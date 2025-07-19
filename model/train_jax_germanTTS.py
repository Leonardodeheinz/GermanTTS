"""
ein notation:
b - batch
n - sequence
f - frequency token dimension
nt - text sequence
nw - raw wave length
d - dimension
dt - dimension text
c - channel dimensions
"""

# Basic Python Imports
from __future__ import annotations
from functools import partial
import functools
from genericpath import exists
from math import inf
import os
from typing import Callable, Dict, Iterator, Tuple
import flax.nnx
from tqdm import tqdm
 
import os
import sys 
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
 
# NNs libaries
# jax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental import mesh_utils

# flax
import flax
import flax.nnx as nnx 
from flax.training import train_state

# numpy 
import numpy as np


# optax

import optax
from optax import adamw
from optax import ema

#type hinting & and checkpointing & Dataloading
from jaxtyping import ArrayLike
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader, Dataset


# tensor helpers 

from GermanTTS.model.models import lens_to_mask, mask_from_frac_lengths, mask_from_start_end_indicies, maybe_masked_mean
from jaxtyping import Float, Array, PyTree, ArrayLike 
from einops import rearrange, reduce, repeat

# logging if needed
from loguru import logger
import wandb
import wandb

# Final Model imports
#from data.DataLoader import Huggingface_Datasetloader
from GermanTTS.model.models import SAMPLE_RATE, GermanTTS, DurationPredictor, DiT
from GermanTTS.model.config import get_config_DurationPredictor, get_config_GermanTTS, get_config_hyperparameters
from ml_collections import config_flags
from data.DataLoader import Huggingface_Datasetloader


 # Constants that are important for model definition
HOP_LENGTH = 256
SAMPLE_RATE = 24_000
global_seed = 24

class DynamicsBatchLoader:
    def __init__(self,
                 dataset, 
                 collate_fn,
                 tokenizer,
                 batch_size:int = 32,
                 max_batch_frames:int = 16384,
                 max_duration:float = 20.0,
                 **dataloader_kwargs,
                 ):
        
        self.max_batch_frames = max_batch_frames
        self.max_duration = max_duration
        
        self.tokenizer = tokenizer
        self.dataloader_collate_fn = functools.partial(collate_fn, tokenizer=self.tokenizer)
        
        self.dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            collate_fn = self.dataloader_collate_fn,
            **dataloader_kwargs,
        )
        self.collate_fn = collate_fn
    
    
    def __len__(self):
        return len(self.dataset)
 
    def _collate_wrapper(self, batch):
        return dynamic_batch_collate_fn(batch, self.collate_fn, self.max_batch_frames, self.max_duration)
 
    def __iter__(self):
        return iter(self.dataloader) 
   
def collate_fn(batch, tokenizer): # tokenizer should be passed during DataLoader init
   
    if not batch:
        print("Warning: collate_fn received an empty batch.")
        return None # Or raise an error, depending on desired behavior

    audio_samples = [item["mel"] for item in batch]

   
    if not audio_samples:
        return None

    audio_lengths = jnp.array([item["audio_duration"] for item in batch])
    max_audio_length = jnp.max(audio_lengths)

    padding_interval = 1 * 2_000
    max_audio_length  = (max_audio_length + padding_interval -1) // padding_interval * padding_interval

    padded_audio = []
    for item_audio_data in audio_samples: # Iterate through the extracted audio samples
        padding = ((0,0), 
                   (0, jnp.int32(max_audio_length - item_audio_data.shape[1])))
                   
        #logger.info(f"padding with value:{padding}")
        padded_spec = jnp.pad(item_audio_data, padding, mode = "empty")
        padded_audio.append(padded_spec)

    padded_audio = jnp.stack(padded_audio, axis = 0)

    text = [item["transcript"] for item in batch]
    text = tokenizer(text)

    return dict(audio = padded_audio, audio_lengths = audio_lengths, text = text)


def dynamic_batch_collate_fn(batch, batch_collate_fn, max_batch_frames, max_duration=None):
    cum_length = 0
    valid_items = []
    audio_tensors = []

    max_duration = max_duration if max_duration is not None else 4096

    for idx, item in enumerate(batch):
        item_audio = jnp.array(item["mp3"]["array"], device="cpu", dtype = jnp.float32)
        item_length = item_audio.shape[-1] // HOP_LENGTH

        if item_length > max_batch_frames or item_length > max_duration:
            continue

        if cum_length + item_length <= max_batch_frames:
            valid_items.append({k: v for k, v in item.items() if k != "mp3"})
            audio_tensors.append(item_audio)
            cum_length += item_length
        else:
            if valid_items:
                break

    if not valid_items:
        print("warning: no valid items in batch")
        return None

    collated_items = batch_collate_fn(valid_items, audio=audio_tensors)

    return collated_items



class TrainState(train_state.TrainState):
    ema_params:flax.core.FrozenDict = None
    
nnx.jit()
def train_step_jitted(model: GermanTTS, optimizer: nnx.Optimizer, batch: Dict[str, jnp.ndarray], 
                       max_grad_norm: float, use_flow_matching_loss: bool, ema_model: GermanTTS = None,
                       rngs:nnx.Rngs = None, ema_decay: float = 0.999
                       ) -> Tuple[Float[Array, ""], float, jax.random.PRNGKey]:
    
    # Value and gradient with respect to model parameters
    (loss, new_rng_key), grads = nnx.value_and_grad(calculate_loss, has_aux=True)(
        model, batch, rngs=rngs, ema_model=ema_model, use_flow_matching_loss=use_flow_matching_loss
    )

    optimizer.update(grads) # Updates model parameters in-place through NNX's binding

    # EMA update (JAX will ensure this is consistent across devices due to replicated model)
    if ema_model and use_flow_matching_loss:
        new_ema_params = optax.incremental_update(
            new_tensors=model.filter(nnx.Param),
            old_tensors=ema_model.filter(nnx.Param),
            step_size= 1.0 - ema_decay  # Corresponds to (1.0 - ema_decay) in your original formula
        )
        
        for name, param_value in new_ema_params.items():
            ema_model.filter(nnx.Param)[name].value = param_value

    # Learning rate calculation (will be identical across devices)
    current_lr = optax.get_current_schedule_value(optimizer.tx, optimizer.state)

    return loss, current_lr, new_rng_key
    
    

def calculate_loss(
    main_model: GermanTTS,
    batch: Dict[str, jnp.ndarray],
    rngs:nnx.Rngs,
    ema_model: GermanTTS = None,
    use_flow_matching_loss: bool = False,
) -> Tuple[Float[Array, ""], jax.random.PRNGKey]:
    
    text_input, audio_input, audio_lengths_raw = batch["text"], batch["audio"], batch["audio_lengths"]

    x1 = audio_input
    
    max_len = x1.shape[-1]
    
    audio_lengths = audio_lengths_raw

    batch_size = x1.shape[0] # This will be local batch size per device
    
    dtype = x1.dtype

    
    #audio_lengths = jnp.full((batch_size,), seq_len, dtype=jnp.int32)
    audio_lengths_in_frames_float = (audio_lengths * SAMPLE_RATE) / HOP_LENGTH
    audio_lengths = jnp.round(audio_lengths_in_frames_float).astype(jnp.int64)

    mask = lens_to_mask(audio_lengths, max_len=max_len)

    rng_frac_lengths = rngs.other()

    frac_lengths = jax.random.uniform(rng_frac_lengths, shape=(batch_size,), dtype=jnp.float32,
                                     minval=main_model.frac_lengths_mask[0],
                                     maxval=main_model.frac_lengths_mask[1])
    

    rand_span_mask = mask_from_frac_lengths(seq_len = audio_lengths, frac_lengths = frac_lengths, max_seq_len = max_len, rngs = rngs)

    
    rand_span_mask = rand_span_mask * mask
    #rand_span_mask = rearrange(rand_span_mask, "b n d -> b d n")

    rng_x0 = rngs.other()
    x0 = jax.random.normal(rng_x0, x1.shape, dtype=dtype)

    rng_time = rngs.other()
    time = jax.random.uniform(rng_time, shape=(batch_size,), dtype=dtype)

    t = rearrange(time, "b -> b 1 1")
    w = (1 - t) * x0 + t * x1
    flow = x1 - x0

    #cond_bool_mask = (rand_span_mask > 0.5)
    
    
    
    cond = jnp.where(rand_span_mask[:,None,:], jnp.zeros_like(x1), x1)

    rng_rand_audio_drop = rngs.other()
    rng_rand_cond_drop = rngs.other()
    
    rand_audio_drop = jax.random.uniform(rng_rand_audio_drop, (1,))[0]
    rand_cond_drop = jax.random.uniform(rng_rand_cond_drop, (1,))[0]
    
    drop_audio_cond_main = rand_audio_drop < main_model.audio_drop_prob
    drop_text_main = rand_cond_drop < main_model.cond_drop_prob
    
    if not use_flow_matching_loss:
        actual_drop_audio_cond = drop_audio_cond_main | drop_text_main
        actual_drop_text = drop_text_main
    else:
        actual_drop_audio_cond = False
        actual_drop_text = False


    pred = main_model(
        inp=w,
        cond=cond,
        text=text_input,
        time=time,
        drop_audio_cond = actual_drop_audio_cond,
        drop_text = actual_drop_text,
        mask = mask,
    )

    if use_flow_matching_loss and ema_model is not None:
        guidance_cond = ema_model(
            inp=w,
            cond=cond,
            text=text_input,
            time=time,
            lens=audio_lengths_raw,
            drop_audio_cond=False,
            drop_text=False,
            mask=mask,
        )

        guidance_uncond = ema_model(
            inp=w,
            cond=cond,
            text=text_input,
            time=time,
            lens=audio_lengths_raw,
            drop_audio_cond=True,
            drop_text=True,
            mask=mask,
        )
        
        # Ensure guidance_scale is also consistent across devices if needed
        guidance_scale = jnp.where(time < 0.75, 0.5, 0.0)[:, None, None]
        guidance = (guidance_cond - guidance_uncond) * guidance_scale
        
        flow_target = flow + guidance
    else:
        flow_target = flow

    loss = optax.losses.squared_error(pred, flow_target).mean()
    
    masked_loss = loss * rand_span_mask[:, :, None]
    total_masked_elements = jnp.sum(rand_span_mask) * x1.shape[-1]
    loss = jnp.sum(masked_loss) / (total_masked_elements + 1e-6)

    return loss, rng_key    

class GermanTTSTrainerJax:
    def __init__ (self, model: GermanTTS,
                  ema_model:GermanTTS,
                  optimizer:optax.adamw = None,
                  num_warmup_steps:int =  2_000,
                  max_grad_norm:float = 1.0,
                  use_mg_loss:bool = False,
                  ema_decay:float = 0.999,
                  total_epochs:int = 10_000,
                  tokenizer:functools.partial = None,
                  rngs: nnx.Rngs = None,
                  project_name: str = "GermanTTS_Training",
                  ):
        
        self.target_sample_rate = SAMPLE_RATE
            
        self.optimizer = nnx.Optimizer(model = model, tx = optimizer)        
        self.model = model
        self.ema_model = ema_model
        self.tokenizer = tokenizer
        
        self.num_warmup_steps = num_warmup_steps
        self.use_mg_loss = use_mg_loss
        self.max_grad_norm = max_grad_norm
        self.ema_decay = ema_decay
        self.total_epochs = total_epochs
        self.rngs = rngs
        
        
        self.num_devices = len(jax.devices("gpu"))
        logger.info(f"Training start with current number of devices GPU:{self.num_devices}")
        
        self.mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count("gpu"),)), ("data",))
        
       
        
        self.chkpt_manager = ocp.CheckpointManager(
            './checkpoints',
            options = ocp.CheckpointManagerOptions(
                save_decision_policy = 10_000,
                max_to_keep= 5
            )
        )
           
    
    def checkpoint_path(self, step:int):
        return f"GermanTTS_{step}"
    
    def save_checkpoint(self, step:int, state: TrainState):
        if jax.process_index() == 0:
            ckpt = {'state': state}
            self.chkpt_manager.save(step, ckpt)
        
    
    def load_checkpoint(self, step: int):
    # Determine the target step to load
        target_step = step if step != 0 else self.chkpt_manager.latest_step()

        if target_step is None:
            logger.info("No checkpoint found. Starting from scratch.")
        # Return None for state/optimizer state, and 0 for start_step
            return None, 0, None, None

        logger.info(f"Loading checkpoint from step: {target_step}")

    # Load the checkpoint data
        ckpt_data = self.chkpt_manager.restore(target_step)

        if ckpt_data is None:
            logger.warning(f"Checkpoint at step {target_step} not found or corrupted. Starting from scratch.")
            return None, 0, None, None

    # Extract the components
        loaded_state = ckpt_data.get('state')
        loaded_optimizer_state = ckpt_data.get('optimizer_state')
        loaded_step = ckpt_data.get('step', 0) # Default to 0 if 'step' isn't explicitly saved
        loaded_ema_params = ckpt_data.get('ema_params')

    # Return the loaded components
        return loaded_state, loaded_step, loaded_optimizer_state, loaded_ema_params

    def _train_epoch(self, dataset, epoch: int, save_step: int, batch_size:int = 16,max_batch_frames:int = None, max_duration:int = None, rngs:nnx.Rngs = None) -> float:
        """
        Performs training for a single epoch.
        Returns the average loss for the epoch.
        """
        epoch_loss = 0.0
        item_count = 0
        total_frames = 0
        
        train_dataloader = DynamicsBatchLoader(dataset = dataset, 
                                               collate_fn = collate_fn,
                                               batch_size = batch_size,
                                               max_batch_frames = max_batch_frames,
                                               max_duration = max_duration,
                                               tokenizer = self.tokenizer)
        
      
        
        progress_bar = tqdm(train_dataloader,
                            desc = f"Epoch {epoch + 1}",
                            unit = "batch",
                            total = len(train_dataloader.dataloader),
                            disable = jax.process_index() != 0,
                            )
        
        
        for batch in progress_bar:
            
            current_global_step = self.optimizer.step.value if isinstance(self.optimizer.step, flax.nnx.optimizer.OptState) else self.optimizer.step
            
            if self.optimizer.step.value >= self.total_epochs:
                break
            
            sharded_batch = jax.tree.map(
                    lambda x: jax.device_put(x) if isinstance(x, jax.Array) else x,
                    batch)
            
            loss_item, current_lr, new_rng_key = train_step_jitted(
                self.model,
                self.optimizer,
                sharded_batch,
                self.max_grad_norm,
                self.use_mg_loss,
                self.ema_model,
                rngs,
                self.ema_decay
            )
            
            self.rng_key = new_rng_key
            
            total_frames += jnp.sum(batch["audio_duration"] / HOP_LENGTH).item()
            
            if jax.process_index() == 0:
                progress_bar.set_postfix(
                    loss=f"{loss_item:.4f}", 
                    lr=f"{current_lr:.2e}", 
                    frames=f"{total_frames / 1e6:.2f}M",
                    step=f"{self.optimizer.step.value.item()}/{self.total_steps}"
                )
                
                wandb.log({
                        "train/loss": loss_item,
                        "train/learning_rate": current_lr,
                        "train/total_frames_processed_M": total_frames / 1e6,
                        "train/step": current_global_step,
                        "epoch": epoch
                }, step=current_global_step)


            epoch_loss += loss_item
            item_count += 1

            # Save checkpoint (only from the first device)
            # Use `optimizer.step.item()` to get the current global step
            if self.optimizer.step.value.item() % save_step == 0:
                self.save_checkpoint(self.optimizer.step.value.item())

        return epoch_loss / (item_count if item_count > 0 else 1.0)

            
    def train(
        self,
        train_dataset,
        batch_size=12,
        max_batch_frames= 16384, # even higher since gpu h100 is default in my case
        max_duration= 16384,
        restore_step = None,
        save_step=1000,
        flow_matching_loss = True,
    ):
        
        # Load checkpoint and set optimizer step
      
        # if exists(restore_step):
        #     start_step = self.load_checkpoint(restore_step)
        # else:
        start_step = 0
        self.optimizer.step.value = start_step

        if jax.process_index() == 0:
            hps = {
                "total_steps": self.total_epochs,
                "num_warmup_steps": self.num_warmup_steps,
                "max_grad_norm": self.max_grad_norm,
                "batch_size": batch_size,
                "max_batch_frames": max_batch_frames,
                "max_duration": max_duration,
                "ema_decay": self.ema_decay,
                "use_flow_matching_loss": flow_matching_loss,
                "save_step": save_step,
                "num_devices_gpu": self.num_devices, 
            }
            print(f"Hyperparameters: {hps}")
        
        # --- Added training loop ---
        for epoch in range(self.total_epochs):
          
            current_loop_step = self.optimizer.step.value
            if current_loop_step >= self.total_epochs:
                logger.info(f"Reached total_steps ({self.total_epochs}), stopping training.")
                break

            avg_loss = self._train_epoch(train_dataset, epoch, save_step, batch_size, max_batch_frames, max_duration, rngs = self.rngs)
            if jax.process_index() == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs} finished, Average Loss: {avg_loss:.4f}")

            # You might want to save a checkpoint at the end of each epoch as well
            # if jax.process_index() == 0:
            #     self.save_checkpoint(self.optimizer.step.item(), TrainState()) # Save end of epoch state
        # --- End of added training loop ---
