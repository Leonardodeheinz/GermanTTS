# Basic Python Imports
from __future__ import annotations
from functools import partial
from genericpath import exists



# NNs libaries
# jax
import jax
import jax.numpy as jnp

# flax
import flax
from flax import nnx
from flax.training import train_state


# optax
import optax
from optax import adamw
from optax import ema

#type hinting & and checkpointing
from jaxtyping import ArrayLike
import orbax.checkpoint as ocp
from etils import epath


# tensor helpers 
from einops import rearrange, reduce, repeat # TODO: maybe not needed after all
import einx

# logging if needed
from loguru import logger

# my data_loader
#from data.PreProcesesAudio import MelSpec, Resampler
#from data.DataLoader import Huggingface_Datasetloader

# Final Model imports
from data.DataLoader import Huggingface_Datasetloader
from models import SAMPLE_RATE, GermanTTS, DurationPredictor, DiT
from config import get_config_DurationPredictor, get_config_GermanTTS, get_config_hyperparameters
from ml_collections import config_flags
from absl import flags

 # Constants that are important for model definition
HOP_LENGTH = 256
SAMPLE_RATE = 24_000
global_seed = 24


class TrainState(train_state.TrainState):
    ema_params:flax.core.FrozenDict = None
    

class GermanTTSTrainerJax:
    def __init__ (self, model: GermanTTS,
                  optimizer_config: optax.GradientTransformation,
                  num_warump_steps:int =  2_000,
                  max_grad_norm:float = 1.0,
                  use_mg_loss:bool = False,
                  ema_decay:float = 0.999 
                  ):
        
        
        
        self.target_sample_rate = SAMPLE_RATE    
        
        self.model = model
        self.optimizer = optimizer_config
        self.num_warump_steps = num_warump_steps
        self.use_mg_loss = use_mg_loss
        self.max_grad_norm = max_grad_norm
        self.ema_decay = ema_decay
        
        self.chkpt_manager = ocp.CheckpointManager(
            './checkpoints',
            options = ocp.CheckpointManagerOptions(
                save_decision_policy = 10_000,
                max_to_keep= 5,
                checkpoint_name = "GermanTTS"
            )
        )
        
        warmup_schedule = optax.linear_schedule(
            init_value = 1e-8, end_value = 1.0, transition_steps = self.num_warump_steps
        )
        decay_steps = self.total_steps - self.num_warump_steps
        decay_schedule = optax.linear_schedule(
            
            init_value = 1.0, end_value = 1e-8, transition_steps=decay_steps
        )
        
        self.learning_rate_schedule_fn = optax.join_schedules(
            schedules = [warmup_schedule, decay_schedule],
            boundaries = [self.num_warump_steps]
            
        )
        
        self.optimizer = nnx.Optimizer(
            optax.chain(
                optax.adamw(learning_rate = self.learning_rate_schedule_fn),
                optax.clip_by_global_norm(self.max_grad_norm),  
            ),
            self.model
        )
        
        #
    
    def checkpoint_path(self, step:int):
        return f"GermanTTS_{step}"
    
    def save_checkpoint(self, step:int, state: TrainState):
        if jax.process_index() == 0:
            ckpt = {'state': state}
            self.chkpt_manager.save(step, ckpt)
        
    
    def load_checkpoint(self,step:int):
        if self.chkpt_manager.latest_step() is None and step == 0:
            return None, 0
        
        target_step = step if step != 0 else self.chkpt_manager.latest_step()
        if target_step is None:
            return None, 0
        
        logger.info(f"Loading checkpoint from step: {target_step}")
            
        rng = jax.random.PRNGKey(seed = global_seed)
        
        dummy_audio = jnp.zeros((1, 24_000), dtype = jnp.bfloat16)
        dummy_text = jnp.zeros((1,128), dtype = jnp.int32)
        dummy_audio_lengths = jnp.array([24_000])
        
        initial_params = self.model.__init__() 












if __name__ == '__main__':
    
    # get configs of GermanTTS 
    
    GermanTTS_config = get_config_GermanTTS()
    
    diffusion_transformer = DiT(*GermanTTS_config)
    
    model = GermanTTS(transformer = diffusion_transformer ,tokenizer=None)

    base_optimizer = optax.adamw(learning_rate=1e-4)

    trainer = GermanTTSTrainerJax(
        model=model,
        base_optimizer_config=base_optimizer,
        total_steps=1000,
        num_warmup_steps=100,
        use_mg_loss=True,
        ema_decay=0.9999,
        max_grad_norm=1.0,
        save_step=100,
    )

    train_dataset = Huggingface_Datasetloader(None) # here i should parse the args

    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        batch_size=2,
        max_batch_frames=24000 * 10,
    )
    print("Training complete.")


















# class DynamicBatchDataLoader:
#     def __init__(self,
#                  collate_fn,
#                  batch_size = 32,
#                  max_batch_frame = 4096,
#                  max_duration = None,
#                  **dataloader_kwargs,
#                  ):
#         self.max_batch_frames = max_batch_frame
#         self.max_duration = max_duration
#         self.dataloader = Huggingface_Datasetloader(
#             **dataloader_kwargs
#         )
#         self.collate_fn = collate_fn
        
        
#     def _collate_wrapper(self, batch):
#         return dynamic_batch_collate_fn(batch, self.collate_fn, self.max_batch_frames, self.max_duration)
    
#     def __iter__(self):
#         batch_iterator = iter(self.dataloader)
#         while True:
#             try:
#                 batch = next(batch_iterator)
                
#                 while batch is None:
#                     batch =next(batch_iterator)
#                 yield batch
#             except StopIteration:
#                 break
        
    
    
    
    

# def collate_fn(batch, audio = None, tokenizer = None):
#     if audio is None:
#         audio = [jnp.array(item[""][""], dtype = jnp.float32) for item in batch]

#     if audio is None or len(audio) == 0:
#         return None
    
#     audio_lengths = jnp.array([item.shape[-1] for item in audio], dtype = jnp.float64)
#     max_audio_length = audio_lengths.amax()
    
#     padding_interval = 1 * 24_000
#     max_audio_length = (max_audio_length + padding_interval -1) // padding_interval + padding_interval
    
#     padded_audio = []
#     for item in audio:
#         padding = (0, max_audio_length - item.size(-1))
#         padded_sec = jnp.pad(item, padding, mode = "empty")
#         padded_audio.append(padded_sec)
    
#     padded_audio = jnp.stack(padded_audio , axis = 0)
    
#     text = [item["text"] for item in batch]
#     text = tokenizer(text)
    
#     return dict(audio = padded_audio, audio_lengths = audio_lengths, text = text)
        
    

# def dynamic_batch_collate_fn(batch, batch_collate_fn, max_batch_frames, max_duration = None):
#     cum_length = 0
#     valid_items = []
#     audio_tensors = []
    
#     max_duration = max_duration if max_duration is not None else 4096
    
#     for idx, item in enumerate(batch):
        


# class GermanTTS_Trainer():
#     def __init__ (self,
#                   model: GermanTTS,
#                   optimizer = optax.adamw,
#                   num_warump_steps:int = 2_000,
#                   max_grad_norm:float = 1.0,
#                   sample_rate:int = 24_000,
#                   use_mg_loss:bool = False,
#                   ema_kwargs:dict = dict()
#                   ):
        
#         self.target_sample_rate = sample_rate
        
#         self.model = model 
#         self.optimizer = optimizer 
        
#         self.num_warmup_steps = num_warump_steps
        
#         self.ema_state = optax.ema(self.model.parameters)
        
#         self.max_grad_norm = max_grad_norm
        


#         self.chkptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler)
        
#     def checkpoint_path(self, step:int):
#         return f"GermanTTS_{step}.pt"
    
#     def save_checkpoint(self, step:str, chkpt_dir:str):             # For now alright but TODO: because it seems a bit of 
#         _, train_state = nnx.split(self.model)
#         path = chkpt_dir + "model_checkpoint" + "_" + "step"
#         self.chkptr.save(path, args = ocp.args.StandardSave(train_state))
#         logger.info(f"State is saved at step:{step} in diretory: {path}")
        
        
#     # def load_checkpoint(self, step:str = 0):   # TODO: but not essentially important right now
#     #     checkpoint
        
        
        
        

# @jax.jit
# def train_GermanTTS(train_dataset, 
#                     model,
#                     optimizer,
#                     * ,total_steps:int = 100_000, batch_size:int = 12, max_batch_frames:int = 4096, max_duration = 4096, save_step:int = 1000): 
    
    
    
    
    
    
    
    

# class DurationTrainer():
#     def __init__ (self, model: DurationPredictor, optimizer = optax.adamw, num_warump_steps:int = 1_000, max_grad_norm:float = 1.0, sample_rate:int = 24_000):
        
#         self.target_sample_rate = sample_rate
#         self.model = model
#         self.optimizer = optimizer
        
#         self.num_warump_steps = num_warump_steps
#         self.max_grad_norm = max_grad_norm
        
#     def checkpoint_path(self, step:int):
#         return f"duration_{step}.pt"
    
       
#     def save_checkpoint(self, step:str, chkpt_dir:str):
#         _, train_state = nnx.split(self.model)
#         path = chkpt_dir + "model_checkpoint" + "_" + "step"
#         self.chkptr.save(path, args = ocp.args.StandardSave(train_state))
#         logger.info(f"State is saved at step:{step} in diretory: {path}")
    
    