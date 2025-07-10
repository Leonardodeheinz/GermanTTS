import jax
import jax.numpy as jnp
import optax
from flax import nnx
import flax.linen as nn # For TrainState, though NNX prefers its own state management
from flax.training import train_state # For TrainState
import orbax.checkpoint as ocp
from functools import partial
import os
from tqdm import tqdm
from typing import Any, Callable, Dict, Tuple, Union, List, Iterator

# Type hints for JAX arrays (assuming you have jaxtyping installed)
from jaxtyping import Float, Int, Array # type: ignore

# For sharding
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

# For logging (dummy logger if not provided)
import logging
try:
    from logs import logger
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


# --- Constants ---
HOP_LENGTH = 256
SAMPLE_RATE = 24_000
global_seed = 24
NUM_MEL_CHANNELS = 100 # Assuming this is consistent with your GermanTTS model



# --- TrainState (as provided) ---
class TrainState(train_state.TrainState):
    # NNX's Optimizer manages its own state and model parameters.
    # If you intend to use this TrainState for something else, keep it.
    # If it's meant to hold model params for a non-NNX setup, it's not directly compatible.
    # For NNX, the model and optimizer instances themselves *are* the state.
    ema_params: flax.core.FrozenDict = None


# --- Loss function (remains largely the same, but comments adjusted for global batch) ---
def calculate_loss(
    main_model: GermanTTS,
    batch: Dict[str, jnp.ndarray],
    *,
    rng_key: jax.random.PRNGKey, # This will be the GLOBAL RNG key managed by the trainer
    ema_model: GermanTTS = None,
    use_flow_matching_loss: bool = False,
) -> Tuple[Float[Array, ""], jax.random.PRPRNGKey]:
    
    text_input, audio_input, audio_lengths_raw = batch["text"], batch["audio"], batch["audio_lengths"]

    # If audio_input is waveform, convert to mel
    if audio_input.ndim == 2:
        x1 = main_model.mel_spec(audio_input)
        # Assuming audio_lengths_raw are for waveforms, convert to mel frame lengths
        audio_lengths = (audio_lengths_raw / main_model.mel_spec.hop_length).astype(jnp.int32)
    else: # Already mel spectrogram
        x1 = audio_input
        audio_lengths = audio_lengths_raw

    seq_len = x1.shape[1]
    batch_size = x1.shape[0] # This is the GLOBAL batch size because inputs are sharded
    dtype = x1.dtype

    if not exists(audio_lengths):
        audio_lengths = jnp.full((batch_size,), seq_len, dtype=jnp.int32)

    mask = lens_to_mask(audio_lengths, length=seq_len)

    rng_key, subkey_frac = jax.random.split(rng_key)
    frac_lengths = jax.random.uniform(subkey_frac, shape=(batch_size,), dtype=jnp.float32,
                                     minval=main_model.frac_lengths_mask[0],
                                     maxval=main_model.frac_lengths_mask[1])
    
    rng_key, subkey_rand_span_mask = jax.random.split(rng_key)
    # The mask_from_frac_lengths function should ideally be designed to handle
    # the sharded nature of `lengths` and `frac_lengths` if they are sharded.
    # JAX will automatically parallelize this if inputs are sharded.
    rand_span_mask = mask_from_frac_lengths(audio_lengths, frac_lengths, seq_len, subkey_rand_span_mask)

    if exists(mask):
        rand_span_mask = rand_span_mask * mask

    rng_key, subkey_x0 = jax.random.split(rng_key)
    x0 = jax.random.normal(subkey_x0, x1.shape, dtype=dtype)

    rng_key, subkey_time = jax.random.split(rng_key)
    time = jax.random.uniform(subkey_time, shape=(batch_size,), dtype=dtype)

    t = time[:, None, None]
    w = (1 - t) * x0 + t * x1
    flow = x1 - x0

    cond_bool_mask = (rand_span_mask > 0.5)
    cond = jnp.where(cond_bool_mask[:, :, None], jnp.zeros_like(x1), x1)

    rng_key, subkey_drop_audio_cond = jax.random.split(rng_key)
    rng_key, subkey_drop_text = jax.random.split(rng_key)
    
    # These random drops should be generated PER-EXAMPLE for effective batch-wise dropout.
    # The original code `(1,)` then `[0]` makes it a single scalar which is then broadcast.
    # This might be intended for "batch-level" dropout, but typically dropout is per-example.
    # For per-example dropout:
    rand_audio_drop = jax.random.uniform(subkey_drop_audio_cond, (batch_size,))
    rand_cond_drop = jax.random.uniform(subkey_drop_text, (batch_size,))
    
    drop_audio_cond_main = rand_audio_drop < main_model.audio_drop_prob
    drop_text_main = rand_cond_drop < main_model.cond_drop_prob
    
    if not use_flow_matching_loss:
        actual_drop_audio_cond = drop_audio_cond_main | drop_text_main
        actual_drop_text = drop_text_main
    else:
        # In flow matching, it's common to always use the unconditional branch (drop=True)
        # and conditional branch (drop=False) for sampling, but for training,
        # typically you still apply dropout. The original code sets them to False here.
        # Let's keep it consistent with your provided logic for now.
        actual_drop_audio_cond = jnp.zeros_like(drop_audio_cond_main, dtype=jnp.bool_)
        actual_drop_text = jnp.zeros_like(drop_text_main, dtype=jnp.bool_)

    pred = main_model(
        inp=w,
        cond=cond,
        text=text_input,
        time=time,
        lens=audio_lengths_raw,
        drop_audio_cond=actual_drop_audio_cond,
        drop_text=actual_drop_text,
        mask=mask,
    )

    if use_flow_matching_loss and ema_model is not None:
        # These predictions will be made by the EMA model which usually has no dropout in evaluation mode
        # You specified False/True directly, so keeping that.
        guidance_cond = ema_model(
            inp=w,
            cond=cond,
            text=text_input,
            time=time,
            lens=audio_lengths_raw,
            drop_audio_cond=jnp.zeros_like(drop_audio_cond_main, dtype=jnp.bool_), # No dropout for conditional
            drop_text=jnp.zeros_like(drop_text_main, dtype=jnp.bool_),
            mask=mask,
        )

        guidance_uncond = ema_model(
            inp=w,
            cond=cond,
            text=text_input,
            time=time,
            lens=audio_lengths_raw,
            drop_audio_cond=jnp.ones_like(drop_audio_cond_main, dtype=jnp.bool_), # Full dropout for unconditional
            drop_text=jnp.ones_like(drop_text_main, dtype=jnp.bool_),
            mask=mask,
        )
        
        guidance_scale = jnp.where(time < 0.75, 0.5, 0.0)[:, None, None]
        guidance = (guidance_cond - guidance_uncond) * guidance_scale
        
        flow_target = flow + guidance
    else:
        flow_target = flow

    loss = optax.losses.squared_error(pred, flow_target)
    
    masked_loss = loss * rand_span_mask[:, :, None]
    total_masked_elements = jnp.sum(rand_span_mask) * x1.shape[-1]
    loss = jnp.sum(masked_loss) / (total_masked_elements + 1e-6)

    return loss, rng_key


# --- Training Step Function (remains the same) ---
@partial(jax.jit, donate_argnums=(0, 1)) # donate_argnums allows XLA to optimize in-place updates
def train_step_jitted(model: GermanTTS, optimizer: nnx.Optimizer, batch: Dict[str, jnp.ndarray], 
                       max_grad_norm: float, use_flow_matching_loss: bool, ema_model: GermanTTS = None,
                       learning_rate_schedule_fn: Callable[[int], float] = None,
                       rng_key: jax.random.PRNGKey = None, ema_decay: float = 0.0
                       ) -> Tuple[Float[Array, ""], float, jax.random.PRNGKey]:
    
    # Value and gradient with respect to model parameters
    (loss, new_rng_key), grads = nnx.value_and_grad(calculate_loss, has_aux=True)(
        model, batch, rng_key=rng_key, ema_model=ema_model, use_flow_matching_loss=use_flow_matching_loss
    )

    optimizer.update(grads) # Updates model parameters in-place through NNX's binding

    # EMA update (JAX will ensure this is consistent across devices due to replicated model)
    if ema_model and use_flow_matching_loss:
        main_params = model.filter(nnx.Param)
        ema_params = ema_model.filter(nnx.Param)
        for name, main_var in main_params.items():
            ema_params[name].value = ema_params[name].value * ema_decay + main_var.value * (1.0 - ema_decay)

    # Learning rate calculation (will be identical across devices)
    current_lr = learning_rate_schedule_fn(optimizer.step)

    return loss, current_lr, new_rng_key


# --- GermanTTSTrainerJax Class ---
class GermanTTSTrainerJax:
    def __init__(self, 
                 model: GermanTTS, # Model is now passed in
                 optimizer: nnx.Optimizer, # Optimizer is now passed in
                 total_steps: int, # This was missing in your provided __init__
                 num_warmup_steps: int = 2_000,
                 max_grad_norm: float = 1.0,
                 use_mg_loss: bool = False, # Renamed from use_flow_matching_loss
                 ema_decay: float = 0.999,
                 ema_model: GermanTTS = None, # EMA model also passed in
                 seed: int = global_seed, # Use global_seed by default
                 checkpoint_dir: str = './checkpoints',
                 ):
        
        self.target_sample_rate = SAMPLE_RATE    
        
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        
        self.total_steps = total_steps # Crucial for learning rate schedule and total training time
        self.num_warmup_steps = num_warmup_steps
        self.use_flow_matching_loss = use_mg_loss # Keep consistent name internally
        self.max_grad_norm = max_grad_norm
        self.ema_decay = ema_decay
        self.seed = seed

        # Discover devices and set up mesh
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        logger.info(f"Training start with current number of devices GPU: {jax.device_count('gpu')}, CPU: {jax.device_count('cpu')}, TPU: {jax.device_count('tpu')}, Total JAX devices: {self.num_devices}")
        
        # Define the device mesh for sharding
        # This creates a 1D mesh named 'data' that spans all devices.
        self.mesh = Mesh(mesh_utils.create_device_mesh((self.num_devices,)), ('data',))

        # Generate a global RNG key for the trainer
        self.rng_key = jax.random.PRNGKey(self.seed)
        
        self.ckpt_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options = ocp.CheckpointManagerOptions(
                save_interval_steps=10_000, # This means saving every 10k steps. Use `save_step` from `train` method for more frequent saves.
                max_to_keep= 5,
                checkpoint_name = "GermanTTS"
            )
        )
        
        warmup_schedule = optax.linear_schedule(
            init_value = 1e-8, end_value = 1.0, transition_steps = self.num_warmup_steps
        )
        decay_steps = self.total_steps - self.num_warmup_steps
        decay_schedule = optax.linear_schedule(
            init_value = 1.0, end_value = 1e-8, transition_steps=decay_steps
        )
        
        self.learning_rate_schedule_fn = optax.join_schedules(
            schedules = [warmup_schedule, decay_schedule],
            boundaries = [self.num_warmup_steps]
        )
        
    def save_checkpoint(self, step: int):
        if jax.process_index() == 0:
            ckpt = {
                'model': self.model,
                'optimizer': self.optimizer,
                'rng_key': self.rng_key,
            }
            if self.ema_model:
                ckpt['ema_model'] = self.ema_model
            
            self.ckpt_manager.save(step, ckpt)
            logger.info(f"Saved checkpoint at step {step}")
        
    
    def load_checkpoint(self, restore_step: int | None):
        current_step = 0
        if jax.process_index() == 0:
            if self.ckpt_manager.latest_step() is None and (restore_step is None or restore_step == 0):
                return 0 # No checkpoint found, start from step 0

            target_step = restore_step if restore_step is not None and restore_step != 0 else self.ckpt_manager.latest_step()
            if target_step is None:
                return 0
                
            logger.info(f"Loading checkpoint from step: {target_step}")
            
            # Need to re-create a template for the model and optimizer with correct sharding
            # The structure must match the saved structure.
            
            # Use self.model and self.optimizer directly as templates.
            # Their existing structure (with replicated jax.Arrays) is what Orbax expects.
            restored_args = {
                'model': self.model, # Orbax will restore *into* this instance's variables
                'optimizer': self.optimizer, # Orbax will restore *into* this instance's state
                'rng_key': jax.random.PRNGKey(0), # Placeholder for RNG key
            }
            if self.ema_model:
                restored_args['ema_model'] = self.ema_model

            restored_state = self.ckpt_manager.restore(
                target_step,
                args=ocp.args.StandardRestore(restored_args)
            )
            
            # Assign restored RNG key
            self.rng_key = restored_state['rng_key']
            
            current_step = self.optimizer.step.item() # Optimizer step is now a jax.Array
        
        # All processes must agree on the step if doing multi-process training
        if jax.process_count() > 1:
            # Need to get the step from process 0 and broadcast it.
            # JAX's `jax.lax.broadcast` works on `jax.Array`s.
            current_step_array = jax.device_put(jnp.array(current_step, dtype=jnp.int32), jax.local_devices(0))
            current_step = jax.lax.broadcast(current_step_array, 0).item()
        
        # Ensure the optimizer's step is updated on all devices to the restored value
        # This is already handled by the `self.optimizer = restored_state['optimizer']`
        # if the optimizer instance itself is fully replaced, but explicit is safer.
        self.optimizer.step = jax.device_put(jnp.array(current_step), P(), self.mesh)

        return current_step


    def _train_epoch(self, train_dataloader: Iterator[Dict[str, jnp.ndarray]], epoch: int, save_step: int) -> float:
        """
        Performs training for a single epoch.
        Returns the average loss for the epoch.
        """
        epoch_loss = 0.0
        item_count = 0
        total_frames = 0 

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            total=len(train_dataloader),
            disable=jax.process_index() != 0, 
        )

        for batch in progress_bar:
            if self.optimizer.step.item() >= self.total_steps:
                break

            # Split the global RNG key for the current step
            self.rng_key, step_rng_key = jax.random.split(self.rng_key)

            # --- Apply Sharding to Batch Data ---
            # Explicitly put batch data onto devices with desired sharding.
            sharded_batch = jax.tree_map(
                lambda x: jax.device_put(x, P('data', *([None] * (x.ndim - 1))), self.mesh),
                batch
            )
            
            # Call the jitted train step function
            loss_item, current_lr, new_rng_key = train_step_jitted(
                self.model,
                self.optimizer,
                sharded_batch,
                self.max_grad_norm,
                self.use_flow_matching_loss,
                self.ema_model,
                self.learning_rate_schedule_fn,
                step_rng_key,
                self.ema_decay
            )
            
            self.rng_key = new_rng_key

            total_frames += jnp.sum(batch["audio_lengths"] / HOP_LENGTH).item()

            if jax.process_index() == 0:
                progress_bar.set_postfix(
                    loss=f"{loss_item:.4f}", 
                    lr=f"{current_lr:.2e}", 
                    frames=f"{total_frames / 1e6:.2f}M",
                    step=f"{self.optimizer.step.item()}/{self.total_steps}"
                )

            epoch_loss += loss_item
            item_count += 1

            # Save checkpoint (only from the first device)
            # Use `optimizer.step.item()` to get the current global step
            if self.optimizer.step.item() % save_step == 0:
                self.save_checkpoint(self.optimizer.step.item())

        return epoch_loss / (item_count if item_count > 0 else 1.0)


    def train(
        self,
        train_dataset,
        batch_size=12,
        max_batch_frames=4096,
        max_duration=4096,
        num_workers=0,
        restore_step=None,
        save_step=1000,
    ):
        # Load checkpoint and set optimizer step
        start_step = self.load_checkpoint(restore_step)
        # self.optimizer.step is already set during load_checkpoint, so no need to repeat.


        if jax.process_index() == 0:
            hps = {
                "total_steps": self.total_steps,
                "num_warmup_steps": self.num_warmup_steps,
                "max_grad_norm": self.max_grad_norm,
                "batch_size": batch_size,
                "max_batch_frames": max_batch_frames,
                "max_duration": max_duration,
                "ema_decay": self.ema_decay,
                "use_flow_matching_loss": self.use_flow_matching_loss,
                "epochs": self.epochs, #epochs parameter was missing in __init__
                "save_step": save_step,
                "seed": self.seed,
                "num_devices": self.num_devices,
            }
            logger.info(f"Hyperparameters: {hps}")


        class MockDynamicBatchDataLoader(Iterator):
            def __init__(self, dataset, global_batch_size, num_devices, max_batch_frames, max_duration, collate_fn):
                self.dataset = dataset
                self.global_batch_size = global_batch_size
                self.num_devices = num_devices
                if global_batch_size % num_devices != 0:
                     logger.warning(f"WARNING: Global batch size ({global_batch_size}) not divisible by number of devices ({num_devices}). JAX sharding will handle padding/truncation, but consider making it divisible for cleaner batching.")
                self.local_batch_size = global_batch_size // num_devices

                self.max_batch_frames = max_batch_frames
                self.max_duration = max_duration
                self.collate_fn = collate_fn
                self._current_batch_idx = 0
                self._total_batches_per_epoch = len(self.dataset) // self.global_batch_size
                if len(self.dataset) % self.global_batch_size != 0:
                    self._total_batches_per_epoch += 1

            def __iter__(self):
                self._current_batch_idx = 0
                return self

            def __next__(self) -> Dict[str, jnp.ndarray]:
                if self._current_batch_idx >= self._total_batches_per_epoch:
                    raise StopIteration
                
                global_text_batch = jnp.array([[1, 2, 3, 0], [4, 5, 6, 7]] * self.local_batch_size, dtype=jnp.int32)
                global_text_batch = global_text_batch[:self.global_batch_size]
                
                # Assume 2 seconds of audio for this dummy batch for simplicity, 
                # converted to mel frames. (SAMPLE_RATE / HOP_LENGTH * 2 = 187.5)
                # Let's say mel frames are ~188 for 2s audio.
                # Mel dims: (batch, mel_frames, mel_channels)
                global_audio_batch = jnp.zeros((self.global_batch_size, 188, NUM_MEL_CHANNELS), dtype=jnp.float32)
                
                # Assuming this refers to mel frame lengths now, not raw audio samples
                global_audio_lengths_batch = jnp.array([188] * self.global_batch_size, dtype=jnp.int32)
                
                batch = {
                    "text": global_text_batch,
                    "audio": global_audio_batch,
                    "audio_lengths": global_audio_lengths_batch,
                }
                
                self._current_batch_idx += 1
                return batch
                    
            def __len__(self):
                return self._total_batches_per_epoch


        train_collate_fn = partial(collate_fn, tokenizer=None) # Assuming collate_fn needs tokenizer
        
        for epoch in range(self.epochs): # Now self.epochs is available
            if self.optimizer.step.item() >= self.total_steps:
                logger.info(f"Reached total_steps ({self.total_steps}). Stopping training.")
                break 

            train_dataloader = MockDynamicBatchDataLoader(
                train_dataset,
                global_batch_size=batch_size,
                num_devices=self.num_devices,
                max_batch_frames=max_batch_frames,
                max_duration=max_duration,
                collate_fn=train_collate_fn
            )

            epoch_avg_loss = self._train_epoch(train_dataloader, epoch, save_step)

            if jax.process_index() == 0:
                logger.info(f"Epoch {epoch + 1} average loss: {epoch_avg_loss:.4f}")

        if jax.process_index() == 0:
            self.ckpt_manager.wait_until_finished()
            logger.info("Training finished.")

# Helper function placeholder for collate_fn (assuming it returns tensors suitable for JAX)
def collate_fn(batch_items: List[Dict[str, Any]], tokenizer: Any) -> Dict[str, jnp.ndarray]:
    # This collate_fn should prepare a GLOBAL batch, which the trainer will shard.
    texts = [item["json"]["text"] for item in batch_items]
    # In this new setup, `item["audio"]` for the mock might be raw waveform,
    # but the `calculate_loss` expects mel or handles conversion.
    # For a real pipeline, your data loader would likely return mel features directly or raw audio.
    # The MockDataLoader above is simplified to return pre-computed mel for consistency.
    audios = [item["audio"] for item in batch_items] # This would be `(num_samples,)` if raw audio
                                                     # or `(mel_frames, mel_channels)` if pre-computed mel

    tokenized_texts_numerical = [jnp.array([ord(c) for c in s], dtype=jnp.int32) for s in texts]
    
    max_text_len = max(len(t) for t in tokenized_texts_numerical)
    padded_texts = jnp.stack([jnp.pad(t, (0, max_text_len - len(t)), constant_values=0) for t in tokenized_texts_numerical])

    # If `audios` from dataset are raw waveforms, you might need to convert them here or pass them as is
    # For the `MockDynamicBatchDataLoader`, it's already providing mel features, so this part needs adjustment
    # based on actual dataset output. Assuming `audios` are already `(mel_frames, mel_channels)` arrays.
    
    # If the collate_fn is meant to process raw audio to mel, that's a different setup.
    # For this example, let's assume raw audio is provided, and the model's `mel_spec` handles it.
    # However, the MockDataLoader will provide already-mel, so this needs to be consistent.
    
    # Let's align `collate_fn` with the `MockDynamicBatchDataLoader` output for `audio` and `audio_lengths`
    # which is `(batch, mel_frames, mel_channels)` for audio and `(batch,)` for lengths.
    
    # If `audios` are raw audio waveforms from `DummyDataset`:
    # max_audio_samples = max(len(audio) for audio in audios)
    # padded_audios = jnp.stack([jnp.pad(audio, (0, max_audio_samples - len(audio)), constant_values=0).astype(jnp.float32) for audio in audios])
    # audio_lengths = jnp.array([len(audio) for audio in audios], dtype=jnp.int32)

    # For the MockDataLoader that returns MEL directly:
    padded_audios = jnp.stack(audios) # Assuming audios are already padded to same length by dataset or previous step
    audio_lengths = jnp.array([item["audio"].shape[0] for item in batch_items], dtype=jnp.int32) # Get length of MEL frames
    
    return {
        "text": padded_texts,
        "audio": padded_audios, # This is now MEL features, not raw waveform
        "audio_lengths": audio_lengths, # These are MEL frame lengths
    }

# --- Dummy config and dataset loader ---



# --- Main execution block ---
if __name__ == '__main__':
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2" # Simulate 2 devices
    
    # 1. Device and Mesh Setup (needed upfront for sharding)
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), ('data',))
    
    # 2. Get Model Configuration
    GermanTTS_config_dict = get_config_GermanTTS()
    
    # 3. Global RNG Key for initial model/optimizer parameter generation
    init_rng_key = jax.random.PRNGKey(global_seed)
    init_rng_key, model_init_rng_params = jax.random.split(init_rng_key)
    init_rng_key, model_init_rng_dropout = jax.random.split(init_rng_key)
    model_init_rngs = nnx.Rngs(params=model_init_rng_params, dropout=model_init_rng_dropout)

    # 4. Initialize DiT and GermanTTS models, and put their parameters on the device mesh
    # This involves creating dummy instances, then using tree_map with jax.device_put
    
    # Initialize DiT and GermanTTS on host first
    # DiT parameters
    dit_instance_host = DiT(
        dim=GermanTTS_config_dict['transformer_dim'],
        num_heads=GermanTTS_config_dict['transformer_num_heads'],
        num_layers=GermanTTS_config_dict['transformer_num_layers'],
        rngs=model_init_rngs
    )
    
    # GermanTTS requires the DiT instance, and then its own MelSpectrogram
    # GermanTTS also has its own parameters (for MelSpectrogram) and uses RNGs
    # So we need to carefully compose them.
    # A cleaner way with NNX if you are passing modules:
    
    # First, create a "template" of variables with correct sharding for DiT
    # This creates the variable tree.
    dit_template_vars = nnx.create_variables(DiT)(
        dim=GermanTTS_config_dict['transformer_dim'],
        num_heads=GermanTTS_config_dict['transformer_num_heads'],
        num_layers=GermanTTS_config_dict['transformer_num_layers'],
        rngs=model_init_rngs
    )
    
    # Then put these variables on device with replicated sharding
    sharded_dit_vars = dit_template_vars.replace(
        params=jax.tree_map(lambda x: jax.device_put(x, P(), mesh), dit_template_vars.params)
    )
    
    # Create the actual sharded DiT instance
    sharded_dit_model = DiT(
        dim=GermanTTS_config_dict['transformer_dim'],
        num_heads=GermanTTS_config_dict['transformer_num_heads'],
        num_layers=GermanTTS_config_dict['transformer_num_layers'],
        rngs=model_init_rngs, # Can still pass rngs for potential usage inside __call__ if needed
        _variables=sharded_dit_vars # Provide the pre-sharded variables
    )

    # Now, initialize GermanTTS using the sharded DiT instance, and its own MelSpec
    german_tts_template_vars = nnx.create_variables(GermanTTS)(
        transformer=sharded_dit_model, # Pass the already sharded DiT model
        tokenizer=None, # your actual tokenizer here
        num_mel_channels=GermanTTS_config_dict['num_mel_channels'],
        audio_drop_prob=GermanTTS_config_dict['audio_drop_prob'],
        cond_drop_prob=GermanTTS_config_dict['cond_drop_prob'],
        frac_lengths_mask=GermanTTS_config_dict['frac_lengths_mask'],
        sample_rate=GermanTTS_config_dict['sample_rate'],
        n_fft=GermanTTS_config_dict['n_fft'],
        hop_length=GermanTTS_config_dict['hop_length'],
        rngs=model_init_rngs # This rngs is for MelSpectrogram's internal params
    )
    
    sharded_german_tts_vars = german_tts_template_vars.replace(
        params=jax.tree_map(lambda x: jax.device_put(x, P(), mesh), german_tts_template_vars.params)
    )
    
    model = GermanTTS(
        transformer=sharded_dit_model, # This model already has sharded parameters
        tokenizer=None,
        num_mel_channels=GermanTTS_config_dict['num_mel_channels'],
        audio_drop_prob=GermanTTS_config_dict['audio_drop_prob'],
        cond_drop_prob=GermanTTS_config_dict['cond_drop_prob'],
        frac_lengths_mask=GermanTTS_config_dict['frac_lengths_mask'],
        sample_rate=GermanTTS_config_dict['sample_rate'],
        n_fft=GermanTTS_config_dict['n_fft'],
        hop_length=GermanTTS_config_dict['hop_length'],
        rngs=model_init_rngs,
        _variables=sharded_german_tts_vars # Only GermanTTS's direct variables (MelSpec) need this
    )

    # 5. Initialize EMA model if used
    ema_model = None
    if GermanTTS_config_dict['use_mg_loss']: # Assuming use_mg_loss is part of config or derived
        ema_german_tts_template_vars = nnx.create_variables(GermanTTS)(
            transformer=sharded_dit_model, # EMA transformer also takes the sharded DiT template
            tokenizer=None,
            num_mel_channels=GermanTTS_config_dict['num_mel_channels'],
            audio_drop_prob=GermanTTS_config_dict['audio_drop_prob'],
            cond_drop_prob=GermanTTS_config_dict['cond_drop_prob'],
            frac_lengths_mask=GermanTTS_config_dict['frac_lengths_mask'],
            sample_rate=GermanTTS_config_dict['sample_rate'],
            n_fft=GermanTTS_config_dict['n_fft'],
            hop_length=GermanTTS_config_dict['hop_length'],
            rngs=model_init_rngs # Use same RNG for EMA init
        )
        sharded_ema_german_tts_vars = ema_german_tts_template_vars.replace(
            params=jax.tree_map(lambda x: jax.device_put(x, P(), mesh), ema_german_tts_template_vars.params)
        )
        ema_model = GermanTTS(
            transformer=sharded_dit_model,
            tokenizer=None,
            num_mel_channels=GermanTTS_config_dict['num_mel_channels'],
            audio_drop_prob=GermanTTS_config_dict['audio_drop_prob'],
            cond_drop_prob=GermanTTS_config_dict['cond_drop_prob'],
            frac_lengths_mask=GermanTTS_config_dict['frac_lengths_mask'],
            sample_rate=GermanTTS_config_dict['sample_rate'],
            n_fft=GermanTTS_config_dict['n_fft'],
            hop_length=GermanTTS_config_dict['hop_length'],
            rngs=model_init_rngs,
            _variables=sharded_ema_german_tts_vars
        )

    # 6. Setup Optimizer
    base_optimizer_config = optax.adamw(learning_rate=1e-4) # Will be overwritten by scheduler in trainer
    optimizer = nnx.Optimizer(
        base_optimizer_config,
        model # Bind to the model whose parameters are already sharded
    )
    # Explicitly put optimizer state (m, v, count) onto devices with P() sharding
    optimizer.state = jax.tree_map(
        lambda x: jax.device_put(x, P(), mesh),
        optimizer.state
    )

    # 7. Instantiate the Trainer
    trainer = GermanTTSTrainerJax(
        model=model,
        optimizer=optimizer,
        total_steps=1000,
        num_warmup_steps=100,
        use_mg_loss=GermanTTS_config_dict['use_mg_loss'], # Ensure this aligns with your config
        ema_decay=0.9999,
        max_grad_norm=1.0,
        ema_model=ema_model, # Pass the EMA model
        checkpoint_dir='./checkpoints_auto_parallel', # Consistent checkpoint dir
    )

    train_dataset = Huggingface_Datasetloader() # Instantiate the dummy dataset
    
    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        batch_size=num_devices * 2, # Global batch size, e.g., 2 samples per device
        max_batch_frames=24000 * 10,
        save_step=100,
        epochs=3, # Add epochs here if you want to control it externally
    )
    print("Training complete.")