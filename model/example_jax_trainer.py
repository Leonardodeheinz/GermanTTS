import jax
import jax.numpy as jnp
import optax
from flax import nnx
import orbax.checkpoint as ocp
from functools import partial
import os
from tqdm import tqdm
from typing import Any, Callable, Dict, Tuple, Union, List


# --- Constants (mocked for example) ---
HOP_LENGTH = 256 # Example value for audio to mel length conversion
NUM_MEL_CHANNELS = 100 # Example mel channels (d in PyTorch code's float["b n d"])



# --- NNX Nanospeech Model Definition (now encapsulating the Transformer) ---
class Nanospeech(nnx.Module):
    """
    Nanospeech model that integrates mel conversion and the core Transformer.
    This __call__ should produce the model's direct prediction, not handle loss.
    """
    tokenizer: Any
    num_channels: int # This refers to the output channels of mel_spec (d in float["b n d"])
    audio_drop_prob: float
    cond_drop_prob: float
    frac_lengths_mask: Tuple[float, float] # (min_frac, max_frac) for masking

    mel_spec: MelSpectrogram
    transformer: Transformer

    def __init__(
        self,
        num_mel_channels: int,
        audio_drop_prob: float,
        cond_drop_prob: float,
        frac_lengths_mask: Tuple[float, float],
        transformer_dim: int,
        transformer_num_heads: int,
        transformer_num_layers: int,
        *,
        rngs: nnx.Rngs,
        tokenizer: Any = None,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.num_channels = num_mel_channels
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

        # Initialize sub-modules within NNX
        self.mel_spec = MelSpectrogram(
            n_mels=num_mel_channels,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            rngs=rngs # Pass rngs to sub-modules as well
        )
        self.transformer = Transformer(
            dim=transformer_dim,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
            rngs=rngs # Pass rngs to sub-modules as well
        )

    def __call__(
        self,
        inp: jnp.ndarray, # mel or raw wave
        text: jnp.ndarray, # tokenized text (already int array)
        time: jnp.ndarray, # Timestep `t`
        *,
        lens: jnp.ndarray = None,
        # NO ema_model here in __call__! It's for loss.
        # Drop flags are also passed here as they affect the core prediction
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        mask: jnp.ndarray = None, # Audio mask (derived from lens)
        cond: jnp.ndarray = None, # Infilling condition
    ) -> jnp.ndarray:
        """
        Forward pass of the Nanospeech model, producing the prediction `pred`.
        This is essentially the `self.transformer(...)` part of your original forward.
        """
        # Handle raw wave -> mel spectrogram
        if inp.ndim == 2: # Assuming (B, NumSamples)
            inp = self.mel_spec(inp) # (B, NumFrames, NumMels)
            # In your PyTorch, it's `rearrange(inp, "b d n -> b n d")`
            # If your mel_spec returns (B, NumMels, NumFrames), then you need to transpose
            # For our dummy, we'll assume it returns (B, NumFrames, NumMels) as needed
            assert inp.shape[-1] == self.num_channels, f"Mel channels mismatch: {inp.shape[-1]} != {self.num_channels}"

        # If `mask` or `cond` are not provided, it means this call is for
        # direct prediction or an intermediate step.
        # In training, `calculate_loss` will generate these.
        # In inference, you'd generate them based on your use case.
        
        # The JAX Transformer will take these processed inputs.
        # It's up to the loss function to prepare `w`, `cond` (the masked input),
        # `text`, `time`, and the drop flags based on the training logic.
        
        # Here, `inp` is effectively `w` or the input for `transformer`.
        # `cond` is the conditional input.
        
        pred = self.transformer(
            x=inp, # This `inp` is already `w` from the loss function's perspective
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
        )
        return pred


# --- 2. Loss Function (Now much more substantial) ---
def calculate_loss(
    main_model: Nanospeech, # The main trainable NNX model instance
    batch: Dict[str, jnp.ndarray],
    *,
    rng_key: jax.random.PRNGKey, # Must pass RNG for stochastic operations
    ema_model: Nanospeech = None, # The EMA NNX model instance
    use_flow_matching_loss: bool = False,
) -> Tuple[jnp.ndarray, jax.random.PRNGKey]: # Return updated RNG key
    """
    Calculates the loss for the Nanospeech model, implementing the logic
    from the original PyTorch forward method that is not part of the core model prediction.
    """
    text_input, audio_input, audio_lengths = batch["text"], batch["audio"], batch["audio_lengths"]

    # --- JAX-equivalent of PyTorch's `forward` data preparation ---

    # Handle raw wave -> mel conversion (this is part of the `Nanospeech` model's __call__)
    # It's better if `Nanospeech.__call__` itself handles this.
    # For now, `audio_input` would be raw wave or mel.
    # The `Nanospeech` model is expected to handle this conversion internally if needed.
    # We pass `audio_input` to `model` and it decides if it needs `mel_spec`.

    batch_size, seq_len = audio_input.shape[0], audio_input.shape[1] if audio_input.ndim == 3 else audio_input.shape[1] # For raw wave or mel

    # Text tokenization is expected to be done in `collate_fn` now.
    # `text_input` is already an `int["b nt"]` JAX array.

    # Lens and mask
    if not exists(audio_lengths):
        audio_lengths = jnp.full((batch_size,), seq_len, dtype=jnp.int32)

    mask = lens_to_mask(audio_lengths, length=seq_len if audio_input.ndim == 2 else audio_input.shape[1]) # Mask for raw audio or mel frames

    # Get a random span to mask out for training conditionally
    rng_key, subkey_frac = jax.random.split(rng_key)
    frac_lengths = jax.random.uniform(subkey_frac, (batch_size,), 
                                     minval=main_model.frac_lengths_mask[0], 
                                     maxval=main_model.frac_lengths_mask[1])
    
    # Need to be careful with mask_from_frac_lengths if it relies on random.
    # It now takes a JAX RNG.
    rand_span_mask = mask_from_frac_lengths(audio_lengths, frac_lengths, seq_len if audio_input.ndim == 2 else audio_input.shape[1])

    if exists(mask):
        rand_span_mask = rand_span_mask & mask

    # mel is x1 (or whatever the processed input is)
    x1 = main_model.mel_spec(audio_input) if audio_input.ndim == 2 else audio_input
    if x1.ndim == 2: # If it's (B, S_raw), reshape to (B, S_mel, Mel_dim)
        x1 = x1[:, :, None] # Dummy: assumes mel_spec returns (B, S_raw) and we want (B, S_raw, 1)

    # x0 is gaussian noise
    rng_key, subkey_x0 = jax.random.split(rng_key)
    x0 = jax.random.normal(subkey_x0, x1.shape, dtype=x1.dtype)

    # Timestep
    rng_key, subkey_time = jax.random.split(rng_key)
    time = jax.random.uniform(subkey_time, (batch_size,), dtype=x1.dtype)

    # Sample x(t)
    t = time[:, None, None] # Reshape for broadcasting (b -> b 1 1)
    w = (1 - t) * x0 + t * x1
    flow = x1 - x0 # Target flow

    # Only predict what is within the random mask span for infilling
    # This means `cond` is `x1` where `rand_span_mask` is False (i.e., known regions)
    # and `zeros_like` where `rand_span_mask` is True (unknown regions)
    # The PyTorch code `torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)`
    # means: if rand_span_mask is True (masked out), then 0, else x1.
    cond = jnp.where(rand_span_mask[:, :, None], jnp.zeros_like(x1), x1) # Ensure dim for rand_span_mask matches x1

    # Classifier-Free Guidance (CFG) / Model Guidance (MG) Setup
    rng_key, subkey_cfg = jax.random.split(rng_key)
    rng_key, subkey_cond_drop = jax.random.split(rng_key) # For random drop flags

    # Determine drop flags based on model's probabilities
    rand_audio_drop = jax.random.uniform(subkey_cfg, (1,))[0]
    rand_cond_drop = jax.random.uniform(subkey_cond_drop, (1,))[0]
    
    # Drop flags are Boolean for JAX/NNX
    drop_audio_cond_main = rand_audio_drop < main_model.audio_drop_prob
    drop_text_main = rand_cond_drop < main_model.cond_drop_prob
    drop_audio_cond_main = drop_audio_cond_main | drop_text_main # Combine

    # Prediction from the main model
    # The `__call__` of Nanospeech will handle `mel_spec` and then `transformer`
    pred = main_model(
        inp=w.squeeze(-1), # Pass `w` to the model. Squeeze if transformer expects 2D
        cond=cond.squeeze(-1), # Squeeze if transformer expects 2D
        text=text_input,
        time=time,
        lens=audio_lengths, # Pass original lengths
        drop_audio_cond=drop_audio_cond_main,
        drop_text=drop_text_main,
        mask=mask,
    )
    
    # Ensure `pred` and `flow` have compatible shapes for MSE Loss
    # `pred` will be (B, S_mel) and `flow` will be (B, S_mel)
    # The `squeezes` above aim to align this.
    pred = pred[:, :x1.shape[1]] # Trim to match x1 length after squeeze

    # Model-guidance loss (MG Loss)
    if use_flow_matching_loss and ema_model is not None:
        # Note: In PyTorch, it's `ema_model.ema_model.transformer`.
        # Here, `ema_model` is directly the `Nanospeech` instance.
        # We need to call its `transformer` (or its main `__call__`) with no drops.
        
        # Predictions from EMA model (without gradient tracking)
        # Using `nnx.no_grad` context manager if needed, but nnx.jit handles this for non-params
        
        guidance_cond = ema_model(
            inp=w.squeeze(-1),
            cond=cond.squeeze(-1),
            text=text_input,
            time=time,
            lens=audio_lengths,
            drop_audio_cond=False, # No drop for conditional path
            drop_text=False,      # No drop for conditional path
            mask=mask,
        )

        guidance_uncond = ema_model(
            inp=w.squeeze(-1),
            cond=cond.squeeze(-1),
            text=text_input, # Text still passed, but `drop_text=True` makes it ignored by transformer
            time=time,
            lens=audio_lengths,
            drop_audio_cond=True, # Drop audio condition for unconditional path
            drop_text=True,       # Drop text for unconditional path
            mask=mask,
        )
        
        # Ensure guidance predictions match length of flow target
        guidance_cond = guidance_cond[:, :flow.shape[1]]
        guidance_uncond = guidance_uncond[:, :flow.shape[1]]

        guidance_scale = jnp.where(time < 0.75, 0.5, 0.0)[:, None] # Add dim for broadcasting
        guidance = (guidance_cond - guidance_uncond) * guidance_scale
        flow = flow.squeeze(-1) + guidance # Ensure flow and guidance have compatible dims

    else: # If not using MG loss, then the main model's drop flags apply
        flow = flow.squeeze(-1) # Ensure flow is 2D for loss

    # Flow matching loss
    # `rand_span_mask` needs to be 2D for indexing `loss[rand_span_mask]`
    loss = jnp.square(pred - flow) # MSE loss
    
    # Ensure rand_span_mask matches the sequence length of pred/flow
    # This is tricky if input raw_wave vs mel leads to different sequence lengths
    # Assuming `rand_span_mask` matches the mel frame length (seq_len of x1)
    
    # Mask the loss
    loss = loss * rand_span_mask # Zero out loss outside the span
    loss = jnp.sum(loss) / jnp.sum(rand_span_mask) # Mean over the masked span only

    return loss, rng_key # Return updated RNG key


# --- 3. Training Step Function ---
# `@nnx.jit` handles the split/merge of the nnx.Module and optimizer.
@nnx.jit
def train_step(model: Nanospeech, optimizer: nnx.Optimizer, batch: Dict[str, jnp.ndarray], 
               max_grad_norm: float, use_flow_matching_loss: bool, ema_model: Nanospeech = None,
               learning_rate_schedule_fn: Callable[[int], float] = None,
               rng_key: jax.random.PRNGKey = None # Passed from trainer
               ) -> Tuple[jnp.ndarray, float, jax.random.PRNGKey]:
    """
    Performs one training step using NNX modules and optimizer.
    """
    # Use nnx.value_and_grad for automatic differentiation
    # The `calculate_loss` now needs the RNG key.
    (loss, new_rng_key), grads = nnx.value_and_grad(calculate_loss, has_aux=True)(
        model, batch, rng_key=rng_key, ema_model=ema_model, use_flow_matching_loss=use_flow_matching_loss
    )

    # Gradient clipping is handled within the optax.chain of the optimizer.

    optimizer.update(grads)

    current_lr = learning_rate_schedule_fn(optimizer.step)

    return loss, current_lr, new_rng_key


# --- 4. NanospeechTrainerFlax Class ---
class NanospeechTrainerFlax:
    def __init__(
        self,
        model_config: Dict[str, Any], # Pass model configuration as a dict
        total_steps: int,
        num_warmup_steps: int = 2_000,
        ema_decay: float = 0.999,
        max_grad_norm: float = 1.0,
        sample_rate: int = 24_000,
        use_flow_matching_loss: bool = False,
    ):
        self.target_sample_rate = sample_rate
        self.total_steps = total_steps
        self.num_warmup_steps = num_warmup_steps
        self.use_flow_matching_loss = use_flow_matching_loss
        self.max_grad_norm = max_grad_norm
        self.ema_decay = ema_decay
        self.model_config = model_config # Store model config

        # --- JAX PRNGKey Management ---
        # Initialize a base RNG key for the entire trainer.
        # This will be split for different stochastic operations (model init, data sampling, etc.)
        self.rng_key = jax.random.PRNGKey(42) # Master RNG

        # --- NNX Specific: Model Initialization (including EMA) ---
        # Initialize the main model
        self.rng_key, model_init_rngs = nnx.split(self.rng_key, {'params': 0, 'dropout': 1})
        self.model = Nanospeech(
            num_mel_channels=model_config['num_mel_channels'],
            audio_drop_prob=model_config['audio_drop_prob'],
            cond_drop_prob=model_config['cond_drop_prob'],
            frac_lengths_mask=model_config['frac_lengths_mask'],
            transformer_dim=model_config['transformer_dim'],
            transformer_num_heads=model_config['transformer_num_heads'],
            transformer_num_layers=model_config['transformer_num_layers'],
            rngs=model_init_rngs,
            tokenizer=model_config['tokenizer'],
            sample_rate=sample_rate,
        )

        if self.use_flow_matching_loss:
            # Initialize EMA model by cloning the main model's structure and initial parameters
            # and then we'll manually update its parameters in the loop.
            self.ema_model = nnx.clone(self.model) # Clone creates a new instance with copied params
        else:
            self.ema_model = None

        # Define the learning rate schedule using optax
        warmup_schedule = optax.linear_schedule(
            init_value=1e-8, end_value=1.0, transition_steps=self.num_warmup_steps
        )
        decay_steps = self.total_steps - self.num_warmup_steps
        decay_schedule = optax.linear_schedule(
            init_value=1.0, end_value=1e-8, transition_steps=decay_steps
        )
        self.learning_rate_schedule_fn = optax.join_schedules(
            schedules=[warmup_schedule, decay_schedule],
            boundaries=[self.num_warmup_steps]
        )
        
        # --- NNX Specific: Initialize Optimizer ---
        self.optimizer = nnx.Optimizer(
            optax.chain(
                optax.adamw(learning_rate=self.learning_rate_schedule_fn),
                optax.clip_by_global_norm(self.max_grad_norm),
            ),
            self.model, # The optimizer is tied to the main model
        )

        # Orbax checkpoint manager
        self.ckpt_manager = ocp.CheckpointManager(
            './checkpoints',
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=1,
                max_to_keep=5,
                checkpoint_name='nanospeech_nnx',
            )
        )

    def save_checkpoint(self, step: int):
        if jax.process_index() == 0:
            ckpt = {
                'model': self.model,
                'optimizer': self.optimizer,
                'rng_key': self.rng_key, # Save the current RNG key state
            }
            if self.ema_model:
                ckpt['ema_model'] = self.ema_model
            
            self.ckpt_manager.save(step, ckpt)
            print(f"Saved checkpoint at step {step}")

    def load_checkpoint(self, step: int):
        if self.ckpt_manager.latest_step() is None and step == 0:
            return 0 # No checkpoint, start from step 0

        target_step = step if step != 0 else self.ckpt_manager.latest_step()
        if target_step is None:
            return 0
            
        print(f"Loading checkpoint from step {target_step}")
        
        # Initialize empty modules/optimizer with the correct structure for Orbax
        # This can be made more robust by having a `create_empty_model_and_optimizer` helper
        # that takes the same configs as __init__.
        rngs = nnx.Rngs(params=jax.random.PRNGKey(0), dropout=jax.random.PRNGKey(1)) # Dummy RNGs for structure
        empty_model = Nanospeech(rngs=rngs, **self.model_config)
        empty_optimizer = nnx.Optimizer(
            optax.chain(
                optax.adamw(learning_rate=self.learning_rate_schedule_fn),
                optax.clip_by_global_norm(self.max_grad_norm),
            ),
            empty_model,
        )
        
        empty_ema_model = None
        if self.ema_model:
            empty_ema_model = nnx.clone(empty_model) # Clone structure of main model

        restored_state = self.ckpt_manager.restore(
            target_step,
            args=ocp.args.StandardRestore(
                {
                    'model': empty_model,
                    'optimizer': empty_optimizer,
                    'ema_model': empty_ema_model,
                    'rng_key': jax.random.PRNGKey(0), # Dummy RNG key to load into
                }
            )
        )
        
        self.model = restored_state['model']
        self.optimizer = restored_state['optimizer']
        self.rng_key = restored_state['rng_key'] # Restore the RNG key
        if self.ema_model:
            self.ema_model = restored_state['ema_model']
        
        return self.optimizer.step

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
        start_step = self.load_checkpoint(restore_step)
        self.optimizer.step = start_step # Ensure optimizer's step is synced

        global_step = self.optimizer.step

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
            }
            print(f"Hyperparameters: {hps}")


        class MockDynamicBatchDataLoader:
            def __init__(self, dataset, batch_size, max_batch_frames, max_duration, collate_fn):
                self.dataset = dataset
                self.batch_size = batch_size
                self.max_batch_frames = max_batch_frames
                self.max_duration = max_duration
                self.collate_fn = collate_fn
                
            def __iter__(self):
                for i in range(200): # Larger dummy loop for more steps
                    # Create dummy data for a batch
                    dummy_text_batch = jnp.array([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=jnp.int32)
                    dummy_audio_batch = jnp.zeros((2, 24000), dtype=jnp.float32) # Raw audio
                    dummy_audio_lengths_batch = jnp.array([24000, 24000], dtype=jnp.int32)
                    
                    batch = {
                        "text": dummy_text_batch,
                        "audio": dummy_audio_batch,
                        "audio_lengths": dummy_audio_lengths_batch,
                    }
                    yield batch
                    
            def __len__(self):
                return 200

        train_collate_fn = partial(collate_fn, tokenizer=self.model.tokenizer)
        train_dataloader = MockDynamicBatchDataLoader(
            train_dataset,
            batch_size=batch_size,
            max_batch_frames=max_batch_frames,
            max_duration=max_duration,
            collate_fn=train_collate_fn
        )

        epoch = 0
        total_frames = 0

        while global_step < self.total_steps:
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}",
                unit="batch",
                disable=jax.process_index() != 0,
            )
            epoch_loss = 0.0
            item_count = 0

            for batch in progress_bar:
                if global_step >= self.total_steps:
                    break

                # --- JAX PRNGKey for stochastic ops in train_step/calculate_loss ---
                # Split a new key for each step, and update the trainer's master key
                self.rng_key, step_rng_key = jax.random.split(self.rng_key)

                loss_item, current_lr, new_step_rng_key = train_step(
                    self.model,
                    self.optimizer,
                    batch,
                    self.max_grad_norm,
                    self.use_flow_matching_loss,
                    self.ema_model,
                    self.learning_rate_schedule_fn,
                    step_rng_key, # Pass the RNG key to train_step
                )
                self.rng_key = new_step_rng_key # Update master RNG with the returned key

                # --- Manual EMA update for Mean Teacher ---
                if self.ema_model and self.use_flow_matching_loss:
                    main_params = self.model.filter(nnx.Param)
                    ema_params = self.ema_model.filter(nnx.Param)
                    for name, main_var in main_params.items():
                        ema_var = ema_params[name]
                        ema_var.value = ema_var.value * self.ema_decay + main_var.value * (1.0 - self.ema_decay)


                total_frames += jnp.sum(batch["audio_lengths"] / HOP_LENGTH).item()
                
                if jax.process_index() == 0:
                    log_data = {
                        "loss": float(loss_item), # Ensure logging scalar Python float
                        "lr": float(current_lr),
                        "frames": total_frames,
                    }
                    progress_bar.set_postfix(loss=f"{loss_item:.4f}", lr=f"{current_lr:.2e}", frames=f"{total_frames / 1e6:.2f}M")
                    # print(f"Step {global_step}, Loss: {loss_item:.4f}, LR: {current_lr:.6f}, Frames: {total_frames}")

                global_step = self.optimizer.step

                epoch_loss += loss_item
                item_count += 1

                if global_step % save_step == 0:
                    self.save_checkpoint(global_step)

            if global_step >= self.total_steps:
                break

            epoch_loss /= item_count if item_count > 0 else 1.0
            if jax.process_index() == 0:
                print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            epoch += 1

        if jax.process_index() == 0:
            self.ckpt_manager.wait_until_finished()
            print("Training finished.")

# Helper function placeholder for collate_fn
def collate_fn(batch_items, tokenizer):
    texts = [item["json"]["text"] for item in batch_items]
    audios = [item["audio"] for item in batch_items] # Now raw audio
    
    # Tokenize texts - simplified dummy for demonstration
    tokenized_texts_numerical = [jnp.array([ord(c) for c in s], dtype=jnp.int32) for s in texts]
    
    max_text_len = max(len(t) for t in tokenized_texts_numerical)
    padded_texts = jnp.stack([jnp.pad(t, (0, max_text_len - len(t)), constant_values=0) for t in tokenized_texts_numerical])

    max_audio_len = max(len(audio) for audio in audios)
    # Ensure raw audio is float32, as MelSpectrogram might expect it, or convert
    padded_audios = jnp.stack([jnp.pad(audio, (0, max_audio_len - len(audio)), constant_values=0).astype(jnp.float32) for audio in audios])
    
    audio_lengths = jnp.array([len(audio) for audio in audios], dtype=jnp.int32)

    return {
        "text": padded_texts,
        "audio": padded_audios,
        "audio_lengths": audio_lengths,
    }

# --- Example Usage (main block) ---
if __name__ == '__main__':
    class DummyNanospeech(Nanospeech):
        pass

    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            return {
                "json": {"text": f"hello world {idx}", "duration": 5.0},
                "audio": jnp.zeros(24000 * 5, dtype=jnp.float32) # Raw audio input
            }
        
        def shuffle(self, seed):
            return self

        def filter(self, fn):
            return self

    # Define model configuration
    nanospeech_model_config = {
        'num_mel_channels': NUM_MEL_CHANNELS,
        'audio_drop_prob': 0.1,
        'cond_drop_prob': 0.1,
        'frac_lengths_mask': (0.2, 0.5), # min and max fraction for span masking
        'transformer_dim': 512,
        'transformer_num_heads': 8,
        'transformer_num_layers': 4,
        'tokenizer': None, # Replace with your actual tokenizer object
    }

    trainer = NanospeechTrainerFlax(
        model_config=nanospeech_model_config,
        total_steps=1000,
        num_warmup_steps=100,
        use_flow_matching_loss=True,
        ema_decay=0.9999,
        max_grad_norm=1.0,
        sample_rate=24000,
    )

    train_dataset = DummyDataset(size=500)

    print("Starting training...")
    trainer.train(
        train_dataset=train_dataset,
        batch_size=2,
        max_batch_frames=24000 * 10,
    )
    print("Training complete.")