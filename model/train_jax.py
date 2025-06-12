# Basic Python Imports
from __future__ import annotations
from functools import partial
import os
from typing import Any, Sequence, Tuple, Optional, Dict


# NNs libaries
# jax
import jax
import jax.numpy as jnp

# flax
import flax
from flax import nnx

# optax
import optax
from optax import adamw

#type hinting
from jaxtyping import ArrayLike
#from jax import Array


# tensor helpers 
from einops import rearrange, reduce, repeat # TODO: maybe not needed after all
import einx

# logging if needed
from loguru import logger

# my data_loader
from data.PreProcesesAudio import MelSpec, Resampler
from data.DataLoader import Huggingface_Datasetloader

# Final Model imports
from model.GermanTTS import GermanTTS, DurationPredictor

HOP_LENGTH = 256

class DynamicBatchDataLoader:
    def __init__(self,
                 collate_fn,
                 batch_size = 32,
                 max_batch_frame = 4096,
                 max_duration = None,
                 **dataloader_kwargs,
                 ):
        self.max_batch_frames = max_batch_frame
        self.max_duration = max_duration
        self.dataloader = Huggingface_Datasetloader(
            **dataloader_kwargs
        )
        self.collate_fn = collate_fn
        
    def _collate_wrapper(self, batch):
        return dynamic_batch_collate_fn(batch, self.collate_fn, self.max_batch_frames, self.max_duration)
    
    def __iter__(self):
        batch_iterator = iter(self.dataloader)
        while True:
            try:
                batch = next(batch_iterator)
                
                while batch is None:
                    batch =next(batch_iterator)
                yield batch
            except StopIteration:
                break
        
    
    
    
    
    
def dynamic_batch_collate_fn():
    pass


def collate_fn():
    pass


class GermanTTS_Trainer():
    pass

class DurationTrainer():
    pass