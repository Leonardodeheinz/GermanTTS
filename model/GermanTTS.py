"""
ein notation:
b - batch
n - sequence
f - frequency token dimension
nt - text sequence
nw - raw wave length
d - dimension
dt - dimension text
"""
# neural network libaries
from ast import Module
from statistics import mean
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# type hinting
from jaxtyping import ArrayLike
from jax import Array

# tensor helpers 
from einops import rearrange, reduce, repeat
import einx

# retransforming mel-spectogram with neural voice encoder
from vocos import Vocos # is pretrained

# logging if needed
from loguru import logging

# for the Rottary Embdedding 
from transformers import RoFormerModel

# import here the helpers functions

def exists(v):
    return v is not None

def default(v,d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num & den) == 0

def xnor(x,y):
    return not (x ^ y)


















# GRN response normalization layer

class GRN(nn.Module):
    dim:int 
    
    def setup(self):
        self.gamma = self.param("gamma", jnp.zeros(1, 1, self.dim))
        self.beta = self.param("beta", jnp.zeros(1, 1, self.dim))
        
    @nn.compact
    def __call__(self, x):
        Gx = jnp.linalg.norm(x, "fro", axis=1, keepdim = True)
        Gx_old = jnp.array(Gx)
        Nx = Gx / Gx_old.mean(axis = 1, keepdims=True) + 1e-6
        return self.gamma * (x * Nx) + self.beta + x 
        
        

















































































class GermanTTS(Module)