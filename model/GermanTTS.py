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
from tokenize import group
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from flax import nnx

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

class GRN(nnx.Module):

    
    def __init__(self, dim:int):
        self.gamma = nnx.Param("gamma", jnp.zeros(1, 1, dim))
        self.beta = nnx.Param("beta", jnp.zeros(1, 1, dim))
    
    def __call__(self, x):
        Gx = jnp.linalg.norm(x, "fro", axis=1, keepdim = True)
        Gx_old = jnp.array(Gx)
        Nx = Gx / Gx_old.mean(axis = 1, keepdims=True) + 1e-6
        return self.gamma * (x * Nx) + self.beta + x 
        
        
class ConvNeXtV2Block(nnx.Module):

    
    def __init__(self, dim:int, intermidiate_dim:int , dilation:int = 1):
        
        # depthwise conv
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nnx.Conv(dim, dim, kernel_size = 7, padding = padding, dilation = dilation)
        self.norm = nnx.LayerNorm(dim, eps = 1e-6) 
    
        # pointwise conv
        self.pwconv1 = nnx.Linear(dim, intermidiate_dim)
        self.act = nnx.gelu()
        self.grn = GRN(intermidiate_dim)
        self.pwconv2 = nnx.Linear(intermidiate_dim, dim)
    

    def __call__(self, x:jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = rearrange(x, "b n d -> b d n")
        x = self.dwconv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x
        

class SinusPositonalEmbedding(nnx.Module):
    def __init__(self, dim:int):
        self.dim = dim
        
    def __call__(self, x:jnp.ndarray, scale = 1000):
        device = x.device
        half_dim = self.dim // 2
        emb = jnp.log(10_000) / half_dim - 1
        emb = jnp.exp(jnp.arange(half_dim, device = device).float() * -emb)
        
        
        


import jax.numpy as jnp
import jax
jax.devices("cpu")
arr = jnp.array([1,2,3])
print(arr.device)







































































class GermanTTS(Module)