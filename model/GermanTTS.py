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
# neural network libaries
from ast import Module
from asyncio import constants
from os import device_encoding
from re import S
from statistics import mean
from tokenize import group
from typing import Text
from wsgiref import headers
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
from numpy.core.umath import ndarray
from vocos import Vocos # is pretrained

# logging if needed
from loguru import logging

# for the Rottary Embdedding 
from transformers import RoFormerModel

# seeds for random numbers

key = 42


# import here the helpers functions

def exists(v):
    return v is not None

def default(v,d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0


def xnor(x,y):
    return not (x ^ y)

# tensor helpers
@jax.jit
def lens_to_mask( t:int["b"],length: int | None = None,)-> bool["b n"]:
    if not exists(length):
        length = jnp.amax(t)
    
    seq = jnp.arange(length, device = t.device)
    return seq[None, :] < t[:, None]

@jax.jit
def mask_from_start_end_indicies(start:int["b"], end:int["b"], max_seq_len:int):
    seq = jnp.arange(max_seq_len, device=start.device, dtype = jnp.int64)
    start_mask = seq[None,:] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask

@jax.jit
def mask_from_frac_lengths(seq_len: int["b"], frac_lengths:float["b"], max_seq_len:int, seed = key):
    lengths = jnp.int64(frac_lengths * seq_len)
    max_start = seq_len - lengths
    
    key = jax.random.key/(seed)
    rand = jax.random.normal(key, shape = (len(frac_lengths), ))
    
    start = jnp.clip(jnp.int64(max_start * rand), min = 0)
    
    end = start + lengths
    
    return mask_from_start_end_indicies(start, end, max_seq_len)

# create Buffer Class so that untrained parameters are also recognized
class Buffer(nnx.Variable):
    pass


class RotaryEmbedding(nnx.Module):
    def __init__(self, dim, use_xpos, scale_base, interpolation_factor, base, base_rescale_factor):
        
        base *= base_rescale_factor ** (dim / (dim-2))
        
        inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2, dtype = jnp.float32) / dim))
        self.buffer_inv_freq = Buffer(inv_freq)
        
        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor
        
        if not use_xpos:
            self.buffer_scale = Buffer(None)
            return
        
        scale = (jnp.arange(0, dim, 2 + 0.4 * dim) / (1.4 * dim))
        
        self.scale_base = scale_base
        self.buffer_scale = Buffer(scale)
        
    
    def forward_from_seq_len(self, seq_len):
        
        t = jnp.arange(seq_len)
        return t
    
    def __call__ (self, t:jnp.ndarray):
        max_pos = t.max() + 1
        
        if t.ndim == 3:
            t = rearrange(t, "n -> 1 n")
            
        freqs = jnp.einsum("bi,j->bij", t.astype(jnp.float32), self.buffer_inv_freq) / self.interpolation_factor
        freqs = jnp.stack((freqs, freqs), axis = -1)
        freqs = rearrange(freqs, "... d r -> ... (d r)")
        
        if not exists(self.buffer_scale):
            return freqs, 1.0
        
        power = (t - (max_pos // 2)) / self.buffer_scale
        scale = self.buffer_scale ** rearrange(power, "n -> n 1")
        scale = jnp.stack((scale, scale), dim = -1)
        scale = rearrange(scale, "... d r -> (d r)")
        
        return freqs, scale












# GRN response normalization layer

class GRN(nnx.Module):

    
    def __init__(self, dim:int):
        self.gamma = nnx.Param("gamma", jnp.zeros(1, 1, dim))
        self.beta = nnx.Param("beta", jnp.zeros(1, 1, dim))
    
    def __call__(self, x):
        Gx = jnp.linalg.norm(x, "fro", axis=1, keepdim = True)
        Gx_old = jnp.array(Gx)
        Nx = Gx / Gx_old.mean(axis = 1, keepdims=True) + 1e-6
        out = self.gamma * (x * Nx) + self.beta + x 
        return out
        
    
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
        x = self.dwconv(x)                      #TODO: Check the dimensions of the input and output
        x = rearrange(x, "b d n -> b n d")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x

# Definition of the embeddings

# sinusoidal position embedding

class SinusPositonalEmbedding(nnx.Module):
    def __init__(self, dim:int):
        self.dim = dim
        
    def __call__(self, x:jnp.ndarray, scale = 1000):
        device = x.device
        half_dim = self.dim // 2
        emb = jnp.log(10_000) / half_dim - 1                                # TODO: check if output and input are valid
        emb = jnp.exp(jnp.arange(half_dim, device = device).float() * -emb)
        emb = scale * jnp.expand_dims(x, 1) * jnp.expand_dims(emb, 0)
        emb = jnp.concat((jnp.sin(emb),jnp.cos(emb)), axis = 1)
        return emb

# Convolutional postion embedding

class ConvPositionalEmbedding(nnx.Module):
    def __init__(self, dim:int, kernel_size=31, groups=16):
        assert divisible_by(dim, 2)
        padding_size = kernel_size // 2
        self.conv1d = nnx.Sequential(
            nnx.Conv(dim, dim, kernel_size = kernel_size, groups = groups, padding = padding_size),
            jax.nn.mish(), #noqa F722
            nnx.Conv(dim, dim, kernel_size = kernel_size, groups = groups, padding = padding_size),
            jax.nn.mish()
        )
    def __call__(self, x:float["b n d"], mask:bool["b n"] | None = None):
        if exists(mask):
            mask = mask[..., None]
            jnp.where(mask, x,0.0)
        
        x = rearrange(x, "b n d -> b d n")
        x = self.conv1d(x)                      # TODO: check the permutations, not sure if right, maybe in test_pipeline
        x = rearrange(x, "b d n -> b n d")

        if exists(mask):
            out = jnp.where(mask , x, fill_value = 0.0)      # TODO: here also check the dimensions
            
        return out 
    
    
class TimestepEmbedding(nnx.Module):
    
    def __init__(self, dim, freq_ebmed_dim=256):
        self.time_embed = SinusPositonalEmbedding(freq_ebmed_dim)
        self.time_mlp = nnx.Sequential(nnx.Linear(freq_ebmed_dim, dim), nnx.silu(), nnx.Linear(dim, dim))
        
    def __call__(self, timestep:float["b"]): 
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time
    
# feed forward class 

class FeedForward(nnx.Module):
    
    def __init__(self, dim, dim_out = None, mult = 4, dropout = 0.0, approximate:str = "none"):
        inner_dim = int(dim * mult)
        if not dim_out:
            pass
        else:
            dim_out = dim
        
        activation = nnx.gelu(approximate=approximate)
        project_fn = nnx.Sequential(nnx.Linear(dim, inner_dim), activation)
        self.ff = nnx.Sequential(project_fn, nnx.Dropout(dropout), nnx.Linear(inner_dim, dim_out))

    def forward(self,x):
        return self.ff(x)    
    
# attention

class Attention(nnx.Module):
    
    def __init__(self, dim:int, heads:int = 8, dim_head:int = 64, dropout:float = 0.0):
        """Standard attention Block for our 

        Args:
            dim (int): _description_
            head (int, optional): _description_. Defaults to 8.
            dim_head (int, optional): _description_. Defaults to 64.
            dropout (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        
        self.dim = dim 
        self.heads = heads 
        self.inner_dim = dim_head * heads
        self.droput = dropout
        
        self.to_q = nnx.Linear(dim, self.inner_dim)
        self.to_k = nnx.Linear(dim, self.inner_dim)
        self.to_v = nnx.Linear(dim, self.inner_dim)
        
        self.to_out = nnx.Sequential(nnx.Linear(self.inner_dim, dim, bias = False), nnx.Dropout(dropout))
    
    
    def __call__(self, 
                x:float["b n d"],
                mask: bool["b n"] | None = None,
                rope = None,
                ) -> jnp.array:
        batch_size = x.shape[0]

        query = self.to_q(x)
        key = self.to_k(x)
        values = self.to_v(x)
        
        if rope is not None:        # TODO: change this code block since it is wrong and only coppied from the original.
            # freqs, xpos_scale = rope         
            # q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            # query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            # key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
            query = None
            key = None

    
        query = rearrange(query, "b n (h d) -> b h n d", h=self.heads)
        key =   rearrange(key, "b n (h d) -> b n h d", h=self.heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.heads)
        
        if mask is not None:
            attn_mask = rearrange(mask, "b n -> b 1 1 n")
            attn_mask = attn_mask.expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None
            
        x = nnx.dot_product_attention(query, key, value, bias = None, dropout_rate = 0.8, deterministic = False)
        
        x = rearrange("b h n d -> b n (h d)")
        x = x.to(query.dtype)
        
        if mask is not None:
            mask = jnp.expand_dims(mask, axis = 1)
            x = jnp.where(mask, x, fill_value = 0.0)
            
        return x
        

# text embedding helper funcitons
@jax.jit
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor: float = 1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    # Generate the frequency components
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    
    # Time steps
    t = jnp.arange(end, dtype=jnp.float32)

    # Outer product to get positions * frequencies matrix
    freqs = jnp.outer(t, freqs).astype(jnp.float32)

    # Compute cos and sin components
    freqs_cos = jnp.cos(freqs)
    freqs_sin = jnp.sin(freqs)

    # Concatenate along the last dimension
    return jnp.concatenate([freqs_cos, freqs_sin], axis=-1)


@jax.jit
def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    # Make sure scale has same shape as start
    scale = scale * jnp.ones_like(start, dtype=jnp.float32)
    
    # Broadcast start and range with scale
    pos = start[:, None] + (jnp.arange(length, dtype=jnp.float32)[None, :] * scale[:, None])
    
    # Convert to integer indices (flooring is implicit in cast)
    pos = pos.astype(jnp.int32)
    
    # Clip positions to max_pos - 1
    pos = jnp.where(pos < max_pos, pos, max_pos - 1)
    
    return pos
    
    

class TextEmbedding(nnx.Module):
    
    
    def __init__(self, text_num_embeds, text_dim, conv_layers = 0, conv_mult = 2):
        
        self.text_embed = jnp.take(text_num_embeds + 1, text_dim) # TODO: check how it differs to torch.nn.Embedding
        
        if conv_layers > 0:
            self.extra_modelling = False
            self.precompute_max_pos = 2048      # TODO: check how much it should differ to our, I think, what most audio is not longer than 20 seconds
            self.buffer = {
                "freqs_cis": precompute_freqs_cis(text_dim, self.precompute_max_pos) #precompu     # TODO: create the precompute_freqs_cis function, which precomputes the position encodings for an max size 
            }
    
            self.text_blocks = nnx.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])        
        
        else:
            self.extra_modelling = False
        
        def forward(self, text:int ["b nt"], seq_len, drop_text = False):
            text = text + 1 
            text = text[:, :seq_len]
            batch, text_len = text.shape[0], text.shape[1]
            text = jnp.pad(text, (0, seq_len - text_len), mode = "constant")  # TODO: check if paddig is alright

            if drop_text:
                text = jnp.zeros_like(text, dtype = np.int64)
                
            text = self.text_embed(text)
            
            if self.extra_modelling:
                
                batch_start = jnp.zeros((batch,), dtype = np.int64)
                pox_idx = get_pos_embed_indices(batch_start, seq_len, max_pos = self.precompute_max_pos)
                text_pos_embed = self.buffer["freq_cis"][pox_idx]
                text = text + text_pos_embed
                
                text = self.text_blocks(text)
                
            return text
            
        
# noised Inputaudio and context mising embedding

class InputEmbedding(nnx.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        
        self.proj = nnx.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionalEmbedding(dim = out_dim)

    def __call__(self, 
                 x: float["b n d"],
                 cond:float["b n d"],
                 text_embed:float["b n d"],
                 drop_audio_cond = False,
                 ):
        if drop_audio_cond:
            cond = jnp.zeros_like(cond)
            
            x = self.proj(jnp.concatenate((x, cond, text_embed), axis = -1))
            x = self.conv_pos_embed(x) + x
            
            return x
        


class AdaLayernNormZero(nnx.Module):
    
    def __init__(self, dim:int, dim_condition:int):
     
      
        self.linear = nnx.Linear(dim, dim * 2)
        
        self.norm = nnx.LayerNorm(dim, use_bias = False, use_scale = False,  epsilon = 1e-6)
        
    def __call__(self, x, emb):
        
        emb = self.linear(nnx.silu(emb))
        
        scale, shift = jnp.split(emb, 2, axis=-1)
        
        normed_x = self.norm(x)
        
        scale_expanded = jnp.expand_dims(scale, axis = 1)
        shift_expanded = jnp.expand_dims(shift, axis = 1)
        
        output = normed_x * (1 + scale_expanded) + shift_expanded 
        return output
        


class DiTBlock(nnx.Module):
    def __init__ (self, dim, heads, dim_head, ff_mult = 4, dropout = 0.1):
        self.attn_norm = AdaLayernNormZero(dim)
        self.attn = Attention(dim = dim, heads = heads, dim_head = dim_head, dropout = dropout)
        
        self.ff_norm = nnx.LayerNorm(dim, use_bias = False, use_scale = False, epsilon = 1e-6)
        self.ff = FeedForward(dim = dim, mult = ff_mult, dropout = dropout, approxiamte = "tanh")
    
    def __call__ (self, x, t, mask = None, rope = None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb = t)
        attn_output = self.attn(x=norm, mask = mask, rope = rope)
        x = x + jnp.expand_dims(gate_msa, axis = -1)  * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + jnp.expand_dims(gate_mlp, axis = 1) * ff_output
        
        return x

class DiT(nnx.Module):
    def __init__ (self, *, dim, depth = 8, heads = 8,dim_head = 64, dropout = 0.1, ff_mult = 4, mel_dim = 100, text_num_embeds = 256, text_dim = None, conv_layers = 0):
        
        if text_dim is None:
            text_dim = mel_dim
            
        self.time_embed = TimestepEmbedding(dim)
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        
        self.dim = dim 
        self.depth = depth
        
        self.transformers_blocks = [DiTBlock(dim=dim, heads=heads,dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        
        self.norm_out = AdaLayernNormZero(dim)
        self.proj_out = nnx.Linear(dim, mel_dim, use_bias=False,kernel_init = nnx.initializers.lecun_normal)
        
    def __call__ (self, x:float["b n d"], cond:float["b n d"], text:int["b nt"], time:float["b"], drop_audio_cond, drop_text, mask:bool["b n"] | None = None):
        
        batch, seq_len = x.shape[0], x.shape[1]
        
        #if time.ndim  == 0 :                       # TODO: rewrite this condition to jax
        # time = repeat(time, " -> b", b=batch)  
        
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text = drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        #rope = self.rotary_embed.forward_from_seq_len(seq_len)  # TODO: not implemented yet

        for block in self.transformers_blocks:
            x = block(x, t, mask=mask, rope=None) # TODO here should be also rope parsed as argument
            
        x = self.norm_out(x,t)
        output = self.proj_out(x)
        
        return output     

# constants for duration predictor
        
SAMPLE_RATE = 24_000
HOP_LENGTH = 256
SAMPLES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH        

def maybe_masked_mean():
    pass

class Rearrange(nnx.Module):                        # TODO: eher unötig oder nicht ?
    def __init__(self, pattern:str):
        self.pattern = pattern
    def __call__(self, x: jnp.array) -> jnp.array:
        return rearrange(x, self.pattern)
    

class DurationInputEmbedding(nnx.Module):
    pass

class DurationBlock(nnx.Module):
    pass

class DurationTransformer(nnx.Module):
    pass

class DurationPredictor(nnx.Module):
    pass

def odeint_euler():
    pass

def odeint_midpoint():
    pass

def odeint_rk4():
    pass


class GermanTTS(nnx.Module):
    pass