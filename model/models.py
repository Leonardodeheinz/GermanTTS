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

from multiprocessing import Value
from typing import Callable, Literal
import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import flax
from flax import nnx
import optax
import torch
import safetensors.flax

# type hinting
from jaxtyping import Float, Array, PyTree, Bool, Int
from typing import Callable, Literal

# standard libary or other useful helpers
from pathlib import Path
from huggingface_hub import snapshot_download
from multiprocessing import Value
from typing import Callable, Literal
from functools import partial

# tensor helpers 
from einops import rearrange, reduce, repeat
import einx

# retransforming mel-spectogram with neural voice encoder
from numpy.core.umath import ndarray, zeros
from vocos import Vocos # is pretrained

# logging if needed
from loguru import logger

# for the Rottary Embdedding 
#from transformers import RoFormerModel
#from data.DataLoader import MelSpec


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

# tokenizer helper


def list_str_to_vocab_tensor(
    texts: list[list[str]], vocab: dict[str, int], padding_value: int = -1
):
    # Convert each string to vocab index, falling back to padding_value if not found
    indexed = [
        [vocab.get(token, padding_value) for token in seq]
        for seq in texts
    ]
    # Find the max sequence length for padding
    max_len = max(len(seq) for seq in indexed)
    # Pad sequences to the same length
    padded = [
        seq + [padding_value] * (max_len - len(seq)) for seq in indexed
    ]
    # Convert to JAX array
    return jnp.array(padded)

# tensor helpers
@jax.jit
def lens_to_mask(lengths, max_len=None):
    if max_len is None:
        max_len = jnp.amax(lengths)
    seq = jnp.arange(max_len)
    return seq[None, :] < lengths[:, None]

@jax.jit
def mask_from_start_end_indicies(start: int,  # noqa: F722 F821
    end: int,  # noqa: F722 F821
    max_seq_len: int,
):
    seq = jnp.arange(max_seq_len, device=start.device, dtype = jnp.int64)
    start_mask = seq[None,:] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask

@jax.jit
def mask_from_frac_lengths(seq_len: int, frac_lengths:float, max_seq_len:int, seed = key):
    lengths = jnp.int64(frac_lengths * seq_len)
    max_start = seq_len - lengths
    
    key = jax.random.key(seed)
    rand = jax.random.uniform(key, shape = (len(frac_lengths), ))
    
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
        scale = jnp.split((scale, scale), axis = -1)  # TODO: jnp.split is not the same as torch.unbind !!!
        scale = rearrange(scale, "... d r -> (d r)")
        
        return freqs, scale


# rotation helper functions

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r = 2)
    x1, x2 = x.unbind(axis=-1)
    x = jnp.stack((-x2, x1),dim=-1)
    x = rearrange(x, "... d r -> ... (d r)")
    return x

def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    
    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, jnp.ndarray) else scale
    
    if t.ndim == 4 and freqs.dim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")
    
    t, t_unrotated = t[..., :rot_dim,], t[..., rot_dim:]
    t = (t * jnp.cos(freqs) * scale) + (rotate_half(t) * jnp.sin(freqs) * scale)
    out = jnp.concat((t, t_unrotated), dim=-1)
    
    return out.astype(orig_dtype)


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
        x = self.dwconv(x)                     
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

    def __call__(self, x:Float[Array, "b n d"], mask:Bool[Array,"b n"] | None = None):
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
        
    def __call__(self, timestep:Float[Array, "b"]): 
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
                x:Float[Array, "b n d"],
                mask: Bool[Array, "b n"] | None = None,
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
                 x: Float[Array,"b n d"],
                 cond:Float[Array,"b n d"],
                 text_embed:Float[Array,"b n d"],
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
        
    def __call__ (self, x:Float[Array,"b n d"], cond:Float[Array,"b n d"], text:Int[Array, "b nt"], time:Float[Array,"b"], drop_audio_cond, drop_text, mask:Bool[Array,"b n"] | None = None):
        
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

def maybe_masked_mean(t: Float[Array, "b n d"], mask: Bool[Array,"b n"] = None) -> Float[Array,"b d"]:  # noqa: F722
    if not exists(mask):
        return  jnp.mean(t, axis = 1)
    
    t = einx.where("b n, b n d -> b n d", mask , t , 0.0)
    num = reduce(t, "b n d -> b d", "sum")
    den = reduce(mask.astype(jnp.float32), "b n -> b", "sum")
    
    return einx.divide("b d, b -> b d", num, jnp.clip(den, min = 1.0))  # TODO: check the if (clip funciton) == torch.clamp()
    
    

class Rearrange(nnx.Module):                        # TODO: eher unötig oder nicht ?
    def __init__(self, pattern:str):
        self.pattern = pattern
    def __call__(self, x: jnp.array) -> jnp.array:
        return rearrange(x, self.pattern)
    

class DurationInputEmbedding(nnx.Module):
    def __init__ (self, mel_dim, text_dim, out_dim):
        self.proj = nnx.Linear(mel_dim + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionalEmbedding(dim=out_dim)
    
    def __call__ (self, x:Float[Array, "b n d"], text_embed: Float[Array,"b nd "]):
        x = self.proj(jnp.concat((x, text_embed), dim = -1))
        x = self.conv_pos_embed(x) + x
        return x
        
        
class DurationBlock(nnx.Module):
    def __init__ (self, dim, heads, dim_head, ff_mult = 4, dropout = 0.1):
        self.attn_normal = nnx.LayerNorm(dim, use_bias= False, use_scale = False, epsilon = 1e-6)
        self.attn = Attention(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            dropout = dropout,   
        )
        self.ff_norm = nnx.LayerNorm(dim, use_bias = False, use_scale = False, eps = 1e-6)
        self.ff =FeedForward(dim = dim, mult = ff_mult, dropout=dropout, approximate = "tanh")
        
    def __call__(self, x, mask = None, rope = None):
        norm = self.attn_norm(x)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + attn_output
        norm = self.ff_norm(x)
        ff_output = self.ff(norm)
        x = x + ff_output
        return x 
        
class DurationTransformer(nnx.Module):
    def __init__(self, * , dim:int, depth:int = 8,
                 heads:int = 8, dim_head:int = 64, droput:float = 0.1,
                 ff_mult:int = 4, mel_dim:int = 100, text_num_embeds:int = 256,
                 text_dim = None, conv_layers = 0,
                 ):
        if text_dim is None:
            text_dim = mel_dim
            
        self.text_embed = TextEmbedding(text_num_embeds = text_num_embeds, text_dim = text_dim, conv_layers = conv_layers)
        self.input_embed = DurationInputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        
        self.dim = dim
        self.depth = depth

        self.transformer_blocks = [DurationBlock(
            dim = dim, 
            heads = heads,
            dim_head = dim_head, 
            ff_mult = ff_mult, 
            dropout = droput,
        ) for _ in range(depth)
                                   ]    
    
        self.norm_out = nnx.RMSNorm(dim)
    
    def __call__ (self, x:Float [Array, "b n d"], text:Int [Array,"b nt"], mask:Bool[Array,"b n"] | None = None):
        seq_len = x.shape[1]
        
        text_embed = self.text_embed(text, seq_len)
        x = self.input_embed(x, text_embed)
         
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        
        for block in self.transformer_blocks:
            x = block(x, mask = mask, rope = rope)
        
        x = self.norm_out(x)
        
        return x
        
# l1 loss function 
@jax.jit
def l1_loss(pred, target, reduction = None):
    return jnp.abs(pred, target)    

class DurationPredictor(nnx.Module):
    def __init__ (self, transformer:nnx.Module, mel_spec_kwargs: dict = dict(), tokenizer: einx.Callable[[str], list[str]] | None = None,):
        if not exists(mel_spec_kwargs):
            self.mel_spec = MelSpec(**mel_spec_kwargs)
            
        self.num_channels = self.mel_spec.n_mel_channels
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.to_pred = nnx.Sequential(nnx.Linear(transformer.dim, 1, use_bias= False), nnx.softplus(), Rearrange("... 1 -> ..."))
    
    
    
    def __call__ (self, inp: Float[Array, "b n d"] | Float[Array, "b nw"], text: Int[Array,"b nt"] | list[str], *, lens:Int[Array,"b"] | None = None, return_loss = False, key:int = 42):
        
        # check if inp is not mel-spectogram (maybe because of an bug)
        
        if inp.ndim == 2:           # TODO: pretty sure to cast the data from torch to flax if it was not preprocessed
            
            inp = self.mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels
        
        batch, seq_len, device = inp.shape[0], inp.shape[1], inp.device
        
        if isinstance(text, list):
            if exists(self.tokenizer):
                text = self.tokenizer(text)
            else:
                assert False, "if text is provided than tokenizer must be a list"
            assert text.shape[0] == batch
        
        # lens and mask
        
        if not exists(lens):
            lens = jnp.full((batch,), seq_len, device=device)            
        
        mask = lens_to_mask(lens, length = seq_len)
        
        if return_loss:
            rand_franc_index = jax.random.uniform(key, shape = (batch,), dtype = inp.dtype, minval=0, maxval=1, device = device)
            rand_index = jnp.int64(rand_franc_index * lens)
            
            seq = jnp.arange(seq_len, device=device)
            mask &= einx.less("n, b -> b n ", seq, rand_index)
            
        inp = jnp.where(
            repeat(mask,"b n  -> b n d", d = self.num_channels),
            inp, 
            jnp.zeros_like(inp),
            )
        
        x = self.transformers(inp, text = text)
        
        x = maybe_masked_mean(x, mask)
        
        pred = self.to_pred(x)
        
        if not return_loss:
            return pred

        duration = jnp.float32(lens) / SAMPLES_PER_SECOND
        return l1_loss(pred, duration)
            

            
            
#@jax.jit this needs to be rewrite but i 

def odeint_euler(func, y0, t, **kwargs):
    """
    Solves ODE using the Euler method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y)
    - y0: Initial state, a PyTorch tensor of any shape
    - t: Array of time steps, a PyTorch tensor
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # compute the next value
        k = func(t_current, y_current)
        y_next = y_current + dt * k
        ys.append(y_next)
        y_current = y_next

    return jnp.stack(ys)

def odeint_midpoint(func, y0, t, **kwargs):
    """
    Solves ODE using the midpoint method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y)
    - y0: Initial state, a PyTorch tensor of any shape
    - t: Array of time steps, a PyTorch tensor
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # midpoint approximation
        k1 = func(t_current, y_current)
        mid = y_current + 0.5 * dt * k1

        # compute the next value
        k2 = func(t_current + 0.5 * dt, mid)
        y_next = y_current + dt * k2
        ys.append(y_next)
        y_current = y_next

    return jnp.stack(ys)

@jax.vmap
def odeint_rk4(func, y0, t, **kwargs):
    """
    Solves ODE using the Runge-Kutta 4th-order (RK4) method.

    Parameters:
    - func: Function representing the ODE, with signature func(t, y)
    - y0: Initial state, a PyTorch tensor of any shape
    - t: Array of time steps, a PyTorch tensor
    """
    ys = [y0]
    y_current = y0

    for i in range(len(t) - 1):
        t_current = t[i]
        dt = t[i + 1] - t_current

        # rk4 steps
        k1 = func(t_current, y_current)
        k2 = func(t_current + 0.5 * dt, y_current + 0.5 * dt * k1)
        k3 = func(t_current + 0.5 * dt, y_current + 0.5 * dt * k2)
        k4 = func(t_current + dt, y_current + dt * k3)

        # compute the next value
        y_next = y_current + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y_next)
        y_current = y_next

    return jnp.stack(ys)


class GermanTTS(nnx.Module):
        def __init__(
        self,
        transformer: nnx.Module,
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        duration_predictor: nnx.Module | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        vocoder: Callable[[],Float[Array, "b d n"]] | None = None,  # noqa: F722
        seed:int = 42,
    ):
            self.frac_lengths_mask = frac_lengths_mask
            
            self.mel_spec = MelSpec(**mel_spec_kwargs)
            self.num_channels = self.mel_spec.n_mel_channels
            
            self.audio_drop_prob = audio_drop_prob
            self.audio_cond_prob = cond_drop_prob
            
            self.transformers = transformer
            dim = transformer.dim
            self.dim = dim
            
            self.tokenizer = tokenizer 
            
            self.vocoder = vocoder 
            
            self._duration_predictor = duration_predictor
            
            self.rng_key = jax.random.key(seed)
            
            self.EMA_model = optax.ema()
            
        def device(self):
            return next((self.parameters())).device
        
        def __call__(self,
                     inp:Float[Array,"b n d"] | Float[Array,"b nw"],
                     text:Int[Array,"b nt"] | list[str],
                     *,
                     time: jnp.ndarray, # (b,) timestep
                     cond: jnp.ndarray, # (b, n, d) or (b, n) conditional input
                     mask: jnp.ndarray, # (b, n) mask for transformer attention
                     drop_audio_cond: bool = False,
                     drop_text: bool = False,
                    ):
           
            
            pred = self.transformers(x = inp,
                                     cond = cond, 
                                     text = text, 
                                     time = time,
                                     drop_audio_cond = drop_audio_cond, 
                                     drop_text = drop_text,
                                     mask = mask,
                                     )
            
            return pred
        
        
        def sample(
            self,
            cond: Float[Array, "b n d"] | Float[Array, "b nw"],  # noqa: F722
            text: Int[Array, "b nt"] | list[str],  # noqa: F722
            duration: int | Int[Array,"b"] | None = None,  # noqa: F821
            *,
            lens: Int[Array, "b"] | None = None,  # noqa: F821
            ode_method: Literal["euler", "midpoint", "rk4"] = "rk4",
            steps=32,
            cfg_strength=1.0,
            speed=1.0,
            sway_sampling_coef=None,
            max_duration=4096,
    ):
            self.eval()
            
            cond = jnp.float32(cond)
            
            if next(self.parameters()).dtype == jnp.float16:
                cond = jnp.astype(x, dtype = jnp.float16)
            
            batch, cond_seq_len, device = cond.shape[0], cond.shape[1], cond.device
            
            if isinstance(text, list):
                if exists(self.tokenizer):
                    text = self.tokenizer(text)
                else:
                    assert False, "if text is a list, a tokenizer must be provided"
                assert text.shape[0] == batch
            
            if not exists(lens):
                lens = jnp.full((batch,), cond_seq_len, dtype = jnp.float32)
                
            if exists(text):
                text_lens = jnp.sum(text != -1, axis = -1)
                lens = jnp.maximum(text_lens, lens)
            
            if cond_seq_len < text.shape[1]:
                cond_seq_len = text.shape[1]
                cond = jnp.pad(cond, (0, 0 , 0, cond_seq_len - cond.shape[1]), mode = "zeros")
                
            if duration is None and  self._duration_predictor is not None:
                duration = self.duration_predictor(cond, text, speed)
            elif duration is None:
                raise ValueError("Duration must be provided or a duration predictor must be set") 
            
            cond_mask = lens_to_mask(lens)
            
            if isinstance(duration, int):
                duration = jnp.full((batch,), duration , dtype = jnp.float32)
            
            assert lens < duration, "duration must be at least as long as the input"
            
            duration = jnp.clip(duration, max = max_duration)
            max_duration = jnp.amax(duration)
            
            cond = jnp.pad(cond, (0,0,0, max_duration - cond_seq_len), value = 0.0)
            cond_mask = jnp.pad(cond_mask, (0, 0, 0 , max_duration - cond_mask.shape[-1]), value = False)
            cond_mask = rearrange(cond_mask, "... -> ... 1")
            
            # at each step conditioning is fixed
            
            step_cond = jnp.where(cond_mask, cond, jnp.zeros_like(cond))
            
            if batch > 1:
                mask = lens_to_mask(cond)
            else:
                mask = None
            
            def fn(t, x):
                pred = self.transformers(
                    x = x, 
                    cond = step_cond, 
                    text = text,
                    time = t,
                    mask = mask,
                    drop_audio_cond = False,
                    drop_text = False,
                    )
                
                if cfg_strength < 1e-5:
                    return pred
                
                null_pred = self.transformers(x = x,
                    cond = step_cond, 
                    text = text,
                    time = t, 
                    mask = mask, 
                    drop_audio_cond = True,
                    dropt_text = True,
                )
                
                output = pred + (pred - null_pred) * cfg_strength
                return output
            
            if ode_method == "euler":
                odeint_fn = odeint_euler
            elif ode_method == "rk4":
                odeint_fn = odeint_rk4
            elif odeint_fn == "midpoint":
                odeint_fn = odeint_midpoint
            else:
                raise ValueError(f"Unknown method: {ode_method}")
            
            y0 = []
            for dur in duration:
                if exists(self.rng_key):
                    y0.append(jax.random.normal(self.rng_key, (dur, self.num_channels), dtype = step_cond.dtype))
            #y0 = pad_seq  #TODO: look for better option to do this
            
            t = jnp.linspace(0, 1, dtype = step_cond.dtype)
            if exists(sway_sampling_coef):
                t = t + sway_sampling_coef * (jnp.cos(jnp.pi / 2 * t) - 1 + t)
                
                
            trajectory = odeint_fn(fn, y0, t)
            
            sampled = trajectory[-1]
            
            # trim the reference audio
            
            out = sampled[:, cond_seq_len]
            
            if exists(self.vocoder):        
                out = np.asarray(out, dtype= out.dtype)
                out = torch.from_numpy(out)
                out = torch.permute(input = out, dims = (0,2,1))
                out = self.vocoder(out.cpu())
            
            return out 
            
        
        @classmethod
        def from_pretrained(cls, hf_model_name_or_path:str) -> "GermanTTS":
            if exists(hf_model_name_or_path):
                path = Path(snapshot_download(repo_id = hf_model_name_or_path, allow_patterns = ["*.safetensors", "*.txt"]))
            else:
                raise ValueError(f"please provide an vailid path for the model path: {hf_model_name_or_path}")
            
            vocab_path = path / "vocab.txt"
            vocab = {v: i for i, v in enumerate(Path(vocab_path).read_text().split("\n"))}
            
            tokenizer = partial(list_str_to_vocab_tensor, vocab=vocab)
            
            vocos = Vocos.from_pretrained("charachtr/vocos-mel-24khz")
            
            duration_model_filename = "duration.safetensors"
            duration_model_path = path / duration_model_filename
            duration_predictor = None
            
            if duration_model_filename.exists():
                duration_predictor = DurationPredictor(
                    transformer = DurationTransformer(
                        dim = 512,
                        depth = 12,
                        heads = 8,
                        text_dim = 12,
                        ff_mult = 2,
                        conv_layers=0,
                        text_num_embeds=len(vocab),
                    ), tokenizer = tokenizer
                )
                
            state_dict = safetensors.flax.load_file(duration_model_path.as_posix())
            duration_predictor = flax.serialization.from_state_dict(duration_predictor, state_dict, "duration_predictor")
            
            
            model_filename = "model.safetensors"
            model_path = path / model_filename
            
            model = GermanTTS(
                transformer=DiT(
                    dim = 512,
                    depth = 18,
                    heads = 12,
                    text_dim = 512,
                    ff_mult = 2,
                    conv_layers= 4,
                    text_num_embeds=len(vocab)
                    ),
                tokenizer=tokenizer,
                vocoder=vocos.decode,
            )
            
            state_dict = safetensors.flax.load_file(model_path.as_posix())
            model = flax.serialization.from_state_dict(model, state_dict, "model")
            
            model._duration_predictor = duration_predictor
            
            return model
