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

import torch
from torch.nn import Module  # torch imports

import torchaudio
from torchaudio.transforms import MelSpectrogram # torchaudio imports
from torchaudio.transforms import Resample
from torch import nn, tensor, Tensor, from_numpy
from einops import rearrange, repeat, reduce, einsum, pack, unpack

class MelSpec(Module):

    def __init__(self, filter_length:int = 1024, 
                 hop_length:int = 256, 
                 win_length:int = 1024, 
                 n_mel_channels:int = 100, 
                 sampling_rate:int = 24_000, 
                 normalize:bool = False, power:int = 1, center:bool = True, norm = None
                ):
                
                super().__init__()
                
                self.n_mel_channels = n_mel_channels
                self.sampling_rate = sampling_rate
                
                self.mel_stft = MelSpectrogram(
                    sample_rate = self.sampling_rate, 
                    n_fft = filter_length,
                    win_length = win_length, 
                    hop_length = hop_length, 
                    n_mels = n_mel_channels, 
                    power = power,
                    center = center,
                    normalized= normalize,
                    norm=norm
                )
                
                self.register_buffer("dummy" , tensor(0), persistent = False)
    
    def forward(self, inp):
        if len(inp.shape) == 3:
            inp = rearrange(inp, 'b 1 w -> b nw')
        
        assert len(inp.shape) == 2
        
        if self.dummy.device != inp.device:
            self.to(inp.device)
        
        mel = self.mel_stft(inp)
        mel = self.log(mel)    
        return mel
    
    def log(self, mel_tensor, eps=1e-5):
        """
        enables us to create an log value of an torch.tensor

        Args:
            mel (torch.tensor): generted mel_spectogram of the wav file
            eps (float): small epsilon to prevent log(0) 
        """
        return torch.clamp(mel_tensor, min = eps).log()
        
        
    
class Resampler(Module):

    def __init__(self,
               orig_freq:int = None,
               new_freq:int = 24_000,
               resampling_method:str = "sinc_interp_hann",
               #lowpass_filter_width:int = 0,
               #rolloff:float = 0.99,
               #beta:float = None, 
               dtype:torch.device =  torch.float32,# for resampling precission
               ):
        super().__init__()
        
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampling_method = resampling_method
        #self.lowpass_filter_width = lowpass_filter_width
        #self.rolloff = rolloff
        #self.beta = beta
        self.dtype = dtype
        
        self.resampler = torchaudio.transforms.Resample(orig_freq = self.orig_freq,
                                                        new_freq = self.new_freq,
                                                        resampling_method = self.resampling_method,
                                                        dtype= self.dtype
                                                        )
        
        
    def forward(self, raw_tensor):
        """
        resample an array to an specific sample rate
        Args:
            raw_tensor (torch.tensor): raw tensor to resample 
        """

        return self.resampler(raw_tensor)