# usefull functions from huggingface
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import get_dataset_config_names
from datasets import Dataset, Features, Array2D

# helpful array handling libaries
import numpy as np
import jax

# for mel-spectogramm maybe use torchaudio ? for this or create by myself in jax
from librosa.feature import melspectrogram
import soundfile

# logging
from loguru import logger

# for better readibility in handling data
from einops import rearrange, reduce, repeat




class HF_dataset():
    """
    This class provides an easy way to load an huggingface audio dataset and convert it quickly for TTS training
    args:
    
    path_to_data:
    target_sample_rate:
    
    """
    
    def __init(self, path_to_data:str, 
               target_sample_rate:int, 
               hop_length:int, 
               if_stream:bool,
               language:str,
               split:str, 
               ):
        # download specific settings
        self.if_stream = stream
        self.language = language
        self.split = split
        
        # feature specific settings
        self.hop_length = hop_length
        self.target_sampling_rate
        
        
        
        self.data = load_dataset(path=path_to_data,
                                 name=self.language,
                                 streaming = self.if_stream)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem(self, index):
        pass
    

class Mozilla_dataset():
    """
    small Dataloader for the Mozilla Dataset, could have a diffrent structure to huggingface dataset
    """
    
    def __init__(self, path_to_data:str, target_sample_rate:int,hop_length:int):
        pass