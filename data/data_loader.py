# usefull functions from huggingface
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import get_dataset_config_names
from datasets import Dataset, Features, Array2D

# helpful array handling libaries
import numpy as np
import jax

# for mel-spectogramm maybe use torchaudio ? for this or create by myself in jax
import librosa
import soundfile

# logging
from loguru import logger

# for better readibility in handling data
from einops import rearrange, reduce, repeat




class HF_dataset():
    
    
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
        
    
    

class Mozilla_dataset():
    
    
    def __init__(self, path_to_data:str, target_sample_rate:int,hop_length:int):
        pass