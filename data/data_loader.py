# usefull functions from huggingface
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import get_dataset_config_names
from datasets import Dataset, Features, Array2D

from torchaudio.transforms._transforms import Resample
from torchaudio.functional.functional import  

# helpful array handling libaries
import datasets
import numpy as np
import jax

# for mel-spectogramm maybe use torchaudio ? for this or create by myself in jax


# logging
from loguru import logger

# for better readibility in handling data
from einops import rearrange, reduce, repeat




class Huggingface_Dataset(datasets.Dataset):
    """
    This class provides an easy way to load an huggingface audio dataset and convert it quickly for TTS training
    args:
    path_to_data:
    target_sample_rate:
    
    """
    
    def __init__(
        self,
        name_of_dataset:str,
        min_duration:float = 0,     # min 
        max_duration:float = 10,    # max duration when i cut of the sample, should be less than > 10 sec
        if_stream: bool = False,
        language:str = "german",
        split:str = "",
        location:str = ""
    ):
        # download specific settings
        
        super().__init__()
        self.if_stream = if_stream  # maybe not neccessary if will just have it already available
        self.language = language  # specify language to load 
        self.split = split  # train, dev or test
        self.name_of_dataset = name_of_dataset
               
        self.data = load_dataset(path=name_of_dataset,
                                 name=self.language,
                                 split = self.split
                                 streaming = self.if_stream)
        
    def __len__(self):
        return len(self.data)
    
    def get_sample(self):
        """
        get an random sample of dataset
        """
        return next(iter(self.data))
    
    
    def set_format(self):
        """
        sets the format for preprocessing
        """

        
    def safe_preprocess(self):
        """
        safe preprocessed data
        """
        
    def load_preprocessed(self):
        """
        load preprocessed data
        """

    def convert_to_jax(self):
        """
        maybe not needed
        """
    
    def get_dataset(self) -> datasets.dataset_dict:
        """
        get an already existing Dataset that was also downloaded        
        Returns:
            datasets.dataset_dict: _description_
        """