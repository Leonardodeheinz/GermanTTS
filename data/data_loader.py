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

# usefull functions from huggingface
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import get_dataset_config_names
from datasets import Dataset, Features, Array2D


from PreProcesesAudio import (
    MelSpec,
    Resampler 
)

# helpful array handling libaries
import datasets
import numpy as np
import jax.numpy as jnp

# for plotting mel spectogram
import matplotlib.pyplot as plt


# logging
from loguru import logger

# for better readibility in handling data
from einops import rearrange, reduce, repeat

# check if directory exits
import os 

class Huggingface_Dataset:
    """
    This class provides an easy way to load an huggingface audio dataset and convert it quickly for TTS training
    args:
    path_to_data:
    target_sample_rate:
    
    """
    
    def __init__(
        self,
        name_of_dataset:str,
        #min_duration:float = 0,     # min 
        #max_duration:float = 10,    # max duration when i cut of the sample, should be less than > 10 sec
        if_stream: bool = False,
        language:str = "german",
        split:str = "",
        cache_dir:str = "",
    ):
        # download specific settings
       
        self.if_stream = if_stream  # maybe not neccessary if will just have it already available
        self.language = language  # specify language to load 
        self.split = split  # train, dev or test
        self.name_of_dataset = name_of_dataset

               
        # check if datset is already loaded in cache
        if os.path.isdir(cache_dir):
            self.data = load_dataset(path = name_of_dataset, cache_dir=cache_dir)
        else:  # relead dataset if not already done
            self.data = load_dataset(path = self.name_of_dataset,
                                     name = self.language,
                                     split = self.split,
                                     streaming = self.if_stream,
                                     )
            self.data = self.data.with_format("torch")
        
        
        # basic information for the preprocessing
        self.sampling_rate = self.data[0]["audio"]["sampling_rate"]
        
     
        
        
    def get_data(self):
        return self.data
        
    def get_shape(self):
        return self.data.shape
    
    def get_sample(self):
        """
        get an random sample of dataset
        """
        return next(iter(self.data))
    
    
    def remove_columns(self, columns_to_remove):
        """
        sets the format for preprocessing
        """
        self.data.remove_columns(column_names = columns_to_remove)
        

        
    def __safe_data__(self, directory_to_save_to:str = ""):
        """
        safe preprocessed data, not load data not needed because is catched if in constructur already
        """
        self.data.save_to_disk(directory_to_save_to)
     
     
    @logger.catch   
    def pre_process(self, audio_threshold_max:int = 20, audio_threshold_min:int = 3, 
                    ):
        """
        pre_process whole dataset and save it 
        """

        dtype_to_resample_to = self.data[0]["audio"]["array"].dtype
        resampler = Resampler(orig_freq = self.sampling_rate.item(), dtype = dtype_to_resample_to)
        mel_spec = MelSpec()
        
        path_to_save_to = "res/preprocessed" + self.name_of_dataset + "/" +  self.split   # path to save the dataset to 
        
        
        logger.info(f"Sampling Rate of our dataset {self.sampling_rate}")
        
        logger.debug("for loop starts")
        breakpoint()
        for file_index in range(self.get_shape()[0]):
            if file_index == 10:
                break
            duration_of_current_file = self.data[file_index]["audio_duration"].item()
            
            if audio_threshold_max >= duration_of_current_file > audio_threshold_min:
                
                logger.info(f"File with index: {file_index}, in Dataset: {self.name_of_dataset, self.split}, has duration:{duration_of_current_file}, it will be used !") # log index of data that is too long or too short
                
                self.data[file_index]["audio"]["array"] = resampler.forward(self.data[file_index]["audio"]["array"])
                breakpoint()
                self.data[file_index] = rearrange(self.data[file_index]["audio"]["array"], "t -> 1 t")
                
                self.data[file_index] = mel_spec.forward(self.data[file_index]["audio"]["array"])
                
            else:
                logger.info(f"File with index: {file_index}, in Dataset: {self.name_of_dataset, self.split}, has duration:{duration_of_current_file}, and will not be used!")
        logger.info()   
        
        breakpoint()
        
    def get_dataset(self) -> datasets.dataset_dict:
        """
        get an already existing Dataset that was also downloaded        
        Returns:
            datasets.dataset_dict: _description_
        """
        return self.data
    
    def group_data_speaker(self):
        """
        This function should group data of one specific speaker, e.g. for OrpheusTTS training
        """
        
    
    
    def plot_mel_spectrogram(self,mel_spec, t, sr=24_000, n_mels=80):
        """Plot the Mel spectrogram using matplotlib."""
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma', extent=[t.min(), t.max(), 0, n_mels])
        plt.colorbar(label="Log Magnitude")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel Frequency Bins")
        plt.title("Mel Spectrogram")
        plt.show()



columns_to_remove = ["original_path", "file", "id","chapter_id"]

facebook_german = Huggingface_Dataset(name_of_dataset="facebook/multilingual_librispeech",
                                       if_stream = False,
                                       language = 'german', 
                                       split = "train",
                                       cache_dir = "res/example/train")
                                       
#common_voice = Huggingface_Dataset(name_of_dataset = "mozilla-foundation/common_voice_17_0",
                      #             if_stream = False, 
                       #            language = "de",
                        #           split = "train"
                         #          chach_dir = "res/example/train")           
                            
facebook_german.remove_columns(columns_to_remove=columns_to_remove)

facebook_german.pre_process()


print("successful run")


