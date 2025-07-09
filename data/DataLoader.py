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
from random import sample
from datasets import load_dataset
from datasets import load_from_disk

from functools import partial
import os

from data.PreProcesesAudio import (
    MelSpec,
    Resampler
)

# helpful array handling libaries
import datasets


# for plotting mel spectogram
import matplotlib.pyplot as plt


# logging
from loguru import logger

# for better readibility in handling data
from einops import rearrange

# check if directory exits
import os 

class Huggingface_Datasetloader:
    """
    This class provides an easy way to load an huggingface audio dataset and convert it quickly for TTS training
    args:
    path_to_data:
    target_sample_rate:
    
    """
    
    def __init__(  #TODO read the data_dir also for pre_processed data
        self,
        name_of_dataset:str,
        #min_duration:float = 0,     # min 
        #max_duration:float = 10,    # max duration when i cut of the sample, should be less than > 10 sec
        if_stream: bool = False,
        language:str = "german",
        split:str = "",
        cache_dir:str = "",
    ):
           
              
        self.if_stream = if_stream  # maybe not neccessary if will just have it already available
        self.language = language  # specify language to load 
        self.split = split  # train, dev or test
        self.name_of_dataset = name_of_dataset

               
        # check if datset is already loaded in cache
        if os.path.isdir(cache_dir):
            self.data = load_dataset(path = name_of_dataset, 
                                     name = self.language, 
                                     cache_dir=cache_dir)
        else:  # relead dataset if not already done
            self.data = load_dataset(path = self.name_of_dataset,
                                     name = self.language,
                                     split = self.split,
                                     streaming = self.if_stream,
                                     )
            self.data = self.data.with_format("torch")
        
        
        # basic information for the preprocessing
        self.sampling_rate = self.data[0]["audio"]["sampling_rate"]
        self.dtype_to_resample_to = self.data[0]["audio"]["array"].dtype
        
        
        
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
    def process_dataset(self, num_proc=4, batch_size = 4, make_preprocessor = None, safe_data_path:str = ""):
        """ give any function and it will tranform the data
            and store the transformed data in a new path so that it is very easy to access later on.
        Args:
            num_proc (int, optional): _description_. Defaults to 4.
            batch_size (int, optional): _description_. Defaults to 4.
            columns_to_remove (_type_, optional): _description_. Defaults to None.
            make_preprocessor (_type_, optional): _description_. Defaults to None.
        """
        
        
        
        preprocessor = partial(make_preprocessor, orig_freq = 16_000, audio_threshold_max = 20, audio_threshold_min = 3)
        logger.info(f"num_proc: {num_proc}, batch_size: {batch_size}, Preprocessing starts, data_size: {self.data.shape}")
        
        self.data = self.data.map(preprocessor,
                                  batched = True,
                                  batch_size = batch_size, 
                                  num_proc= num_proc,
                                  remove_columns=self.data.column_names,
                                  load_from_cache_file=True, 
                                  keep_in_memory= True
                                  )
        
        # safe data if data path is provided
        if os.path.exists(safe_data_path):
            self.__safe_data__(safe_data_path)
        
        logger.info(f"Preprocessing finished, data_size: {self.data.shape}, saved to {safe_data_path}")
        
        
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
        pass
    
    
    def plot_mel_spectrogram(self,mel_spec, t, sr=24_000, n_mels=80):
        """Plot the Mel spectrogram using matplotlib."""
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma', extent=[t.min(), t.max(), 0, n_mels])
        plt.colorbar(label="Log Magnitude")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel Frequency Bins")
        plt.title("Mel Spectrogram")
        plt.show()

    def group_data(self, speaker_id):
        """Group data by speaker_id for example 300 samples of one exact speaker

        Returns:
            _type_: _description_
        """ 
        filterd_data = self.data.filter(lambda example: example["speaker_id"] == speaker_id)
        return filterd_data

  
    
# example

# facebook_german = Huggingface_Dataset(name_of_dataset="facebook/multilingual_librispeech",
#                                       if_stream = False,
#                                       language = "german", 
#                                       split = "1_hours",
#                                       cache_dir = "/home/dheinz/Documents/GermanTTS/res/example/1_hours",
#                                       )
                                       
common_voice = Huggingface_Datasetloader(name_of_dataset = "mozilla-foundation/common_voice_17_0",
                                    if_stream = False, 
                                    language = "de",
                                     split = "train",
                                    cache_dir = "res/example/train")           


#facebook_german.process_dataset(num_proc=1, batch_size= 16, make_preprocessor = preprocess_facebook_librispeech, safe_data_path= "/home/dheinz/Documents/GermanTTS/res/example/1_hours")


filterd_dataset = load_from_disk("/home/dheinz/Documents/GermanTTS/res/example/1_hours")

print("successful run")

