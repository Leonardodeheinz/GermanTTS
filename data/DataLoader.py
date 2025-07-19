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
from json import load
from random import sample
from datasets import load_dataset
from datasets import load_from_disk
import jax.numpy as jnp
import tqdm

from functools import partial
import os

from GermanTTS.data.PreProcesesAudio import Resample, MelSpec

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
        name_of_dataset:str = this variable should point to the huggingface repo to download the data from 
        if_strea:bool = decide if audio should be streamable when loading dataset
        language:str = decide which language to load from
        split:str = indicate which split is desired
        from_disk_directory:str = path to directory of preprocessed data to load form 
    
    """
    
    def __init__(
        self,
        name_of_dataset:str = None,
        if_stream: bool = False,
        language:str = "",
        split:str = "",
        from_disk_directory:str = "",
    ):
           
              
        self.if_stream = if_stream  
        self.language = language

        self.split = split  
        self.name_of_dataset = name_of_dataset

        self.from_disk_directory = from_disk_directory
               
        # check if datset is already loaded in cache
        if name_of_dataset is not None:
            self.data = load_dataset(path = name_of_dataset, 
                                     name = self.language,
                                     split = self.split)
            self.sampling_rate = self.data[0]["audio"]["sampling_rate"]
            self.dtype_to_resample_to = self.data[0]["audio"]["array"].dtype
            self.data = self.data.with_format("torch")
        else: 
            self.data = load_from_disk(dataset_path = self.from_disk_directory,
                                       keep_in_memory = False,
                                     )
            
        
        
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
        self.data = self.data.with_format("jax")
        self.data.save_to_disk(directory_to_save_to)
        logger.info(f"Data got saved to this path: {directory_to_save_to}")
    
    

    @logger.catch
    def process_data(self, num_proc=4, batch_size = 4, make_preprocessor = None, safe_data_path:str = ""):
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
       
        self.__safe_data__(safe_data_path)
        
        logger.info(f"Preprocessing finished, data_size: {self.data.shape}, saved to {safe_data_path}")
        
    def extract_vocab(self, num_proc=4, batch_size = 4,name_of_text:str = None, safe_data_path:str = ""):
        """
        Extracts a set of unique characters from the specified text column of the dataset.
        """

       
        logger.info(f"num_proc: {num_proc}, batch_size: {batch_size}, creation of vocab for data_size: {self.data.shape}")

        def extract_batch_vocab(batch):
            all_chars = set()
            for text in batch[name_of_text]:
                all_chars.update(text)
            return {"vocab": [list(all_chars)]}  # Return wrapped in list to preserve batch formatting



        vocab_data = self.data.map(extract_batch_vocab,
                              batched = True,
                              batch_size = batch_size, 
                              num_proc= num_proc,
                              remove_columns=self.data.column_names,
                              load_from_cache_file=True, 
                              keep_in_memory= True
                              )
        

        merged_chars = set()
        for batch_vocab in vocab_data["vocab"]:
            for char in batch_vocab:
                merged_chars.add(char)

        unique_chars = sorted(merged_chars)
        logger.info(f"Extracted vocab of size {len(unique_chars)}")
        
        # safe data if data path is provided
       
        if safe_data_path:
            vocab_file = f"{safe_data_path.rstrip('/')}/vocab.txt"
            with open(vocab_file, "w", encoding="utf-8") as f:
               for char in unique_chars:
                    if char == "\n":
                      f.write("\\n\n")  # escape newline for readability
                    elif char == "\t":
                      f.write("\\t\n")  # escape tab for readability
                    else:
                      f.write(f"{char}\n")
        logger.info(f"Vocabulary saved to {vocab_file}")

        return unique_chars
        
        logger.info(f"Preprocessing finished, data_size: {self.data.shape}, saved to {safe_data_path}")


        
    def plot_mel_spectrogram(self, sr=24_000):
        """Plot the Mel spectrogram using matplotlib."""
        # take sample
        sample_mel = self.data[:1]["mel"]
        sampel_transcript = self.data[:1]["transcript"]
        n_mels = sample_mel.shape[0]
        t = jnp.arange(0, sample_mel.shape[1]) 
        
        plt.figure(figsize=(10, 4))
        plt.imshow(sample_mel, aspect='auto', origin='lower', cmap='magma', extent=[jnp.min(t), jnp.max(t), 0, n_mels])
        plt.colorbar(label="Log Magnitude")
        plt.xlabel("Time (s)")
        plt.ylabel("Mel Frequency Bins")
        plt.title(sampel_transcript)
        plt.show()

    def group_data(self, speaker_id):
        """Group data by speaker_id for example 300 samples of one exact speaker

        Returns:
            _type_: _description_
        """ 
        filterd_data = self.data.filter(lambda example: example["speaker_id"] == speaker_id)
        return filterd_data

