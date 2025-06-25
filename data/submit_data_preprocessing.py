# usefull functions from huggingface
from random import sample
from datasets import load_dataset
from datasets import load_from_disk

from functools import partial
import os

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
from einops import rearrange

# check if directory exits
import os 

from DataLoader import Huggingface_Datasetloader


def preprocess_facebook_librispeech(batch, orig_freq=None, audio_threshold_max=20, audio_threshold_min=3):
    resampler = Resampler(orig_freq=orig_freq, new_freq=24_000)
    melspec = MelSpec(sampling_rate=orig_freq)

    audio_arrays = batch["audio"]
    audio_duration = batch["audio_duration"]

    mels = []

    logger.info("PreProcessing starts")
    for index in range(len(audio_arrays)):
        if audio_threshold_min < audio_duration[index] <= audio_threshold_max:
            audio_array = audio_arrays[index]["array"]

            audio_array = resampler.forward(audio_array)
            audio_array = rearrange(audio_array, "t -> 1 t")
            audio_array = melspec.forward(audio_array)
            mel_spec_array = rearrange(audio_array, "1 d t -> d t")
            logger.info(f"mel_spec_array shape: {mel_spec_array.shape}")

            mels.append(mel_spec_array)
        else:
            logger.info(f"File skipped (duration: {audio_duration[index]})")
            mels.append(None)  # Maintain alignment

    return {
        "mel": mels,
        "transcript": batch["transcript"],
        "speaker_id": batch["speaker_id"],
        "audio_duration": batch["audio_duration"],
    }
    
    
# def preprocess_commonvoice(batch, orig_freq=None, audio_threshold_max=20, audio_threshold_min=3):
#     resampler = Resampler(orig_freq=orig_freq, new_freq=24_000)
#     melspec = MelSpec(sampling_rate=orig_freq)
#     pass
#     columns_to_remove = ["original_path", "file", "id","chapter_id"]


if __name__ == "__main__":
    # load the dataset
    dataset = Huggingface_Datasetloader(
        name_of_dataset="facebook/multilingual_librispeech",
        if_stream=False,
        language="german",
        split="train"
    )
    
    # preprocess the dataset
    dataset.process_dataset(num_proc=1, batch_size= 256, make_preprocessor = preprocess_facebook_librispeech, safe_data_path= "GermanTTS/data/processed_data/librispeech_german_preprocessed")
    
    logger.info("Dataset preprocessing completed successfully.")