# Default configs for training our model

import ml_collections

def get_config_GermanTTS():
    """
    Get the default Hyperparameters configurations
    """
    config = ml_collections.ConfigDict()
    
    config.dim = 512
    config.depth = 18
    config.heads = 12
    config.text_dim = 512
    config.ff_mult = 2
    
    return config
    
def get_config_hyperparameters():
    
    config = ml_collections.ConfigDict()
    
    config.start_factor = 1e-8
    config.end_factor = 1.0
    config.warmup_steps = 1_000
    config.batch_size = 16

    return config

def get_config_DurationPredictor():
    
    config = ml_collections.ConfigDict()
    
    config.batch_size = 12
    config.max_batch_frames = 4096
    config.max_duration = 4096
    
    return config