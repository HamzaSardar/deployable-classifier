from pathlib import Path
import ml_collections


def get_config() -> ml_collections.ConfigDict:

    config = ml_collections.ConfigDict()

    # inference parameters 
    config.inference = ml_collections.ConfigDict()
    config.inference.data_path = Path('/Users/user/Projects/CIFAR-10-images/test')
    
    return config

