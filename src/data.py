from datasets import load_dataset
from .utils import load_config

def load_bhagwad_gita_dataset():
    config = load_config()
    dataset = load_dataset(config['dataset']['name'], split="train")
    return dataset