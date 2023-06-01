import os
import argparse
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    args, unparsed = parser.parse_known_args()
    return args

def parse_config():
    args = parse_args()
    if args.config_path is None:
        return {}
    with open(args.config_path) as f:
        config = eval(f.read())
    assert isinstance(config, dict)
    return config

def _singleton(cls):
    _instance = {}
    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner
    
@_singleton
class Config(Dict):
    def __init__(self):
        dict.__init__(self)
        config = parse_config()
        if config is not None:
            self.merge(config)

    def merge(self, config: dict):
        for key, value in config.items():
            self.__setitem__(key, value)
        
