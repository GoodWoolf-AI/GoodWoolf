from pathlib import Path
from transformers import AutoTokenizer

from goodwoolf.utils.models import config_path


class Tokenizer:
    def __init__(self, model_name):
        path = config_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)        
