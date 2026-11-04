import os
import sys
import torch.nn as nn
from transformers import AutoModelForCausalLM
from utils.models import model_name_from_path


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        name = model_name_from_path(__file__)
        print(f"Initializing model {name}", file=sys.stderr)
        self.model = AutoModelForCausalLM.from_config(config)

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids=input_ids, labels=labels, **kwargs)

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(k.startswith("model.") for k in state_dict):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)
