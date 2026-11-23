import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from goodwoolf.utils.models import model_name_from_path


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        name = model_name_from_path(__file__)
        print(f"Initializing model {name}", file=sys.stderr)
        self.config = config
        self.model = AutoModelForCausalLM.from_config(config)

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids=input_ids, labels=labels, **kwargs)

    def generate(
        self,
        tokenizer,
        prompt_ids,
        num_samples=1,
        max_new_tokens=256,
        temperature=0.6,
        top_k=50,
        do_sample=True,
        seed=42,
    ):
        device = next(self.parameters()).device
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        sampling = bool(do_sample and temperature > 0)
        if num_samples > 1:
            sampling = True
        if seed is not None:
            torch.manual_seed(seed)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sampling,
            "num_return_sequences": num_samples,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        }
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id
        if sampling:
            kwargs["temperature"] = max(temperature, 1e-6)
            if top_k and top_k > 0:
                kwargs["top_k"] = top_k
        output = self.model.generate(input_ids=input_ids, **kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        rows = sequences[:, len(prompt_ids):].tolist()
        full_sequences = [list(prompt_ids) + row for row in rows]
        masks = [[0] * len(prompt_ids) + [1] * len(row) for row in rows]
        return full_sequences, masks

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(k.startswith("model.") for k in state_dict):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)
