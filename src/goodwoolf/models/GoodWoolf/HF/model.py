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

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    @torch.no_grad()
    def chat(self, tokenizer, messages, max_tokens=256, temperature=0.6, top_k=50, do_sample=True):
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise ValueError("Tokenizer must provide apply_chat_template() for chat")
        max_position = int(getattr(self.config, "max_position_embeddings", 4096))
        prompt_limit = max(1, max_position - max_tokens)
        text = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            prompt_ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            prompt_ids = tokenizer.encode(text)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is not None and (not prompt_ids or prompt_ids[0] != bos_token_id):
            prompt_ids = [bos_token_id] + prompt_ids
        if not prompt_ids:
            raise ValueError("Chat template rendered an empty prompt")
        prompt_ids = prompt_ids[:prompt_limit]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=next(self.parameters()).device)
        kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": bool(do_sample and temperature > 0),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        }
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            kwargs["eos_token_id"] = eos_token_id
        if kwargs["do_sample"]:
            kwargs["temperature"] = temperature
            if top_k and top_k > 0:
                kwargs["top_k"] = top_k
        output = self.model.generate(input_ids=input_ids, **kwargs)
        sequences = output.sequences if hasattr(output, "sequences") else output
        response_ids = sequences[0, len(prompt_ids):].tolist()
        return tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if any(k.startswith("model.") for k in state_dict):
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        return self.model.load_state_dict(state_dict, strict=strict, assign=assign)
