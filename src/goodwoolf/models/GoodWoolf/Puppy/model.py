"""Puppy GPT model adapted from karpathy/nanochat."""

import signal
import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from goodwoolf.utils.models import model_name_from_path


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def _eval_with_timeout(formula, max_time=3):
    def timeout_handler(signum, frame):
        raise TimeoutError(formula)

    try:
        previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_time)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
    except Exception:
        return None


def _use_calculator(expr):
    expr = expr.replace(",", "")
    if all(char in "0123456789*+-/.() " for char in expr):
        if "**" in expr:
            return None
        return _eval_with_timeout(expr)

    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all(char in allowed_chars for char in expr):
        return None
    dangerous_patterns = (
        "__", "import", "exec", "eval", "compile", "open", "file",
        "input", "raw_input", "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr", "hasattr",
    )
    if any(pattern in expr.lower() for pattern in dangerous_patterns):
        return None
    if ".count(" not in expr:
        return None
    return _eval_with_timeout(expr)


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if window_size[0] < 0 or window_size[0] >= T:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            positions = torch.arange(T, device=x.device)
            mask = (positions[:, None] <= positions[None, :]) & (
                positions[None, :] >= positions[:, None] - window_size[0] + 1
            )
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx, targets=None, loss_reduction="mean"):
        B, T = idx.size()

        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        cos_sin = self.cos[:, :T].to(idx.device), self.sin[:, :T].to(idx.device)

        x = self.transformer.wte(idx)
        x = norm(x)

        if T > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)[..., :self.config.vocab_size]
        logits = softcap * torch.tanh(logits.float() / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        return logits


class Model(GPT):
    def __init__(self, config):
        name = model_name_from_path(__file__)
        print(f"Initializing model {name}", file=sys.stderr)

        depth = int(getattr(config, "depth", 12))
        aspect_ratio = int(getattr(config, "aspect_ratio", 64))
        head_dim = int(getattr(config, "head_dim", 128))
        n_embd = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
        n_head = n_embd // head_dim
        n_kv_head = int(getattr(config, "n_kv_head", n_head))

        gpt_config = GPTConfig(
            sequence_len=int(getattr(config, "max_position_embeddings", 2048)),
            vocab_size=int(getattr(config, "vocab_size", 32768)),
            n_layer=depth,
            n_head=n_head,
            n_kv_head=n_kv_head,
            n_embd=n_embd,
            window_pattern=str(getattr(config, "window_pattern", "SSSL")),
        )
        super().__init__(gpt_config)

    def forward(self, input_ids, labels=None, **kwargs):
        loss_reduction = kwargs.get("loss_reduction", "mean")
        output = super().forward(input_ids, targets=labels, loss_reduction=loss_reduction)
        if labels is None:
            return SimpleNamespace(loss=None, logits=output)
        return SimpleNamespace(loss=output, logits=None)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                key = key.removeprefix("_orig_mod.")
            normalized[key] = value
        return super().load_state_dict(normalized, strict=strict, assign=assign)

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=50,
        temperature=1.0,
        top_k=None,
        do_sample=True,
        pad_token_id=None,
        **kwargs,
    ):
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = super().forward(ids)[:, -1, :]
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            if do_sample and temperature > 0:
                logits = logits / temperature
                next_ids = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
        return ids

    @torch.inference_mode()
    def chat(
        self,
        tokenizer,
        messages,
        max_tokens=256,
        temperature=0.6,
        top_k=50,
        do_sample=True,
    ):
        prompt_ids = tokenizer.render_prompt(messages)
        if not prompt_ids:
            raise ValueError("Conversation rendered to an empty prompt")

        device = next(self.parameters()).device
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        python_start = tokenizer.encode_special("<|python_start|>")
        python_end = tokenizer.encode_special("<|python_end|>")
        output_start = tokenizer.encode_special("<|output_start|>")
        output_end = tokenizer.encode_special("<|output_end|>")
        stop_ids = {
            tokenizer.encode_special("<|assistant_end|>"),
            tokenizer.get_bos_token_id(),
        }
        forced_tokens = []
        python_expression = []
        in_python = False
        generated = 0

        while generated < max_tokens:
            if forced_tokens:
                next_id = forced_tokens.pop(0)
                next_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
            else:
                logits = super().forward(ids)[:, -1, :]
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                if do_sample and temperature > 0:
                    next_ids = torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1)
                else:
                    next_ids = torch.argmax(logits, dim=-1, keepdim=True)
                next_id = next_ids.item()

            generated += 1
            if next_id in stop_ids:
                break
            if next_id == python_start:
                in_python = True
                python_expression = []
            elif next_id == python_end and in_python:
                in_python = False
                result = _use_calculator(tokenizer.decode(python_expression))
                python_expression = []
                if result is not None:
                    forced_tokens.extend([output_start] + tokenizer.encode(str(result)) + [output_end])
            elif in_python:
                python_expression.append(next_id)
            ids = torch.cat((ids, next_ids), dim=1)

        response_ids = ids[0, len(prompt_ids):].tolist()
        return tokenizer.decode(response_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_sequences(
        self,
        tokenizer,
        prompt_ids,
        num_samples=1,
        max_new_tokens=256,
        temperature=1.0,
        top_k=50,
        seed=42,
    ):
        device = next(self.parameters()).device
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        python_start = tokenizer.encode_special("<|python_start|>")
        python_end = tokenizer.encode_special("<|python_end|>")
        output_start = tokenizer.encode_special("<|output_start|>")
        output_end = tokenizer.encode_special("<|output_end|>")
        stop_ids = {
            tokenizer.encode_special("<|assistant_end|>"),
            tokenizer.get_bos_token_id(),
        }
        sequences, masks = [], []

        for sample_idx in range(num_samples):
            rng = torch.Generator(device=device)
            rng.manual_seed(seed + sample_idx)
            ids = prompt.clone()
            token_mask = [0] * len(prompt_ids)
            forced_tokens = []
            python_expression = []
            in_python = False

            for _ in range(max_new_tokens):
                if forced_tokens:
                    next_id = forced_tokens.pop(0)
                    next_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    sampled = False
                else:
                    logits = super().forward(ids)[:, -1, :]
                    if top_k is not None and top_k > 0:
                        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
                    if temperature > 0:
                        next_ids = torch.multinomial(
                            F.softmax(logits / temperature, dim=-1),
                            num_samples=1,
                            generator=rng,
                        )
                    else:
                        next_ids = torch.argmax(logits, dim=-1, keepdim=True)
                    next_id = next_ids.item()
                    sampled = True

                token_mask.append(1 if sampled else 0)
                if next_id in stop_ids:
                    ids = torch.cat((ids, next_ids), dim=1)
                    break
                if next_id == python_start:
                    in_python = True
                    python_expression = []
                elif next_id == python_end and in_python:
                    in_python = False
                    result = _use_calculator(tokenizer.decode(python_expression))
                    python_expression = []
                    if result is not None:
                        forced_tokens.extend([output_start] + tokenizer.encode(str(result)) + [output_end])
                elif in_python:
                    python_expression.append(next_id)
                ids = torch.cat((ids, next_ids), dim=1)

            sequences.append(ids[0].tolist())
            masks.append(token_mask[:ids.size(1)])

        return sequences, masks
