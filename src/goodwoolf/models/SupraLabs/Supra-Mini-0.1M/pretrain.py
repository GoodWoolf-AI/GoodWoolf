"""
Pretrain a Supra-Mini model on Shakespeare.
"""

import os
import sys
import time
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch

from goodwoolf.utils.system import models_path, data_path
from goodwoolf.utils.models import load_config, load_model, model_name_from_path


class Trainer:
    def __init__(self, model_config, train_config):
        self.train_config = train_config

        has_cuda = torch.cuda.is_available()
        self.device = self.train_config.device if self.train_config.device else ("cuda" if has_cuda else "cpu")
        self.device_type = "cuda" if "cuda" in self.device and has_cuda else "cpu"

        self.compute_dtype = self.train_config.dtype
        compute_ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.compute_dtype]
        self.ctx = nullcontext() if self.device_type == "cpu" else torch.amp.autocast(device_type=self.device_type, dtype=compute_ptdtype)

        if has_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        torch.manual_seed(1337)
        self.data_dir = data_path(self.train_config.model_name, self.train_config.data_name)

        block_size = model_config.max_position_embeddings
        self.block_size = block_size
        tpi = self.train_config.gradient_accumulation_steps * self.train_config.batch_size * block_size
        print(f"tokens per iteration will be: {tpi:,}", file=sys.stderr)
        print(f"device: {self.device_type}  compute: {self.compute_dtype}", file=sys.stderr)

        # ---- model ----
        if os.path.exists(self.train_config.checkpoint_path):
            print(f"Resuming model from {self.train_config.checkpoint_path}", file=sys.stderr)
            self.model = load_model(self.train_config.model_name, self.train_config.checkpoint_path)
            ckpt = torch.load(self.train_config.checkpoint_path, map_location=self.device)
            self.iter_num = ckpt.get("iter_num", 0)
            self.best_val_loss = ckpt.get("best_val_loss", 1e9)
        elif os.path.exists(self.train_config.load_model):
            print(f"Initializing model from {self.train_config.load_model} with config", file=sys.stderr)
            self.model = load_model(self.train_config.model_name, self.train_config.load_model)
            self.iter_num = 0
            self.best_val_loss = 1e9
        else:
            print("Training requires initial weights (checkpoint or safetensors)", file=sys.stderr)
            sys.exit(1)

        self.model.to(self.device)
        if self.device_type == "cuda":
            self.model.to(dtype=compute_ptdtype)

        # ---- optimizer / scaler ----
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.compute_dtype == "float16"))
        self.optimizer = self.configure_optimizers()

        resume_ckpt = locals().get("ckpt")
        if resume_ckpt is not None and "optimizer" in resume_ckpt:
            self.optimizer.load_state_dict(resume_ckpt["optimizer"])

        # ---- compile ----
        if self.train_config.compile and self.device_type == "cuda":
            print("compiling the model... (takes a ~minute)", file=sys.stderr)
            self.model = torch.compile(self.model)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M", file=sys.stderr)

    def configure_optimizers(self):
        tc = self.train_config
        decay = set()
        no_decay = set()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.add(name)
            else:
                decay.add(name)
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": tc.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=tc.learning_rate, betas=(tc.beta1, tc.beta2), fused=self.device_type == "cuda")

    def get_batch(self, split):
        data = np.memmap(os.path.join(self.data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - self.block_size, (self.train_config.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        if self.device_type == "cuda":
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "test"]:
            losses = torch.zeros(self.train_config.eval_iters)
            for k in range(self.train_config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    loss = self.model(input_ids=X, labels=Y).loss
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def get_lr(self, it):
        tc = self.train_config
        if it < tc.warmup_iters:
            return tc.learning_rate * it / tc.warmup_iters
        if it > tc.lr_decay_iters:
            return tc.min_lr
        decay_ratio = (it - tc.warmup_iters) / (tc.lr_decay_iters - tc.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)

    def train(self):
        tc = self.train_config
        X, Y = self.get_batch("train")
        t0 = time.time()
        local_iter_num = 0

        while True:
            lr = self.get_lr(self.iter_num) if tc.decay_lr else tc.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # evaluate & checkpoint
            if self.iter_num % tc.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"step {self.iter_num}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}", file=sys.stderr)
                if tc.always_save_checkpoint or losses["test"] < self.best_val_loss:
                    self.best_val_loss = losses["test"]
                    torch.save({
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "iter_num": self.iter_num,
                        "best_val_loss": self.best_val_loss,
                    }, tc.checkpoint_path)
                    print(f"checkpoint saved to {tc.checkpoint_path}", file=sys.stderr)

            # forward-backward
            for _ in range(tc.gradient_accumulation_steps):
                with self.ctx:
                    loss = self.model(input_ids=X, labels=Y, use_cache=False).loss
                    loss = loss / tc.gradient_accumulation_steps
                X, Y = self.get_batch("train")
                self.scaler.scale(loss).backward()

            if tc.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % tc.log_interval == 0:
                lossf = loss.item() * tc.gradient_accumulation_steps
                print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms", file=sys.stderr)

            self.iter_num += 1
            local_iter_num += 1

            if self.iter_num > tc.max_iters:
                break


def main(argv=None):
    default_model_name = model_name_from_path(__file__)
    default_model_path = models_path(default_model_name)
    default_chpt_path = f"{default_model_path}/chpt.pt"

    parser = argparse.ArgumentParser(description="Pretrain a SupraLabs model")
    parser.add_argument("--model-name", default=default_model_name, help=f"Model name (default: {default_model_name})")
    parser.add_argument("--load-model", default=default_model_path, help="safetensors file with model weights")
    parser.add_argument("--data-name", default="shakespeare", help="Dataset name (e.g. shakespeare)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--lr-decay-iters", type=int, default=10000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--decay-lr", action="store_true", default=True)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--always-save-checkpoint", action="store_true", default=True)
    parser.add_argument("--checkpoint-path", default=default_chpt_path)
    parser.add_argument("--device", default="")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)

    configs = load_config(args.model_name)
    Trainer(configs["model"], args).train()


if __name__ == "__main__":
    main()
