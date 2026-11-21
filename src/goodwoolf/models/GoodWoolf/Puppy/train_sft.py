"""Supervised fine-tuning for the Puppy model."""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import torch

from goodwoolf.utils.data import SFTDataLoader, load_data
from goodwoolf.utils.models import load_config, load_model, model_name_from_path
from goodwoolf.utils.system import model_path


class Trainer:
    def __init__(self, model_config, tokenizer, train_config):
        self.train_config = train_config
        self.tokenizer = tokenizer

        has_cuda = torch.cuda.is_available()
        self.device = train_config.device or ("cuda" if has_cuda else "cpu")
        self.device_type = "cuda" if "cuda" in self.device and has_cuda else "cpu"
        compute_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[train_config.dtype]
        self.ctx = nullcontext() if self.device_type == "cpu" else torch.amp.autocast(
            device_type=self.device_type,
            dtype=compute_dtype,
        )

        if has_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(1337)

        self.max_seq_len = train_config.max_seq_len or model_config.max_position_embeddings
        if train_config.total_batch_size is not None:
            tokens_per_micro_batch = train_config.batch_size * self.max_seq_len
            if train_config.total_batch_size % tokens_per_micro_batch:
                raise ValueError("total batch size must be a multiple of batch_size * max_seq_len")
            train_config.gradient_accumulation_steps = (
                train_config.total_batch_size // tokens_per_micro_batch
            )

        train_dataset = load_data(train_config.data_name, split=train_config.train_split)
        val_dataset = load_data(train_config.data_name, split=train_config.val_split)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(
                f"Dataset {train_config.data_name} is missing "
                f"{train_config.train_split}/{train_config.val_split} records"
            )
        print(
            f"{train_config.data_name} SFT: {len(train_dataset):,} train records, "
            f"{len(val_dataset):,} val records",
            file=sys.stderr,
        )

        loader_kwargs = dict(
            tokenizer=tokenizer,
            batch_size=train_config.batch_size,
            max_seq_len=self.max_seq_len,
            device=self.device,
            device_type=self.device_type,
            buffer_size=train_config.buffer_size,
        )
        self.train_loader = SFTDataLoader(
            train_dataset,
            num_iterations=train_config.num_iterations,
            **loader_kwargs,
        )
        self.val_dataset = val_dataset
        self.val_loader_kwargs = loader_kwargs

        if os.path.exists(train_config.checkpoint_path):
            print(f"Resuming model from {train_config.checkpoint_path}", file=sys.stderr)
            self.model = load_model(train_config.model_name, train_config.checkpoint_path)
            ckpt = torch.load(train_config.checkpoint_path, map_location="cpu")
            self.iter_num = ckpt.get("iter_num", 0)
            self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        elif os.path.exists(train_config.load_model):
            print(f"Initializing model from {train_config.load_model}", file=sys.stderr)
            self.model = load_model(train_config.model_name, train_config.load_model)
            self.iter_num = 0
            self.best_val_loss = float("inf")
        else:
            print("Training requires initial weights (checkpoint or safetensors)", file=sys.stderr)
            sys.exit(1)

        self.model.to(self.device)
        if self.device_type == "cuda":
            self.model.to(dtype=compute_dtype)

        self.scaler = torch.amp.GradScaler(
            self.device_type,
            enabled=(train_config.dtype == "float16"),
        )
        self.optimizer = self.configure_optimizers(model_config)
        resume_ckpt = locals().get("ckpt")
        if resume_ckpt is not None and "optimizer" in resume_ckpt:
            self.optimizer.load_state_dict(resume_ckpt["optimizer"])

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M", file=sys.stderr)

    def configure_optimizers(self, model_config):
        n_embd = getattr(model_config, "n_embd", None)
        if n_embd is None:
            n_embd = int(getattr(model_config, "depth", 12)) * int(getattr(model_config, "aspect_ratio", 64))
        scale = (n_embd / 768) ** -0.5
        embedding_lr = self.train_config.embedding_lr * scale
        unembedding_lr = self.train_config.unembedding_lr * scale
        matrix_lr = self.train_config.matrix_lr * scale

        params = dict(self.model.named_parameters())
        groups = []

        def add(names, lr, betas=(0.9, 0.95), weight_decay=0.0):
            names = [name for name in names if name in params]
            if names:
                groups.append({
                    "params": [params[name] for name in names],
                    "lr": lr,
                    "betas": betas,
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                    "base_lr": lr,
                })

        add(["lm_head.weight"], unembedding_lr)
        add(["transformer.wte.weight"], embedding_lr, betas=(0.9, 0.95))
        add(
            [name for name in params if name.startswith("value_embeds.")],
            embedding_lr * 0.5,
        )
        add(["resid_lambdas"], 0.005 * scale, betas=(0.9, 0.999))
        add(["x0_lambdas"], 0.5, betas=(0.9, 0.999))
        add(["smear_gate.weight", "smear_lambda", "backout_lambda"], 0.2, betas=(0.9, 0.999))

        matrix_names = []
        for name in params:
            if params[name].ndim >= 2:
                if any(part in name for part in ("lm_head", "wte", "value_embeds", "smear_gate")):
                    continue
                matrix_names.append(name)
        add(matrix_names, matrix_lr)

        scalar_names = [
            name for name in params
            if params[name].ndim < 2
            and name not in {"resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda"}
        ]
        add(scalar_names, matrix_lr)

        for group in groups:
            group["lr"] *= self.train_config.init_lr_frac
            group["base_lr"] = group["lr"]
        return torch.optim.AdamW(groups, fused=self.device_type == "cuda")

    def get_lr_multiplier(self, progress):
        tc = self.train_config
        if progress < tc.warmup_ratio:
            return (progress + 1e-8) / tc.warmup_ratio
        if progress <= 1.0 - tc.warmdown_ratio:
            return 1.0
        decay = (progress - (1.0 - tc.warmdown_ratio)) / tc.warmdown_ratio
        return (1 - decay) + decay * tc.final_lr_frac

    @torch.no_grad()
    def estimate_loss(self):
        self.model.eval()
        loader = SFTDataLoader(self.val_dataset, **self.val_loader_kwargs)
        losses = []
        for _ in range(self.train_config.eval_iters):
            inputs, targets, _, _, _ = loader.next_batch()
            with self.ctx:
                loss = self.model(input_ids=inputs, labels=targets).loss
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def save_checkpoint(self, last_step=False):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iter_num": self.iter_num,
            "best_val_loss": self.best_val_loss,
            "last_step": last_step,
        }
        parent = os.path.dirname(self.train_config.checkpoint_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(checkpoint, self.train_config.checkpoint_path)
        print(f"checkpoint saved to {self.train_config.checkpoint_path}", file=sys.stderr)

    def train(self):
        tc = self.train_config
        t0 = time.time()
        self.model.train()

        while True:
            inputs, targets, progress, last_step, epoch = self.train_loader.next_batch()
            multiplier = self.get_lr_multiplier(progress)
            for group in self.optimizer.param_groups:
                group["lr"] = group["base_lr"] * multiplier

            loss = None
            for _ in range(tc.gradient_accumulation_steps):
                with self.ctx:
                    loss = self.model(input_ids=inputs, labels=targets).loss
                    loss = loss / tc.gradient_accumulation_steps
                inputs, targets, next_progress, step_last, epoch = self.train_loader.next_batch()
                progress = next_progress
                last_step = last_step or step_last
                self.scaler.scale(loss).backward()

            if tc.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.iter_num += 1

            finish = last_step or (tc.num_iterations > 0 and self.iter_num >= tc.num_iterations)
            if finish or (tc.eval_interval > 0 and self.iter_num % tc.eval_interval == 0):
                val_loss = self.estimate_loss()
                print(
                    f"step {self.iter_num:05d}: val loss {val_loss:.4f} | epoch {epoch}",
                    file=sys.stderr,
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(finish)

            t1 = time.time()
            if self.iter_num % tc.log_interval == 0:
                print(
                    f"iter {self.iter_num}: loss {loss.item() * tc.gradient_accumulation_steps:.4f}, "
                    f"lr scale {multiplier:.4f}, time {(t1 - t0) * 1000:.2f}ms",
                    file=sys.stderr,
                )
            t0 = t1

            if finish:
                self.save_checkpoint(last_step=True)
                break


def main(argv=None):
    default_model_name = model_name_from_path(__file__)
    default_model_path = model_path(default_model_name)
    default_checkpoint_path = f"{default_model_path}/chpt_sft.pt"

    parser = argparse.ArgumentParser(description="Supervised fine-tuning of the Puppy model")
    parser.add_argument("--load-model", default=default_model_path, help="Path to model weights")
    parser.add_argument("--checkpoint-path", default=default_checkpoint_path)
    parser.add_argument("--data-name", default="HuggingFaceTB/smoltalk", help="Conversation dataset")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", "--device-batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--total-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--embedding-lr", type=float, default=0.3)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--init-lr-frac", type=float, default=0.8)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-frac", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", "--eval-every", dest="eval_interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--eval-tokens", type=int, default=40 * 524288)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)
    args.model_name = default_model_name
    if args.eval_iters is None:
        args.eval_iters = max(1, args.eval_tokens // (args.batch_size * (args.max_seq_len or 2048)))

    configs = load_config(args.model_name)
    Trainer(configs["model"], configs["tokenizer"], args).train()


if __name__ == "__main__":
    main()
