"""Pretrain the Puppy model on a prepared dataset."""

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import torch

from goodwoolf.utils import training
from goodwoolf.utils.data import NTPDataLoader, load_data
from goodwoolf.utils.models import load_config, load_model, model_name_from_path
from goodwoolf.utils.system import dataset_path, model_path


class Trainer:
    def __init__(self, model_config, tokenizer, train_config):
        self.train_config = train_config
        self.tokenizer = tokenizer

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
        self.data_dir = dataset_path(self.train_config.data_name)

        block_size = model_config.max_position_embeddings
        self.block_size = block_size
        print(
            f"examples per update will be: {self.train_config.examples_per_update:,}",
            file=sys.stderr,
        )
        print(f"device: {self.device_type}  compute: {self.compute_dtype}", file=sys.stderr)

        if tokenizer is None:
            print(f"Tokenizer not found for {self.train_config.model_name}", file=sys.stderr)
            sys.exit(1)

        self.token_bytes = training.load_token_bytes(self.train_config.model_name, self.device)

        loader_kwargs = dict(
            tokenizer=tokenizer,
            batch_size=self.train_config.batch_size,
            block_size=block_size,
            device=self.device,
            device_type=self.device_type,
            buffer_size=self.train_config.buffer_size,
            tokenizer_batch_size=self.train_config.tokenizer_batch_size,
        )
        print(f"Loading {self.train_config.data_name} samples from {self.data_dir}", file=sys.stderr)
        train_dataset = load_data(self.train_config.data_name, split="train")
        val_dataset = load_data(self.train_config.data_name, split="val")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(f"Dataset {self.train_config.data_name} is missing train/val documents")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.total_examples = len(train_dataset) * self.train_config.num_epochs
        self.loader_kwargs = loader_kwargs
        self.train_loader = NTPDataLoader(
            train_dataset,
            num_epochs=self.train_config.num_epochs,
            **loader_kwargs,
        )

        ckpt = None
        if os.path.exists(self.train_config.checkpoint_path):
            print(f"Resuming model from {self.train_config.checkpoint_path}", file=sys.stderr)
            self.model = load_model(self.train_config.model_name, self.train_config.checkpoint_path)
            ckpt = torch.load(self.train_config.checkpoint_path, map_location=self.device)
            self.best_val_bpb = ckpt.get("best_val_bpb", ckpt.get("best_val_loss", 1e9))
        elif os.path.exists(self.train_config.load_model):
            print(f"Initializing model from {self.train_config.load_model} with config", file=sys.stderr)
            self.model = load_model(self.train_config.model_name, self.train_config.load_model)
            self.best_val_bpb = 1e9
        else:
            print("Training requires initial weights (checkpoint or safetensors)", file=sys.stderr)
            sys.exit(1)

        examples_consumed = ckpt.get("examples_consumed", 0) if ckpt is not None else 0
        update_count = ckpt.get("update_count", ckpt.get("iter_num", 0)) if ckpt is not None else 0
        self.tracker = training.ExampleUpdateTracker(
            examples_per_update=self.train_config.examples_per_update,
            total_examples=self.total_examples,
            examples_consumed=examples_consumed,
            update_count=update_count,
        )
        if ckpt is not None and "loader_state" in ckpt:
            self.train_loader.load_state_dict(ckpt["loader_state"])
        elif ckpt is not None:
            print("Checkpoint has no data stream state; restarting data stream", file=sys.stderr)

        self.model.to(self.device)
        if self.device_type == "cuda":
            self.model.to(dtype=compute_ptdtype)

        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(self.compute_dtype == "float16"))
        self.optimizer = self.configure_optimizers()
        if ckpt is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

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

    def evaluate(self):
        out = {}
        self.model.eval()
        for split, dataset in (("train", self.train_dataset), ("val", self.val_dataset)):
            loader = NTPDataLoader(dataset, **self.loader_kwargs)

            def batches(loader=loader):
                for _ in range(self.train_config.eval_iters):
                    batch = loader.next_update_batch()
                    if batch is None:
                        break
                    yield batch[0], batch[1]

            out[split] = training.evaluate_bpb(
                self.model,
                batches(),
                self.train_config.eval_iters,
                self.token_bytes,
                self.ctx,
            )
        self.model.train()
        return out

    def get_lr(self, progress):
        tc = self.train_config
        if tc.warmup_ratio > 0 and progress < tc.warmup_ratio:
            return tc.learning_rate * progress / tc.warmup_ratio
        if progress <= 1.0 - tc.warmdown_ratio:
            return tc.learning_rate
        denominator = max(tc.warmdown_ratio, 1e-8)
        decay_ratio = min((progress - (1.0 - tc.warmdown_ratio)) / denominator, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return tc.min_lr + coeff * (tc.learning_rate - tc.min_lr)

    def save_checkpoint(self, last_update=False):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.tracker.update_count,
            "examples_consumed": self.tracker.examples_consumed,
            "best_val_bpb": self.best_val_bpb,
            "last_update": last_update,
            "loader_state": self.train_loader.state_dict(),
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

        while not self.tracker.done:
            group_size = self.tracker.group_size()
            progress = min(
                (self.tracker.examples_consumed + group_size) / self.total_examples,
                1.0,
            )
            lr = self.get_lr(progress) if tc.decay_lr else tc.learning_rate
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            update = self.tracker.update_count
            if tc.eval_interval > 0 and update % tc.eval_interval == 0:
                metrics = self.evaluate()
                print(
                    f"update {update}: train bpb {metrics['train']:.4f}, val bpb {metrics['val']:.4f}",
                    file=sys.stderr,
                )
                if tc.always_save_checkpoint or metrics["val"] < self.best_val_bpb:
                    self.best_val_bpb = metrics["val"]
                    self.save_checkpoint(False)

            pending_examples = 0
            micro_batches = 0
            loss_value = None
            while pending_examples < group_size:
                batch = self.train_loader.next_update_batch(max_examples=group_size - pending_examples)
                if batch is None:
                    raise RuntimeError("NTP data stream ended before the update boundary")
                inputs, targets, docs_used, _ = batch
                with self.ctx:
                    loss = self.model(input_ids=inputs, labels=targets, use_cache=False).loss
                self.scaler.scale(loss).backward()
                pending_examples += docs_used
                self.tracker.consume(docs_used)
                micro_batches += 1
                loss_value = loss.detach()
            if pending_examples != group_size:
                raise RuntimeError("NTP update consumed the wrong number of examples")

            training.finish_gradient_update(
                self.scaler,
                self.optimizer,
                self.model.parameters(),
                tc.grad_clip,
                micro_batches,
            )
            self.tracker.finish_update()
            update = self.tracker.update_count
            finish = self.tracker.done and self.train_loader.done

            if finish and tc.eval_interval > 0:
                metrics = self.evaluate()
                print(
                    f"update {update}: train bpb {metrics['train']:.4f}, val bpb {metrics['val']:.4f}",
                    file=sys.stderr,
                )
                if tc.always_save_checkpoint or metrics["val"] < self.best_val_bpb:
                    self.best_val_bpb = metrics["val"]

            t1 = time.time()
            if update % tc.log_interval == 0:
                print(
                    f"update {update}: loss {loss_value.item():.4f}, time {(t1 - t0)*1000:.2f}ms",
                    file=sys.stderr,
                )
            t0 = t1

            if finish:
                self.save_checkpoint(last_update=True)
                break


def main(argv=None):
    default_model_name = model_name_from_path(__file__)
    default_model_path = model_path(default_model_name)
    default_chpt_path = f"{default_model_path}/chpt.pt"

    parser = argparse.ArgumentParser(description="Pretrain the Puppy model")
    parser.add_argument("--load-model", default=default_model_path, help="Path to model weights")
    parser.add_argument(
        "--data-name",
        default="nvidia/climbmix",
        help="Dataset name (e.g. nvidia/climbmix)",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--examples-per-update", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--tokenizer-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmdown-ratio", type=float, default=0.9)
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
    args.model_name = default_model_name
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.examples_per_update <= 0:
        parser.error("--examples-per-update must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.buffer_size <= 0:
        parser.error("--buffer-size must be positive")
    if args.tokenizer_batch_size <= 0:
        parser.error("--tokenizer-batch-size must be positive")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("--warmup-ratio must be in [0, 1)")
    if not 0 <= args.warmdown_ratio <= 1:
        parser.error("--warmdown-ratio must be in [0, 1]")
    if args.warmup_ratio + args.warmdown_ratio > 1:
        parser.error("--warmup-ratio + --warmdown-ratio must not exceed 1")

    configs = load_config(args.model_name)
    Trainer(configs["model"], configs["tokenizer"], args).train()


if __name__ == "__main__":
    main()
