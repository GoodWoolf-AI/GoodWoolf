"""Supervised fine-tuning for the Puppy model."""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import torch

from goodwoolf.utils import training
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
        self.token_bytes = training.load_token_bytes(train_config.model_name, self.device)

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

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.total_examples = len(train_dataset) * train_config.num_epochs
        self.source_kwargs = dict(
            tokenizer=tokenizer,
            max_seq_len=self.max_seq_len,
            batch_size=train_config.batch_size,
            buffer_size=train_config.buffer_size,
            device=self.device,
            device_type=self.device_type,
        )
        self.train_loader = SFTDataLoader(train_dataset, num_epochs=train_config.num_epochs, **self.source_kwargs)

        ckpt = None
        if os.path.exists(train_config.checkpoint_path):
            print(f"Resuming model from {train_config.checkpoint_path}", file=sys.stderr)
            self.model = load_model(train_config.model_name, train_config.checkpoint_path)
            ckpt = torch.load(train_config.checkpoint_path, map_location="cpu")
            self.best_val_bpb = ckpt.get("best_val_bpb", ckpt.get("best_val_loss", float("inf")))
        elif os.path.exists(train_config.load_model):
            print(f"Initializing model from {train_config.load_model}", file=sys.stderr)
            self.model = load_model(train_config.model_name, train_config.load_model)
            self.best_val_bpb = float("inf")
        else:
            print("Training requires initial weights (checkpoint or safetensors)", file=sys.stderr)
            sys.exit(1)

        examples_consumed = ckpt.get("examples_consumed", 0) if ckpt is not None else 0
        update_count = ckpt.get("update_count", ckpt.get("iter_num", 0)) if ckpt is not None else 0
        self.tracker = training.ExampleUpdateTracker(
            examples_per_update=train_config.examples_per_update,
            total_examples=self.total_examples,
            examples_consumed=examples_consumed,
            update_count=update_count,
        )
        if ckpt is not None and "loader_state" in ckpt:
            self.train_loader.load_state_dict(ckpt["loader_state"])
        elif ckpt is not None:
            print("Checkpoint has no data stream state; restarting data streams", file=sys.stderr)

        self.model.to(self.device)
        if self.device_type == "cuda":
            self.model.to(dtype=compute_dtype)

        self.scaler = torch.amp.GradScaler(
            self.device_type,
            enabled=(train_config.dtype == "float16"),
        )
        self.optimizer = self.configure_optimizers(model_config)
        if ckpt is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

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
        if tc.warmup_ratio > 0 and progress < tc.warmup_ratio:
            return (progress + 1e-8) / tc.warmup_ratio
        if progress <= 1.0 - tc.warmdown_ratio:
            return 1.0
        denominator = max(tc.warmdown_ratio, 1e-8)
        decay = min((progress - (1.0 - tc.warmdown_ratio)) / denominator, 1.0)
        return (1 - decay) + decay * tc.final_lr_frac

    def evaluate(self):
        self.model.eval()
        val_loader = SFTDataLoader(self.val_dataset, **self.source_kwargs)

        def batches():
            for _ in range(self.train_config.eval_iters):
                batch = val_loader.next_batch()
                if batch is None:
                    break
                yield batch[0], batch[1]

        metrics = {
            "val/bpb": training.evaluate_bpb(
                self.model,
                batches(),
                self.train_config.eval_iters,
                self.token_bytes,
                self.ctx,
            )
        }
        self.model.train()
        return metrics

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
            multiplier = self.get_lr_multiplier(progress)
            for group in self.optimizer.param_groups:
                group["lr"] = group["base_lr"] * multiplier

            pending_examples = 0
            micro_batches = 0
            loss_value = None
            while pending_examples < group_size:
                batch = self.train_loader.next_batch(max_examples=group_size - pending_examples)
                if batch is None:
                    raise RuntimeError("SFT data stream ended before the update boundary")
                inputs, targets, items_used, _ = batch
                with self.ctx:
                    loss = self.model(input_ids=inputs, labels=targets).loss
                self.scaler.scale(loss).backward()
                pending_examples += items_used
                self.tracker.consume(items_used)
                micro_batches += 1
                loss_value = loss.detach()
            if pending_examples != group_size:
                raise RuntimeError("SFT update consumed the wrong number of examples")

            training.finish_gradient_update(
                self.scaler,
                self.optimizer,
                self.model.parameters(),
                tc.grad_clip,
                micro_batches,
            )
            self.tracker.finish_update()
            update = self.tracker.update_count
            epoch = min(self.tracker.examples_consumed // len(self.train_dataset) + 1, tc.num_epochs)
            finish = self.tracker.done and self.train_loader.done

            if finish or (tc.eval_interval > 0 and update % tc.eval_interval == 0):
                metrics = self.evaluate()
                val_bpb = metrics["val/bpb"]
                print(
                    f"update {update:05d}: val bpb {val_bpb:.4f} | epoch {epoch}",
                    file=sys.stderr,
                )
                if val_bpb < self.best_val_bpb:
                    self.best_val_bpb = val_bpb
                    self.save_checkpoint(finish)

            t1 = time.time()
            if update % tc.log_interval == 0:
                print(
                    f"update {update}: loss {loss_value.item():.4f}, "
                    f"lr scale {multiplier:.4f}, time {(t1 - t0) * 1000:.2f}ms",
                    file=sys.stderr,
                )
            t0 = t1

            if finish:
                self.save_checkpoint(last_update=True)
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
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--examples-per-update", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", "--device-batch-size", dest="batch_size", type=int, default=8)
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
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.examples_per_update <= 0:
        parser.error("--examples-per-update must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.buffer_size <= 0:
        parser.error("--buffer-size must be positive")
    if args.eval_iters is None:
        args.eval_iters = max(1, args.eval_tokens // (args.batch_size * (args.max_seq_len or 2048)))

    configs = load_config(args.model_name)
    Trainer(configs["model"], configs["tokenizer"], args).train()


if __name__ == "__main__":
    main()
