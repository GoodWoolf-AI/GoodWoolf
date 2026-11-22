import argparse
import os
import sys
import time
from pathlib import Path

import torch

from goodwoolf.utils import training
from goodwoolf.utils.data import NTPDataLoader, load_data
from goodwoolf.utils.models import load_config, model_name_from_path
from goodwoolf.utils.system import import_file, model_path

_common = import_file(
    Path(__file__).with_name("common.py"), f"common_{model_name_from_path(__file__)}"
)
configure_optimizers = _common.configure_optimizers
get_lr = _common.get_lr
load_model_state = _common.load_model_state
make_context = _common.make_context
resolve_device = _common.resolve_device
restore_progress = _common.restore_progress
set_lr = _common.set_lr


class Trainer:
    def __init__(self, model_config, tokenizer, train_config):
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.device, self.device_type = resolve_device(train_config)
        self.compute_dtype, self.ctx = make_context(self.device_type, train_config.dtype)

        if self.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(1337)

        if tokenizer is None:
            print(f"Tokenizer not found for {train_config.model_name}", file=sys.stderr)
            sys.exit(1)

        self.block_size = int(model_config.max_position_embeddings)
        self.token_bytes = training.load_token_bytes(train_config.model_name, self.device)
        train_dataset = load_data(train_config.data_name, split="train")
        val_dataset = load_data(train_config.data_name, split="val")
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError(f"Dataset {train_config.data_name} is missing train/val documents")
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.total_examples = len(train_dataset) * train_config.num_epochs

        self.loader_kwargs = dict(
            tokenizer=tokenizer,
            batch_size=train_config.batch_size,
            block_size=self.block_size,
            device=self.device,
            device_type=self.device_type,
            buffer_size=train_config.buffer_size,
            tokenizer_batch_size=train_config.tokenizer_batch_size,
        )
        self.train_loader = NTPDataLoader(
            train_dataset,
            num_epochs=train_config.num_epochs,
            **self.loader_kwargs,
        )

        self.model, checkpoint = load_model_state(
            train_config.model_name,
            train_config.load_model,
            train_config.checkpoint_path,
        )
        self.best_val_bpb = checkpoint.get("best_val_bpb", checkpoint.get("best_val_loss", float("inf"))) if checkpoint else float("inf")
        examples_consumed, update_count = restore_progress(
            checkpoint,
            train_config.examples_per_update,
            self.total_examples,
        )
        self.tracker = training.ExampleUpdateTracker(
            examples_per_update=train_config.examples_per_update,
            total_examples=self.total_examples,
            examples_consumed=examples_consumed,
            update_count=update_count,
        )
        if checkpoint and "loader_state" in checkpoint:
            self.train_loader.load_state_dict(checkpoint["loader_state"])
        elif checkpoint:
            print("Checkpoint has no data stream state; restarting data stream", file=sys.stderr)

        self.model.to(self.device)
        if self.device_type == "cuda":
            self.model.to(dtype=self.compute_dtype)
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=(train_config.dtype == "float16"))
        self.optimizer = configure_optimizers(self.model, train_config, self.device_type)
        if checkpoint and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if train_config.compile and self.device_type == "cuda":
            self.model = torch.compile(self.model)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M", file=sys.stderr)

    def evaluate(self):
        self.model.eval()
        results = {}
        for split, dataset in (("train", self.train_dataset), ("val", self.val_dataset)):
            loader = NTPDataLoader(dataset, **self.loader_kwargs)

            def batches(source=loader):
                for _ in range(self.train_config.eval_iters):
                    batch = source.next_update_batch()
                    if batch is None:
                        break
                    yield batch[0], batch[1]

            results[split] = training.evaluate_bpb(
                self.model,
                batches(),
                self.train_config.eval_iters,
                self.token_bytes,
                self.ctx,
            )
        self.model.train()
        return results

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
            progress = min((self.tracker.examples_consumed + group_size) / self.total_examples, 1.0)
            lr = get_lr(progress, tc) if tc.decay_lr else tc.learning_rate
            set_lr(self.optimizer, lr)

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
                losses = training.token_losses(self.model, inputs, targets, self.ctx)
                loss = training.masked_mean(losses, targets)
                self.scaler.scale(loss).backward()
                pending_examples += docs_used
                self.tracker.consume(docs_used)
                micro_batches += 1
                loss_value = loss.detach()
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

            if update % tc.log_interval == 0:
                elapsed = (time.time() - t0) * 1000
                print(f"update {update}: loss {loss_value.item():.4f}, time {elapsed:.2f}ms", file=sys.stderr)
            t0 = time.time()
            if finish:
                self.save_checkpoint(last_update=True)
                break


def main(argv=None):
    default_model_name = model_name_from_path(__file__)
    default_model_path = model_path(default_model_name)
    parser = argparse.ArgumentParser(description="Pretrain an HF model")
    parser.add_argument("--load-model", default=default_model_path)
    parser.add_argument("--checkpoint-path", default=f"{default_model_path}/chpt.pt")
    parser.add_argument("--data-name", default="nvidia/climbmix")
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
    parser.add_argument("--device", default="")
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)
    args.model_name = default_model_name
    if args.num_epochs <= 0:
        parser.error("--num-epochs must be positive")
    if args.examples_per_update <= 0:
        parser.error("--examples-per-update must be positive")
    configs = load_config(args.model_name)
    Trainer(configs["model"], configs["tokenizer"], args).train()


if __name__ == "__main__":
    main()
