import argparse
import os
import sys
import time

import torch

from goodwoolf.utils import training
from goodwoolf.utils.common import get_pad_token_id
from goodwoolf.utils.data import RLDataLoader, load_data
from goodwoolf.utils.models import load_config, model_name_from_path
from goodwoolf.utils.system import model_path
from goodwoolf.utils.training import (
    configure_optimizers,
    load_model_state,
    make_context,
    resolve_device,
    restore_progress,
    set_lr,
)


def rl_loss(logp, targets, rewards, scale):
    advantages = rewards - rewards.mean()
    pg_obj = (logp * advantages.unsqueeze(-1)).sum()
    num_valid = (targets >= 0).sum().clamp(min=1)
    return -(pg_obj / (num_valid * scale))


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

        self.train_loader = RLDataLoader(load_data(train_config.data_name, train_config.train_split), tokenizer)
        self.val_loader = RLDataLoader(load_data(train_config.data_name, train_config.val_split), tokenizer)
        self.num_train_records = len(self.train_loader.dataset)
        if self.num_train_records == 0 or len(self.val_loader.dataset) == 0:
            raise ValueError(
                f"Dataset {train_config.data_name} is missing "
                f"{train_config.train_split}/{train_config.val_split} records"
            )
        self.total_examples = self.num_train_records * train_config.num_epochs

        self.model, checkpoint = load_model_state(
            train_config.model_name,
            train_config.load_model,
            train_config.checkpoint_path,
        )
        examples_consumed, update_count = restore_progress(
            checkpoint,
            train_config.examples_per_update,
            self.total_examples,
        )
        self.example_idx = checkpoint.get("example_idx", 0) if checkpoint else 0
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
        self.pad_token_id = get_pad_token_id(tokenizer)
        self.optimizer = configure_optimizers(self.model, train_config, self.device_type)
        if checkpoint and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.apply_lr(self.tracker.progress)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M", file=sys.stderr)

    def apply_lr(self, progress):
        lr = self.train_config.learning_rate * max(0.0, 1.0 - progress)
        set_lr(self.optimizer, lr)
        return lr

    def sample_rollout(self, update):
        tc = self.train_config
        prompt_ids, answer = self.train_loader.next_batch()
        example_idx = self.example_idx
        self.example_idx += 1
        self.model.eval()
        sequences, masks = self.model.generate(
            self.tokenizer,
            prompt_ids,
            num_samples=tc.num_samples,
            max_new_tokens=tc.max_new_tokens,
            temperature=tc.temperature,
            top_k=tc.top_k,
            do_sample=True,
            seed=update * 1000003 + example_idx * 9176,
        )
        rewards = [
            training.reward(answer, self.tokenizer.decode(sequence[len(prompt_ids):]))
            for sequence in sequences
        ]
        inputs, targets = training.pack_rollout(sequences, masks, self.pad_token_id, self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        return inputs, targets, reward_tensor, sequences

    @torch.no_grad()
    def evaluate(self, update):
        tc = self.train_config
        self.model.eval()
        k_max = min(tc.device_batch_size, tc.num_samples)
        passk = [0.0] * k_max
        num_examples = 0
        for example_idx in range(tc.eval_examples):
            tokens, answer = self.val_loader.next_batch()
            sequences, _ = self.model.generate(
                self.tokenizer,
                tokens,
                num_samples=k_max,
                max_new_tokens=tc.max_new_tokens,
                temperature=tc.temperature,
                top_k=tc.top_k,
                do_sample=True,
                seed=update * 1000003 + example_idx,
            )
            outcomes = [
                training.evaluate(answer, self.tokenizer.decode(sequence[len(tokens):]))
                for sequence in sequences
            ]
            for k in range(1, k_max + 1):
                passk[k - 1] += float(any(outcomes[:k]))
            num_examples += 1
        if num_examples == 0:
            return
        passk = [value / num_examples for value in passk]
        print(
            f"update {update}: " + ", ".join(f"pass@{k}: {value:.4f}" for k, value in enumerate(passk, 1)),
            file=sys.stderr,
        )

    def save_checkpoint(self, last_update=False):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.tracker.update_count,
            "examples_consumed": self.tracker.examples_consumed,
            "example_idx": self.example_idx,
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
        while not self.tracker.done:
            update = self.tracker.update_count
            group_size = self.tracker.group_size()
            lr_scale = self.apply_lr(self.tracker.progress)
            if tc.eval_every > 0 and update % tc.eval_every == 0:
                self.evaluate(update)

            rewards_list = []
            sequence_lengths = []
            for _ in range(group_size):
                inputs_all, targets_all, rewards_all, sequences = self.sample_rollout(update)
                self.model.train()
                num_passes = (inputs_all.size(0) + tc.device_batch_size - 1) // tc.device_batch_size
                for pass_idx in range(num_passes):
                    b0 = pass_idx * tc.device_batch_size
                    b1 = min(b0 + tc.device_batch_size, inputs_all.size(0))
                    inputs = inputs_all[b0:b1]
                    targets = targets_all[b0:b1]
                    rewards = rewards_all[b0:b1]
                    logp = -training.token_losses(self.model, inputs, targets, self.ctx)
                    loss = rl_loss(logp, targets, rewards, num_passes * group_size)
                    loss.backward()
                rewards_list.append(rewards_all.mean().item())
                sequence_lengths.extend(len(sequence) for sequence in sequences)
                self.tracker.consume(1)

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.tracker.finish_update()
            update = self.tracker.update_count
            mean_reward = sum(rewards_list) / max(1, len(rewards_list))
            mean_length = sum(sequence_lengths) / max(1, len(sequence_lengths))
            if update % tc.log_interval == 0:
                elapsed = (time.time() - t0) * 1000
                print(
                    f"update {update}/{self.tracker.total_updates}: reward {mean_reward:.4f}, "
                    f"length {mean_length:.2f}, lr scale {lr_scale:.4f}, time {elapsed:.2f}ms",
                    file=sys.stderr,
                )
            t0 = time.time()
            if tc.save_every > 0 and ((update > 0 and update % tc.save_every == 0) or self.tracker.done):
                self.save_checkpoint(last_update=self.tracker.done)


def main(argv=None):
    default_model_name = model_name_from_path(__file__)
    default_model_path = model_path(default_model_name)
    parser = argparse.ArgumentParser(description="Reinforcement learning for an HF model")
    parser.add_argument("--load-model", default=default_model_path)
    parser.add_argument("--checkpoint-path", default=f"{default_model_path}/chpt_rl.pt")
    parser.add_argument("--data-name", default="openai/gsm8k")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--examples-per-update", type=int, default=16)
    parser.add_argument("--batch-size", "--device-batch-size", dest="device_batch_size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--eval-examples", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--device", default="")
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
