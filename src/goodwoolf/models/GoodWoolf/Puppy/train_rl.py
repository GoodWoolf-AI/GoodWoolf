"""Reinforcement learning for the Puppy model."""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import torch

from goodwoolf.utils import training
from goodwoolf.utils.data import RLDataLoader, load_data
from goodwoolf.utils.models import load_config, load_model, model_name_from_path
from goodwoolf.utils.system import model_path


def rl_loss(logp, targets, rewards, scale):
    advantages = rewards - rewards.mean()
    pg_obj = (logp * advantages.unsqueeze(-1)).sum()
    num_valid = (targets >= 0).sum().clamp(min=1)
    return -(pg_obj / (num_valid * scale))


class Trainer:
    def __init__(self, model_config, tokenizer, train_config):
        self.train_config = train_config
        self.tokenizer = tokenizer

        has_cuda = torch.cuda.is_available()
        self.device = train_config.device or ("cuda" if has_cuda else "cpu")
        self.device_type = "cuda" if "cuda" in self.device and has_cuda else "cpu"
        compute_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[train_config.dtype]
        self.ctx = nullcontext() if self.device_type == "cpu" else torch.amp.autocast(
            device_type=self.device_type,
            dtype=compute_dtype,
        )

        if has_cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(1337)

        self.train_loader = RLDataLoader(
            load_data(train_config.data_name, train_config.train_split),
            tokenizer,
        )
        self.val_loader = RLDataLoader(
            load_data(train_config.data_name, train_config.val_split),
            tokenizer,
        )
        self.num_train_records = len(self.train_loader.dataset)
        if self.num_train_records == 0 or len(self.val_loader.dataset) == 0:
            raise ValueError(
                f"Dataset {train_config.data_name} is missing "
                f"{train_config.train_split}/{train_config.val_split} records"
            )
        self.num_steps = (self.num_train_records // train_config.examples_per_step) * train_config.num_epochs
        if self.num_steps < 1:
            raise ValueError("Training horizon is zero; increase --num-epochs or reduce --examples-per-step")
        print(
            f"{train_config.data_name} RL: {self.num_train_records:,} train records, "
            f"{self.num_steps:,} steps",
            file=sys.stderr,
        )

        if os.path.exists(train_config.checkpoint_path):
            print(f"Resuming model from {train_config.checkpoint_path}", file=sys.stderr)
            self.model = load_model(train_config.model_name, train_config.checkpoint_path)
            ckpt = torch.load(train_config.checkpoint_path, map_location="cpu")
            self.step = ckpt.get("step", 0)
        elif os.path.exists(train_config.load_model):
            print(f"Initializing model from {train_config.load_model}", file=sys.stderr)
            self.model = load_model(train_config.model_name, train_config.load_model)
            self.step = 0
            ckpt = None
        else:
            print("Training requires initial weights (checkpoint or safetensors)", file=sys.stderr)
            sys.exit(1)

        self.model.to(self.device)
        if self.device_type == "cuda":
            self.model.to(dtype=compute_dtype)

        self.example_idx = 0
        self.pad_token_id = tokenizer.encode_special("<|assistant_end|>")

        self.optimizer = self.configure_optimizers(model_config)
        if ckpt is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.apply_lr(0)

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"number of parameters: {n_params / 1e6:.2f}M", file=sys.stderr)

    def configure_optimizers(self, model_config):
        n_embd = getattr(model_config, "n_embd", None)
        if n_embd is None:
            n_embd = int(getattr(model_config, "depth", 12)) * int(getattr(model_config, "aspect_ratio", 64))
        scale = (n_embd / 768) ** -0.5
        tc = self.train_config
        params = dict(self.model.named_parameters())
        groups = []

        def add(suffixes, lr, betas=(0.9, 0.95), weight_decay=0.0):
            names = [f"model.{name}" for name in suffixes if f"model.{name}" in params]
            if names:
                groups.append({
                    "params": [params[name] for name in names],
                    "lr": lr,
                    "betas": betas,
                    "eps": 1e-10,
                    "weight_decay": weight_decay,
                    "initial_lr": lr,
                })

        add(["lm_head.weight"], tc.unembedding_lr * scale, weight_decay=tc.weight_decay)
        add(["transformer.wte.weight"], tc.embedding_lr * scale, weight_decay=tc.weight_decay)
        add(
            [name.removeprefix("model.") for name in params if name.startswith("model.value_embeds.")],
            tc.embedding_lr * scale * 0.5,
            weight_decay=tc.weight_decay,
        )
        add(["resid_lambdas"], 0.005 * scale, betas=(0.9, 0.999))
        add(["x0_lambdas"], 0.5, betas=(0.9, 0.999))
        add(["smear_gate.weight", "smear_lambda", "backout_lambda"], 0.2, betas=(0.9, 0.999))

        matrix_names = []
        for name in params:
            if name.startswith("model.") and params[name].ndim >= 2:
                if any(part in name for part in ("lm_head", "wte", "value_embeds", "smear_gate")):
                    continue
                matrix_names.append(name.removeprefix("model."))
        add(matrix_names, tc.matrix_lr * scale)

        scalar_names = [
            name.removeprefix("model.")
            for name in params
            if name.startswith("model.") and params[name].ndim < 2
            and name.removeprefix("model.") not in {"resid_lambdas", "x0_lambdas", "smear_lambda", "backout_lambda"}
        ]
        add(scalar_names, tc.matrix_lr * scale)

        for group in groups:
            group["lr"] *= tc.init_lr_frac
            group["initial_lr"] = group["lr"]
        return torch.optim.AdamW(groups, fused=self.device_type == "cuda")

    def apply_lr(self, step):
        multiplier = max(0.0, 1.0 - step / self.num_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        return multiplier

    def sample_rollout(self, step):
        tc = self.train_config
        prompt_ids, answer = self.train_loader.next_batch()
        example_idx = self.example_idx
        self.example_idx += 1
        self.model.eval()

        sequences, masks = [], []
        num_sampling_steps = (tc.num_samples + tc.device_batch_size - 1) // tc.device_batch_size
        for sampling_step in range(num_sampling_steps):
            sample_count = min(tc.device_batch_size, tc.num_samples - sampling_step * tc.device_batch_size)
            seed = (step * 1000003 + example_idx * 9176 + sampling_step) & 0x7FFFFFFF
            sample_sequences, sample_masks = self.model.generate_sequences(
                self.tokenizer,
                prompt_ids,
                num_samples=sample_count,
                max_new_tokens=tc.max_new_tokens,
                temperature=tc.temperature,
                top_k=tc.top_k,
                seed=seed,
            )
            sequences.extend(sample_sequences)
            masks.extend(sample_masks)

        rewards = [
            training.reward(
                answer,
                self.tokenizer.decode(sequence[len(prompt_ids):]),
            )
            for sequence in sequences
        ]
        inputs, targets = training.pack_rollout(sequences, masks, self.pad_token_id, self.device)
        reward_tensor = torch.tensor(rewards, dtype=torch.float, device=self.device)
        return inputs, targets, reward_tensor, sequences

    @torch.no_grad()
    def evaluate(self, step):
        tc = self.train_config
        self.model.eval()
        k_max = min(tc.device_batch_size, tc.num_samples)
        passk = [0.0] * k_max
        num_examples = 0

        for example_idx in range(tc.eval_examples):
            tokens, answer = self.val_loader.next_batch()
            seed = (step * 1000003 + example_idx) & 0x7FFFFFFF
            sequences, _ = self.model.generate_sequences(
                self.tokenizer,
                tokens,
                num_samples=k_max,
                max_new_tokens=tc.max_new_tokens,
                temperature=tc.temperature,
                top_k=tc.top_k,
                seed=seed,
            )
            outcomes = [
                training.evaluate(
                    answer,
                    self.tokenizer.decode(sequence[len(tokens):]),
                )
                for sequence in sequences
            ]
            for k in range(1, k_max + 1):
                passk[k - 1] += float(any(outcomes[:k]))
            num_examples += 1

        if num_examples == 0:
            return
        passk = [value / num_examples for value in passk]
        print(
            f"step {step}: " + ", ".join(f"pass@{k}: {value:.4f}" for k, value in enumerate(passk, 1)),
            file=sys.stderr,
        )

    def save_checkpoint(self, step):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "num_steps": self.num_steps,
        }
        parent = os.path.dirname(self.train_config.checkpoint_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(checkpoint, self.train_config.checkpoint_path)
        print(f"checkpoint saved to {self.train_config.checkpoint_path}", file=sys.stderr)

    def train(self):
        tc = self.train_config
        t0 = time.time()

        for step in range(self.step, self.num_steps):
            if tc.eval_every > 0 and step % tc.eval_every == 0:
                self.evaluate(step)

            rewards_list, sequence_lengths = [], []
            for example_step in range(tc.examples_per_step):
                inputs_all, targets_all, rewards_all, sequences = self.sample_rollout(step)
                self.model.train()
                num_passes = (inputs_all.size(0) + tc.device_batch_size - 1) // tc.device_batch_size
                for pass_idx in range(num_passes):
                    b0 = pass_idx * tc.device_batch_size
                    b1 = min(b0 + tc.device_batch_size, inputs_all.size(0))
                    inputs = inputs_all[b0:b1]
                    targets = targets_all[b0:b1]
                    rewards = rewards_all[b0:b1]
                    with self.ctx:
                        logp = -self.model(
                            input_ids=inputs,
                            labels=targets,
                            loss_reduction="none",
                        ).loss.view_as(inputs)
                    loss = rl_loss(logp, targets, rewards, num_passes * tc.examples_per_step)
                    loss.backward()
                rewards_list.append(rewards_all.mean().item())
                sequence_lengths.extend(len(sequence) for sequence in sequences)

            mean_reward = sum(rewards_list) / len(rewards_list)
            mean_length = sum(sequence_lengths) / len(sequence_lengths)
            multiplier = self.apply_lr(step)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step = step + 1

            if step % tc.log_interval == 0:
                print(
                    f"step {step}/{self.num_steps}: reward {mean_reward:.4f}, "
                    f"length {mean_length:.2f}, lr scale {multiplier:.4f}, "
                    f"time {(time.time() - t0) * 1000:.2f}ms",
                    file=sys.stderr,
                )
            t0 = time.time()

            if tc.save_every > 0 and ((step > 0 and step % tc.save_every == 0) or step == self.num_steps - 1):
                self.save_checkpoint(self.step)


def main(argv=None):
    default_model_name = model_name_from_path(__file__)
    default_model_path = model_path(default_model_name)
    default_checkpoint_path = f"{default_model_path}/chpt_rl.pt"

    parser = argparse.ArgumentParser(description="Reinforcement learning for the Puppy model")
    parser.add_argument("--load-model", default=default_model_path, help="Path to model weights")
    parser.add_argument("--checkpoint-path", default=default_checkpoint_path)
    parser.add_argument(
        "--data-name",
        default="openai/gsm8k",
        help="Dataset with answer records (e.g. openai/gsm8k)",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--batch-size", "--device-batch-size", dest="device_batch_size", type=int, default=8)
    parser.add_argument("--examples-per-step", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--embedding-lr", type=float, default=0.2)
    parser.add_argument("--unembedding-lr", type=float, default=0.004)
    parser.add_argument("--matrix-lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--init-lr-frac", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=60)
    parser.add_argument("--eval-examples", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=60)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)
    args.model_name = default_model_name

    configs = load_config(args.model_name)
    Trainer(configs["model"], configs["tokenizer"], args).train()


if __name__ == "__main__":
    main()
