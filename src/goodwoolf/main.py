#!/usr/bin/env python3
"""Entry point for GoodWoolf CLI commands."""

import click
import importlib.util
from pathlib import Path
import sys

import goodwoolf.utils.models as gw_models
import goodwoolf.utils.ledger as gw_ledger
from goodwoolf.utils.system import import_file, model_path

TRAIN_STAGES = ("TOK", "NTP", "SFT", "RL")

@click.group()
def main():
    """GoodWoolf builds foundation models on commodity hardware."""


@main.command()
@click.option("--list", "list_items", is_flag=True, default=False, help="List model vendors.")
@click.option("--remote", required=False, metavar="REPO", help="Remote repo for vendors (supported: hf).")
def vendor(list_items, remote):
    """Inspect model vendors."""
    if not list_items:
        raise click.UsageError("You must specify --list.")
    if remote and remote.lower() == "hf":
        gw_models.list_remote_vendors()
    else:
        gw_models.list_vendors()


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--create", required=False, metavar="TYPE", help="Create a new ledger entry with an operator TYPE.")
@click.option("--check", is_flag=True, default=False, help="Validate the ledger entry agains the local model.")
def ledger(model_name, create, check):
    "Update the model training ledger."
    try:
        if create:
            gw_ledger.create_entry(model_name, operator=create)
        if check:
            gw_ledger.check_entry(model_name)
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--id", required=False, metavar="ID", help="The model ledger entry id.")
def pull(model_name, id):
    "Pull a model from the training ledger."
    try:
        gw_ledger.pull_model(model_name, id)
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--id", required=True, metavar="ID", help="The model ledger entry id.")
def checkout(model_name, id):
    "Check out a model checkpoint as the active model."
    try:
        gw_ledger.model_checkout(model_name, id)
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.argument("model-name", required=False, help="Model name in format <vendor>/<model>, or vendor for --list.")
@click.option("--list", "list_items", is_flag=True, default=False, help="List models.")
@click.option("--remote", required=False, metavar="REPO", help="Remote repo for models (supported: hf).")
@click.option("--config", is_flag=True, default=False, help="Fetch model configuration to the model folder.")
@click.option(
    "--template",
    required=False,
    is_flag=False,
    flag_value="GoodWoolf/HF",
    default=None,
    metavar="MODEL_NAME",
    help="Create model training code from a template"
)
@click.option("--weights", is_flag=True, default=False, help="Create randomized model weights in the input folder")
def model(model_name, list_items, remote, config, template, weights):
    """Create or list models."""
    if list_items:
        if config or template is not None or weights:
            raise click.UsageError(
                "--list cannot be combined with --config, --template, or --weights."
            )
        if remote and remote.lower() == "hf":
            gw_models.list_remote_models(model_name)
        elif model_name:
            if "/" in model_name:
                raise click.UsageError("Use a vendor name with --list, not <vendor>/<model>.")
            gw_models.list_models(model_name)
        else:
            gw_models.list_all_models()
        return
    if remote:
        raise click.UsageError("--remote only applies with --list.")
    if model_name is None:
        raise click.UsageError("You must specify MODEL_NAME or use --list.")
    if not config and template is None and not weights:
        raise click.UsageError(
            "You must specify at least one action option: "
            "--config, --template, or --weights."
        )
    gw_models.create_model(model_name, config, template, weights)


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--load-path", metavar="PATH", help="Load weights from safetensors or checkpoint.")
@click.option("--prompt", metavar="PROMPT", default="", help="Model input prompt.")
@click.option("--max-tokens", metavar="TOKENS", type=int, default=100, help="Max number of tokens to generate.")
def run(model_name, load_path, prompt, max_tokens):
    """Run a model to generate text from stdin prompt."""
    if load_path is None:
        load_path = model_path(model_name)
    response = gw_models.generate(model_name, load_path, prompt, max_tokens)
    print(f"Output:\n{response}")


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--load-path", metavar="PATH", help="Path to model weights.")
@click.option("--prompt", metavar="PROMPT", default="", help="Prompt the model and get a single response.")
@click.option("--max-tokens", metavar="TOKENS", type=int, default=256, help="Max number of tokens in the response.")
@click.option("--temperature", type=float, default=0.6, help="Sampling temperature.")
@click.option("--top-k", type=int, default=50, help="Top-k sampling parameter.")
@click.option("--device", default="", help="Device to run on (cuda, cpu, or torch device string).")
def chat(model_name, load_path, prompt, max_tokens, temperature, top_k, device):
    """Chat with a model."""
    import torch

    if load_path is None:
        load_path = model_path(model_name)
    try:
        configs = gw_models.load_config(model_name)
        tokenizer = configs["tokenizer"]
        model = gw_models.load_model(model_name, load_path)
        if not hasattr(model, "chat"):
            raise ValueError(f"Model {model_name} does not implement chat()")

        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)
        if "cuda" in str(target_device):
            model.to(dtype=torch.bfloat16)
        model.eval()

        messages = []
        if not prompt:
            print("\nGoodWoolf Chat")
            print("-" * 50)
            print("Type 'quit' or 'exit' to end the conversation")
            print("Type 'clear' to start a new conversation")
            print("-" * 50)

        while True:
            if prompt:
                user_input = prompt
            else:
                try:
                    user_input = input("\nUser: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                messages = []
                print("Conversation cleared.")
                continue
            if not user_input:
                continue

            messages.append({"role": "user", "content": user_input})
            response = model.chat(
                tokenizer,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            print(f"\nAssistant: {response}")
            messages.append({"role": "assistant", "content": response})
            if prompt:
                break
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--data-name", required=True, metavar="DATASET", help="Dataset name (e.g. openai/gsm8k).")
@click.option("--model-id", "id", required=True, metavar="ID", help="Model checkpoint id to merge.")
@click.option("--batch-size", type=int, default=8, help="Validation batch size.")
@click.option("--eval-iters", type=int, default=100, help="Validation iteration count.")
@click.option("--dtype", metavar="TYPE", default="bfloat16", help="Model runtime type.")
def merge(model_name, data_name, id, batch_size, eval_iters, dtype):
    """Merge a checkpoint into the active model."""
    gw_models.merge(model_name, data_name, id, batch_size, eval_iters, dtype)


@main.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument("data-name", required=False, help="Dataset name (e.g. openai/gsm8k).")
@click.option("--list", "list_items", is_flag=True, default=False, help="List training datasets.")
@click.pass_context
def dataset(ctx, data_name, list_items):
    """Prepare or list datasets."""
    if list_items:
        gw_models.list_datasets()
        return
    if data_name is None:
        raise click.UsageError("You must specify DATASET or use --list.")
    module_path = gw_models.DATASET_BASE / data_name / "dataset.py"
    package_name = f"goodwoolf.data.{data_name.replace('/', '.')}.dataset"
    import_file(str(module_path), package_name)

    sys.modules[package_name].main(ctx.args)


@main.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>")
@click.option(
    "--stage",
    "train_stage",
    type=click.Choice(TRAIN_STAGES, case_sensitive=False),
    default="NTP",
    show_default=True,
    help=(
        "TOK: tokenizer training (HF templates only prepare the token-byte cache). "
        "NTP: next token prediction (base pretraining). "
        "SFT: supervised fine-tuning (full answer prediction). "
        "RL: reinforcement learning (free thinking with final answer prediction)."
    ),
)
@click.pass_context
def train(ctx, model_name, train_stage):
    """Train a model."""
    script_name = f"train_{train_stage.lower()}.py"
    vendor, model_dir = model_name.split("/")
    module_path = gw_models.MODEL_BASE / vendor / model_dir / script_name
    package_name = f"pipe.{vendor}.{model_dir}.{Path(script_name).stem}"
    import_file(module_path, package_name)

    sys.modules[package_name].main(ctx.args)


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>")
@click.option("--load-path", metavar="PATH", help="Path to a training checkpoint")
def save(model_name, load_path):
    """Save a model checkpoint as safetensors."""
    save_path = gw_models.model_path(model_name)
    if load_path is None:
        load_path = save_path + "/chpt.pt"
    gw_models.save_model(model_name, save_path, load_path)


if __name__ == "__main__":
    main()