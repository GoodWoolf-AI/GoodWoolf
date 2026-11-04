#!/usr/bin/env python3
"""Entry point for GoodWoolf CLI commands."""

import click
import importlib.util
import sys
from pathlib import Path

import goodwoolf.utils.models as gw_models
import goodwoolf.utils.ledger as gw_ledger
from goodwoolf.utils.system import import_file, models_path

@click.group()
def main():
    """GoodWoolf builds foundation models on commodity hardware."""


@main.command()
def datasets():
    """List local training datasets."""
    if datasets:
        gw_models.list_datasets()


@main.command()
@click.option("--remote", required=False, metavar="REPO", help="Remote repo for vendors (supported: hf).")
def vendors(remote):
    """List local or remote model vendors."""
    if remote == "hf":
        gw_models.list_remote_vendors()
    else:
        gw_models.list_vendors()


@main.command()
@click.argument("vendor-name", required=True, help="Model vendor name.")
@click.option("--remote", required=False, metavar="REPO", help="Remote repo for vendors (supported: hf).")
def models(vendor_name, remote):
    """List local or remote models."""
    if remote == "hf":
        gw_models.list_remote_models(vendor_name)
    else:
        gw_models.list_models(vendor_name)


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
    "Pull the current model from the training ledger."
    try:
        gw_ledger.pull_model(model_name, id)
    except Exception as e:
        raise click.ClickException(str(e))


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--config", is_flag=True, default=False, help="Fetch model configuration to the config folder.")
@click.option(
    "--model",
    required=False,
    is_flag=False,
    flag_value="GoodWoolf/default",
    default=None,
    help="Create model training code from a template"
)
@click.option("--weights", is_flag=True, default=False, help="Create randomized model weights in the input folder")
def create(model_name, config, model, weights):
    """Create a new model config, code and initial weights."""
    if not config and model is None and not weights:
        raise click.UsageError(
            "You must specify at least one action option: "
            "--config, --model, or --weights."
        )    
    gw_models.create_model(model_name, config, model, weights)


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--load-path", metavar="PATH", help="Load weights from safetensors or checkpoint.")
@click.option("--prompt", metavar="PROMPT", default="", help="Model input prompt.")
@click.option("--max-tokens", metavar="TOKENS", type=int, default=100, help="Max number of tokens to generate.")
def run(model_name, load_path, prompt, max_tokens):
    """Run a model to generate text from stdin prompt."""
    if load_path is None:
        load_path = models_path(model_name)
    response = gw_models.generate(model_name, load_path, prompt, max_tokens)
    print(f"Output:\n{response}")


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>.")
@click.option("--data-name", required=True, metavar="DATASET", help="Dataset name (e.g. shakespeare).")
@click.option("--base-model", required=True, metavar="PATH", help="Path to base model safetensors checkpoint.")
@click.option("--new-model", required=True, metavar="PATH", help="Path to new model safetensors checkpoint.")
@click.option("--batch-size", type=int, default=8, help="Validation batch size.")
@click.option("--eval-iters", type=int, default=100, help="Validation iteration count.")
@click.option("--dtype", metavar="TYPE", default="bfloat16", help="Model runtime type.")
def update(model_name, data_name, base_model, new_model, batch_size, eval_iters, dtype):
    """Update the base model with delta weights from a new model."""
    gw_models.update(model_name, data_name, base_model, new_model, batch_size, eval_iters, dtype)


@main.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>")
@click.pass_context
def prepare(ctx, model_name, data_name):
    """Prepare data for training and validation."""
    module_path = gw_models.DATASET_BASE / data_name / "prepare.py"
    package_name = f"goodwoolf.data.{data_name}.prepare"
    import_file(str(module_path), package_name)

    argv = ["--model-name", model_name] + ctx.args
    sys.modules[package_name].main(argv)


@main.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>")
@click.pass_context
def pretrain(ctx, model_name):
    """Pretrain a model."""
    vendor, model_dir = model_name.split("/")
    module_path = gw_models.MODEL_BASE / vendor / model_dir / "pretrain.py"
    package_name = f"pipe.{vendor}.{model_dir}.pretrain"
    import_file(module_path, package_name)

    argv = ["--model-name", model_name] + ctx.args
    sys.modules[package_name].main(argv)


@main.command()
@click.argument("model-name", required=True, help="Model name in format <vendor>/<model>")
@click.option("--load-path", metavar="PATH", help="Path to a training checkpoint")
def save(model_name, load_path):
    """Save a model checkpoint as safetensors."""
    save_path = gw_models.models_path(model_name)
    if load_path is None:
        load_path = save_path + "/chpt.pt"
    gw_models.save_model(model_name, save_path, load_path)


if __name__ == "__main__":
    main()