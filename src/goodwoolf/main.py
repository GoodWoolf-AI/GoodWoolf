#!/usr/bin/env python3
"""Entry point for GoodWoolf CLI commands."""

import argparse
from pathlib import Path
import sys

import goodwoolf.utils.common as gw_common
import goodwoolf.utils.models as gw_models
import goodwoolf.utils.ledger as gw_ledger
import goodwoolf.utils.machine as gw_machine
from goodwoolf.utils.system import import_file, model_path

TRAIN_STAGES = ("TOK", "NTP", "SFT", "RL")


class CommandParser(argparse.ArgumentParser):
    def error(self, message):
        if message.startswith("the following arguments are required: "):
            self.print_help()
            raise SystemExit(0)
        super().error(message)


class PassthroughHelpAction(argparse.Action):
    def __init__(self, option_strings, dest, positional=None, **kwargs):
        self.positional = positional
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.positional, None) is None:
            parser.print_help()
            parser.exit()
        namespace.passthrough_help = option_string


# ============================================================
# ROOT
# ============================================================
def _error(message):
    print(f"gw: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _run_action(action, *args, **kwargs):
    try:
        action(*args, **kwargs)
    except Exception as e:
        _error(str(e))


def _load_module(path, package_name):
    import_file(str(path), package_name)
    return sys.modules[package_name]


def _dataset_path(data_name):
    vendor_name, dataset_name = data_name.split("/", 1)
    path = gw_models.DATASET_BASE / vendor_name / dataset_name / "dataset.py"
    return path if path.exists() else None


def _dataset_package_name(data_name):
    vendor_name, dataset_name = data_name.split("/", 1)
    return f"goodwoolf.data.{vendor_name}.{dataset_name}.dataset"


def _dataset_module(args):
    data_name = args.data_name
    if "/" not in data_name:
        _error(f"Invalid dataset name: {data_name}")
    module_path = _dataset_path(data_name)
    if module_path is None:
        _error(f"Unknown dataset: {data_name}")
    return _load_module(module_path, _dataset_package_name(data_name))


# ============================================================
# VENDOR
# ============================================================
def _vendor_list(args):
    if args.remote and args.remote.lower() == "hf":
        _run_action(gw_models.list_remote_vendors)
    else:
        _run_action(gw_models.list_vendors)


# ============================================================
# LEDGER > CREATE
# ============================================================
def _ledger_create(args):
    _run_action(gw_ledger.create_entry, args.model_name, operator=args.stage.upper())


# ============================================================
# LEDGER > CHECK
# ============================================================
def _ledger_check(args):
    _run_action(gw_ledger.check_entry, args.model_name)


# ============================================================
# PULL
# ============================================================
def _pull(args):
    _run_action(gw_ledger.pull_model, args.model_name, args.model_id)


# ============================================================
# CHECKOUT
# ============================================================
def _checkout(args):
    _run_action(gw_ledger.model_checkout, args.model_name, args.model_id)


# ============================================================
# MODEL > LIST
# ============================================================
def _model_list(args):
    if args.remote and args.remote.lower() == "hf":
        _run_action(gw_models.list_remote_models, args.vendor)
    elif args.vendor:
        if "/" in args.vendor:
            _error("Use a vendor name with `gw model list`, not <vendor>/<model>.")
        _run_action(gw_models.list_models, args.vendor)
    else:
        _run_action(gw_models.list_all_models)


# ============================================================
# MODEL > CONFIG
# ============================================================
def _model_config(args):
    _run_action(gw_models.create_model, args.model_name, config=True, template=None, weights=False)


# ============================================================
# MODEL > TEMPLATE
# ============================================================
def _model_template(args):
    _run_action(
        gw_models.create_model,
        args.model_name,
        config=False,
        template=args.template,
        weights=False,
    )


# ============================================================
# MODEL > WEIGHTS
# ============================================================
def _model_weights(args):
    _run_action(gw_models.create_model, args.model_name, config=False, template=None, weights=True)


# ============================================================
# RUN
# ============================================================
def _run(args):
    load_path = args.load_path or model_path(args.model_name)
    response = gw_models.generate(args.model_name, load_path, args.prompt, args.max_tokens)
    print(f"Output:\n{response}")


# ============================================================
# CHAT
# ============================================================
def _chat(args):
    _run_action(
        gw_common.chat,
        args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )


# ============================================================
# MERGE
# ============================================================
def _merge(args):
    _run_action(
        gw_models.merge,
        args.model_name,
        args.data_name,
        args.model_id,
        args.batch_size,
        args.eval_iters,
        args.dtype,
    )


# ============================================================
# DATASET > LIST
# ============================================================
def _dataset_list(args):
    _run_action(gw_models.list_datasets)


# ============================================================
# DATASET > FETCH
# ============================================================
def _dataset_fetch(args):
    module = _dataset_module(args)
    module.fetch()


# ============================================================
# DATASET > PREPARE
# ============================================================
def _dataset_prepare(args):
    module = _dataset_module(args)
    module.prepare()


# ============================================================
# DATASET > METADATA
# ============================================================
def _dataset_metadata(args):
    module = _dataset_module(args)
    split = args.split
    if split is not None and split not in module.SPLITS:
        _error(f"Invalid split: {split}")
    module.metadata(split)


# ============================================================
# DATASET > SAMPLE
# ============================================================
def _dataset_sample(args):
    module = _dataset_module(args)
    split = args.split
    if split is not None and split not in module.SPLITS:
        _error(f"Invalid split: {split}")
    module.sample(args.start, args.count, split=split)


# ============================================================
# TRAIN
# ============================================================
def _train(args):
    vendor, model_dir = args.model_name.split("/")
    script_name = f"train_{args.stage.lower()}.py"
    module_path = gw_models.MODEL_BASE / vendor / model_dir / script_name
    package_name = f"pipe.{vendor}.{model_dir}.{Path(script_name).stem}"
    trainer_args = list(getattr(args, "trainer_args", []))
    help_option = getattr(args, "passthrough_help", None)
    if help_option is not None:
        trainer_args.append(help_option)
    module = _load_module(module_path, package_name)
    module.main(trainer_args)


# ============================================================
# SAVE
# ============================================================
def _save(args):
    save_path = gw_models.model_path(args.model_name)
    load_path = args.load_path or save_path + "/chpt.pt"
    _run_action(gw_models.save_model, args.model_name, save_path, load_path)


# ============================================================
# MACHINE > CREATE
# ============================================================
def _machine_create(args):
    _run_action(
        gw_machine.machine_create,
        cpus=args.cpus,
        disk_size=args.disk_size,
        memory=args.memory,
        swap_size=args.swap_size,
    )


# ============================================================
# MACHINE > SHELL
# ============================================================
def _machine_shell(args):
    _run_action(gw_machine.machine_shell)


# ============================================================
# MACHINE > REMOVE
# ============================================================
def _machine_remove(args):
    _run_action(gw_machine.machine_remove)


# ============================================================
# CONTAINER > CREATE
# ============================================================
def _container_create(args):
    _run_action(
        gw_machine.container_create,
        cpus=args.cpus,
        disk_size=args.disk_size,
        memory=args.memory,
        swap_size=args.swap_size,
    )


# ============================================================
# CONTAINER > START
# ============================================================
def _container_start(args):
    _run_action(gw_machine.container_start)


# ============================================================
# CONTAINER > STOP
# ============================================================
def _container_stop(args):
    _run_action(gw_machine.container_stop)


# ============================================================
# CONTAINER > SHELL
# ============================================================
def _container_shell(args):
    _run_action(gw_machine.container_shell)


# ============================================================
# CONTAINER > REMOVE
# ============================================================
def _container_remove(args):
    _run_action(gw_machine.container_remove)


# ============================================================
# ROOT > PARSER
# ============================================================
def build_parser():
    # ROOT
    parser = CommandParser(
        prog="gw",
        description="GoodWoolf builds foundation models on commodity hardware.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
        parser_class=CommandParser,
    )

    # VENDOR
    vendor = commands.add_parser("vendor", help="Inspect model vendors.")
    vendor_commands = vendor.add_subparsers(
        dest="vendor_command",
        required=True,
        metavar="COMMAND",
        parser_class=CommandParser,
    )
    vendor_list = vendor_commands.add_parser("list", help="List model vendors.")
    vendor_list.add_argument(
        "--remote",
        metavar="REPO",
        help="Remote repo for vendors (supported: hf).",
    )
    vendor_list.set_defaults(func=_vendor_list)

    # LEDGER
    ledger = commands.add_parser("ledger", help="Update the model training ledger.")
    ledger_commands = ledger.add_subparsers(
        dest="ledger_command",
        required=True,
        metavar="COMMAND",
        parser_class=CommandParser,
    )

    ledger_create = ledger_commands.add_parser("create", help="Create a new ledger entry.")
    ledger_create.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    ledger_create.add_argument(
        "--stage",
        required=True,
        type=str.upper,
        choices=gw_ledger.LEDGER_OPERATORS,
        help="Stage-code operator: RND, NTP, SFT, RL, or AGG.",
    )
    ledger_create.set_defaults(func=_ledger_create)

    ledger_check = ledger_commands.add_parser(
        "check",
        help="Validate the ledger entry against the local model.",
    )
    ledger_check.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    ledger_check.set_defaults(func=_ledger_check)

    # PULL
    pull = commands.add_parser("pull", help="Pull a model from the training ledger.")
    pull.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    pull.add_argument("--model-id", help="The model ledger entry id.")
    pull.set_defaults(func=_pull)

    # CHECKOUT
    checkout = commands.add_parser("checkout", help="Check out a model checkpoint as the active model.")
    checkout.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    checkout.add_argument("--model-id", required=True, help="The model ledger entry id.")
    checkout.set_defaults(func=_checkout)

    # MODEL
    model = commands.add_parser("model", help="Manage models.")
    model_commands = model.add_subparsers(
        dest="model_command",
        required=True,
        metavar="COMMAND",
        parser_class=CommandParser,
    )

    model_list = model_commands.add_parser("list", help="List models.")
    model_list.add_argument("vendor", nargs="?", help="Vendor name or Hugging Face search string.")
    model_list.add_argument(
        "--remote",
        metavar="REPO",
        help="Remote repo for models (supported: hf).",
    )
    model_list.set_defaults(func=_model_list)

    model_config = model_commands.add_parser("config", help="Fetch model configuration to the model folder.")
    model_config.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    model_config.set_defaults(func=_model_config)

    model_template = model_commands.add_parser("template", help="Create model training code from a template.")
    model_template.add_argument("model_name", help="New model name in format <vendor>/<model>.")
    model_template.add_argument(
        "template",
        nargs="?",
        default="OpenAI-Community/GPT2",
        help="Template model name in format <vendor>/<model>.",
    )
    model_template.set_defaults(func=_model_template)

    model_weights = model_commands.add_parser("weights", help="Create randomized model weights in the input folder.")
    model_weights.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    model_weights.set_defaults(func=_model_weights)

    # RUN
    run = commands.add_parser("run", help="Run a model to generate text from stdin prompt.")
    run.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    run.add_argument("--load-path", metavar="PATH", help="Load weights from safetensors or checkpoint.")
    run.add_argument("--prompt", metavar="PROMPT", default="", help="Model input prompt.")
    run.add_argument("--max-tokens", metavar="TOKENS", type=int, default=100, help="Max number of tokens to generate.")
    run.set_defaults(func=_run)

    # CHAT
    chat = commands.add_parser("chat", help="Chat with a model.")
    chat.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    chat.add_argument("--max-tokens", metavar="TOKENS", type=int, default=256, help="Max number of tokens in the response.")
    chat.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    chat.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter.")
    chat.add_argument("--device", default="", help="Device to run on (cuda, cpu, or torch device string).")
    chat.set_defaults(func=_chat)

    # MERGE
    merge = commands.add_parser("merge", help="Merge a checkpoint into the active model.")
    merge.add_argument("model_name", help="Model name in format <vendor>/<model>.")
    merge.add_argument("--data-name", required=True, metavar="DATASET", help="Dataset name (e.g. openai/gsm8k).")
    merge.add_argument("--model-id", required=True, metavar="MODEL_ID", help="Model checkpoint id to merge.")
    merge.add_argument("--batch-size", type=int, default=8, help="Validation batch size.")
    merge.add_argument("--eval-iters", type=int, default=100, help="Validation iteration count.")
    merge.add_argument("--dtype", metavar="TYPE", default="bfloat16", help="Model runtime type.")
    merge.set_defaults(func=_merge)

    # DATASET
    dataset = commands.add_parser("dataset", help="Manage datasets.")
    dataset_commands = dataset.add_subparsers(
        dest="dataset_command",
        required=True,
        metavar="COMMAND",
        parser_class=CommandParser,
    )

    dataset_list = dataset_commands.add_parser("list", help="List training datasets.")
    dataset_list.set_defaults(func=_dataset_list)

    dataset_fetch = dataset_commands.add_parser(
        "fetch",
        help="Download the source dataset.",
        description="Download the source dataset.",
    )
    dataset_fetch.add_argument("data_name", metavar="DATA_NAME", help="Dataset name in format <vendor>/<dataset>.")
    dataset_fetch.set_defaults(func=_dataset_fetch)

    dataset_prepare = dataset_commands.add_parser(
        "prepare",
        help="Write the prepared dataset.",
        description="Write the prepared dataset.",
    )
    dataset_prepare.add_argument("data_name", metavar="DATA_NAME", help="Dataset name in format <vendor>/<dataset>.")
    dataset_prepare.set_defaults(func=_dataset_prepare)

    dataset_metadata = dataset_commands.add_parser(
        "metadata",
        help="Print dataset statistics and splits.",
        description="Print dataset statistics and splits.",
    )
    dataset_metadata.add_argument("data_name", metavar="DATA_NAME", help="Dataset name in format <vendor>/<dataset>.")
    dataset_metadata.add_argument("--split", help="Split to inspect (default: all).")
    dataset_metadata.set_defaults(func=_dataset_metadata)

    dataset_sample = dataset_commands.add_parser(
        "sample",
        help="Print COUNT samples starting at START.",
        description="Print COUNT samples starting at START.",
    )
    dataset_sample.add_argument("data_name", metavar="DATA_NAME", help="Dataset name in format <vendor>/<dataset>.")
    dataset_sample.add_argument("start", nargs="?", type=int, default=0, help="Start index.")
    dataset_sample.add_argument("count", nargs="?", type=int, default=1, help="Number of samples.")
    dataset_sample.add_argument("--split", help="Split to inspect (default: all).")
    dataset_sample.set_defaults(func=_dataset_sample)

    # TRAIN
    train = commands.add_parser(
        "train",
        help="Train a model.",
        add_help=False,
        usage=f"%(prog)s [-h] [--stage {{{','.join(TRAIN_STAGES)}}}] model_name [OPTIONS]",
    )
    train.add_argument("model_name", help="Model name in format <vendor>/<model>")
    train.add_argument(
        "--stage",
        type=str.upper,
        choices=TRAIN_STAGES,
        default="NTP",
        help=(
            "TOK: tokenizer training (HF templates only prepare the token-byte cache). "
            "NTP: next token prediction (base pretraining). "
            "SFT: supervised fine-tuning (full answer prediction). "
            "RL: reinforcement learning (free thinking with final answer prediction)."
        ),
    )
    train.add_argument(
        "-h",
        "--help",
        action=PassthroughHelpAction,
        positional="model_name",
        help="Show this help message and exit.",
    )
    train.set_defaults(func=_train, passthrough=True)

    # SAVE
    save = commands.add_parser("save", help="Save a model checkpoint as safetensors.")
    save.add_argument("model_name", help="Model name in format <vendor>/<model>")
    save.add_argument("--load-path", metavar="PATH", help="Path to a training checkpoint")
    save.set_defaults(func=_save)

    # MACHINE
    machine = commands.add_parser("machine", help="Manage a Podman machine.")
    machine_commands = machine.add_subparsers(
        dest="machine_command",
        required=True,
        metavar="COMMAND",
        parser_class=CommandParser,
    )

    machine_create = machine_commands.add_parser("create", help="Create and configure a Podman machine.")
    machine_create.add_argument(
        "--cpus",
        type=int,
        default=gw_machine.DEFAULT_CPUS,
        help=f"Machine CPU count (default: {gw_machine.DEFAULT_CPUS}).",
    )
    machine_create.add_argument(
        "--disk-size",
        type=int,
        default=gw_machine.DEFAULT_DISK_SIZE,
        help=f"Machine disk size in GiB (default: {gw_machine.DEFAULT_DISK_SIZE}G).",
    )
    machine_create.add_argument(
        "--memory",
        "--ram-size",
        dest="memory",
        type=int,
        default=gw_machine.DEFAULT_MEMORY,
        help=f"Machine memory in GiB (default: {gw_machine.DEFAULT_MEMORY}G).",
    )
    machine_create.add_argument(
        "--swap-size",
        type=int,
        default=gw_machine.DEFAULT_SWAP_SIZE,
        help=f"Machine swap in GiB (default: {gw_machine.DEFAULT_SWAP_SIZE}G).",
    )
    machine_create.set_defaults(func=_machine_create)

    machine_shell = machine_commands.add_parser("shell", help="Open a shell in a Podman machine.")
    machine_shell.set_defaults(func=_machine_shell)

    machine_remove = machine_commands.add_parser("remove", help="Remove a Podman machine.")
    machine_remove.set_defaults(func=_machine_remove)

    # CONTAINER
    container = commands.add_parser("container", help="Manage a Podman container.")
    container_commands = container.add_subparsers(
        dest="container_command",
        required=True,
        metavar="COMMAND",
        parser_class=CommandParser,
    )

    container_create = container_commands.add_parser("create", help="Create a Podman container.")
    container_create.add_argument(
        "--cpus",
        type=int,
        default=gw_machine.DEFAULT_CPUS,
        help=f"Container CPU count (default: {gw_machine.DEFAULT_CPUS}).",
    )
    container_create.add_argument(
        "--disk-size",
        type=int,
        default=gw_machine.DEFAULT_DISK_SIZE,
        help=f"Container disk size in GiB (default: {gw_machine.DEFAULT_DISK_SIZE}G).",
    )
    container_create.add_argument(
        "--memory",
        "--ram-size",
        dest="memory",
        type=int,
        default=gw_machine.DEFAULT_MEMORY,
        help=f"Container memory in GiB (default: {gw_machine.DEFAULT_MEMORY}G).",
    )
    container_create.add_argument(
        "--swap-size",
        type=int,
        default=gw_machine.DEFAULT_SWAP_SIZE,
        help=f"Container swap in GiB (default: {gw_machine.DEFAULT_SWAP_SIZE}G).",
    )
    container_create.set_defaults(func=_container_create)

    for name, description, handler in (
        ("start", "Start a Podman container.", _container_start),
        ("stop", "Stop a Podman container.", _container_stop),
        ("shell", "Open an interactive shell in a Podman container.", _container_shell),
        ("remove", "Remove a Podman container.", _container_remove),
    ):
        subparser = container_commands.add_parser(name, help=description)
        subparser.set_defaults(func=handler)

    return parser


# ============================================================
# ROOT
# ============================================================
def main(argv=None):
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if remaining and not getattr(args, "passthrough", False):
        parser.error("unrecognized arguments: " + " ".join(remaining))

    if getattr(args, "passthrough", False):
        args.trainer_args = remaining

    handler = getattr(args, "func", None)
    if handler is not None:
        handler(args)


if __name__ == "__main__":
    main()
