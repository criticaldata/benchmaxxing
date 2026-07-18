"""Command-line entry point for benchmaxxing.

Dependency-light: standard-library argparse only. Optional pieces (the datasets registry,
config resolution) are imported lazily inside each command so ``benchmaxxing --help`` and
``benchmaxxing version`` keep working even while the rest of the package is mid-build.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with the version, datasets, and config-show commands."""
    parser = argparse.ArgumentParser(
        prog="benchmaxxing",
        description=(
            "Referee agents for catching benchmark gaming in clinical multi-agent systems."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    sub.add_parser("version", help="print the installed benchmaxxing version")
    sub.add_parser("datasets", help="list the available dataset adapters")
    sub.add_parser("smoke", help="run the offline end-to-end pipeline smoke on synthetic data")

    p_config = sub.add_parser("config-show", help="print the resolved default config")
    p_config.add_argument(
        "-c",
        "--config",
        default=None,
        metavar="PATH",
        help="optional YAML config file to resolve and print (defaults are shown otherwise)",
    )
    return parser


def _cmd_version(_args: argparse.Namespace) -> int:
    import benchmaxxing

    print(benchmaxxing.__version__)
    return 0


def _dataset_names() -> list[str]:
    """Best-effort list of registered dataset names, empty if none or on any import error."""
    try:
        from benchmaxxing import datasets
    except Exception:  # noqa: BLE001 - datasets may be mid-build; degrade gracefully
        return []
    registry = getattr(datasets, "registry", None)
    if not registry:
        return []
    items = registry.keys() if isinstance(registry, Mapping) else registry
    names = []
    try:
        for item in items:
            label = getattr(item, "name", None)
            names.append(str(label) if label is not None else str(item))
    except TypeError:
        return []
    return sorted(names)


def _cmd_datasets(_args: argparse.Namespace) -> int:
    names = _dataset_names()
    if names:
        for name in names:
            print(name)
    else:
        print("No dataset adapters are registered yet.")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    from benchmaxxing.config import load_config

    config = load_config(getattr(args, "config", None))
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_smoke(_args: argparse.Namespace) -> int:
    from benchmaxxing.demo import format_report, run_smoke

    print(format_report(run_smoke()))
    return 0


_HANDLERS = {
    "version": _cmd_version,
    "datasets": _cmd_datasets,
    "config-show": _cmd_config_show,
    "smoke": _cmd_smoke,
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (or ``sys.argv``) and dispatch to a subcommand. Returns an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = _HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
