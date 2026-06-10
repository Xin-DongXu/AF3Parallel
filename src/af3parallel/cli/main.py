"""Unified CLI dispatcher for all AF3Parallel tools."""

from __future__ import annotations

import importlib
import signal
import sys
from typing import Callable, Dict, Optional

from af3parallel.__version__ import __version__

# Maps subcommand name -> (module path, entry function name, needs_signal_handlers)
_COMMANDS: Dict[str, tuple[str, str, bool]] = {
    "run": ("af3parallel.parallel", "main", True),
    "profile": ("af3parallel.gpu_memory_profiler", "main", False),
    "profile-ts": ("af3parallel.gpu_memory_timeseries_profiler", "main", False),
    "estimate-gpu": ("af3parallel.gpu_time_estimate", "main", False),
    "estimate-cpu": ("af3parallel.cpu_time_estimate", "main", False),
    "monitor": ("af3parallel.gpu_monitor", "main", False),
}

_USAGE = f"""\
AF3Parallel v{__version__} - AlphaFold 3 multi-GPU toolkit

Usage:
  af3parallel <command> [arguments]

Commands:
  run            Multi-GPU batch executor
  profile        Peak VRAM profiler
  profile-ts     Time-series VRAM profiler
  estimate-gpu   Serial GPU runtime estimator
  estimate-cpu   CPU/MSA runtime estimator
  json           Input JSON integrator (set-seeds, add-ligand, ...)
  monitor        Standalone GPU memory monitor

Examples:
  af3parallel run -i ./af_input -o results.tsv --sif alphafold3.sif
  af3parallel profile -i ./profile_inputs -o profile.tsv --sif alphafold3.sif
  af3parallel json set-seeds -i input.json -o output.json --seeds 1 2 3

Run 'af3parallel <command> --help' for command-specific options.
"""


def _print_usage() -> None:
    sys.stdout.write(_USAGE)


def _load_entry(module_path: str, func_name: str) -> Callable:
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _install_parallel_signal_handlers() -> None:
    parallel = importlib.import_module("af3parallel.parallel")
    signal.signal(signal.SIGINT, parallel.signal_handler)
    signal.signal(signal.SIGTERM, parallel.signal_handler)


def main(argv: Optional[list[str]] = None) -> int:
    """Dispatch to the selected AF3Parallel subcommand."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        _print_usage()
        return 0

    if args[0] in ("-V", "--version"):
        sys.stdout.write(f"af3parallel {__version__}\n")
        return 0

    command = args[0]
    rest = args[1:]

    if command == "json":
        sys.argv = ["af3parallel-json"] + rest
        entry = _load_entry("af3parallel.json_integrator", "main")
        result = entry()
        return int(result) if result is not None else 0

    if command not in _COMMANDS:
        sys.stderr.write(f"af3parallel: unknown command {command!r}\n\n")
        _print_usage()
        return 2

    module_path, func_name, needs_signals = _COMMANDS[command]
    sys.argv = [f"af3parallel-{command}"] + rest

    if needs_signals:
        _install_parallel_signal_handlers()

    entry = _load_entry(module_path, func_name)
    result = entry()
    return int(result) if result is not None else 0
