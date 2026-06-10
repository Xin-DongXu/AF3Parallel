#!/usr/bin/env python3
"""
GPU memory usage monitor with idle detection and TSV logging.

Polls nvidia-smi at a fixed interval and writes one TSV row per sample.
Optionally exits automatically once all monitored GPUs stay below a memory
threshold for a sustained idle duration -- useful as a sidecar that wraps a
long-running GPU job and stops itself when the workload is done.

Python: >= 3.8
External requirements: NVIDIA driver providing nvidia-smi in PATH.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Optional


# ---------------------------------------------------------------------------
# nvidia-smi helpers
# ---------------------------------------------------------------------------

def _query_nvidia_smi() -> List[int]:
    """Return per-GPU memory.used in MiB. Exit on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
    except FileNotFoundError:
        sys.exit("Error: nvidia-smi not found. Is the NVIDIA driver installed?")
    except subprocess.TimeoutExpired:
        sys.exit("Error: nvidia-smi timed out (10 s).")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Error running nvidia-smi (exit {e.returncode}): "
                 f"{e.stderr.strip() or e.stdout.strip()}")

    out = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(int(line))
        except ValueError:
            sys.exit(f"Error: cannot parse nvidia-smi output line: {line!r}")
    if not out:
        sys.exit("Error: nvidia-smi returned no GPUs. Are any GPUs visible?")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_gpu_indices(arg: Optional[str], n_total: int) -> List[int]:
    """Parse the --gpus CLI value into a validated list of indices."""
    if arg is None:
        return list(range(n_total))

    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if not parts:
        sys.exit("Error: --gpus is empty.")

    indices: List[int] = []
    for p in parts:
        try:
            idx = int(p)
        except ValueError:
            sys.exit(f"Error: --gpus contains non-integer entry: {p!r}")
        if idx < 0:
            sys.exit(f"Error: GPU index must be non-negative, got {idx}.")
        if idx >= n_total:
            sys.exit(f"Error: GPU index {idx} out of range "
                     f"(found {n_total} GPUs, valid: 0..{n_total - 1}).")
        if idx in indices:
            # Allow duplicates only if user clearly intends to repeat;
            # warn instead of failing.
            print(f"Warning: GPU index {idx} listed more than once; "
                  f"keeping a single occurrence.", file=sys.stderr)
            continue
        indices.append(idx)
    return indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Monitor GPU memory usage and log to a TSV file. "
                     "Optionally stop when ALL monitored GPUs stay below a "
                     "threshold for a sustained period."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to the output TSV file.",
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=5.0,
        help="Sampling interval in seconds (must be > 0).",
    )
    parser.add_argument(
        "-t", "--threshold", type=int, default=100,
        help="Memory threshold in MiB. If ALL monitored GPUs stay strictly "
             "below this, the idle timer starts.",
    )
    parser.add_argument(
        "-d", "--duration", type=int, default=300,
        help="Idle duration in seconds. Script exits after ALL monitored GPUs "
             "remain below the threshold for this long. Use 0 to disable "
             "auto-exit and run until interrupted.",
    )
    parser.add_argument(
        "-g", "--gpus", type=str, default=None,
        help="Comma-separated GPU indices to monitor, e.g. '0,1,3'. "
             "Default: all visible GPUs.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to the output TSV instead of overwriting (no header is "
             "written if the file already exists and is non-empty).",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress per-sample stdout (TSV is still written).",
    )
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be > 0.")
    if args.threshold < 0:
        parser.error("--threshold must be >= 0.")
    if args.duration < 0:
        parser.error("--duration must be >= 0.")
    return args


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Discover GPUs and validate the requested subset.
    initial = _query_nvidia_smi()
    gpu_indices = _parse_gpu_indices(args.gpus, len(initial))
    n_gpus = len(gpu_indices)

    print(f"Monitoring {n_gpus} GPU(s): {gpu_indices}")
    auto_exit = "disabled" if args.duration == 0 else f"{args.duration} s"
    print(f"Interval: {args.interval} s | Threshold: {args.threshold} MiB | "
          f"Idle auto-exit: {auto_exit}")
    print(f"Output:   {args.output}  (mode={'append' if args.append else 'write'})")

    # Decide whether to write a TSV header.
    write_header = True
    open_mode = "w"
    if args.append:
        open_mode = "a"
        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            write_header = False

    idle_since: Optional[float] = None
    exit_code = 0

    try:
        with open(args.output, open_mode, encoding="utf-8") as f:
            if write_header:
                header = ["timestamp"] + [f"gpu{idx}_mem_MiB"
                                          for idx in gpu_indices]
                f.write("\t".join(header) + "\n")
                f.flush()

            while True:
                mem_all = _query_nvidia_smi()
                if len(mem_all) < len(initial):
                    # GPU went offline mid-run -- abort cleanly rather than
                    # silently misalign columns.
                    print(f"Error: nvidia-smi reports {len(mem_all)} GPUs now "
                          f"vs {len(initial)} at start; aborting.",
                          file=sys.stderr)
                    exit_code = 2
                    break

                mem = [mem_all[i] for i in gpu_indices]
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                f.write("\t".join([ts] + [str(m) for m in mem]) + "\n")
                f.flush()

                if not args.quiet:
                    cells = "  ".join(f"GPU{gpu_indices[i]}: {mem[i]} MiB"
                                      for i in range(n_gpus))
                    print(f"[{ts}]  {cells}")

                # Idle detection (only if auto-exit is enabled).
                if args.duration > 0:
                    all_below = all(m < args.threshold for m in mem)
                    if all_below:
                        if idle_since is None:
                            idle_since = time.monotonic()
                        elapsed = time.monotonic() - idle_since
                        if elapsed >= args.duration:
                            print(f"\nAll monitored GPUs below "
                                  f"{args.threshold} MiB for "
                                  f"{args.duration} s. Exiting.")
                            break
                        if not args.quiet:
                            print(f"  [idle] {elapsed:.0f} s / "
                                  f"{args.duration} s")
                    else:
                        if idle_since is not None and not args.quiet:
                            print("  [active] idle timer reset.")
                        idle_since = None

                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Data saved.", file=sys.stderr)

    print(f"Log written to {args.output}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
