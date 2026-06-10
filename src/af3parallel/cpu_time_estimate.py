#!/usr/bin/env python3
"""
af3_CPU_time_estimate.py
================================================================================
AlphaFold 3 CPU / MSA runtime estimator.

This tool answers ONE question:

    "If I ran the MSA (data-pipeline) step for every AF3 JSON in <input-dir>
     serially on a single CPU node, how long would it take in total?"

Background
----------
AlphaFold 3 splits work into two stages:

  1. CPU stage  - MSA search + template search (jackhmmer, hhblits, nhmmer ...)
                  Runtime is dominated by protein chain length, NOT total token
                  count, because database search time scales with amino-acid
                  residues.

  2. GPU stage  - Structure inference (neural network forward passes)
                  Runtime scales with total token count.

This script handles Stage 1.  For Stage 2, see AF3_estimate_GPU_time.py.

Profile format (--profile)
--------------------------
Tab-separated file produced by af3_CPU_Test_stat.tsv (or equivalent).
Required columns:
    protein_length      integer - longest protein chain length in the job
    total_runtime_s     float   - measured wall-clock MSA runtime (seconds)

Optional columns (loaded but not strictly required):
    peak_ram_mb         integer - peak RAM usage (MB, informational only)
    success             True/False - rows with success=False are skipped
                        unless --include-failed is set.

Any extra columns (json_file, protein_name, token_count, step_*, ...) are
silently ignored.

Key difference vs GPU estimator
--------------------------------
The GPU estimator uses total token count (all chains + ligand heavy atoms).
The CPU estimator uses longest protein chain length because MSA runtime is
driven by the largest protein sequence submitted to jackhmmer / hhblits.
Ligands, DNA, RNA, and stoichiometry copies do NOT add meaningful MSA cost.

Protein length extraction from AF3 JSON
----------------------------------------
Both AF3 JSON dialects are supported:

  Type-based  : {"type": "protein", "sequence": "MAST...", "count": 2}
  Key-based   : {"protein": {"sequence": "MAST..."}, "count": 2}

The longest protein sequence across all protein entities in the file is used
as the lookup key into the profile.

Lookup model: linear interpolation (default) or step lookup
-------------------------------------------------------------
Default behaviour is linear interpolation between the two nearest measured
data points.  Pass --no-gap-fill to use step lookup instead - i.e. return
the runtime of the largest measured length that is <= the query length.
Linear interpolation is usually the better choice for smooth runtime curves;
step lookup is appropriate when measurements are very dense or when you
explicitly do not want to extrapolate between measured ranges.

In both modes:
  - Below the minimum measured length -> first measured runtime.
  - Above the maximum measured length -> last measured runtime
    (conservative; exposed via --timeout-runtime is NOT applied to lengths
    inside the profile range, only documented as a hard fallback).

Publication audit (release-ready):
==========================================================================
  - macOS multiprocessing now uses 'spawn' (fork is unsafe on Darwin since
    Python 3.8).  'fork' is restricted to Linux; macOS and Windows use
    'spawn'.
  - Elapsed-time measurements switched from time.time() to time.monotonic()
    so durations are immune to NTP step adjustments and wall-clock changes.
  - --no-gap-fill is now actually honoured.  Previously the flag was
    parsed and passed to the loader but the loader ignored it (always
    used linear interpolation).  The flag now selects step lookup.
  - Docstring filename and argparse prog= aligned with the actual
    filename (af3_CPU_time_estimate.py).

Outputs
-------
  stderr   : human-readable progress, warnings, and summary table.
  stdout   : total estimated MSA runtime in seconds (machine-readable).
  TSV      : optional per-file breakdown  (--output-tsv).

Usage
-----
    python af3_CPU_time_estimate.py \\
        --input-dir   /path/to/af3_jsons \\
        --profile     af3_CPU_Test_stat.tsv \\
        --output-tsv  msa_breakdown.tsv \\
        --workers     16

    # Recursive scan + custom pattern
    python af3_CPU_time_estimate.py \\
        --input-dir   /data/jobs \\
        --profile     af3_CPU_Test_stat.tsv \\
        --recursive \\
        --pattern     "*.json" \\
        --output-tsv  msa_estimate.tsv
"""

# -----------------------------------------------------------------------------
#  Imports
# -----------------------------------------------------------------------------
import argparse
import bisect
import csv
import json
import multiprocessing
import os
import platform as _platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
#  Tiny logging helpers  (stderr, flush-safe)
# -----------------------------------------------------------------------------
def info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


def warning(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def error_log(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
#  Protein length extraction from AF3 JSON
# -----------------------------------------------------------------------------
def extract_max_protein_length(data: Dict) -> int:
    """Return the length of the longest protein chain in an AF3 JSON dict.

    MSA runtime is dominated by the longest protein sequence submitted to
    jackhmmer / hhblits.  Ligands, DNA, RNA, and stoichiometric copies do
    not add MSA search cost and are therefore ignored here.

    Supports both AF3 JSON dialects:
      - Type-based  : {"type": "protein", "sequence": "MAST...", "count": N}
      - Key-based   : {"protein": {"sequence": "MAST..."}, "count": N}

    Returns at least 1 (guards against empty / ligand-only files).
    """
    max_len = 0
    for entry in data.get("sequences", []):
        seq_type = entry.get("type", "").lower()

        if seq_type == "protein":
            # Type-based dialect
            seq = entry.get("sequence", "")
            max_len = max(max_len, len(seq))

        elif "protein" in entry:
            # Key-based dialect
            seq = entry["protein"].get("sequence", "")
            max_len = max(max_len, len(seq))

        # RNA, DNA, ligand entities are intentionally skipped.

    return max(1, max_len)


# -----------------------------------------------------------------------------
#  CPU MSA Runtime Profile
# -----------------------------------------------------------------------------

class CpuRuntimeProfileLoader:
    """Load a CPU MSA runtime profile TSV and answer
    "how long does a protein of length L take for MSA?".

    Algorithm
    ---------
    Unlike the GPU estimator (which groups by VRAM tier), CPU MSA runtime
    is NOT well-modelled by memory-based grouping.  Peak RAM for MSA varies
    due to OS paging and jackhmmer internal buffering, not because of a
    discrete capacity boundary.  Instead this loader uses:

      1. Parse TSV rows -> (protein_length, runtime_seconds) pairs.
         Duplicate lengths -> keep longest (worst-case) runtime.
      2. Sort measurements by protein_length.
      3. Lookup by linear interpolation between the two nearest measured
         points.  For lengths below the minimum, use the first measured
         runtime.  For lengths above the maximum, use the last measured
         runtime (conservative extrapolation).

    This approach avoids spurious WARN messages caused by RAM oscillations
    that are normal for MSA workloads and produces smooth estimates across
    the full length range.

    Required TSV columns
    ---------------------
      protein_length    integer  - longest protein chain length in the job
      total_runtime_s   float    - measured wall-clock MSA runtime (seconds)

    Optional columns (loaded but not used in estimation)
    ------------------------------------------------------
      peak_ram_mb       integer  - peak RAM usage (informational)
      success           True/False  - rows with False are skipped unless
                        --include-failed is set
    """

    REQUIRED_COLUMNS = {"protein_length", "total_runtime_s"}

    def __init__(
        self,
        profile_file: Path,
        gap_fill: bool = True,           # True = linear interpolation, False = step lookup
        include_failed: bool = False,
        timeout_runtime: float = 86400.0,   # 24 h default for MSA
    ):
        self.profile_file    = profile_file
        self.gap_fill        = gap_fill        # publication audit: now actually used
        self.include_failed  = include_failed
        self.timeout_runtime = timeout_runtime

        # Sorted parallel arrays for bisect + linear interpolation
        self._lens:     List[int]   = []   # protein lengths (sorted ascending)
        self._runtimes: List[float] = []   # corresponding measured runtimes

        self._load()

        if not self._lens:
            raise RuntimeError(
                f"Profile file produced no usable rows: {profile_file}"
            )

        info(
            f"Profile ready: {len(self._lens)} measured points, "
            f"length range [{self._lens[0]}, {self._lens[-1]}] aa, "
            f"lookup mode = {'linear interpolation' if gap_fill else 'step lookup'}"
        )

    # --- TSV parsing ------------------------------------------------------
    def _load(self) -> None:
        profiles: Dict[int, float] = {}   # protein_length -> max runtime_s

        with open(self.profile_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise RuntimeError("Profile file is empty (no header row).")
            stripped_fields = {c.strip() for c in reader.fieldnames}
            missing = self.REQUIRED_COLUMNS - stripped_fields
            if missing:
                raise RuntimeError(
                    f"Profile file missing required column(s): {sorted(missing)}. "
                    f"Found: {[c.strip() for c in reader.fieldnames]}"
                )

            n_loaded = n_skipped = 0
            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items() if k}
                try:
                    plen    = int(float(row["protein_length"]))
                    runtime = float(row["total_runtime_s"])
                    success = row.get("success", "True").lower() == "true"
                except (ValueError, KeyError):
                    continue

                if not success and not self.include_failed:
                    n_skipped += 1
                    continue

                # Keep worst-case runtime for duplicate lengths
                if plen not in profiles or runtime > profiles[plen]:
                    profiles[plen] = runtime
                n_loaded += 1

        info(
            f"Loaded {n_loaded} rows from {self.profile_file.name} "
            f"({len(profiles)} unique protein lengths"
            + (f", skipped {n_skipped} failed rows" if n_skipped else "")
            + ")"
        )

        # Build sorted parallel arrays
        for plen in sorted(profiles.keys()):
            self._lens.append(plen)
            self._runtimes.append(profiles[plen])

    # --- Lookup ----------------------------------------------------------
    def estimate_runtime_seconds(self, protein_length: int) -> float:
        """Return MSA runtime estimate for the given protein length.

        Default mode (gap_fill=True): linear interpolation between the two
        nearest measured points.

        Step-lookup mode (gap_fill=False): return the runtime of the
        largest measured length that is <= the query length.  This is what
        the user gets by passing --no-gap-fill.

        In both modes:
          - Below the minimum measured length -> first measured runtime.
          - Above the maximum measured length -> last measured runtime.
        """
        if not self._lens:
            return self.timeout_runtime

        if protein_length <= self._lens[0]:
            return self._runtimes[0]

        if protein_length >= self._lens[-1]:
            return self._runtimes[-1]

        # Locate the bracketing pair: lens[idx-1] < protein_length <= lens[idx]
        idx = bisect.bisect_right(self._lens, protein_length)
        lo_len, lo_rt = self._lens[idx - 1], self._runtimes[idx - 1]
        hi_len, hi_rt = self._lens[idx],     self._runtimes[idx]

        if not self.gap_fill:
            # Step lookup: return the lower-neighbour runtime as-is.
            return lo_rt

        # Linear interpolation between (lo_len, lo_rt) and (hi_len, hi_rt).
        frac = (protein_length - lo_len) / (hi_len - lo_len)
        return lo_rt + frac * (hi_rt - lo_rt)

    @property
    def max_measured_length(self) -> int:
        """Largest protein length covered by the profile."""
        return self._lens[-1] if self._lens else 0

    @property
    def n_steps(self) -> int:
        """Number of measured data points (for summary reporting)."""
        return len(self._lens)


# -----------------------------------------------------------------------------
#  Parallel JSON parsing worker + driver
# -----------------------------------------------------------------------------
def _parse_json_worker(
    json_path_str: str,
) -> Tuple[str, Optional[int], Optional[str]]:
    """Parse one AF3 JSON and return its max protein length.

    Kept at module top-level so it is picklable by multiprocessing on
    'spawn' platforms (Windows / macOS default).

    Returns (path_str, max_protein_length_or_None, error_or_None).
    """
    try:
        with open(json_path_str, "r", encoding="utf-8") as f:
            data = json.load(f)
        plen = extract_max_protein_length(data)
        return (json_path_str, plen, None)
    except Exception as exc:
        return (json_path_str, None, f"{type(exc).__name__}: {exc}")


def parse_protein_lengths_batch(
    json_files: List[Path],
    workers: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, str]]]:
    """Parse protein lengths for a batch of AF3 JSON files in parallel.

    Strategy mirrors count_tokens_batch() in AF3_estimate_GPU_time.py:
      - 'fork' on Linux/macOS; 'spawn' on Windows.
      - imap_unordered + chunksize for load balancing.
      - Single-threaded fast path for small batches (<= 10 files).

    Returns
    -------
    ok_rows  : [(path, max_protein_length), ...]
    err_rows : [(path, error_message), ...]
    """
    n_files = len(json_files)
    if n_files == 0:
        return [], []

    # -- Decide worker count ----------------------------------------------
    try:
        usable_cpus = len(os.sched_getaffinity(0))   # Linux cgroup-aware
    except (AttributeError, OSError):
        usable_cpus = multiprocessing.cpu_count()
    mp_cpu = multiprocessing.cpu_count()

    user_specified = workers is not None and workers > 0
    if not user_specified:
        workers = max(1, min(usable_cpus, n_files))
    else:
        workers = max(1, min(workers, n_files))

    if usable_cpus != mp_cpu:
        info(
            f"CPU affinity reports {usable_cpus}/{mp_cpu} cores "
            "available (cgroup/taskset/SLURM quota)"
        )
        if not user_specified and usable_cpus == 1 and n_files > 10:
            warning(
                "Auto-detected 1 usable CPU, falling back to single-threaded. "
                f"If more cores are available, override with --workers N "
                f"(e.g. --workers {mp_cpu})."
            )
    info(
        f"JSON parsing will use {workers} worker process(es)"
        + (" (user-specified)" if user_specified else " (auto)")
    )

    # -- Fast path for small batches --------------------------------------
    if n_files <= 10 or workers == 1:
        reason = "small file count" if n_files <= 10 else "workers=1"
        info(f"Using single-threaded parsing ({reason}, {n_files} file(s))")
        ok_rows:  List[Tuple[Path, int]] = []
        err_rows: List[Tuple[Path, str]] = []
        for jf in json_files:
            _, plen, err = _parse_json_worker(str(jf))
            if plen is None:
                err_rows.append((jf, err or "unknown error"))
            else:
                ok_rows.append((jf, plen))
        return ok_rows, err_rows

    # -- Parallel path ----------------------------------------------------
    # Publication audit: macOS now uses 'spawn' (fork is unsafe on Darwin
    # since Python 3.8 due to libdispatch / Objective-C runtime).
    preferred_ctx = "fork" if _platform.system() == "Linux" else "spawn"
    try:
        ctx = multiprocessing.get_context(preferred_ctx)
    except ValueError:
        ctx = multiprocessing.get_context("spawn")
        preferred_ctx = "spawn"

    chunksize = max(1, min(10, n_files // (workers * 4) or 1))
    info(
        f"Parallel parsing: {workers} worker(s), "
        f"start_method={preferred_ctx}, chunksize={chunksize}"
    )

    json_path_strs = [str(jf) for jf in json_files]
    ok_rows  = []
    err_rows = []
    completed = 0
    # Publication audit: monotonic clock (immune to NTP step adjustments)
    start = time.monotonic()
    last_pct = -1

    info("Starting worker pool...")
    with ctx.Pool(processes=workers) as pool:
        info(f"Pool ready, dispatching {n_files} file(s)")
        for path_str, plen, err in pool.imap_unordered(
            _parse_json_worker, json_path_strs, chunksize=chunksize
        ):
            completed += 1
            jf = Path(path_str)
            if plen is None:
                err_rows.append((jf, err or "unknown error"))
            else:
                ok_rows.append((jf, plen))

            if show_progress:
                pct = int(completed / n_files * 100)
                if pct >= last_pct + 2 or completed % 50 == 0 or completed == n_files:
                    elapsed = time.monotonic() - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta  = (n_files - completed) / rate if rate > 0 else 0.0
                    print(
                        f"\r  Parsing: {completed}/{n_files} ({pct:3d}%) | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"Speed: {rate:.1f} files/s | ETA: {eta:.1f}s   ",
                        end="", flush=True, file=sys.stderr,
                    )
                    last_pct = pct

    if show_progress:
        elapsed = time.monotonic() - start
        rate    = n_files / elapsed if elapsed > 0 else 0.0
        print(
            f"\r  Parsing: {n_files}/{n_files} (100%) | "
            f"Time: {elapsed:.1f}s | Avg speed: {rate:.1f} files/s          ",
            file=sys.stderr,
        )

    return ok_rows, err_rows


# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------
def human_time(seconds: float) -> str:
    """Convert seconds to a human-readable string like '2d 3h 15m 42s'."""
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts: List[str] = []
    if d:           parts.append(f"{d}d")
    if h or d:      parts.append(f"{h}h")
    if m or h or d: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def percentile(sorted_vals: List[float], pct: float) -> float:
    """Return the p-th percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * pct / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="af3_CPU_time_estimate.py",
        description=(
            "Estimate total serial CPU/MSA runtime for a batch of AlphaFold 3 "
            "JSON files, using a protein-length -> runtime profile measured on "
            "your CPU hardware.  Reads JSON files in parallel; produces a "
            "human-readable summary on stderr and a machine-readable total on stdout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python af3_CPU_time_estimate.py \\\n"
            "      --input-dir  /data/af3_jobs \\\n"
            "      --profile    af3_CPU_Test_stat.tsv \\\n"
            "      --output-tsv msa_estimate.tsv \\\n"
            "      --workers    32\n\n"
            "  # Recursive scan + verbose per-file output\n"
            "  python af3_CPU_time_estimate.py \\\n"
            "      -i /data/jobs -p cpu_profile.tsv -r -v\n"
        ),
    )

    # -- Required / primary arguments --------------------------------------
    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help=(
            "Directory containing AF3 input *.json files (relative or absolute "
            "path).  Use --recursive to also search sub-directories."
        ),
    )
    parser.add_argument(
        "-p", "--profile",
        required=True,
        type=Path,
        metavar="TSV",
        help=(
            "CPU runtime profile TSV file.  Must contain columns: "
            "protein_length, total_runtime_s, peak_ram_mb.  "
            "Produced by af3_CPU_Test_stat or a similar benchmarking tool.  "
            "Example: af3_CPU_Test_stat.tsv"
        ),
    )

    # -- Output arguments --------------------------------------------------
    parser.add_argument(
        "-o", "--output-tsv",
        type=Path,
        default=None,
        metavar="TSV",
        help=(
            "Optional output TSV file with per-file breakdown.  "
            "Columns: json_file, max_protein_length, estimated_msa_runtime_s.  "
            "Directory is created automatically if it does not exist."
        ),
    )

    # -- Scan behaviour ----------------------------------------------------
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help=(
            "Recursively scan --input-dir and all sub-directories for JSON files. "
            "Without this flag only the top-level directory is scanned."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.json",
        metavar="GLOB",
        help=(
            "Glob pattern used to identify AF3 JSON files inside --input-dir.  "
            "Change if your files have a different extension or naming convention."
        ),
    )

    # -- Profile behaviour -------------------------------------------------
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help=(
            "Use step lookup instead of linear interpolation between "
            "measured protein lengths.  In step mode, queries inside the "
            "profile range return the runtime of the largest measured "
            "length that is <= the query length.  Default is linear "
            "interpolation, which produces smoother estimates for "
            "sparsely-measured ranges."
        ),
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help=(
            "Include profile rows where success=False when building the step "
            "function.  Useful to capture timeout runtimes for very long proteins "
            "that exceeded the measurement wall-clock limit."
        ),
    )
    parser.add_argument(
        "--timeout-runtime",
        type=float,
        default=86400.0,
        metavar="SECONDS",
        help=(
            "Fallback runtime (seconds) applied to protein lengths that exceed "
            "the maximum measured value in the profile.  Default: 86400 (24 h)."
        ),
    )

    # -- Filtering thresholds ----------------------------------------------
    parser.add_argument(
        "--max-protein-length",
        type=int,
        default=None,
        metavar="AA",
        help=(
            "Skip any JSON whose longest protein chain exceeds this length "
            "(amino acids).  Useful to exclude anomalously slow jobs (e.g. "
            "proteins > 1350 aa that trigger jackhmmer memory spikes) from the "
            "total estimate.  Skipped files appear in the summary and are written "
            "to --output-tsv with status=skipped_length.  Default: no limit."
        ),
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Skip any job whose *estimated* MSA runtime exceeds this value "
            "(seconds).  Applied after protein-length interpolation, so it "
            "filters on the estimated value rather than a raw measurement.  "
            "Skipped files appear in the summary and are written to --output-tsv "
            "with status=skipped_runtime.  "
            "Example: --max-runtime 3600 excludes jobs estimated > 1 h.  "
            "Default: no limit."
        ),
    )

    # -- Parallelism -------------------------------------------------------
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of parallel worker processes used to parse JSON files.  "
            "0 or negative = auto (use all CPU cores available to this process, "
            "respecting cgroup / SLURM affinity).  "
            "1 = force single-threaded (useful for debugging).  "
            "Explicit positive values override CPU affinity detection."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress the live progress line printed to stderr during parsing.",
    )

    # -- Verbosity ---------------------------------------------------------
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Print per-file estimation details (protein length + estimated runtime) "
            "to stderr.  Also print warnings for files outside the profile range."
        ),
    )

    args = parser.parse_args()

    # -- Validate inputs --------------------------------------------------
    if not args.input_dir.is_dir():
        error_log(f"--input-dir is not a directory: {args.input_dir}")
        return 2
    if not args.profile.is_file():
        error_log(f"--profile file not found: {args.profile}")
        return 2

    # -- Load CPU runtime profile -----------------------------------------
    info(f"Loading CPU MSA runtime profile: {args.profile}")
    try:
        profile = CpuRuntimeProfileLoader(
            profile_file=args.profile,
            gap_fill=not args.no_gap_fill,
            include_failed=args.include_failed,
            timeout_runtime=args.timeout_runtime,
        )
    except RuntimeError as exc:
        error_log(str(exc))
        return 2

    info(
        f"Profile has {profile.n_steps} measured point(s); "
        f"lookup mode: {'step (largest measured <= query)' if args.no_gap_fill else 'linear interpolation'}; "
        f"max_measured_length={profile.max_measured_length}"
    )

    # -- Enumerate JSON files ---------------------------------------------
    info(f"Scanning {'recursively ' if args.recursive else ''}for '{args.pattern}' "
         f"under: {args.input_dir}")
    if args.recursive:
        json_files = sorted(args.input_dir.rglob(args.pattern))
    else:
        json_files = sorted(args.input_dir.glob(args.pattern))

    if not json_files:
        error_log(
            f"No files matching '{args.pattern}' found under {args.input_dir}. "
            "Use --recursive to search sub-directories."
        )
        return 1

    info(f"Found {len(json_files):,} JSON file(s)")

    # -- Parse all JSON files (parallel) ---------------------------------
    # Publication audit: monotonic clock (immune to NTP step adjustments)
    parse_start = time.monotonic()
    ok_rows, err_rows = parse_protein_lengths_batch(
        json_files,
        workers=args.workers if args.workers and args.workers > 0 else None,
        show_progress=not args.no_progress,
    )
    parse_elapsed = time.monotonic() - parse_start

    if err_rows:
        warning(f"{len(err_rows)} file(s) failed to parse (see --verbose for details)")
    if args.verbose:
        for jf, msg in err_rows:
            warning(f"  Parse error: {jf.name}: {msg}")

    # Sort by filename for reproducible TSV output
    ok_rows.sort(key=lambda r: r[0].name)

    # -- Estimate MSA runtime per file + apply thresholds ---------------
    # Each row: (filename, plen, estimated_s, status)
    # status: "ok" | "skipped_length" | "skipped_runtime"
    all_rows: List[Tuple[str, int, float, str]] = []
    total_runtime  = 0.0
    n_beyond_profile = 0

    for jf, plen in ok_rows:
        if plen > profile.max_measured_length:
            n_beyond_profile += 1
            if args.verbose:
                warning(
                    f"  {jf.name}: protein_length={plen} exceeds profile max "
                    f"({profile.max_measured_length}) - using last-step runtime."
                )

        # -- Threshold 1: protein length ----------------------------------
        if args.max_protein_length is not None and plen > args.max_protein_length:
            all_rows.append((jf.name, plen, 0.0, "skipped_length"))
            if args.verbose:
                print(
                    f"[skip] {jf.name}  protein_length={plen} "
                    f"> --max-protein-length={args.max_protein_length}",
                    file=sys.stderr,
                )
            continue

        rt = profile.estimate_runtime_seconds(plen)

        # -- Threshold 2: estimated runtime -------------------------------
        if args.max_runtime is not None and rt > args.max_runtime:
            all_rows.append((jf.name, plen, rt, "skipped_runtime"))
            if args.verbose:
                print(
                    f"[skip] {jf.name}  protein_length={plen}  "
                    f"est_msa_runtime={rt:.2f}s "
                    f"> --max-runtime={args.max_runtime}s",
                    file=sys.stderr,
                )
            continue

        # -- Accepted -----------------------------------------------------
        total_runtime += rt
        all_rows.append((jf.name, plen, rt, "ok"))
        if args.verbose:
            print(
                f"[file] {jf.name}  protein_length={plen}  "
                f"est_msa_runtime={rt:.2f}s",
                file=sys.stderr,
            )

    info(f"Parsed {len(ok_rows):,} file(s) in {parse_elapsed:.2f}s")

    # Separate accepted vs skipped rows for reporting
    rows_ok      = [(n, p, r) for n, p, r, s in all_rows if s == "ok"]
    rows_sk_len  = [(n, p, r) for n, p, r, s in all_rows if s == "skipped_length"]
    rows_sk_rt   = [(n, p, r) for n, p, r, s in all_rows if s == "skipped_runtime"]

    # -- Optional per-file breakdown TSV ---------------------------------
    if args.output_tsv is not None and all_rows:
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_tsv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow([
                "json_file", "max_protein_length",
                "estimated_msa_runtime_s", "status",
            ])
            for name, plen, rt, status in all_rows:
                w.writerow([name, plen, f"{rt:.3f}" if rt else "0.000", status])
        info(f"Per-file breakdown written to {args.output_tsv}")

    # -- Summary ----------------------------------------------------------
    n_ok  = len(rows_ok)
    n_err = len(err_rows)
    n_sk_len = len(rows_sk_len)
    n_sk_rt  = len(rows_sk_rt)

    plen_list = sorted(r[1] for r in rows_ok)
    rt_list   = sorted(r[2] for r in rows_ok)

    print("", file=sys.stderr)
    print("=" * 66, file=sys.stderr)
    print(f"  AF3 CPU MSA Runtime Estimate", file=sys.stderr)
    print("-" * 66, file=sys.stderr)
    print(f"  Input directory        : {args.input_dir}", file=sys.stderr)
    print(f"  Profile file           : {args.profile.name}", file=sys.stderr)
    print(f"  Profile steps          : {profile.n_steps}", file=sys.stderr)
    print(f"  Profile max length     : {profile.max_measured_length:,} aa", file=sys.stderr)
    if args.max_protein_length is not None:
        print(f"  Threshold: max length  : {args.max_protein_length:,} aa", file=sys.stderr)
    if args.max_runtime is not None:
        print(f"  Threshold: max runtime : {args.max_runtime:.0f} s "
              f"({human_time(args.max_runtime)})", file=sys.stderr)
    print("-" * 66, file=sys.stderr)
    print(f"  Files found            : {len(json_files):,}", file=sys.stderr)
    print(f"  Files parsed           : {len(ok_rows):,}", file=sys.stderr)
    if n_err:
        print(f"  Files with parse errors: {n_err:,}", file=sys.stderr)
    if n_sk_len:
        print(f"  Skipped (length > {args.max_protein_length:,} aa) : {n_sk_len:,}",
              file=sys.stderr)
    if n_sk_rt:
        print(f"  Skipped (runtime > {args.max_runtime:.0f} s) : {n_sk_rt:,}",
              file=sys.stderr)
    print(f"  Files included in total: {n_ok:,}", file=sys.stderr)
    if n_beyond_profile:
        print(
            f"  Files beyond profile   : {n_beyond_profile:,} "
            "(used last-step runtime as fallback)",
            file=sys.stderr,
        )
    if n_ok:
        print("-" * 66, file=sys.stderr)
        print(
            f"  Protein length  min    : {plen_list[0]:,} aa",
            file=sys.stderr,
        )
        print(
            f"  Protein length  median : {plen_list[len(plen_list)//2]:,} aa",
            file=sys.stderr,
        )
        print(
            f"  Protein length  max    : {plen_list[-1]:,} aa",
            file=sys.stderr,
        )
        print("-" * 66, file=sys.stderr)
        print(
            f"  Runtime per file  min  : {rt_list[0]:.2f} s",
            file=sys.stderr,
        )
        print(
            f"  Runtime per file  mean : {total_runtime / n_ok:.2f} s",
            file=sys.stderr,
        )
        print(
            f"  Runtime per file  p95  : {percentile(rt_list, 95):.2f} s",
            file=sys.stderr,
        )
        print(
            f"  Runtime per file  max  : {rt_list[-1]:.2f} s",
            file=sys.stderr,
        )
    print("-" * 66, file=sys.stderr)
    print(
        f"  Total serial MSA runtime : {total_runtime:.2f} s "
        f"({human_time(total_runtime)})",
        file=sys.stderr,
    )
    print("=" * 66, file=sys.stderr)

    # Machine-readable result on stdout (total seconds only)
    print(f"{total_runtime:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
