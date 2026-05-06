#!/usr/bin/env python3
"""
AF3_estimate_GPU_time.py
============================================================================
AlphaFold 3 serial-runtime estimator (parallel token counting).

Outputs
-------
- stdout  : total estimated serial runtime in seconds (machine-readable)
- stderr  : human summary + per-file progress
- TSV     : optional per-file breakdown
            (json_file, token_count, estimated_runtime_seconds)

Usage
-----
  python AF3_estimate_GPU_time.py \\
      --input-dir  /path/to/af3_jsons \\
      --profile    AF3_A800_80G_All_Len_stat.tsv \\
      --output-tsv breakdown.tsv \\
      --workers    16
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
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
#  CCD heavy-atom table (copied verbatim from AF3Parallel.py)
# -----------------------------------------------------------------------------
COMMON_CCD_HEAVY_ATOMS: Dict[str, int] = {
    # Nucleotide triphosphates / di / mono
    "ATP": 31, "ADP": 27, "AMP": 23, "GTP": 32, "GDP": 28, "GMP": 24,
    "CTP": 30, "CDP": 26, "CMP": 22, "UTP": 29, "UDP": 25, "UMP": 21,
    # Cofactors
    "NAD": 44, "NADH": 44, "NADP": 47, "FAD": 53, "FMN": 31, "SAM": 27,
    "SAH": 26, "COA": 51, "HEM": 43, "CLA": 55,
    # Sugars
    "GLC": 12, "GAL": 12, "FUC": 12, "MAN": 12, "FRU": 12, "SUC": 23,
    "BGC": 12, "SIA": 14, "NDG": 15, "NAG": 14,
    # Ions (single heavy atom each)
    "MG": 1, "ZN": 1, "CA": 1, "MN": 1, "FE": 1, "CU": 1, "NI": 1,
    "CO": 1, "CD": 1, "HG": 1, "PT": 1, "NA": 1, "K": 1, "CL": 1,
    "BR": 1, "IOD": 1, "SO4": 5, "PO4": 5, "NO3": 4, "ACT": 4,
    # Porphyrins, buffers, crystallisation agents
    "HEC": 43, "GOL": 6, "EDO": 6, "PEG": 14, "EPE": 22, "MES": 18,
    "TRS": 11, "BTB": 22, "IMD": 5, "DMS": 4, "ACE": 4, "ACY": 4,
    "FMT": 3, "PCA": 9, "CSO": 8, "CME": 12, "MSE": 9, "TPO": 12,
    # Modified residues
    "SEP": 11, "PTR": 18, "M3L": 15, "KCX": 12,
    # Select kinase inhibitors
    "STI": 35, "GNF": 31, "LAP": 28, "ERL": 27, "GEF": 29,
}
DEFAULT_CCD_HEAVY_ATOMS = 20
DEFAULT_SMILES_HEAVY_ATOMS = 20


# -----------------------------------------------------------------------------
#  Tiny logging helpers (no color - stderr only, single-line contract)
# -----------------------------------------------------------------------------
def info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)

def warning(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)

def error_log(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
#  Token counting (copied from AF3Parallel.py)
# -----------------------------------------------------------------------------
def _count_smiles_heavy_atoms(smiles: str) -> int:
    """Count heavy (non-hydrogen) atoms in a SMILES string.

    Prefers RDKit when available; falls back to a regex heuristic that
    handles bracket atoms ([Fe2+], [NH3+], [13C@@H]) and organic subset
    atoms (C, N, O, S, P, F, Cl, Br, I).  The fallback is intentionally
    conservative - it may over-count by a small margin but should never
    under-count significantly.

    Publication audit: the RDKit branch now catches Exception (not just
    ImportError), so a corrupt RDKit install falls through to the regex
    parser instead of crashing the whole job.
    """
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol.GetNumHeavyAtoms()
    except Exception:
        pass

    # -- Regex fallback (no RDKit) ------------------------------------
    # Bracket atoms first, then organic-subset atoms.  H is never in the
    # regex character class so we never need to filter it out afterwards.
    count = 0
    bracket_re = re.compile(r'\[([^]]+)\]')
    for match in bracket_re.finditer(smiles):
        inner = match.group(1)
        elem = re.match(r'\d*([A-Z][a-z]?)', inner)
        if elem and elem.group(1) != 'H':
            count += 1
    bracket_stripped = bracket_re.sub('', smiles)
    organic_atoms = re.findall(r'Cl|Br|[BCNOPSFInbcnops]', bracket_stripped)
    count += len(organic_atoms)
    return max(1, count)


def _count_ccd_heavy_atoms(ccd_id: str) -> int:
    """Return the heavy-atom count for a CCD ligand code."""
    cid = ccd_id.strip().upper()
    return COMMON_CCD_HEAVY_ATOMS.get(cid, DEFAULT_CCD_HEAVY_ATOMS)


def count_tokens_from_af3_json(data: Dict) -> int:
    """Count total AF3 tokens from a parsed JSON dict.

    Token counting rules (per AlphaFold3 documentation):
      Protein : 1 token per amino-acid residue (sequence length)
      RNA/DNA : 1 token per nucleotide         (sequence length)
      Ligand  : 1 token per heavy (non-H) atom
      Ion     : 1 token (always 1 heavy atom)

    Supports both AF3 JSON dialects:
      - Type-based:  {"type": "protein", "sequence": "...", "count": 2}
      - Key-based:   {"protein": {"sequence": "..."}, "count": 2}

    The 'count' field (stoichiometry / copy number) multiplies all tokens
    for that entity.  Returns max(1, total_tokens) to avoid zero-token
    edge cases.
    """
    total_tokens = 0
    sequences = data.get('sequences', [])

    for seq_entry in sequences:
        seq_type = seq_entry.get('type', '').lower()
        count = max(1, int(seq_entry.get('count', 1)))

        if seq_type == 'protein':
            total_tokens += len(seq_entry.get('sequence', '')) * count
        elif seq_type in ('rna', 'dna'):
            total_tokens += len(seq_entry.get('sequence', '')) * count
        elif seq_type == 'ligand':
            ccd_ids = seq_entry.get('ccd_ids', [])
            smiles = seq_entry.get('smiles', '')
            if ccd_ids:
                for cid in ccd_ids:
                    total_tokens += _count_ccd_heavy_atoms(cid) * count
            elif smiles:
                total_tokens += _count_smiles_heavy_atoms(smiles) * count
            else:
                total_tokens += DEFAULT_CCD_HEAVY_ATOMS * count

        # Key-based dialect
        elif 'protein' in seq_entry:
            p = seq_entry['protein']
            c = max(1, int(seq_entry.get('count', 1)))
            total_tokens += len(p.get('sequence', '')) * c
        elif 'rna' in seq_entry:
            r = seq_entry['rna']
            c = max(1, int(seq_entry.get('count', 1)))
            total_tokens += len(r.get('sequence', '')) * c
        elif 'dna' in seq_entry:
            d = seq_entry['dna']
            c = max(1, int(seq_entry.get('count', 1)))
            total_tokens += len(d.get('sequence', '')) * c
        elif 'ligand' in seq_entry:
            lig = seq_entry['ligand']
            c = max(1, int(seq_entry.get('count', 1)))
            ccd_ids = lig.get('ccdCodes', lig.get('ccd_ids', []))
            smiles = lig.get('smilesString', lig.get('smiles', ''))
            if isinstance(ccd_ids, str):
                ccd_ids = [ccd_ids]
            if ccd_ids:
                for cid in ccd_ids:
                    total_tokens += _count_ccd_heavy_atoms(cid) * c
            elif smiles:
                total_tokens += _count_smiles_heavy_atoms(smiles) * c
            else:
                total_tokens += DEFAULT_CCD_HEAVY_ATOMS * c

    return max(1, total_tokens)


# -----------------------------------------------------------------------------
#  Runtime profile (simplified from TokenMemoryProfileLoader -
#  no VRAM logic, just token_count -> runtime_seconds)
# -----------------------------------------------------------------------------
@dataclass
class TokenMemoryProfile:
    """Profile entry keyed by total token count."""
    token_count: int
    memory_usage_mb: int
    runtime_seconds: float
    success: bool = True


class RuntimeProfileLoader:
    """Loads a runtime profile TSV and answers "how long does N tokens take?".

    This is a trimmed-down version of TokenMemoryProfileLoader from
    AF3Parallel.py (same step-function model + gap-fill
    algorithm), stripped of:
      - VRAM overflow detection
      - Built-in profiles (A800 / RTX 4090)
      - GPU-VRAM interaction
    because estimation here only needs runtime.

    Required TSV columns: token_count, peak_memory_mb, runtime_seconds
    Optional column     : success (default True if missing)
    """

    REQUIRED_COLUMNS = {'token_count', 'peak_memory_mb', 'runtime_seconds'}

    def __init__(self, profile_file: Path, gap_fill: bool = True,
                 include_failed: bool = False, timeout_runtime: float = 7200.0):
        self.profile_file = profile_file
        self.gap_fill = gap_fill
        self.include_failed = include_failed
        self.timeout_runtime = timeout_runtime
        # token_count -> profile entry
        self.profiles: Dict[int, TokenMemoryProfile] = {}
        # step function: list of (min_token, max_token|None, memory_mb, runtime_avg)
        self._steps: List[Tuple[int, Optional[int], int, float]] = []
        self._step_min_tokens: List[int] = []

        self._load()
        self._build_steps()
        if gap_fill:
            self._fill_gaps()
        self._rebuild_step_index()

        if not self._steps:
            raise RuntimeError(
                f"Profile file produced no usable steps: {profile_file}"
            )

    # --- TSV parsing ------------------------------------------------------
    def _load(self) -> None:
        with open(self.profile_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            if reader.fieldnames is None:
                raise RuntimeError("Profile file is empty (no header row)")
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise RuntimeError(
                    f"Profile file missing required column(s): {sorted(missing)}. "
                    f"Found: {list(reader.fieldnames)}"
                )

            n_rows = 0
            n_skipped_failed = 0
            for row in reader:
                try:
                    token_count = int(row['token_count'])
                    memory_mb = int(row['peak_memory_mb'])
                    runtime = float(row['runtime_seconds'])
                    success_flag = row.get('success', 'True').strip().lower() == 'true'
                except (ValueError, KeyError):
                    continue

                if not success_flag and not self.include_failed:
                    n_skipped_failed += 1
                    continue

                existing = self.profiles.get(token_count)
                if existing is None:
                    self.profiles[token_count] = TokenMemoryProfile(
                        token_count=token_count,
                        memory_usage_mb=memory_mb,
                        runtime_seconds=runtime,
                        success=success_flag,
                    )
                else:
                    # Keep the worst case for the same token_count
                    if memory_mb > existing.memory_usage_mb:
                        existing.memory_usage_mb = memory_mb
                    if runtime > existing.runtime_seconds:
                        existing.runtime_seconds = runtime
                    if not success_flag:
                        existing.success = False
                n_rows += 1

        info(f"Loaded {n_rows} rows from {self.profile_file.name}"
             f" ({len(self.profiles)} unique token counts"
             + (f", skipped {n_skipped_failed} failed rows"
                if n_skipped_failed else "") + ")")

    # --- Step-function construction --------------------------------------
    def _build_steps(self) -> None:
        """Group consecutive token counts with identical memory into steps.

        The logic mirrors TokenMemoryProfileLoader._build_steps_from_profiles()
        from the parallel script (v2.9 gap-safe version).
        """
        if not self.profiles:
            return
        sorted_tokens = sorted(self.profiles.keys())

        step_start_token = sorted_tokens[0]
        step_tokens: List[int] = []
        prev_mem: Optional[int] = None

        for token in sorted_tokens:
            mem = self.profiles[token].memory_usage_mb
            if prev_mem is not None and mem != prev_mem:
                avg_rt = (
                    sum(self.profiles[t].runtime_seconds for t in step_tokens)
                    / len(step_tokens)
                )
                self._steps.append((step_start_token, token, prev_mem, avg_rt))
                step_start_token = token
                step_tokens = []
            step_tokens.append(token)
            prev_mem = mem

        if step_tokens and prev_mem is not None:
            avg_rt = (
                sum(self.profiles[t].runtime_seconds for t in step_tokens)
                / len(step_tokens)
            )
            self._steps.append((step_start_token, None, prev_mem, avg_rt))

        # Drop non-monotonic tail steps (> 30% memory drop is almost always
        # an OOM-crash artifact - same sanity check as the parallel script).
        if len(self._steps) > 1:
            cleaned = [self._steps[0]]
            for step in self._steps[1:]:
                prev_mem_val = cleaned[-1][2]
                cur_mem = step[2]
                drop = (prev_mem_val - cur_mem) / prev_mem_val if prev_mem_val else 0
                if drop > 0.30:
                    warning(
                        f"Skipping anomalous memory step at token >= {step[0]}: "
                        f"{cur_mem:,} MB drops >30% from previous step "
                        f"{prev_mem_val:,} MB. Likely an OOM-crash artifact."
                    )
                else:
                    cleaned.append(step)
            self._steps = cleaned

    # --- Gap filling -----------------------------------------------------
    def _fill_gaps(self) -> None:
        """Insert interpolated steps where consecutive steps do not touch."""
        if len(self._steps) < 2:
            return
        filled: List[Tuple[int, Optional[int], int, float]] = []
        gaps_found = 0
        for i, step in enumerate(self._steps):
            filled.append(step)
            if i >= len(self._steps) - 1:
                continue
            curr_min, curr_max, curr_mem, curr_rt = step
            next_min, next_max, next_mem, next_rt = self._steps[i + 1]
            if curr_max is None:
                continue
            gap = next_min - curr_max
            if gap <= 0:
                continue
            gap_mem = int(curr_mem + (next_mem - curr_mem) * 0.5)
            gap_rt = curr_rt + (next_rt - curr_rt) * 0.5
            filled.append((curr_max, next_min, gap_mem, gap_rt))
            gaps_found += 1
        if gaps_found:
            info(f"Profile gap-fill: {gaps_found} gap(s) interpolated "
                 f"({len(filled)} total steps after fill)")
            self._steps = filled

    def _rebuild_step_index(self) -> None:
        self._step_min_tokens = [s[0] for s in self._steps]

    # --- Lookup ----------------------------------------------------------
    def _lookup_step(self, token_count: int) -> int:
        """O(log N) bisect lookup into the step function."""
        if not self._step_min_tokens:
            return -1
        idx = bisect.bisect_right(self._step_min_tokens, token_count) - 1
        return max(0, min(idx, len(self._steps) - 1))

    def estimate_runtime_seconds(self, token_count: int) -> float:
        if not self._steps:
            return self.timeout_runtime
        return self._steps[self._lookup_step(token_count)][3]

    @property
    def n_steps(self) -> int:
        return len(self._steps)


# -----------------------------------------------------------------------------
#  Parallel token counting worker + driver
# -----------------------------------------------------------------------------
def _count_tokens_worker(json_path_str: str) -> Tuple[str, Optional[int], Optional[str]]:
    """Parse a single JSON file in a subprocess worker.

    Returns (path_str, token_count_or_None, error_or_None).
    Kept at module top level so it is picklable by ProcessPoolExecutor
    on 'spawn' platforms.
    """
    try:
        with open(json_path_str, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokens = count_tokens_from_af3_json(data)
        return (json_path_str, tokens, None)
    except Exception as e:
        return (json_path_str, None, f"{type(e).__name__}: {e}")


def count_tokens_batch(
    json_files: List[Path],
    workers: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, str]]]:
    """Count tokens for a batch of JSON files in parallel.

    Directly mirrors the parallel strategy from
    AF3Parallel.py::collect_json_files:

      - 'fork' start method on Linux/macOS (instant worker creation,
        no module reimport per worker)
      - 'spawn' on Windows (mandatory)
      - imap_unordered + chunksize for load balancing and low IPC overhead
      - single-threaded fast path for tiny batches (<= 10 files)

    Returns:
        ok_rows  : [(path, token_count), ...]  for successfully parsed files
        err_rows : [(path, error_message), ...] for failed parses
    """
    n_files = len(json_files)
    if n_files == 0:
        return [], []

    # --- Decide worker count --------------------------------------------
    # multiprocessing.cpu_count() reports the machine's total logical
    # cores, which is WRONG inside containers / SLURM / cgroups that
    # pin the process to a subset. os.sched_getaffinity() (Linux only)
    # returns the cores actually schedulable.
    try:
        usable_cpus = len(os.sched_getaffinity(0))  # Linux
    except (AttributeError, OSError):
        usable_cpus = multiprocessing.cpu_count()
    mp_cpu = multiprocessing.cpu_count()

    user_specified = workers is not None and workers > 0
    if not user_specified:
        # Auto mode: respect CPU affinity to avoid oversubscribing a cgroup
        workers = max(1, min(usable_cpus, n_files))
    else:
        # User explicitly asked for N - do NOT clamp to usable_cpus.
        # Affinity detection can mislead on login shells / detached sessions.
        workers = max(1, min(workers, n_files))

    # --- Always log the decision (not gated on verbose) ------------------
    if usable_cpus != mp_cpu:
        info(f"CPU affinity reports {usable_cpus}/{mp_cpu} cores "
             f"available (cgroup/taskset/SLURM quota)")
        if not user_specified and usable_cpus == 1 and n_files > 10:
            warning(
                f"Auto-detected 1 usable CPU, falling back to single-threaded. "
                f"If you believe more cores are available, override with "
                f"--workers N (e.g. --workers {mp_cpu})."
            )
    info(f"Token counting will use {workers} worker process(es)"
         + (" (user-specified)" if user_specified else " (auto)"))

    # --- Fast path for small batches: avoid process-pool overhead -------
    if n_files <= 10 or workers == 1:
        reason = "small file count" if n_files <= 10 else "workers=1"
        info(f"Using single-threaded parsing ({reason}, {n_files} file(s))")
        ok_rows: List[Tuple[Path, int]] = []
        err_rows: List[Tuple[Path, str]] = []
        for jf in json_files:
            _, tokens, err = _count_tokens_worker(str(jf))
            if tokens is None:
                err_rows.append((jf, err or "unknown error"))
            else:
                ok_rows.append((jf, tokens))
        return ok_rows, err_rows

    # --- Parallel path --------------------------------------------------
    # Prefer 'fork' on Linux only - same reasoning as the parallel runner:
    # spawn would restart Python and re-import this module in every worker,
    # adding multi-second overhead per worker on many-core machines.
    # Publication audit: macOS now uses 'spawn' too (fork is unsafe on
    # Darwin since Python 3.8 due to libdispatch / Objective-C runtime).
    preferred_ctx = 'fork' if _platform.system() == 'Linux' else 'spawn'
    try:
        ctx = multiprocessing.get_context(preferred_ctx)
    except ValueError:
        ctx = multiprocessing.get_context('spawn')
        preferred_ctx = 'spawn'

    # chunksize: same heuristic as parallel runner. Too small = IPC overhead;
    # too large = poor load balance for mixed file sizes.
    chunksize = max(1, min(10, n_files // (workers * 4) or 1))

    info(f"Parallel parsing: {workers} worker(s), "
         f"start_method={preferred_ctx}, chunksize={chunksize}")

    json_path_strs = [str(jf) for jf in json_files]
    ok_rows = []
    err_rows = []
    completed = 0
    # Publication audit: monotonic clock (immune to NTP step adjustments)
    start = time.monotonic()
    last_progress_pct = -1

    info(f"Starting worker pool...")
    with ctx.Pool(processes=workers) as pool:
        info(f"Pool ready, dispatching {n_files} file(s)")
        for path_str, tokens, err in pool.imap_unordered(
            _count_tokens_worker, json_path_strs, chunksize=chunksize
        ):
            completed += 1
            jf = Path(path_str)
            if tokens is None:
                err_rows.append((jf, err or "unknown error"))
            else:
                ok_rows.append((jf, tokens))

            if show_progress:
                pct = int(completed / n_files * 100)
                if (pct >= last_progress_pct + 2
                        or completed % 50 == 0
                        or completed == n_files):
                    elapsed = time.monotonic() - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (n_files - completed) / rate if rate > 0 else 0.0
                    print(
                        f"\r  Parsing: {completed}/{n_files} ({pct:3d}%) | "
                        f"Elapsed: {elapsed:.1f}s | "
                        f"Speed: {rate:.1f} files/s | ETA: {eta:.1f}s   ",
                        end='', flush=True, file=sys.stderr,
                    )
                    last_progress_pct = pct

    if show_progress:
        elapsed = time.monotonic() - start
        rate = n_files / elapsed if elapsed > 0 else 0.0
        print(
            f"\r  Parsing: {n_files}/{n_files} (100%) | "
            f"Time: {elapsed:.1f}s | Avg Speed: {rate:.1f} files/s          ",
            file=sys.stderr,
        )

    return ok_rows, err_rows


# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------
def human_time(seconds: float) -> str:
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h or d: parts.append(f"{h}h")
    if m or h or d: parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate total serial (single-GPU) runtime for a batch of "
            "AlphaFold3 JSON files using a token->runtime profile. "
            "Parses JSON files in parallel (ProcessPoolExecutor) using the "
            "same strategy as AF3Parallel.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input-dir", required=True, type=Path,
        help="Directory containing AF3 input *.json files.",
    )
    parser.add_argument(
        "-p", "--profile", required=True, type=Path,
        help="Profile TSV file (columns: token_count, peak_memory_mb, "
             "runtime_seconds, [success]). "
             "Example: AF3_A800_80G_All_Len_stat.tsv",
    )
    parser.add_argument(
        "-o", "--output-tsv", type=Path, default=None,
        help="Optional output TSV file with per-file breakdown "
             "(json_file, token_count, estimated_runtime_seconds).",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recursively scan input-dir for *.json files.",
    )
    parser.add_argument(
        "--pattern", type=str, default="*.json",
        help="Glob pattern for JSON files.",
    )
    parser.add_argument(
        "--no-gap-fill", action="store_true",
        help="Disable automatic gap-filling between sparse profile steps.",
    )
    parser.add_argument(
        "--include-failed", action="store_true",
        help="Include profile rows with success=False when building the "
             "step function (useful to capture timeout runtimes for very "
             "large inputs).",
    )
    parser.add_argument(
        "--timeout-runtime", type=float, default=7200.0,
        help="Fallback runtime (seconds) for token counts beyond the profile.",
    )
    parser.add_argument(
        "-w", "--workers", type=int, default=0,
        help="Number of parallel worker processes for JSON parsing. "
             "0 or negative = auto (use all available CPU cores). "
             "1 = force single-threaded. Explicit values override CPU "
             "affinity detection.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress the live progress line during parallel parsing.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-file estimation details to stderr.",
    )
    args = parser.parse_args()

    # -- Validate inputs -------------------------------------------------
    if not args.input_dir.is_dir():
        error_log(f"input-dir is not a directory: {args.input_dir}")
        return 2
    if not args.profile.is_file():
        error_log(f"profile file not found: {args.profile}")
        return 2

    # -- Load profile ----------------------------------------------------
    profile = RuntimeProfileLoader(
        profile_file=args.profile,
        gap_fill=not args.no_gap_fill,
        include_failed=args.include_failed,
        timeout_runtime=args.timeout_runtime,
    )
    info(f"Built {profile.n_steps} step(s) "
         f"(gap_fill={'off' if args.no_gap_fill else 'on'})")

    # -- Enumerate JSON files --------------------------------------------
    if args.recursive:
        json_files = sorted(args.input_dir.rglob(args.pattern))
    else:
        json_files = sorted(args.input_dir.glob(args.pattern))
    if not json_files:
        error_log(f"no files matching '{args.pattern}' under {args.input_dir}")
        return 1
    info(f"Found {len(json_files)} JSON file(s) under {args.input_dir}")

    # -- Parse all JSON files in parallel, then look up runtimes ---------
    # Parsing is the expensive step (RDKit SMILES + I/O) - do it in a
    # process pool. Runtime lookup is a trivial bisect over the step list,
    # so it stays in the main process (also avoids shipping the profile
    # object to every worker).
    # Publication audit: monotonic clock (immune to NTP step adjustments)
    parse_start = time.monotonic()
    ok_rows, err_rows = count_tokens_batch(
        json_files,
        workers=args.workers if args.workers and args.workers > 0 else None,
        show_progress=not args.no_progress,
    )
    parse_elapsed = time.monotonic() - parse_start

    if args.verbose and err_rows:
        for jf, msg in err_rows:
            warning(f"Could not parse {jf.name}: {msg}")

    # Sort by filename so TSV output is reproducible (imap_unordered
    # returns results in completion order, not input order).
    ok_rows.sort(key=lambda r: r[0].name)

    # -- Estimate runtime per file + accumulate --------------------------
    rows: List[Tuple[str, int, float]] = []
    total_runtime = 0.0
    for jf, tokens in ok_rows:
        rt = profile.estimate_runtime_seconds(tokens)
        total_runtime += rt
        rows.append((jf.name, tokens, rt))
        if args.verbose:
            print(f"[file  ] {jf.name}\ttokens={tokens}\test_runtime={rt:.2f}s",
                  file=sys.stderr)

    info(f"Parsed {len(ok_rows)} file(s) in {parse_elapsed:.2f}s")

    # -- Optional per-file breakdown TSV ---------------------------------
    if args.output_tsv is not None and rows:
        args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_tsv, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f, delimiter='\t', lineterminator='\n')
            w.writerow(['json_file', 'token_count', 'estimated_runtime_seconds'])
            for name, tokens, rt in rows:
                w.writerow([name, tokens, f"{rt:.3f}"])
        info(f"Per-file breakdown written to {args.output_tsv}")

    # -- Summary ---------------------------------------------------------
    n_ok = len(rows)
    n_parse_err = len(err_rows)
    print("", file=sys.stderr)
    print("=" * 64, file=sys.stderr)
    print(f"  Files estimated        : {n_ok}", file=sys.stderr)
    if n_parse_err:
        print(f"  Files with parse errors: {n_parse_err}", file=sys.stderr)
    if n_ok:
        print(f"  Mean runtime per file  : {total_runtime / n_ok:.2f} s",
              file=sys.stderr)
        # Token distribution summary
        tokens_list = [r[1] for r in rows]
        tokens_list.sort()
        tmin = tokens_list[0]
        tmax = tokens_list[-1]
        tmed = tokens_list[len(tokens_list) // 2]
        print(f"  Token range (min/med/max): {tmin} / {tmed} / {tmax}",
              file=sys.stderr)
    print(f"  Total serial runtime   : {total_runtime:.2f} s "
          f"({human_time(total_runtime)})", file=sys.stderr)
    print("=" * 64, file=sys.stderr)

    # Machine-readable result on stdout: total runtime in seconds
    print(f"{total_runtime:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
