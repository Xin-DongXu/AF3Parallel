#!/usr/bin/env python3
"""
AlphaFold3 GPU Memory Time-Series Profiler
==========================================

Important: AF3 uses Abseil flags. The correct boolean form is --noFLAG
(no hyphen, no underscore separator). Examples::

    --norun_data_pipeline       CORRECT
    --no-run-data_pipeline      WRONG (parses as a fatal flag error)
    --no_run_data_pipeline      WRONG

This script auto-corrects the most common wrong forms passed via
--extra-args.

Output TSV columns
------------------
    json_file | token_count | elapsed_seconds | memory_used_mb |
    memory_total_mb | memory_percent | gpu_util | temperature |
    success | job_runtime_seconds

Usage
-----
    python af3_GPU_memory_timeseries.py \\
        --stat-file  AF3_A800_80G_All_Len_stat.txt \\
        --input-dir  ./af_input \\
        --output-dir ./timeseries_output \\
        --sif        alphafold3.sif \\
        --af3-db     ./af3_DB \\
        --models     ./models \\
        --monitor-interval 0.5 \\
        --n-per-step 3 \\
        --workers    8

Requirements
------------
* Python >= 3.8
* nvidia-smi (NVIDIA driver) in PATH
* singularity in PATH
* AlphaFold3 Singularity image (--sif), model directory and database
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Console output helpers
# ===========================================================================

class _Colors:
    RED    = "\033[0;31m"
    GREEN  = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE   = "\033[0;34m"
    NC     = "\033[0m"


_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _emit(prefix: str, msg: str, color: str) -> None:
    if _USE_COLOR:
        print(f"{color}[{prefix}]{_Colors.NC} {msg}", flush=True)
    else:
        print(f"[{prefix}] {msg}", flush=True)


def info(msg: str) -> None:    _emit("INFO",  msg, _Colors.BLUE)
def warning(msg: str) -> None: _emit("WARN",  msg, _Colors.YELLOW)
def error(msg: str) -> None:   _emit("ERROR", msg, _Colors.RED)
def ok_msg(msg: str) -> None:  _emit("OK",    msg, _Colors.GREEN)


# --- BEGIN AF3 TOKEN COUNTER ---------------------------------------------

# Heavy-atom counts for common CCD ligands (PDB Chemical Component
# Dictionary, hydrogens excluded). Values are computed from the molecular
# formula. This list is illustrative and not exhaustive; unknown codes fall
# back to _CCD_FALLBACK.
_CCD_HEAVY: Dict[str, int] = {
    # nucleotides (computed from formula, H excluded)
    "ATP": 31, "ADP": 27, "AMP": 23,
    "GTP": 32, "GDP": 28, "GMP": 24,
    "CTP": 29, "CDP": 25, "CMP": 21,
    "UTP": 29, "UDP": 25, "UMP": 21,
    # cofactors
    "NAD": 44, "NDP": 48, "NAP": 48,        # NAD+, NADPH, NADP+
    "FAD": 53, "FMN": 31,
    "COA": 51, "SAM": 27, "SAH": 26,
    "PLP": 16, "FES": 4,  "SF4": 8,
    # heme / chlorophyll
    "HEM": 43, "HEC": 43, "CLA": 55,
    # sugars
    "GLC": 12, "GAL": 12, "MAN": 12, "FUC": 11, "FRU": 12,
    "BGC": 12, "NAG": 15, "NDG": 15, "SIA": 20,
    # ions and small anions
    "MG": 1, "ZN": 1, "CA": 1, "MN": 1, "FE": 1, "CU": 1, "NI": 1,
    "CO": 1, "CD": 1, "HG": 1, "PT": 1, "NA": 1, "K":  1, "CL": 1,
    "BR": 1, "IOD": 1,
    "SO4": 5, "PO4": 5, "PO3": 4, "NO3": 4,
    # buffers / cryo-protectants / common additives
    "GOL": 6,  "EDO": 4,  "PEG": 7,  "MPD": 8,  "EPE": 15,
    "MES": 12, "TRS": 8,  "BTB": 19, "IMD": 5,  "DMS": 4,
    "ACE": 4,  "ACY": 4,  "ACT": 4,  "FMT": 3,  "EOH": 3,  "DMF": 4,
    # modified residues / common drugs (heavy-atom counts only; included as
    # rough estimates for bucketing rather than absolute accuracy)
    "PCA": 9,  "MSE": 9,  "TPO": 12, "SEP": 11, "PTR": 18,
    "M3L": 12, "KCX": 12, "CSO": 8,  "CME": 12,
    "STI": 37, "GNF": 31, "LAP": 28, "ERL": 27, "GEF": 29,
    "BEN": 7,  "LAC": 8,  "MLI": 6,
}
_CCD_FALLBACK = 15  # heavy-atom fallback for CCD codes not in the table

# SMILES "organic subset" atoms (single-letter, may appear outside brackets).
# Aromatic forms (lowercase b,c,n,o,p,s) are also single heavy atoms.
_SMILES_ORGANIC = set("BCNOPSFI") | set("bcnops")


def _smiles_heavy_atoms(smiles: str) -> int:
    """Count heavy (non-H) atoms in a SMILES string without RDKit.

    Handles:
      - organic-subset atoms outside brackets (B, C, N, O, P, S, F, Cl, Br, I);
      - aromatic lowercase (b, c, n, o, p, s);
      - bracketed atoms [Fe], [CH4], [14C@H], [Ce+3] etc.;
      - isotope and charge specifications inside brackets.

    Returns at least 1 for any non-empty input -- AF3 charges 1 token per
    heavy atom, so an unparseable single-atom ligand still consumes a token.
    """
    if not smiles:
        return 0
    n = len(smiles)
    i = 0
    count = 0
    while i < n:
        ch = smiles[i]
        if ch == "[":
            j = i + 1
            while j < n and smiles[j] != "]":
                j += 1
            inner = smiles[i + 1:j]
            # skip leading isotope digits
            k = 0
            while k < len(inner) and inner[k].isdigit():
                k += 1
            rest = inner[k:]
            if rest:
                # element symbol is 1 or 2 letters, second letter lowercase
                if len(rest) >= 2 and rest[0].isalpha() and rest[1].islower():
                    elem = rest[:2]
                else:
                    elem = rest[:1]
                if elem.upper() != "H":
                    count += 1
            i = j + 1
            continue
        if ch.isalpha():
            # two-letter organic-subset elements outside brackets
            if (ch in ("C", "B")
                    and i + 1 < n
                    and smiles[i + 1] in ("l", "r")):
                count += 1
                i += 2
                continue
            if ch in _SMILES_ORGANIC:
                count += 1
            i += 1
            continue
        # bonds, ring closures, parentheses, stereo markers, '%', digits, etc.
        i += 1
    return max(count, 1)


def _entity_copies(entity: dict) -> int:
    """Number of chain copies for a sequence/ligand entity.

    AF3 schema: id is a string for a single chain or a list of strings
    for multiple chain copies.
    """
    id_field = entity.get("id", "A")
    if isinstance(id_field, (list, tuple)):
        return max(len(id_field), 1)
    return 1


def _ccd_codes_heavy(codes_field) -> int:
    """Sum of heavy atoms across one or more CCD codes."""
    if isinstance(codes_field, str):
        codes = [codes_field]
    elif isinstance(codes_field, (list, tuple)):
        codes = list(codes_field)
    else:
        return _CCD_FALLBACK
    return sum(_CCD_HEAVY.get(str(c).upper(), _CCD_FALLBACK) for c in codes)


def af3_count_tokens(data: dict) -> int:
    """Total AF3 token count for a parsed input JSON object.

    Iterates data["sequences"] and applies the official rules:
      - protein/RNA/DNA: len(sequence) * copies;
      - ligand: heavy-atom count * copies (SMILES or ccdCodes).
    Unknown entity types are skipped silently to forward-compatibly support
    future schema extensions.
    """
    total = 0
    for entry in data.get("sequences", []) or []:
        if not isinstance(entry, dict):
            continue
        if "protein" in entry:
            ent = entry["protein"]
            total += len(ent.get("sequence", "")) * _entity_copies(ent)
        elif "rna" in entry:
            ent = entry["rna"]
            total += len(ent.get("sequence", "")) * _entity_copies(ent)
        elif "dna" in entry:
            ent = entry["dna"]
            total += len(ent.get("sequence", "")) * _entity_copies(ent)
        elif "ligand" in entry:
            lig = entry["ligand"]
            copies = _entity_copies(lig)
            if "smiles" in lig:
                total += _smiles_heavy_atoms(lig["smiles"]) * copies
            elif "ccdCodes" in lig:
                total += _ccd_codes_heavy(lig["ccdCodes"]) * copies
            else:
                total += _CCD_FALLBACK * copies
    return total

# --- END AF3 TOKEN COUNTER -----------------------------------------------


# ===========================================================================
# Worker for parallel JSON parsing
# ===========================================================================

def _parse_worker(path_str: str) -> Dict:
    """Top-level (picklable) worker for ProcessPoolExecutor."""
    p = Path(path_str)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {"json_file": p.name, "json_path": str(p),
                "tokens": af3_count_tokens(data), "error": None}
    except (OSError, json.JSONDecodeError) as exc:
        return {"json_file": p.name, "json_path": str(p),
                "tokens": 0, "error": str(exc)}


# ===========================================================================
# Stat file -> gradient steps
# ===========================================================================

def parse_stat_file(stat_file: Path,
                    bucket_mb: float = 100.0,
                    merge_gap_mb: float = 200.0
                    ) -> List[Dict]:
    """Read a peak-memory TSV and derive discrete VRAM gradient steps.

    The input TSV must include columns token_count and peak_memory_mb
    (extra columns are ignored). Rows with non-numeric values are skipped.

    Returns a list (sorted by min_token) of::

        {"min_token": int, "max_token": int,
         "memory_mb": float, "token_list": List[int]}
    """
    rows: List[Tuple[int, float]] = []
    with open(stat_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or \
                "token_count" not in reader.fieldnames or \
                "peak_memory_mb" not in reader.fieldnames:
            raise ValueError(
                f"{stat_file}: missing required columns "
                f"'token_count' and/or 'peak_memory_mb'. "
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            try:
                rows.append((int(row["token_count"]),
                             float(row["peak_memory_mb"])))
            except (KeyError, ValueError, TypeError):
                continue

    if not rows:
        raise ValueError(f"No valid rows found in {stat_file}")

    # Bucket by memory rounded to nearest bucket_mb MB
    buckets: Dict[float, List[int]] = defaultdict(list)
    for tokens, mem in rows:
        key = round(mem / bucket_mb) * bucket_mb
        buckets[key].append(tokens)

    steps = sorted(
        ({"memory_mb": float(k),
          "token_list": sorted(v),
          "min_token": min(v),
          "max_token": max(v)} for k, v in buckets.items()),
        key=lambda s: s["min_token"],
    )

    # Merge adjacent steps within merge_gap_mb of each other
    merged: List[Dict] = []
    for s in steps:
        if merged and abs(s["memory_mb"] - merged[-1]["memory_mb"]) <= merge_gap_mb:
            merged[-1]["max_token"]   = max(merged[-1]["max_token"],  s["max_token"])
            merged[-1]["min_token"]   = min(merged[-1]["min_token"],  s["min_token"])
            merged[-1]["token_list"]  = sorted(merged[-1]["token_list"]
                                               + s["token_list"])
            # weighted-ish midpoint memory; average is fine for reporting
            merged[-1]["memory_mb"]   = (merged[-1]["memory_mb"]
                                         + s["memory_mb"]) / 2.0
        else:
            merged.append(s)

    info(f"Detected {len(merged)} VRAM gradient step(s):")
    for s in merged:
        info(f"  tokens [{s['min_token']:>6} - {s['max_token']:>6}]  "
             f"->  ~{s['memory_mb']:.0f} MB VRAM")
    return merged


# ===========================================================================
# Representative file selection
# ===========================================================================

def select_representatives(steps: List[Dict],
                           parsed: List[Dict],
                           n: int = 3) -> List[Dict]:
    """For each gradient step, pick up to n JSON files spread across
    the [low, mid, high] of its token range. Returns a deduplicated list."""
    selected: List[Dict] = []
    seen: set = set()

    for step in steps:
        lo, hi = step["min_token"], step["max_token"]
        cands = sorted(
            [p for p in parsed
             if p["error"] is None and lo <= p["tokens"] <= hi],
            key=lambda x: x["tokens"],
        )
        if not cands:
            warning(f"  gradient [{lo}-{hi}]: no matching JSON files; skipping.")
            continue

        if len(cands) <= n:
            picks = list(cands)
        else:
            target_idx = [0, len(cands) // 2, len(cands) - 1][:n]
            picks: List[Dict] = []
            taken: set = set()
            for tgt in target_idx:
                # Walk outward from tgt to find an unused token count.
                for delta in range(len(cands)):
                    placed = False
                    for sign in (0, 1, -1):
                        probe = tgt + sign * delta
                        if 0 <= probe < len(cands):
                            c = cands[probe]
                            if c["tokens"] not in taken:
                                picks.append(c)
                                taken.add(c["tokens"])
                                placed = True
                                break
                    if placed:
                        break
            picks = picks[:n]

        for p in picks:
            if p["json_file"] not in seen:
                seen.add(p["json_file"])
                selected.append(p)
                info(f"  gradient [{lo:>6}-{hi:>6}] -> {p['json_file']}  "
                     f"(tokens={p['tokens']})")

    info(f"Total files selected for profiling: {len(selected)}")
    return selected


# ===========================================================================
# GPU sampling (fine-grained, background thread, interruptible)
# ===========================================================================

def _gpu_query() -> Dict:
    """One nvidia-smi snapshot. Returns zeros on failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu,"
             "temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            first = r.stdout.strip().splitlines()[0]
            parts = [x.strip() for x in first.split(",")]
            if len(parts) == 4:
                mu, mt = int(parts[0]), int(parts[1])
                return {
                    "memory_used_mb":  mu,
                    "memory_total_mb": mt,
                    "memory_percent":  round(mu / mt * 100, 2) if mt else 0.0,
                    "gpu_util":        int(parts[2]),
                    "temperature":     int(parts[3]),
                }
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return {"memory_used_mb": 0, "memory_total_mb": 0,
            "memory_percent": 0.0, "gpu_util": 0, "temperature": 0}


class TimedGPUMonitor:
    """Thread-safe interval sampler. stop() returns within one poll
    even when the child wait is mid-interval (uses Event.wait)."""

    def __init__(self, interval: float = 0.5) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.interval = float(interval)
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._records: List[Tuple[float, Dict]] = []

    def start(self) -> None:
        with self._lock:
            self._records = []
        self._stop.clear()
        self._t0 = time.time()
        self._thread = Thread(target=self._loop, daemon=True,
                              name="TimedGPUMonitor")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            elapsed = round(time.time() - self._t0, 3)
            sample = _gpu_query()
            with self._lock:
                self._records.append((elapsed, sample))
            self._stop.wait(timeout=self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)

    def get_records(self) -> List[Tuple[float, Dict]]:
        with self._lock:
            return list(self._records)


# ===========================================================================
# Singularity command builder
# ===========================================================================

# Common wrong forms users may pass via --extra-args -> correct AF3 form.
_FLAG_FIX = {
    "--no-run-data_pipeline": "--norun_data_pipeline",
    "--no_run_data_pipeline": "--norun_data_pipeline",
    "--no-run-inference":     "--norun_inference",
    "--no_run_inference":     "--norun_inference",
}


def _normalise_extra_args(raw: List[str]) -> List[str]:
    """Apply known boolean-flag corrections and warn about each fix."""
    out: List[str] = []
    for a in raw:
        a = a.strip()
        fixed = _FLAG_FIX.get(a, a)
        if fixed != a:
            warning(f"Auto-corrected AF3 flag: {a!r} -> {fixed!r}")
        out.append(fixed)
    return out


def _build_cmd(parsed: Dict, sif: str, af3_db: str, models: str,
               out_dir: str, extra_args: List[str]) -> List[str]:
    """Build the full singularity exec --nv ... command."""
    json_path = Path(parsed["json_path"])
    abs_input  = str(json_path.parent.resolve())
    abs_output = os.path.abspath(out_dir)
    abs_db     = os.path.abspath(af3_db)
    abs_models = os.path.abspath(models)
    os.makedirs(abs_output, exist_ok=True)

    af3_flags = _normalise_extra_args(extra_args)
    if ("--norun_data_pipeline" not in af3_flags
            and "--norun_inference" not in af3_flags):
        af3_flags.append("--norun_data_pipeline")

    return [
        "singularity", "exec", "--nv",
        "--bind", f"{abs_db}:/root/public_databases",
        "--bind", f"{abs_input}:/root/af_input",
        "--bind", f"{abs_output}:/root/af_output",
        "--bind", f"{abs_models}:/root/models",
        sif,
        "python", "run_alphafold.py",
        f"--json_path=/root/af_input/{parsed['json_file']}",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output",
    ] + af3_flags


# ===========================================================================
# AF3 invocation with concurrent GPU monitoring
# ===========================================================================

# Track currently-running AF3 subprocess so signal handlers can kill it.
_CHILD_LOCK = threading.Lock()
_CURRENT_CHILD: Optional[subprocess.Popen] = None


def run_af3_job(parsed: Dict, sif: str, af3_db: str, models: str,
                out_dir: str, extra_args: List[str],
                monitor_interval: float, timeout_seconds: int,
                log_dir: Optional[Path]
                ) -> Dict:
    """Execute one AF3 job and record GPU memory over time.

    Returns {"success": bool, "runtime_seconds": float,
             "records": List[(elapsed_s, gpu_dict)]}.
    """
    global _CURRENT_CHILD
    cmd = _build_cmd(parsed, sif, af3_db, models, out_dir, extra_args)
    info(f"  CMD: {' '.join(shlex.quote(x) for x in cmd)}")

    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{Path(parsed['json_file']).stem}.log"

    monitor = TimedGPUMonitor(interval=monitor_interval)
    monitor.start()
    t0 = time.time()
    rc = -1
    log_f = open(log_path, "w", encoding="utf-8") if log_path else \
        subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f if log_path else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        with _CHILD_LOCK:
            _CURRENT_CHILD = proc
        try:
            rc = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            warning(f"  AF3 timed out (>{timeout_seconds} s); killing.")
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            rc = 124
    finally:
        with _CHILD_LOCK:
            _CURRENT_CHILD = None
        if log_path:
            log_f.close()  # type: ignore[union-attr]
        runtime = time.time() - t0
        monitor.stop()

    success_flag = (rc == 0)
    tag = "SUCCESS" if success_flag else f"FAILED (rc={rc})"
    (ok_msg if success_flag else warning)(f"  {tag}  runtime={runtime:.1f} s "
                                          f"(log: {log_path})")

    return {"success": success_flag, "runtime_seconds": runtime,
            "records": monitor.get_records()}


# ===========================================================================
# TSV output
# ===========================================================================

_TSV_HEADER = (
    "json_file", "token_count", "elapsed_seconds",
    "memory_used_mb", "memory_total_mb", "memory_percent",
    "gpu_util", "temperature",
    "success", "job_runtime_seconds",
)


def append_timeseries(out_tsv: Path, json_file: str, tokens: int,
                      records: List[Tuple[float, Dict]],
                      success_flag: bool, runtime: float,
                      write_header: bool) -> None:
    """Append time-series rows for one job to the output TSV."""
    mode = "w" if write_header else "a"
    with open(out_tsv, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if write_header:
            w.writerow(_TSV_HEADER)
        for elapsed, gpu in records:
            w.writerow([
                json_file, tokens, elapsed,
                gpu["memory_used_mb"], gpu["memory_total_mb"],
                gpu["memory_percent"], gpu["gpu_util"], gpu["temperature"],
                success_flag, f"{runtime:.2f}",
            ])


# ===========================================================================
# Signal handling
# ===========================================================================

def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        sig = signal.Signals(signum).name
        warning(f"Received {sig}; killing AF3 subprocess (if any) and exiting.")
        with _CHILD_LOCK:
            child = _CURRENT_CHILD
        if child is not None and child.poll() is None:
            try:
                child.kill()
            except OSError:
                pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ===========================================================================
# CLI / Main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="af3_GPU_memory_timeseries.py",
        description=("AF3 GPU memory time-series profiler. Detects VRAM "
                     "gradient steps from a peak-memory stat file, picks "
                     "representative JSONs per step, runs AF3 sequentially, "
                     "and records sub-second GPU memory curves into a "
                     "single TSV."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stat-file", type=Path, required=True,
                   help="Peak-memory stat TSV with columns "
                        "'token_count' and 'peak_memory_mb'.")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing AF3 input JSON files.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("./timeseries_output"),
                   help="AF3 result output directory.")
    p.add_argument("--output-tsv", type=Path, default=None,
                   help="Output TSV path "
                        "[default: <output-dir>/gpu_memory_timeseries.tsv].")
    p.add_argument("--sif", type=str, required=True,
                   help="Path to AlphaFold3 Singularity image (.sif).")
    p.add_argument("--af3-db", type=str, default="./af3_DB",
                   help="AF3 genetic database directory.")
    p.add_argument("--models", type=str, default="./models",
                   help="AF3 model parameters directory.")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Directory to stream per-job AF3 logs. "
                        "Default: <output-dir>/_af3_logs.")
    p.add_argument("--monitor-interval", type=float, default=0.5,
                   help="GPU sampling interval in seconds.")
    p.add_argument("--n-per-step", type=int, default=3,
                   help="Representative JSON files per gradient step.")
    p.add_argument("--workers", type=int,
                   default=min(8, multiprocessing.cpu_count()),
                   help="Parallel workers for JSON token counting.")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Per-job timeout in seconds (default: 2 hours).")
    p.add_argument(
        "--extra-args", type=str, default="",
        help=("Single quoted string of extra flags forwarded to "
              "run_alphafold.py, parsed with shlex. Example: "
              "--extra-args '--num_recycles=3 --num_seeds=1'. Use Abseil "
              "flag syntax: --norun_data_pipeline (CORRECT) vs "
              "--no-run-data_pipeline (WRONG). If neither --norun_data_pipeline "
              "nor --norun_inference is given, --norun_data_pipeline is "
              "appended automatically."),
    )
    args = p.parse_args()
    if args.monitor_interval <= 0:
        p.error("--monitor-interval must be > 0.")
    if args.n_per_step <= 0:
        p.error("--n-per-step must be > 0.")
    if args.workers <= 0:
        p.error("--workers must be > 0.")
    if args.timeout <= 0:
        p.error("--timeout must be > 0.")
    return args


def main() -> None:
    args = _parse_args()
    _install_signal_handlers()

    # ---- Validation ----
    if not args.stat_file.is_file():
        error(f"Stat file not found: {args.stat_file}")
        sys.exit(1)
    if not args.input_dir.is_dir():
        error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not Path(args.sif).is_file():
        error(f"Singularity image not found: {args.sif}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_tsv is None:
        args.output_tsv = args.output_dir / "gpu_memory_timeseries.tsv"
    log_dir = args.log_dir or (args.output_dir / "_af3_logs")
    extra_args = shlex.split(args.extra_args) if args.extra_args else []

    info("=" * 64)
    info("AF3 GPU Memory Time-Series Profiler")
    info("=" * 64)
    info(f"Stat file         : {args.stat_file}")
    info(f"Input dir         : {args.input_dir}")
    info(f"Output dir        : {args.output_dir}")
    info(f"Output TSV        : {args.output_tsv}")
    info(f"Log dir           : {log_dir}")
    info(f"Monitor interval  : {args.monitor_interval} s")
    info(f"Files per gradient: {args.n_per_step}")
    info(f"Parse workers     : {args.workers}")
    info(f"SIF image         : {args.sif}")
    info(f"Per-job timeout   : {args.timeout} s")
    if extra_args:
        info(f"Extra AF3 flags   : {extra_args}")

    # ---- Tool checks ----
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError(f"nvidia-smi exit={r.returncode}")
        info("nvidia-smi        : OK")
    except (FileNotFoundError, subprocess.TimeoutExpired, RuntimeError) as exc:
        error(f"nvidia-smi unavailable: {exc}")
        sys.exit(1)
    try:
        subprocess.run(["singularity", "--version"],
                       capture_output=True, check=True, timeout=10)
        info("singularity       : OK")
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        error("singularity not available in PATH.")
        sys.exit(1)

    # ---- Step 1: gradient detection ----
    info("=" * 64)
    info("Step 1/4 - Detecting VRAM gradient steps from stat file ...")
    steps = parse_stat_file(args.stat_file)

    # ---- Step 2: parallel token counting ----
    info("=" * 64)
    info("Step 2/4 - Counting tokens in input JSON files (parallel) ...")
    all_json = sorted(args.input_dir.glob("*.json"))
    if not all_json:
        error(f"No JSON files found in {args.input_dir}")
        sys.exit(1)
    info(f"Found {len(all_json)} JSON file(s) - parsing with "
         f"{args.workers} worker(s) ...")

    parsed_all: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_parse_worker, str(p)): p for p in all_json}
        done = 0
        for fut in as_completed(futs):
            parsed_all.append(fut.result())
            done += 1
            if done % 1000 == 0 or done == len(all_json):
                info(f"  Parsed {done}/{len(all_json)} ...")

    bad = [p for p in parsed_all if p["error"]]
    if bad:
        warning(f"  {len(bad)} file(s) failed to parse (showing first 5):")
        for p in bad[:5]:
            warning(f"    {p['json_file']}: {p['error']}")

    parsed_all.sort(key=lambda x: x["tokens"])
    valid = [p for p in parsed_all if p["error"] is None]
    if not valid:
        error("No valid JSON files parsed; aborting.")
        sys.exit(1)
    info(f"  Dataset token range: {valid[0]['tokens']} - {valid[-1]['tokens']}")

    # ---- Step 3: select representatives ----
    info("=" * 64)
    info(f"Step 3/4 - Selecting up to {args.n_per_step} "
         f"representative(s) per gradient ...")
    selected = select_representatives(steps, parsed_all, n=args.n_per_step)
    if not selected:
        error("No files selected. Verify that token ranges in the input "
              "directory overlap with the gradient steps from the stat file.")
        sys.exit(1)

    # ---- Step 4: run AF3 + monitor ----
    info("=" * 64)
    info("Step 4/4 - Running AF3 jobs and recording GPU memory curves ...")

    first_write = True
    total = len(selected)
    for idx, parsed in enumerate(selected, 1):
        info(f"[Job {idx}/{total}] {parsed['json_file']} "
             f"(tokens={parsed['tokens']})")
        result = run_af3_job(
            parsed=parsed,
            sif=args.sif,
            af3_db=args.af3_db,
            models=args.models,
            out_dir=str(args.output_dir),
            extra_args=extra_args,
            monitor_interval=args.monitor_interval,
            timeout_seconds=args.timeout,
            log_dir=log_dir,
        )
        append_timeseries(
            out_tsv=args.output_tsv,
            json_file=parsed["json_file"],
            tokens=parsed["tokens"],
            records=result["records"],
            success_flag=result["success"],
            runtime=result["runtime_seconds"],
            write_header=first_write,
        )
        first_write = False

        n_pts = len(result["records"])
        peak = max((r[1]["memory_used_mb"] for r in result["records"]),
                   default=0)
        ok_msg(f"  {n_pts} data points  |  peak VRAM = {peak} MB  |  "
               f"runtime = {result['runtime_seconds']:.1f} s  |  "
               f"{'OK' if result['success'] else 'FAILED'}")

    # ---- Done ----
    info("=" * 64)
    ok_msg(f"All {total} job(s) complete.")
    ok_msg(f"Time-series TSV: {args.output_tsv}")
    info("Filter rows by 'json_file' to plot each job's VRAM curve "
         "independently.")


if __name__ == "__main__":
    main()
