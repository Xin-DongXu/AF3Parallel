#!/usr/bin/env python3
"""
AlphaFold3 GPU Memory Profiler -- Peak-memory scan
==================================================

For each AF3 input JSON in a directory, runs AlphaFold3 (via Singularity)
and records the peak GPU memory observed during the run, together with
runtime and metadata. One representative file is kept per unique AF3 token
count, then jobs are executed from smallest to largest.

The token counter follows the official AF3 specification (Abramson et al.,
Nature 2024) and is bit-identical to the one in
af3_GPU_memory_timeseries.py so that gradient buckets selected from this
script's output remain consistent in the time-series profiler.

Token counting rules
--------------------
* protein / RNA / DNA: 1 token per residue (len(sequence)),
  multiplied by the number of copies (length of the id list, or 1 if
  id is a string).
* ligand with smiles: 1 token per heavy atom, computed via a small
  SMILES tokenizer that supports aromatic lowercase atoms and two-letter
  elements (Cl, Br) outside brackets.
* ligand with ccdCodes: sum of heavy atoms per code from a built-in
  CCD table (fallback _CCD_FALLBACK for unknown codes).

CLI
---
Run ./af3_GPU_stat_v2.py --help for full options.

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
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
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


def info(msg: str) -> None:    _emit("INFO",    msg, _Colors.BLUE)
def warning(msg: str) -> None: _emit("WARN",    msg, _Colors.YELLOW)
def error(msg: str) -> None:   _emit("ERROR",   msg, _Colors.RED)
def success(msg: str) -> None: _emit("SUCCESS", msg, _Colors.GREEN)


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
# Per-file metadata extraction
# ===========================================================================

def parse_json_file(json_path: Path) -> Optional[Dict]:
    """Parse one AF3 input JSON and return token count + metadata.

    Returns None and logs a warning if the file cannot be parsed.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warning(f"Failed to parse {json_path.name}: {exc}")
        return None

    sequences = data.get("sequences", []) or []
    protein_residues = 0
    ligand_count = 0
    for seq in sequences:
        if not isinstance(seq, dict):
            continue
        if "protein" in seq:
            protein_residues += len(seq["protein"].get("sequence", "")) * \
                _entity_copies(seq["protein"])
        elif "ligand" in seq:
            ligand_count += _entity_copies(seq["ligand"])

    return {
        "name": data.get("name", json_path.stem),
        "token_count": af3_count_tokens(data),
        "protein_length": protein_residues,
        "ligand_count": ligand_count,
        "total_sequences": len(sequences),
        "dialect": data.get("dialect", "unknown"),
        "version": data.get("version", 0),
    }


def collect_json_files(input_dir: Path,
                       keep_all: bool = False
                       ) -> List[Tuple[Path, Dict]]:
    """Scan input_dir for AF3 input JSONs and return them sorted.

    By default, when several files share the same token count only the first
    one encountered (lexicographic order) is kept. Pass keep_all=True to
    keep every file.
    """
    info(f"Scanning JSON files in: {input_dir}")
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        error(f"No JSON files found in {input_dir}")
        sys.exit(1)
    info(f"Found {len(json_files)} JSON file(s)")

    seen_tokens: Dict[int, Path] = {}
    kept: List[Tuple[Path, Dict]] = []
    skipped = 0
    for jf in json_files:
        meta = parse_json_file(jf)
        if meta is None:
            continue
        tc = meta["token_count"]
        if not keep_all and tc in seen_tokens:
            skipped += 1
            continue
        seen_tokens[tc] = jf
        kept.append((jf, meta))
        info(f"  [{tc:>5} tokens] {jf.name}  "
             f"name={meta['name']!r} prot={meta['protein_length']} "
             f"lig={meta['ligand_count']}")

    if not kept:
        error("No valid JSON files found.")
        sys.exit(1)
    if skipped and not keep_all:
        info(f"Skipped {skipped} duplicate-token-count file(s) "
             f"(use --keep-all to keep them).")

    kept.sort(key=lambda pair: pair[1]["token_count"])
    info(f"Selected {len(kept)} file(s) for profiling.")
    return kept


# ===========================================================================
# GPU monitor (background thread, interruptible wait)
# ===========================================================================

class GPUMonitor:
    """Sample nvidia-smi at a fixed interval in a daemon thread.

    The polling thread uses Event.wait so stop() returns promptly.
    """

    _QUERY = ("memory.used,memory.total,utilization.gpu,temperature.gpu")

    def __init__(self, interval: float = 1.0) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.interval = float(interval)
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._lock = threading.Lock()
        self.max_memory = 0
        self.samples: List[Dict] = []

    @staticmethod
    def check_nvidia_smi() -> bool:
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True,
                               text=True, timeout=10)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _poll(self) -> Dict:
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 f"--query-gpu={self._QUERY}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                # If multiple GPUs are visible, take the first row.
                first = r.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first.split(",")]
                if len(parts) == 4:
                    mu = int(parts[0]); mt = int(parts[1])
                    return {
                        "memory_used": mu,
                        "memory_total": mt,
                        "memory_percent": (mu / mt * 100.0) if mt else 0.0,
                        "gpu_util": int(parts[2]),
                        "temperature": int(parts[3]),
                    }
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return {"memory_used": 0, "memory_total": 0, "memory_percent": 0.0,
                "gpu_util": 0, "temperature": 0}

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._poll()
            with self._lock:
                if sample["memory_used"] > self.max_memory:
                    self.max_memory = sample["memory_used"]
                self.samples.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **sample,
                })
            self._stop.wait(timeout=self.interval)

    def start(self) -> None:
        with self._lock:
            self.max_memory = 0
            self.samples = []
        self._stop.clear()
        self._thread = Thread(target=self._loop, daemon=True,
                              name="GPUMonitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)


# ===========================================================================
# AF3 invocation (Singularity)
# ===========================================================================

# Track currently-running AF3 subprocess so SIGINT/SIGTERM can kill it.
_CHILD_LOCK = threading.Lock()
_CURRENT_CHILD: Optional[subprocess.Popen] = None


def _build_af3_cmd(json_file: Path, sif: str, af3_db: str, models: str,
                   output_dir: str) -> List[str]:
    """Construct the singularity exec command for one AF3 job."""
    return [
        "singularity", "exec", "--nv",
        "--bind", f"{af3_db}:/root/public_databases",
        "--bind", f"{json_file.parent.resolve()}:/root/af_input",
        "--bind", f"{output_dir}:/root/af_output",
        "--bind", f"{models}:/root/models",
        sif,
        "python", "run_alphafold.py",
        f"--json_path=/root/af_input/{json_file.name}",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output",
        "--norun_data_pipeline",
    ]


def run_alphafold3_prediction(json_file: Path, sif: str, af3_db: str,
                              models: str, output_dir: str,
                              timeout_seconds: int,
                              log_dir: Optional[Path]
                              ) -> Tuple[bool, float]:
    """Run one AF3 job. Streams stdout/stderr to a log file (no in-memory
    buffering of the entire output) and returns (success, runtime)."""
    global _CURRENT_CHILD
    cmd = _build_af3_cmd(json_file, sif, af3_db, models, output_dir)
    info(f"Running: {' '.join(cmd)}")

    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{json_file.stem}.log"

    start = time.time()
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
            warning(f"Timeout after {timeout_seconds} s; killing AF3.")
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
    runtime = time.time() - start

    ok = (rc == 0)
    if not ok:
        warning(f"AF3 exited with code {rc} (log: {log_path})")
    else:
        info(f"AF3 succeeded in {runtime:.1f} s")
    return ok, runtime


# ===========================================================================
# TSV output
# ===========================================================================

_TSV_HEADER = [
    "json_file", "protein_name", "token_count", "protein_length",
    "ligand_count", "total_sequences", "peak_memory_mb",
    "runtime_seconds", "success", "timestamp",
]


def write_results_header(output_file: Path) -> None:
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerow(_TSV_HEADER)


def append_result(output_file: Path, json_file: Path, meta: Dict,
                  peak_memory: int, runtime: float, ok: bool) -> None:
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        csv.writer(f, delimiter="\t").writerow([
            json_file.name,
            meta["name"],
            meta["token_count"],
            meta["protein_length"],
            meta["ligand_count"],
            meta["total_sequences"],
            peak_memory,
            f"{runtime:.2f}",
            ok,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
# Main
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Monitor GPU memory while running AlphaFold3 on JSON "
                     "inputs and write a TSV summary."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s -i ./af_input -o results.tsv \\
           --sif alphafold3.sif --af3-db ./af3_DB --models ./models
""",
    )
    p.add_argument("-i", "--input-dir", type=Path, required=True,
                   help="Directory containing AF3 JSON input files.")
    p.add_argument("-o", "--output-file", type=Path, required=True,
                   help="Output TSV file for peak-memory results.")
    p.add_argument("--sif", "--singularity-image", dest="sif", type=str,
                   required=True,
                   help="Path to AlphaFold3 Singularity image (.sif).")
    p.add_argument("--af3-db", type=str, default="./af3_DB",
                   help="AF3 genetic database directory.")
    p.add_argument("--models", type=str, default="./models",
                   help="AF3 model parameters directory.")
    p.add_argument("--output-dir", type=str, default="./af_output",
                   help="AF3 result output directory (inside the container "
                        "this is bound to /root/af_output).")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Directory to stream per-job AF3 stdout/stderr logs. "
                        "Default: <output-dir>/_af3_logs.")
    p.add_argument("--interval", type=float, default=1.0,
                   help="GPU sampling interval in seconds.")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Per-job timeout in seconds (default: 2 hours).")
    p.add_argument("--keep-all", action="store_true",
                   help="Keep all input JSONs even when several share the "
                        "same token count (default: keep one per token "
                        "count).")
    p.add_argument("--inter-job-sleep", type=float, default=5.0,
                   help="Seconds to wait between jobs to let GPU memory "
                        "settle (default: 5).")
    args = p.parse_args()

    if args.interval <= 0:
        p.error("--interval must be > 0.")
    if args.timeout <= 0:
        p.error("--timeout must be > 0.")
    if args.inter_job_sleep < 0:
        p.error("--inter-job-sleep must be >= 0.")
    return args


def main() -> None:
    args = _parse_args()
    _install_signal_handlers()

    # ---- Validation ----
    if not args.input_dir.is_dir():
        error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not Path(args.sif).is_file():
        error(f"Singularity image not found: {args.sif}")
        sys.exit(1)

    af3_db_path = os.path.abspath(args.af3_db)
    models_path = os.path.abspath(args.models)
    output_path = os.path.abspath(args.output_dir)
    sif_path = os.path.abspath(args.sif)

    for path, label in [(af3_db_path, "AF3 database"),
                        (models_path, "Models")]:
        if not os.path.exists(path):
            error(f"{label} directory not found: {path}")
            sys.exit(1)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    log_dir = args.log_dir or (Path(output_path) / "_af3_logs")

    if not GPUMonitor.check_nvidia_smi():
        error("nvidia-smi is not available or no GPU is accessible.")
        sys.exit(1)
    try:
        subprocess.run(["singularity", "--version"],
                       capture_output=True, check=True, timeout=10)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        error("singularity not available in PATH.")
        sys.exit(1)

    info("AlphaFold3 GPU memory profiler")
    info(f"  input dir    : {args.input_dir}")
    info(f"  output TSV   : {args.output_file}")
    info(f"  Singularity  : {sif_path}")
    info(f"  AF3 database : {af3_db_path}")
    info(f"  models       : {models_path}")
    info(f"  AF3 outputs  : {output_path}")
    info(f"  log dir      : {log_dir}")
    info(f"  interval     : {args.interval} s")
    info(f"  per-job TO   : {args.timeout} s")

    # ---- Collect ----
    selected = collect_json_files(args.input_dir, keep_all=args.keep_all)

    # ---- Run ----
    monitor = GPUMonitor(args.interval)
    write_results_header(args.output_file)
    info(f"Starting predictions for {len(selected)} file(s) "
         f"in ascending token order...")

    for i, (json_file, meta) in enumerate(selected, 1):
        info(f"--- [{i}/{len(selected)}] {json_file.name} "
             f"(tokens={meta['token_count']}, prot={meta['protein_length']}, "
             f"lig={meta['ligand_count']}) ---")

        monitor.start()
        try:
            ok, runtime = run_alphafold3_prediction(
                json_file=json_file,
                sif=sif_path,
                af3_db=af3_db_path,
                models=models_path,
                output_dir=output_path,
                timeout_seconds=args.timeout,
                log_dir=log_dir,
            )
        finally:
            monitor.stop()

        peak = monitor.max_memory
        append_result(args.output_file, json_file, meta,
                      peak, runtime, ok)
        status = "SUCCESS" if ok else "FAILED"
        success(f"{json_file.name}  peak={peak} MB  "
                f"runtime={runtime:.1f} s  status={status}")

        if args.inter_job_sleep > 0 and i < len(selected):
            time.sleep(args.inter_job_sleep)

    success(f"All predictions completed. Results: {args.output_file}")


if __name__ == "__main__":
    main()
