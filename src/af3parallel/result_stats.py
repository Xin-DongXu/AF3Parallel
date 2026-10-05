#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarise AlphaFold 3 ``*_summary_confidences.json`` outputs.

Modes
-----
**ligand** (default)
    Multi-protein + ligand complexes. Global ``iptm`` mixes interfaces;
    this mode reports ligand–protein pairwise metrics from
    ``chain_pair_iptm`` / ``chain_pair_pae_min``. Default layout: ligand is
    the last entity (``--ligand-index -1``).

**general**
    Any AF3 prediction: collect global ``ptm``, ``iptm``, ``ranking_score``,
    and per-chain ``chain_iptm`` without assuming a ligand.

Recursive scans prune ``seed-*`` directories by default. Parsing uses a
process pool (``-j``).

Examples
--------
  af3parallel stats -i ./af_output -r -o detail.csv -s summary.csv
  af3parallel stats --mode general -i ./af_output -r -o scores.csv
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, unit=None, **kwargs):  # type: ignore
        if iterable is None:
            class _Dummy:
                def __enter__(self):
                    return self

                def __exit__(self, *a):
                    return False

                def update(self, n=1):
                    pass

            return _Dummy()
        return iterable


DEFAULT_PATTERN: str = "*_summary_confidences.json"
DEFAULT_IPTM_HIGH: float = 0.80
DEFAULT_IPTM_MEDIUM: float = 0.60

# Ranking key for confidence_class: max ligand–protein pair ipTM.
RANK_METRIC: str = "iptm_final"


def _default_jobs() -> int:
    """CPUs visible to this process (honours cgroup/SLURM affinity)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 2
    return max(1, n - 1)

DETAIL_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "protein_id",
    "ligand_id",
    "file",
    "status",
    "n_chains",
    "ligand_index",
    "protein_indices",
    "confidence_class",
    # Global (mixed interfaces — kept for reference only)
    "ptm",
    "iptm_global",
    "ranking_score",
    # Ligand–protein (primary); iptm_final = max(P0, P1, ...)
    "iptm_final",
    "iptm_ligand_min",
    "iptm_ligand_mean",
    "iptm_ligand_max",
    "chain_iptm_ligand",
    "pae_ligand_min",
    "pae_ligand_mean",
    # Explicit pairs (A/B proteins vs ligand); empty if <2 proteins
    "iptm_P0_ligand",
    "iptm_P1_ligand",
    "pae_P0_ligand",
    "pae_P1_ligand",
    # Protein–protein (secondary)
    "iptm_protein_protein",
    "pae_protein_protein",
)

GENERAL_COLUMNS: Tuple[str, ...] = (
    "sample_id",
    "file",
    "status",
    "n_chains",
    "confidence_class",
    "ptm",
    "iptm",
    "ranking_score",
    "chain_iptm_max",
    "chain_iptm_mean",
    "chain_iptm_min",
    "fraction_disordered",
    "has_clash",
)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="af3parallel stats",
        description=(
            "Summarise AF3 *_summary_confidences.json "
            "(ligand interface metrics or general scores)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=("ligand", "general"),
        default="ligand",
        help="ligand = pairwise ligand–protein ipTM/PAE; general = global scores",
    )
    p.add_argument("-i", "--input-dir", type=Path, required=True)
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Per-prediction CSV (detail).",
    )
    p.add_argument("-p", "--pattern", type=str, default=DEFAULT_PATTERN)
    p.add_argument("-r", "--recursive", action="store_true")
    p.add_argument(
        "--include-seed-dirs",
        action="store_true",
        help=(
            "Include JSON under intermediate folders named seed-* "
            "(default: skip those AF3 per-sample folders)."
        ),
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=_default_jobs(),
        help=(
            "Worker processes for JSON parse (not threads). "
            "JSON decoding is CPU-bound under the GIL; a thread pool "
            "barely speeds this up."
        ),
    )
    p.add_argument(
        "-s",
        "--summary",
        type=Path,
        default=None,
        help="Optional aggregate summary CSV.",
    )
    p.add_argument(
        "--ligand-index",
        type=int,
        default=-1,
        help=(
            "0-based index of the ligand entity in chain_pair_iptm "
            "(-1 = last chain, typical for protein A/B + ligand C)."
        ),
    )
    p.add_argument("--iptm-high", type=float, default=DEFAULT_IPTM_HIGH)
    p.add_argument("--iptm-medium", type=float, default=DEFAULT_IPTM_MEDIUM)
    p.add_argument(
        "--rank-metric",
        choices=(
            "iptm_final",
            "iptm_ligand_max",
            "iptm_ligand_min",
            "iptm_ligand_mean",
            "chain_iptm_ligand",
            "iptm",
            "ptm",
            "ranking_score",
        ),
        default=RANK_METRIC,
        help="Metric for confidence_class (ligand mode default: iptm_final).",
    )
    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args(argv)
    if not args.input_dir.is_dir():
        p.error(f"--input-dir is not a directory: {args.input_dir}")
    if args.jobs < 1:
        p.error("--jobs must be >= 1")
    if not (0.0 <= args.iptm_medium <= args.iptm_high <= 1.0):
        p.error("Require 0 <= --iptm-medium <= --iptm-high <= 1")
    if args.mode == "general" and args.rank_metric in (
        "iptm_final",
        "iptm_ligand_max",
        "iptm_ligand_min",
        "iptm_ligand_mean",
        "chain_iptm_ligand",
    ):
        args.rank_metric = "iptm"
    return args


def find_json_files(
    directory: Path,
    pattern: str,
    recursive: bool,
    skip_seed_dirs: bool = True,
) -> List[Path]:
    """Collect matching JSON paths.

    When ``skip_seed_dirs`` is set, ``seed-*`` directories are pruned
    during ``os.walk`` so the scanner never descends into AF3 per-sample
    folders.  ``Path.rglob`` would still walk those trees and only filter
    afterwards — that is the usual reason this script feels hung before
    any progress bar appears.
    """
    root = Path(directory)
    if not recursive:
        return sorted(p for p in root.glob(pattern) if p.is_file())

    files: List[Path] = []
    n_pruned = 0
    t0 = time.perf_counter()
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        if skip_seed_dirs:
            keep: List[str] = []
            for d in dirnames:
                if d.startswith("seed-"):
                    n_pruned += 1
                else:
                    keep.append(d)
            dirnames[:] = keep
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                files.append(Path(dirpath) / name)
    logging.info(
        "Scan finished in %.2fs (%d file(s), pruned %d seed-* dir(s))",
        time.perf_counter() - t0,
        len(files),
        n_pruned,
    )
    return sorted(files)

def _classify(val: Optional[float], high: float, medium: float) -> str:
    if val is None:
        return "Unknown"
    if val >= high:
        return "High"
    if val >= medium:
        return "Medium"
    return "Low"


def _split_sample_id(sample_id: str) -> Tuple[str, str]:
    if not sample_id:
        return "", ""
    parts = sample_id.split("_", 2)
    protein = parts[0].upper() if parts and parts[0] else ""
    ligand = parts[1].upper() if len(parts) >= 2 and parts[1] else ""
    return protein, ligand


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _matrix_get(
    matrix: Optional[Sequence[Sequence[Any]]],
    i: int,
    j: int,
) -> Optional[float]:
    if not matrix:
        return None
    try:
        return _as_float(matrix[i][j])
    except (IndexError, TypeError):
        return None


def _pair_sym(
    matrix: Optional[Sequence[Sequence[Any]]],
    i: int,
    j: int,
) -> Optional[float]:
    """Prefer average of (i,j) and (j,i) when both present; else whichever exists."""
    a = _matrix_get(matrix, i, j)
    b = _matrix_get(matrix, j, i)
    if a is not None and b is not None:
        return 0.5 * (a + b)
    return a if a is not None else b


def _resolve_ligand_index(n_chains: int, ligand_index: int) -> int:
    if n_chains < 2:
        raise ValueError(f"need >=2 chains, got {n_chains}")
    idx = ligand_index if ligand_index >= 0 else n_chains + ligand_index
    if not (0 <= idx < n_chains):
        raise ValueError(f"ligand index {ligand_index} invalid for n_chains={n_chains}")
    return idx


def extract_metrics(
    json_path_str: str,
    ligand_index: int = -1,
    iptm_high: float = DEFAULT_IPTM_HIGH,
    iptm_medium: float = DEFAULT_IPTM_MEDIUM,
    rank_metric: str = RANK_METRIC,
) -> Dict[str, Any]:
    json_path = Path(json_path_str)
    record: Dict[str, Any] = {col: "" for col in DETAIL_COLUMNS}
    record["file"] = json_path.name
    record["sample_id"] = json_path.name.replace("_summary_confidences.json", "")
    record["protein_id"], record["ligand_id"] = _split_sample_id(record["sample_id"])
    record["status"] = "OK"

    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            record["status"] = "MALFORMED:not_an_object"
            return record

        pair_iptm = data.get("chain_pair_iptm") or []
        pair_pae = data.get("chain_pair_pae_min") or []
        chain_iptm = data.get("chain_iptm") or []

        n_chains = len(pair_iptm) if pair_iptm else len(chain_iptm)
        record["n_chains"] = n_chains
        if n_chains < 2:
            record["status"] = "MALFORMED:need_>=2_chains"
            return record

        lig_i = _resolve_ligand_index(n_chains, ligand_index)
        protein_idx = [i for i in range(n_chains) if i != lig_i]
        record["ligand_index"] = lig_i
        record["protein_indices"] = ",".join(str(i) for i in protein_idx)

        record["ptm"] = data.get("ptm")
        record["iptm_global"] = data.get("iptm")
        record["ranking_score"] = data.get("ranking_score")

        lig_prot_iptm: List[float] = []
        lig_prot_pae: List[float] = []
        pair_cols_iptm = ("iptm_P0_ligand", "iptm_P1_ligand")
        pair_cols_pae = ("pae_P0_ligand", "pae_P1_ligand")
        for k, pi in enumerate(protein_idx):
            iptm_lp = _pair_sym(pair_iptm, pi, lig_i)
            pae_lp = _pair_sym(pair_pae, pi, lig_i)
            if iptm_lp is not None:
                lig_prot_iptm.append(iptm_lp)
            if pae_lp is not None:
                lig_prot_pae.append(pae_lp)
            if k < 2:
                record[pair_cols_iptm[k]] = iptm_lp
                record[pair_cols_pae[k]] = pae_lp

        record["iptm_ligand_min"] = min(lig_prot_iptm) if lig_prot_iptm else None
        record["iptm_ligand_mean"] = (
            sum(lig_prot_iptm) / len(lig_prot_iptm) if lig_prot_iptm else None
        )
        record["iptm_ligand_max"] = max(lig_prot_iptm) if lig_prot_iptm else None
        # Final ipTM for screening: strongest ligand–protein interface.
        record["iptm_final"] = record["iptm_ligand_max"]
        record["pae_ligand_min"] = min(lig_prot_pae) if lig_prot_pae else None
        record["pae_ligand_mean"] = (
            sum(lig_prot_pae) / len(lig_prot_pae) if lig_prot_pae else None
        )

        if isinstance(chain_iptm, list) and lig_i < len(chain_iptm):
            record["chain_iptm_ligand"] = _as_float(chain_iptm[lig_i])
        else:
            record["chain_iptm_ligand"] = None

        if len(protein_idx) >= 2:
            record["iptm_protein_protein"] = _pair_sym(
                pair_iptm, protein_idx[0], protein_idx[1]
            )
            record["pae_protein_protein"] = _pair_sym(
                pair_pae, protein_idx[0], protein_idx[1]
            )
        else:
            record["iptm_protein_protein"] = None
            record["pae_protein_protein"] = None

        rank_val = record.get(rank_metric)
        record["confidence_class"] = _classify(
            _as_float(rank_val),
            iptm_high,
            iptm_medium,
        )
        return record

    except FileNotFoundError:
        record["status"] = "FILE_NOT_FOUND"
        return record
    except json.JSONDecodeError as exc:
        record["status"] = f"JSON_ERROR:{exc.msg}"
        return record
    except ValueError as exc:
        record["status"] = f"BAD_LIGAND_INDEX:{exc}"
        return record
    except OSError as exc:
        record["status"] = f"IO_ERROR:{exc.strerror or exc}"
        return record
    except Exception as exc:  # noqa: BLE001
        record["status"] = f"ERROR:{type(exc).__name__}:{exc}"
        return record


def extract_metrics_general(
    json_path_str: str,
    iptm_high: float = DEFAULT_IPTM_HIGH,
    iptm_medium: float = DEFAULT_IPTM_MEDIUM,
    rank_metric: str = "iptm",
) -> Dict[str, Any]:
    json_path = Path(json_path_str)
    record: Dict[str, Any] = {col: "" for col in GENERAL_COLUMNS}
    record["file"] = json_path.name
    record["sample_id"] = json_path.name.replace("_summary_confidences.json", "")
    record["status"] = "OK"
    try:
        with json_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            record["status"] = "MALFORMED:not_an_object"
            return record
        chain_iptm = data.get("chain_iptm") or []
        record["n_chains"] = len(chain_iptm) if chain_iptm else 0
        record["ptm"] = data.get("ptm")
        record["iptm"] = data.get("iptm")
        record["ranking_score"] = data.get("ranking_score")
        record["fraction_disordered"] = data.get("fraction_disordered")
        record["has_clash"] = data.get("has_clash")
        vals = [_as_float(x) for x in chain_iptm]
        vals = [v for v in vals if v is not None]
        if vals:
            record["chain_iptm_max"] = max(vals)
            record["chain_iptm_min"] = min(vals)
            record["chain_iptm_mean"] = sum(vals) / len(vals)
        rank_val = record.get(rank_metric)
        record["confidence_class"] = _classify(
            _as_float(rank_val), iptm_high, iptm_medium
        )
        return record
    except FileNotFoundError:
        record["status"] = "FILE_NOT_FOUND"
        return record
    except json.JSONDecodeError as exc:
        record["status"] = f"JSON_ERROR:{exc.msg}"
        return record
    except Exception as exc:  # noqa: BLE001
        record["status"] = f"ERROR:{type(exc).__name__}:{exc}"
        return record


class _ParseJob(NamedTuple):
    path: str
    mode: str
    ligand_index: int
    iptm_high: float
    iptm_medium: float
    rank_metric: str


def _process_one(job: _ParseJob) -> Dict[str, Any]:
    """Top-level worker so ProcessPoolExecutor can pickle it (Windows spawn)."""
    try:
        if job.mode == "general":
            return extract_metrics_general(
                job.path,
                iptm_high=job.iptm_high,
                iptm_medium=job.iptm_medium,
                rank_metric=job.rank_metric,
            )
        return extract_metrics(
            job.path,
            ligand_index=job.ligand_index,
            iptm_high=job.iptm_high,
            iptm_medium=job.iptm_medium,
            rank_metric=job.rank_metric,
        )
    except Exception as exc:  # noqa: BLE001
        p = Path(job.path)
        sid = p.name.replace("_summary_confidences.json", "")
        if job.mode == "general":
            return {
                **{c: "" for c in GENERAL_COLUMNS},
                "file": p.name,
                "sample_id": sid,
                "status": f"WORKER_ERROR:{type(exc).__name__}:{exc}",
            }
        prot, lig = _split_sample_id(sid)
        return {
            **{c: "" for c in DETAIL_COLUMNS},
            "file": p.name,
            "sample_id": sid,
            "protein_id": prot,
            "ligand_id": lig,
            "status": f"WORKER_ERROR:{type(exc).__name__}:{exc}",
        }


def _chunksize(n_items: int, jobs: int) -> int:
    """Batch many small JSON files per IPC round-trip."""
    if n_items <= 0:
        return 1
    return max(8, min(256, n_items // max(1, jobs * 4) or 8))


def process_files(
    paths: Sequence[Path],
    jobs: int,
    ligand_index: int,
    iptm_high: float,
    iptm_medium: float,
    rank_metric: str,
    mode: str = "ligand",
) -> List[Dict[str, Any]]:
    """Parse JSON files with a process pool (not threads).

    A ThreadPoolExecutor cannot overlap ``json.load`` in CPython: the GIL
    is held while Python objects are built, so extra threads look parallel
    in the log but run almost serially.  Submitting one Future per file
    also blows up at 10k+ paths.  ``Executor.map`` + chunksize keeps IPC
    coarse-grained.
    """
    use_parallel = jobs > 1 and len(paths) > 1
    desc = f"Processing ({'process x' + str(jobs) if use_parallel else 'serial'})"
    records: List[Dict[str, Any]] = []

    if not use_parallel:
        for p in tqdm(paths, desc=desc, unit="file"):
            if mode == "general":
                records.append(
                    extract_metrics_general(
                        str(p),
                        iptm_high=iptm_high,
                        iptm_medium=iptm_medium,
                        rank_metric=rank_metric,
                    )
                )
            else:
                records.append(
                    extract_metrics(
                        str(p),
                        ligand_index=ligand_index,
                        iptm_high=iptm_high,
                        iptm_medium=iptm_medium,
                        rank_metric=rank_metric,
                    )
                )
    else:
        jobs_list = [
            _ParseJob(
                path=str(p),
                mode=mode,
                ligand_index=ligand_index,
                iptm_high=iptm_high,
                iptm_medium=iptm_medium,
                rank_metric=rank_metric,
            )
            for p in paths
        ]
        cs = _chunksize(len(jobs_list), jobs)
        logging.info(
            "Process pool: %d workers, chunksize=%d, %d file(s)",
            jobs,
            cs,
            len(jobs_list),
        )
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for rec in tqdm(
                ex.map(_process_one, jobs_list, chunksize=cs),
                total=len(jobs_list),
                desc=desc,
                unit="file",
            ):
                records.append(rec)

    records.sort(key=lambda r: r.get("file", ""))
    return records


def _fmt(val: Any) -> str:
    if val is None or val == "":
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def write_detailed_csv(
    records: Sequence[Dict[str, Any]],
    output: Path,
    columns: Sequence[str] = DETAIL_COLUMNS,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({c: _fmt(rec.get(c, "")) for c in columns})
    logging.info("Detail CSV written: %s", output)


class MetricSummary(NamedTuple):
    name: str
    n: int
    mean: float
    median: float
    stdev: float
    minimum: float
    q1: float
    q3: float
    maximum: float


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summarise(name: str, values: Sequence[Any]) -> Optional[MetricSummary]:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    s = sorted(clean)
    sd = 0.0 if len(s) < 2 else statistics.stdev(s)
    return MetricSummary(
        name=name,
        n=len(s),
        mean=sum(s) / len(s),
        median=statistics.median(s),
        stdev=sd,
        minimum=s[0],
        q1=_quantile(s, 0.25),
        q3=_quantile(s, 0.75),
        maximum=s[-1],
    )


def build_summary(
    records: Sequence[Dict[str, Any]],
    iptm_high: float,
    iptm_medium: float,
    rank_metric: str,
    mode: str = "ligand",
) -> Dict[str, Any]:
    ok = [r for r in records if r.get("status") == "OK"]
    failed = [r for r in records if r.get("status") != "OK"]
    if mode == "general":
        metric_keys = (
            "ptm",
            "iptm",
            "ranking_score",
            "chain_iptm_max",
            "chain_iptm_mean",
            "chain_iptm_min",
            "fraction_disordered",
        )
    else:
        metric_keys = (
            "ptm",
            "iptm_global",
            "iptm_final",
            "iptm_ligand_min",
            "iptm_ligand_mean",
            "iptm_ligand_max",
            "chain_iptm_ligand",
            "pae_ligand_min",
            "pae_ligand_mean",
            "iptm_P0_ligand",
            "iptm_P1_ligand",
            "iptm_protein_protein",
            "pae_protein_protein",
            "ranking_score",
        )
    metric_stats = [
        m
        for m in (
            _summarise(k, [r.get(k) for r in ok]) for k in metric_keys
        )
        if m is not None
    ]
    classes = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}
    for r in ok:
        classes[r.get("confidence_class", "Unknown")] = (
            classes.get(r.get("confidence_class", "Unknown"), 0) + 1
        )
    return {
        "mode": mode,
        "total": len(records),
        "ok": len(ok),
        "failed": len(failed),
        "failed_records": failed,
        "metric_stats": metric_stats,
        "class_counts": classes,
        "iptm_high": iptm_high,
        "iptm_medium": iptm_medium,
        "rank_metric": rank_metric,
    }


def write_summary_csv(summary: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value"])
        w.writerow(["meta", "total", summary["total"]])
        w.writerow(["meta", "ok", summary["ok"]])
        w.writerow(["meta", "failed", summary["failed"]])
        w.writerow(["meta", "rank_metric", summary["rank_metric"]])
        w.writerow(["meta", "iptm_high", summary["iptm_high"]])
        w.writerow(["meta", "iptm_medium", summary["iptm_medium"]])
        for k in ("High", "Medium", "Low", "Unknown"):
            w.writerow(["confidence_class", k, summary["class_counts"].get(k, 0)])
        w.writerow([])
        w.writerow(
            ["metric", "n", "mean", "median", "stdev", "min", "q1", "q3", "max"]
        )
        for m in summary["metric_stats"]:
            w.writerow(
                [
                    m.name,
                    m.n,
                    f"{m.mean:.4f}",
                    f"{m.median:.4f}",
                    f"{m.stdev:.4f}",
                    f"{m.minimum:.4f}",
                    f"{m.q1:.4f}",
                    f"{m.q3:.4f}",
                    f"{m.maximum:.4f}",
                ]
            )
        if summary["failed_records"]:
            w.writerow([])
            w.writerow(["failed_file", "status"])
            for r in summary["failed_records"]:
                w.writerow([r.get("file", ""), r.get("status", "")])
    logging.info("Summary CSV written: %s", path)


def print_console_summary(summary: Dict[str, Any]) -> None:
    line = "=" * 64
    print(line, file=sys.stderr)
    print(
        f" AF3 stats ({summary.get('mode', 'ligand')})  | total={summary['total']}  "
        f"ok={summary['ok']}  failed={summary['failed']}",
        file=sys.stderr,
    )
    print(
        f" rank_metric={summary['rank_metric']}  "
        f"High>={summary['iptm_high']}  Medium>={summary['iptm_medium']}",
        file=sys.stderr,
    )
    cc = summary["class_counts"]
    print(
        f" Confidence          | High={cc.get('High', 0)}  "
        f"Medium={cc.get('Medium', 0)}  Low={cc.get('Low', 0)}  "
        f"Unknown={cc.get('Unknown', 0)}",
        file=sys.stderr,
    )
    highlight = (
        "iptm",
        "ptm",
        "iptm_final",
        "iptm_ligand_max",
        "iptm_ligand_min",
        "iptm_ligand_mean",
        "chain_iptm_ligand",
        "iptm_protein_protein",
        "iptm_global",
        "ranking_score",
    )
    for m in summary["metric_stats"]:
        if m.name in highlight:
            print(
                f" {m.name:<22}| n={m.n}  mean={m.mean:.3f}  "
                f"median={m.median:.3f}  min={m.minimum:.3f}  max={m.maximum:.3f}",
                file=sys.stderr,
            )
    print(line, file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    setup_logging(args.verbose)
    t0 = time.perf_counter()

    logging.info(
        "Scanning %s (mode=%s, pattern=%s, recursive=%s, skip_seed_dirs=%s)",
        args.input_dir,
        args.mode,
        args.pattern,
        args.recursive,
        not args.include_seed_dirs,
    )
    paths = find_json_files(
        args.input_dir,
        args.pattern,
        args.recursive,
        skip_seed_dirs=not args.include_seed_dirs,
    )
    if not paths:
        logging.error("No files matching %r in %s", args.pattern, args.input_dir)
        return 1
    logging.info("Found %d file(s); workers=%d", len(paths), args.jobs)

    records = process_files(
        paths,
        args.jobs,
        args.ligand_index,
        args.iptm_high,
        args.iptm_medium,
        args.rank_metric,
        mode=args.mode,
    )
    cols = GENERAL_COLUMNS if args.mode == "general" else DETAIL_COLUMNS
    write_detailed_csv(records, args.output, columns=cols)

    summary = build_summary(
        records,
        args.iptm_high,
        args.iptm_medium,
        args.rank_metric,
        mode=args.mode,
    )
    if args.summary is not None:
        write_summary_csv(summary, args.summary)
    print_console_summary(summary)

    logging.info("Done in %.2fs.", time.perf_counter() - t0)
    return 0 if summary["failed"] == 0 else 3


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())