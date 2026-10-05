#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel AF3 data-pipeline (MSA / templates) on CPU nodes.

Runs ``run_alphafold.py --norun_inference`` via Singularity/Apptainer with a
fixed concurrency pool. When a job finishes, the next pending JSON is started
immediately (thread-pool gap filling).

Default workers = ``max(1, cpu_count // 4)`` (overridable with ``--workers``).

Alternative MSA route (GPU, not wrapped here)
---------------------------------------------
ColabFold / MMseqs2 search is often faster on GPUs. Install and follow:

  https://github.com/sokrypton/ColabFold
  https://github.com/steineggerlab/colabfold

Then convert ColabFold outputs into AF3 JSON (or place MSA strings into the
``unpairedMsa`` / ``pairedMsa`` fields) before ``af3parallel run
--norun-data-pipeline``.

Example
-------
  af3parallel msa \\
      --input-dir ./json_seq_only \\
      --output-dir ./msa_af_output \\
      --harvest-dir ./json_with_msa \\
      --sif alphafold3.sif \\
      --af3-db ~/af3_DB --models ./models --af3-home ~/alphafold3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def default_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        n = os.cpu_count() or 4
    return max(1, int(n) // 4)


def sanitize_job_name(name: str) -> str:
    return (
        name.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    )


def json_has_msa(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for item in data.get("sequences") or []:
        if not isinstance(item, dict):
            continue
        prot = item.get("protein")
        if isinstance(prot, dict) and (
            prot.get("unpairedMsa") or prot.get("pairedMsa")
        ):
            return True
    return False


def job_output_dir(output_root: Path, json_path: Path) -> Path:
    try:
        name = json.loads(json_path.read_text(encoding="utf-8")).get("name")
    except Exception:
        name = None
    if not name:
        name = json_path.stem
    return output_root / sanitize_job_name(str(name))


def is_job_done(output_root: Path, json_path: Path) -> bool:
    if json_has_msa(json_path):
        return True
    jdir = job_output_dir(output_root, json_path)
    if not jdir.is_dir():
        return False
    for p in jdir.rglob("*.json"):
        low = p.name.lower()
        if "confidence" in low or "summary" in low:
            continue
        if low.endswith("_data.json") or json_has_msa(p):
            return True
    try:
        size = sum(f.stat().st_size for f in jdir.rglob("*") if f.is_file())
        if size > 10_000:
            return True
    except OSError:
        pass
    return False


def summarize_log(text: str, limit: int = 400) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    skip_re = re.compile(
        r"SINGULARITYENV_|APPTAINERENV_|INFO:\s+Environment variable",
        re.I,
    )
    useful = [ln for ln in lines if not skip_re.search(ln)]
    hit = [
        ln
        for ln in useful
        if re.search(
            r"Error|Exception|Traceback|FAILED|Unable|not found|CUDA|JAX|Aborted",
            ln,
            re.I,
        )
    ]
    pick = hit[-12:] if hit else useful[-12:]
    out = " | ".join(pick) if pick else (useful[-1] if useful else "no-stderr")
    return out[-limit:]


def find_runtime() -> str:
    for name in ("singularity", "apptainer"):
        if shutil.which(name):
            return name
    raise SystemExit("neither singularity nor apptainer found in PATH")


def run_one(
    json_path: str,
    sif: str,
    af3_db: str,
    models: str,
    output_root: str,
    af3_home: str,
    log_dir: str,
    timeout: int,
    use_nv: bool,
    runtime: str,
) -> Tuple[str, bool, float, str]:
    jp = Path(json_path)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{jp.stem}.log"

    input_dir = str(jp.parent.resolve())
    af3_home_abs = str(Path(af3_home).resolve())
    run_py = str((Path(af3_home) / "run_alphafold.py").resolve())

    cmd: List[str] = [runtime, "exec"]
    if use_nv:
        cmd.append("--nv")
    cmd += [
        "--pwd",
        af3_home_abs,
        "--bind",
        f"{af3_home_abs}:{af3_home_abs}",
        "--bind",
        f"{af3_db}:/root/public_databases",
        "--bind",
        f"{input_dir}:/root/af_input",
        "--bind",
        f"{out_root.resolve()}:/root/af_output",
        "--bind",
        f"{models}:/root/models",
        "--env",
        "JAX_PLATFORMS=cpu",
        "--env",
        "CUDA_VISIBLE_DEVICES=",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=",
        "--env",
        "TF_CPP_MIN_LOG_LEVEL=3",
        "--env",
        "XLA_PYTHON_CLIENT_PREALLOCATE=false",
        sif,
        "python",
        run_py,
        f"--json_path=/root/af_input/{jp.name}",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output",
        "--norun_inference",
    ]

    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=af3_home_abs,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={
                **os.environ,
                "JAX_PLATFORMS": "cpu",
                "CUDA_VISIBLE_DEVICES": "",
                "NVIDIA_VISIBLE_DEVICES": "",
            },
        )
        dt = time.monotonic() - t0
        blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
        log_path.write_text(
            f"# cmd: {' '.join(cmd)}\n# rc={proc.returncode} wall={dt:.1f}s\n\n"
            + blob,
            encoding="utf-8",
            errors="replace",
        )
        ok = proc.returncode == 0 and is_job_done(out_root, jp)
        msg = (
            ""
            if ok
            else f"rc={proc.returncode} {summarize_log(blob)}  log={log_path}"
        )
        return jp.name, ok, dt, msg
    except subprocess.TimeoutExpired as exc:
        dt = time.monotonic() - t0
        blob = ((exc.stdout or b"") + b"\n" + (exc.stderr or b"")).decode(
            "utf-8", errors="replace"
        )
        log_path.write_text(blob + "\nTIMEOUT\n", encoding="utf-8", errors="replace")
        return jp.name, False, float(timeout), f"TIMEOUT log={log_path}"
    except Exception as exc:  # noqa: BLE001
        return jp.name, False, time.monotonic() - t0, str(exc)


def harvest(work_in: Path, af_out: Path, dst: Path) -> int:
    """Copy / merge MSA-bearing JSON into ``dst`` for inference."""
    dst.mkdir(parents=True, exist_ok=True)
    by_name = {}

    for p in sorted(work_in.glob("*.json")):
        if p.name == "build_manifest.json":
            continue
        if json_has_msa(p):
            by_name[p.name] = p

    if af_out.is_dir():
        for p in af_out.rglob("*.json"):
            low = p.name.lower()
            if "confidence" in low or "summary" in low:
                continue
            if not (low.endswith("_data.json") or json_has_msa(p)):
                continue
            key = p.name
            if key.endswith("_data.json"):
                key = key[: -len("_data.json")] + ".json"
            if key not in by_name:
                by_name[key] = p
            parent = p.parent.name
            for src in work_in.glob("*.json"):
                if src.name == "build_manifest.json":
                    continue
                try:
                    nm = (
                        json.loads(src.read_text(encoding="utf-8")).get("name")
                        or src.stem
                    )
                except Exception:
                    nm = src.stem
                if sanitize_job_name(str(nm)) == parent and src.name not in by_name:
                    if json_has_msa(p) or low.endswith("_data.json"):
                        by_name[src.name] = p

    n_ok = 0
    for name, src in sorted(by_name.items()):
        out = dst / (name if name.endswith(".json") else f"{name}.json")
        seq_src = work_in / out.name
        if seq_src.is_file() and src.resolve() != seq_src.resolve():
            try:
                base = json.loads(seq_src.read_text(encoding="utf-8"))
                feat = json.loads(src.read_text(encoding="utf-8"))
                if "sequences" in feat:
                    base["sequences"] = feat["sequences"]
                    for k in ("bondedAtomPairs", "userCCD", "dialect", "version"):
                        if k in feat and k not in base:
                            base[k] = feat[k]
                    out.write_text(
                        json.dumps(base, indent=2) + "\n", encoding="utf-8"
                    )
                elif json_has_msa(src):
                    out.write_bytes(src.read_bytes())
                else:
                    continue
            except Exception:
                out.write_bytes(src.read_bytes())
        else:
            out.write_bytes(src.read_bytes())
        n_ok += 1
    return n_ok


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="af3parallel msa",
        description=(
            "Parallel AF3 data pipeline (--norun_inference). "
            f"Default --workers = cpu_count//4 (currently {default_workers()}). "
            "For GPU ColabFold MSA see https://github.com/sokrypton/ColabFold"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Directory of sequence-only AF3 JSON (natural mode)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="AF3 af_output root for data-pipeline jobs")
    ap.add_argument(
        "--harvest-dir",
        type=Path,
        default=None,
        help="Where to write MSA-bearing JSON for inference "
        "(default: <input-dir>/../json_with_msa)",
    )
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--sif", type=Path, required=True)
    ap.add_argument("--af3-db", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--af3-home", type=Path, required=True,
                    help="Host directory containing run_alphafold.py")
    ap.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max concurrent data-pipeline jobs (default: cpu_count//4)",
    )
    ap.add_argument("--timeout", type=int, default=86400)
    ap.add_argument("--limit", type=int, default=0,
                    help="Only run first N pending jobs (smoke test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--nv",
        action="store_true",
        help="Pass singularity --nv (usually unnecessary on CPU nodes)",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workers = args.workers if args.workers is not None else default_workers()
    if workers < 1:
        print("ERROR: --workers must be >= 1", file=sys.stderr)
        return 2

    for p, label in [
        (args.input_dir, "input-dir"),
        (args.sif, "sif"),
        (args.af3_db, "af3-db"),
        (args.models, "models"),
        (args.af3_home / "run_alphafold.py", "run_alphafold.py"),
    ]:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}", file=sys.stderr)
            return 1

    print(
        "[INFO] MSA strategies: (1) this command = parallel AF3 data pipeline; "
        "(2) ColabFold GPU search = https://github.com/sokrypton/ColabFold",
        flush=True,
    )

    runtime = find_runtime()
    jsons = sorted(
        p
        for p in args.input_dir.glob("*.json")
        if p.name != "build_manifest.json"
    )
    if not jsons:
        print(f"[ERROR] no JSON in {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    harvest_dir = args.harvest_dir or (args.input_dir.parent / "json_with_msa")
    log_dir = args.log_dir or (args.input_dir.parent / "logs" / "msa_jobs")
    log_dir.mkdir(parents=True, exist_ok=True)

    todo: List[Path] = []
    skipped = 0
    for jp in jsons:
        if not args.force and is_job_done(args.output_dir, jp):
            skipped += 1
        else:
            todo.append(jp)
    if args.limit and args.limit > 0:
        todo = todo[: args.limit]

    print(
        f"[INFO] MSA CPU-parallel ({runtime}): n={len(jsons)}  todo={len(todo)}  "
        f"skip={skipped}  workers={workers}  nv={args.nv}  logs={log_dir}",
        flush=True,
    )
    if not todo:
        n = harvest(args.input_dir, args.output_dir, harvest_dir)
        print(f"[INFO] nothing to run; harvested {n} -> {harvest_dir}")
        return 0

    print(
        f"[INFO] first pending: {todo[0].name}  "
        f"(errors → {log_dir / (todo[0].stem + '.log')})",
        flush=True,
    )

    ok_n = fail_n = 0
    t0 = time.monotonic()
    # ThreadPoolExecutor: as_completed fills free slots as soon as a job ends.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                run_one,
                str(jp),
                str(args.sif.resolve()),
                str(args.af3_db.resolve()),
                str(args.models.resolve()),
                str(args.output_dir.resolve()),
                str(args.af3_home.resolve()),
                str(log_dir.resolve()),
                args.timeout,
                bool(args.nv),
                runtime,
            ): jp
            for jp in todo
        }
        for i, fut in enumerate(as_completed(futs), 1):
            name, ok, dt, msg = fut.result()
            if ok:
                ok_n += 1
                print(f"[OK   {i}/{len(todo)}] {name}  {dt:.0f}s", flush=True)
            else:
                fail_n += 1
                print(
                    f"[FAIL {i}/{len(todo)}] {name}  {dt:.0f}s  {msg}",
                    flush=True,
                )
                if fail_n == 1:
                    print(
                        "[HINT] Inspect the job log. Smoke-test with "
                        "--workers 1 --limit 1",
                        flush=True,
                    )

    wall = time.monotonic() - t0
    n_harv = harvest(args.input_dir, args.output_dir, harvest_dir)
    print(
        f"[INFO] done wall={wall:.0f}s  ok={ok_n}  fail={fail_n}  "
        f"skipped={skipped}  harvested={n_harv} -> {harvest_dir}",
        flush=True,
    )
    return 0 if fail_n == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
