# AF3Parallel

A toolkit for running [AlphaFold 3](https://github.com/google-deepmind/alphafold3)
inference at scale on multi-GPU clusters. The repository contains a
profile-driven multi-GPU executor, two memory profilers (peak and
time-series), GPU and CPU runtime estimators, an input-JSON manipulation
utility, and a standalone GPU memory monitor.

The tools are designed to work together: profile your hardware once, use
the resulting TSV to estimate batch runtime ahead of time, then run the
batch in parallel across all available GPUs with VRAM-aware scheduling.

---

## Repository contents

| Script | Purpose |
| --- | --- |
| `AF3Parallel.py` | Main executor. Distributes AF3 jobs across multiple GPUs with token-balanced LPT scheduling, joint VRAM and wall-clock batching, and a multi-anchor temporal-wave scheduler that hides small jobs in the VRAM shadow of long ones. |
| `AF3_GPU_Memory_Profiler.py` | One-shot peak-VRAM scan. Runs AF3 on a representative sample of inputs and records peak GPU memory plus runtime per token count, producing the TSV consumed by the executor and the estimators. |
| `AF3_GPU_Memory_Time-Series_Profiler.py` | Time-series VRAM profiler. Samples GPU memory at sub-second intervals during AF3 runs to study allocation curves and detect transient spikes. |
| `AF3_GPU_time_estimate.py` | Estimates total serial GPU runtime for a directory of AF3 input JSONs against a token-runtime profile. |
| `AF3_CPU_time_estimate.py` | Estimates total CPU/MSA (data-pipeline) runtime for a directory of AF3 input JSONs against a protein-length profile. |
| `AF3_JSON_Integrator.py` | Batch-modify AF3 input JSONs: set seeds, add/replace ligands (SMILES or CCD codes), nucleic-acid chains, or metal ions. Single-file, bulk and CSV fan-out modes. |
| `GPU_monitor.py` | Lightweight standalone GPU memory monitor. Polls `nvidia-smi`, writes TSV, optional auto-exit when all monitored GPUs idle for a sustained period. |

---

## Prerequisites

### 1. AlphaFold 3 (v3.0.1) — required first

**These scripts are wrappers around AlphaFold 3 and live alongside the
official source tree.** Before installing AF3Parallel, clone the
AlphaFold 3 repository at tag `v3.0.1`, build a Singularity image from
the bundled `docker/Dockerfile`, fetch the genetic databases, and place
the model weights. The cloned `alphafold3/` directory then becomes the
working directory into which the AF3Parallel scripts are copied (see
the *Installation* section below).

The five items you need to assemble are summarised below; for
authoritative, up-to-date instructions follow the
[official installation guide](https://github.com/google-deepmind/alphafold3/blob/v3.0.1/docs/installation.md).

| Component         | What it is                                                                                                       | Path convention used in this README |
| ----------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| AF3 source tree   | The repository at tag `v3.0.1` (provides `docker/`, `src/alphafold3/`, `run_alphafold.py`, `fetch_databases.sh`) | `./alphafold3/` (working directory) |
| Singularity image | `alphafold3.sif` built from `docker/Dockerfile`                                                                  | `./alphafold3/alphafold3.sif`       |
| Model weights     | `af3.bin` (or `af3.bin.zst`) — request access from Google DeepMind                                               | `./alphafold3/models/`              |
| Genetic databases | ~252 GB compressed / ~628 GB unpacked, fetched by `fetch_databases.sh`                                           | **outside** the AF3 repo (see *Installation* layout) |
| `nvidia-smi`      | Ships with the NVIDIA driver; verify with `nvidia-smi`                                                           | in `PATH`                           |

> **Verify your AF3 setup before installing AF3Parallel.** From inside
> the cloned `alphafold3/` directory, run a single-job prediction with a
> plain `singularity exec` and your own `fold_input.json`:
> ```bash
> singularity exec --nv \
>     --bind $HOME/af_input:/root/af_input \
>     --bind $HOME/af_output:/root/af_output \
>     --bind ./models:/root/models \
>     --bind ~/af3_DB:/root/public_databases \
>     alphafold3.sif \
>     python /app/alphafold/run_alphafold.py \
>         --json_path=/root/af_input/fold_input.json \
>         --model_dir=/root/models \
>         --db_dir=/root/public_databases \
>         --output_dir=/root/af_output
> ```
> If this finishes without error, your environment is ready for AF3Parallel.

**Hardware requirements (from the AF3 installation guide):**
- Linux (Ubuntu 22.04 LTS recommended)
- NVIDIA GPU with Compute Capability ≥ 8.0 (A100, H100, RTX 4090, etc.)
- ≥ 64 GB RAM recommended for the genetic-search (MSA) stage
- ≥ 1 TB SSD for the genetic databases

### 2. Python and optional packages

- **Python** >= 3.8
- `psutil` *(optional)* — enables `AF3Parallel.py`'s automatic global
  concurrency cap, which derives a safe `--max-concurrent-tasks` from
  total system RAM. Without it the auto-cap is silently skipped; you can
  still pass `--max-concurrent-tasks N` explicitly.
- `rdkit` *(optional)* — gives more accurate ligand heavy-atom counts for
  SMILES strings; both `AF3Parallel.py` and `AF3_GPU_time_estimate.py`
  fall back to a built-in regex tokenizer if `rdkit` is not importable.

---

## Installation

> Before you start, make sure the four AF3 components from the
> *Prerequisites* section are in place:
> 1. The `alphafold3/` repository is cloned at tag `v3.0.1`.
> 2. `alphafold3/alphafold3.sif` exists and `singularity exec --nv alphafold3.sif sh -c 'nvidia-smi'` works.
> 3. `alphafold3/models/` contains `af3.bin` (or `af3.bin.zst`).
> 4. The genetic databases are downloaded **outside** the repository, e.g. at `~/af3_DB/`.

```bash
# Step 1 — change into the cloned AF3 working directory
cd /path/to/alphafold3

# Step 2 — clone AF3Parallel anywhere and copy its scripts into ./alphafold3/
#         so they sit next to AF3's own run_alphafold.py
git clone https://github.com/Xin-DongXu/AF3Parallel.git /tmp/AF3Parallel
cp /tmp/AF3Parallel/*.py            .
cp /tmp/AF3Parallel/requirements.txt .       # optional dependencies

# Step 3 — install the optional Python dependencies (psutil, rdkit)
pip install -r requirements.txt

# Step 4 — (optional) make scripts directly executable
chmod +x AF3Parallel.py AF3_*.py GPU_monitor.py
```

> Do **not** copy this repository's `README.md` or `LICENSE` into the
> AF3 tree — they would overwrite AF3's own files. Only the `*.py`
> scripts (and, optionally, `requirements.txt`) need to be copied in.

After Step 2, the on-disk layout looks like this. Items marked **[AF3]**
ship with the official AlphaFold 3 repository at `v3.0.1`; **[you]**
items are produced or placed by you during AF3 setup; **[this repo]**
items are the AF3Parallel scripts you just copied in.

```
~/                                                ← anywhere on disk
├── af3_DB/                                       # [you] genetic databases (~628 GB), kept OUTSIDE
│   ├── bfd-first_non_consensus_sequences.fasta   #       the AF3 repo per the official guide
│   ├── mgy_clusters_2022_05.fa
│   ├── pdb_2022_09_28_mmcif_files/
│   ├── uniref90_2022_05.fa
│   └── ...                                       #       (full list created by fetch_databases.sh)
│
└── alphafold3/                                   # ← cloned AF3 v3.0.1 = AF3Parallel working directory
    ├── docker/                                   # [AF3] Dockerfile + container helpers
    │   └── Dockerfile
    ├── docs/                                     # [AF3] official AF3 documentation
    ├── src/                                      # [AF3] AlphaFold 3 Python package source
    │   └── alphafold3/                           #       (model, data pipeline, scripts/, ...)
    ├── run_alphafold.py                          # [AF3] AF3 inference entry point
    ├── fetch_databases.sh                        # [AF3] genetic-database downloader
    ├── pyproject.toml                            # [AF3] AF3 package metadata
    ├── README.md                                 # [AF3] AF3's own README — do NOT overwrite
    ├── LICENSE                                   # [AF3] CC BY-NC-SA 4.0
    ├── WEIGHTS_TERMS_OF_USE.md                   # [AF3] terms governing model weights
    │
    ├── alphafold3.sif                            # [you] Singularity image built from docker/Dockerfile
    ├── models/                                   # [you] AF3 model weights
    │   └── af3.bin.zst
    ├── af_input/                                 # [you] your AF3 input JSON files
    │
    ├── AF3Parallel.py                            # [this repo] multi-GPU executor
    ├── AF3_GPU_Memory_Profiler.py                # [this repo] peak-VRAM profiler
    ├── AF3_GPU_Memory_Time-Series_Profiler.py    # [this repo] time-series VRAM profiler
    ├── AF3_GPU_time_estimate.py                  # [this repo] GPU runtime estimator
    ├── AF3_CPU_time_estimate.py                  # [this repo] CPU/MSA runtime estimator
    ├── AF3_JSON_Integrator.py                    # [this repo] input-JSON manipulator
    ├── GPU_monitor.py                            # [this repo] standalone GPU memory monitor
    └── requirements.txt                          # [this repo] optional Python deps (psutil, rdkit)
```

The Python scripts are self-contained — there is no package to install.
Invoke them with `python script.py ...` or, after `chmod +x`, run them
directly. Every command shown later in this README assumes you run it
from inside the `alphafold3/` working directory and uses `~/af3_DB` as
the database path; substitute the actual location you passed to
`fetch_databases.sh` if it differs.

---

## Typical workflow

```
                   +--------------------------------+
                   | AF3 input JSONs                |
                   | (generated by AF3_JSON_        |
                   |  Integrator.py or by hand)     |
                   +---------------+----------------+
                                   |
        +--------------------------+--------------------------+
        |                                                     |
        v                                                     v
+---------------------+                            +-----------------------+
| AF3_GPU_Memory_     |  one-shot, per GPU model   | AF3_GPU_time_estimate |
| Profiler.py         | -----TSV profile---------> | AF3_CPU_time_estimate |
| (run once per GPU)  |                            | (estimate batch wall) |
+----------+----------+                            +-----------------------+
           |
           | TSV profile (token_count, peak_memory_mb, runtime_seconds)
           v
+-------------------------------------------+
| AF3Parallel.py                            |
|   - LPT distribution across GPUs          |
|   - VRAM-aware batching                   |
|   - Temporal-wave scheduling              |
|   - Streaming TSV log                     |
+-------------------------------------------+
```

A typical end-to-end run:

1. **Profile once per GPU model** (skip if your GPU is one of the built-ins
   listed below):
   ```bash
   python AF3_GPU_Memory_Profiler.py \
       -i ./profile_inputs \
       -o my_gpu_profile.tsv \
       --sif alphafold3.sif \
       --af3-db ~/af3_DB \
       --models ./models
   ```
   > `AF3_GPU_Memory_Profiler.py` monitors GPU 0 by default. To profile a
   > different GPU, prefix the command with
   > `CUDA_VISIBLE_DEVICES=<index>`.

2. **(Optional) Estimate runtime** before launching a large batch:
   ```bash
   python AF3_GPU_time_estimate.py \
       --input-dir ./af_input \
       --profile   my_gpu_profile.tsv \
       --output-tsv estimate_breakdown.tsv \
       --workers   16
   ```

3. **Run the batch** across all GPUs:
   ```bash
   python AF3Parallel.py \
       -i  ./af_input \
       -o  results.tsv \
       --output-dir ./af_output \
       --sif alphafold3.sif \
       --af3-db ~/af3_DB \
       --models ./models \
       --gpus 0,1,2,3 \
       --memory-profile my_gpu_profile.tsv
   ```

---

## Built-in GPU profiles

`AF3Parallel.py` ships with built-in profiles measured on two GPU models.
**Only these two are officially supported**; all other presets listed in
`--help` are unmeasured extrapolations and should not be relied upon.

| Preset key | VRAM   | Tested on             |
| ---------- | ------ | --------------------- |
| `a800-80g` | 80 GB  | NVIDIA A800 80 GB     |
| `rtx4090`  | 24 GB  | NVIDIA RTX 4090 24 GB |

Pick one with `--gpu-preset`:

```bash
python AF3Parallel.py --gpu-preset a800-80g ...
python AF3Parallel.py --gpu-preset rtx4090 ...
```

**Any other GPU model requires a custom profile.** Run
`AF3_GPU_Memory_Profiler.py` on your hardware first, then pass the
resulting TSV via `--memory-profile`:

```bash
python AF3_GPU_Memory_Profiler.py \
    -i ./profile_inputs -o my_gpu_profile.tsv \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models

python AF3Parallel.py ... --memory-profile my_gpu_profile.tsv
```

If you pass neither `--gpu-preset` nor `--memory-profile`, `AF3Parallel.py`
queries `nvidia-smi`, attempts to match the detected VRAM against the
preset table within a +/-512 MB tolerance, and falls back to the A800
profile if nothing matches (a warning is printed). **This fallback is not
a substitute for proper profiling on untested hardware.**

---

## Profile TSV format

Profilers and estimators exchange data through a tab-separated file with
this header:

```
token_count    peak_memory_mb    runtime_seconds    success
```

`AF3_GPU_Memory_Profiler.py` writes additional diagnostic columns
(`json_file`, `protein_name`, `protein_length`, `ligand_count`,
`total_sequences`, `timestamp`); these are tolerated and ignored by the
loaders. Rows with `success=False` are discarded by the loaders unless
explicitly enabled. Sparse profiles (non-contiguous token counts) are
filled by linear interpolation between adjacent measurements.

---

## Per-script quick reference

Every script supports `--help` for the full option list. Selected examples:

### AF3Parallel.py — multi-GPU executor

```bash
python AF3Parallel.py \
    -i ./af_input -o results.tsv \
    --output-dir ./af_output \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models \
    --gpus 0,1,2,3 \
    --memory-profile my_gpu_profile.tsv

# Disable temporal-wave scheduling (fall back to plain VRAM-packed batches)
python AF3Parallel.py ... --no-temporal-waves

# Skip jobs whose predicted footprint exceeds GPU VRAM instead of
# routing them through CPU memory offloading
python AF3Parallel.py ... --skip-vram-overflow

# Drop oversized jobs by token count before scheduling
python AF3Parallel.py ... --max-tokens 4000

# Validate config + show the planned schedule without running AF3
python AF3Parallel.py ... --test-only

# Re-run jobs whose output directory already exists
python AF3Parallel.py ... --no-skip-existing
```

**I/O paths.** `-i DIR` / `-o FILE` are required. `--output-dir DIR`
(default `./af_output_parallel`) sets where AF3 writes its per-job result
folders; always specify it explicitly to avoid mixing outputs across runs.

**Resource & runtime budgets.** The executor enforces three independent
budgets — one VRAM, two wall-clock — plus an estimation safety factor:

- `--gpu-memory MB` / `--vram-margin RATIO` (default 0.95) /
  `--safety-margin RATIO` (default 0.10) — effective per-GPU VRAM is
  `gpu_memory * vram_margin * (1 - safety_margin)`. Auto-detected from
  `nvidia-smi` when not set.
- `--max-batch-runtime SECONDS` (default 7200) — wall-clock budget per
  batch; influences how the scheduler packs tasks into batches.
- `--task-timeout SECONDS` (default 7200) — hard timeout for any single
  AF3 subprocess.
- `--memory-estimation-factor FACTOR` (default 1.0) — multiplier applied
  to the profile's VRAM estimate; raise to 1.1–1.2 if you observe OOMs
  near the VRAM ceiling.

**Global concurrency cap (CPU-RAM-aware).** When many small-token tasks
fit in GPU VRAM simultaneously, they can still exhaust system RAM /
swap. `AF3Parallel.py` therefore applies a *global* cap on the number
of concurrent AF3 subprocesses across **all** GPUs:

```bash
# Set the cap explicitly
python AF3Parallel.py ... --max-concurrent-tasks 8

# Tune the auto-derivation (used when --max-concurrent-tasks is unset)
python AF3Parallel.py ... \
    --cpu-memory-per-task-mb 4500 \
    --cpu-memory-reserve-mb  8192

# Disable the auto-derivation entirely
python AF3Parallel.py ... --no-cpu-memory-autocap
```

If `--max-concurrent-tasks` is not given and `psutil` is installed, the
cap is computed as
`max(1, (total_ram_mb - cpu_memory_reserve_mb) // cpu_memory_per_task_mb)`.
Without `psutil` the auto-cap is skipped and concurrency is bounded only
by per-GPU VRAM.

**Temporal-wave tuning.** `--min-anchor-ratio` (default 2.0) sets how
much longer an anchor must be than its candidate wave tasks; the
companion `--max-anchor-group-ratio` (default 1.5) limits how
heterogeneous a group of anchors running on the same GPU is allowed to
be (largest anchor runtime / smallest anchor runtime).

**Other options of note.** `--temp-dir DIR` overrides the default
`<input-dir>/gpu_work/` workspace; `--norun-data-pipeline` is a direct
shortcut for the corresponding AF3 flag (equivalent to
`--af3-extra-args --norun_data_pipeline`); `--no-profile-gap-fill`
disables linear interpolation between non-contiguous profile rows;
`--strict-errors` aborts the run on the first task failure instead of
isolating it for retry.

#### Full option reference

The tables below list every option exposed by `python AF3Parallel.py
--help`, grouped exactly as `argparse` groups them. *Type* uses Python
conventions; *required* options have no default. Boolean flags are
shown with the `store_true` toggle they actually map to.

**Input/Output Options**

| Option | Type / metavar | Default | Description |
| --- | --- | --- | --- |
| `-i, --input-dir DIR` | `Path` | *(required)* | Directory containing AF3 input JSON files. Every `*.json` inside is treated as one prediction job. |
| `-o, --output-file FILE` | `Path` | *(required)* | TSV log written incrementally as tasks finish (one row per task; columns documented under *Output formats*). |
| `--output-dir DIR` | `str` | `./af_output_parallel` | Where AF3 writes its per-job result folders. Always set this explicitly to avoid mixing outputs across runs. |
| `--no-skip-existing` | flag | off (skip is enabled) | Re-run jobs even if their output directory already exists. By default jobs whose output dir is present are skipped. |
| `--skip-vram-overflow` | flag | off | Drop jobs whose predicted VRAM footprint exceeds the GPU's effective budget *instead of* letting them spill into CPU memory. |
| `--max-tokens N` | `int` | `None` | Hard upper bound on token count: any job above this is dropped before scheduling. Useful for excluding huge inputs from a batch. |
| `--temp-dir DIR` | `str` | `<input-dir>/gpu_work/` | Workspace where staged JSONs and per-task scratch live. Override if your input directory is read-only or on a slow filesystem. |

**AlphaFold3 Configuration**

| Option | Type / metavar | Default | Description |
| --- | --- | --- | --- |
| `--sif, --singularity-image FILE` | `str` | *(required)* | Path to the `alphafold3.sif` Singularity image built from `docker/Dockerfile`. |
| `--af3-db DIR` | `str` | `./af3_DB` | Path to the genetic-database directory created by `fetch_databases.sh`. Use an absolute path (e.g. `~/af3_DB`) since the databases live outside the AF3 repo. |
| `--models DIR` | `str` | `./models` | Path to the directory containing `af3.bin` (or `af3.bin.zst`). |
| `--norun-data-pipeline` | flag | off | Shortcut for AF3's `--norun_data_pipeline` flag. Skips the CPU-bound MSA/template stage; assumes inputs already contain MSAs. |
| `--af3-extra-args ARG [ARG ...]` | `str*` | `[]` | Extra arguments passed straight to `run_alphafold.py`. Use Abseil-style booleans (`--noFLAG`, no separator). Example: `--af3-extra-args --num_recycles=5 --save_distogram`. |

**GPU Configuration**

| Option | Type / metavar | Default | Description |
| --- | --- | --- | --- |
| `--gpus LIST` | `str` | all visible GPUs | Comma-separated GPU indices to use (e.g. `0,1,2,3`). Equivalent to setting `CUDA_VISIBLE_DEVICES` for the executor itself. |
| `--gpu-preset PRESET` | `str` | auto-detect | Selects a built-in profile and matching VRAM. Valid keys: `a800-80g`, `a100-80g`, `a100-40g`, `h100-80g`, `h100-94g`, `a6000-48g`, `v100-32g`, `rtx4090`, `rtx3090`, `rtx3090ti`, `rtx4080`. Only `a800-80g` and `rtx4090` are *measured*; the rest are extrapolations — see *Built-in GPU profiles* above. |
| `--gpu-memory MB` | `int` | `81920` (= 80 GB) | Per-GPU VRAM ceiling in MB. Auto-detected from `nvidia-smi` when not set; overridden by `--gpu-preset`. |
| `--vram-margin RATIO` | `float` in `[0.5, 1.0]` | `0.95` | Fraction of physical VRAM the executor is allowed to budget against (leaves headroom for CUDA overhead and fragmentation). |
| `--safety-margin RATIO` | `float` in `[0.0, 0.5]` | `0.10` | Additional shrink factor applied on top of `--vram-margin`. Effective per-GPU budget = `gpu_memory * vram_margin * (1 - safety_margin)`. |
| `--max-workers N` | `int` | unlimited | Per-batch concurrency cap on a *single* GPU (bounds how many tasks share one GPU within one batch). |
| `--max-concurrent-tasks N` | `int` | auto from CPU RAM | **Global** cap on concurrent AF3 subprocesses across all GPUs. Prevents CPU RAM/swap exhaustion when many small-token tasks fit in VRAM but collectively overflow system memory. Auto-derived from `psutil` if installed: `max(1, (total_ram_mb - cpu_memory_reserve_mb) // cpu_memory_per_task_mb)`. |
| `--cpu-memory-per-task-mb MB` | `int` | `4500` | Estimated CPU RAM per AF3 subprocess; used only for the auto-derivation above. The default matches observed RES on A800 nodes. |
| `--cpu-memory-reserve-mb MB` | `int` | `8192` | CPU RAM reserved for the OS, the parent Python process, `nvidia-smi`, etc.; subtracted from total RAM in the auto-derivation. |
| `--no-cpu-memory-autocap` | flag | off | Disable the auto-derivation entirely. Concurrency is then bounded only by per-GPU VRAM (and `--max-workers` if set). |
| `--max-batch-runtime SECONDS` | `float` ≥ 60 | `7200.0` | Wall-clock budget per batch; influences how the scheduler packs tasks. |
| `--task-timeout SECONDS` | `int` ≥ 60 | `7200` | Hard timeout for any single AF3 subprocess. |

**Temporal Wave Scheduling**

| Option | Type / metavar | Default | Description |
| --- | --- | --- | --- |
| `--no-temporal-waves` | flag | off (waves enabled) | Disable temporal-wave scheduling and fall back to plain VRAM-packed batches. |
| `--min-anchor-ratio RATIO` | `float` ≥ 1.1 | `2.0` | Minimum runtime ratio between an anchor and its candidate wave tasks. Larger values pick longer anchors and admit only much shorter tasks into their VRAM shadow. |
| `--max-anchor-group-ratio RATIO` | `float` | `1.5` | Limits how heterogeneous a group of anchors on the same GPU may be (longest anchor runtime ÷ shortest). |

**Memory Profiling Options**

| Option | Type / metavar | Default | Description |
| --- | --- | --- | --- |
| `--memory-profile FILE` | `Path` | built-in profile for the detected/preset GPU | Custom TSV produced by `AF3_GPU_Memory_Profiler.py`. Required for any GPU model not listed in `--gpu-preset`. |
| `--no-profile-gap-fill` | flag | off | Disable linear interpolation between non-contiguous rows in the profile. With this on, gaps in the profile cause hard failures rather than estimates. |
| `--memory-estimation-factor FACTOR` | `float` in `[0.5, 5.0]` | `1.0` | Multiplier applied to the profile's VRAM estimate. Raise to `1.1`–`1.2` if you observe OOMs near the VRAM ceiling. |

**Monitoring & Debug Options**

| Option | Type / metavar | Default | Description |
| --- | --- | --- | --- |
| `--monitor-interval SECONDS` | `int` | `5` | Polling interval for the in-process GPU memory monitor (separate from `GPU_monitor.py`). |
| `--cpu-workers N` | `int` | auto (CPU count) | Number of parallel workers used for the JSON-parsing / token-counting pre-pass. |
| `-v, --verbose` | flag | off | Print extra diagnostic output (per-task timing breakdown, scheduling decisions). |
| `--strict-errors` | flag | off | Abort the entire run on the first task failure instead of isolating it for retry. |
| `--test-only` | flag | off | Dry-run mode: validate config, load profiles, plan batches/waves, and print the schedule — but do not invoke AF3. Recommended before any large run. |

### AF3_GPU_Memory_Profiler.py — peak VRAM profiler

```bash
python AF3_GPU_Memory_Profiler.py \
    -i ./profile_inputs -o profile.tsv \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models \
    --interval 1.0
```

### AF3_GPU_Memory_Time-Series_Profiler.py — time-series VRAM profiler

```bash
python AF3_GPU_Memory_Time-Series_Profiler.py \
    --stat-file AF3_A800_80G_All_Len_stat.txt \
    --input-dir ./af_input \
    --output-dir ./timeseries_output \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models \
    --monitor-interval 0.5 \
    --n-per-step 3 --workers 8
```

### AF3_GPU_time_estimate.py — GPU runtime estimator

```bash
python AF3_GPU_time_estimate.py \
    --input-dir  ./af_input \
    --profile    my_gpu_profile.tsv \
    --output-tsv breakdown.tsv \
    --workers    16
# stdout: total estimated serial runtime in seconds (machine-readable)
```

### AF3_CPU_time_estimate.py — CPU/MSA runtime estimator

```bash
python AF3_CPU_time_estimate.py \
    --input-dir  ./af_input \
    --profile    af3_CPU_Test_stat.tsv \
    --output-tsv msa_breakdown.tsv \
    --workers    16
```

The CPU profile is keyed on the longest protein chain length (MSA cost is
driven by the largest protein passed to jackhmmer/hhblits), not by AF3
token count.

### AF3_JSON_Integrator.py — input JSON manipulation

A programmatic, protein-preserving editor for AlphaFold 3 input JSON files
(the [`alphafold3` dialect](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)).
It is designed for **screening workflows** in which a single
protein-of-interest is repeatedly evaluated against many different ligands,
nucleic-acid partners, metal ions, or random seeds — without ever touching
the protein chains themselves.

> The legacy `alphafoldserver` dialect (top-level JSON list) is *not*
> supported. Inputs must be the alphafold3-dialect dict-form JSON.

#### Subcommands

| Subcommand        | Effect                                                                                                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `set-seeds`       | Modify only `modelSeeds`; everything else (including the `name` field) stays byte-identical.                        |
| `add-ligand`      | Append a new small-molecule ligand entry (SMILES *or* one-or-more CCD codes for multi-component ligands).            |
| `replace-ligand`  | Replace an existing small-molecule ligand identified by chain ID.                                                    |
| `add-nucleic`     | Append a new RNA or DNA chain.                                                                                       |
| `replace-nucleic` | Replace an existing RNA/DNA chain identified by chain ID.                                                            |
| `add-ion`         | Append a metal ion (CCD code only, e.g. `MG`, `ZN`; per AF3 spec ions are simply ligands with single-element codes). |
| `replace-ion`     | Replace an existing metal ion identified by chain ID.                                                                |

#### Three I/O modes

Every operation supports the same three modes (with one exception noted below):

| Mode        | Required flags                                  | When to use                                                       |
| ----------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| **Single**  | `-i FILE -o FILE`                               | One input → one output. Quick edits, scripting.                   |
| **Bulk**    | `--input-dir DIR --output-dir DIR`              | Apply the *same* operation to every `*.json` in a directory.      |
| **Fan-out** | `-i FILE --from-csv CSV --output-dir DIR`       | One base JSON → many outputs, one per CSV row (parameter sweeps). |

`set-seeds` does not support Fan-out mode (use Single or Bulk instead).

In **Bulk** and **Fan-out** modes the script runs tasks in parallel via
`--workers N` (default: `min(8, os.cpu_count())`). Failed tasks are logged
but never abort the whole run; the final stderr summary lists success /
failure counts plus the first few error messages. The exit code is `0`
on full success, `2` if at least one task failed, and `1` for argument
errors.

#### `--in-place` mode

A space-saving alternative to writing copies into `--output-dir`. It is
particularly useful when the JSONs are large (e.g. contain pre-computed
MSAs). It works with `-i FILE` and with `--input-dir DIR`, but is **not**
compatible with `-o`, `--output-dir`, or `--from-csv`. The on-disk filename
is preserved; the internal `name` field is still updated normally per the
operation. Writes are atomic (temp-file + rename) so the original file is
never corrupted on validation errors, disk-full conditions, or process
interruption.

#### `name` field convention

The `name` field is treated as `<protein_prefix>_<ligand_tag>`, split on the
**first** underscore. Therefore:

- The protein-ID prefix should not itself contain underscores.
- `set-seeds` is name-stable (does not modify `name`).
- Every other operation rewrites only the `<ligand_tag>` suffix; the
  protein-ID prefix is preserved verbatim.
- In **Bulk** and **Fan-out** modes the on-disk output filename is derived
  from the resulting `name` field (`<name>.json`). Filename collisions
  abort the run before writing anything to disk, so a sweep that would
  silently overwrite outputs is detected up front.

#### Invariants enforced by every operation

1. Protein chains are never modified — every transformed JSON is
   deep-compared against the input for protein-entry equality.
2. The protein-ID prefix in `name` is preserved verbatim.
3. `set-seeds` is name-stable.
4. Chain IDs are kept unique across the entire `sequences` list.
5. Output-filename collisions in Bulk / Fan-out modes abort the run before
   any write to disk.

#### CSV manifest format (Fan-out mode)

The CSV must have a header row. Required and optional columns depend on
the operation. In Fan-out mode the per-task parameters come from the CSV,
so the corresponding CLI flags (`--smiles`, `--ccd`, `--sequence`,
`--ligand-tag`, `--target-id`, `--copies`, `--chain-id`, ...) are
**rejected** to prevent silent overrides — the integrator fails fast with
a clear error if you mix them.

| Operation                          | Required columns                                  | Optional columns                                  |
| ---------------------------------- | ------------------------------------------------- | ------------------------------------------------- |
| `add-ligand` / `replace-ligand`    | `ligand_tag`, one of `smiles` or `ccd`; `target_id` (replace only) | `copies` (default 1, add only), `chain_id` (add only) |
| `add-nucleic` / `replace-nucleic`  | `ligand_tag`, `type` (`rna` or `dna`), `sequence`; `target_id` (replace only) | `copies` (default 1, add only), `chain_id` (add only) |
| `add-ion` / `replace-ion`          | `ligand_tag`, `ccd` (single ion code, e.g. `MG`, `ZN`); `target_id` (replace only) | `copies` (default 1, add only), `chain_id` (add only) |

The `ccd` column for `add-ligand` / `replace-ligand` accepts a
**comma-separated** list to define a multi-component ligand
(e.g. `ATP,MG`).

#### Examples

**1. `set-seeds` — fix or randomise the seed list**

```bash
# Replace seeds with an explicit list (single-file)
python AF3_JSON_Integrator.py set-seeds \
    -i input.json -o output.json --seeds 1 2 3

# Generate 5 consecutive seeds starting at 1000
python AF3_JSON_Integrator.py set-seeds \
    -i input.json -o output.json --num-seeds 5 --seed-base 1000

# Generate 5 random seeds reproducibly across an entire directory, in place
python AF3_JSON_Integrator.py set-seeds \
    --input-dir ./inputs --in-place \
    --num-seeds 5 --rng-seed 42 --workers 8
```

**2. `add-ligand` — append a ligand to every JSON in a directory**

```bash
# Add paracetamol (SMILES) to every JSON in ./inputs, in parallel
python AF3_JSON_Integrator.py add-ligand \
    --input-dir ./inputs --output-dir ./outputs \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --ligand-tag paracetamol \
    --copies 1 --workers 8

# Add a multi-component CCD ligand (ATP coordinated with Mg)
python AF3_JSON_Integrator.py add-ligand \
    -i base.json -o base_with_ATP_MG.json \
    --ccd ATP MG --ligand-tag ATP_MG
```

**3. `replace-ligand` — fan-out screen across many candidates**

A typical CSV (`ligands.csv`) for a SMILES screen:

```csv
ligand_tag,target_id,smiles
cmpd001,L,CC(=O)Nc1ccc(O)cc1
cmpd002,L,O=C(O)c1ccccc1O
cmpd003,L,CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

Run the screen — one output JSON per row, named after `<protein_prefix>_<ligand_tag>.json`:

```bash
python AF3_JSON_Integrator.py replace-ligand \
    -i base.json --from-csv ligands.csv \
    --output-dir ./outputs --workers 16
```

**4. `add-nucleic` / `replace-nucleic` — RNA or DNA partners**

```bash
# Add a single-stranded RNA hairpin to one base JSON
python AF3_JSON_Integrator.py add-nucleic \
    -i base.json -o base_with_RNA.json \
    --type rna --sequence GGGAAACCC \
    --ligand-tag hairpin

# Replace whatever RNA is on chain B with a new sequence (bulk, in place)
python AF3_JSON_Integrator.py replace-nucleic \
    --input-dir ./inputs --in-place \
    --target-id B --type rna --sequence AUGCAUGCAU \
    --ligand-tag rna_v2 --workers 8
```

**5. `add-ion` / `replace-ion` — metal ions**

```bash
# Add 2 Mg2+ ions to a base JSON
python AF3_JSON_Integrator.py add-ion \
    -i base.json -o base_with_2MG.json \
    --ccd MG --copies 2 --ligand-tag 2MG

# Swap the existing ion on chain C for Zn2+
python AF3_JSON_Integrator.py replace-ion \
    -i base.json -o base_ZN.json \
    --target-id C --ccd ZN --ligand-tag ZN
```

#### Common flags

| Flag                       | Applies to                | Description                                                                            |
| -------------------------- | ------------------------- | -------------------------------------------------------------------------------------- |
| `-i, --input FILE`         | all                       | Input JSON file (Single-file or Fan-out mode).                                         |
| `--input-dir DIR`          | all                       | Input directory of `*.json` files (Bulk mode).                                         |
| `-o, --output FILE`        | all                       | Output JSON file (Single-file mode only).                                              |
| `--output-dir DIR`         | all                       | Output directory (Bulk or Fan-out mode).                                               |
| `--in-place`               | all                       | Edit input files in place; incompatible with `-o`, `--output-dir`, `--from-csv`.       |
| `--from-csv CSV`           | all except `set-seeds`    | CSV manifest for Fan-out mode.                                                          |
| `--ligand-tag STR`         | all add/replace ops       | Suffix written into the `name` field; required in Single and Bulk, ignored in Fan-out. |
| `--workers N`              | all                       | Parallel worker processes (default: `min(8, os.cpu_count())`).                          |
| `--quiet`                  | all                       | Suppress progress / summary on stderr.                                                  |
| `--version`                | top-level                 | Print version and exit.                                                                 |

For the full per-subcommand option list, see
`python AF3_JSON_Integrator.py <subcommand> --help`.

### GPU_monitor.py — standalone GPU memory monitor

```bash
# Sample all GPUs every 5 s, exit when ALL stay below 100 MiB for 5 minutes
python GPU_monitor.py -o gpu_log.tsv -i 5 -t 100 -d 300

# Monitor only GPUs 0 and 2, run forever
python GPU_monitor.py -o gpu_log.tsv -g 0,2 -d 0
```

---

## Output formats

All scripts log structured TSV. The main `AF3Parallel.py` per-task log
has the following columns (written as a single header row):

```
gpu_id  batch_id  is_retry  batch_type  wave_id
batch_peak_memory_mb  batch_runtime_seconds
task_id  json_file  protein_name
token_count  protein_length  rna_length  dna_length  ligand_count  total_sequences
estimated_memory_mb  estimated_runtime_s
task_peak_memory_mb  runtime_seconds
timeout_risk  success  timestamp
```

Records are written immediately as each task finishes, so the file is
useful even if the run is interrupted.

---

## Tips and gotchas

- AlphaFold 3 uses [Abseil flags](https://abseil.io/docs/python/guides/flags),
  whose boolean negations are written `--noFLAG` (no separator). Pass extra
  AF3 flags with `--af3-extra-args`, e.g. `--af3-extra-args --norun_data_pipeline`,
  or use the dedicated shortcut `--norun-data-pipeline`.
  `AF3_GPU_Memory_Time-Series_Profiler.py` auto-corrects the most common
  wrong forms.
- Run with `--test-only` first to dry-run the full pipeline up to (but not
  including) AF3 invocation. The summary prints the auto-detected GPU
  preset, the resolved global concurrency cap, the per-GPU task
  distribution, and the planned batches / temporal waves — useful for
  catching misconfigurations before committing to a long run.
- If you launch on a GPU that doesn't match any built-in preset and you
  haven't set `--memory-profile`, the run continues with the detected
  VRAM size and the default A800 profile, after printing a warning.
  Profile your GPU once with `AF3_GPU_Memory_Profiler.py` to silence the
  warning and improve scheduling accuracy.
- On many-small-task workloads (hundreds of <500-token jobs), the global
  concurrency cap is what stops the OS from swapping itself to death.
  Install `psutil` and let the auto-cap do the right thing, or pass
  `--max-concurrent-tasks` explicitly.
- On SIGINT/SIGTERM, `AF3Parallel.py` tries to restore staged JSON files
  back to the original input directory before exiting. If a run is killed
  uncleanly, look for stragglers in `./<input-dir>/gpu_work/` (or the
  directory passed via `--temp-dir`).
- For mixed workloads (many small jobs + a few huge ones), enable temporal
  waves (default). The scheduler launches one or more long-running anchor
  jobs and packs successive waves of light jobs into the unused VRAM
  inside the anchor runtime window.

---

## License

This project is released under the [MIT License](LICENSE).

AlphaFold 3 itself is licensed separately by Google DeepMind and is **not**
distributed by this repository — see the upstream project for details.

---

## Citation

If you use these tools in academic work, please cite AlphaFold 3:

> Abramson, J., Adler, J., Dunger, J. *et al.* Accurate structure
> prediction of biomolecular interactions with AlphaFold 3. *Nature*
> **630**, 493-500 (2024). https://doi.org/10.1038/s41586-024-07487-w

A `CITATION.cff` for this repository can be added once a release is
tagged; please open an issue if you would like one provided.
