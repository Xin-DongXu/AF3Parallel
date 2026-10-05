# AF3Parallel

[![PyPI](https://img.shields.io/pypi/v/af3parallel.svg)](https://pypi.org/project/af3parallel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AlphaFold 3](https://img.shields.io/badge/AlphaFold_3-v3.0.1-green.svg)](https://github.com/google-deepmind/alphafold3)

Profile-driven toolkit for running [AlphaFold 3](https://github.com/google-deepmind/alphafold3) inference at scale on multi-GPU Linux clusters.

AF3Parallel wraps the official AF3 Singularity workflow with VRAM-aware scheduling, temporal-wave batching, and companion utilities for profiling, runtime estimation, and input JSON preparation. Install from **PyPI** and invoke all tools through a single CLI.

**PyPI:** https://pypi.org/project/af3parallel/

---

## Overview

Profile-driven VRAM scheduling: peak-memory and runtime profiles (**top**), MSA cache construction (**1**), token-balanced LPT allocation (**2**), anchor / wave packing (**3–5**), and temporal-wave execution (**6**).

<p align="center">
  <img src="docs/images/scheduling-overview.png" alt="AF3Parallel overview: MSA cache, GPU profiles, LPT allocation, anchor/wave packing, and temporal-wave execution" width="100%">
</p>

---

## What's included

| Tool | CLI command | Purpose |
| --- | --- | --- |
| JSON builder | `af3parallel build-json` | Build AF3 inputs from FASTA/PDB/manifest (**natural** = MSA later; **denovo** = empty MSA) |
| Parallel MSA | `af3parallel msa` | CPU data pipeline with fixed concurrency (default `cpu_count//4`); or use [ColabFold](https://github.com/sokrypton/ColabFold) |
| Multi-GPU executor | `af3parallel run` | Distribute AF3 jobs across GPUs with LPT scheduling, VRAM-aware batching, and temporal-wave packing |
| Peak VRAM profiler | `af3parallel profile` | One-shot peak-memory scan → TSV profile for scheduling |
| Time-series profiler | `af3parallel profile-ts` | Sub-second VRAM sampling during AF3 runs |
| GPU runtime estimator | `af3parallel estimate-gpu` | Predict serial GPU wall time from a token profile |
| CPU/MSA estimator | `af3parallel estimate-cpu` | Predict data-pipeline wall time from a protein-length profile |
| JSON integrator | `af3parallel json` | Batch-edit existing AF3 inputs (seeds, ligands, nucleic acids, ions) |
| Result stats | `af3parallel stats` | Summarise `*_summary_confidences.json` (ligand interfaces or general scores) |
| GPU monitor | `af3parallel monitor` | Standalone `nvidia-smi` memory logger |

Built-in VRAM/runtime profiles are **measured** on NVIDIA A800 80 GB and RTX 4090 24 GB; other GPUs require a one-time custom profile from `af3parallel profile`.

---

## Installation

### Prerequisites

Complete the [official AF3 v3.0.1 installation](https://github.com/google-deepmind/alphafold3/blob/v3.0.1/docs/installation.md) first (Singularity image, model weights, genetic databases). Details: [docs/installation.md](docs/installation.md).

| Component | Required | Notes |
| --- | --- | --- |
| AlphaFold 3 v3.0.1 + Singularity | Yes | Run from your AF3 working directory |
| Linux + NVIDIA GPU (CC ≥ 8.0) | Yes | e.g. A100, H100, RTX 4090 |
| Python ≥ 3.8 | Yes | Core tools use the standard library |
| `psutil` | Optional | Auto `--max-concurrent-tasks` in `af3parallel run` |
| `rdkit` | Optional | More accurate SMILES heavy-atom counts |

### pip

```bash
pip install "af3parallel[extras]"
```

Verify:

```bash
af3parallel --version
af3parallel --help
```

---

## Quick start

Run from your AF3 working tree (`alphafold3/`):

```bash
# 0a. Natural proteins: build JSON (no MSA fields) → data pipeline later
af3parallel build-json natural \
    --fasta ./proteome.fasta --ccd ATP --out-dir ./json_seq_only

# 0b. De novo / designed proteins: empty MSA → inference without data pipeline
af3parallel build-json denovo \
    --fasta ./designs.fasta --out-dir ./json_denovo

# 0c. Parallel MSA (AF3 data pipeline). Default workers = cpu_count/4
af3parallel msa \
    --input-dir ./json_seq_only --output-dir ./msa_af_output \
    --harvest-dir ./json_with_msa \
    --sif alphafold3.sif --af3-db ~/af3_DB --models ./models \
    --af3-home .
# Alternative: ColabFold GPU MSA — https://github.com/sokrypton/ColabFold

# 1. Profile once per GPU model (skip for built-in a800-80g / rtx4090)
af3parallel profile \
    -i ./profile_inputs -o my_gpu_profile.tsv \
    --sif alphafold3.sif --af3-db ~/af3_DB --models ./models

# 2. (Optional) estimate batch runtime
af3parallel estimate-gpu \
    --input-dir ./json_with_msa --profile my_gpu_profile.tsv \
    --output-tsv estimate_breakdown.tsv --workers 16

# 3. Run the batch across all GPUs (MSA-bearing or denovo empty-MSA JSON)
af3parallel run \
    -i ./json_with_msa -o results.tsv --output-dir ./af_output \
    --sif alphafold3.sif --af3-db ~/af3_DB --models ./models \
    --gpus 0,1,2,3 --memory-profile my_gpu_profile.tsv

# 4. Summarise confidence JSONs
af3parallel stats -i ./af_output -r -o detail.csv -s summary.csv
```

Dry-run: `af3parallel run ... --test-only`

Bulk ligand replacement (directory of JSON files, in place):

```bash
af3parallel json replace-ligand \
    --input-dir ./inputs --in-place \
    --target-id L \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --ligand-tag paracetamol \
    --workers 8
```

Applies the same replacement to every `*.json` in `./inputs`; files are updated atomically in place.

---

## Typical workflow

```
  AF3 input JSONs  ──►  af3parallel profile  ──►  TSV profile
         │                                              │
         │                                              ▼
         ├──►  af3parallel estimate-gpu/cpu             │
         │                                              ▼
         └──────────────────────────────►  af3parallel run  ──►  results.tsv
```

See [docs/workflow.md](docs/workflow.md).

---

## CLI reference

```bash
af3parallel <command> [arguments]
af3parallel run --help
python -m af3parallel --help
```

| Subcommand | Standalone alias |
| --- | --- |
| `run` | `af3parallel-run` |
| `profile` | `af3parallel-profile` |
| `profile-ts` | `af3parallel-profile-ts` |
| `estimate-gpu` | `af3parallel-estimate-gpu` |
| `estimate-cpu` | `af3parallel-estimate-cpu` |
| `json` | `af3parallel-json` |
| `monitor` | `af3parallel-monitor` |

Full flags: [docs/cli-reference.md](docs/cli-reference.md)

---

## Features

- Token-balanced **LPT** multi-GPU distribution
- **VRAM-aware** batching with temporal-wave scheduling
- Built-in profiles for **A800 80 GB** and **RTX 4090 24 GB**
- Streaming TSV logs, per-task retry, SIGINT cleanup
- Optional `psutil` CPU-RAM autocap and `rdkit` SMILES parsing

---

## Documentation

| Topic | Guide |
| --- | --- |
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/installation.md](docs/installation.md) | Install & prerequisites |
| [docs/workflow.md](docs/workflow.md) | End-to-end workflow |
| [docs/gpu-profiles.md](docs/gpu-profiles.md) | GPU presets & profile TSV |
| [docs/cli-reference.md](docs/cli-reference.md) | CLI flags & outputs |
| [docs/json-integrator.md](docs/json-integrator.md) | Input JSON editor |
| [docs/tips.md](docs/tips.md) | Tips & troubleshooting |

---

## License & citation

MIT License — see [LICENSE](LICENSE). AlphaFold 3 is licensed separately by Google DeepMind.

> Abramson, J., Adler, J., Dunger, J. *et al.* Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024). https://doi.org/10.1038/s41586-024-07487-w

See [CITATION.cff](CITATION.cff).
