# Installation

AF3Parallel wraps the official AlphaFold 3 v3.0.1 Singularity workflow. AlphaFold 3 itself must be installed separately.

## Prerequisites

Complete the [official AF3 installation guide](https://github.com/google-deepmind/alphafold3/blob/v3.0.1/docs/installation.md) first.

| Component | Typical path |
| --- | --- |
| AF3 source tree (v3.0.1) | `./alphafold3/` |
| Singularity image | `./alphafold3/alphafold3.sif` |
| Model weights | `./alphafold3/models/` |
| Genetic databases | `~/af3_DB/` (outside the AF3 repo) |
| NVIDIA GPU + driver | `nvidia-smi` in `PATH` |

**Hardware:** Linux, NVIDIA GPU (CC ≥ 8.0), ≥ 64 GB RAM recommended, ≥ 1 TB SSD for databases.

---

## pip install (recommended)

**PyPI:** https://pypi.org/project/af3parallel/

```bash
pip install "af3parallel[extras]"
```

Optional extras:

| Package | Purpose |
| --- | --- |
| `psutil` | Auto `--max-concurrent-tasks` cap in `af3parallel run` |
| `rdkit` | Accurate SMILES heavy-atom counts |

Verify:

```bash
af3parallel --version
af3parallel --help
af3parallel run --help
```

### Unified CLI

| Command | Tool |
| --- | --- |
| `af3parallel run` | Multi-GPU batch executor |
| `af3parallel profile` | Peak VRAM profiler |
| `af3parallel profile-ts` | Time-series VRAM profiler |
| `af3parallel estimate-gpu` | GPU runtime estimator |
| `af3parallel estimate-cpu` | CPU/MSA runtime estimator |
| `af3parallel json` | Input JSON integrator |
| `af3parallel monitor` | GPU memory monitor |

Individual entry points: `af3parallel-run`, `af3parallel-profile`, …

Example (from your AF3 working directory):

```bash
cd /path/to/alphafold3

af3parallel run \
    -i ./af_input -o results.tsv --output-dir ./af_output \
    --sif alphafold3.sif --af3-db ~/af3_DB --models ./models \
    --gpus 0,1,2,3 --memory-profile my_gpu_profile.tsv
```

---

## Working directory layout

```
~/af3_DB/                          # genetic databases
/path/to/alphafold3/
    alphafold3.sif
    models/
    af_input/
    af_output/
```

See [workflow.md](workflow.md) for the full pipeline.
