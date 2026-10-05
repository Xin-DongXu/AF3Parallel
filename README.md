# AF3Parallel

[![PyPI](https://img.shields.io/pypi/v/af3parallel.svg)](https://pypi.org/project/af3parallel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AlphaFold 3](https://img.shields.io/badge/AlphaFold_3-v3.0.1-green.svg)](https://github.com/google-deepmind/alphafold3)

Process-level scheduler for large-scale [AlphaFold 3](https://github.com/google-deepmind/alphafold3) inference on multi-GPU Linux systems.

AF3Parallel does not modify AF3 weights, kernels, or the Singularity image. Each prediction is an independent `run_alphafold.py` process. The scheduler tokenises inputs, assigns them by longest-processing-time (LPT), and packs concurrent jobs under a measured VRAM envelope, with optional temporal-wave co-scheduling. All utilities are invoked as `af3parallel <command>`.

**Package:** https://pypi.org/project/af3parallel/ · **Source:** https://github.com/Xin-DongXu/AF3Parallel

<p align="center">
  <img src="docs/images/scheduling-overview.png" alt="AF3Parallel scheduling: measured VRAM profiles, MSA cache, LPT allocation, anchor and wave packing, and temporal-wave execution" width="100%">
</p>

*Figure. Measured peak-memory and runtime profiles (top); MSA cache construction (1); token-balanced LPT assignment (2); anchor selection and VRAM packing (3–5); temporal-wave execution (6).*

---

## Scope

The official AF3 release executes one job per GPU. AF3Parallel is the layer that turns a directory of JSON inputs into a multi-GPU campaign:

1. **Profile.** Peak VRAM and runtime are tabulated against token count. Presets `a800-80g` and `rtx4090` were measured on NVIDIA A800 80 GB and RTX 4090 24 GB. Other devices require `af3parallel profile`.
2. **Assign.** Jobs are placed across GPUs by LPT so token load stays balanced.
3. **Pack.** On each GPU, additional jobs are admitted while the sum of profiled peaks fits the memory budget. A long job is the anchor; shorter jobs may run as temporal waves inside that window.
4. **Launch.** Each job is a separate Singularity process. Model weights are not shared across concurrent jobs.

Inference scheduling (`af3parallel run`) expects MSA-bearing JSON, or empty MSA for designed sequences. MSA generation is a separate stage (`af3parallel msa`, or [ColabFold](https://github.com/sokrypton/ColabFold)) and is not included in inference wall-clock.

---

## Requirements

Complete the [AlphaFold 3 v3.0.1 installation](https://github.com/google-deepmind/alphafold3/blob/v3.0.1/docs/installation.md) before using this package (Singularity image, model weights, genetic databases). Layout notes: [docs/installation.md](docs/installation.md).

| Requirement | Role |
| --- | --- |
| Linux, NVIDIA GPU, compute capability ≥ 8.0 | Execution host |
| AF3 v3.0.1 working directory | Contains `run_alphafold.py`, the `.sif`, and `models/` |
| Python ≥ 3.8 | Host-side scheduler |
| `nvidia-smi` | Occupancy monitoring |

Optional extras (`psutil`, `rdkit`, `tqdm`) are installed with `af3parallel[extras]`. `psutil` enables an automatic host-memory cap on concurrent tasks.

---

## Installation

```bash
pip install "af3parallel[extras]"
af3parallel --version
```

From source:

```bash
git clone https://github.com/Xin-DongXu/AF3Parallel.git
cd AF3Parallel
pip install -e ".[extras]"
```

Commands that invoke AF3 (`run`, `msa`, `profile`, `profile-ts`) must be started from the AF3 working directory, or given absolute paths for `--sif`, `--af3-db`, `--models`, and `--af3-home`.

---

## Usage

Set the AF3 paths once, then follow the branch that matches the inputs.

```bash
cd /path/to/alphafold3

export SIF=alphafold3.sif
export AF3_DB="${HOME}/af3_DB"
export MODELS=./models
# a800-80g | rtx4090
export GPU_PRESET=a800-80g
```

### A. Natural proteins

Sequence-only JSON is written without MSA keys so the data pipeline can populate them. Inference is a later step.

```bash
af3parallel build-json natural \
    --fasta ./proteome.fasta --ccd ATP --out-dir ./json_seq_only

af3parallel msa \
    --input-dir ./json_seq_only \
    --output-dir ./msa_af_output \
    --harvest-dir ./json_with_msa \
    --sif "${SIF}" --af3-db "${AF3_DB}" --models "${MODELS}" \
    --af3-home .

af3parallel run \
    -i ./json_with_msa -o results.tsv --output-dir ./af_output \
    --sif "${SIF}" --af3-db "${AF3_DB}" --models "${MODELS}" \
    --gpus 0,1,2,3 --gpu-preset "${GPU_PRESET}"

af3parallel stats -i ./af_output -r -o detail.csv -s summary.csv
```

A new ligand against the same proteins reuses the MSA cache:

```bash
af3parallel json replace-ligand \
    --input-dir ./json_with_msa --in-place \
    --target-id L --smiles "CC(=O)Nc1ccc(O)cc1" \
    --ligand-tag paracetamol --workers 8
```

### B. Designed sequences

`build-json denovo` writes an empty MSA. Skip `msa`.

```bash
af3parallel build-json denovo --fasta ./designs.fasta --out-dir ./json_denovo

af3parallel run \
    -i ./json_denovo -o results.tsv --output-dir ./af_output \
    --sif "${SIF}" --af3-db "${AF3_DB}" --models "${MODELS}" \
    --gpus 0,1,2,3 --gpu-preset "${GPU_PRESET}"
```

### C. Existing MSA JSON

If inputs already contain MSA (or an empty de novo MSA), only `run` is required. Add `--test-only` to print the GPU assignment, batches, and waves without launching AF3.

```bash
af3parallel run \
    -i ./json_with_msa -o results.tsv --output-dir ./af_output \
    --sif "${SIF}" --af3-db "${AF3_DB}" --models "${MODELS}" \
    --gpus 0,1,2,3 --gpu-preset "${GPU_PRESET}" \
    --test-only
```

### D. Unprofiled hardware

Do not rely on VRAM auto-detection. Measure a profile once and pass it explicitly.

```bash
af3parallel profile \
    -i ./profile_inputs -o my_gpu_profile.tsv \
    --sif "${SIF}" --af3-db "${AF3_DB}" --models "${MODELS}"

af3parallel run ... --memory-profile my_gpu_profile.tsv
af3parallel estimate-gpu \
    -i ./json_with_msa -p my_gpu_profile.tsv \
    -o estimate_breakdown.tsv --workers 16
```

`estimate-gpu` reports the sum of profiled serial task times. It is a planning figure, not a measured one-GPU wall-clock.

---

## Scheduling

| Control | Default | Effect |
| --- | --- | --- |
| `--gpu-preset` | auto from `nvidia-smi` | `a800-80g` or `rtx4090` when the device matches a measured profile |
| `--memory-profile` | — | Overrides the preset with a user TSV |
| `--safety-margin` | `0.10` | Fraction of the packing budget held in reserve |
| `--vram-margin` | `0.95` | Fraction of physical VRAM treated as usable for overflow classification |
| temporal waves | on | Disable with `--no-temporal-waves` |
| `--max-workers 1` | off | One job per GPU (exclusive execution; no packing) |
| `--norun-data-pipeline` | passed by `run` | Inference only; MSA must already be in the JSON |

Wave co-scheduling increases campaign throughput when residual VRAM remains after the anchor. It also lengthens per-task runtime, because concurrent processes share the device. On 24 GB cards the residual after a long anchor is often too small for waves to help; packing without waves is then the appropriate setting.

`af3parallel run` streams a per-task TSV (`-o`). Failed tasks can be retried after the main schedule. SIGINT restores staged JSON files to the input directory.

---

## Commands

`af3parallel <command> --help` prints the full option list. Standalone entry points (`af3parallel-run`, …) call the same functions.

| Command | Function |
| --- | --- |
| `build-json` | AF3 JSON from FASTA, PDB, or a manifest (`natural` or `denovo`) |
| `msa` | Parallel AF3 data pipeline |
| `run` | Multi-GPU inference |
| `json` | Edit existing JSON (seeds, ligands, nucleic acids, ions) |
| `stats` | Tables from `*_summary_confidences.json` |
| `profile` | Peak-VRAM profile for a new GPU |
| `profile-ts` | VRAM time series during inference |
| `estimate-gpu` | Serial GPU-time estimate from a token profile |
| `estimate-cpu` | Data-pipeline time estimate from a length profile |
| `monitor` | `nvidia-smi` logger |

Extended notes live under [docs/](docs/README.md): [inputs](docs/build-inputs.md), [MSA](docs/msa.md), [workflow](docs/workflow.md), [profiles](docs/gpu-profiles.md), [CLI](docs/cli-reference.md), [JSON editor](docs/json-integrator.md), [statistics](docs/result-stats.md), [operational notes](docs/tips.md).

---

## Citation

MIT License — see [LICENSE](LICENSE). AlphaFold 3 is distributed separately by Google DeepMind.

> Abramson, J., Adler, J., Dunger, J. *et al.* Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature* **630**, 493–500 (2024). https://doi.org/10.1038/s41586-024-07487-w

Software citation metadata: [CITATION.cff](CITATION.cff).
