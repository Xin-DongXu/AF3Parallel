# CLI reference

Every command supports `--help` for the full option list. Examples below use the
unified `af3parallel` CLI and assume you run from your AF3 working directory
(`alphafold3/`).

Install first: `pip install -e ".[extras]"` — see [installation.md](installation.md).

---

## `af3parallel run` — multi-GPU executor

```bash
af3parallel run \
    -i ./af_input -o results.tsv \
    --output-dir ./af_output \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models \
    --gpus 0,1,2,3 \
    --memory-profile my_gpu_profile.tsv

af3parallel run ... --no-temporal-waves      # plain VRAM-packed batches
af3parallel run ... --skip-vram-overflow     # drop oversized jobs
af3parallel run ... --max-tokens 4000        # drop by token count
af3parallel run ... --test-only              # dry-run schedule only
af3parallel run ... --no-skip-existing     # re-run existing outputs
```

### Key option groups

**I/O**

| Option | Description |
| --- | --- |
| `-i, --input-dir DIR` | Input JSON directory *(required)* |
| `-o, --output-file FILE` | Streaming TSV log *(required)* |
| `--output-dir DIR` | AF3 per-job output root (default `./af_output_parallel`) |
| `--temp-dir DIR` | Staging workspace (default `<input-dir>/gpu_work/`) |

**AF3 configuration**

| Option | Description |
| --- | --- |
| `--sif FILE` | Singularity image *(required)* |
| `--af3-db DIR` | Genetic database directory |
| `--models DIR` | Model weights directory |
| `--norun-data-pipeline` | Skip MSA stage (inputs must contain MSAs) |
| `--af3-extra-args ...` | Extra flags passed to `run_alphafold.py` |

**GPU and scheduling**

| Option | Default | Description |
| --- | --- | --- |
| `--gpus LIST` | all visible | Comma-separated GPU indices |
| `--gpu-preset PRESET` | auto | Built-in profile key |
| `--memory-profile FILE` | built-in | Custom TSV from profiler |
| `--vram-margin` | 0.95 | Fraction of physical VRAM to budget |
| `--safety-margin` | 0.10 | Additional shrink on top of vram-margin |
| `--max-batch-runtime` | 7200 | Wall-clock budget per batch (s) |
| `--task-timeout` | 7200 | Hard per-task timeout (s) |
| `--max-concurrent-tasks N` | auto | Global subprocess cap across all GPUs |
| `--memory-estimation-factor` | 1.0 | VRAM estimate multiplier (try 1.1–1.2 on OOM) |

**Temporal waves**

| Option | Default | Description |
| --- | --- | --- |
| `--no-temporal-waves` | off | Disable wave scheduling |
| `--min-anchor-ratio` | 2.0 | Anchor must be this many times longer than wave tasks |
| `--max-anchor-group-ratio` | 1.5 | Max anchor runtime heterogeneity on one GPU |

Run `af3parallel run --help` for the complete grouped reference.

---

## `af3parallel profile` — peak VRAM profiler

```bash
af3parallel profile \
    -i ./profile_inputs -o profile.tsv \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models \
    --interval 1.0
```

---

## `af3parallel profile-ts` — time-series VRAM profiler

```bash
af3parallel profile-ts \
    --stat-file AF3_A800_80G_All_Len_stat.txt \
    --input-dir ./af_input \
    --output-dir ./timeseries_output \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models \
    --monitor-interval 0.5 \
    --n-per-step 3 --workers 8
```

---

## `af3parallel estimate-gpu` — GPU runtime estimator

```bash
af3parallel estimate-gpu \
    --input-dir  ./af_input \
    --profile    my_gpu_profile.tsv \
    --output-tsv breakdown.tsv \
    --workers    16
# stdout: total estimated serial runtime in seconds
```

---

## `af3parallel estimate-cpu` — CPU/MSA runtime estimator

```bash
af3parallel estimate-cpu \
    --input-dir  ./af_input \
    --profile    af3_CPU_Test_stat.tsv \
    --output-tsv msa_breakdown.tsv \
    --workers    16
```

---

## `af3parallel monitor` — GPU memory monitor

```bash
af3parallel monitor -o gpu_log.tsv -i 5 -t 100 -d 300

af3parallel monitor -o gpu_log.tsv -g 0,2 -d 0
```

---

## Output formats

The main `af3parallel run` per-task TSV columns:

```
gpu_id  batch_id  is_retry  batch_type  wave_id
batch_peak_memory_mb  batch_runtime_seconds
task_id  json_file  protein_name
token_count  protein_length  rna_length  dna_length  ligand_count  total_sequences
estimated_memory_mb  estimated_runtime_s
task_peak_memory_mb  runtime_seconds
timeout_risk  success  timestamp
```

Records are written immediately as each task finishes.

See also [JSON Integrator](json-integrator.md) and [tips and gotchas](tips.md).
