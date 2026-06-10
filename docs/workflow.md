# Typical workflow

The tools are designed to work together: profile your hardware once, estimate
batch runtime ahead of time, then run the batch in parallel across all available
GPUs with VRAM-aware scheduling.

```
                   +--------------------------------+
                   | AF3 input JSONs                |
                   | (af3parallel json              |
                   |  or hand-written)              |
                   +---------------+----------------+
                                   |
        +--------------------------+--------------------------+
        |                                                     |
        v                                                     v
+---------------------+                            +-----------------------+
| af3parallel profile |  one-shot, per GPU model   | af3parallel           |
|                     | -----TSV profile---------> | estimate-gpu/cpu      |
| (run once per GPU)  |                            | (estimate batch wall) |
+----------+----------+                            +-----------------------+
           |
           | TSV profile (token_count, peak_memory_mb, runtime_seconds)
           v
+-------------------------------------------+
| af3parallel run                           |
|   - LPT distribution across GPUs          |
|   - VRAM-aware batching                   |
|   - Temporal-wave scheduling              |
|   - Streaming TSV log                     |
+-------------------------------------------+
```

## End-to-end example

All commands below assume your current directory is the AF3 working tree
(`alphafold3/`).

### 1. Profile once per GPU model

Skip this step if your GPU matches a [built-in profile](gpu-profiles.md)
(`a800-80g` or `rtx4090`).

```bash
af3parallel profile \
    -i ./profile_inputs \
    -o my_gpu_profile.tsv \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB \
    --models ./models
```

> Monitors GPU 0 by default. Prefix with `CUDA_VISIBLE_DEVICES=<index>` to
> profile a different GPU.

### 2. (Optional) Estimate runtime

```bash
af3parallel estimate-gpu \
    --input-dir ./af_input \
    --profile   my_gpu_profile.tsv \
    --output-tsv estimate_breakdown.tsv \
    --workers   16
```

For CPU/MSA stage estimates, use `af3parallel estimate-cpu` with a
protein-length profile instead.

### 3. Run the batch across all GPUs

```bash
af3parallel run \
    -i  ./af_input \
    -o  results.tsv \
    --output-dir ./af_output \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB \
    --models ./models \
    --gpus 0,1,2,3 \
    --memory-profile my_gpu_profile.tsv
```

### 4. Dry-run before large batches

```bash
af3parallel run ... --test-only
```

Prints the resolved GPU preset, global concurrency cap, per-GPU task
distribution, and planned batches / temporal waves without invoking AF3.

---

## Command roles

| Command | Role |
| --- | --- |
| `af3parallel run` | Main multi-GPU executor |
| `af3parallel profile` | One-shot peak-VRAM profiling |
| `af3parallel profile-ts` | Sub-second VRAM time-series profiling |
| `af3parallel estimate-gpu` | Serial GPU runtime estimator |
| `af3parallel estimate-cpu` | CPU/MSA runtime estimator |
| `af3parallel json` | Batch input JSON editor |
| `af3parallel monitor` | Standalone `nvidia-smi` memory logger |

See [CLI reference](cli-reference.md) for common flags and [JSON Integrator](json-integrator.md) for input preparation workflows.
