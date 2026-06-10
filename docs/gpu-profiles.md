# GPU profiles and TSV format

## Built-in GPU profiles

`af3parallel run` ships with built-in profiles measured on two GPU models.
**Only these two are officially supported**; other preset keys in `--help` are
unmeasured extrapolations.

| Preset key | VRAM | Tested on |
| --- | --- | --- |
| `a800-80g` | 80 GB | NVIDIA A800 80 GB |
| `rtx4090` | 24 GB | NVIDIA RTX 4090 24 GB |

```bash
af3parallel run --gpu-preset a800-80g ...
af3parallel run --gpu-preset rtx4090 ...
```

### Custom profiles

Any other GPU requires a custom TSV from `af3parallel profile`:

```bash
af3parallel profile \
    -i ./profile_inputs -o my_gpu_profile.tsv \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB --models ./models

af3parallel run ... --memory-profile my_gpu_profile.tsv
```

### Auto-detection fallback

If neither `--gpu-preset` nor `--memory-profile` is set, `af3parallel run`
queries `nvidia-smi`, matches VRAM against the preset table (±512 MB), and
falls back to the A800 profile with a warning. **This is not a substitute for
proper profiling on untested hardware.**

---

## Profile TSV format

Profilers and estimators exchange data through a tab-separated file with this
header:

```
token_count    peak_memory_mb    runtime_seconds    success
```

`af3parallel profile` writes additional diagnostic columns (`json_file`,
`protein_name`, `protein_length`, `ligand_count`, `total_sequences`,
`timestamp`); loaders tolerate and ignore them. Rows with `success=False` are
discarded unless explicitly enabled. Sparse profiles (non-contiguous token
counts) are filled by linear interpolation between adjacent measurements unless
`--no-profile-gap-fill` is set on `af3parallel run`.

---

## CPU/MSA profiles

`af3parallel estimate-cpu` uses a separate profile keyed on the **longest protein
chain length** (MSA cost is driven by the largest protein passed to
jackhmmer/hhblits), not AF3 token count.

```bash
af3parallel estimate-cpu \
    --input-dir  ./af_input \
    --profile    af3_CPU_Test_stat.tsv \
    --output-tsv msa_breakdown.tsv \
    --workers    16
```
