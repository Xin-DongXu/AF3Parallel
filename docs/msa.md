# MSA / data pipeline

Two practical routes are supported. Pick one per project (do not mix blindly).

## 1. Parallel AF3 data pipeline (CPU)

`af3parallel msa` runs official AF3 `run_alphafold.py --norun_inference` via
Singularity/Apptainer. Concurrency is a fixed worker pool; when a job
finishes, the next pending JSON starts immediately (gap filling).

**Default workers = `max(1, cpu_count // 4)`.** Override with `--workers`.

```bash
af3parallel msa \
    --input-dir ./json_seq_only \
    --output-dir ./msa_af_output \
    --harvest-dir ./json_with_msa \
    --log-dir ./logs/msa_jobs \
    --sif alphafold3.sif \
    --af3-db ~/af3_DB \
    --models ./models \
    --af3-home /path/to/alphafold3
# optional: --workers 12 --limit 1   # smoke test one job
```

Inputs must be **natural-mode** JSON (no empty MSA keys). Harvested
`json_with_msa/` is ready for `af3parallel run` (inference).

## 2. ColabFold / MMseqs2 (GPU, external)

Often faster for large proteomes on GPU nodes. AF3Parallel does not vendor
ColabFold — install and follow upstream docs:

- https://github.com/sokrypton/ColabFold
- https://github.com/steineggerlab/colabfold

After search, place MSA strings into AF3 JSON `unpairedMsa` / `pairedMsa`
(or convert with your own helper), then run inference with
`--norun_data_pipeline`.

## De novo proteins

Skip this stage. Use `af3parallel build-json denovo` so MSA fields are
already empty, then go straight to `af3parallel run`.
