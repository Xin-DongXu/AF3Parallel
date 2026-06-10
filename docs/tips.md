# Tips and gotchas

## AlphaFold 3 flags

AlphaFold 3 uses [Abseil flags](https://abseil.io/docs/python/guides/flags).
Boolean negations are written `--noFLAG` (no separator). Pass extra AF3 flags
with `--af3-extra-args`, or use `--norun-data-pipeline` on `af3parallel run`.

`af3parallel profile-ts` auto-corrects common wrong forms.

## Before large runs

Run `af3parallel run` with `--test-only` to validate config and inspect the
planned schedule (GPU preset, concurrency cap, batches, temporal waves) without
invoking AF3.

## Untested GPUs

Without `--memory-profile`, unknown GPUs fall back to the A800 profile after a
warning. Profile once with `af3parallel profile` for accurate scheduling.

## CPU RAM exhaustion

On many-small-task workloads (hundreds of <500-token jobs), the **global**
concurrency cap prevents OS swap thrashing. Install `psutil` (`pip install
"af3parallel[extras]"`) for auto-cap, or set `--max-concurrent-tasks` explicitly.

## Interrupted runs

On SIGINT/SIGTERM, `af3parallel run` restores staged JSON files to the input
directory. After an unclean kill, check `./<input-dir>/gpu_work/` (or
`--temp-dir`) for stragglers.

## Mixed workloads

For many small jobs plus a few large ones, keep temporal waves enabled
(default). Long anchor jobs run while successive waves of light tasks fill unused
VRAM inside the anchor runtime window.

## VRAM near the ceiling

If you observe OOMs near the predicted limit, raise
`--memory-estimation-factor` to `1.1`–`1.2`.

## Streaming logs

All commands write structured TSV. The main executor log is updated per task, so
partial results remain useful if a run is interrupted.
