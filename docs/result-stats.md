# Result statistics

`af3parallel stats` walks AF3 output trees for `*_summary_confidences.json`
and writes detail (+ optional summary) CSV.

## Modes

| `--mode` | Use when | Primary metrics |
| --- | --- | --- |
| `ligand` (default) | Protein(s) + ligand | Ligand–protein pair ipTM/PAE; `iptm_final` = max pair |
| `general` | Monomer / any complex | Global `ptm`, `iptm`, `ranking_score` |

Recursive scans **skip** intermediate `seed-*` folders unless
`--include-seed-dirs` is set.

```bash
# Ligand screen (ligand = last entity by default)
af3parallel stats \
    -i ./af_output -r \
    -o ligand_detail.csv -s ligand_summary.csv \
    --ligand-index -1

# General scores
af3parallel stats --mode general \
    -i ./af_output -r \
    -o scores.csv -s scores_summary.csv \
    --rank-metric iptm
```

Optional: `pip install tqdm` for a progress bar (`af3parallel[extras]`).
