# Build AF3 input JSON

`af3parallel build-json` creates AlphaFold 3 alphafold3-dialect JSON from
FASTA, PDB, or a TSV/CSV manifest.

## Natural vs de novo

| Mode | CLI | Protein MSA fields | Typical use |
| --- | --- | --- | --- |
| **natural** | `build-json natural` | *Omitted* (`id` + `sequence` only) | Proteomes / evolutionary proteins; run data pipeline next |
| **denovo** | `build-json denovo` | `unpairedMsa=""`, `pairedMsa=""`, `templates=[]` | Designed / *de novo* sequences; skip MSA |

AF3 computes MSA/templates only when those keys are absent. Empty strings
tell AF3 to skip search — required for designed proteins where MSA is
misleading or unavailable.

This tool does **not** run ProteinMPNN / LigandMPNN. Run your design
pipeline separately, then pass FASTA or PDB here.

## Examples

```bash
# Natural monomer + CCD ligand
af3parallel build-json natural \
    --fasta prot.fa --ccd ATP --ligand-id L --out-dir ./json_seq_only

# Natural batch from directory (one JSON per FASTA record)
af3parallel build-json natural \
    --fasta-dir ./fastas --pattern '*.fasta' --out-dir ./json_seq_only

# De novo multimer from multi-record FASTA
af3parallel build-json denovo \
    --fasta dimer.fa --multimer --chains A,B --out-dir ./json_denovo

# De novo from PDB chains + SMILES
af3parallel build-json denovo \
    --pdb complex.pdb --protein-chains A,B \
    --smiles 'c1ccccc1' --ligand-id C --out-dir ./json_denovo

# Manifest (TSV): name, fasta|sequence, optional ccd|smiles, ligand_id
af3parallel build-json natural --manifest jobs.tsv --out-dir ./json_seq_only
```

Next steps:

- Natural → [`msa.md`](msa.md) or ColabFold, then `af3parallel run`
- De novo → `af3parallel run` with inference-only flags / empty-MSA JSON

Related: edit existing JSON with [`json-integrator.md`](json-integrator.md).
