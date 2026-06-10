# AF3 JSON Integrator

`af3parallel json` is a protein-preserving editor for AlphaFold 3 input JSON
files ([alphafold3 dialect](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)).

Designed for **screening workflows**: one protein evaluated against many ligands,
nucleic-acid partners, metal ions, or random seeds — without modifying protein
chains.

> The legacy `alphafoldserver` dialect (top-level JSON list) is **not** supported.

Standalone alias: `af3parallel-json`.

---

## Subcommands

| Subcommand | Effect |
| --- | --- |
| `set-seeds` | Modify only `modelSeeds`; `name` stays byte-identical |
| `add-ligand` | Append SMILES or CCD ligand |
| `replace-ligand` | Replace ligand by chain ID |
| `add-nucleic` | Append RNA or DNA chain |
| `replace-nucleic` | Replace RNA/DNA by chain ID |
| `add-ion` | Append metal ion (CCD code, e.g. `MG`, `ZN`) |
| `replace-ion` | Replace ion by chain ID |

---

## I/O modes

| Mode | Flags | Use case |
| --- | --- | --- |
| **Single** | `-i FILE -o FILE` | One input → one output |
| **Bulk** | `--input-dir DIR --output-dir DIR` | Same operation on every `*.json` |
| **Fan-out** | `-i FILE --from-csv CSV --output-dir DIR` | One base JSON → many outputs |

`set-seeds` does not support Fan-out mode.

Bulk and Fan-out modes run in parallel via `--workers N` (default:
`min(8, os.cpu_count())`). Exit code: `0` success, `2` partial failure, `1`
argument error.

### `--in-place`

Edit files without copying. Works with `-i FILE` or `--input-dir DIR`.
Incompatible with `-o`, `--output-dir`, and `--from-csv`. Writes are atomic
(temp-file + rename).

---

## Name field convention

The `name` field is `<protein_prefix>_<ligand_tag>`, split on the **first**
underscore:

- Protein prefix must not contain underscores
- `set-seeds` does not modify `name`
- Other operations rewrite only the `<ligand_tag>` suffix
- Bulk/Fan-out output filenames derive from `name`; collisions abort before writing

---

## CSV manifest (Fan-out mode)

Header row required. Per-task parameters come from the CSV — corresponding CLI
flags (`--smiles`, `--ccd`, …) are **rejected** in Fan-out mode.

| Operation | Required columns | Optional columns |
| --- | --- | --- |
| `add-ligand` / `replace-ligand` | `ligand_tag`, `smiles` or `ccd`; `target_id` (replace) | `copies`, `chain_id` |
| `add-nucleic` / `replace-nucleic` | `ligand_tag`, `type` (`rna`/`dna`), `sequence`; `target_id` (replace) | `copies`, `chain_id` |
| `add-ion` / `replace-ion` | `ligand_tag`, `ccd`; `target_id` (replace) | `copies`, `chain_id` |

The `ccd` column accepts comma-separated codes for multi-component ligands
(e.g. `ATP,MG`).

See [`examples/ligands.csv`](../examples/ligands.csv) for a SMILES screen template.

---

## Examples

**Set seeds**

```bash
af3parallel json set-seeds \
    -i input.json -o output.json --seeds 1 2 3

af3parallel json set-seeds \
    --input-dir ./inputs --in-place \
    --num-seeds 5 --rng-seed 42 --workers 8
```

**Add ligand (bulk, in place)**

```bash
af3parallel json add-ligand \
    --input-dir ./inputs --in-place \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --ligand-tag paracetamol --workers 8
```

**Replace ligand (bulk, in place — same change on every JSON in a directory)**

```bash
af3parallel json replace-ligand \
    --input-dir ./inputs --in-place \
    --target-id L \
    --smiles "CC(=O)Nc1ccc(O)cc1" \
    --ligand-tag paracetamol \
    --workers 8
```

`--target-id` is the chain ID of the ligand to replace (e.g. `L`).

**Replace ligand (fan-out — one base JSON, many outputs from CSV)**

```bash
af3parallel json replace-ligand \
    -i base.json --from-csv examples/ligands.csv \
    --output-dir ./outputs --workers 16
```

**Add nucleic acid**

```bash
af3parallel json add-nucleic \
    -i base.json -o base_with_RNA.json \
    --type rna --sequence GGGAAACCC --ligand-tag hairpin
```

**Add metal ions**

```bash
af3parallel json add-ion \
    -i base.json -o base_with_2MG.json \
    --ccd MG --copies 2 --ligand-tag 2MG
```

For the full per-subcommand option list:

```bash
af3parallel json <subcommand> --help
```
