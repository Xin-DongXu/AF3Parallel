#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build AlphaFold 3 input JSON from sequences / FASTA / PDB.

Two MSA modes (critical distinction)
------------------------------------
**natural** (default for proteome / evolutionary proteins)
    Protein entries contain *only* ``id`` + ``sequence`` (no MSA/template
    keys). AF3's data pipeline then computes MSA + templates.

**denovo** (designed proteins, *de novo* folds, MPNN designs, …)
    Protein entries include empty ``unpairedMsa`` / ``pairedMsa`` and
    ``templates: []``. AF3 skips the data pipeline for those chains; use
    with ``--norun_data_pipeline`` (or let AF3 detect empty MSA and skip).

Ligands are optional (CCD codes and/or SMILES). This tool does **not**
run LigandMPNN / ProteinMPNN — feed sequences (FASTA) or PDB-extracted
chains produced by your own design pipeline.

Examples
--------
  # Natural monomer + ATP (MSA will be computed later)
  af3parallel build-json natural \\
      --fasta prot.fa --ccd ATP --out-dir ./af_input

  # De novo dimer apo from FASTA (empty MSA, inference-ready)
  af3parallel build-json denovo \\
      --fasta designs.fa --chains A,B --out-dir ./af_input_denovo

  # De novo complex from PDB + SMILES ligand
  af3parallel build-json denovo \\
      --pdb complex.pdb --protein-chains A,B \\
      --smiles 'c1ccccc1' --ligand-id C --out-dir ./af_input

  # Batch from TSV (columns: name, fasta|sequence[, ccd|smiles, ligand_id])
  af3parallel build-json natural --manifest jobs.tsv --out-dir ./af_input
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name)[:200]


def parse_chain_list(s: str) -> List[str]:
    chains = [c.strip() for c in s.split(",") if c.strip()]
    if not chains:
        raise argparse.ArgumentTypeError("chain list must not be empty")
    return chains


def generate_model_seeds(n: int, rng_seed: Optional[int]) -> List[int]:
    if n < 1:
        raise ValueError("--n-model-seeds must be >= 1")
    rng = random.Random(rng_seed)
    return rng.sample(range(1, 2_147_483_647), k=n)


def parse_fasta(text: str) -> List[Tuple[str, str]]:
    """Return list of (header, sequence) from FASTA text."""
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    chunks: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                seq = "".join(chunks).upper().replace(" ", "")
                if seq:
                    records.append((header, seq))
            header = line[1:].strip() or f"seq{len(records)+1}"
            chunks = []
        else:
            chunks.append(re.sub(r"[^A-Za-z]", "", line))
    if header is not None:
        seq = "".join(chunks).upper().replace(" ", "")
        if seq:
            records.append((header, seq))
    return records


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    return parse_fasta(path.read_text(encoding="utf-8", errors="replace"))


def fasta_id(header: str) -> str:
    """First whitespace-delimited token; strip UniProt-style junk."""
    tok = header.split()[0] if header else "protein"
    # sp|P12345|NAME → P12345
    if "|" in tok:
        parts = tok.split("|")
        if len(parts) >= 2 and parts[1]:
            tok = parts[1]
    return sanitize_filename(tok)


def _atom_fields(line: str) -> Optional[Tuple[str, str, str, int, str]]:
    if len(line) < 26:
        return None
    record = line[0:6].strip()
    if record not in ("ATOM", "HETATM"):
        return None
    name = line[12:16].strip()
    resname = line[17:20].strip()
    chain = line[21:22].strip() or " "
    try:
        resseq = int(line[22:26])
    except ValueError:
        parts = line.split()
        if len(parts) < 6:
            return None
        name, resname, chain = parts[2], parts[3], parts[4]
        try:
            resseq = int(parts[5])
        except ValueError:
            return None
    return record, name, resname, resseq, chain


def extract_protein_sequences_from_pdb(
    pdb_path: Path,
    protein_chains: Sequence[str],
    *,
    ca_only: bool = True,
) -> Dict[str, str]:
    wanted = set(protein_chains)
    residues: Dict[str, Dict[int, str]] = {c: {} for c in protein_chains}
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _atom_fields(line)
        if parsed is None:
            continue
        _record, name, resname, resseq, chain = parsed
        if chain not in wanted:
            continue
        aa = AA3_TO_1.get(resname.upper())
        if aa is None:
            continue
        if ca_only and name.strip().upper() != "CA":
            continue
        if resseq not in residues[chain]:
            residues[chain][resseq] = aa
    sequences: Dict[str, str] = {}
    for cid in protein_chains:
        if not residues[cid]:
            raise ValueError(
                f"{pdb_path.name}: no standard amino acids for chain {cid!r}"
            )
        order = sorted(residues[cid])
        sequences[cid] = "".join(residues[cid][i] for i in order)
    return sequences


def _protein_entry(
    chain_id: Union[str, List[str]],
    sequence: str,
    *,
    msa_mode: str,
) -> Dict[str, Any]:
    prot: Dict[str, Any] = {
        "id": chain_id,
        "sequence": sequence,
    }
    if msa_mode == "denovo":
        # Explicit empty MSA → AF3 skips jackhmmer / templates for this chain.
        prot["unpairedMsa"] = ""
        prot["pairedMsa"] = ""
        prot["templates"] = []
    elif msa_mode != "natural":
        raise ValueError(f"unknown msa_mode={msa_mode!r}")
    return {"protein": prot}


def _ligand_entry(
    ligand_id: Union[str, List[str]],
    *,
    ccd: Optional[Sequence[str]] = None,
    smiles: Optional[str] = None,
) -> Dict[str, Any]:
    lig: Dict[str, Any] = {"id": ligand_id}
    if smiles:
        lig["smiles"] = smiles
    elif ccd:
        codes = [c.strip().upper() for c in ccd if c and c.strip()]
        if not codes:
            raise ValueError("empty CCD list")
        lig["ccdCodes"] = codes if len(codes) > 1 else codes
        # AF3 accepts either a string or list; prefer list for multi-component.
        if len(codes) == 1:
            lig["ccdCodes"] = codes  # keep list form (AF3-compatible)
    else:
        raise ValueError("ligand requires --ccd or --smiles")
    return {"ligand": lig}


def build_af3_job(
    name: str,
    chain_sequences: Dict[str, str],
    *,
    msa_mode: str,
    model_seeds: Sequence[int],
    af3_version: int = 2,
    homo: bool = False,
    ligand_id: Optional[str] = None,
    ccd: Optional[Sequence[str]] = None,
    smiles: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble one AF3 alphafold3-dialect JSON object."""
    chains = list(chain_sequences.keys())
    sequences: List[Dict[str, Any]] = []
    if homo and len(chains) > 1:
        monomer = chain_sequences[chains[0]]
        for c in chains[1:]:
            if chain_sequences[c] != monomer:
                raise ValueError(
                    f"homo mode requires identical sequences; "
                    f"{chains[0]}!={c}"
                )
        sequences.append(
            _protein_entry(list(chains), monomer, msa_mode=msa_mode)
        )
    else:
        for cid in chains:
            sequences.append(
                _protein_entry(cid, chain_sequences[cid], msa_mode=msa_mode)
            )

    if ligand_id and (ccd or smiles):
        sequences.append(
            _ligand_entry(ligand_id, ccd=ccd, smiles=smiles)
        )

    return {
        "name": name,
        "sequences": sequences,
        "modelSeeds": list(model_seeds),
        "dialect": "alphafold3",
        "version": int(af3_version),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _assign_chain_ids(n: int, explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        if len(explicit) != n:
            raise ValueError(
                f"--chains has {len(explicit)} id(s) but {n} sequence(s)"
            )
        return list(explicit)
    if n > len(CHAIN_IDS):
        raise ValueError(f"need {n} chain IDs; provide --chains explicitly")
    return list(CHAIN_IDS[:n])


def jobs_from_fasta(
    fasta_path: Path,
    *,
    msa_mode: str,
    one_job_per_record: bool,
    chains: Optional[Sequence[str]],
    homo: bool,
    model_seeds: Sequence[int],
    af3_version: int,
    name_prefix: str,
    ligand_id: Optional[str],
    ccd: Optional[Sequence[str]],
    smiles: Optional[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    records = read_fasta(fasta_path)
    if not records:
        raise ValueError(f"no sequences in {fasta_path}")

    jobs: List[Tuple[str, Dict[str, Any]]] = []
    if one_job_per_record:
        for header, seq in records:
            name = f"{name_prefix}{fasta_id(header)}"
            chain_id = (chains[0] if chains else "A")
            payload = build_af3_job(
                name,
                {chain_id: seq},
                msa_mode=msa_mode,
                model_seeds=model_seeds,
                af3_version=af3_version,
                homo=False,
                ligand_id=ligand_id,
                ccd=ccd,
                smiles=smiles,
            )
            jobs.append((name, payload))
    else:
        ids = _assign_chain_ids(len(records), chains)
        name = f"{name_prefix}{fasta_id(records[0][0])}"
        if len(records) > 1:
            name = f"{name_prefix}{sanitize_filename(fasta_path.stem)}"
        chain_sequences = {cid: seq for cid, (_h, seq) in zip(ids, records)}
        payload = build_af3_job(
            name,
            chain_sequences,
            msa_mode=msa_mode,
            model_seeds=model_seeds,
            af3_version=af3_version,
            homo=homo,
            ligand_id=ligand_id,
            ccd=ccd,
            smiles=smiles,
        )
        jobs.append((name, payload))
    return jobs


def jobs_from_pdb(
    pdb_path: Path,
    *,
    protein_chains: Sequence[str],
    msa_mode: str,
    model_seeds: Sequence[int],
    af3_version: int,
    name_prefix: str,
    name_suffix: str,
    ca_only: bool,
    homo: bool,
    ligand_id: Optional[str],
    ccd: Optional[Sequence[str]],
    smiles: Optional[str],
) -> List[Tuple[str, Dict[str, Any]]]:
    seqs = extract_protein_sequences_from_pdb(
        pdb_path, protein_chains, ca_only=ca_only
    )
    name = f"{name_prefix}{sanitize_filename(pdb_path.stem)}{name_suffix}"
    payload = build_af3_job(
        name,
        seqs,
        msa_mode=msa_mode,
        model_seeds=model_seeds,
        af3_version=af3_version,
        homo=homo,
        ligand_id=ligand_id,
        ccd=ccd,
        smiles=smiles,
    )
    return [(name, payload)]


def jobs_from_manifest(
    manifest: Path,
    *,
    msa_mode: str,
    model_seeds: Sequence[int],
    af3_version: int,
    default_ligand_id: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """TSV/CSV with header. Required: name. Sequence via fasta or sequence.

    Optional: chains (comma-separated for multi-seq FASTA as one job),
    ccd, smiles, ligand_id, homo (0/1).
    """
    text = manifest.read_text(encoding="utf-8", errors="replace")
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",\t")
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError(f"empty manifest: {manifest}")
    fields = {f.lower().strip(): f for f in reader.fieldnames}

    def col(*names: str) -> Optional[str]:
        for n in names:
            if n in fields:
                return fields[n]
        return None

    c_name = col("name", "id", "job")
    c_fasta = col("fasta", "fasta_path", "fa")
    c_seq = col("sequence", "seq")
    c_chains = col("chains", "protein_chains")
    c_ccd = col("ccd", "ccd_codes")
    c_smiles = col("smiles")
    c_lig = col("ligand_id", "ligand")
    c_homo = col("homo")
    if not c_name:
        raise ValueError("manifest needs a 'name' column")
    if not c_fasta and not c_seq:
        raise ValueError("manifest needs 'fasta' or 'sequence' column")

    jobs: List[Tuple[str, Dict[str, Any]]] = []
    for row in reader:
        name = (row.get(c_name) or "").strip()
        if not name:
            continue
        homo = False
        if c_homo and (row.get(c_homo) or "").strip() in ("1", "true", "True", "yes"):
            homo = True
        ccd_raw = (row.get(c_ccd) or "").strip() if c_ccd else ""
        smiles = (row.get(c_smiles) or "").strip() if c_smiles else ""
        lig_id = (
            (row.get(c_lig) or "").strip()
            if c_lig
            else (default_ligand_id if (ccd_raw or smiles) else None)
        )
        ccd = [x.strip() for x in ccd_raw.split(",") if x.strip()] if ccd_raw else None

        if c_fasta and (row.get(c_fasta) or "").strip():
            fa = Path((row.get(c_fasta) or "").strip())
            if not fa.is_file():
                raise FileNotFoundError(f"manifest fasta not found: {fa}")
            records = read_fasta(fa)
            chain_arg = None
            if c_chains and (row.get(c_chains) or "").strip():
                chain_arg = parse_chain_list(row[c_chains])
            ids = _assign_chain_ids(len(records), chain_arg)
            chain_sequences = {cid: seq for cid, (_h, seq) in zip(ids, records)}
        else:
            seq = (row.get(c_seq) or "").strip().upper()
            if not seq:
                raise ValueError(f"row {name}: empty sequence")
            cid = "A"
            if c_chains and (row.get(c_chains) or "").strip():
                cid = parse_chain_list(row[c_chains])[0]
            chain_sequences = {cid: seq}

        payload = build_af3_job(
            name,
            chain_sequences,
            msa_mode=msa_mode,
            model_seeds=model_seeds,
            af3_version=af3_version,
            homo=homo,
            ligand_id=lig_id or None,
            ccd=ccd,
            smiles=smiles or None,
        )
        jobs.append((name, payload))
    return jobs


def _add_common_json_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("AF3 JSON")
    g.add_argument("--out-dir", type=Path, required=True)
    g.add_argument("--n-model-seeds", type=int, default=1)
    g.add_argument("--model-seed-rng", type=int, default=None)
    g.add_argument("--model-seeds", type=int, nargs="+", default=None)
    g.add_argument(
        "--af3-version",
        type=int,
        default=2,
        help="alphafold3 dialect version (cluster SIF often accepts 1–2 only)",
    )
    g.add_argument("--prefix", type=str, default="")
    g.add_argument(
        "--homo",
        action="store_true",
        help="Identical multi-chain protein (single sequence, shared MSA slot)",
    )

    gl = p.add_argument_group("ligand (optional)")
    gl.add_argument("--ccd", type=str, default=None,
                    help="CCD code(s), comma-separated (e.g. ATP or NAD,NAP)")
    gl.add_argument("--smiles", type=str, default=None)
    gl.add_argument("--ligand-id", type=str, default="L",
                    help="Ligand chain id (default L)")


def _resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.model_seeds is not None:
        return list(args.model_seeds)
    return generate_model_seeds(args.n_model_seeds, args.model_seed_rng)


def _parse_ccd(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [x.strip().upper() for x in raw.split(",") if x.strip()]


def _write_jobs(
    jobs: List[Tuple[str, Dict[str, Any]]],
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in jobs:
        path = out_dir / f"{sanitize_filename(name)}.json"
        write_json(path, payload)
        n_prot = sum(1 for s in payload["sequences"] if "protein" in s)
        has_lig = any("ligand" in s for s in payload["sequences"])
        print(f"[ok] {path.name}  proteins={n_prot}  ligand={has_lig}")
    return len(jobs)


def cmd_natural(args: argparse.Namespace) -> int:
    """Natural proteins: omit MSA keys so AF3 data pipeline runs."""
    return _cmd_build(args, msa_mode="natural")


def cmd_denovo(args: argparse.Namespace) -> int:
    """De novo / designed: empty MSA + templates so inference can skip MSA."""
    return _cmd_build(args, msa_mode="denovo")


def _cmd_build(args: argparse.Namespace, msa_mode: str) -> int:
    seeds = _resolve_seeds(args)
    ccd = _parse_ccd(args.ccd)
    smiles = args.smiles
    if ccd and smiles:
        print("ERROR: provide either --ccd or --smiles, not both", file=sys.stderr)
        return 2
    lig_id = args.ligand_id if (ccd or smiles) else None

    jobs: List[Tuple[str, Dict[str, Any]]] = []
    try:
        if args.manifest:
            jobs = jobs_from_manifest(
                args.manifest,
                msa_mode=msa_mode,
                model_seeds=seeds,
                af3_version=args.af3_version,
                default_ligand_id=args.ligand_id,
            )
        elif args.pdb:
            if not args.protein_chains:
                print("ERROR: --pdb requires --protein-chains", file=sys.stderr)
                return 2
            suffix = getattr(args, "suffix", "") or ""
            jobs = jobs_from_pdb(
                args.pdb,
                protein_chains=args.protein_chains,
                msa_mode=msa_mode,
                model_seeds=seeds,
                af3_version=args.af3_version,
                name_prefix=args.prefix,
                name_suffix=suffix,
                ca_only=not getattr(args, "all_atom_seq", False),
                homo=args.homo,
                ligand_id=lig_id,
                ccd=ccd,
                smiles=smiles,
            )
        elif args.fasta:
            one_per = bool(getattr(args, "one_job_per_record", True))
            if msa_mode == "denovo" and getattr(args, "multimer", False):
                one_per = False
            jobs = jobs_from_fasta(
                args.fasta,
                msa_mode=msa_mode,
                one_job_per_record=one_per,
                chains=getattr(args, "chains", None),
                homo=args.homo,
                model_seeds=seeds,
                af3_version=args.af3_version,
                name_prefix=args.prefix,
                ligand_id=lig_id,
                ccd=ccd,
                smiles=smiles,
            )
        elif args.fasta_dir:
            pattern = getattr(args, "pattern", "*.fa*")
            files = sorted(args.fasta_dir.glob(pattern))
            if not files:
                print(f"ERROR: no FASTA under {args.fasta_dir}", file=sys.stderr)
                return 1
            for fa in files:
                jobs.extend(
                    jobs_from_fasta(
                        fa,
                        msa_mode=msa_mode,
                        one_job_per_record=True,
                        chains=getattr(args, "chains", None),
                        homo=False,
                        model_seeds=seeds,
                        af3_version=args.af3_version,
                        name_prefix=args.prefix,
                        ligand_id=lig_id,
                        ccd=ccd,
                        smiles=smiles,
                    )
                )
        else:
            print(
                "ERROR: provide --fasta, --fasta-dir, --pdb, or --manifest",
                file=sys.stderr,
            )
            return 2
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not jobs:
        print("ERROR: no jobs produced", file=sys.stderr)
        return 1

    print(
        f"[INFO] msa_mode={msa_mode}  n_jobs={len(jobs)}  "
        f"modelSeeds={list(seeds)}  out={args.out_dir}"
    )
    if msa_mode == "natural":
        print(
            "[INFO] Natural mode: MSA/template keys omitted → run data pipeline "
            "(af3parallel msa or ColabFold) before inference."
        )
    else:
        print(
            "[INFO] De novo mode: empty unpairedMsa/pairedMsa/templates → "
            "skip MSA; use af3parallel run with --norun-data-pipeline."
        )

    n = _write_jobs(jobs, args.out_dir)
    manifest = {
        "msa_mode": msa_mode,
        "n_json": n,
        "model_seeds": list(seeds),
        "af3_version": args.af3_version,
        "jobs": [name for name, _ in jobs],
    }
    write_json(args.out_dir / "build_manifest.json", manifest)
    print(f"Done. Wrote {n} JSON file(s) → {args.out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="af3parallel build-json",
        description=(
            "Build AF3 input JSON from FASTA/PDB/manifest. "
            "Use 'natural' for evolutionary proteins (MSA later) or "
            "'denovo' for designed sequences (empty MSA, inference-ready)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def add_inputs(sp: argparse.ArgumentParser, *, denovo: bool) -> None:
        gi = sp.add_argument_group("input")
        gi.add_argument("--fasta", type=Path, default=None)
        gi.add_argument("--fasta-dir", type=Path, default=None)
        gi.add_argument("--pattern", type=str, default="*.fa*")
        gi.add_argument("--pdb", type=Path, default=None)
        gi.add_argument("--manifest", type=Path, default=None,
                        help="TSV/CSV: name + fasta|sequence [+ ccd|smiles]")
        gi.add_argument(
            "--chains",
            type=parse_chain_list,
            default=None,
            help="Chain IDs when one FASTA has multiple records as one multimer job",
        )
        gi.add_argument(
            "--protein-chains",
            type=parse_chain_list,
            default=None,
            help="PDB protein chain IDs to extract",
        )
        if denovo:
            gi.add_argument(
                "--multimer",
                action="store_true",
                help="Treat multi-record FASTA as one multimer job (not one-per-record)",
            )
            gi.add_argument("--suffix", type=str, default="",
                            help="Optional name suffix (e.g. _apo)")
            gi.add_argument(
                "--all-atom-seq",
                action="store_true",
                help="PDB: use any atom for residue presence (default: CA only)",
            )
        else:
            gi.add_argument(
                "--one-job-per-record",
                dest="one_job_per_record",
                action="store_true",
                default=True,
                help="Each FASTA record → one JSON (default)",
            )
            gi.add_argument(
                "--one-multimer-job",
                dest="one_job_per_record",
                action="store_false",
                help="All FASTA records → one multimer JSON",
            )
        _add_common_json_args(sp)

    sp_n = sub.add_parser(
        "natural",
        help="Omit MSA keys → AF3 data pipeline computes MSA/templates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_inputs(sp_n, denovo=False)
    sp_n.set_defaults(func=cmd_natural)

    sp_d = sub.add_parser(
        "denovo",
        help="Empty MSA/templates → skip data pipeline (designed proteins)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_inputs(sp_d, denovo=True)
    sp_d.set_defaults(func=cmd_denovo)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Allow `af3parallel build-json natural ...` and bare `python -m ... natural`
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(args_list)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
