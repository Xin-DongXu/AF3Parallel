#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AF3 JSON Integrator
===================

A user-friendly command-line utility for modifying AlphaFold 3 input JSON
files (the alphafold3 dialect, see
https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) in
either single-file or large-scale batch / parallel modes.

The tool is designed for screening workflows in which a single
protein-of-interest is repeatedly evaluated against many different ligands,
nucleic-acid partners, metal ions, or random seeds.

Supported operations
--------------------
    set-seeds         Modify only modelSeeds (everything else, including
                      the name field, stays byte-identical).
    add-ligand        Append a new small-molecule ligand entry (SMILES or CCD).
    replace-ligand    Replace an existing small-molecule ligand entry.
    add-nucleic       Append a new RNA or DNA chain.
    replace-nucleic   Replace an existing RNA or DNA chain.
    add-ion           Append a new metal ion (CCD code only; per AF3 spec
                      ions are simply ligands with codes such as MG, ZN, ...).
    replace-ion       Replace an existing metal ion (CCD code only).

Three I/O modes per operation
-----------------------------
    Single   :  -i FILE  -o FILE
    Bulk     :  --input-dir DIR  --output-dir DIR
                (same operation applied to every *.json in the directory)
    Fan-out  :  -i FILE  --from-csv CSV  --output-dir DIR
                (one base JSON, one output per CSV row)

In Bulk and Fan-out modes the script runs tasks in parallel using
--workers N (default: min(8, os.cpu_count() or 1)). Failed tasks
are logged but never abort the whole run; the final summary lists
success / failure counts and the first few error messages.

Name-field convention
---------------------
The name field is treated as <protein_prefix>_<ligand_tag>, split
on the first underscore. Therefore the protein-ID prefix should not
itself contain underscores. set-seeds does not modify name;
every other operation rewrites only the ligand-tag suffix.

Invariants enforced by every operation
--------------------------------------
1. Protein chains are never touched. Every transformed JSON is
   deep-compared against the input for protein-entry equality.
2. The protein-ID prefix in name is preserved verbatim.
3. set-seeds is name-stable.
4. Chain IDs are kept unique across the whole sequences list.
5. In Bulk and Fan-out modes the output filename is derived from the
   resulting name field (<name>.json); collisions abort the run
   before writing anything to disk.

CSV manifest format (Fan-out mode)
----------------------------------
The CSV must have a header row. Required and optional columns depend on the
operation:

    add-ligand / replace-ligand:
        ligand_tag (required)
        smiles                (one of these two)
        ccd                   (comma-separated CCD codes for multi-component)
        target_id             (replace-ligand only; required)
        copies                (default 1, add-ligand only)
        chain_id              (optional, add-ligand only)

    add-nucleic / replace-nucleic:
        ligand_tag (required)
        type      (required, "rna" or "dna")
        sequence  (required)
        target_id (replace-nucleic only; required)
        copies    (default 1, add-nucleic only)
        chain_id  (optional, add-nucleic only)

    add-ion / replace-ion:
        ligand_tag (required)
        ccd       (required, single ion code, e.g. MG, ZN)
        target_id (replace-ion only; required)
        copies    (default 1, add-ion only)
        chain_id  (optional, add-ion only)

set-seeds does not support Fan-out mode.

"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import random
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENTITY_KEYS: Tuple[str, ...] = ("protein", "rna", "dna", "ligand")
RNA_ALPHABET: frozenset = frozenset("ACGU")
DNA_ALPHABET: frozenset = frozenset("ACGT")
SEED_RANDOM_UPPER: int = 2_147_483_647
DEFAULT_WORKERS: int = min(8, os.cpu_count() or 1)
SUPPORTED_DIALECT: str = "alphafold3"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> Dict[str, Any]:
    """Load and minimally validate an AF3 alphafold3-dialect JSON file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Top-level JSON must be a dict (alphafold3 dialect). "
            "Lists are alphafoldserver dialect and are not supported."
        )
    for required in ("name", "sequences", "modelSeeds"):
        if required not in data:
            raise ValueError(f"Missing required top-level field: '{required}'")
    if not isinstance(data["sequences"], list):
        raise ValueError("'sequences' must be a list.")
    if not isinstance(data["modelSeeds"], list) or not data["modelSeeds"]:
        raise ValueError("'modelSeeds' must be a non-empty list of integers.")
    # AF3 alphafold3-dialect files declare their dialect explicitly.  Reject
    # other dialects up front so users get a clean error rather than a
    # confusing downstream failure.
    dialect = data.get("dialect")
    if dialect is not None and dialect != SUPPORTED_DIALECT:
        raise ValueError(
            f"Unsupported dialect '{dialect}' in {path}: this tool only "
            f"handles the '{SUPPORTED_DIALECT}' dialect."
        )
    return data


def save_json(data: Dict[str, Any], path: str) -> None:
    """Pretty-print the JSON to path atomically.

    Writes to a temporary file in the same directory and then renames it
    over the destination via os.replace (atomic on POSIX and Windows).
    This guarantees that --in-place operations never corrupt the
    original file on validation errors, disk-full conditions, or process
    interruption.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=parent or ".",
        prefix=".af3_int_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the partial temp file on any error.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def entity_type(entry: Dict[str, Any]) -> str:
    keys = [k for k in entry.keys() if k in ENTITY_KEYS]
    if len(keys) != 1:
        raise ValueError(
            f"Each sequences element must contain exactly one of "
            f"{ENTITY_KEYS}; got {list(entry.keys())}"
        )
    return keys[0]


def get_chain_ids(entry: Dict[str, Any]) -> List[str]:
    body = entry[entity_type(entry)]
    raw = body.get("id")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw]
    raise ValueError(f"Unexpected 'id' value (must be str or list): {raw!r}")


def all_chain_ids(sequences: Sequence[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for entry in sequences:
        out.extend(get_chain_ids(entry))
    return out


def find_entity_by_chain_id(
    sequences: Sequence[Dict[str, Any]], target_id: str
) -> Tuple[int, str, Dict[str, Any]]:
    for idx, entry in enumerate(sequences):
        if target_id in get_chain_ids(entry):
            return idx, entity_type(entry), entry
    raise ValueError(
        f"No entity with chain ID '{target_id}' found in sequences."
    )


def _all_chain_ids_of_length(n: int) -> Iterable[str]:
    for tup in itertools.product("ABCDEFGHIJKLMNOPQRSTUVWXYZ", repeat=n):
        yield "".join(tup)


def next_chain_id(used: Iterable[str]) -> str:
    used_set = set(used)
    n = 1
    while True:
        for cid in _all_chain_ids_of_length(n):
            if cid not in used_set:
                return cid
        n += 1


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

def split_name(name: str) -> Tuple[str, str]:
    """Split name into (protein-ID prefix, ligand-tag suffix)."""
    if "_" in name:
        head, tail = name.split("_", 1)
        return head, tail
    return name, ""


def compose_name(protein_prefix: str, ligand_tag: str) -> str:
    if not ligand_tag:
        return protein_prefix
    return f"{protein_prefix}_{ligand_tag}"


# ---------------------------------------------------------------------------
# Protein-integrity guards
# ---------------------------------------------------------------------------

def extract_protein_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        copy.deepcopy(e["protein"])
        for e in data["sequences"]
        if "protein" in e
    ]


def assert_proteins_unchanged(
    before: List[Dict[str, Any]], after: List[Dict[str, Any]]
) -> None:
    # Check the simpler invariant first so the error message is more
    # informative on failure (the equality check would otherwise mask
    # length changes behind a generic "differ" error).
    if len(before) != len(after):
        raise RuntimeError(
            f"Number of protein entries changed: "
            f"{len(before)} -> {len(after)}"
        )
    if before != after:
        raise RuntimeError(
            "Protein entries differ between input and output. This script "
            "guarantees that no operation modifies any protein chain."
        )


def assert_protein_prefix_preserved(
    original_name: str, new_name: str
) -> None:
    orig_prefix, _ = split_name(original_name)
    new_prefix, _ = split_name(new_name)
    if orig_prefix != new_prefix:
        raise RuntimeError(
            f"Protein-ID prefix in 'name' changed: "
            f"'{orig_prefix}' -> '{new_prefix}'."
        )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _ligand_id_field(ids: List[str]) -> Union[str, List[str]]:
    return ids[0] if len(ids) == 1 else list(ids)


def build_ligand_entry(
    chain_ids: List[str],
    smiles: Optional[str],
    ccd_codes: Optional[List[str]],
) -> Dict[str, Any]:
    if (smiles is None) == (ccd_codes is None):
        raise ValueError(
            "Exactly one of SMILES or CCD codes must be provided for ligands."
        )
    body: Dict[str, Any] = {"id": _ligand_id_field(chain_ids)}
    if smiles is not None:
        body["smiles"] = smiles
    else:
        body["ccdCodes"] = list(ccd_codes)  # type: ignore[arg-type]
    return {"ligand": body}


def build_nucleic_entry(
    kind: str, chain_ids: List[str], sequence: str
) -> Dict[str, Any]:
    kind = kind.lower()
    if kind not in ("rna", "dna"):
        raise ValueError(f"Nucleic kind must be 'rna' or 'dna', got {kind!r}")
    seq = sequence.strip().upper().replace(" ", "")
    if not seq:
        raise ValueError(f"{kind.upper()} sequence is empty.")
    alphabet = RNA_ALPHABET if kind == "rna" else DNA_ALPHABET
    bad = sorted(set(seq) - alphabet)
    if bad:
        raise ValueError(
            f"{kind.upper()} sequence contains invalid bases: {bad}. "
            f"Allowed alphabet: {''.join(sorted(alphabet))}"
        )
    body: Dict[str, Any] = {
        "id": _ligand_id_field(chain_ids),
        "sequence": seq,
    }
    return {kind: body}


def allocate_chain_ids(
    used: Iterable[str], copies: int, requested: Optional[str]
) -> List[str]:
    if copies < 1:
        raise ValueError("copies must be >= 1")
    used_set = set(used)
    if requested is not None:
        if copies != 1:
            raise ValueError(
                "chain_id can only be combined with copies=1; "
                "for multiple copies the script auto-allocates chain IDs."
            )
        if requested in used_set:
            raise ValueError(
                f"Chain ID '{requested}' is already used in the JSON."
            )
        return [requested]
    out: List[str] = []
    for _ in range(copies):
        cid = next_chain_id(used_set)
        out.append(cid)
        used_set.add(cid)
    return out


# ---------------------------------------------------------------------------
# Operations (pure transforms)
# ---------------------------------------------------------------------------

def op_set_seeds(
    data: Dict[str, Any],
    explicit_seeds: Optional[List[int]],
    num_seeds: Optional[int],
    seed_base: Optional[int],
    rng_seed: Optional[int],
) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    if explicit_seeds:
        seeds = list(explicit_seeds)
    elif num_seeds is not None:
        if num_seeds < 1:
            raise ValueError("num_seeds must be >= 1")
        if seed_base is not None:
            seeds = [seed_base + i for i in range(num_seeds)]
        else:
            rng = random.Random(rng_seed)
            seeds = rng.sample(range(1, SEED_RANDOM_UPPER), num_seeds)
    else:
        raise ValueError("Provide either explicit seeds or num_seeds.")
    if any(not isinstance(s, int) for s in seeds):
        raise ValueError("All seeds must be integers.")
    out["modelSeeds"] = seeds
    return out


def op_add_ligand(
    data: Dict[str, Any],
    smiles: Optional[str],
    ccd_codes: Optional[List[str]],
    copies: int,
    chain_id: Optional[str],
    new_ligand_tag: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    used = all_chain_ids(out["sequences"])
    ids = allocate_chain_ids(used, copies, chain_id)
    entry = build_ligand_entry(ids, smiles, ccd_codes)
    out["sequences"].append(entry)
    out["name"] = compose_name(split_name(out["name"])[0], new_ligand_tag)
    return out


def op_replace_ligand(
    data: Dict[str, Any],
    target_id: str,
    smiles: Optional[str],
    ccd_codes: Optional[List[str]],
    new_ligand_tag: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    idx, etype, entry = find_entity_by_chain_id(out["sequences"], target_id)
    if etype == "protein":
        raise ValueError(
            f"Chain '{target_id}' is a protein chain. "
            "Protein chains cannot be replaced by this tool."
        )
    if etype != "ligand":
        raise ValueError(
            f"Chain '{target_id}' is a {etype} entity, not a ligand. "
            f"Use the appropriate replace-* subcommand instead."
        )
    preserved_ids = get_chain_ids(entry)
    new_entry = build_ligand_entry(preserved_ids, smiles, ccd_codes)
    out["sequences"][idx] = new_entry
    out["name"] = compose_name(split_name(out["name"])[0], new_ligand_tag)
    return out


def op_add_nucleic(
    data: Dict[str, Any],
    kind: str,
    sequence: str,
    copies: int,
    chain_id: Optional[str],
    new_ligand_tag: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    used = all_chain_ids(out["sequences"])
    ids = allocate_chain_ids(used, copies, chain_id)
    entry = build_nucleic_entry(kind, ids, sequence)
    out["sequences"].append(entry)
    out["name"] = compose_name(split_name(out["name"])[0], new_ligand_tag)
    return out


def op_replace_nucleic(
    data: Dict[str, Any],
    target_id: str,
    kind: str,
    sequence: str,
    new_ligand_tag: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(data)
    idx, etype, entry = find_entity_by_chain_id(out["sequences"], target_id)
    if etype == "protein":
        raise ValueError(
            f"Chain '{target_id}' is a protein chain. "
            "Protein chains cannot be replaced by this tool."
        )
    if etype not in ("rna", "dna"):
        raise ValueError(
            f"Chain '{target_id}' is a {etype} entity, not RNA/DNA. "
            f"Use the appropriate replace-* subcommand instead."
        )
    preserved_ids = get_chain_ids(entry)
    new_entry = build_nucleic_entry(kind, preserved_ids, sequence)
    out["sequences"][idx] = new_entry
    out["name"] = compose_name(split_name(out["name"])[0], new_ligand_tag)
    return out


# ---------------------------------------------------------------------------
# Task dispatch
# ---------------------------------------------------------------------------

def apply_operation(data: Dict[str, Any], op: str,
                    params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one operation to a parsed JSON dict and return the new dict."""
    if op == "set-seeds":
        return op_set_seeds(
            data,
            explicit_seeds=params.get("seeds"),
            num_seeds=params.get("num_seeds"),
            seed_base=params.get("seed_base"),
            rng_seed=params.get("rng_seed"),
        )
    if op == "add-ligand":
        return op_add_ligand(
            data,
            smiles=params.get("smiles"),
            ccd_codes=params.get("ccd"),
            copies=params.get("copies", 1),
            chain_id=params.get("chain_id"),
            new_ligand_tag=params["ligand_tag"],
        )
    if op == "replace-ligand":
        return op_replace_ligand(
            data,
            target_id=params["target_id"],
            smiles=params.get("smiles"),
            ccd_codes=params.get("ccd"),
            new_ligand_tag=params["ligand_tag"],
        )
    if op == "add-nucleic":
        return op_add_nucleic(
            data,
            kind=params["type"],
            sequence=params["sequence"],
            copies=params.get("copies", 1),
            chain_id=params.get("chain_id"),
            new_ligand_tag=params["ligand_tag"],
        )
    if op == "replace-nucleic":
        return op_replace_nucleic(
            data,
            target_id=params["target_id"],
            kind=params["type"],
            sequence=params["sequence"],
            new_ligand_tag=params["ligand_tag"],
        )
    if op == "add-ion":
        return op_add_ligand(
            data,
            smiles=None,
            ccd_codes=([params["ccd"]] if isinstance(params["ccd"], str)
                       else list(params["ccd"])),
            copies=params.get("copies", 1),
            chain_id=params.get("chain_id"),
            new_ligand_tag=params["ligand_tag"],
        )
    if op == "replace-ion":
        return op_replace_ligand(
            data,
            target_id=params["target_id"],
            smiles=None,
            ccd_codes=([params["ccd"]] if isinstance(params["ccd"], str)
                       else list(params["ccd"])),
            new_ligand_tag=params["ligand_tag"],
        )
    raise ValueError(f"Unknown operation: {op}")


# ---------------------------------------------------------------------------
# Single-task runner (used by both serial and parallel paths)
# ---------------------------------------------------------------------------

def _new_result(task: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a fresh per-task result dict with all keys set."""
    return {
        "input_path": task.get("input_path", "?"),
        "output_path": task.get("output_path", "?"),
        "operation": task.get("operation", "?"),
        "success": False,
        "error": None,
        "name_before": None,
        "name_after": None,
    }


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one task (load -> transform -> validate -> save) and report.

    The argument is a plain dict so it pickles cleanly for multiprocessing.
    Returns a dict describing success or failure; never raises.
    """
    result = _new_result(task)
    try:
        data_in = load_json(task["input_path"])
        proteins_before = extract_protein_entries(data_in)
        data_out = apply_operation(data_in, task["operation"], task["params"])

        if task["operation"] == "set-seeds":
            if data_out["name"] != data_in["name"]:
                raise RuntimeError("set-seeds must not change 'name'")
        proteins_after = extract_protein_entries(data_out)
        assert_proteins_unchanged(proteins_before, proteins_after)
        assert_protein_prefix_preserved(data_in["name"], data_out["name"])
        cids = all_chain_ids(data_out["sequences"])
        if len(cids) != len(set(cids)):
            raise RuntimeError(f"Duplicate chain IDs in output: {cids}")

        save_json(data_out, task["output_path"])

        result["success"] = True
        result["name_before"] = data_in["name"]
        result["name_after"] = data_out["name"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# Output path planning
# ---------------------------------------------------------------------------

def _plan_output_path(output_dir: str, name_after: str) -> str:
    safe = name_after.strip()
    if not safe:
        raise ValueError("Resulting JSON 'name' field is empty.")
    return os.path.join(output_dir, f"{safe}.json")


def _peek_output_name(input_path: str, op: str,
                      params: Dict[str, Any]) -> str:
    """Run the transform in-memory just to discover the output name."""
    data = load_json(input_path)
    out = apply_operation(data, op, params)
    return out["name"]


# ---------------------------------------------------------------------------
# Task builders for the three I/O modes
# ---------------------------------------------------------------------------

def _params_from_args_single(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a per-task parameter dict from the CLI namespace."""
    op = args.op
    p: Dict[str, Any] = {}
    if op == "set-seeds":
        p["seeds"] = args.seeds
        p["num_seeds"] = args.num_seeds
        p["seed_base"] = args.seed_base
        p["rng_seed"] = args.rng_seed
    elif op == "add-ligand":
        p["smiles"] = args.smiles
        p["ccd"] = args.ccd
        p["copies"] = args.copies
        p["chain_id"] = args.chain_id
        p["ligand_tag"] = args.ligand_tag
    elif op == "replace-ligand":
        p["smiles"] = args.smiles
        p["ccd"] = args.ccd
        p["target_id"] = args.target_id
        p["ligand_tag"] = args.ligand_tag
    elif op == "add-nucleic":
        p["type"] = args.type
        p["sequence"] = args.sequence
        p["copies"] = args.copies
        p["chain_id"] = args.chain_id
        p["ligand_tag"] = args.ligand_tag
    elif op == "replace-nucleic":
        p["target_id"] = args.target_id
        p["type"] = args.type
        p["sequence"] = args.sequence
        p["ligand_tag"] = args.ligand_tag
    elif op == "add-ion":
        p["ccd"] = args.ccd
        p["copies"] = args.copies
        p["chain_id"] = args.chain_id
        p["ligand_tag"] = args.ligand_tag
    elif op == "replace-ion":
        p["ccd"] = args.ccd
        p["target_id"] = args.target_id
        p["ligand_tag"] = args.ligand_tag
    else:
        raise ValueError(f"Unknown operation: {op}")
    return p


def _parse_csv_row(op: str, row: Dict[str, str]) -> Dict[str, Any]:
    def _maybe(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s = s.strip()
        return s if s else None

    def _int_or(value: Optional[str], default: int) -> int:
        v = _maybe(value)
        if v is None:
            return default
        try:
            return int(v)
        except ValueError as exc:
            raise ValueError(
                f"CSV column expects an integer, got {v!r}"
            ) from exc

    def _copies_or(value: Optional[str]) -> int:
        n = _int_or(value, 1)
        if n < 1:
            raise ValueError(
                f"CSV 'copies' column must be a positive integer, got {n}"
            )
        return n

    p: Dict[str, Any] = {}
    if op in ("add-ligand", "replace-ligand"):
        p["ligand_tag"] = _maybe(row.get("ligand_tag"))
        smiles = _maybe(row.get("smiles"))
        ccd = _maybe(row.get("ccd"))
        if smiles is not None and ccd is not None:
            raise ValueError(
                "CSV row has both 'smiles' and 'ccd' filled; provide only one."
            )
        if smiles is None and ccd is None:
            raise ValueError("CSV row has neither 'smiles' nor 'ccd' filled.")
        p["smiles"] = smiles
        p["ccd"] = [c.strip() for c in ccd.split(",")] if ccd else None
        if op == "replace-ligand":
            p["target_id"] = _maybe(row.get("target_id"))
            if p["target_id"] is None:
                raise ValueError("replace-ligand requires 'target_id' column.")
        else:
            p["copies"] = _copies_or(row.get("copies"))
            p["chain_id"] = _maybe(row.get("chain_id"))
    elif op in ("add-nucleic", "replace-nucleic"):
        p["ligand_tag"] = _maybe(row.get("ligand_tag"))
        p["type"] = _maybe(row.get("type"))
        p["sequence"] = _maybe(row.get("sequence"))
        if p["type"] is None or p["sequence"] is None:
            raise ValueError(f"{op} requires 'type' and 'sequence' columns.")
        if op == "replace-nucleic":
            p["target_id"] = _maybe(row.get("target_id"))
            if p["target_id"] is None:
                raise ValueError("replace-nucleic requires 'target_id' column.")
        else:
            p["copies"] = _copies_or(row.get("copies"))
            p["chain_id"] = _maybe(row.get("chain_id"))
    elif op in ("add-ion", "replace-ion"):
        p["ligand_tag"] = _maybe(row.get("ligand_tag"))
        p["ccd"] = _maybe(row.get("ccd"))
        if p["ccd"] is None:
            raise ValueError(f"{op} requires 'ccd' column.")
        if op == "replace-ion":
            p["target_id"] = _maybe(row.get("target_id"))
            if p["target_id"] is None:
                raise ValueError("replace-ion requires 'target_id' column.")
        else:
            p["copies"] = _copies_or(row.get("copies"))
            p["chain_id"] = _maybe(row.get("chain_id"))
    else:
        raise ValueError(f"Operation '{op}' does not support fan-out CSV mode.")

    if not p.get("ligand_tag"):
        raise ValueError("CSV row missing required 'ligand_tag' value.")
    return p


def _read_csv(csv_path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        return [dict(r) for r in reader]


def _list_input_jsons(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    paths = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".json") and os.path.isfile(os.path.join(input_dir, f))
    )
    if not paths:
        raise ValueError(f"No *.json files found in: {input_dir}")
    return paths


def build_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Materialise the full task list from CLI arguments."""
    op: str = args.op

    has_input_file = bool(args.input)
    has_input_dir = bool(args.input_dir)
    has_output_file = bool(args.output)
    has_output_dir = bool(args.output_dir)
    has_csv = bool(getattr(args, "from_csv", None))
    in_place = bool(getattr(args, "in_place", False))

    if has_input_file == has_input_dir:
        raise ValueError(
            "Specify exactly one of --input/-i (single file) or --input-dir."
        )
    if has_csv and op == "set-seeds":
        raise ValueError("set-seeds does not support fan-out CSV mode.")
    if has_csv and has_input_dir:
        raise ValueError(
            "Fan-out CSV mode (--from-csv) requires a single -i FILE input, "
            "not --input-dir."
        )
    if in_place:
        if has_csv:
            raise ValueError(
                "--in-place is incompatible with --from-csv "
                "(fan-out produces multiple outputs per input)."
            )
        if has_output_file or has_output_dir:
            raise ValueError(
                "--in-place cannot be combined with -o/--output or "
                "--output-dir; the input file(s) are modified in place."
            )
    else:
        if has_csv and not has_output_dir:
            raise ValueError("Fan-out CSV mode requires --output-dir.")
        if has_input_dir and not has_output_dir:
            raise ValueError(
                "--input-dir requires --output-dir (or use --in-place to "
                "modify files in place)."
            )
        if (not has_csv) and (not has_input_dir):
            if not has_output_file:
                raise ValueError(
                    "Single-file mode requires -o/--output (or use "
                    "--in-place to modify the input file in place)."
                )
            if has_output_dir:
                raise ValueError(
                    "Single-file mode uses -o/--output, not --output-dir."
                )

    tasks: List[Dict[str, Any]] = []

    # ---- In-place: single file -----------------------------------------
    if in_place and has_input_file:
        params = _params_from_args_single(args)
        in_path = os.path.abspath(args.input)
        tasks.append({
            "input_path": in_path,
            "output_path": in_path,  # overwrite source atomically
            "operation": op,
            "params": params,
        })
        return tasks

    # ---- In-place: bulk directory --------------------------------------
    if in_place and has_input_dir:
        params = _params_from_args_single(args)
        for in_path in _list_input_jsons(args.input_dir):
            tasks.append({
                "input_path": in_path,
                "output_path": in_path,  # overwrite source atomically
                "operation": op,
                "params": params,
            })
        # No collision check needed: directory listing is unique by
        # construction and every output_path equals its own input_path.
        return tasks

    # ---- Single mode ----------------------------------------------------
    if has_input_file and has_output_file and not has_csv:
        params = _params_from_args_single(args)
        tasks.append({
            "input_path": os.path.abspath(args.input),
            "output_path": os.path.abspath(args.output),
            "operation": op,
            "params": params,
        })
        return tasks

    # ---- Bulk (directory) mode ------------------------------------------
    if has_input_dir and not has_csv:
        params = _params_from_args_single(args)
        out_dir = os.path.abspath(args.output_dir)
        for in_path in _list_input_jsons(args.input_dir):
            try:
                if op == "set-seeds":
                    # set-seeds is name-stable; just keep input filename.
                    out_path = os.path.join(out_dir, os.path.basename(in_path))
                else:
                    name_after = _peek_output_name(in_path, op, params)
                    out_path = _plan_output_path(out_dir, name_after)
            except Exception as exc:
                tasks.append({
                    "input_path": in_path,
                    "output_path": os.path.join(out_dir, "__failed__.json"),
                    "operation": op,
                    "params": params,
                    "_planning_error": f"{type(exc).__name__}: {exc}",
                })
                continue
            tasks.append({
                "input_path": in_path,
                "output_path": out_path,
                "operation": op,
                "params": params,
            })
        _check_output_collisions(tasks)
        return tasks

    # ---- Fan-out CSV mode ------------------------------------------------
    if has_input_file and has_csv and has_output_dir:
        rows = _read_csv(args.from_csv)
        if not rows:
            raise ValueError("CSV is empty.")
        in_path = os.path.abspath(args.input)
        out_dir = os.path.abspath(args.output_dir)
        for i, row in enumerate(rows, start=1):
            try:
                params = _parse_csv_row(op, row)
                name_after = _peek_output_name(in_path, op, params)
                out_path = _plan_output_path(out_dir, name_after)
            except Exception as exc:
                tasks.append({
                    "input_path": in_path,
                    "output_path": os.path.join(
                        out_dir, f"__row{i}_failed__.json"),
                    "operation": op,
                    "params": {},
                    "_planning_error":
                        f"row {i}: {type(exc).__name__}: {exc}",
                })
                continue
            tasks.append({
                "input_path": in_path,
                "output_path": out_path,
                "operation": op,
                "params": params,
            })
        _check_output_collisions(tasks)
        return tasks

    raise ValueError("Unrecognised combination of input/output arguments.")


def _check_output_collisions(tasks: List[Dict[str, Any]]) -> None:
    seen: Dict[str, int] = {}
    for t in tasks:
        if "_planning_error" in t:
            continue
        seen[t["output_path"]] = seen.get(t["output_path"], 0) + 1
    dups = sorted(p for p, n in seen.items() if n > 1)
    if dups:
        raise ValueError(
            "Output filename collision detected (different tasks would "
            f"overwrite the same file). Examples: {dups[:5]}. "
            "Make ligand_tag values unique in your CSV / inputs."
        )


# ---------------------------------------------------------------------------
# Execution: serial or parallel
# ---------------------------------------------------------------------------

def _run_planning_failed(task: Dict[str, Any]) -> Dict[str, Any]:
    result = _new_result(task)
    result["error"] = task["_planning_error"]
    return result


def execute_tasks(tasks: List[Dict[str, Any]],
                  workers: int,
                  quiet: bool) -> List[Dict[str, Any]]:
    """Run all tasks (parallel if workers > 1 and len(tasks) > 1)."""
    results: List[Dict[str, Any]] = []
    if not tasks:
        return results

    runnable: List[Dict[str, Any]] = []
    for t in tasks:
        if "_planning_error" in t:
            results.append(_run_planning_failed(t))
        else:
            runnable.append(t)

    if not runnable:
        return results

    use_pool = workers > 1 and len(runnable) > 1

    if not use_pool:
        for i, task in enumerate(runnable, start=1):
            results.append(run_task(task))
            if not quiet:
                _progress(i, len(runnable))
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, t) for t in runnable]
        done = 0
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:  # pragma: no cover - safety net
                crashed = _new_result({})
                crashed["error"] = (
                    f"Worker crashed: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                results.append(crashed)
            done += 1
            if not quiet:
                _progress(done, len(runnable))

    return results


def _progress(done: int, total: int) -> None:
    pct = 100.0 * done / total if total else 100.0
    end = "\n" if done == total else "\r"
    sys.stderr.write(f"  [progress] {done}/{total} ({pct:5.1f}%){end}")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_io_args(p: argparse.ArgumentParser, allow_csv: bool) -> None:
    p.add_argument("-i", "--input", default=None,
                   help="Input JSON file (single-file or fan-out mode).")
    p.add_argument("--input-dir", default=None,
                   help="Input directory containing many *.json files (bulk).")
    p.add_argument("-o", "--output", default=None,
                   help="Output JSON file (single-file mode only).")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (bulk or fan-out mode).")
    p.add_argument(
        "--in-place", action="store_true",
        help=("Modify input files in place rather than writing copies to a "
              "separate output directory (useful when JSONs are large, "
              "e.g. contain MSAs). Compatible with -i FILE and --input-dir; "
              "NOT compatible with --from-csv or with -o/--output-dir. "
              "The on-disk filename is preserved; the internal 'name' field "
              "is updated normally per the operation. Atomic write "
              "(temp-file + rename) ensures the original is never corrupted "
              "on failure."),
    )
    if allow_csv:
        p.add_argument("--from-csv", default=None,
                       help=("CSV manifest for fan-out mode: one input JSON "
                             "is expanded into one output per row."))
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help=(f"Parallel worker processes (default: "
                         f"{DEFAULT_WORKERS}). Ignored when only 1 task."))
    p.add_argument("--quiet", action="store_true",
                   help="Suppress progress and summary messages on stderr.")


def _add_tag_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ligand-tag", default=None,
        help=("New ligand-tag suffix for the 'name' field. The protein-ID "
              "prefix is preserved automatically. Required in single-file "
              "and bulk modes; ignored in fan-out mode (each CSV row "
              "supplies its own ligand_tag)."),
    )


def build_parser() -> argparse.ArgumentParser:
    desc = (
        "AF3 JSON Integrator: programmatic, protein-preserving editor for "
        "AlphaFold 3 input JSON files, with single-file, bulk-directory, "
        "and CSV-fan-out batch modes."
    )
    parser = argparse.ArgumentParser(
        prog="af3_json_integrator",
        description=desc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="op", required=True, metavar="OPERATION")

    # ---- set-seeds -------------------------------------------------------
    p = sub.add_parser(
        "set-seeds",
        help="Modify only modelSeeds (everything else, including 'name', "
             "stays exactly as in the input).",
    )
    _add_io_args(p, allow_csv=False)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--seeds", type=int, nargs="+",
                     help="Explicit list of integer seeds, e.g. --seeds 1 2 3.")
    grp.add_argument("--num-seeds", type=int,
                     help="Number of seeds to generate.")
    p.add_argument("--seed-base", type=int, default=None,
                   help="With --num-seeds, generate consecutive seeds "
                        "starting at this value.")
    p.add_argument("--rng-seed", type=int, default=None,
                   help="With --num-seeds (no --seed-base), use this as the "
                        "RNG seed for reproducible random sampling.")

    # ---- add-ligand ------------------------------------------------------
    p = sub.add_parser("add-ligand",
                       help="Add a small-molecule ligand (SMILES or CCD).")
    _add_io_args(p, allow_csv=True)
    p.add_argument("--smiles", type=str, default=None, help="SMILES string.")
    p.add_argument("--ccd", type=str, nargs="+", default=None,
                   help="One or more CCD codes (multi-component ligand).")
    p.add_argument("--copies", type=int, default=1,
                   help="Number of identical copies (default 1).")
    p.add_argument("--chain-id", type=str, default=None,
                   help="Force a specific chain ID (only with --copies 1).")
    _add_tag_arg(p)

    # ---- replace-ligand --------------------------------------------------
    p = sub.add_parser(
        "replace-ligand",
        help="Replace an existing small-molecule ligand by chain ID.")
    _add_io_args(p, allow_csv=True)
    p.add_argument("--target-id", type=str, default=None,
                   help="Chain ID of the ligand to replace.")
    p.add_argument("--smiles", type=str, default=None, help="New SMILES.")
    p.add_argument("--ccd", type=str, nargs="+", default=None,
                   help="New CCD code(s).")
    _add_tag_arg(p)

    # ---- add-nucleic -----------------------------------------------------
    p = sub.add_parser("add-nucleic",
                       help="Add a new RNA or DNA chain.")
    _add_io_args(p, allow_csv=True)
    p.add_argument("--type", choices=("rna", "dna"), default=None,
                   help="Nucleic acid type.")
    p.add_argument("--sequence", type=str, default=None,
                   help="Nucleotide sequence (1-letter codes).")
    p.add_argument("--copies", type=int, default=1,
                   help="Number of identical copies (default 1).")
    p.add_argument("--chain-id", type=str, default=None,
                   help="Force a specific chain ID (only with --copies 1).")
    _add_tag_arg(p)

    # ---- replace-nucleic -------------------------------------------------
    p = sub.add_parser("replace-nucleic",
                       help="Replace an existing RNA/DNA chain by chain ID.")
    _add_io_args(p, allow_csv=True)
    p.add_argument("--target-id", type=str, default=None,
                   help="Chain ID of the RNA/DNA chain to replace.")
    p.add_argument("--type", choices=("rna", "dna"), default=None,
                   help="New nucleic-acid type (rna or dna).")
    p.add_argument("--sequence", type=str, default=None,
                   help="New nucleotide sequence.")
    _add_tag_arg(p)

    # ---- add-ion ---------------------------------------------------------
    p = sub.add_parser("add-ion",
                       help="Add a metal ion (CCD code only, e.g. MG, ZN).")
    _add_io_args(p, allow_csv=True)
    p.add_argument("--ccd", type=str, default=None,
                   help="Single ion CCD code.")
    p.add_argument("--copies", type=int, default=1,
                   help="Number of identical ions (default 1).")
    p.add_argument("--chain-id", type=str, default=None,
                   help="Force a specific chain ID (only with --copies 1).")
    _add_tag_arg(p)

    # ---- replace-ion -----------------------------------------------------
    p = sub.add_parser("replace-ion",
                       help="Replace an existing metal ion by chain ID.")
    _add_io_args(p, allow_csv=True)
    p.add_argument("--target-id", type=str, default=None,
                   help="Chain ID of the ion to replace.")
    p.add_argument("--ccd", type=str, default=None,
                   help="Replacement ion CCD code.")
    _add_tag_arg(p)

    return parser


# ---------------------------------------------------------------------------
# CLI validation that depends on the chosen mode
# ---------------------------------------------------------------------------

def _validate_required_args(args: argparse.Namespace) -> None:
    op = args.op
    csv_mode = bool(getattr(args, "from_csv", None))

    if csv_mode:
        # In fan-out mode the per-task params come from the CSV, so any
        # operation-specific CLI flag is rejected to avoid silent overrides.
        # copies is handled separately because its argparse default of 1
        # is indistinguishable from "not set" via getattr alone.
        forbidden_map = {
            "add-ligand": ("smiles", "ccd", "ligand_tag", "chain_id"),
            "replace-ligand": ("smiles", "ccd", "target_id", "ligand_tag"),
            "add-nucleic": ("type", "sequence", "ligand_tag", "chain_id"),
            "replace-nucleic": ("target_id", "type", "sequence", "ligand_tag"),
            "add-ion": ("ccd", "ligand_tag", "chain_id"),
            "replace-ion": ("target_id", "ccd", "ligand_tag"),
        }
        for k in forbidden_map.get(op, ()):
            if getattr(args, k, None) is not None:
                raise ValueError(
                    f"--{k.replace('_','-')} cannot be combined with "
                    "--from-csv (fan-out mode reads per-task values from CSV)."
                )
        # copies default is 1; only complain if the user explicitly
        # supplied a different value.
        if op in ("add-ligand", "add-nucleic", "add-ion"):
            if getattr(args, "copies", 1) != 1:
                raise ValueError(
                    "--copies cannot be combined with --from-csv "
                    "(supply 'copies' column in the CSV instead)."
                )
        return

    needed = {
        "set-seeds": (),  # already enforced by argparse mutually-exclusive
        "add-ligand": ("ligand_tag",),
        "replace-ligand": ("target_id", "ligand_tag"),
        "add-nucleic": ("type", "sequence", "ligand_tag"),
        "replace-nucleic": ("target_id", "type", "sequence", "ligand_tag"),
        "add-ion": ("ccd", "ligand_tag"),
        "replace-ion": ("target_id", "ccd", "ligand_tag"),
    }[op]
    missing = [k for k in needed if getattr(args, k, None) in (None,)]
    if missing:
        raise ValueError(
            f"Operation '{op}' requires: "
            + ", ".join(f"--{k.replace('_','-')}" for k in missing)
        )

    if op in ("add-ligand", "replace-ligand"):
        if (args.smiles is None) == (args.ccd is None):
            raise ValueError(
                f"{op} requires exactly one of --smiles or --ccd."
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(results: List[Dict[str, Any]], quiet: bool) -> None:
    if quiet:
        return
    n = len(results)
    ok = sum(1 for r in results if r["success"])
    failed = n - ok
    sys.stderr.write("\n--- AF3 JSON Integrator: summary ---\n")
    sys.stderr.write(f"Total tasks : {n}\n")
    sys.stderr.write(f"Succeeded   : {ok}\n")
    sys.stderr.write(f"Failed      : {failed}\n")
    if failed:
        sys.stderr.write("First failures:\n")
        shown = 0
        for r in results:
            if r["success"]:
                continue
            sys.stderr.write(
                f"  - [{r['operation']}] {r['input_path']}\n"
                f"      -> {r['output_path']}\n"
                f"      ERROR: {r['error']}\n"
            )
            shown += 1
            if shown >= 10:
                sys.stderr.write(f"  ... and {failed - shown} more failures\n")
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_required_args(args)
        tasks = build_tasks(args)
    except Exception as exc:
        sys.stderr.write(f"[af3_json_integrator] ERROR: {exc}\n")
        return 1

    if not args.quiet:
        sys.stderr.write(
            f"[af3_json_integrator] Operation = {args.op}; "
            f"#tasks = {len(tasks)}; workers = {args.workers}\n"
        )
    results = execute_tasks(tasks, args.workers, args.quiet)
    _print_summary(results, args.quiet)

    failed = sum(1 for r in results if not r["success"])
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
