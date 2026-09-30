#!/usr/bin/env python3
"""Print the dataset table needed for the manuscript from a curated split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _column(row: dict[str, str], candidates: tuple[str, ...]) -> str:
    for name in candidates:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def _split_summary(path: Path) -> dict[str, Any]:
    drugs: set[str] = set()
    targets: set[str] = set()
    pairs = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pairs += 1
            drug = _column(row, ("canonical_smiles", "SMILES"))
            target = _column(row, ("protein_identity_key", "target", "protein_id"))
            if drug:
                drugs.add(drug)
            if target:
                targets.add(target)
    return {"pairs": pairs, "drugs": drugs, "targets": targets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--original-drugs", type=int, default=0)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs", {})
    summaries: dict[str, dict[str, Any]] = {}
    all_drugs: set[str] = set()
    all_targets: set[str] = set()
    for split in ("train", "val", "test"):
        entry = outputs.get(split, {})
        path = _resolve(manifest_path, str(entry.get("path", "")))
        summaries[split] = _split_summary(path)
        all_drugs.update(summaries[split]["drugs"])
        all_targets.update(summaries[split]["targets"])
    result = {
        "dataset": str(manifest.get("configuration", {}).get("dataset", data_dir.name)),
        "retained_drugs": len(all_drugs),
        "retained_targets": len(all_targets),
        "train_pairs": summaries["train"]["pairs"],
        "validation_pairs": summaries["val"]["pairs"],
        "test_pairs": summaries["test"]["pairs"],
        "original_drugs": int(args.original_drugs) or None,
        "retention_percent": (
            100.0 * len(all_drugs) / int(args.original_drugs)
            if int(args.original_drugs) > 0
            else None
        ),
    }
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    retention = "NR" if result["retention_percent"] is None else f"{result['retention_percent']:.2f}"
    original = "NR" if result["original_drugs"] is None else str(result["original_drugs"])
    print("| Dataset | Original drugs | Retained drugs | Retention (%) | Targets | Train pairs | Validation pairs | Test pairs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    print(
        "| {dataset} | {original} | {retained_drugs} | {retention} | {retained_targets} | "
        "{train_pairs} | {validation_pairs} | {test_pairs} |".format(
            **result, original=original, retention=retention
        )
    )
    print("NR = not recorded in the supplied manifest; provide --original-drugs to compute retention.")


if __name__ == "__main__":
    main()
