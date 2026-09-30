#!/usr/bin/env python3
"""Summarize pocket coverage and fallback rates for manuscript reporting."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pocket-contract", required=True)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()

    path = Path(args.pocket_contract).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {})
    if not isinstance(entries, dict) or not entries:
        raise SystemExit("contract has no target entries")
    methods = collections.Counter(
        str(entry.get("selection_method", "unreported")) for entry in entries.values()
    )
    sources = collections.Counter(
        str(entry.get("source_group", "unreported")) for entry in entries.values()
    )
    total = len(entries)
    fallback = methods["whole_target_small_protein_fallback"]
    available = sum(bool(entry.get("pocket_available", False)) for entry in entries.values())
    result = {
        "targets": total,
        "available_fpockets": available,
        "whole_target_small_protein_fallback": fallback,
        "fallback_fraction": fallback / total,
        "selection_methods": dict(sorted(methods.items())),
        "source_groups": dict(sorted(sources.items())),
    }
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"Targets: {total}")
    print(f"Usable fpocket entries: {available}")
    print(f"Whole-protein fallback: {fallback}/{total} = {fallback / total:.4%}")
    print("Selection methods:")
    for name, count in sorted(methods.items()):
        print(f"  - {name}: {count}")
    print("Structure source groups:")
    for name, count in sorted(sources.items()):
        print(f"  - {name}: {count}")


if __name__ == "__main__":
    main()
