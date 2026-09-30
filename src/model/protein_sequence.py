"""Strict protein-sequence resolution and compact categorical encoding."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

PROTEIN_PAD_ID = 0
# The categorical sequence route follows the conventional 25-letter protein
# alphabet.  Keeping this small vocabulary here makes the release independent
# of comparison-baseline code.
CHARPROTSET: dict[str, int] = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
    "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
    "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}
PROTEIN_UNKNOWN_ID = CHARPROTSET["X"]
PROTEIN_VOCAB_SIZE = max(CHARPROTSET.values()) + 1


def normalize_sequence(value: object) -> str:
    """Normalize a protein sequence and reject empty/non-letter values."""

    sequence = "".join(str(value or "").split()).upper()
    if not sequence or sequence in {"NAN", "NONE", "NULL"}:
        raise ValueError("protein sequence is empty")
    invalid = sorted({character for character in sequence if not "A" <= character <= "Z"})
    if invalid:
        raise ValueError(f"protein sequence has non-uppercase-letter symbols: {invalid}")
    return sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_component_sequences(
    path: str | Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"component sequence catalog is missing: {source}")
    frame = pd.read_csv(
        source, usecols=["protein_component_id", "sequence"], low_memory=False
    )
    if frame.empty or frame["protein_component_id"].isna().any():
        raise RuntimeError("component sequence catalog is empty or has missing IDs")
    mapping: dict[str, str] = {}
    for component_id, raw_sequence in zip(
        frame["protein_component_id"], frame["sequence"], strict=True
    ):
        key = str(component_id).strip()
        sequence = normalize_sequence(raw_sequence)
        if not key:
            raise RuntimeError("component sequence catalog has an empty ID")
        previous = mapping.get(key)
        if previous is not None and previous != sequence:
            raise RuntimeError(f"component sequence catalog conflicts for {key}")
        mapping[key] = sequence
    return mapping, {
        "path": str(source),
        "sha256": _sha256_file(source),
        "rows": int(len(frame)),
        "unique_components": int(len(mapping)),
    }


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NAN", "NONE", "NULL"}:
        return None
    number = float(text)
    if not math.isfinite(number) or not number.is_integer():
        raise RuntimeError(f"invalid integer-like protein metadata: {value!r}")
    return int(number)


def resolve_row_protein_sequence(
    row: Mapping[str, object],
    *,
    dataset_name: str,
    component_sequences: Mapping[str, str],
) -> tuple[str, str]:
    """Resolve an auditable sequence without a protein-ID fallback."""

    dataset = str(dataset_name).strip().lower()
    expected_length = _optional_int(row.get("protein_total_sequence_length"))
    if dataset != "bindingdb":
        sequence = normalize_sequence(row.get("protein_sequence"))
        source = "row_protein_sequence"
    else:
        raw_ids = str(row.get("protein_component_ids", "")).strip()
        try:
            component_ids = json.loads(raw_ids)
        except json.JSONDecodeError as exc:
            raise RuntimeError("BindingDB protein_component_ids is invalid JSON") from exc
        if not isinstance(component_ids, list) or not component_ids or any(
            not isinstance(item, str) or not item.strip() for item in component_ids
        ):
            raise RuntimeError("BindingDB protein_component_ids must be a non-empty list")
        component_ids = [item.strip() for item in component_ids]
        expected_count = _optional_int(row.get("protein_component_count"))
        if expected_count != len(component_ids):
            raise RuntimeError("BindingDB protein component count does not match its ID list")
        missing = [item for item in component_ids if item not in component_sequences]
        if missing:
            raise RuntimeError(
                f"component sequence catalog misses IDs; examples={missing[:3]}"
            )
        sequence = "".join(component_sequences[item] for item in component_ids)
        source = "ordered_component_concatenation"
    if expected_length is not None and len(sequence) != expected_length:
        raise RuntimeError(
            "protein sequence length differs from row metadata: "
            f"{len(sequence)} != {expected_length}"
        )
    return sequence, source


def encode_protein_sequence(sequence: str, *, max_length: int) -> np.ndarray:
    if int(max_length) <= 0:
        raise ValueError("max protein sequence length must be positive")
    normalized = normalize_sequence(sequence)
    encoded = np.zeros(int(max_length), dtype=np.uint8)
    for index, residue in enumerate(normalized[: int(max_length)]):
        encoded[index] = CHARPROTSET.get(residue, PROTEIN_UNKNOWN_ID)
    return encoded
