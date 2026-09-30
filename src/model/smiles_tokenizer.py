"""Dependency-light regex SMILES tokenizer used by the paper model."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np


SMILES_VOCAB_PATH = Path(__file__).resolve().parent / "assets" / "smiles_regex_vocab.json"
MAX_SMILES_TOKENS = 128
_PATTERN = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
_SPECIAL_IDS = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}
_ATOM_TOKENS = frozenset({"B", "C", "N", "O", "S", "P", "F", "I", "Br", "Cl", "b", "c", "n", "o", "s", "p"})


def load_smiles_vocabulary(path: str | Path = SMILES_VOCAB_PATH) -> dict[str, int]:
    """Load and validate the fixed regex-SMILES vocabulary."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SMILES vocabulary is missing: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid SMILES vocabulary JSON: {source}") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise RuntimeError("SMILES vocabulary must be a non-empty object")
    vocabulary = {str(token): int(index) for token, index in raw.items()}
    if {key: vocabulary.get(key) for key in _SPECIAL_IDS} != _SPECIAL_IDS:
        raise RuntimeError("SMILES vocabulary has incompatible special-token IDs")
    if sorted(vocabulary.values()) != list(range(len(vocabulary))):
        raise RuntimeError("SMILES vocabulary IDs must be contiguous from zero")
    return vocabulary


def regex_smiles_tokens(smiles: str) -> list[str]:
    """Tokenize SMILES using the fixed paper-release regular expression."""

    return _PATTERN.findall(str(smiles))


def is_smiles_atom_token(token: str) -> bool:
    """Return whether one fixed-regex token introduces a SMILES atom."""

    text = str(token)
    return (text.startswith("[") and text.endswith("]")) or text in _ATOM_TOKENS


def regex_smiles_token_atom_indices(smiles: str) -> tuple[list[str], list[int | None]]:
    """Map each regex-SMILES token to its zero-based atom encounter index.

    SMILES introduces atoms in textual encounter order. Branch, bond, ring,
    stereochemical, and punctuation tokens therefore receive ``None``; every
    atom token receives its corresponding atom index. The RDKit alignment
    layer validates that the resulting atom count equals the parsed molecule.
    """

    tokens = regex_smiles_tokens(smiles)
    mapping: list[int | None] = []
    atom_index = 0
    for token in tokens:
        if is_smiles_atom_token(token):
            mapping.append(atom_index)
            atom_index += 1
        else:
            mapping.append(None)
    return tokens, mapping


def encode_smiles_regex(
    smiles: str,
    *,
    vocabulary: Mapping[str, int],
    max_length: int = MAX_SMILES_TOKENS,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Encode one SMILES as ``<s> tokens </s>`` plus fixed-length padding."""

    length = int(max_length)
    if length < 2:
        raise ValueError("max_length must be at least two")
    raw = str(smiles).strip()
    if not raw:
        raise ValueError("SMILES is empty")
    tokens = regex_smiles_tokens(raw)
    ids = [int(vocabulary["<s>"])]
    unknown = 0
    for token in tokens[: length - 2]:
        value = vocabulary.get(token)
        if value is None:
            unknown += 1
            value = int(vocabulary["<unk>"])
        ids.append(int(value))
    ids.append(int(vocabulary["</s>"]))
    output = np.full((length,), int(vocabulary["<pad>"]), dtype=np.uint16)
    mask = np.zeros((length,), dtype=np.uint8)
    output[: len(ids)] = np.asarray(ids, dtype=np.uint16)
    mask[: len(ids)] = 1
    return output, mask, len(tokens), unknown

