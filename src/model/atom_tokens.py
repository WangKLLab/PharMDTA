"""Exact atom-token positions in fixed-regex SMILES sequences."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch_geometric.data import Data

from .smiles_tokenizer import regex_smiles_token_atom_indices


@lru_cache(maxsize=131_072)
def _cached_atom_token_positions(
    smiles: str, max_length: int
) -> tuple[int, ...]:
    """Return retained SMILES positions for atom tokens only.

    This is an index-alignment record, not a molecular graph: it has no bond
    edges, coordinates, or topology-derived feature.
    """

    text = str(smiles).strip()
    length = int(max_length)
    if not text:
        raise ValueError("SMILES is empty")
    if length < 2:
        raise ValueError("max_length must be at least two")
    _tokens, token_atom_indices = regex_smiles_token_atom_indices(text)
    positions = tuple(
        token_position
        for token_position, atom_index in enumerate(
            token_atom_indices[: length - 2], start=1
        )
        if atom_index is not None
    )
    if not positions:
        raise ValueError("SMILES truncation retained no atom token")
    return positions


def build_atom_token_selection(smiles: str, *, max_length: int) -> Data:
    """Build a ragged atom-token index record for one ligand."""

    token_positions = _cached_atom_token_positions(str(smiles).strip(), int(max_length))
    atom_count = len(token_positions)
    return Data(
        token_position=torch.tensor(token_positions, dtype=torch.long),
        atom_id=torch.arange(atom_count, dtype=torch.long),
        num_nodes=atom_count,
    )
