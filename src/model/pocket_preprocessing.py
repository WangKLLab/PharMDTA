"""Protein-only fpocket parsing, alignment, and graph construction.

Adapted from the project's established offline preprocessing implementation.
"""
from __future__ import annotations
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from Bio.Align import PairwiseAligner
from torch_geometric.data import Data
from .pocket_data import RESIDUE_PHYSCHEM_DIM, RESIDUE_PHYSCHEM_SCHEMA
from .residue_features import residue_physchem_vector

POCKET_FEATURE_DIM = 2568

DSSP_DIM = 8

AMINO_ACIDS = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

STANDARD_AMINO_ACIDS = frozenset(AMINO_ACIDS)

DSSP_MAP = {
    "H": 0,
    "B": 1,
    "E": 2,
    "G": 3,
    "I": 4,
    "T": 5,
    "S": 6,
    "C": 7,
    " ": 7,
}

POCKET_FILE_RE = re.compile(r"pocket(\d+)_atm\.pdb$")

def _pdb_residue_key(line: str) -> tuple[str, int]:
    return line[21:22].strip(), int(line[22:26].strip())

def _normalize_sequence(sequence: str) -> str:
    return "".join(
        character for character in str(sequence).strip().upper() if character.isalpha()
    )

def _alignment_index_map(
    reference: str,
    observed: str,
) -> tuple[dict[int, int], float, float]:
    reference = _normalize_sequence(reference)
    observed = _normalize_sequence(observed)
    if not reference or not observed:
        return {}, 0.0, 0.0
    aligner = PairwiseAligner(
        mode="local",
        match_score=2.0,
        mismatch_score=-1.0,
        open_gap_score=-3.0,
        extend_gap_score=-0.5,
    )
    alignments = aligner.align(reference, observed)
    if len(alignments) == 0:
        return {}, 0.0, 0.0
    alignment = alignments[0]
    aligned_reference, aligned_observed = alignment.aligned
    mapping: dict[int, int] = {}
    matches = 0
    for (ref_start, ref_end), (obs_start, obs_end) in zip(
        aligned_reference,
        aligned_observed,
    ):
        block_length = min(
            int(ref_end - ref_start),
            int(obs_end - obs_start),
        )
        for offset in range(block_length):
            reference_index = int(ref_start) + offset
            observed_index = int(obs_start) + offset
            mapping[observed_index] = reference_index
            matches += int(reference[reference_index] == observed[observed_index])
    coverage = len(mapping) / max(1, len(observed))
    identity = matches / max(1, len(mapping))
    return mapping, float(coverage), float(identity)

def parse_residues(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    residues: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            res_name = line[17:20].strip().upper()
            if res_name not in STANDARD_AMINO_ACIDS:
                continue
            try:
                key = _pdb_residue_key(line)
                atom_name = line[12:16].strip()
                coord = np.asarray(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float32,
                )
            except (TypeError, ValueError):
                continue
            residue = residues.setdefault(
                key,
                {"res_name": res_name, "atoms": [], "coords": []},
            )
            residue["atoms"].append(atom_name)
            residue["coords"].append(coord)
    for residue in residues.values():
        residue["coords"] = np.asarray(residue["coords"], dtype=np.float32)
    return residues

def parse_pocket_residue_keys(path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[17:20].strip().upper() not in STANDARD_AMINO_ACIDS:
                continue
            try:
                keys.add(_pdb_residue_key(line))
            except (TypeError, ValueError):
                continue
    return keys

def parse_dssp(path: Path) -> dict[tuple[str, int], str]:
    result: dict[tuple[str, int], str] = {}
    if not path.is_file():
        return result
    started = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "  #  RESIDUE AA STRUCTURE" in line:
                started = True
                continue
            if not started:
                continue
            try:
                number = int(line[5:10].strip())
                chain = line[11:12].strip()
                result[(chain, number)] = line[16:17]
            except (TypeError, ValueError):
                continue
    return result

def parse_fpocket_scores(path: Path) -> dict[int, float]:
    scores: dict[int, float] = {}
    if not path.is_file():
        return scores
    current: int | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = re.match(r"Pocket\s+(\d+)\s*:", line.strip())
            if match:
                current = int(match.group(1))
                continue
            if current is not None and line.strip().startswith("Score"):
                try:
                    scores[current] = float(line.split(":", 1)[1].strip())
                except (IndexError, ValueError):
                    pass
    return scores

def build_graph(
    *,
    residue_keys: set[tuple[str, int]],
    residues: dict[tuple[str, int], dict[str, Any]],
    residue_to_sequence: dict[tuple[str, int], int],
    embedding: np.ndarray,
    dssp: dict[tuple[str, int], str],
    max_sequence_index: int,
    edge_distance_a: float,
) -> Data:
    nodes: list[tuple[tuple[str, int], int]] = []
    for key in sorted(residue_keys):
        sequence_index = residue_to_sequence.get(key)
        if sequence_index is None or sequence_index > int(max_sequence_index):
            continue
        embedding_index = sequence_index + 1
        if embedding_index >= int(embedding.shape[0]):
            continue
        nodes.append((key, embedding_index))
    if not nodes:
        raise ValueError("fpocket residues do not map to the contracted sequence")

    features: list[np.ndarray] = []
    residue_features: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    residue_labels: list[str] = []
    for key, embedding_index in nodes:
        residue = residues[key]
        atom_names = residue["atoms"]
        if "CA" in atom_names:
            coordinate = residue["coords"][atom_names.index("CA")]
        else:
            coordinate = residue["coords"].mean(axis=0)
        dssp_feature = np.zeros(DSSP_DIM, dtype=np.float32)
        dssp_feature[DSSP_MAP.get(dssp.get(key, " "), 7)] = 1.0
        features.append(
            np.concatenate(
                (embedding[embedding_index].astype(np.float32), dssp_feature)
            )
        )
        residue_features.append(
            residue_physchem_vector(AMINO_ACIDS[residue["res_name"]])
        )
        coordinates.append(coordinate.astype(np.float32))
        residue_labels.append(f"{key[0]}:{key[1]}")

    x = torch.from_numpy(np.stack(features)).to(dtype=torch.float32)
    residue_physchem = torch.from_numpy(
        np.stack(residue_features)
    ).to(dtype=torch.float32)
    pos = torch.from_numpy(np.stack(coordinates)).to(dtype=torch.float32)
    if int(x.shape[1]) != POCKET_FEATURE_DIM:
        raise ValueError(f"pocket feature dimension mismatch: {tuple(x.shape)}")
    if tuple(residue_physchem.shape) != (len(nodes), RESIDUE_PHYSCHEM_DIM):
        raise ValueError(
            "residue physicochemical feature dimension mismatch: "
            f"{tuple(residue_physchem.shape)}"
        )
    if int(pos.shape[0]) == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        distance = torch.cdist(pos, pos)
        mask = (distance < float(edge_distance_a)) & ~torch.eye(
            len(pos), dtype=torch.bool
        )
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    if edge_index.numel() == 0:
        raise ValueError("predicted pocket graph has no radius edges")
    graph = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        residue_physchem=residue_physchem,
    )
    graph.residue_labels = residue_labels
    graph.residue_feature_schema = RESIDUE_PHYSCHEM_SCHEMA["name"]
    return graph

def _independent_chain_map(
    residues: dict[tuple[str, int], dict[str, Any]], sequence: str
) -> tuple[dict[tuple[str, int], int], dict[str, tuple[float, float]]]:
    by_chain: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (chain, number), residue in residues.items():
        by_chain[chain].append((number, AMINO_ACIDS[residue["res_name"]]))
    mapping: dict[tuple[str, int], int] = {}
    quality: dict[str, tuple[float, float]] = {}
    for chain, values in by_chain.items():
        values.sort()
        local, coverage, identity = _alignment_index_map(
            sequence, "".join(letter for _, letter in values)
        )
        quality[chain] = (float(coverage), float(identity))
        if coverage >= 0.60 and identity >= 0.65:
            for observed, sequence_index in local.items():
                mapping[(chain, values[int(observed)][0])] = int(sequence_index)
    return mapping, quality

def _labels_to_indices(labels: list[str], mapping: dict[tuple[str, int], int]) -> list[int | None]:
    result: list[int | None] = []
    for label in labels:
        try:
            chain, number = str(label).split(":", 1)
            result.append(mapping.get((chain, int(number))))
        except (TypeError, ValueError):
            result.append(None)
    return result

def _graph_sequence_identity_ok(graph: Data, sequence: str) -> bool:
    """Require every graph node to agree with its persisted sequence index."""

    indices = getattr(graph, "sequence_index", None)
    residue = getattr(graph, "residue_physchem", None)
    if not torch.is_tensor(indices) or not torch.is_tensor(residue):
        return False
    if indices.ndim != 1 or residue.ndim != 2 or residue.shape[0] != indices.numel():
        return False
    if residue.shape[1] < 20 or bool((indices < 0).any()) or bool((indices >= len(sequence)).any()):
        return False
    order = "ACDEFGHIKLMNPQRSTVWY"
    observed = residue[:, :20].argmax(dim=1).tolist()
    return all(order[int(aa)] == sequence[int(index)] for aa, index in zip(observed, indices.tolist()))

def _connected_edges(pos: torch.Tensor, radius: float) -> tuple[torch.Tensor, int, float]:
    n = int(pos.shape[0])
    if n == 1:
        return torch.tensor([[0], [0]], dtype=torch.long), 0, 0.0
    distance = torch.cdist(pos.float(), pos.float())
    edges = ((distance < radius) & ~torch.eye(n, dtype=torch.bool)).nonzero().t().contiguous()
    if edges.numel() == 0:
        raise RuntimeError("rebuilt pocket has no radius edges")
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for left, right in edges.t().tolist():
        left, right = find(int(left)), find(int(right))
        if left != right:
            parent[right] = left
    groups: dict[int, list[int]] = defaultdict(list)
    for node in range(n):
        groups[find(node)].append(node)
    components = list(groups.values())
    if len(components) == 1:
        return edges, 0, 0.0
    connected = {0}; bridges: list[tuple[int, int]] = []; maximum = 0.0
    while len(connected) < len(components):
        best: tuple[float, int, int, int, int] | None = None
        for left_group in sorted(connected):
            for right_group in range(len(components)):
                if right_group in connected:
                    continue
                block = distance[components[left_group]][:, components[right_group]]
                flat = int(torch.argmin(block).item())
                li, ri = divmod(flat, len(components[right_group]))
                candidate = (float(block[li, ri]), left_group, right_group,
                             components[left_group][li], components[right_group][ri])
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        value, _, right_group, left_node, right_node = best
        connected.add(right_group); maximum = max(maximum, value)
        bridges.extend([(left_node, right_node), (right_node, left_node)])
    return torch.cat((edges, torch.tensor(bridges, dtype=torch.long).t()), dim=1), len(bridges), maximum

def _write_chain_subset(source: Path, output: Path, chains: set[str]) -> None:
    keep: list[str] = []
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("CRYST1"):
            keep.append(line)
        elif line.startswith("ATOM") and line[21:22].strip() in chains and line[17:20].strip().upper() in AMINO_ACIDS:
            keep.append(line)
    if not any(line.startswith("ATOM") for line in keep):
        raise RuntimeError("no runtime-sequence-aligned protein chain in source structure")
    output.write_text("\n".join(keep) + "\nTER\nEND\n", encoding="utf-8")

def _quality(score: float, nodes: int, mapping_coverage: float, bridge_edges: int) -> torch.Tensor:
    # An explicit bridge flag is deliberately retained rather than hiding this
    # construction artifact inside a distance-derived score: quality gating can
    # then learn whether a disconnected pocket that needed graph bridging is
    # trustworthy for the task.
    bridge_indicator = float(int(bridge_edges) > 0)
    score_confidence = 1.0 / (1.0 + math.exp(-((float(score) - 0.20) / 0.10)))
    node_confidence = min(1.0, math.log1p(max(0, int(nodes))) / math.log1p(30.0))
    return torch.tensor(
        [[bridge_indicator, score_confidence, node_confidence, float(mapping_coverage)]],
        dtype=torch.float32,
    )
