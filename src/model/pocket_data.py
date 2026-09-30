"""Self-contained runtime contract for native-v6 single-pocket graphs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


TARGET_POCKET_CONTRACT_FORMAT = "rl_mtl.target_ligand_independent_pockets"
TARGET_POCKET_QUALITY_CONTRACT_VERSION = 4
TARGET_POCKET_QUALITY_CONTRACT_SEMANTICS = (
    "target_level_ligand_independent_sequence_aligned_quality_v4"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RUNTIME_GRAPH_FIELDS = frozenset(
    {
        "x",
        "pos",
        "edge_index",
        "residue_physchem",
        # ``torch_geometric.data.Batch`` increments attributes whose names
        # contain ``index``.  Keep the validated zero-based sequence mapping
        # under a neutral runtime name so batching cannot corrupt it.
        "sequence_position",
        "pocket_quality_valid",
    }
)

# This is the immutable schema recorded by the current fpocket-v2 contract.
# Keeping the literal here avoids importing the offline pocket builder during
# training while retaining an exact manifest/graph schema check.
RESIDUE_PHYSCHEM_DIM = 28
RESIDUE_PHYSCHEM_SCHEMA: dict[str, Any] = {
    "name": "canonical_aa_identity_physchem_v1",
    "version": 1,
    "field": "residue_physchem",
    "dim": 28,
    "residue_order": "ACDEFGHIKLMNPQRSTVWY",
    "feature_names": [
        "aa_A", "aa_C", "aa_D", "aa_E", "aa_F", "aa_G", "aa_H",
        "aa_I", "aa_K", "aa_L", "aa_M", "aa_N", "aa_P", "aa_Q",
        "aa_R", "aa_S", "aa_T", "aa_V", "aa_W", "aa_Y",
        "hydropathy_z", "sidechain_volume_z", "positive", "negative",
        "aromatic", "sidechain_hbond_donor", "sidechain_hbond_acceptor",
        "flexibility_z",
    ],
    "continuous_scaling": "zscore_over_20_canonical_amino_acids",
    "sha256": "22a851d51bfbbcefa683e63f76667833743519a58f936eca5881c92f9e16e892",
}


def torch_load_compat(path: str | Path) -> Any:
    """Read a trusted, hash-validated graph artifact using PyTorch 2.6."""
    return torch.load(path, map_location="cpu", weights_only=False)


@dataclass(frozen=True)
class TargetPocketContract:
    path: Path
    version: int
    semantics: str
    identity_column: str
    feature_dim: int
    residue_feature_dim: int
    residue_feature_schema: dict[str, Any]
    graph_paths: dict[str, Path]
    graph_sha256: dict[str, str]
    metric_groups: dict[str, str]
    available_targets: frozenset[str]
    requires_sequence_alignment: bool
    payload: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_sequence_sha256(sequence: str) -> str:
    """Digest the exact normalized sequence used by a v4 pocket graph."""

    normalized = "".join(character for character in str(sequence).upper() if character.isalpha())
    if not normalized:
        raise ValueError("cannot hash an empty target sequence")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _require_flag(payload: dict[str, Any], key: str, expected: bool, path: Path) -> None:
    if payload.get(key) is not expected:
        raise ValueError(
            f"{path}: {key} must be {expected!r}, got {payload.get(key)!r}"
        )


def _resolve_graph_path(contract_path: Path, raw_path: Any) -> Path:
    graph_path = Path(str(raw_path))
    if not graph_path.is_absolute():
        graph_path = contract_path.parent / graph_path
    return graph_path.resolve()


def load_target_pocket_contract(path: str | Path) -> TargetPocketContract:
    """Validate the exact ligand-independent sequence-aligned fpocket-v4 training contract."""

    contract_path = Path(path).expanduser().resolve()
    if not contract_path.is_file() or contract_path.is_symlink():
        raise FileNotFoundError(f"target-pocket contract is not a regular file: {contract_path}")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("format") != TARGET_POCKET_CONTRACT_FORMAT:
        raise ValueError(f"{contract_path}: unsupported target-pocket format")
    version = int(payload.get("version", -1))
    semantics = str(payload.get("semantics", ""))
    is_v4 = (version, semantics) == (
        TARGET_POCKET_QUALITY_CONTRACT_VERSION,
        TARGET_POCKET_QUALITY_CONTRACT_SEMANTICS,
    )
    if not is_v4:
        raise ValueError(
            f"{contract_path}: DTA requires the native-v6 fpocket-v4 contract, "
            f"got version={version}, semantics={semantics!r}"
        )

    for key, expected in (
        ("target_level", True),
        ("query_specific", False),
        ("ligand_independent", True),
        ("ligand_coordinates_used", False),
        ("ligand_identity_used", False),
        ("fpocket_custom_ligand_argument", False),
        ("standard_amino_acid_only", True),
        ("ligand_residue_records_retained", False),
        ("apo_conformation_guaranteed", False),
    ):
        _require_flag(payload, key, expected, contract_path)
    if payload.get("multi_pocket") not in (None, False):
        raise ValueError(f"{contract_path}: fpocket-v2 cannot enable multi_pocket")
    if payload.get("protein_only_record_types") != ["CRYST1", "ATOM", "TER", "END"]:
        raise ValueError(f"{contract_path}: protein-only record contract is invalid")
    if payload.get("smiles_fields_read") != []:
        raise ValueError(f"{contract_path}: smiles_fields_read must be empty")
    if payload.get("affinity_label_fields_read") != []:
        raise ValueError(f"{contract_path}: affinity_label_fields_read must be empty")
    if int(payload.get("num_failed_targets", -1)) != 0:
        raise ValueError(f"{contract_path}: partial pocket coverage is forbidden")
    for metadata_key in ("builder", "fpocket", "mkdssp"):
        metadata = payload.get(metadata_key)
        if not isinstance(metadata, dict) or not SHA256_RE.fullmatch(
            str(metadata.get("sha256", "")).lower()
        ):
            raise ValueError(f"{contract_path}: {metadata_key} SHA256 is required")

    identity_column = str(payload.get("identity_column", "")).strip()
    feature_dim = int(payload.get("feature_dim", 0))
    residue_feature_dim = int(payload.get("residue_feature_dim", 0) or 0)
    residue_schema = payload.get("residue_feature_schema")
    if not identity_column:
        raise ValueError(f"{contract_path}: identity_column is required")
    if feature_dim <= 0:
        raise ValueError(f"{contract_path}: feature_dim must be positive")
    if residue_feature_dim != RESIDUE_PHYSCHEM_DIM:
        raise ValueError(
            f"{contract_path}: residue_feature_dim must be {RESIDUE_PHYSCHEM_DIM}"
        )
    if residue_schema != RESIDUE_PHYSCHEM_SCHEMA:
        raise ValueError(f"{contract_path}: residue feature schema mismatch")

    raw_paths = payload.get("graph_paths")
    raw_hashes = payload.get("graph_sha256")
    entries = payload.get("entries")
    if not isinstance(raw_paths, dict) or not raw_paths:
        raise ValueError(f"{contract_path}: graph_paths must be a non-empty mapping")
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(raw_paths):
        raise ValueError(f"{contract_path}: graph_sha256 keys must match graph_paths")
    if not isinstance(entries, dict) or set(entries) != set(raw_paths):
        raise ValueError(f"{contract_path}: entry keys must match graph_paths")
    if int(payload.get("num_targets", -1)) != len(raw_paths):
        raise ValueError(f"{contract_path}: num_targets does not match graph_paths")
    if int(payload.get("num_requested_targets", -1)) != len(raw_paths):
        raise ValueError(f"{contract_path}: complete requested coverage is required")

    graph_paths: dict[str, Path] = {}
    graph_hashes: dict[str, str] = {}
    metric_groups: dict[str, str] = {}
    available_targets: set[str] = set()
    missing_files: list[str] = []
    for raw_target, raw_path in raw_paths.items():
        target = str(raw_target).strip()
        if not target or target in graph_paths:
            raise ValueError(f"{contract_path}: empty/duplicate normalized graph target")
        graph_path = _resolve_graph_path(contract_path, raw_path)
        if not graph_path.is_file() or graph_path.is_symlink():
            missing_files.append(str(graph_path))
        expected_hash = str(raw_hashes[raw_target]).strip().lower()
        if not SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"{contract_path}: invalid graph SHA256 for {target}")
        entry = entries[raw_target]
        if not isinstance(entry, dict):
            raise ValueError(f"{contract_path}: malformed entry for {target}")
        if str(entry.get("target", "")).strip() != target or entry.get("status") != "ok":
            raise ValueError(f"{contract_path}: invalid success entry for {target}")
        source_group = str(entry.get("source_group", "")).strip()
        if source_group == "internal_target":
            metric_group = "target_fpocket_internal_structure"
        elif source_group == "external_only_target":
            metric_group = "target_fpocket_external_structure"
        else:
            raise ValueError(
                f"{contract_path}: unsupported source_group={source_group!r}"
            )
        if str(entry.get("selection_method", "")).strip() not in {
            "fpocket",
            "whole_target_small_protein_fallback",
            "masked_unavailable",
        }:
            raise ValueError(f"{contract_path}: invalid selection method for {target}")
        expected_sequence_hash = str(entry.get("target_sequence_sha256", "")).lower()
        if not SHA256_RE.fullmatch(expected_sequence_hash):
            raise ValueError(
                f"{contract_path}: v4 entry lacks target_sequence_sha256 for {target}"
            )
        if bool(entry.get("pocket_available", False)):
            if str(entry.get("selection_method", "")).strip() != "fpocket":
                raise ValueError(
                    f"{contract_path}: v4 pocket_available target is not fpocket: {target}"
                )
            available_targets.add(target)
        graph_paths[target] = graph_path
        graph_hashes[target] = expected_hash
        metric_groups[target] = metric_group
    if missing_files:
        raise FileNotFoundError(
            "target-pocket graph files are missing; "
            f"count={len(missing_files)}, examples={missing_files[:5]}"
        )

    return TargetPocketContract(
        path=contract_path,
        version=version,
        semantics=semantics,
        identity_column=identity_column,
        feature_dim=feature_dim,
        residue_feature_dim=residue_feature_dim,
        residue_feature_schema=dict(residue_schema),
        graph_paths=graph_paths,
        graph_sha256=graph_hashes,
        metric_groups=metric_groups,
        available_targets=frozenset(available_targets),
        requires_sequence_alignment=True,
        payload=payload,
    )


def canonicalize_target_pocket_graph_for_batching(graph: Any) -> Any:
    sequence_index = getattr(graph, "sequence_index", None)
    if torch.is_tensor(sequence_index):
        graph.sequence_position = sequence_index.clone()
    for field in tuple(graph.keys()):
        if field not in RUNTIME_GRAPH_FIELDS:
            del graph[field]
    return graph


def load_validated_target_pocket_graph(
    contract: TargetPocketContract, target: str, *, expected_sequence: str | None = None
) -> Any:
    normalized_target = str(target).strip()
    if normalized_target not in contract.graph_paths:
        raise KeyError(f"target is absent from pocket contract: {normalized_target!r}")
    graph_path = contract.graph_paths[normalized_target]
    actual_hash = sha256_file(graph_path)
    expected_hash = contract.graph_sha256[normalized_target]
    if actual_hash != expected_hash:
        raise RuntimeError(
            "target-pocket graph hash mismatch: "
            f"target={normalized_target}, expected={expected_hash}, actual={actual_hash}"
        )
    graph = torch_load_compat(graph_path)
    x = getattr(graph, "x", None)
    edge_index = getattr(graph, "edge_index", None)
    if (
        not torch.is_tensor(x)
        or x.ndim != 2
        or int(x.shape[0]) < 1
        or int(x.shape[1]) != contract.feature_dim
        or not torch.isfinite(x).all()
    ):
        raise RuntimeError(f"invalid target-pocket node features: {graph_path}")
    if (
        not torch.is_tensor(edge_index)
        or edge_index.ndim != 2
        or tuple(edge_index.shape[:1]) != (2,)
        or int(edge_index.shape[1]) < 1
    ):
        raise RuntimeError(f"invalid target-pocket edge_index: {graph_path}")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(x.shape[0]):
        raise RuntimeError(f"target-pocket edge index is out of bounds: {graph_path}")
    if str(getattr(graph, "target_identity", "")).strip() != normalized_target:
        raise RuntimeError(f"target-pocket graph identity mismatch: {graph_path}")
    if getattr(graph, "ligand_independent", None) is not True:
        raise RuntimeError(f"target-pocket graph lacks ligand-independent marker: {graph_path}")
    if str(getattr(graph, "pocket_contract", "")).strip() != contract.semantics:
        raise RuntimeError(f"target-pocket graph semantics mismatch: {graph_path}")
    residue_features = getattr(graph, "residue_physchem", None)
    if (
        not torch.is_tensor(residue_features)
        or residue_features.ndim != 2
        or tuple(residue_features.shape)
        != (int(x.shape[0]), contract.residue_feature_dim)
        or not torch.isfinite(residue_features).all()
    ):
        raise RuntimeError(f"invalid target-pocket residue features: {graph_path}")
    schema = contract.residue_feature_schema
    if (
        str(getattr(graph, "residue_feature_schema", "")).strip()
        != str(schema.get("name", ""))
        or str(getattr(graph, "residue_feature_schema_sha256", "")).strip()
        != str(schema.get("sha256", ""))
    ):
        raise RuntimeError(f"target-pocket residue feature schema mismatch: {graph_path}")
    if expected_sequence is None:
        raise RuntimeError("v4 target-pocket loading requires the runtime target sequence")
    expected_hash = normalized_sequence_sha256(expected_sequence)
    graph_hash = str(getattr(graph, "target_sequence_sha256", "")).lower()
    entry_hash = str(contract.payload["entries"][normalized_target].get(
        "target_sequence_sha256", ""
    )).lower()
    if graph_hash != expected_hash or entry_hash != expected_hash:
        raise RuntimeError(
            "target-pocket sequence hash mismatch: "
            f"target={normalized_target}"
        )
    sequence_index = getattr(graph, "sequence_index", None)
    if (
        not torch.is_tensor(sequence_index)
        or sequence_index.dtype != torch.long
        or sequence_index.ndim != 1
        or int(sequence_index.numel()) != int(x.shape[0])
        or bool((sequence_index < 0).any())
        or bool((sequence_index >= len("".join(c for c in expected_sequence.upper() if c.isalpha()))).any())
    ):
        raise RuntimeError(f"invalid v4 sequence_index: {graph_path}")
    pocket_quality = getattr(graph, "pocket_quality", None)
    pocket_quality_valid = getattr(graph, "pocket_quality_valid", None)
    if (
        not torch.is_tensor(pocket_quality)
        or tuple(pocket_quality.shape) != (1, 4)
        or not torch.isfinite(pocket_quality).all()
        or not torch.is_tensor(pocket_quality_valid)
        or tuple(pocket_quality_valid.shape) != (1, 1)
        or not torch.isfinite(pocket_quality_valid).all()
        or bool((pocket_quality_valid < 0).any())
        or bool((pocket_quality_valid > 1).any())
    ):
        raise RuntimeError(f"invalid v4 pocket quality fields: {graph_path}")
    if bool((pocket_quality_valid > 0.5).all()):
        sequence = "".join(c for c in expected_sequence.upper() if c.isalpha())
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        expected_aa = torch.tensor(
            [aa_order.index(sequence[int(index)]) for index in sequence_index.tolist()],
            dtype=torch.long,
        )
        observed_aa = residue_features[:, :20].argmax(dim=1).to(dtype=torch.long)
        if not torch.equal(expected_aa, observed_aa.cpu()):
            raise RuntimeError(
                f"v4 residue identity / sequence_index mismatch: {graph_path}"
            )
    if normalized_target not in contract.available_targets or not bool((pocket_quality_valid > 0.5).all()):
        raise RuntimeError(f"paper inference requires an available, valid fpocket: {graph_path}")
    return canonicalize_target_pocket_graph_for_batching(graph)
