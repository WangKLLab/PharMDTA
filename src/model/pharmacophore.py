"""Token-aligned RDKit pharmacophore features from canonical SMILES.

Every fixed-regex SMILES token position receives a 12-dimensional local
feature vector. Atom tokens receive RDKit atom pharmacophore/context features;
special and non-atom syntax tokens receive exact zero vectors. The
representation is ligand-only and does not depend on a pose, docking
calculation, or pair-specific interaction record.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import ChemicalFeatures

from .smiles_tokenizer import regex_smiles_token_atom_indices


PHARMACOPHORE_SCHEMA = "rdkit_smiles_token_aligned_atom_pharmacophore_v2"
PHARMACOPHORE_FEATURE_NAMES = (
    "hbond_donor",
    "hbond_acceptor",
    "aromatic",
    "positive_ionizable",
    "negative_ionizable",
    "hydrophobe",
    "zinc_binder",
    "ring_member",
    "heteroatom",
    "halogen",
    "sp2",
    "formal_charge_scaled",
)
PHARMACOPHORE_FEATURE_DIM = len(PHARMACOPHORE_FEATURE_NAMES)
assert PHARMACOPHORE_FEATURE_DIM == 12


def _base_features_path() -> Path:
    path = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
    if not path.is_file():
        raise RuntimeError(f"RDKit BaseFeatures definition is missing: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _feature_factory():
    """Load RDKit's pharmacophore definition exactly once per process."""

    return ChemicalFeatures.BuildFeatureFactory(str(_base_features_path()))


def _feature_atom_ids(feature, molecule: Chem.Mol) -> tuple[int, ...]:
    """Return chemically local atom indices for one RDKit feature hit."""

    atom_ids = tuple(int(index) for index in feature.GetAtomIds())
    if feature.GetFamily() not in {"NegIonizable", "PosIonizable", "ZnBinder"}:
        return atom_ids
    heteroatoms = tuple(
        index
        for index in atom_ids
        if molecule.GetAtomWithIdx(index).GetAtomicNum() not in {1, 6}
    )
    return heteroatoms or atom_ids


def _atom_feature_matrix(molecule: Chem.Mol) -> np.ndarray:
    """Construct the 12-column local pharmacophore/context matrix."""

    atom_count = molecule.GetNumAtoms()
    if atom_count <= 0:
        raise ValueError("SMILES contains no heavy atoms")
    features = np.zeros((atom_count, PHARMACOPHORE_FEATURE_DIM), dtype=np.float32)
    family_columns = {
        "Donor": 0,
        "Acceptor": 1,
        "Aromatic": 2,
        "PosIonizable": 3,
        "NegIonizable": 4,
        "Hydrophobe": 5,
        "LumpedHydrophobe": 5,
        "ZnBinder": 6,
    }
    for feature in _feature_factory().GetFeaturesForMol(molecule):
        column = family_columns.get(feature.GetFamily())
        if column is None:
            continue
        for atom_index in _feature_atom_ids(feature, molecule):
            if not 0 <= atom_index < atom_count:
                raise RuntimeError("RDKit pharmacophore atom index is out of range")
            features[atom_index, column] = 1.0

    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        atomic_number = atom.GetAtomicNum()
        features[index, 2] = max(features[index, 2], float(atom.GetIsAromatic()))
        features[index, 7] = float(atom.IsInRing())
        features[index, 8] = float(atomic_number not in {1, 6})
        features[index, 9] = float(atomic_number in {9, 17, 35, 53})
        features[index, 10] = float(
            atom.GetHybridization() == Chem.HybridizationType.SP2
        )
        features[index, 11] = float(max(-2, min(2, atom.GetFormalCharge()))) / 2.0
    if not np.isfinite(features).all():
        raise RuntimeError("pharmacophore extraction produced non-finite values")
    return features


@lru_cache(maxsize=131_072)
def _cached_atom_features(smiles: str) -> tuple[tuple[float, ...], ...]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit rejected SMILES: {smiles!r}")
    matrix = _atom_feature_matrix(molecule)
    return tuple(tuple(float(value) for value in row) for row in matrix)


def compute_token_aligned_pharmacophore_features(
    smiles: str,
    *,
    max_length: int,
) -> np.ndarray:
    """Return [max_length, 12] features aligned to regex-SMILES tokens.

    Row zero is reserved for the BOS token. Atom features are placed at
    positions one through max_length - 2. The EOS position, padding, and all
    non-atom SMILES tokens remain zero. Atom encounter order is verified
    against RDKit's parsed atom order.
    """

    text = str(smiles).strip()
    if not text:
        raise ValueError("SMILES is empty")
    length = int(max_length)
    if length < 2:
        raise ValueError("max_length must be at least two")
    _tokens, token_atom_indices = regex_smiles_token_atom_indices(text)
    atom_features = np.asarray(_cached_atom_features(text), dtype=np.float32)
    atom_token_count = sum(index is not None for index in token_atom_indices)
    if atom_features.shape != (atom_token_count, PHARMACOPHORE_FEATURE_DIM):
        raise RuntimeError(
            "RDKit atom count does not match fixed-regex SMILES atom tokens: "
            f"atoms={atom_features.shape[0]}, atom_tokens={atom_token_count}, "
            f"smiles={text!r}"
        )
    output = np.zeros((length, PHARMACOPHORE_FEATURE_DIM), dtype=np.float32)
    for position, atom_index in enumerate(token_atom_indices[: length - 2], start=1):
        if atom_index is not None:
            output[position] = atom_features[atom_index]
    if not np.isfinite(output).all():
        raise RuntimeError("token-aligned pharmacophore extraction produced non-finite values")
    return output


def pharmacophore_contract() -> dict[str, object]:
    """Return the immutable feature definition recorded with each run."""

    definition_path = _base_features_path()
    return {
        "schema": PHARMACOPHORE_SCHEMA,
        "source": "canonical_smiles",
        "feature_dim": PHARMACOPHORE_FEATURE_DIM,
        "feature_names": list(PHARMACOPHORE_FEATURE_NAMES),
        "alignment": "fixed_regex_smiles_token_position",
        "atom_assignment": "SMILES_atom_encounter_order_validated_against_RDKit",
        "non_atom_and_special_tokens": "all_zero_feature_vector",
        "feature_definition": "rdkit_BaseFeatures.fdef_plus_local_atom_context",
        "formal_charge_scaled": "clip(formal_charge,-2,2)/2",
        "rdkit_version": str(rdBase.rdkitVersion),
        "base_features_fdef_sha256": _sha256_file(definition_path),
    }
