"""Canonical residue physicochemical features for pocket preprocessing."""
from __future__ import annotations
import numpy as np

RESIDUE_ORDER = "ACDEFGHIKLMNPQRSTVWY"

RESIDUE_TO_INDEX = {residue: index for index, residue in enumerate(RESIDUE_ORDER)}

HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

SIDECHAIN_VOLUME = {
    "A": 88.6,
    "C": 108.5,
    "D": 111.1,
    "E": 138.4,
    "F": 189.9,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "K": 168.6,
    "L": 166.7,
    "M": 162.9,
    "N": 114.1,
    "P": 112.7,
    "Q": 143.8,
    "R": 173.4,
    "S": 89.0,
    "T": 116.1,
    "V": 140.0,
    "W": 227.8,
    "Y": 193.6,
}

FLEXIBILITY = {
    "A": 0.357,
    "C": 0.346,
    "D": 0.511,
    "E": 0.497,
    "F": 0.314,
    "G": 0.544,
    "H": 0.323,
    "I": 0.462,
    "K": 0.466,
    "L": 0.365,
    "M": 0.295,
    "N": 0.463,
    "P": 0.509,
    "Q": 0.493,
    "R": 0.529,
    "S": 0.507,
    "T": 0.444,
    "V": 0.386,
    "W": 0.305,
    "Y": 0.420,
}

POSITIVE = frozenset("HKR")

NEGATIVE = frozenset("DE")

AROMATIC = frozenset("FHWY")

SIDECHAIN_DONOR = frozenset("CHK NQRSTWY".replace(" ", ""))

SIDECHAIN_ACCEPTOR = frozenset("CDEHNQSTY")

def _standardized(table: dict[str, float]) -> dict[str, float]:
    values = np.asarray([table[residue] for residue in RESIDUE_ORDER], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(std) or std <= 0.0:
        raise RuntimeError("residue property table has zero or invalid variance")
    return {
        residue: float((table[residue] - mean) / std)
        for residue in RESIDUE_ORDER
    }

_HYDROPATHY_Z = _standardized(HYDROPATHY)

_VOLUME_Z = _standardized(SIDECHAIN_VOLUME)

_FLEXIBILITY_Z = _standardized(FLEXIBILITY)

def residue_physchem_vector(one_letter_code: str) -> np.ndarray:
    """Return the fixed 28-dimensional feature vector for one canonical residue."""

    residue = str(one_letter_code).strip().upper()
    if residue not in RESIDUE_TO_INDEX:
        raise ValueError(f"unsupported residue code: {one_letter_code!r}")
    one_hot = np.zeros(len(RESIDUE_ORDER), dtype=np.float32)
    one_hot[RESIDUE_TO_INDEX[residue]] = 1.0
    properties = np.asarray(
        [
            _HYDROPATHY_Z[residue],
            _VOLUME_Z[residue],
            float(residue in POSITIVE),
            float(residue in NEGATIVE),
            float(residue in AROMATIC),
            float(residue in SIDECHAIN_DONOR),
            float(residue in SIDECHAIN_ACCEPTOR),
            _FLEXIBILITY_Z[residue],
        ],
        dtype=np.float32,
    )
    return np.concatenate((one_hot, properties))
