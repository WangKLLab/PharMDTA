"""Immutable input contracts for the SMILES-based affinity model.

The trainer reads already curated train/validation/test CSV files. Its formal
default never repairs molecules, changes labels, reassigns splits, or
constructs a pair-specific interaction descriptor at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PHYSICAL_PAFFINITY_LABEL_KIND = "physical_paffinity"
KIBA_RAW_LABEL_KIND = "kiba_raw"
CLEAN_CONTRACT_FORMAT = "rl_mtl.minimal_joint_clean_splits"
CLEAN_CONTRACT_VERSION = 4
DATASET_NATIVE_REVISION = 6
FIXED_SPLIT_PROTOCOL = "deepdtagen_official_fixed_test_single_pair_holdout_val_no_cv"
BINDINGDB_FULL_IC50_SPLIT_PROTOCOL = "bindingdb_full_ic50_random_pair_64_16_20_seed42_no_cv"
FIXED_SPLIT_PROTOCOLS = frozenset(
    {FIXED_SPLIT_PROTOCOL, BINDINGDB_FULL_IC50_SPLIT_PROTOCOL, "deepdtagen_drug_wise_64_16_20_no_cv"}
)
LIPINSKI_FILTER_FORMAT = "rdkit_lipinski_filter"
LIPINSKI_FILTER_VERSION = 1
LIPINSKI_RULE_VERSION = "lipinski_mw_lt500_logp_lt5_no_lower_hbd_le5_hba_le10_rot_strict_le10_v1"
LIPINSKI_DESCRIPTOR_COLUMNS = (
    "lipinski_mw",
    "lipinski_logp",
    "lipinski_hbd",
    "lipinski_hba",
    "lipinski_rotatable_bonds",
)
LIPINSKI_AUDIT_COLUMNS = frozenset(
    (*LIPINSKI_DESCRIPTOR_COLUMNS, "lipinski_pass", "lipinski_filter_version")
)


@dataclass(frozen=True)
class DatasetRegressionContract:
    """Endpoint and label semantics allowed for one benchmark."""

    dataset_name: str
    endpoint: str
    label_kind: str
    label_column: str
    label_semantics: str
    label_scale: str
    unit: str
    transform: str
    label_range_inclusive: tuple[float, float] | None
    higher_is_stronger: bool | None

    @property
    def affinity_label_kind(self) -> str:
        return self.label_kind


DATASET_REGRESSION_CONTRACTS: dict[str, DatasetRegressionContract] = {
    "bindingdb": DatasetRegressionContract(
        dataset_name="bindingdb",
        endpoint="IC50",
        label_kind=PHYSICAL_PAFFINITY_LABEL_KIND,
        label_column="pAffinity",
        label_semantics="pIC50=-log10(IC50[M])",
        label_scale="negative_log10_molar",
        unit="pIC50",
        transform="9-log10(IC50[nM])",
        label_range_inclusive=(0.0, 12.0),
        higher_is_stronger=True,
    ),
    "kiba": DatasetRegressionContract(
        dataset_name="kiba",
        endpoint="KIBA",
        label_kind=KIBA_RAW_LABEL_KIND,
        label_column="kiba_score_raw",
        label_semantics="released integrated KIBA score copied without transformation",
        label_scale="integrated_kiba_score_raw",
        unit="KIBA_score",
        transform="identity",
        label_range_inclusive=None,
        higher_is_stronger=None,
    ),
}


def dataset_regression_contract(dataset_name: str) -> DatasetRegressionContract:
    key = str(dataset_name).strip().lower()
    try:
        return DATASET_REGRESSION_CONTRACTS[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dataset_name={dataset_name!r}; "
            f"expected one of {sorted(DATASET_REGRESSION_CONTRACTS)}"
        ) from exc


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _normalized_endpoint(value: Any) -> str:
    return str(value).strip().casefold().replace("_", "").replace("-", "")


def _resolve_declared_path(manifest_path: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_split_inventory(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    split_paths: Mapping[str, Path],
) -> None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise RuntimeError(f"{manifest_path}: manifest lacks outputs")
    expected_splits = {"train", "val", "test"}
    if set(split_paths) not in (expected_splits, {"train", "val"}) or not expected_splits.issubset(outputs):
        raise RuntimeError(f"{manifest_path}: train/val/test inventory is incomplete")
    for split, requested in split_paths.items():
        entry = outputs[split]
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"{manifest_path}: invalid output entry for {split}")
        declared = _resolve_declared_path(manifest_path, entry.get("path", ""))
        requested = Path(requested).expanduser().resolve()
        if declared != requested:
            raise RuntimeError(
                f"{manifest_path}: {split} path differs from requested input; "
                f"manifest={declared}, runtime={requested}"
            )
        if requested.is_symlink() or not requested.is_file():
            raise FileNotFoundError(f"missing {split} split: {requested}")
        if int(entry.get("rows", 0)) <= 0:
            raise RuntimeError(f"{manifest_path}: {split} split must be non-empty")


def _validate_fixed_split_protocol(manifest: Mapping[str, Any], path: Path) -> None:
    configuration = manifest.get("configuration", {})
    if not isinstance(configuration, Mapping):
        raise RuntimeError(f"{path}: manifest configuration must be an object")
    protocol = configuration.get("split_protocol", manifest.get("split_protocol"))
    if protocol is None:
        raise RuntimeError(f"{path}: manifest lacks split_protocol")
    if isinstance(protocol, Mapping):
        name = str(protocol.get("name", protocol.get("protocol", ""))).strip()
        if name not in FIXED_SPLIT_PROTOCOLS:
            raise RuntimeError(f"{path}: unsupported split protocol {name!r}")
        if bool(protocol.get("cross_validation", False)):
            raise RuntimeError(f"{path}: cross-validation is not a formal run protocol")
        if bool(protocol.get("test_used_for_model_selection", False)):
            raise RuntimeError(f"{path}: test split may not select the model")
        if protocol.get("fixed_single_split") is False:
            raise RuntimeError(f"{path}: split protocol is not fixed")
    elif str(protocol).strip().lower() not in FIXED_SPLIT_PROTOCOLS:
        raise RuntimeError(f"{path}: unsupported split_protocol={protocol!r}")


def _validate_lipinski_manifest(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError("manifest lacks configuration.lipinski_filter")
    expected = {
        "format": LIPINSKI_FILTER_FORMAT,
        "version": LIPINSKI_FILTER_VERSION,
        "filter_version": LIPINSKI_RULE_VERSION,
        "descriptor_columns": list(LIPINSKI_DESCRIPTOR_COLUMNS),
        "model_role": "audit_only_not_model_input",
        "non_finite_policy": "drop",
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"invalid Lipinski filter contract: {mismatches}")
    expected_rules = {
        "molecular_weight": {"descriptor": "Descriptors.MolWt", "operator": "<", "threshold": 500.0},
        "mol_logp": {"descriptor": "Crippen.MolLogP", "operator": "<", "threshold": 5.0, "lower_bound": None},
        "hbond_donors": {"descriptor": "Lipinski.NumHDonors", "operator": "<=", "threshold": 5},
        "hbond_acceptors": {"descriptor": "Lipinski.NumHAcceptors", "operator": "<=", "threshold": 10},
        "rotatable_bonds": {"descriptor": "rdMolDescriptors.CalcNumRotatableBonds", "operator": "<=", "threshold": 10, "option": "Strict"},
    }
    rules = value.get("rules")
    if not isinstance(rules, Mapping) or dict(rules) != expected_rules:
        raise RuntimeError("Lipinski rule definition differs from the formal contract")


def _validate_frame_lipinski_contract(frame: pd.DataFrame, path: Path) -> None:
    missing = sorted(LIPINSKI_AUDIT_COLUMNS.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: Lipinski audit columns are missing: {missing}")
    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        for column in LIPINSKI_DESCRIPTOR_COLUMNS
    }
    if any(not np.isfinite(values).all() for values in numeric.values()):
        raise RuntimeError(f"{path}: Lipinski descriptors contain non-finite values")
    for column in ("lipinski_hbd", "lipinski_hba", "lipinski_rotatable_bonds"):
        values = numeric[column]
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise RuntimeError(f"{path}: {column} must contain non-negative integers")
    passed = frame["lipinski_pass"].map(
        lambda value: value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower()
    )
    if not passed.isin({True, "true", "1"}).all():
        raise RuntimeError(f"{path}: retained rows must all pass the Lipinski filter")
    versions = {str(value).strip() for value in frame["lipinski_filter_version"].tolist()}
    if versions != {LIPINSKI_RULE_VERSION}:
        raise RuntimeError(f"{path}: Lipinski filter version mismatch")
    valid = (
        (numeric["lipinski_mw"] < 500.0)
        & (numeric["lipinski_logp"] < 5.0)
        & (numeric["lipinski_hbd"] <= 5.0)
        & (numeric["lipinski_hba"] <= 10.0)
        & (numeric["lipinski_rotatable_bonds"] <= 10.0)
    )
    if not valid.all():
        raise RuntimeError(f"{path}: retained rows violate the Lipinski filter")


def _clean_manifest_for_csv(csv_path: Path) -> tuple[dict[str, Any], str]:
    if csv_path.parent.name != "splits":
        raise RuntimeError(f"split CSV must live in a splits/ directory: {csv_path}")
    manifest_path = csv_path.parent.parent / "manifest.json"
    manifest = _load_json(manifest_path, label="data manifest")
    if manifest.get("contract_format") != CLEAN_CONTRACT_FORMAT:
        raise RuntimeError(f"unsupported data contract: {manifest_path}")
    if int(manifest.get("contract_version", -1)) != CLEAN_CONTRACT_VERSION:
        raise RuntimeError(f"unsupported data contract version: {manifest_path}")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping) or int(
        configuration.get("dataset_native_revision", -1)
    ) != DATASET_NATIVE_REVISION:
        raise RuntimeError(f"{manifest_path}: expected dataset_native_revision={DATASET_NATIVE_REVISION}")
    _validate_split_inventory(
        manifest,
        manifest_path=manifest_path.resolve(),
        split_paths={
            name: csv_path.parent / f"{name}.csv"
            for name in (("train", "val", "test") if csv_path.stem == "test" else ("train", "val"))
        },
    )
    overlaps = manifest.get("pair_overlap", {})
    if not isinstance(overlaps, Mapping) or any(
        int(overlaps.get(name, -1)) != 0 for name in ("train_val", "train_test", "val_test")
    ):
        raise RuntimeError(f"{manifest_path}: cross-split pair leakage is reported")
    _validate_fixed_split_protocol(manifest, manifest_path)
    _validate_lipinski_manifest(configuration.get("lipinski_filter"))
    return manifest, csv_path.stem


def _manifest_regression_contract(
    manifest: Mapping[str, Any], *, expected_dataset_name: str, path: Path
) -> DatasetRegressionContract:
    expected = dataset_regression_contract(expected_dataset_name)
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RuntimeError(f"{path}: manifest configuration must be an object")
    actual_dataset = str(configuration.get("dataset", configuration.get("dataset_name", ""))).strip().lower()
    if actual_dataset != expected.dataset_name:
        raise RuntimeError(f"{path}: dataset contract mismatch")
    if _normalized_endpoint(configuration.get("endpoint", "")) != _normalized_endpoint(expected.endpoint):
        raise RuntimeError(f"{path}: endpoint contract mismatch")
    required_values = {
        "affinity_label_kind": expected.label_kind,
        "label_column": expected.label_column,
        "label_scale": expected.label_scale,
        "label_unit": expected.unit,
        "label_transform": expected.transform,
    }
    for key, value in required_values.items():
        if str(configuration.get(key, "")).strip() != str(value):
            raise RuntimeError(f"{path}: {key} contract mismatch")
    if str(configuration.get("label_semantics", "")).strip().casefold() != expected.label_semantics.casefold():
        raise RuntimeError(f"{path}: label_semantics contract mismatch")
    raw_range = configuration.get("label_range_inclusive")
    if expected.label_range_inclusive is None:
        if raw_range is not None:
            raise RuntimeError(f"{path}: KIBA must not declare a pAffinity range")
    else:
        try:
            observed_range = tuple(float(value) for value in raw_range)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{path}: invalid label_range_inclusive") from exc
        if observed_range != expected.label_range_inclusive:
            raise RuntimeError(f"{path}: label range contract mismatch")
    if configuration.get("higher_is_stronger") is not expected.higher_is_stronger:
        raise RuntimeError(f"{path}: higher_is_stronger contract mismatch")
    _validate_fixed_split_protocol(manifest, path)
    statistics = manifest.get("train_label_statistics")
    if not isinstance(statistics, Mapping):
        raise RuntimeError(f"{path}: manifest lacks train label statistics")
    expected_statistics = {
        "fit_split": "train",
        "fit_scope": "final_unique_train_pairs_only",
        "label_column": expected.label_column,
        "label_kind": expected.label_kind,
        "unit": expected.unit,
        "transform": expected.transform,
        "validation_or_test_used": False,
    }
    if any(statistics.get(key) != value for key, value in expected_statistics.items()):
        raise RuntimeError(f"{path}: invalid train label-statistics provenance")
    for key in ("count", "mean", "std_population", "epsilon", "scale", "min", "max"):
        try:
            value = float(statistics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"{path}: invalid train_label_statistics[{key!r}]") from exc
        if not np.isfinite(value):
            raise RuntimeError(f"{path}: non-finite train_label_statistics[{key!r}]")
    if int(statistics["count"]) != int(manifest["outputs"]["train"].get("rows", -1)):
        raise RuntimeError(f"{path}: train label count does not match train split")
    if not np.isclose(
        float(statistics["scale"]),
        max(float(statistics["std_population"]), float(statistics["epsilon"])),
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError(f"{path}: invalid train label normalization scale")
    return expected


def _validate_train_label_statistics(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    path: Path,
    contract: DatasetRegressionContract,
) -> None:
    statistics = manifest["train_label_statistics"]
    required = {"canonical_smiles", "protein_identity_key", contract.label_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: cannot verify train label statistics; missing={missing}")
    values = pd.to_numeric(frame[contract.label_column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{path}: train labels contain non-finite values")
    observed = {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
    for key, value in observed.items():
        expected = float(statistics[key])
        matches = int(value) == int(expected) if key == "count" else np.isclose(
            value, expected, rtol=1.0e-12, atol=1.0e-12
        )
        if not bool(matches):
            raise RuntimeError(f"{path}: train label statistic {key} mismatch")


def _validate_frame_regression_contract(
    frame: pd.DataFrame, path: Path, contract: DatasetRegressionContract
) -> None:
    required = {
        "dataset",
        "affinity_label_kind",
        "affinity_label_semantics",
        "affinity_measurement_source",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"{path} lacks regression-contract columns: {missing}")
    datasets = {str(value).strip().lower() for value in frame["dataset"].dropna()}
    kinds = {str(value).strip().lower() for value in frame["affinity_label_kind"].dropna()}
    semantics = {
        str(value).strip().casefold() for value in frame["affinity_label_semantics"].dropna()
    }
    endpoints = {
        _normalized_endpoint(value) for value in frame["affinity_measurement_source"].dropna()
    }
    if datasets != {contract.dataset_name} or kinds != {contract.label_kind}:
        raise RuntimeError(f"{path}: dataset or label-kind contract mismatch")
    if semantics != {contract.label_semantics.casefold()}:
        raise RuntimeError(f"{path}: label-semantics contract mismatch")
    if endpoints != {_normalized_endpoint(contract.endpoint)}:
        raise RuntimeError(f"{path}: endpoint contract mismatch")


def validate_native_v6_training_input(
    *, data_dir: str | Path, dataset_name: str, split_paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Validate the immutable curated input set before training.

    The historical ``native_v6`` label identifies the source-data revision;
    it does not add an extra model modality. The returned receipt is written
    into each run directory for manuscript-to-result traceability.
    """

    expected = dataset_regression_contract(dataset_name)
    root = Path(data_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, label="data manifest")
    if manifest.get("contract_format") != CLEAN_CONTRACT_FORMAT or int(
        manifest.get("contract_version", -1)
    ) != CLEAN_CONTRACT_VERSION:
        raise RuntimeError("formal training requires the curated clean-split contract")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping) or int(
        configuration.get("dataset_native_revision", -1)
    ) != DATASET_NATIVE_REVISION:
        raise RuntimeError("formal training requires the declared source-data revision")
    if str(configuration.get("dataset", "")).strip().lower() != expected.dataset_name:
        raise RuntimeError("requested dataset does not match the data manifest")
    normalized = {name: Path(path).expanduser().resolve() for name, path in split_paths.items()}
    _validate_split_inventory(manifest, manifest_path=manifest_path, split_paths=normalized)
    _validate_fixed_split_protocol(manifest, manifest_path)
    _validate_lipinski_manifest(configuration.get("lipinski_filter"))
    overlaps = manifest.get("pair_overlap", {})
    if not isinstance(overlaps, Mapping) or any(
        int(overlaps.get(name, -1)) != 0 for name in ("train_val", "train_test", "val_test")
    ):
        raise RuntimeError("data manifest reports cross-split pair leakage")
    _manifest_regression_contract(manifest, expected_dataset_name=expected.dataset_name, path=manifest_path)
    return {
        "manifest": {"path": str(manifest_path)},
        "contract_format": CLEAN_CONTRACT_FORMAT,
        "contract_version": CLEAN_CONTRACT_VERSION,
        "dataset_native_revision": DATASET_NATIVE_REVISION,
        "split_protocol": configuration.get("split_protocol", manifest.get("split_protocol")),
        "split_paths": {
            split: str(path) for split, path in sorted(normalized.items())
        },
    }

