"""Immutable curated BindingDB/KIBA splits and frozen ESM-C residue inputs."""
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from .data_contracts import (_clean_manifest_for_csv, _manifest_regression_contract,
    _validate_frame_lipinski_contract, _validate_frame_regression_contract,
    _validate_train_label_statistics)
from .protein_sequence import resolve_row_protein_sequence, encode_protein_sequence
from .pocket_data import load_target_pocket_contract, load_validated_target_pocket_graph
from .smiles_tokenizer import encode_smiles_regex
from .pharmacophore import compute_token_aligned_pharmacophore_features
from .atom_tokens import build_atom_token_selection

class CachedESMC6BStore:
    """Load frozen ESM-C-6B residue states keyed by target identity.

    ``source`` may be a directory containing ``<target>.pt``/``.pkl`` files or
    one torch/pickle mapping from target identity to a residue tensor.  The
    tensors are deliberately loaded on CPU and converted to float16 so the
    language model remains frozen and outside the trainable parameter count.
    """

    def __init__(self, source: str | Path, *, feature_dim: int = 2560) -> None:
        self.source = Path(source).expanduser().resolve()
        self.feature_dim = int(feature_dim)
        if not self.source.exists() or self.feature_dim <= 0:
            raise ValueError("ESM-C embedding source must exist and feature_dim must be positive")
        self._mapping: Mapping[str, Any] | None = None
        if self.source.is_file():
            self._mapping = self._load_file(self.source)
            if not isinstance(self._mapping, Mapping):
                raise RuntimeError("single ESM-C cache file must be a target-to-tensor mapping")

    @staticmethod
    def _load_file(path: Path) -> Any:
        if path.suffix.lower() in {".pt", ".pth"}:
            return torch.load(path, map_location="cpu", weights_only=False)
        with path.open("rb") as handle:
            return pickle.load(handle)

    def _raw(self, target: str) -> Any:
        if self._mapping is not None:
            if target not in self._mapping:
                raise RuntimeError(f"ESM-C cache misses target {target!r}")
            return self._mapping[target]
        for suffix in (".pt", ".pth", ".pkl", ".pickle"):
            candidate = self.source / f"{target}{suffix}"
            if candidate.exists():
                return self._load_file(candidate)
        raise RuntimeError(f"ESM-C cache misses target file for {target!r}")

    def residue_states(self, target: str, *, sequence_length: int, max_length: int) -> np.ndarray:
        value = self._raw(target)
        if isinstance(value, Mapping):
            for key in ("representations", "embedding", "embeddings", "residue_embeddings"):
                if key in value:
                    value = value[key]
                    break
        tensor = torch.as_tensor(value).detach().cpu()
        if tensor.ndim != 2 or tensor.size(1) != self.feature_dim:
            raise RuntimeError(
                f"ESM-C cache for {target!r} must be [residues, {self.feature_dim}]; got {tuple(tensor.shape)}"
            )
        # Accept conventional BOS/EOS framing, but never silently crop any
        # other length mismatch.
        if tensor.size(0) == sequence_length + 2:
            tensor = tensor[1:-1]
        if tensor.size(0) != sequence_length:
            raise RuntimeError(
                f"ESM-C cache length mismatch for {target!r}: cache={tensor.size(0)}, sequence={sequence_length}"
            )
        result = np.zeros((max_length, self.feature_dim), dtype=np.float16)
        kept = min(sequence_length, max_length)
        result[:kept] = tensor[:kept].to(dtype=torch.float16).numpy()
        return result


class AffinityDataset(Dataset):
    def __init__(self, csv_path, *, dataset_name, pocket_contract, component_sequences,
                 vocabulary, esmc6b_embeddings, max_seq_len=128, max_protein_seq_len=1000,
                 esmc6b_embedding_dim=2560):
        self.csv_path = Path(csv_path).resolve()
        self.manifest, self.split = _clean_manifest_for_csv(self.csv_path)
        self.contract = _manifest_regression_contract(self.manifest,
            expected_dataset_name=dataset_name, path=self.csv_path)
        frame = pd.read_csv(self.csv_path, float_precision='round_trip', low_memory=False)
        _validate_frame_regression_contract(frame, self.csv_path, self.contract)
        _validate_frame_lipinski_contract(frame, self.csv_path)
        if self.split == 'train':
            _validate_train_label_statistics(frame, manifest=self.manifest,
                path=self.csv_path, contract=self.contract)
        self.pocket_contract = load_target_pocket_contract(pocket_contract)
        if not self.pocket_contract.requires_sequence_alignment:
            raise ValueError('the paper requires a sequence-aligned v4 pocket contract')
        self.target_column = self.pocket_contract.identity_column
        if frame.duplicated(['canonical_smiles', self.target_column]).any():
            raise ValueError('each curated split must contain unique ligand-target pairs')
        if 'SMILES' not in frame:
            frame['SMILES'] = frame['canonical_smiles']
        declared = self.manifest['configuration'].get('max_seq_len_including_bos_eos')
        if declared is not None and int(declared) != max_seq_len:
            raise ValueError('SMILES length differs from the curated manifest')
        missing = set(frame[self.target_column]) - set(self.pocket_contract.graph_paths)
        if missing:
            raise ValueError(f'pocket contract is missing {len(missing)} targets')
        keep = frame[self.target_column].isin(self.pocket_contract.available_targets)
        self.excluded_unavailable_pockets = int((~keep).sum())
        self.df = frame.loc[keep].reset_index(drop=True)
        if self.df.empty:
            raise ValueError('no pairs with valid pockets remain')
        self.vocabulary = vocabulary
        self.max_seq_len, self.max_protein_seq_len = max_seq_len, max_protein_seq_len
        self.embeddings = CachedESMC6BStore(esmc6b_embeddings, feature_dim=esmc6b_embedding_dim)
        self.sequences = {}
        columns = [self.target_column] + [key for key in (
            'protein_sequence', 'protein_component_ids', 'protein_component_count',
            'protein_total_sequence_length') if key in self.df]
        for row in self.df[columns].drop_duplicates().to_dict('records'):
            seq, _ = resolve_row_protein_sequence(row, dataset_name=dataset_name,
                                                 component_sequences=component_sequences)
            target = str(row[self.target_column])
            if target in self.sequences and self.sequences[target] != seq:
                raise ValueError(f'conflicting sequences for {target}')
            self.sequences[target] = seq
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.df)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'] = OrderedDict()
        return state

    def __getitem__(self, index):
        row = self.df.iloc[index]
        target = str(row[self.target_column])
        smiles = str(row.canonical_smiles)
        sequence = self.sequences[target]
        if target not in self._cache:
            graph = load_validated_target_pocket_graph(self.pocket_contract, target,
                                                       expected_sequence=sequence)
            embeddings = self.embeddings.residue_states(target, sequence_length=len(sequence),
                                                        max_length=self.max_protein_seq_len)
            self._cache[target] = (graph, embeddings)
            if len(self._cache) > 16:
                self._cache.popitem(last=False)
        self._cache.move_to_end(target)
        graph, embeddings = self._cache[target]
        ids, _, _, unknown = encode_smiles_regex(smiles, vocabulary=self.vocabulary,
                                                max_length=self.max_seq_len)
        if unknown:
            raise ValueError(f'unknown SMILES tokens at row {index}: {smiles}')
        return dict(molecule_tokens=torch.from_numpy(ids.astype(np.int64)),
            pharmacophore_features=torch.from_numpy(compute_token_aligned_pharmacophore_features(
                smiles, max_length=self.max_seq_len)),
            protein_tokens=torch.from_numpy(encode_protein_sequence(sequence, max_length=self.max_protein_seq_len)),
            protein_embeddings=torch.from_numpy(embeddings), pocket=graph.clone(),
            atom_tokens=build_atom_token_selection(smiles, max_length=self.max_seq_len),
            affinity=torch.tensor(float(row[self.contract.label_column]), dtype=torch.float32),
            row_index=int(index), smiles=smiles, target=target)


def collate_affinity(rows):
    if not rows:
        raise ValueError('empty batch')
    tensors = ('molecule_tokens', 'pharmacophore_features', 'protein_tokens',
               'protein_embeddings', 'affinity')
    result = {key: torch.stack([row[key] for row in rows]) for key in tensors}
    result.update({key: Batch.from_data_list([row[key] for row in rows]) for key in ('pocket', 'atom_tokens')})
    result.update({key: [row[key] for row in rows] for key in ('row_index', 'smiles', 'target')})
    return result


def move_affinity_batch(batch, device):
    return {key: value.to(device) if hasattr(value, 'to') else value for key, value in batch.items()}


def forward_batch(model, batch, *, interpret=False):
    return model(batch['molecule_tokens'], pocket_batch=batch['pocket'],
        protein_tokens=batch['protein_tokens'], protein_embeddings=batch['protein_embeddings'],
        pharmacophore_features=batch['pharmacophore_features'], atom_tokens=batch['atom_tokens'],
        return_interpretability=interpret)
