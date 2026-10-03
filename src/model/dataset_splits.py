"""Prepare original pair or cold-drug splits without changing curated labels."""
import csv
import json
import copy
import hashlib
import shutil
import tempfile
import math
import random

from contextlib import ExitStack
from pathlib import Path

SPLITS = ('train', 'val', 'test')
PROTOCOL = 'deepdtagen_drug_wise_64_16_20_no_cv'


def repartition(data_dir, source_manifest, seed):
    backup = data_dir / 'original_pair_splits'
    if backup.exists():
        raise RuntimeError(f'Backup already exists: {backup}; refusing to overwrite')
    paths = {s: data_dir / 'splits' / f'{s}.csv' for s in SPLITS}
    ligands, pairs = set(), set()
    inputs = {}
    fields = None
    for split, path in paths.items():
        count = 0
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if fields is None:
                fields = reader.fieldnames
            if reader.fieldnames != fields:
                raise ValueError('Split schemas differ')
            for row in reader:
                ligand = row['canonical_smiles']
                pair = (ligand, row['protein_identity_key'])
                if not ligand or pair in pairs:
                    raise ValueError(f'Missing ligand or duplicate pair: {pair}')
                ligands.add(ligand)
                pairs.add(pair)
                count += 1
        inputs[split] = {'path': f'original_pair_splits/{split}.csv', 'rows': count}
    ordered = sorted(ligands)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = math.ceil(len(ordered) * .2)
    remaining = ordered[n_test:]
    rng.shuffle(remaining)
    n_val = math.ceil(len(remaining) * .2)
    groups = {'test': set(ordered[:n_test]), 'val': set(remaining[:n_val]), 'train': set(remaining[n_val:])}
    if not all(groups.values()):
        raise ValueError('Too few ligands for three nonempty splits')
    assignment = {ligand: split for split, group in groups.items() for ligand in group}
    template = json.loads(source_manifest.read_text())
    label = template['configuration']['label_column']
    stage = data_dir / 'drug_wise_staging'
    stage.mkdir()
    counts = dict.fromkeys(SPLITS, 0)
    targets = {s: set() for s in SPLITS}
    n, mean, m2 = 0, 0., 0.
    minimum, maximum = math.inf, -math.inf
    with ExitStack() as stack:
        writers = {}
        for split in SPLITS:
            writer = csv.DictWriter(stack.enter_context((stage / f'{split}.csv').open('w', newline='')), fieldnames=fields)
            writer.writeheader()
            writers[split] = writer
        for path in paths.values():
            with path.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    split = assignment[row['canonical_smiles']]
                    row['contract_split'] = split
                    writers[split].writerow(row)
                    counts[split] += 1
                    targets[split].add(row['protein_identity_key'])
                    if split == 'train':
                        value = float(row[label])
                        if not math.isfinite(value):
                            raise ValueError('Nonfinite label')
                        n += 1
                        delta = value - mean
                        mean += delta / n
                        m2 += delta * (value - mean)
                        minimum, maximum = min(minimum, value), max(maximum, value)
    assert sum(counts.values()) == len(pairs)
    overlaps = {a + '_' + b: len(groups[a] & groups[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
    assert not any(overlaps.values())
    configuration = template['configuration'].copy()
    configuration.update(split_scope='cold_ligand', split_scope_semantics='canonical_smiles disjoint across all three splits; targets may overlap', split_protocol={'name': PROTOCOL, 'seed': seed, 'fixed_single_split': True, 'cross_validation': False, 'test_used_for_model_selection': False, 'fractions': {'train': .64, 'val': .16, 'test': .2}, 'unit': 'unique_canonical_smiles', 'rounding': 'ceil(0.2*N) test; ceil(0.2*remaining) validation', 'algorithm': 'sorted ligands; Python random.Random shuffle twice'})
    stats = {key: template['train_label_statistics'][key] for key in (
        'fit_split', 'fit_scope', 'label_column', 'label_kind', 'unit',
        'transform', 'validation_or_test_used', 'epsilon',
    )}
    std = math.sqrt(m2 / n)
    stats.update(count=n, mean=mean, std_population=std, scale=max(std, float(stats['epsilon'])), min=minimum, max=maximum)
    manifest = {'contract_format': template['contract_format'], 'contract_version': template['contract_version'], 'configuration': configuration, 'train_label_statistics': stats, 'pair_overlap': overlaps, 'ligand_overlap': overlaps, 'inputs': inputs, 'source_manifest': {'path': str(source_manifest.resolve())}, 'aggregation': {'input_rows': sum(counts.values()), 'retained_rows': sum(counts.values()), 'dropped_rows': 0, 'unique_output_pairs': len(pairs), 'split_reassignment': True}, 'outputs': {s: {'path': f'splits/{s}.csv', 'rows': counts[s], 'unique_ligands': len(groups[s]), 'unique_targets': len(targets[s])} for s in SPLITS}}
    with (stage / 'ligand_assignments.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['canonical_smiles', 'split'])
        writer.writerows((ligand, assignment[ligand]) for ligand in sorted(ligands))
    backup.mkdir()
    for s, path in paths.items():
        path.rename(backup / path.name)
        (stage / path.name).rename(path)
    (stage / 'ligand_assignments.csv').rename(data_dir / 'ligand_assignments.csv')
    original_manifest = copy.deepcopy(template)
    for split in SPLITS:
        original_manifest['outputs'][split]['path'] = f'original_pair_splits/{split}.csv'
    (data_dir / 'original_pair_manifest.json').write_text(json.dumps(original_manifest, indent=2) + '\n')
    manifest['source_manifest'] = {'path': 'original_pair_manifest.json'}
    (data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stage.rmdir()
    print(json.dumps({'dataset': data_dir.name, 'outputs': manifest['outputs'], 'ligand_overlap': overlaps}, indent=2))


def _original_source(data_dir, source_manifest=None):
    from .data_contracts import FIXED_SPLIT_PROTOCOL, BINDINGDB_FULL_IC50_SPLIT_PROTOCOL
    manifest_path = Path(source_manifest) if source_manifest else data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    protocol = manifest['configuration']['split_protocol']
    name = protocol['name'] if isinstance(protocol, dict) else protocol
    if name == PROTOCOL:
        backup = data_dir / 'original_pair_splits'
        reference = manifest.get('source_manifest', {}).get('path')
        if not reference:
            raise ValueError('Drug-wise input lacks its original pair manifest; provide --source-manifest')
        original_manifest = Path(reference)
        if not original_manifest.is_absolute():
            original_manifest = manifest_path.parent / original_manifest
        manifest = json.loads(original_manifest.read_text())
        protocol = manifest['configuration']['split_protocol']
        name = protocol['name'] if isinstance(protocol, dict) else protocol
        paths = {split: backup / f'{split}.csv' for split in SPLITS}
    else:
        # An explicit original manifest can accompany a legacy drug-wise backup.
        backup = data_dir / 'original_pair_splits'
        directory = backup if source_manifest and backup.is_dir() else data_dir / 'splits'
        paths = {split: directory / f'{split}.csv' for split in SPLITS}
    if name not in {FIXED_SPLIT_PROTOCOL, BINDINGDB_FULL_IC50_SPLIT_PROTOCOL}:
        raise ValueError('Pair mode requires an original fixed pair split, not a newly shuffled approximation')
    return manifest, paths


def prepare_splits(data_dir, output_dir, method, seed=42, source_manifest=None):
    """Write a new portable dataset; pair mode preserves original CSV bytes."""
    from .data_contracts import (
        _validate_fixed_split_protocol, _validate_train_label_statistics,
        dataset_regression_contract,
    )
    import pandas as pd
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if method not in {'pair', 'drug-wise'}:
        raise ValueError('method must be pair or drug-wise')
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite: {output_dir}')
    if data_dir == output_dir or data_dir in output_dir.parents:
        raise ValueError('Output must be outside the source dataset directory')
    manifest, paths = _original_source(data_dir, source_manifest)
    _validate_fixed_split_protocol(manifest, data_dir / 'manifest.json')
    contract = dataset_regression_contract(manifest['configuration']['dataset'])
    if manifest['configuration']['label_column'] != contract.label_column:
        raise ValueError('Dataset label column differs from the regression contract')
    pairs = set()
    for split, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f'Missing original {split} split: {path}')
        entry = manifest['outputs'][split]
        expected_hash = entry.get('sha256')
        if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'{split}: original CSV does not match its manifest')
        frame = pd.read_csv(path, float_precision='round_trip', low_memory=False)
        required = {'canonical_smiles', 'protein_identity_key', 'contract_split', contract.label_column}
        if not required.issubset(frame.columns):
            raise ValueError(f'{split}: missing required pair columns')
        if len(frame) != int(entry['rows']) or frame.empty:
            raise ValueError(f'{split}: row count differs from its manifest')
        if frame[list(required)].isna().any().any() or not frame.contract_split.eq(split).all():
            raise ValueError(f'{split}: missing pair values or incorrect contract_split')
        keys = set(zip(frame.canonical_smiles, frame.protein_identity_key))
        if len(keys) != len(frame) or pairs & keys:
            raise ValueError(f'{split}: duplicate drug-target pair within or across splits')
        pairs.update(keys)
        if not frame[contract.label_column].map(lambda x: math.isfinite(float(x))).all():
            raise ValueError(f'{split}: nonfinite affinity label')
        if split == 'train':
            _validate_train_label_statistics(frame, manifest=manifest, path=path, contract=contract)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare_splits_', dir=output_dir.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        (stage / 'splits').mkdir(parents=True)
        portable = copy.deepcopy(manifest)
        for split, path in paths.items():
            shutil.copy2(path, stage / 'splits' / f'{split}.csv')
            portable['outputs'][split]['path'] = f'splits/{split}.csv'
        (stage / 'manifest.json').write_text(json.dumps(portable, indent=2) + '\n')
        if method == 'drug-wise':
            import contextlib
            import io
            # Reuse the legacy assignment algorithm; perform all mutations in staging.
            with contextlib.redirect_stdout(io.StringIO()):
                repartition(stage, stage / 'manifest.json', seed)
        if output_dir.exists():
            raise FileExistsError(f'Refusing to overwrite: {output_dir}')
        stage.rename(output_dir)
    return output_dir
