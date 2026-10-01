#!/usr/bin/env python3
"""Repartition all curated pairs by unique canonical ligand, preserving originals."""
import argparse
import csv
import json
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
    (data_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    stage.rmdir()
    print(json.dumps({'dataset': data_dir.name, 'outputs': manifest['outputs'], 'ligand_overlap': overlaps}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    repartition(args.data_dir.resolve(), args.source_manifest.resolve(), args.seed)
