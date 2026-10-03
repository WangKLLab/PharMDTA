"""Split semantics, source preservation, and portable original-split recovery."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
from model.dataset_splits import prepare_splits
from model.data_contracts import FIXED_SPLIT_PROTOCOL


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        (self.source / 'splits').mkdir(parents=True)
        rows = [(f'drug{i}', f'target{j}', float(i + j)) for j in range(3) for i in range(12)]
        manifest = dict(configuration=dict(dataset='kiba', label_column='kiba_score_raw',
                         split_protocol=FIXED_SPLIT_PROTOCOL), outputs={})
        for split, group in zip(('train', 'val', 'test'), (rows[:20], rows[20:28], rows[28:])):
            path = self.source / 'splits' / f'{split}.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['canonical_smiles', 'protein_identity_key', 'kiba_score_raw', 'contract_split'])
                writer.writerows((drug, target, label, split) for drug, target, label in group)
            manifest['outputs'][split] = dict(path=f'splits/{split}.csv', rows=len(group), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        y = np.array([label for _, _, label in rows[:20]])
        manifest['train_label_statistics'] = dict(fit_split='train', fit_scope='final_unique_train_pairs_only',
            label_column='kiba_score_raw', label_kind='kiba_raw', unit='KIBA_score', transform='identity',
            validation_or_test_used=False, epsilon=1e-8, count=len(y), mean=float(y.mean()),
            std_population=float(y.std()), scale=float(y.std()), min=float(y.min()), max=float(y.max()))
        manifest.update(contract_format='rl_mtl.minimal_joint_clean_splits', contract_version=4)
        (self.source / 'manifest.json').write_text(json.dumps(manifest))

    def fingerprints(self, directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob('*') if p.is_file()}

    def read_pairs(self, directory):
        groups, labels = {}, {}
        for split in ('train', 'val', 'test'):
            with (directory / 'splits' / f'{split}.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(r['contract_split'] == split for r in rows))
            groups[split] = {r['canonical_smiles'] for r in rows}
            for row in rows:
                pair = (row['canonical_smiles'], row['protein_identity_key'])
                self.assertNotIn(pair, labels)
                labels[pair] = row['kiba_score_raw']
        return groups, labels

    def test_pair_preserves_bytes_and_source(self):
        before = self.fingerprints(self.source)
        output = prepare_splits(self.source, self.root / 'pair', 'pair')
        for split in ('train', 'val', 'test'):
            self.assertEqual((output / 'splits' / f'{split}.csv').read_bytes(),
                             (self.source / 'splits' / f'{split}.csv').read_bytes())
        self.assertEqual(before, self.fingerprints(self.source))
        groups, _ = self.read_pairs(output)
        self.assertTrue(groups['train'] & groups['test'])

    def test_drug_wise_is_disjoint_deterministic_and_preserves_labels(self):
        before = self.fingerprints(self.source)
        a = prepare_splits(self.source, self.root / 'cold_a', 'drug-wise')
        b = prepare_splits(self.source, self.root / 'cold_b', 'drug-wise')
        groups, labels = self.read_pairs(a)
        self.assertFalse(groups['train'] & groups['val'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['val'] & groups['test'])
        self.assertEqual(labels, self.read_pairs(self.source)[1])
        for split in ('train', 'val', 'test'):
            self.assertEqual((a / 'splits' / f'{split}.csv').read_bytes(),
                             (b / 'splits' / f'{split}.csv').read_bytes())
        self.assertEqual(before, self.fingerprints(self.source))
        m = json.loads((a / 'manifest.json').read_text())
        with (a / 'splits/train.csv').open() as handle:
            y = np.array([float(r['kiba_score_raw']) for r in csv.DictReader(handle)])
        self.assertAlmostEqual(m['train_label_statistics']['mean'], float(y.mean()), places=12)
        self.assertAlmostEqual(m['train_label_statistics']['std_population'], float(y.std()), places=12)

    def test_original_recovery_survives_relocation_and_source_removal(self):
        original = {s: (self.source / 'splits' / f'{s}.csv').read_bytes() for s in ('train', 'val', 'test')}
        cold = prepare_splits(self.source, self.root / 'cold', 'drug-wise')
        shutil.rmtree(self.source)
        moved = self.root / 'moved'
        cold.rename(moved)
        restored = prepare_splits(moved, self.root / 'restored', 'pair')
        for split, content in original.items():
            self.assertEqual(content, (restored / 'splits' / f'{split}.csv').read_bytes())

    def test_bindingdb_label_column_and_statistics(self):
        manifest = json.loads((self.source / 'manifest.json').read_text())
        manifest['configuration'].update(dataset='bindingdb', label_column='pAffinity',
            split_protocol='bindingdb_full_ic50_random_pair_64_16_20_seed42_no_cv')
        manifest['train_label_statistics'].update(label_column='pAffinity', label_kind='physical_paffinity',
            unit='pIC50', transform='9-log10(IC50[nM])')
        for split in ('train', 'val', 'test'):
            path = self.source / 'splits' / f'{split}.csv'
            path.write_text(path.read_text().replace('kiba_score_raw', 'pAffinity'))
            manifest['outputs'][split]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.source / 'manifest.json').write_text(json.dumps(manifest))
        for method in ('pair', 'drug-wise'):
            output = prepare_splits(self.source, self.root / method, method)
            result = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(result['configuration']['label_column'], 'pAffinity')
            self.assertEqual(result['train_label_statistics']['label_column'], 'pAffinity')

    def test_cross_split_duplicate_is_rejected_even_with_matching_hash(self):
        train = (self.source / 'splits/train.csv').read_text().splitlines()
        path = self.source / 'splits/test.csv'
        with path.open('a') as handle:
            handle.write(train[1].replace(',train', ',test') + '\n')
        manifest = json.loads((self.source / 'manifest.json').read_text())
        manifest['outputs']['test'].update(rows=9, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (self.source / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare_splits(self.source, self.root / 'duplicate', 'pair')

    def test_existing_output_and_nested_output_are_rejected(self):
        before = self.fingerprints(self.source)
        with self.assertRaises(FileExistsError):
            prepare_splits(self.source, self.source, 'pair')
        with self.assertRaises(ValueError):
            prepare_splits(self.source, self.source / 'nested', 'pair')
        self.assertEqual(before, self.fingerprints(self.source))

    def test_corrupted_original_is_rejected_before_output_is_created(self):
        with (self.source / 'splits/test.csv').open('a') as handle:
            handle.write('drug0,target0,0,test\n')
        output = self.root / 'bad'
        with self.assertRaises(ValueError):
            prepare_splits(self.source, output, 'drug-wise')
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
