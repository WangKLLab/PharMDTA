"""Prepare exact target sequences and source structures for feature generation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from model.protein_sequence import load_component_sequences, resolve_row_protein_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--component-sequences', type=Path, default=Path('data/component_sequences.csv'))
    parser.add_argument('--source-contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/features/targets.json'))
    args = parser.parse_args()
    components, _ = load_component_sequences(args.component_sequences)
    source = json.loads(args.source_contract.read_text())
    records = {}
    for dataset in ('bindingdb', 'kiba'):
        columns = ['protein_identity_key', 'protein_total_sequence_length'] + (
            ['protein_component_ids', 'protein_component_count'] if dataset == 'bindingdb' else ['protein_sequence'])
        for split in ('train', 'val', 'test'):
            frame = pd.read_csv(args.data_root/dataset/'splits'/f'{split}.csv', usecols=columns)
            for row in frame.drop_duplicates('protein_identity_key').to_dict('records'):
                target = row['protein_identity_key']
                sequence, _ = resolve_row_protein_sequence(row, dataset_name=dataset, component_sequences=components)
                if target in records:
                    if records[target]['sequence'] != sequence:
                        raise ValueError(f'conflicting sequence for {target}')
                    if dataset not in records[target]['datasets']:
                        records[target]['datasets'].append(dataset)
                    continue
                entry = source['entries'][target]
                structure = Path(entry['structure_path'])
                if not structure.is_file():
                    raise FileNotFoundError(structure)
                records[target] = dict(target=target, sequence=sequence, datasets=[dataset],
                                       structure_path=str(structure.resolve()), source_group=entry['source_group'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([records[t] for t in sorted(records)], indent=2)+'\n')
    print(f'Prepared {len(records)} targets')


if __name__ == '__main__':
    main()
