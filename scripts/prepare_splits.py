#!/usr/bin/env python3
"""Prepare original pair or drug-wise splits in a separate dataset directory."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model.dataset_splits import prepare_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Curated source dataset or drug-wise dataset with original backups')
    parser.add_argument('--output-dir', type=Path, required=True, help='New dataset directory; must not already exist')
    parser.add_argument('--method', choices=('pair', 'drug-wise'), required=True)
    parser.add_argument('--seed', type=int, default=42, help='Drug-wise assignment seed; pair mode preserves the existing assignment')
    parser.add_argument('--source-manifest', type=Path, help='Original pair manifest for a legacy backup')
    args = parser.parse_args()
    output = prepare_splits(args.data_dir, args.output_dir, args.method, args.seed, args.source_manifest)
    print(output)


if __name__ == '__main__':
    main()
