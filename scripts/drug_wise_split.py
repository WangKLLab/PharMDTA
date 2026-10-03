#!/usr/bin/env python3
"""Repartition a local dataset in place, preserving its original pair splits."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from model.dataset_splits import repartition

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    repartition(args.data_dir.resolve(), args.source_manifest.resolve(), args.seed)
