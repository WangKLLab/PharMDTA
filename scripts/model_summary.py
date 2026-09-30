"""Count parameters from the supplied configuration without allocating model storage."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from model.model import PharMacyDTA
from model.train import read_config
from model.smiles_tokenizer import load_smiles_vocabulary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    with torch.device('meta'):
        model = PharMacyDTA(**config['model'], vocab_size=len(load_smiles_vocabulary()))
    print(json.dumps(dict(trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_ESMC_included=False, config=str(args.config)), indent=2))
