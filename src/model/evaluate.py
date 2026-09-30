"""Evaluate a selected checkpoint and export prediction-path atom/residue evidence."""
import argparse
import json
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .model import PharMacyDTA
from .data import collate_affinity, move_affinity_batch, forward_batch
from .train import CHECKPOINT_FORMAT, make_dataset, make_loader, evaluate, atomic_json
from .protein_sequence import load_component_sequences
from .smiles_tokenizer import load_smiles_vocabulary, SMILES_VOCAB_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--micro-batch-size', type=int, default=16)
    parser.add_argument('--attention-limit', type=int, default=0,
                        help='export M and top-15 marginals for the first N split rows; 0 disables export')
    for name in ('data-dir', 'pocket-contract', 'component-sequences', 'esmc6b-embeddings'):
        parser.add_argument('--'+name, type=Path, help='override a relocated input path')
    args = parser.parse_args()
    if args.micro_batch_size < 1 or args.attention_limit < 0:
        parser.error('micro-batch size must be positive and attention limit nonnegative')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty output: {args.output_dir}')
    run = json.loads((args.checkpoint.parent/'config.json').read_text())
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['format'] != CHECKPOINT_FORMAT:
        raise ValueError('checkpoint must come from this cleaned paper implementation')
    paths = SimpleNamespace(**{key: getattr(args, key) or Path(value) for key, value in run['paths'].items()})
    components, _ = load_component_sequences(paths.component_sequences)
    vocabulary = load_smiles_vocabulary(SMILES_VOCAB_PATH)
    dataset = make_dataset(paths.data_dir/'splits'/f'{args.split}.csv', run['config'], paths, components, vocabulary)
    device = torch.device(args.device)
    model = PharMacyDTA(**checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    mean, std = checkpoint['train_mean'], checkpoint['train_std']
    amp = bool(run['config']['training']['amp']) and device.type == 'cuda'
    metrics, predictions = evaluate(model, make_loader(dataset, 128), device=device,
        mean=mean, std=std, micro_batch_size=args.micro_batch_size, amp=amp)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir/'predictions.csv', index=False)
    atomic_json(args.output_dir/'metrics.json', metrics)
    with torch.inference_mode():
        for index in range(min(args.attention_limit, len(dataset))):
            row = dataset[index]
            batch = move_affinity_batch(collate_affinity([row]), device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                _, _, evidence = forward_batch(model, batch, interpret=True)
            arrays = {field.name: getattr(evidence, field.name)[0].detach().cpu().numpy()
                      for field in fields(evidence)}
            np.savez_compressed(args.output_dir/f'attention_{index:06d}.npz', **arrays)
            atom_valid, residue_valid = arrays['atom_valid_mask'], arrays['residue_valid_mask']
            atoms = np.flatnonzero(atom_valid)
            residues = np.flatnonzero(residue_valid)
            atoms = atoms[np.argsort(-arrays['atom_importance'][atoms], kind='stable')[:15]]
            residues = residues[np.argsort(-arrays['residue_importance'][residues], kind='stable')[:15]]
            atomic_json(args.output_dir/f'attention_{index:06d}.json', dict(
                row_index=index, canonical_smiles=row['smiles'], target=row['target'],
                index_convention='zero-based canonical atom and full sequence positions',
                atoms=[dict(atom_index=int(arrays['atom_indices'][i]), score=float(arrays['atom_importance'][i])) for i in atoms],
                residues=[dict(sequence_position=int(arrays['residue_sequence_positions'][i]),
                    score=float(arrays['residue_importance'][i])) for i in residues]))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
