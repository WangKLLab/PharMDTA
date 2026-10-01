"""Train the manuscript model on immutable BindingDB/KIBA pair splits."""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import PharMDTA
from .data import AffinityDataset, collate_affinity, move_affinity_batch, forward_batch
from .data_contracts import validate_native_v6_training_input
from .metrics import compute_native_affinity_metrics
from .protein_sequence import load_component_sequences
from .smiles_tokenizer import load_smiles_vocabulary, SMILES_VOCAB_PATH

CHECKPOINT_FORMAT = 'PharMDTA.paper.v1'


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['dataset'] not in ('bindingdb', 'kiba'):
        raise ValueError('only BindingDB and KIBA are used in this manuscript')
    if cfg['training']['optimizer'] != 'AdamW' or cfg['training']['loss'] != 'standardized_mse':
        raise ValueError('the manuscript uses AdamW and standardized-label MSE')
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--pocket-contract', type=Path)
    p.add_argument('--component-sequences', type=Path)
    p.add_argument('--esmc6b-embeddings', type=Path)
    p.add_argument('--output-dir', type=Path, default=Path('runs'))
    p.add_argument('--run-name', default='')
    p.add_argument('--device', default='cuda')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--micro-batch-size', type=int, default=16)
    p.add_argument('--resume', type=Path)
    p.add_argument('--dry-run', action='store_true', help='print resolved configuration without opening data')
    return p.parse_args()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def identity_collate(rows):
    # Micro-batch collation prevents allocation of an entire logical batch of
    # [1000, 2560] cached residue tensors on the accelerator.
    return rows


def make_loader(dataset, batch_size, workers=0, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, collate_fn=identity_collate, generator=generator)


def micro_batches(rows, size, device):
    for start in range(0, len(rows), size):
        part = rows[start:start+size]
        yield move_affinity_batch(collate_affinity(part), device), len(part)


def train_epoch(model, loader, optimizer, scheduler, scaler, *, device, mean, std, micro_batch_size, amp):
    model.train()
    numerator, count = 0.0, 0
    for rows in tqdm(loader, desc='Train', leave=False):
        optimizer.zero_grad(set_to_none=True)
        for batch, size in micro_batches(rows, micro_batch_size, device):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                prediction, _ = forward_batch(model, batch)
                loss = ((prediction.float() - (batch['affinity']-mean)/std)**2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite training loss')
            scaler.scale(loss * size/len(rows)).backward()
            numerator += float(loss.detach())*size
            count += size
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
    return numerator/count


@torch.inference_mode()
def evaluate(model, loader, *, device, mean, std, micro_batch_size, amp):
    model.eval()
    records = []
    for rows in tqdm(loader, desc='Evaluate', leave=False):
        for batch, _ in micro_batches(rows, micro_batch_size, device):
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                prediction, _ = forward_batch(model, batch)
            for index, smiles, target, label, pred in zip(batch['row_index'], batch['smiles'],
                    batch['target'], batch['affinity'].cpu().tolist(), prediction.float().cpu().tolist()):
                records.append(dict(row_index=index, canonical_smiles=smiles, target=target,
                    label_native=label, prediction_normalized=pred, prediction_native=pred*std+mean))
    frame = pd.DataFrame(records)
    metrics = compute_native_affinity_metrics(frame.label_native, frame.prediction_normalized, mean, std)
    metrics['standardized_MSE'] = metrics['MSE']/(std**2)
    return metrics, frame


def make_dataset(path, cfg, args, components, vocabulary):
    return AffinityDataset(path, dataset_name=cfg['dataset'], pocket_contract=args.pocket_contract,
        component_sequences=components, vocabulary=vocabulary, esmc6b_embeddings=args.esmc6b_embeddings,
        max_seq_len=cfg['data']['max_seq_len'], max_protein_seq_len=cfg['data']['max_protein_seq_len'],
        esmc6b_embedding_dim=cfg['model']['protein_cached_embedding_dim'])


def main():
    args = parse_args()
    cfg = read_config(args.config)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    for key in ('data_dir', 'pocket_contract', 'component_sequences', 'esmc6b_embeddings'):
        if getattr(args, key) is None:
            raise ValueError(f'--{key.replace("_", "-")} is required')
    settings = cfg['training']
    if min(args.micro_batch_size, settings['batch_size'], settings['epochs'], settings['patience']) < 1:
        raise ValueError('batch sizes, epochs and patience must be positive')
    seed = settings['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for smoke tests')
    amp = bool(settings['amp']) and device.type == 'cuda'
    run = (args.output_dir/(args.run_name or f"{cfg['dataset']}_seed{seed}")).resolve()
    if args.resume is None and run.exists() and any(run.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty run: {run}')
    if args.resume is not None and args.resume.resolve().parent != run:
        raise ValueError('--resume must refer to a checkpoint in the selected run directory')
    split_paths = {key: args.data_dir/'splits'/f'{key}.csv' for key in ('train', 'val')}
    receipt = validate_native_v6_training_input(data_dir=args.data_dir,
        dataset_name=cfg['dataset'], split_paths=split_paths)
    components, _ = load_component_sequences(args.component_sequences)
    vocabulary = load_smiles_vocabulary(SMILES_VOCAB_PATH)
    datasets = {key: make_dataset(path, cfg, args, components, vocabulary) for key, path in split_paths.items()}
    train_pairs = set(zip(datasets['train'].df.canonical_smiles, datasets['train'].df[datasets['train'].target_column]))
    val_pairs = set(zip(datasets['val'].df.canonical_smiles, datasets['val'].df[datasets['val'].target_column]))
    if train_pairs & val_pairs:
        raise ValueError('train/validation pair leakage')
    labels = datasets['train'].df[datasets['train'].contract.label_column].to_numpy(dtype=np.float64)
    mean, std = float(labels.mean()), max(float(labels.std(ddof=0)), 1e-8)
    model_kwargs = dict(cfg['model'], vocab_size=len(vocabulary), pad_id=vocabulary['<pad>'])
    if datasets['train'].pocket_contract.feature_dim != model_kwargs['pocket_in_dim']:
        raise ValueError('pocket input dimension differs from configuration')
    model = PharMDTA(**model_kwargs).to(device)
    model.grad_clip = float(settings['grad_clip'])
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
                   weight_decay=settings['weight_decay']),
              dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim < 2], weight_decay=0.0)]
    optimizer = torch.optim.AdamW(groups, lr=settings['lr'], betas=tuple(settings['betas']))
    generator = torch.Generator().manual_seed(seed+10000)
    loader = make_loader(datasets['train'], settings['batch_size'], args.num_workers, True, generator)
    val_loader = make_loader(datasets['val'], settings['batch_size'], args.num_workers)
    total_steps, warmup = len(loader)*settings['epochs'], settings['warmup_steps']
    def schedule(step):
        if step < warmup:
            return max(1e-8, (step+1)/max(1, warmup))
        progress = min(1.0, (step-warmup)/max(1, total_steps-warmup))
        return 0.5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best, bad, best_epoch, history = 1, math.inf, 0, 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if checkpoint['format'] != CHECKPOINT_FORMAT:
            raise ValueError('resume checkpoint format differs')
        previous_run = json.loads((run/'config.json').read_text())
        if previous_run['config'] != cfg or previous_run['model_kwargs'] != model_kwargs:
            raise ValueError('resume configuration differs')
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start, best, bad, best_epoch = checkpoint['epoch']+1, checkpoint['best'], checkpoint['bad'], checkpoint['best_epoch']
        history = checkpoint['history']
        random.setstate(checkpoint['rng_python']); np.random.set_state(checkpoint['rng_numpy'])
        torch.set_rng_state(checkpoint['rng_torch'])
        generator.set_state(checkpoint['rng_loader'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    run.mkdir(parents=True, exist_ok=True)
    (run/'predictions').mkdir(exist_ok=True)
    atomic_json(run/'config.json', dict(config=cfg, model_kwargs=model_kwargs,
        paths={k: str(getattr(args, k).resolve()) for k in ('data_dir', 'pocket_contract', 'component_sequences', 'esmc6b_embeddings')},
        train_label_statistics=dict(mean=mean, std=std),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        excluded_unavailable_pockets={key: ds.excluded_unavailable_pockets for key, ds in datasets.items()}))
    atomic_json(run/'data_contract.json', receipt)
    eval_options = dict(device=device, mean=mean, std=std, micro_batch_size=args.micro_batch_size, amp=amp)
    for epoch in range(start, settings['epochs']+1):
        if bad >= settings['patience']:
            break
        train_loss = train_epoch(model, loader, optimizer, scheduler, scaler, **eval_options)
        metrics, _ = evaluate(model, val_loader, **eval_options)
        value = metrics['standardized_MSE']
        improved = value < best-settings['min_delta']
        if improved:
            best, bad, best_epoch = value, 0, epoch
        else:
            bad += 1
        history.append(dict(epoch=epoch, train_standardized_MSE=train_loss,
            **{f'val_{key}': val for key, val in metrics.items()}, lr=optimizer.param_groups[0]['lr']))
        pd.DataFrame(history).to_csv(run/'history.csv', index=False)
        checkpoint = dict(format=CHECKPOINT_FORMAT, epoch=epoch,
            model_kwargs=model_kwargs, model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), best=best, bad=bad,
            best_epoch=best_epoch, history=history, train_mean=mean, train_std=std,
            rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch=torch.get_rng_state(),
            rng_loader=generator.get_state(), rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        atomic_checkpoint(run/'last.pt', checkpoint)
        if improved:
            atomic_checkpoint(run/'best.pt', checkpoint)
        print(f'epoch={epoch} train_MSE_z={train_loss:.6f} val_MSE_z={value:.6f} best_epoch={best_epoch}', flush=True)
    checkpoint = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    final = dict(best_epoch=best_epoch, metric_scale='native')
    final['val'], predictions = evaluate(model, val_loader, **eval_options)
    predictions.to_csv(run/'predictions/val_best.csv', index=False)
    split_paths['test'] = args.data_dir/'splits/test.csv'
    validate_native_v6_training_input(data_dir=args.data_dir, dataset_name=cfg['dataset'], split_paths=split_paths)
    test = make_dataset(split_paths['test'], cfg, args, components, vocabulary)
    test_pairs = set(zip(test.df.canonical_smiles, test.df[test.target_column]))
    if test_pairs & (train_pairs | val_pairs):
        raise ValueError('held-out test contains train/validation pairs')
    final['test'], predictions = evaluate(model, make_loader(test, settings['batch_size'], args.num_workers), **eval_options)
    predictions.to_csv(run/'predictions/test_best.csv', index=False)
    atomic_json(run/'final_metrics.json', final)
    atomic_json(run/'training_stop.json', dict(best_epoch=best_epoch, completed_epochs=len(history),
        early_stopped=bad >= settings['patience']))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
