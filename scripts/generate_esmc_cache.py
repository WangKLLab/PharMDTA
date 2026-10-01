"""Generate local ESM-C-6B residue caches from the feature target inventory."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from model.esmc_preprocessing import encode_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--chunk-residues', type=int, default=2046)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards or not 1 <= args.chunk_residues <= 2046:
        parser.error('invalid shard or chunk size')
    rows = json.loads(args.targets.read_text())[args.shard::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoConfig, AutoModel, AutoTokenizer, __version__
    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'esmc' or config.d_model != 2560:
        raise ValueError('a local ESM-C-6B checkpoint with dimension 2560 is required')
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
        dtype=dtype, device_map={'': str(device)}, attn_implementation='sdpa',
        low_cpu_mem_usage=True,
    ).eval()
    started = time.monotonic()
    for number, row in enumerate(rows, 1):
        destination = args.output_dir / f"{row['target']}.pt"
        metadata_path = destination.with_suffix('.json')
        if destination.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get('sequence') != row['sequence'] or metadata.get('model_dir') != str(args.model_dir.resolve()):
                raise ValueError(f'existing cache metadata differs: {destination}')
            print(f"existing {number}/{len(rows)} {row['target']}", flush=True)
            continue
        tensor, chunks = encode_request(SimpleNamespace(sequence=row['sequence']), model=model,
                                         tokenizer=tokenizer, max_residues=args.chunk_residues)
        # Runtime accepts residue-only states and converts them to FP16.
        residues = tensor[1:-1].to(dtype=torch.float16).contiguous()
        if residues.shape != (len(row['sequence']), 2560) or not torch.isfinite(residues).all():
            raise RuntimeError('invalid ESM-C residue cache')
        temporary = destination.with_suffix('.pt.tmp')
        torch.save(residues, temporary)
        temporary.replace(destination)
        metadata = dict(target=row['target'], sequence=row['sequence'], shape=list(residues.shape),
                        dtype='float16', model_dir=str(args.model_dir.resolve()),
                        transformers_version=__version__, inference_dtype=str(dtype),
                        chunks=chunks, chunk_policy='full_sequence_nonoverlap', special_tokens_removed=True)
        temporary_metadata = metadata_path.with_suffix('.json.tmp')
        temporary_metadata.write_text(json.dumps(metadata, indent=2)+'\n')
        temporary_metadata.replace(metadata_path)
        print(f"generated {number}/{len(rows)} length={len(row['sequence'])} elapsed={time.monotonic()-started:.1f}s {row['target']}", flush=True)
    print('ESM-C shard complete', flush=True)


if __name__ == '__main__':
    main()
