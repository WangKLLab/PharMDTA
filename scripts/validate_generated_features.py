"""Validate generated caches through the training loader and CUDA model path."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from model.train import read_config, make_dataset
from model.data import collate_affinity, move_affinity_batch, forward_batch
from model.model import PharMDTA
from model.protein_sequence import load_component_sequences
from model.smiles_tokenizer import load_smiles_vocabulary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,default=Path('data/features'))
    parser.add_argument('--data-root',type=Path,default=Path('data'))
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    components,_=load_component_sequences(args.data_root/'component_sequences.csv')
    vocabulary=load_smiles_vocabulary()
    report={}
    device=torch.device(args.device)
    for name in ('bindingdb','kiba'):
        cfg=read_config(Path('configs')/f'{name}.json')
        paths=SimpleNamespace(pocket_contract=args.features/'pockets/contract.json',
                              esmc6b_embeddings=args.features/'esmc6b')
        model=PharMDTA(**cfg['model'],vocab_size=len(vocabulary),pad_id=vocabulary['<pad>']).to(device).eval()
        report[name]={}
        for split in ('train','val','test'):
            dataset=make_dataset(args.data_root/name/'splits'/f'{split}.csv',cfg,paths,components,vocabulary)
            batch=move_affinity_batch(collate_affinity([dataset[0]]),device)
            with torch.inference_mode(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                prediction,representation=forward_batch(model,batch)
            if not torch.isfinite(prediction).all() or not torch.isfinite(representation).all():
                raise RuntimeError(f'nonfinite model output on {name}/{split}')
            report[name][split]=dict(retained_pairs=len(dataset),excluded_unavailable_pockets=dataset.excluded_unavailable_pockets,
                                     model_forward_finite=True)
            print(name,split,report[name][split],flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    (args.features/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Generated features pass all dataset loaders and model smoke runs',flush=True)


if __name__=='__main__':
    main()
