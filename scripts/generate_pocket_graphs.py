"""Rerun fpocket and DSSP and construct sequence-aligned target pocket graphs."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from torch_geometric.data import Data
from model.pocket_data import (RESIDUE_PHYSCHEM_SCHEMA, TARGET_POCKET_CONTRACT_FORMAT,
    TARGET_POCKET_QUALITY_CONTRACT_SEMANTICS, load_target_pocket_contract,
    load_validated_target_pocket_graph)
from model.pocket_preprocessing import (_independent_chain_map, _write_chain_subset,
    parse_residues, parse_dssp, parse_fpocket_scores, parse_pocket_residue_keys,
    build_graph, _connected_edges, _labels_to_indices, _graph_sequence_identity_ok, _quality)


def build_one(row, output, fpocket, mkdssp, radius, fpocket_timeout):
    torch.set_num_threads(1)
    destination = Path(output) / 'graphs' / f"{row['target']}.pt"
    entry_path = destination.with_suffix('.json')
    if destination.exists() and entry_path.exists():
        return row['target'], json.loads(entry_path.read_text())
    sequence = row['sequence']
    residues = parse_residues(Path(row['structure_path']))
    _, quality = _independent_chain_map(residues, sequence)
    chains = {c for c, (coverage, identity) in quality.items() if coverage >= .60 and identity >= .65}
    graph = None
    reason = 'no_aligned_structure_chain'
    details = {}
    if chains:
        with tempfile.TemporaryDirectory(prefix='PharMDTA_pocket_') as raw:
            temp = Path(raw)
            protein = temp / 'protein_only.pdb'
            _write_chain_subset(Path(row['structure_path']), protein, chains)
            residues = parse_residues(protein)
            mapping, _ = _independent_chain_map(residues, sequence)
            # First detect geometry; ESM-C node inputs are attached after inference completes.
            embedding = np.zeros((len(sequence)+2, 2560), dtype=np.float16)
            dssp_path = temp / 'protein.dssp'
            dssp_run = subprocess.run([mkdssp, str(protein), str(dssp_path)], capture_output=True, text=True, timeout=180)
            dssp = parse_dssp(dssp_path) if dssp_run.returncode == 0 else {}
            pocket_run = subprocess.run([fpocket, '-f', str(protein)], cwd=temp, capture_output=True, text=True, timeout=fpocket_timeout)
            pocket_root = temp / 'protein_only_out'
            reason = 'fpocket_failed'
            if pocket_run.returncode == 0 and pocket_root.is_dir():
                reason = 'no_sequence_mapped_fpocket_candidate'
                scores = parse_fpocket_scores(pocket_root / 'protein_only_info.txt')
                candidates = sorted((pocket_root/'pockets').glob('pocket*_atm.pdb'),
                                    key=lambda p: int(p.name.split('_')[0][6:]))
                candidate_errors = []
                for candidate in candidates:
                    try:
                        candidate_graph = build_graph(residue_keys=parse_pocket_residue_keys(candidate),
                            residues=residues, residue_to_sequence=mapping, embedding=embedding,
                            dssp=dssp, max_sequence_index=len(sequence)-1, edge_distance_a=radius)
                        if candidate_graph.num_nodes < 3:
                            continue
                        candidate_graph.edge_index, bridges, max_bridge = _connected_edges(candidate_graph.pos, radius)
                        candidate_graph.sequence_index = torch.tensor(_labels_to_indices(candidate_graph.residue_labels, mapping), dtype=torch.long)
                        if not _graph_sequence_identity_ok(candidate_graph, sequence):
                            continue
                        rank = int(candidate.name.split('_')[0][6:])
                        graph = candidate_graph
                        details = dict(fpocket_rank=rank, fpocket_score=float(scores.get(rank,0.)),
                                       bridge_edges=bridges, bridge_max_distance_a=max_bridge,
                                       dssp_available=bool(dssp), selected_chains=','.join(sorted(chains)))
                        break
                    except (ValueError, RuntimeError, TypeError) as exc:
                        candidate_errors.append(str(exc))
                if candidate_errors:
                    details['candidate_errors'] = candidate_errors
    available = graph is not None
    if graph is None:
        residue = torch.zeros(1,28); residue[0,'ACDEFGHIKLMNPQRSTVWY'.index(sequence[0])] = 1.
        graph = Data(x=torch.zeros(1,2568), pos=torch.zeros(1,3), edge_index=torch.tensor([[0],[0]]),
                     residue_physchem=residue, sequence_index=torch.zeros(1,dtype=torch.long))
    graph.target_identity = row['target']
    graph.pocket_contract = TARGET_POCKET_QUALITY_CONTRACT_SEMANTICS
    graph.ligand_independent = True
    graph.selection_method = 'fpocket' if available else 'masked_unavailable'
    graph.residue_feature_schema = RESIDUE_PHYSCHEM_SCHEMA['name']
    graph.pocket_quality = _quality(details['fpocket_score'],graph.num_nodes,1.,details['bridge_edges']) if available else torch.zeros(1,4)
    graph.pocket_quality_valid = torch.tensor([[float(available)]])
    temporary = destination.with_suffix('.pt.tmp'); torch.save(graph,temporary); temporary.replace(destination)
    entry = dict(target=row['target'], status='ok', source_group=row['source_group'],
                 structure_path=row['structure_path'], selection_method=graph.selection_method,
                 pocket_available=available, reason='' if available else reason,
                 num_nodes=graph.num_nodes, num_edges=graph.edge_index.shape[1], **details)
    entry_path.write_text(json.dumps(entry,indent=2)+'\n')
    return row['target'],entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--embeddings',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--fpocket',default='/usr/local/bin/fpocket')
    parser.add_argument('--mkdssp',default='/usr/bin/mkdssp')
    parser.add_argument('--radius',type=float,default=8.)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--fpocket-timeout',type=int,default=1800)
    args=parser.parse_args()
    rows=json.loads(args.targets.read_text())
    if args.limit:rows=rows[:args.limit]
    args.output_dir=args.output_dir.resolve()
    (args.output_dir/'graphs').mkdir(parents=True,exist_ok=True)
    entries={}; pending={}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(build_one,row,args.output_dir,args.fpocket,args.mkdssp,args.radius,args.fpocket_timeout):row for row in rows}
        for job in as_completed(jobs):
            row=jobs[job];target,entry=job.result();entries[target]=entry;pending[target]=row
            print(f"geometry {len(entries)}/{len(rows)} available={entry['pocket_available']} {target}",flush=True)
    attached=0
    while pending:
        for target,row in list(pending.items()):
            cache=args.embeddings/f'{target}.pt'
            if not cache.is_file() or not cache.with_suffix('.json').is_file():continue
            metadata=json.loads(cache.with_suffix('.json').read_text())
            if metadata['sequence']!=row['sequence']:raise ValueError('cache sequence mismatch')
            tensor=torch.load(cache,map_location='cpu',weights_only=False)
            if tensor.shape!=(len(row['sequence']),2560) or not torch.isfinite(tensor).all():raise ValueError('invalid cache')
            destination=args.output_dir/'graphs'/f'{target}.pt'
            graph=torch.load(destination,map_location='cpu',weights_only=False)
            if entries[target]['pocket_available']:
                graph.x[:,:2560]=tensor[graph.sequence_index].float()
            temporary=destination.with_suffix('.pt.tmp');torch.save(graph,temporary);temporary.replace(destination)
            entries[target]['embedding_path']=str(cache.resolve())
            del pending[target];attached+=1
            if attached%25==0 or not pending:print(f'features attached {attached}/{len(rows)}',flush=True)
        if pending:time.sleep(5)
    payload=dict(format=TARGET_POCKET_CONTRACT_FORMAT,version=4,semantics=TARGET_POCKET_QUALITY_CONTRACT_SEMANTICS,
        identity_column='protein_identity_key',feature_dim=2568,residue_feature_dim=28,residue_feature_schema=RESIDUE_PHYSCHEM_SCHEMA,
        target_level=True,query_specific=False,ligand_independent=True,ligand_coordinates_used=False,ligand_identity_used=False,
        fpocket_custom_ligand_argument=False,standard_amino_acid_only=True,ligand_residue_records_retained=False,
        apo_conformation_guaranteed=False,protein_only_record_types=['CRYST1','ATOM','TER','END'],
        smiles_fields_read=[],affinity_label_fields_read=[],num_failed_targets=0,num_targets=len(rows),
        num_requested_targets=len(rows),graph_paths={t:f'graphs/{t}.pt' for t in sorted(entries)},entries=entries,
        builder={'name':Path(__file__).name},edge_distance_a=args.radius,
        graph_connectivity_policy='radius_edges_plus_component_mst_bridges')
    path=args.output_dir/'contract.json';path.write_text(json.dumps(payload,indent=2)+'\n')
    contract=load_target_pocket_contract(path)
    for row in rows:
        if row['target'] in contract.available_targets:
            load_validated_target_pocket_graph(contract,row['target'],expected_sequence=row['sequence'])
    summary=dict(targets=len(rows),available=len(contract.available_targets),unavailable=len(rows)-len(contract.available_targets),validated=True)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
