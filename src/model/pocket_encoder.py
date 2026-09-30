"""Encode one ligand-independent, sequence-aligned pocket per target."""
import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch
from .egnn import EGNN


class PocketEncoder(nn.Module):
    def __init__(self, *, in_node_nf, hidden_nf, out_node_nf, n_layers,
                 graph_out_dim, node_proj_dim, residue_feature_dim):
        super().__init__()
        self.out_node_nf = out_node_nf
        self.input_node_proj = nn.Sequential(
            nn.LayerNorm(in_node_nf), nn.Linear(in_node_nf, node_proj_dim),
            nn.GELU(), nn.LayerNorm(node_proj_dim))
        self.egnn = EGNN(node_proj_dim, hidden_nf, out_node_nf, n_layers)
        combined = node_proj_dim + out_node_nf
        self.graph_proj = nn.Sequential(nn.Linear(combined, graph_out_dim), nn.LayerNorm(graph_out_dim))
        self.node_out_proj = nn.Sequential(nn.Linear(combined, graph_out_dim), nn.LayerNorm(graph_out_dim))
        self.residue_physchem_proj = nn.Linear(residue_feature_dim, graph_out_dim, bias=False)
        nn.init.zeros_(self.residue_physchem_proj.weight)

    def forward(self, graph):
        if graph.x.ndim != 2 or not len(graph.x):
            raise ValueError("pocket graph must contain node features")
        if graph.pos.shape != (len(graph.x), 3):
            raise ValueError("pocket positions must be [nodes, 3]")
        edges = graph.edge_index
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.numel() == 0:
            raise ValueError("precomputed pocket edges must be nonempty [2, edges]")
        if edges.min() < 0 or edges.max() >= len(graph.x):
            raise ValueError("pocket edge index is out of range")
        if not all(torch.isfinite(v).all() for v in (graph.x, graph.pos, graph.residue_physchem)):
            raise ValueError("pocket graph contains non-finite features")
        batch = graph.batch
        if batch is None:
            batch = torch.zeros(len(graph.x), dtype=torch.long, device=graph.x.device)
        if not torch.equal(batch[edges[0]], batch[edges[1]]):
            raise ValueError("pocket edges cannot cross batch items")
        initial = self.input_node_proj(graph.x)
        geometric = self.egnn(initial, graph.pos, edges)
        combined = torch.cat((initial, geometric), -1)
        physchem = self.residue_physchem_proj(graph.residue_physchem.to(initial.dtype))
        vector = self.graph_proj(global_mean_pool(combined, batch)) + global_mean_pool(physchem, batch)
        nodes, valid = to_dense_batch(self.node_out_proj(combined) + physchem, batch)
        positions, _ = to_dense_batch(graph.sequence_position, batch, fill_value=-1)
        return vector, nodes, valid, positions
