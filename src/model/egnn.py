"""Residue EGNN: invariant messages and equivariant coordinate updates (II-F)."""
import torch
from torch import nn


def segment_sum(values, indices, size):
    result = values.new_zeros(size, values.shape[-1])
    return result.index_add(0, indices, values)


class EquivariantLayer(nn.Module):
    def __init__(self, width, update_coordinates=True):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2 * width + 1, width), nn.SiLU(),
                                      nn.Linear(width, width), nn.SiLU())
        self.node_mlp = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(),
                                      nn.Linear(width, width))
        self.coord_mlp = None
        if update_coordinates:
            projection = nn.Linear(width, 1, bias=False)
            nn.init.xavier_uniform_(projection.weight, gain=0.001)
            self.coord_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                           projection, nn.Tanh())

    def forward(self, h, positions, edges):
        row, col = edges
        displacement = positions[row] - positions[col]
        radial = displacement.square().sum(-1, keepdim=True)
        messages = self.edge_mlp(torch.cat((h[row], h[col], radial), dim=-1))
        if self.coord_mlp is not None:
            direction = displacement / (radial.sqrt().detach() + 1e-8)
            update = segment_sum(direction * self.coord_mlp(messages), row, len(h))
            count = segment_sum(torch.ones_like(radial), row, len(h)).clamp_min(1)
            positions = positions + update / count
        h = h + self.node_mlp(torch.cat((h, segment_sum(messages, row, len(h))), -1))
        return h, positions


class EGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super().__init__()
        if layers < 1:
            raise ValueError("EGNN must have at least one layer")
        self.embedding_in = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            EquivariantLayer(hidden_dim, update_coordinates=i < layers - 1)
            for i in range(layers)
        )
        self.embedding_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, h, positions, edges):
        h = self.embedding_in(h)
        for layer in self.layers:
            h, positions = layer(h, positions, edges)
        # Final coordinates are not consumed by the regressor, so the last
        # coordinate-update MLP is not allocated.
        return self.embedding_out(h)
