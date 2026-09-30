"""Ligand Transformer and ligand-independent pocket Transformer (II-B/E/F)."""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import Tensor, nn
from .encoder_blocks import EncoderBlock
from .pocket_encoder import PocketEncoder


def masked_mean(states, valid):
    weight = valid.to(states.dtype).unsqueeze(-1)
    return (states * weight).sum(1) / weight.sum(1).clamp_min(1)


@dataclass
class ConditionEncoding:
    pocket_states: Tensor
    pocket_key_padding_mask: Tensor
    pocket_vector: Tensor
    pocket_sequence_positions: Tensor


class ConditionEncoder(nn.Module):
    def __init__(self, *, dim, n_head, n_layer, dropout, pocket_in_dim,
                 pocket_node_dim, pocket_input_proj_dim, pocket_hidden_dim,
                 pocket_layers, pocket_residue_feature_dim):
        super().__init__()
        self.pocket_type = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.pocket_type, std=0.02)
        self.pocket_encoder = PocketEncoder(
            in_node_nf=pocket_in_dim, hidden_nf=pocket_hidden_dim,
            out_node_nf=pocket_node_dim, n_layers=pocket_layers,
            graph_out_dim=dim, node_proj_dim=pocket_input_proj_dim,
            residue_feature_dim=pocket_residue_feature_dim)
        layer = nn.TransformerEncoderLayer(dim, n_head, dim_feedforward=4*dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, *, pocket_batch):
        vector, nodes, valid, positions = self.pocket_encoder(pocket_batch)
        states = torch.cat((vector.unsqueeze(1), nodes), 1) + self.pocket_type
        valid = torch.cat((torch.ones(len(nodes), 1, dtype=torch.bool, device=nodes.device), valid), 1)
        states = self.out_norm(self.encoder(states, src_key_padding_mask=~valid))
        return ConditionEncoding(states, ~valid, masked_mean(states, valid), positions)


class MoleculeEncoder(nn.Module):
    """Bidirectional encoder over SMILES and token-aligned pharmacophores.

    Atom-level RDKit pharmacophore/context features are aligned with the same
    fixed-regex SMILES token positions. Non-atom tokens, BOS/EOS, and padding
    carry zero pharmacophore vectors, so ligand self-attention and subsequent
    ligand--pocket cross-attention retain positional chemical correspondence.
    """

    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        dim: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        pharmacophore_feature_dim: int = 12,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        token_dim = int(self.token_embedding.embedding_dim)
        self.pharmacophore_feature_dim = int(pharmacophore_feature_dim)
        if self.pharmacophore_feature_dim <= 0:
            raise ValueError("pharmacophore_feature_dim must be positive")
        self.token_projection = nn.Linear(token_dim, int(dim))
        self.token_layer_norm = nn.LayerNorm(int(dim))
        self.pharmacophore_projection = nn.Linear(
            self.pharmacophore_feature_dim, int(dim), bias=False
        )
        self.pharmacophore_layer_norm = nn.LayerNorm(int(dim), elementwise_affine=False)
        self.pharmacophore_scale = nn.Parameter(torch.tensor(1.0))
        cfg = SimpleNamespace(
            n_embd=dim,
            n_head=n_head,
            attn_pdrop=dropout,
            resid_pdrop=dropout,
            use_rope=True,
        )
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(dim)
        with torch.no_grad():
            if token_dim == int(dim):
                nn.init.eye_(self.token_projection.weight)
            else:
                nn.init.xavier_uniform_(self.token_projection.weight)
            nn.init.zeros_(self.token_projection.bias)
            nn.init.xavier_uniform_(self.pharmacophore_projection.weight)

    def forward(
        self,
        idx: Tensor,
        *,
        pharmacophore_features: Tensor,
        pad_id: int = 0,
    ) -> tuple[Tensor, Tensor]:
        if idx.ndim != 2:
            raise ValueError("SMILES token IDs must have shape [batch, tokens]")
        if not torch.is_floating_point(pharmacophore_features):
            raise TypeError(
                "pharmacophore_features must be a floating point "
                "[batch, tokens, features] tensor"
            )
        expected_shape = (
            int(idx.size(0)),
            int(idx.size(1)),
            self.pharmacophore_feature_dim,
        )
        if tuple(pharmacophore_features.shape) != expected_shape:
            raise ValueError(
                "pharmacophore feature shape mismatch: "
                f"expected={expected_shape}, got={tuple(pharmacophore_features.shape)}"
            )
        if not bool(torch.isfinite(pharmacophore_features).all()):
            raise ValueError("pharmacophore_features contains a non-finite value")
        valid = idx.ne(int(pad_id))
        if not bool(valid.any(dim=1).all()):
            raise RuntimeError("MoleculeEncoder received an all-pad molecule sequence")
        token_states = self.token_embedding(idx)
        token_states = self.token_layer_norm(self.token_projection(token_states))
        pharmacophore_state = self.pharmacophore_layer_norm(
            self.pharmacophore_projection(
                pharmacophore_features.to(dtype=token_states.dtype)
            )
        )
        x = token_states + self.pharmacophore_scale * pharmacophore_state
        x = self.drop(x)
        key_padding_mask = ~valid
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x, masked_mean(x, valid)
