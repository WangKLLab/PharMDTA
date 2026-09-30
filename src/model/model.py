"""PharMacyDTA: the manuscript's ligand, target, and shared-relation model."""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import torch
from torch import Tensor, nn
from torch_geometric.utils import to_dense_batch
from .encoder_blocks import EncoderBlock
from .encoders import ConditionEncoder, MoleculeEncoder, masked_mean

@dataclass
class MutualAttentionInterpretability:
    pair_attention: Tensor
    atom_to_residue_attention: Tensor
    residue_to_atom_attention: Tensor
    atom_importance: Tensor
    residue_importance: Tensor
    atom_indices: Tensor
    atom_token_positions: Tensor
    residue_sequence_positions: Tensor
    residue_is_pocket: Tensor
    atom_valid_mask: Tensor
    residue_valid_mask: Tensor

class AtomTokenSelector(nn.Module):
    """Gather contextual states at exact SMILES atom-token positions.

    This is index alignment only. It intentionally contains no molecular
    graph, bond edge, coordinate, or topology-derived feature.
    """

    def forward(self, token_states: Tensor, atom_tokens: Any) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        token_position = getattr(atom_tokens, 'token_position', None)
        atom_id = getattr(atom_tokens, 'atom_id', None)
        batch = getattr(atom_tokens, 'batch', None)
        if not torch.is_tensor(token_position) or token_position.ndim != 1:
            raise ValueError('atom_tokens.token_position must be [atoms]')
        if not torch.is_tensor(atom_id) or atom_id.shape != token_position.shape:
            raise ValueError('atom_tokens.atom_id must align with token positions')
        if not torch.is_tensor(batch) or batch.shape != token_position.shape:
            raise ValueError('atom_tokens.batch must assign every atom')
        if bool((token_position < 0).any()) or bool((token_position >= token_states.size(1)).any()):
            raise ValueError('atom token positions are out of range')
        atom_states = token_states[batch.long(), token_position.long()]
        (dense_atoms, atom_valid) = to_dense_batch(atom_states, batch.long())
        (dense_token_positions, positions_valid) = to_dense_batch(token_position.long(), batch.long(), fill_value=-1)
        (dense_atom_indices, indices_valid) = to_dense_batch(atom_id.long(), batch.long(), fill_value=-1)
        if not torch.equal(atom_valid, positions_valid):
            raise RuntimeError('atom states and token positions are misaligned')
        if not torch.equal(atom_valid, indices_valid):
            raise RuntimeError('atom states and atom indices are misaligned')
        return (dense_atoms, atom_valid, dense_atom_indices, dense_token_positions)

class AlignedBidirectionalCoAttention(nn.Module):
    """One shared atom--residue score matrix normalized in both directions.

    Atom queries and residue keys define a single multi-head relation tensor.
    Row- and column-wise normalization of that tensor produces the two directed
    cross-attention distributions.  Consequently, the exported mutual evidence
    is the same evidence that updates both modalities and drives affinity
    pooling; it is not a post-hoc auxiliary branch.
    """

    def __init__(self, *, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if int(dim) <= 0 or int(heads) <= 0 or int(dim) % int(heads) != 0:
            raise ValueError('co-attention dim must be divisible by positive heads')
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.scale = self.head_dim ** (-0.5)
        self.atom_query_norm = nn.LayerNorm(dim)
        self.residue_key_norm = nn.LayerNorm(dim)
        self.atom_query_projection = nn.Linear(dim, dim)
        self.residue_key_projection = nn.Linear(dim, dim)
        self.atom_value_projection = nn.Linear(dim, dim)
        self.residue_value_projection = nn.Linear(dim, dim)
        self.atom_output_projection = nn.Linear(dim, dim)
        self.residue_output_projection = nn.Linear(dim, dim)
        self.atom_gate_norm = nn.LayerNorm(2 * dim)
        self.residue_gate_norm = nn.LayerNorm(2 * dim)
        self.atom_gate = nn.Linear(2 * dim, dim)
        self.residue_gate = nn.Linear(2 * dim, dim)
        self.atom_output_norm = nn.LayerNorm(dim)
        self.residue_output_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.atom_gate.weight)
        nn.init.zeros_(self.residue_gate.weight)
        nn.init.constant_(self.atom_gate.bias, -2.0)
        nn.init.constant_(self.residue_gate.bias, -2.0)

    def _split_heads(self, states: Tensor) -> Tensor:
        (batch, length, _) = states.shape
        return states.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _masked_softmax(scores: Tensor, valid: Tensor, *, dim: int) -> Tensor:
        mask = valid.unsqueeze(1)
        masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked, dim=dim)
        weights = weights.masked_fill(~mask, 0.0)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)

    def forward(self, atom_states: Tensor, atom_valid: Tensor, residue_states: Tensor, residue_valid: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if atom_states.shape[:2] != atom_valid.shape:
            raise ValueError('atom states and mask are misaligned')
        if residue_states.shape[:2] != residue_valid.shape:
            raise ValueError('residue states and mask are misaligned')
        if bool((~atom_valid.any(dim=1)).any()):
            raise ValueError('mutual attention received an atom-empty ligand')
        if bool((~residue_valid.any(dim=1)).any()):
            raise ValueError('co-attention received a residue-empty target')
        atom_base = self.atom_query_norm(atom_states)
        residue_base = self.residue_key_norm(residue_states)
        atom_query = self._split_heads(self.atom_query_projection(atom_base))
        residue_key = self._split_heads(self.residue_key_projection(residue_base))
        relation = torch.matmul(atom_query, residue_key.transpose(-2, -1)) * self.scale
        pair_valid = atom_valid.unsqueeze(2) & residue_valid.unsqueeze(1)
        atom_to_residue_heads = self._masked_softmax(relation, pair_valid, dim=-1)
        residue_to_atom_heads = self._masked_softmax(relation.transpose(-2, -1), pair_valid.transpose(1, 2), dim=-1)
        residue_value = self._split_heads(self.residue_value_projection(residue_base))
        atom_value = self._split_heads(self.atom_value_projection(atom_base))
        atom_context = torch.matmul(atom_to_residue_heads, residue_value)
        residue_context = torch.matmul(residue_to_atom_heads, atom_value)
        atom_context = atom_context.transpose(1, 2).reshape_as(atom_states)
        residue_context = residue_context.transpose(1, 2).reshape_as(residue_states)
        atom_delta = self.atom_output_projection(atom_context)
        residue_delta = self.residue_output_projection(residue_context)
        atom_gate = torch.sigmoid(self.atom_gate(self.atom_gate_norm(torch.cat((atom_base, atom_delta), dim=-1))))
        residue_gate = torch.sigmoid(self.residue_gate(self.residue_gate_norm(torch.cat((residue_base, residue_delta), dim=-1))))
        atom_delta = atom_gate * atom_delta * atom_valid.unsqueeze(-1)
        residue_delta = residue_gate * residue_delta * residue_valid.unsqueeze(-1)
        atom_updated = self.atom_output_norm(atom_states + self.dropout(atom_delta))
        residue_updated = self.residue_output_norm(residue_states + self.dropout(residue_delta))
        atom_to_residue = atom_to_residue_heads.mean(dim=1)
        residue_to_atom = residue_to_atom_heads.mean(dim=1)
        pair_attention = torch.sqrt((atom_to_residue * residue_to_atom.transpose(1, 2)).clamp_min(1e-12))
        pair_attention = pair_attention.masked_fill(~pair_valid, 0.0)
        pair_attention = pair_attention / pair_attention.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        return (atom_updated, residue_updated, pair_attention, atom_to_residue, residue_to_atom)

class TargetSequenceTransformerEncoder(nn.Module):
    """Bidirectional, decoder-free Transformer encoder for protein sequences.

    Protein residues are embedded categorically, projected to the model width,
    and contextualized by full self-attention with rotary positions.  The
    encoder deliberately has no decoder or causal mask.  It can return both
    residue-level states for ligand--target interaction and their masked global
    summary for target fusion.
    """

    def __init__(self, *, vocab_size: int, output_dim: int, cached_embedding_dim: int=2560, heads: int=8, layers: int=4, dropout: float=0.1, pad_id: int=0) -> None:
        super().__init__()
        if int(vocab_size) <= 1 or int(output_dim) < 8:
            raise ValueError('protein sequence encoder dimensions are invalid')
        if int(heads) <= 0 or int(output_dim) % int(heads) != 0:
            raise ValueError('protein Transformer output_dim must be divisible by heads')
        if int(layers) <= 0:
            raise ValueError('protein_transformer_layers must be positive')
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError('protein sequence dropout must lie in [0, 1)')
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.output_dim = int(output_dim)
        self.heads = int(heads)
        self.layers = int(layers)
        self.cached_embedding_dim = int(cached_embedding_dim)
        self.input_projection = nn.Linear(self.cached_embedding_dim, self.output_dim)
        self.input_norm = nn.LayerNorm(self.output_dim)
        self.input_dropout = nn.Dropout(float(dropout))
        config = SimpleNamespace(n_embd=self.output_dim, n_head=self.heads, attn_pdrop=float(dropout), resid_pdrop=float(dropout), use_rope=True)
        self.blocks = nn.ModuleList((EncoderBlock(config) for _ in range(self.layers)))
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(self, tokens: Tensor, *, embeddings: Tensor | None=None, return_states: bool=False) -> Tensor | tuple[Tensor, Tensor]:
        if tokens.ndim != 2:
            raise ValueError('protein_tokens must be [batch, residues]')
        if bool(((tokens < 0) | (tokens >= self.vocab_size)).any()):
            raise ValueError('protein_tokens contains an out-of-vocabulary residue ID')
        valid = tokens.ne(self.pad_id)
        if bool((~valid.any(dim=1)).any()):
            raise ValueError('protein sequence batch contains an all-padding row')
        if embeddings is None or embeddings.ndim != 3:
            raise ValueError('ESM-C embeddings must be [batch, residues, features]')
        if embeddings.shape[:2] != tokens.shape or embeddings.size(-1) != self.cached_embedding_dim:
            raise ValueError('ESM-C embeddings must align with protein tokens')
        input_states = embeddings.to(dtype=self.input_projection.weight.dtype)
        states = self.input_dropout(self.input_norm(self.input_projection(input_states)))
        key_padding_mask = ~valid
        for block in self.blocks:
            states = block(states, key_padding_mask=key_padding_mask)
        states = self.output_norm(states)
        vector = masked_mean(states, valid)
        return (states, vector) if return_states else vector

class PharMacyDTA(nn.Module):
    """Four-vector affinity prediction with the complete manuscript model."""

    def __init__(self, *, vocab_size, n_embd=384, n_head=4, dropout=0.1, affinity_dropout=0.1, encoder_layers=8, molecule_encoder_layers=8, molecule_transformer_heads=8, protein_transformer_layers=4, protein_transformer_heads=8, cross_attention_layers=3, pocket_in_dim=2568, pocket_node_dim=128, pocket_input_proj_dim=384, pocket_hidden_dim=384, pocket_layers=3, pocket_residue_feature_dim=28, pocket_transformer_heads=4, protein_cached_embedding_dim=2560, pad_id=0):
        super().__init__()
        if cross_attention_layers < 1:
            raise ValueError('cross_attention_layers must be positive')
        (self.n_embd, self.pad_id) = (n_embd, pad_id)
        self.molecule_encoder = MoleculeEncoder(token_embedding=nn.Embedding(vocab_size, n_embd, padding_idx=pad_id), dim=n_embd, n_head=molecule_transformer_heads, n_layer=molecule_encoder_layers, dropout=dropout)
        self.protein_sequence_encoder = TargetSequenceTransformerEncoder(vocab_size=26, output_dim=n_embd, cached_embedding_dim=protein_cached_embedding_dim, heads=protein_transformer_heads, layers=protein_transformer_layers, dropout=dropout)
        self.condition_encoder = ConditionEncoder(dim=n_embd, n_head=pocket_transformer_heads, n_layer=encoder_layers, dropout=dropout, pocket_in_dim=pocket_in_dim, pocket_node_dim=pocket_node_dim, pocket_input_proj_dim=pocket_input_proj_dim, pocket_hidden_dim=pocket_hidden_dim, pocket_layers=pocket_layers, pocket_residue_feature_dim=pocket_residue_feature_dim, )
        self.pocket_alignment_norm = nn.LayerNorm(n_embd)
        self.pocket_alignment_projection = nn.Linear(n_embd, n_embd, bias=False)
        nn.init.eye_(self.pocket_alignment_projection.weight)
        self.target_sequence_type = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.target_pocket_type = nn.Parameter(torch.zeros(1, 1, n_embd))
        nn.init.normal_(self.target_sequence_type, std=0.02)
        nn.init.normal_(self.target_pocket_type, std=0.02)
        self.aligned_residue_norm = nn.LayerNorm(n_embd)
        self.pocket_alignment_gate_norm = nn.LayerNorm(2 * n_embd)
        self.pocket_alignment_gate = nn.Linear(2 * n_embd, n_embd)
        nn.init.zeros_(self.pocket_alignment_gate.weight)
        nn.init.constant_(self.pocket_alignment_gate.bias, -2.0)
        self.target_fusion = nn.Sequential(nn.LayerNorm(2 * n_embd), nn.Linear(2 * n_embd, n_embd), nn.GELU(), nn.Linear(n_embd, n_embd))
        self.atom_token_selector = AtomTokenSelector()
        self.coattention = nn.ModuleList(AlignedBidirectionalCoAttention(dim=n_embd, heads=n_head, dropout=dropout) for _ in range(cross_attention_layers))
        self.affinity_concat_projection = nn.Sequential(nn.LayerNorm(4 * n_embd), nn.Linear(4 * n_embd, 2 * n_embd), nn.GELU(), nn.Dropout(affinity_dropout), nn.Linear(2 * n_embd, n_embd))
        self.affinity_repr = nn.Sequential(nn.LayerNorm(n_embd), nn.Linear(n_embd, n_embd), nn.GELU(), nn.Dropout(affinity_dropout), nn.LayerNorm(n_embd))
        self.affinity_head = nn.Sequential(nn.Linear(n_embd, n_embd // 2), nn.GELU(), nn.Dropout(affinity_dropout), nn.Linear(n_embd // 2, 1))

    def align_pocket(self, sequence, valid, condition):
        """Scatter-average pocket nodes into unique sequence positions (II-G)."""
        positions = condition.pocket_sequence_positions
        nodes = condition.pocket_states[:, 1:]
        mapped = ~condition.pocket_key_padding_mask[:, 1:] & positions.ge(0) & positions.lt(sequence.shape[1])
        indices = positions.clamp(0, sequence.shape[1] - 1).long()
        mapped = mapped & valid.gather(1, indices)
        sequence = sequence + self.target_sequence_type
        nodes = self.pocket_alignment_projection(self.pocket_alignment_norm(nodes + self.target_pocket_type)).to(sequence.dtype)
        aligned = torch.zeros_like(sequence)
        aligned.scatter_add_(1, indices.unsqueeze(-1).expand_as(nodes), nodes * mapped.unsqueeze(-1))
        counts = sequence.new_zeros(sequence.shape[:2])
        counts.scatter_add_(1, indices, mapped.to(sequence.dtype))
        aligned = aligned / counts.clamp_min(1).unsqueeze(-1)
        in_pocket = counts.gt(0)
        gate = torch.sigmoid(self.pocket_alignment_gate(self.pocket_alignment_gate_norm(torch.cat((sequence, aligned), -1))))
        fused = sequence + gate * aligned
        return (self.aligned_residue_norm(fused) * valid.unsqueeze(-1), in_pocket)

    def forward(self, molecule_tokens, *, pocket_batch, protein_tokens, protein_embeddings, pharmacophore_features, atom_tokens, return_interpretability=False):
        (ligand, z_ligand) = self.molecule_encoder(molecule_tokens, pharmacophore_features=pharmacophore_features, pad_id=self.pad_id)
        (atoms, atom_valid, atom_indices, atom_positions) = self.atom_token_selector(ligand, atom_tokens)
        (sequence, z_sequence) = self.protein_sequence_encoder(protein_tokens, embeddings=protein_embeddings, return_states=True)
        valid = protein_tokens.ne(self.protein_sequence_encoder.pad_id)
        condition = self.condition_encoder(pocket_batch=pocket_batch)
        (residues, in_pocket) = self.align_pocket(sequence, valid, condition)
        fusion = self.target_fusion(torch.cat((condition.pocket_vector, z_sequence), -1))
        gate = torch.sigmoid(fusion)
        z_target = gate * condition.pocket_vector + (1 - gate) * z_sequence
        (initial_atoms, initial_residues) = (atoms, residues)
        for layer in self.coattention:
            (updated_atoms, updated_residues, pair, a_to_r, r_to_a) = layer(atoms, atom_valid, residues, valid)
            atoms = updated_atoms
            residues = updated_residues
        (atom_weights, residue_weights) = (pair.sum(2), pair.sum(1))
        u_atom = ((atoms - initial_atoms) * atom_weights.unsqueeze(-1)).sum(1)
        u_residue = ((residues - initial_residues) * residue_weights.unsqueeze(-1)).sum(1)
        fused = self.affinity_concat_projection(torch.cat((z_ligand, z_target, u_atom, u_residue), -1))
        representation = self.affinity_repr(fused)
        prediction = self.affinity_head(representation).squeeze(-1)
        if not return_interpretability:
            return (prediction, representation)
        positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid).masked_fill(~valid, -1)
        evidence = MutualAttentionInterpretability(pair, a_to_r, r_to_a, atom_weights, residue_weights, atom_indices, atom_positions, positions, in_pocket, atom_valid, valid)
        return (prediction, representation, evidence)
