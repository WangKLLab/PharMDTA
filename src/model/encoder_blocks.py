"""Bidirectional transformer blocks used by the DTA ligand encoder.

This is the affinity-relevant extraction of the former decoder module.  It
contains no autoregressive block, language-model head, or GPT implementation.
The submodule names intentionally match the historical ``EncoderBlock`` so
existing affinity checkpoint keys remain loadable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.size(-1)
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)
    cos = cos[..., : head_dim // 2]
    sin = sin[..., : head_dim // 2]
    q = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
    k = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
    return q, k


class BidirectionalSelfAttention(nn.Module):
    """Full-context multi-head attention with optional rotary positions."""

    def __init__(self, config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_rope = bool(getattr(config, "use_rope", False))
        if self.use_rope:
            if self.head_dim % 2:
                raise ValueError("head_dim must be even for RoPE")
            inv_freq = 1.0 / (
                10000
                ** (
                    torch.arange(0, self.head_dim, 2).float()
                    / self.head_dim
                )
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer("_cos_cached", None, persistent=False)
            self.register_buffer("_sin_cached", None, persistent=False)
            self._cached_seq_len = 0

    def _get_rope_cache(
        self, seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_rope:
            raise RuntimeError("RoPE cache requested while use_rope=False")
        cache_missing = self._cos_cached is None
        cache_too_short = seq_len > self._cached_seq_len
        cache_wrong_device = (
            self._cos_cached is not None and self._cos_cached.device != device
        )
        if cache_missing or cache_too_short or cache_wrong_device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
            self._sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
            self._cached_seq_len = seq_len
        return (
            self._cos_cached[:, :, :seq_len],
            self._sin_cached[:, :, :seq_len],
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, channels = x.size()
        key = self.key(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        query = self.query(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        value = self.value(x).view(
            batch_size, seq_len, self.n_head, self.head_dim
        ).transpose(1, 2)
        if self.use_rope:
            query, key = apply_rotary_pos_emb(
                query,
                key,
                *self._get_rope_cache(seq_len, x.device),
            )
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask[:, None, None, :].to(torch.bool)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )
        return self.resid_drop(self.proj(output))


class EncoderBlock(nn.Module):
    """Pre-normalized bidirectional self-attention and feed-forward block."""

    def __init__(self, config) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x
