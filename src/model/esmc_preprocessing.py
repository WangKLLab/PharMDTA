"""Exact residue tokenization and full-sequence ESM-C chunk inference."""
from __future__ import annotations
from typing import Any, Sequence
import torch

DEFAULT_CHUNK_RESIDUES = 2046

EXPECTED_DIM = 2560

def chunk_sequence(sequence: str, max_residues: int) -> tuple[tuple[int, str], ...]:
    if not sequence:
        raise ValueError("sequence must be non-empty")
    if max_residues < 1 or max_residues > DEFAULT_CHUNK_RESIDUES:
        raise ValueError(
            f"max_residues must be in [1, {DEFAULT_CHUNK_RESIDUES}] for ESMC"
        )
    chunks = tuple(
        (offset, sequence[offset : offset + max_residues])
        for offset in range(0, len(sequence), max_residues)
    )
    if "".join(chunk for _, chunk in chunks) != sequence:
        raise AssertionError("non-overlap chunk reconstruction drift")
    return chunks

def reassemble_chunk_states(
    chunk_states: Sequence[torch.Tensor],
    chunk_lengths: Sequence[int],
) -> torch.Tensor:
    """Reassemble ``[chunk_length + 2, dim]`` states into ``[L + 2, dim]``."""

    if not chunk_states or len(chunk_states) != len(chunk_lengths):
        raise ValueError("chunk states and lengths must be non-empty and aligned")
    dim: int | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    residues: list[torch.Tensor] = []
    for index, (states, length) in enumerate(zip(chunk_states, chunk_lengths)):
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError(f"chunk {index}: length must be a positive integer")
        if not isinstance(states, torch.Tensor) or states.ndim != 2:
            raise ValueError(f"chunk {index}: states must be a rank-2 tensor")
        if int(states.shape[0]) != int(length) + 2:
            raise ValueError(
                f"chunk {index}: shape={tuple(states.shape)} does not match L+2={length + 2}"
            )
        if dim is None:
            dim = int(states.shape[1])
            dtype = states.dtype
            device = states.device
        elif int(states.shape[1]) != dim:
            raise ValueError("chunk embedding dimensions differ")
        elif states.dtype != dtype or states.device != device:
            raise ValueError("chunk tensors must have the same dtype and device")
        if not states.is_floating_point() or not bool(torch.isfinite(states).all()):
            raise ValueError(f"chunk {index}: non-finite or non-floating embeddings")
        residues.append(states[1:-1])
    combined = torch.cat(
        [chunk_states[0][0:1], *residues, chunk_states[-1][-1:]], dim=0
    ).contiguous()
    expected_rows = int(sum(chunk_lengths)) + 2
    if int(combined.shape[0]) != expected_rows:
        raise AssertionError("reassembled embedding does not cover every residue exactly once")
    return combined

def encode_request(
    request: Any,
    *,
    model: Any,
    tokenizer: Any,
    max_residues: int,
) -> tuple[torch.Tensor, list[dict[str, int]]]:
    states: list[torch.Tensor] = []
    records: list[dict[str, int]] = []
    chunks = chunk_sequence(request.sequence, max_residues)
    device = model.device
    for offset, sequence in chunks:
        encoded = tokenizer(
            sequence,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        required_fields = {"input_ids", "attention_mask", "special_tokens_mask"}
        if not required_fields.issubset(encoded):
            raise ValueError("ESMC tokenizer omitted a required strict mask")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        special_mask = encoded["special_tokens_mask"]
        expected_shape = (1, len(sequence) + 2)
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError("ESMC tokenizer did not preserve the exact residue count")
        if tuple(attention_mask.shape) != expected_shape or not bool(
            attention_mask.bool().all()
        ):
            raise ValueError("ESMC attention mask must cover every unpadded input token")
        if tuple(special_mask.shape) != expected_shape:
            raise ValueError("ESMC special-token mask has an unexpected shape")
        expected_special_mask = torch.zeros_like(special_mask, dtype=torch.bool)
        expected_special_mask[:, 0] = True
        expected_special_mask[:, -1] = True
        if not torch.equal(special_mask.bool(), expected_special_mask):
            raise ValueError("ESMC tokenizer must place exactly one BOS and one EOS at the ends")
        cls_token_id = getattr(tokenizer, "cls_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if (
            cls_token_id is None
            or eos_token_id is None
            or int(input_ids[0, 0]) != int(cls_token_id)
            or int(input_ids[0, -1]) != int(eos_token_id)
        ):
            raise ValueError("ESMC tokenizer BOS/EOS token IDs do not match its contract")
        expected_residue_ids = tokenizer.convert_tokens_to_ids(list(sequence))
        if not isinstance(expected_residue_ids, list) or len(expected_residue_ids) != len(
            sequence
        ):
            raise ValueError("ESMC tokenizer does not expose one token per residue")
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if unk_token_id is not None and any(
            int(token_id) == int(unk_token_id) for token_id in expected_residue_ids
        ):
            raise ValueError("ESMC tokenizer mapped a supported residue to <unk>")
        expected_residue_tensor = torch.tensor(
            expected_residue_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        if not torch.equal(input_ids[0, 1:-1], expected_residue_tensor):
            raise ValueError("ESMC tokenizer changed residue identity or order")
        model_inputs = {
            key: encoded[key].to(device) for key in ("input_ids", "attention_mask")
        }
        with torch.inference_mode():
            output = model(**model_inputs)
        hidden = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("ESMC model did not return rank-3 last_hidden_state")
        value = hidden[0].to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(value.shape) != (len(sequence) + 2, EXPECTED_DIM):
            raise ValueError(f"unexpected ESMC hidden-state shape={tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("ESMC produced NaN or infinity")
        states.append(value)
        records.append({"offset": offset, "residues": len(sequence)})
        del output, hidden, model_inputs
    return reassemble_chunk_states(states, [len(chunk) for _, chunk in chunks]), records
