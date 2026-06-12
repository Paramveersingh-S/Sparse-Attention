"""
sparse_attn/kernels/pattern_compiler.py
=========================================
Convert a SparseBlockPattern into kernel-ready tensors.

This module handles:
1. Extracting CSR arrays from SparseBlockPattern.to_kernel_format()
2. Preparing chunk-activity masks for the chunked-KV decode kernel
3. Computing expected SM utilization for a given configuration
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from sparse_attn.patterns.base import SparseBlockPattern


def compile_pattern(
    pattern: SparseBlockPattern,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Convert a SparseBlockPattern to kernel-ready CSR tensors on the target device.

    Parameters
    ----------
    pattern : SparseBlockPattern to compile.
    device  : Target device. If None, uses the pattern's block_mask device.

    Returns
    -------
    dict with keys:
        col_indices : LongTensor [num_active_blocks]
        row_ptrs    : LongTensor [num_blocks + 1]
        causal_mask : BoolTensor [num_active_blocks]
        num_blocks  : int
        block_size  : int
        sparsity    : float
    """
    if device is None:
        device = pattern.block_mask.device

    # Cache device-specific tensors to avoid host-to-device transfer in hot loops
    cache_key = f"_cached_compiled_{device}"
    if hasattr(pattern, cache_key) and getattr(pattern, cache_key) is not None:
        return getattr(pattern, cache_key)

    kf = pattern.to_kernel_format()
    result = {
        "col_indices": kf["col_indices"].to(device, dtype=torch.int32),
        "row_ptrs":    kf["row_ptrs"].to(device, dtype=torch.int32),
        "causal_mask": kf["causal_mask"].to(device, dtype=torch.int8),
        "num_blocks":  kf["num_blocks"],
        "block_size":  kf["block_size"],
        "sparsity":    pattern.sparsity,
    }
    setattr(pattern, cache_key, result)
    return result


def build_chunk_active_mask(
    pattern: SparseBlockPattern,
    chunk_size: int,
    batch_size: int,
    num_heads: int,
    query_position: int = -1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build the chunk_active mask for the chunked-KV decode kernel.

    For decode (single-token query at position query_position), determines
    which KV chunks fall within the sparse pattern's active region.

    Parameters
    ----------
    pattern        : SparseBlockPattern defining the attention mask.
    chunk_size     : KV chunk size in tokens (e.g., 512).
    batch_size     : Batch dimension.
    num_heads      : Number of attention heads.
    query_position : Current decode position (token index). -1 = last token.
    device         : Target device.

    Returns
    -------
    chunk_active : BoolTensor [batch_size, num_heads, num_chunks]
        True if the chunk is within the sparse attention pattern.
    """
    seq_len = pattern.seq_len
    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    if query_position < 0:
        query_position = seq_len - 1

    # Determine which block the query is in
    query_block = query_position // pattern.block_size
    active_kv_blocks = set(pattern.get_active_cols(min(query_block, pattern.num_blocks - 1)))

    # Map block indices to chunk indices
    chunk_active_1d = torch.zeros(num_chunks, dtype=torch.bool)
    for block_idx in active_kv_blocks:
        token_start = block_idx * pattern.block_size
        token_end   = token_start + pattern.block_size
        chunk_start = token_start // chunk_size
        chunk_end   = (token_end - 1) // chunk_size + 1
        for c in range(chunk_start, min(chunk_end, num_chunks)):
            chunk_active_1d[c] = True

    # Broadcast to [B, H, num_chunks]
    chunk_active = chunk_active_1d.view(1, 1, num_chunks).expand(
        batch_size, num_heads, num_chunks
    ).contiguous()

    if device is not None:
        chunk_active = chunk_active.to(device)

    return chunk_active


def expected_sm_utilization(
    batch: int,
    heads: int,
    seq_len: int,
    chunk_size: int,
    sparsity: float,
    num_sms: int = 128,
) -> float:
    """
    Estimate SM utilization for the chunked-KV decode kernel.

    Parameters
    ----------
    batch      : Batch size.
    heads      : Number of attention heads.
    seq_len    : KV sequence length.
    chunk_size : KV chunk size in tokens.
    sparsity   : Fraction of chunks NOT processed (0.0 → dense, 1.0 → fully sparse).
    num_sms    : Number of SMs on the GPU (128 for A100, 82 for RTX 3090, 128 for RTX 4090).

    Returns
    -------
    Utilization ratio (0.0–1.0, clipped). Values > 1.0 mean oversubscription → 100% util.
    """
    num_chunks = seq_len // chunk_size
    active_chunks = int(num_chunks * (1.0 - sparsity))
    total_blocks = batch * heads * active_chunks
    return min(total_blocks / max(num_sms, 1), 1.0)


def compute_memory_reduction(
    seq_len: int,
    sparsity: float,
    dtype_bytes: int = 2,  # float16
) -> Tuple[float, float, float]:
    """
    Compute memory usage for dense vs sparse attention.

    Returns
    -------
    (dense_gb, sparse_gb, reduction_factor)
    """
    n = seq_len
    dense_bytes  = n * n * dtype_bytes
    sparse_bytes = dense_bytes * (1.0 - sparsity)
    dense_gb     = dense_bytes / 1e9
    sparse_gb    = sparse_bytes / 1e9
    reduction    = dense_gb / max(sparse_gb, 1e-9)
    return dense_gb, sparse_gb, reduction
