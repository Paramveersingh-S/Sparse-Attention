"""
sparse_attn/kernels/sparse_prefill.py
=======================================
Triton-based block-sparse prefill kernel with online softmax.

Architecture
------------
Grid: (num_active_query_blocks, num_heads, batch_size)
Each CTA processes one (query_block, head, batch) triple.
Inner loop: iterate over active KV blocks from CSR column list.

Online Softmax (Flash-Attention style)
--------------------------------------
For each KV block:
  1. Compute attention scores S = Q·Kᵀ * scale
  2. Apply causal mask (diagonal blocks only, token-level)
  3. Update running (m, l, O) via online softmax merge
Final: O = O / l

Skip Logic
----------
If a query block has zero active KV blocks (empty CSR row),
the output for that block is filled with zeros — no divergence.

Fallback
--------
When Triton is not installed (CPU/non-GPU environments),
sparse_prefill() falls back to sparse_prefill_reference()
which implements the same algorithm in pure PyTorch.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from sparse_attn.patterns.base import SparseBlockPattern
from sparse_attn.kernels.pattern_compiler import compile_pattern

# --------------------------------------------------------------------------- #
#  Try to import Triton; fall back gracefully                                  #
# --------------------------------------------------------------------------- #
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# --------------------------------------------------------------------------- #
#  Pure-Python / PyTorch Reference Implementation                              #
# --------------------------------------------------------------------------- #

def sparse_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pattern: SparseBlockPattern,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Reference (pure-PyTorch) implementation of block-sparse prefill.

    Semantically equivalent to the Triton kernel but runs on any device.
    Used for:
    - Correctness validation against the Triton kernel
    - CPU / non-GPU environments (Google Colab without GPU)
    - Small sequences where kernel launch overhead dominates

    Parameters
    ----------
    q, k, v : Tensors of shape [B, H, S, D].
    pattern : SparseBlockPattern defining which blocks are attended.
    softmax_scale : Scale factor for dot products. Default: 1/sqrt(D).

    Returns
    -------
    output : Tensor [B, H, S, D]
    """
    B, H, S, D = q.shape
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    bs   = pattern.block_size
    nb   = pattern.num_blocks
    assert S == pattern.seq_len, f"q sequence length {S} != pattern seq_len {pattern.seq_len}"

    # Output tensor
    out = torch.zeros_like(q)

    for b in range(B):
        for h in range(H):
            for query_block in range(nb):
                active_cols = pattern.get_active_cols(query_block)
                if not active_cols:
                    # No KV blocks attended: output stays zero
                    continue

                q_start = query_block * bs
                q_end   = q_start + bs
                q_tile  = q[b, h, q_start:q_end, :]  # [bs, D]

                # Running accumulators for online softmax
                m = torch.full((bs,), float('-inf'), dtype=torch.float32, device=q.device)
                l = torch.zeros((bs,), dtype=torch.float32, device=q.device)
                o = torch.zeros((bs, D), dtype=torch.float32, device=q.device)

                for kv_block in active_cols:
                    kv_start = kv_block * bs
                    kv_end   = kv_start + bs
                    k_tile = k[b, h, kv_start:kv_end, :]  # [bs, D]
                    v_tile = v[b, h, kv_start:kv_end, :]  # [bs, D]

                    # Attention scores [bs, bs]
                    s = torch.matmul(
                        q_tile.float(),
                        k_tile.float().T
                    ) * softmax_scale

                    # Causal masking (diagonal blocks need token-level mask)
                    if kv_block == query_block:
                        # Token-level causal mask
                        q_ids  = torch.arange(q_start, q_end, device=q.device)[:, None]
                        kv_ids = torch.arange(kv_start, kv_end, device=q.device)[None, :]
                        causal = q_ids >= kv_ids
                        s = s.masked_fill(~causal, float('-inf'))
                    elif kv_block > query_block:
                        # Fully future block — should not be in active list
                        # but mask entirely for safety
                        s = torch.full_like(s, float('-inf'))

                    # Online softmax update
                    row_max = s.max(dim=1).values   # [bs]
                    m_new   = torch.maximum(m, row_max)

                    p     = torch.exp(s - m_new[:, None])  # [bs, bs]
                    l_new = torch.exp(m - m_new) * l + p.sum(dim=1)
                    o     = torch.exp(m - m_new)[:, None] * o + torch.matmul(p, v_tile.float())

                    m, l = m_new, l_new

                # Normalize
                # Handle zero l (fully masked rows → output zero)
                safe_l = l.clone()
                safe_l[safe_l == 0] = 1.0
                o = o / safe_l[:, None]
                out[b, h, q_start:q_end, :] = o.to(q.dtype)

    return out


# --------------------------------------------------------------------------- #
#  Triton Kernel                                                               #
# --------------------------------------------------------------------------- #

if _TRITON_AVAILABLE:

    @triton.jit
    def _sparse_prefill_kernel(
        # Input pointers
        q_ptr,  # [B, H, S, D]
        k_ptr,
        v_ptr,
        o_ptr,  # Output
        # CSR sparse pattern arrays
        col_indices_ptr,  # [num_active_blocks]
        row_ptrs_ptr,     # [num_blocks + 1]
        causal_flags_ptr, # [num_active_blocks] (bool/int8)
        # Dimensions
        B, H, S, D,
        # Strides for q, k, v, o: [B, H, S, D]
        stride_qb, stride_qh, stride_qs, stride_qd,
        stride_kb, stride_kh, stride_ks, stride_kd,
        stride_vb, stride_vh, stride_vs, stride_vd,
        stride_ob, stride_oh, stride_os, stride_od,
        softmax_scale,
        # Constexpr
        BLOCK_SIZE: tl.constexpr,  # tokens per block
        BLOCK_D:    tl.constexpr,  # head dimension
    ):
        # ------------------------------------------------------------------ #
        #  Block/thread assignment                                            #
        # ------------------------------------------------------------------ #
        query_block = tl.program_id(0)
        head_idx    = tl.program_id(1)
        batch_idx   = tl.program_id(2)

        q_block_start = query_block * BLOCK_SIZE

        # ------------------------------------------------------------------ #
        #  Load query tile [BLOCK_SIZE, BLOCK_D]                             #
        # ------------------------------------------------------------------ #
        offs_m = tl.arange(0, BLOCK_SIZE)   # token offsets within block
        offs_d = tl.arange(0, BLOCK_D)      # head dim offsets

        # Q pointer for this (batch, head, query_block)
        q_base = (
            batch_idx * stride_qb
            + head_idx * stride_qh
            + (q_block_start + offs_m[:, None]) * stride_qs
            + offs_d[None, :] * stride_qd
        )
        # Bounds check
        q_mask = (q_block_start + offs_m[:, None] < S) & (offs_d[None, :] < D)
        q = tl.load(q_ptr + q_base, mask=q_mask, other=0.0)  # [BLOCK_SIZE, BLOCK_D]

        # ------------------------------------------------------------------ #
        #  Initialize online softmax accumulators                            #
        # ------------------------------------------------------------------ #
        m = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
        l = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        o = tl.zeros([BLOCK_SIZE, BLOCK_D], dtype=tl.float32)

        # ------------------------------------------------------------------ #
        #  CSR row bounds                                                     #
        # ------------------------------------------------------------------ #
        row_start = tl.load(row_ptrs_ptr + query_block)
        row_end   = tl.load(row_ptrs_ptr + query_block + 1)

        # ------------------------------------------------------------------ #
        #  Inner loop: iterate active KV blocks                              #
        # ------------------------------------------------------------------ #
        for kv_ptr_idx in range(row_start, row_end):
            kv_block   = tl.load(col_indices_ptr + kv_ptr_idx)
            is_causal  = tl.load(causal_flags_ptr + kv_ptr_idx)  # 0 or 1
            kv_start   = kv_block * BLOCK_SIZE

            # Load K tile
            k_base = (
                batch_idx * stride_kb
                + head_idx * stride_kh
                + (kv_start + offs_m[None, :]) * stride_ks
                + offs_d[:, None] * stride_kd
            )
            k_mask = (kv_start + offs_m[None, :] < S) & (offs_d[:, None] < D)
            k = tl.load(k_ptr + k_base, mask=k_mask, other=0.0)  # [BLOCK_D, BLOCK_SIZE]

            # Load V tile
            v_base = (
                batch_idx * stride_vb
                + head_idx * stride_vh
                + (kv_start + offs_m[:, None]) * stride_vs
                + offs_d[None, :] * stride_vd
            )
            v_mask = (kv_start + offs_m[:, None] < S) & (offs_d[None, :] < D)
            v = tl.load(v_ptr + v_base, mask=v_mask, other=0.0)  # [BLOCK_SIZE, BLOCK_D]

            # Attention scores: [BLOCK_SIZE, BLOCK_SIZE]
            # q: [BLOCK_SIZE, BLOCK_D], k: [BLOCK_D, BLOCK_SIZE]
            s = tl.dot(q, k) * softmax_scale

            # Causal masking (diagonal blocks: token-level)
            if is_causal:
                row_ids = (q_block_start + offs_m)[:, None]
                col_ids = (kv_start + offs_m)[None, :]
                s = tl.where(col_ids <= row_ids, s, float('-inf'))

            # Online softmax update
            row_max = tl.max(s, axis=1)              # [BLOCK_SIZE]
            m_new   = tl.maximum(m, row_max)
            p       = tl.exp(s - m_new[:, None])    # [BLOCK_SIZE, BLOCK_SIZE]
            l_new   = tl.exp(m - m_new) * l + tl.sum(p, axis=1)
            o       = tl.exp(m - m_new)[:, None] * o + tl.dot(p.to(tl.float16), v)

            m = m_new
            l = l_new

        # ------------------------------------------------------------------ #
        #  Normalize output                                                   #
        # ------------------------------------------------------------------ #
        safe_l = tl.where(l == 0.0, 1.0, l)
        o = o / safe_l[:, None]

        # ------------------------------------------------------------------ #
        #  Store output [BLOCK_SIZE, BLOCK_D]                                #
        # ------------------------------------------------------------------ #
        o_base = (
            batch_idx * stride_ob
            + head_idx * stride_oh
            + (q_block_start + offs_m[:, None]) * stride_os
            + offs_d[None, :] * stride_od
        )
        o_mask = (q_block_start + offs_m[:, None] < S) & (offs_d[None, :] < D)
        tl.store(o_ptr + o_base, o.to(tl.float16), mask=o_mask)


def _sparse_prefill_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pattern: SparseBlockPattern,
    softmax_scale: float,
) -> torch.Tensor:
    """
    Launch the Triton sparse prefill kernel.
    q, k, v : [B, H, S, D] float16, contiguous, on CUDA device.
    """
    B, H, S, D = q.shape
    nb   = pattern.num_blocks
    bs   = pattern.block_size

    # Compile pattern to CSR format on GPU
    kf = compile_pattern(pattern, device=q.device)
    col_indices  = kf["col_indices"].to(torch.int32)
    row_ptrs     = kf["row_ptrs"].to(torch.int32)
    causal_flags = kf["causal_mask"].to(torch.int8)

    # Output tensor
    out = torch.zeros_like(q)

    # Grid: (num_query_blocks, num_heads, batch)
    grid = (nb, H, B)

    # Constexpr must be power-of-2; find next power of 2 for D
    BLOCK_D = 2 ** math.ceil(math.log2(D))

    _sparse_prefill_kernel[grid](
        q, k, v, out,
        col_indices, row_ptrs, causal_flags,
        B, H, S, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        softmax_scale,
        BLOCK_SIZE=bs,
        BLOCK_D=BLOCK_D,
    )
    return out


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def sparse_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pattern: SparseBlockPattern,
    softmax_scale: Optional[float] = None,
    force_reference: bool = False,
) -> torch.Tensor:
    """
    Block-sparse prefill attention.

    Automatically selects:
    - Triton kernel (if available and on CUDA)
    - Pure-PyTorch reference (CPU or Triton unavailable)

    Parameters
    ----------
    q, k, v        : Tensors [B, H, S, D]. float16 for Triton, float32 for reference.
    pattern        : SparseBlockPattern (must match seq_len S).
    softmax_scale  : Score scale. Default: D^{-0.5}.
    force_reference: If True, always use the reference implementation.

    Returns
    -------
    output : Tensor [B, H, S, D]
    """
    B, H, S, D = q.shape
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    assert S == pattern.seq_len, (
        f"Tensor sequence length {S} != pattern.seq_len {pattern.seq_len}"
    )

    use_triton = (
        _TRITON_AVAILABLE
        and q.is_cuda
        and not force_reference
    )

    if use_triton:
        # Ensure float16 for Triton
        q_fp16 = q.half() if q.dtype != torch.float16 else q
        k_fp16 = k.half() if k.dtype != torch.float16 else k
        v_fp16 = v.half() if v.dtype != torch.float16 else v
        return _sparse_prefill_triton(q_fp16, k_fp16, v_fp16, pattern, softmax_scale)
    else:
        return sparse_prefill_reference(q, k, v, pattern, softmax_scale)
