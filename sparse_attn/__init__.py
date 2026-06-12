"""
sparse_attn — Custom Sparse Attention Kernels for Infinite Context Windows
==========================================================================

Implements heterogeneous local-stride block-sparse attention patterns,
Triton prefill kernels, CUDA chunked-KV decode kernels, and vLLM integration.

Quick start:
    from sparse_attn.patterns import make_local_stride_pattern
    from sparse_attn.kernels import sparse_prefill, sparse_decode

    pattern = make_local_stride_pattern(seq_len=32768, block_size=64)
    out = sparse_prefill(q, k, v, pattern)
"""

__version__ = "0.1.0"
__author__ = "Sparse Attention Team"

from sparse_attn.patterns import (
    make_local_stride_pattern,
    LocalWindowPattern,
    StridedPattern,
    HeterogeneousPattern,
    CustomPattern,
    SparseBlockPattern,
)

__all__ = [
    "make_local_stride_pattern",
    "LocalWindowPattern",
    "StridedPattern",
    "HeterogeneousPattern",
    "CustomPattern",
    "SparseBlockPattern",
]
