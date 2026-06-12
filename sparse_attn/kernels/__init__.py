"""
sparse_attn/kernels — Sparse Attention Kernel Implementations
=============================================================

Provides:
  - sparse_prefill  : Triton-based block-sparse prefill (full sequence attention)
  - sparse_decode   : CUDA chunked-KV decode kernel (single-token decode)
  - pattern_compiler: Convert SparseBlockPattern to kernel-ready buffers
"""

from sparse_attn.kernels.pattern_compiler import compile_pattern
from sparse_attn.kernels.sparse_prefill import sparse_prefill, sparse_prefill_reference
from sparse_attn.kernels.sparse_decode import sparse_decode_chunked

__all__ = [
    "compile_pattern",
    "sparse_prefill",
    "sparse_prefill_reference",
    "sparse_decode_chunked",
]
