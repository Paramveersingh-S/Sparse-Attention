"""
sparse_attn/patterns — Sparse Attention Pattern Library
=======================================================

This module provides:
  - SparseBlockPattern: core data structure with CSR serialization
  - LocalWindowPattern: causal local-window pattern
  - StridedPattern: global landmark stride pattern
  - HeterogeneousPattern: mixed local+stride (the primary production pattern)
  - CustomPattern: user-defined lambda-rule pattern
  - make_local_stride_pattern: factory function (main entry point)
"""

from sparse_attn.patterns.base import SparseBlockPattern
from sparse_attn.patterns.local_window import LocalWindowPattern
from sparse_attn.patterns.strided import StridedPattern
from sparse_attn.patterns.heterogeneous import HeterogeneousPattern, make_local_stride_pattern
from sparse_attn.patterns.custom import CustomPattern

__all__ = [
    "SparseBlockPattern",
    "LocalWindowPattern",
    "StridedPattern",
    "HeterogeneousPattern",
    "CustomPattern",
    "make_local_stride_pattern",
]
