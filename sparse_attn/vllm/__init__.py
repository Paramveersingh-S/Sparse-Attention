"""
sparse_attn/vllm — vLLM Attention Backend Integration
======================================================

Provides a vLLM-compatible sparse attention backend and a model patching
utility for drop-in replacement of dense attention with sparse attention.

Usage
-----
    from sparse_attn.vllm import patch_vllm_model_with_sparse_attention

    model = patch_vllm_model_with_sparse_attention(
        "meta-llama/Meta-Llama-3-8B",
        {"local_window": 1024, "stride": 2048, "global_blocks": 4}
    )
    output = model.generate(prompt, max_length=1000)

Or use the backend directly:
    from sparse_attn.vllm import SparseAttentionBackend
"""

from sparse_attn.vllm.sparse_attn_backend import (
    SparseAttentionBackend,
    SparseAttentionImpl,
    MockAttentionMetadata,
)
from sparse_attn.vllm.patch_model import (
    patch_vllm_model_with_sparse_attention,
    SparseAttentionModule,
)

__all__ = [
    "SparseAttentionBackend",
    "SparseAttentionImpl",
    "MockAttentionMetadata",
    "patch_vllm_model_with_sparse_attention",
    "SparseAttentionModule",
]
