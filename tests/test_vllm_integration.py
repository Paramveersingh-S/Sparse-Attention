"""
tests/test_vllm_integration.py
================================
Tests for the vLLM backend integration (uses mock metadata when vLLM absent).
"""

import pytest
import torch
from sparse_attn.vllm import (
    SparseAttentionBackend,
    SparseAttentionImpl,
    MockAttentionMetadata,
)


class TestSparseAttentionImpl:

    def setup_method(self):
        self.impl = SparseAttentionImpl(
            num_heads=4,
            head_size=32,
            scale=32 ** -0.5,
            num_kv_heads=4,
            max_seq_len=512,
            sparse_config={
                "block_size":          64,
                "local_window_blocks": 4,
                "stride_blocks":       8,
                "global_blocks":       2,
                "chunk_size":          64,
            }
        )

    def test_prefill_forward_shape(self):
        B, H, S, D = 1, 4, 512, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        meta = MockAttentionMetadata(is_prompt=True, seq_lens=[S], max_seq_len=S)
        out = self.impl.forward(q, k, v, attn_metadata=meta)
        assert out.shape == (B, H, S, D)

    def test_decode_forward_shape(self):
        B, H, D = 1, 4, 32
        S_kv = 512
        q = torch.randn(B, H, 1, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)
        meta = MockAttentionMetadata(is_prompt=False, seq_lens=[S_kv], max_seq_len=S_kv)
        out = self.impl.forward(q, k, v, attn_metadata=meta)
        assert out.shape == (B, H, 1, D)

    def test_pattern_cache(self):
        """Pattern should be cached and reused."""
        impl = SparseAttentionImpl(
            num_heads=2, head_size=32, scale=None,
            max_seq_len=512
        )
        p1 = impl._get_or_build_pattern(512)
        p2 = impl._get_or_build_pattern(512)
        assert p1 is p2  # Same object (cached)

    def test_no_nan_in_output(self):
        B, H, S, D = 1, 4, 512, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        meta = MockAttentionMetadata(is_prompt=True, seq_lens=[S], max_seq_len=S)
        out = self.impl.forward(q, k, v, attn_metadata=meta)
        assert not torch.isnan(out).any(), "NaN in vLLM backend output"


class TestSparseAttentionBackend:

    def test_get_name(self):
        assert SparseAttentionBackend.get_name() == "SPARSE_ATTN"

    def test_get_impl_cls(self):
        assert SparseAttentionBackend.get_impl_cls() is SparseAttentionImpl

    def test_make_metadata(self):
        meta = SparseAttentionBackend.make_metadata(is_prompt=True, max_seq_len=1024)
        assert meta.is_prompt is True

    def test_kv_cache_shape(self):
        shape = SparseAttentionBackend.get_kv_cache_shape(
            num_blocks=16, block_size=16, num_kv_heads=4, head_size=32
        )
        # Should be (2, num_blocks, block_size, num_kv_heads, head_size)
        assert shape == (2, 16, 16, 4, 32)
