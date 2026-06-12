"""
tests/test_oom_free.py
========================
OOM boundary tests — verify sparse attention doesn't OOM where dense would.

On GPU (Colab T4/A100): tests large sequences up to available memory.
On CPU: tests reference implementation scalability.
"""

import pytest
import torch
from sparse_attn.patterns import make_local_stride_pattern
from sparse_attn.kernels import sparse_prefill_reference


def _try_sparse_attn(seq_len: int, block_size: int = 64, B: int = 1, H: int = 1, D: int = 32):
    """Attempt sparse attention at given seq_len. Returns True if successful."""
    try:
        pattern = make_local_stride_pattern(
            seq_len=seq_len, block_size=block_size,
            local_window_blocks=8, stride_blocks=16, global_blocks=4
        )
        q = torch.randn(B, H, seq_len, D)
        k = torch.randn(B, H, seq_len, D)
        v = torch.randn(B, H, seq_len, D)
        out = sparse_prefill_reference(q, k, v, pattern)
        return out.shape == (B, H, seq_len, D)
    except (MemoryError, torch.cuda.OutOfMemoryError):
        return False


class TestOOMFree:

    def test_no_oom_4k(self):
        assert _try_sparse_attn(4096), "OOM at 4K tokens"

    def test_no_oom_8k(self):
        assert _try_sparse_attn(8192), "OOM at 8K tokens"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU-only test (skip on CPU Colab)"
    )
    def test_no_oom_128k_gpu(self):
        """128K tokens — requires GPU. Dense would need 32GB just for attention matrix."""
        device = "cuda"
        seq_len, block_size = 131072, 64
        B, H, D = 1, 1, 64

        pattern = make_local_stride_pattern(
            seq_len=seq_len, block_size=block_size,
            local_window_blocks=8, stride_blocks=16, global_blocks=4
        )
        try:
            from sparse_attn.kernels import sparse_prefill
            q = torch.randn(B, H, seq_len, D, dtype=torch.float16, device=device)
            k = torch.randn(B, H, seq_len, D, dtype=torch.float16, device=device)
            v = torch.randn(B, H, seq_len, D, dtype=torch.float16, device=device)
            out = sparse_prefill(q, k, v, pattern)
            assert out.shape == (B, H, seq_len, D)
        except torch.cuda.OutOfMemoryError:
            pytest.fail("OOM at 128K tokens — sparse attention should handle this on A100/T4")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU-only test"
    )
    def test_memory_reduction_8x(self):
        """
        Verify memory usage is ≥8× less than dense for 128K context.
        Dense: 128K² × 2 bytes = 32GB
        Sparse (99% sparse): 32GB × 0.01 = 320MB
        """
        from sparse_attn.kernels.pattern_compiler import compute_memory_reduction

        seq_len = 131072
        pattern = make_local_stride_pattern(seq_len=seq_len, block_size=64)
        dense_gb, sparse_gb, reduction = compute_memory_reduction(
            seq_len=seq_len, sparsity=pattern.sparsity
        )

        print(f"\nMemory: dense={dense_gb:.1f}GB, sparse={sparse_gb:.2f}GB, "
              f"reduction={reduction:.1f}×")
        assert reduction >= 8.0, (
            f"Expected ≥8× memory reduction at 128K, got {reduction:.1f}×"
        )


def find_oom_boundary(
    block_size: int = 64,
    heads: int = 1,
    batch: int = 1,
    head_dim: int = 32,
    lo: int = 1024,
    hi: int = 65536,
) -> int:
    """Binary search for maximum sequence length before OOM."""
    while lo < hi - block_size:
        mid = ((lo + hi) // 2 // block_size) * block_size  # align to block size
        if _try_sparse_attn(mid, block_size, batch, heads, head_dim):
            lo = mid
        else:
            hi = mid
    return lo


class TestOOMBoundarySearch:

    def test_find_boundary(self):
        """Binary search should find a valid OOM boundary."""
        boundary = find_oom_boundary(block_size=64, hi=16384)
        assert boundary >= 1024, f"OOM boundary too low: {boundary}"
        print(f"\nSparse OOM boundary (CPU): {boundary:,} tokens")
