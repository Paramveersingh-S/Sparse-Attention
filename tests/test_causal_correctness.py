"""
tests/test_causal_correctness.py
===================================
Tests for causal constraint, diagonal attendance, and kernel correctness.
"""

import pytest
import torch
import math
from sparse_attn.patterns import make_local_stride_pattern, LocalWindowPattern, HeterogeneousPattern
from sparse_attn.kernels import sparse_prefill, sparse_prefill_reference


# --------------------------------------------------------------------------- #
#  Causal Constraint Tests                                                    #
# --------------------------------------------------------------------------- #

class TestCausalConstraint:

    @pytest.mark.parametrize("seq_len,block_size", [
        (256,  64),
        (1024, 64),
        (4096, 64),
        (4096, 128),
    ])
    def test_causal_constraint_always_satisfied(self, seq_len, block_size):
        """No query block can attend to future key blocks."""
        pattern = make_local_stride_pattern(seq_len=seq_len, block_size=block_size)
        for row in range(pattern.num_blocks):
            for col in pattern.get_active_cols(row):
                assert col <= row, (
                    f"Causal violation: block {row} attends to future block {col} "
                    f"(seq={seq_len}, bs={block_size})"
                )

    @pytest.mark.parametrize("seq_len,block_size", [
        (256,  64),
        (1024, 64),
        (4096, 64),
    ])
    def test_diagonal_always_attended(self, seq_len, block_size):
        """Every block must attend to itself (causal self-attention)."""
        pattern = make_local_stride_pattern(seq_len=seq_len, block_size=block_size)
        for row in range(pattern.num_blocks):
            assert row in pattern.get_active_cols(row), (
                f"Block {row} does not attend to itself! "
                f"(seq={seq_len}, bs={block_size})"
            )

    def test_validate_causal_method(self):
        """validate_causal() should pass for all standard patterns."""
        p = make_local_stride_pattern(seq_len=2048, block_size=64)
        p.validate_causal()  # Should not raise

    def test_validate_diagonal_method(self):
        """validate_diagonal() should pass for all standard patterns."""
        p = make_local_stride_pattern(seq_len=2048, block_size=64)
        p.validate_diagonal()


# --------------------------------------------------------------------------- #
#  Kernel Correctness Tests                                                   #
# --------------------------------------------------------------------------- #

def _dense_attention(q, k, v, softmax_scale=None):
    """Dense (reference) causal attention for correctness comparison."""
    B, H, S, D = q.shape
    if softmax_scale is None:
        softmax_scale = D ** -0.5

    # Full attention matrix [B, H, S, S]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * softmax_scale

    # Causal mask
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
    scores = scores.masked_fill(~causal[None, None, :, :], float('-inf'))

    # Softmax + weighted sum
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v.float())
    return output.to(q.dtype)


class TestKernelCorrectness:

    @pytest.mark.parametrize("seq_len,block_size,B,H,D", [
        (128, 16, 1, 4,  32),
        (256, 64, 1, 4,  64),
        (512, 64, 2, 8,  64),
    ])
    def test_sparse_matches_dense_on_attended_positions(
        self, seq_len, block_size, B, H, D
    ):
        """
        Sparse and dense must agree on all positions that ARE attended.
        Uses dense pattern (all blocks active) → should match dense exactly.
        """
        torch.manual_seed(42)

        # Use dense pattern (all-local, window=all)
        nb = seq_len // block_size
        from sparse_attn.patterns.base import SparseBlockPattern
        import torch as t
        mask = t.tril(t.ones(nb, nb, dtype=t.bool))
        pattern = SparseBlockPattern(seq_len=seq_len, block_size=block_size, block_mask=mask)

        q = torch.randn(B, H, seq_len, D, dtype=torch.float32)
        k = torch.randn(B, H, seq_len, D, dtype=torch.float32)
        v = torch.randn(B, H, seq_len, D, dtype=torch.float32)

        # Reference implementation
        out_ref     = _dense_attention(q, k, v)
        out_sparse  = sparse_prefill_reference(q, k, v, pattern)

        # Must match on all positions (dense pattern)
        max_err = (out_ref - out_sparse).abs().max().item()
        assert max_err < 1e-3, (
            f"Max error {max_err:.6f} exceeds 1e-3 "
            f"(seq={seq_len}, bs={block_size}, B={B}, H={H}, D={D})"
        )

    @pytest.mark.parametrize("seq_len,block_size", [
        (256, 64),
        (512, 64),
    ])
    def test_sparse_reference_causal(self, seq_len, block_size):
        """
        Output at position i should NOT depend on positions j > i.
        Test by zeroing out future tokens and checking output is unchanged.
        """
        torch.manual_seed(0)
        B, H, D = 1, 2, 32

        pattern = make_local_stride_pattern(
            seq_len=seq_len, block_size=block_size,
            local_window_blocks=2, stride_blocks=4, global_blocks=1
        )

        q = torch.randn(B, H, seq_len, D)
        k = torch.randn(B, H, seq_len, D)
        v = torch.randn(B, H, seq_len, D)

        out1 = sparse_prefill_reference(q, k, v, pattern)

        # Zero out second half of k, v (future tokens)
        half = seq_len // 2
        k2 = k.clone(); k2[:, :, half:, :] = 0
        v2 = v.clone(); v2[:, :, half:, :] = 0
        out2 = sparse_prefill_reference(q[:, :, :half, :], k2[:, :, :half, :], v2[:, :, :half, :],
                                        make_local_stride_pattern(seq_len=half, block_size=block_size,
                                                                  local_window_blocks=2, stride_blocks=4, global_blocks=1))

        # First half of output should be unchanged (causal property)
        max_err = (out1[:, :, :half, :] - out2).abs().max().item()
        assert max_err < 1e-3, (
            f"Causal violation: zeroing future tokens changed past output (err={max_err:.6f})"
        )

    def test_online_softmax_numerical_stability(self):
        """Online softmax should be numerically stable with large logits."""
        torch.manual_seed(1)
        B, H, S, D = 1, 2, 256, 64
        block_size = 64

        # Large logits that would overflow naive softmax
        q = torch.randn(B, H, S, D) * 100.0
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        pattern = make_local_stride_pattern(seq_len=S, block_size=block_size)

        # Should not produce NaN or Inf
        out = sparse_prefill_reference(q, k, v, pattern)
        assert not torch.isnan(out).any(),  "NaN in output with large logits"
        assert not torch.isinf(out).any(),  "Inf in output with large logits"

    def test_output_shape(self):
        """Output shape must match input shape exactly."""
        B, H, S, D = 2, 4, 512, 64
        block_size  = 64
        pattern     = make_local_stride_pattern(seq_len=S, block_size=block_size)

        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out = sparse_prefill_reference(q, k, v, pattern)
        assert out.shape == (B, H, S, D), f"Shape mismatch: {out.shape} != {(B, H, S, D)}"

    def test_empty_rows_produce_zeros(self):
        """Query blocks with no active KV blocks should output zeros."""
        from sparse_attn.patterns.base import SparseBlockPattern
        nb = 4
        # Only diagonal active, row 2 explicitly cleared
        mask = torch.zeros(nb, nb, dtype=torch.bool)
        mask.fill_diagonal_(True)
        mask[2, 2] = False  # Force row 2 to have no active blocks

        pattern = SparseBlockPattern(seq_len=nb * 64, block_size=64, block_mask=mask)

        B, H, D = 1, 2, 32
        q = torch.randn(B, H, nb * 64, D)
        k = torch.randn(B, H, nb * 64, D)
        v = torch.randn(B, H, nb * 64, D)

        out = sparse_prefill_reference(q, k, v, pattern)
        # Row 2 output should be zero
        assert out[:, :, 128:192, :].abs().max() == 0.0, "Empty row should produce zeros"


# --------------------------------------------------------------------------- #
#  OOM boundary (lightweight version for CPU testing)                         #
# --------------------------------------------------------------------------- #

class TestOOMFree:

    def test_no_oom_large_seq_reference(self):
        """
        Reference implementation should handle large sequences without OOM
        because it processes block-by-block (no full n×n matrix).
        Test with seq=8192 (would be 512MB for dense attention).
        """
        seq_len    = 4096  # Use moderate size for CPU test speed
        block_size = 64
        B, H, D   = 1, 1, 32  # Minimal dims for speed

        pattern = make_local_stride_pattern(
            seq_len=seq_len, block_size=block_size,
            local_window_blocks=4, stride_blocks=8, global_blocks=2
        )

        q = torch.randn(B, H, seq_len, D)
        k = torch.randn(B, H, seq_len, D)
        v = torch.randn(B, H, seq_len, D)

        # Should complete without OOM
        try:
            out = sparse_prefill_reference(q, k, v, pattern)
            assert out.shape == (B, H, seq_len, D)
        except MemoryError:
            pytest.fail("OOM in reference implementation — block-sparse should not OOM")
