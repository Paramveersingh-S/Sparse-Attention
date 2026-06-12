"""
tests/test_pattern_correctness.py
===================================
Tests for pattern representation, sparsity ratios, and custom rules.
"""

import pytest
import torch
from sparse_attn.patterns import (
    SparseBlockPattern,
    LocalWindowPattern,
    StridedPattern,
    HeterogeneousPattern,
    CustomPattern,
    make_local_stride_pattern,
)


# --------------------------------------------------------------------------- #
#  SparseBlockPattern base tests                                              #
# --------------------------------------------------------------------------- #

class TestSparseBlockPattern:

    def test_basic_construction(self):
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask.fill_diagonal_(True)
        p = SparseBlockPattern(seq_len=512, block_size=64, block_mask=mask)
        assert p.num_blocks == 8
        assert p.num_active_blocks == 8

    def test_invalid_seq_len(self):
        with pytest.raises(ValueError, match="divisible"):
            SparseBlockPattern(seq_len=500, block_size=64)

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="power of 2"):
            SparseBlockPattern(seq_len=528, block_size=33)

    def test_block_size_too_small(self):
        with pytest.raises(ValueError, match="power of 2"):
            SparseBlockPattern(seq_len=512, block_size=8)

    def test_sparsity_zero(self):
        # All-dense mask: sparsity = 0
        nb = 8
        mask = torch.ones(nb, nb, dtype=torch.bool)
        p = SparseBlockPattern(seq_len=nb * 64, block_size=64, block_mask=mask)
        assert p.sparsity == 0.0
        assert p.density  == 1.0

    def test_sparsity_identity(self):
        # Only diagonal: sparsity = (n²-n)/n² = (n-1)/n
        nb = 8
        mask = torch.zeros(nb, nb, dtype=torch.bool)
        mask.fill_diagonal_(True)
        p = SparseBlockPattern(seq_len=nb * 64, block_size=64, block_mask=mask)
        expected = (nb * nb - nb) / (nb * nb)
        assert abs(p.sparsity - expected) < 1e-6

    def test_get_active_cols(self):
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[2, 0] = True
        mask[2, 2] = True
        p = SparseBlockPattern(seq_len=256, block_size=64, block_mask=mask)
        assert set(p.get_active_cols(2)) == {0, 2}
        assert p.get_active_cols(0) == []

    def test_to_kernel_format(self):
        nb = 4
        mask = torch.zeros(nb, nb, dtype=torch.bool)
        for i in range(nb):
            mask[i, i] = True  # Only diagonal
        p = SparseBlockPattern(seq_len=nb * 64, block_size=64, block_mask=mask)
        kf = p.to_kernel_format()

        assert "col_indices" in kf
        assert "row_ptrs"    in kf
        assert "causal_mask" in kf

        # All active blocks are diagonal → all causal flags True
        assert kf["causal_mask"].all()
        assert len(kf["col_indices"]) == nb
        assert len(kf["row_ptrs"])    == nb + 1

    def test_ascii_art(self):
        p = LocalWindowPattern(seq_len=512, block_size=64, window_blocks=2)
        art = p.ascii_art(max_blocks=8)
        assert "[■]" in art
        assert "Sparsity" in art


# --------------------------------------------------------------------------- #
#  LocalWindowPattern tests                                                   #
# --------------------------------------------------------------------------- #

class TestLocalWindowPattern:

    def test_window_1(self):
        """Window=1 → only diagonal active."""
        p = LocalWindowPattern(seq_len=512, block_size=64, window_blocks=1)
        for row in range(p.num_blocks):
            assert p.get_active_cols(row) == [row]

    def test_window_full(self):
        """Window=num_blocks → lower triangular (causal full attention)."""
        nb = 8
        p = LocalWindowPattern(seq_len=nb * 64, block_size=64, window_blocks=nb)
        for row in range(nb):
            expected = list(range(row + 1))
            assert p.get_active_cols(row) == expected

    def test_sliding_window(self):
        """Window=3 → each row has at most 3 active blocks."""
        p = LocalWindowPattern(seq_len=1024, block_size=64, window_blocks=3)
        for row in range(p.num_blocks):
            cols = p.get_active_cols(row)
            assert len(cols) <= 3
            assert row in cols  # diagonal always active

    def test_no_future_blocks(self):
        """No active column should be > row index."""
        p = LocalWindowPattern(seq_len=1024, block_size=64, window_blocks=4)
        for row in range(p.num_blocks):
            for col in p.get_active_cols(row):
                assert col <= row, f"Future block: row={row}, col={col}"


# --------------------------------------------------------------------------- #
#  StridedPattern tests                                                       #
# --------------------------------------------------------------------------- #

class TestStridedPattern:

    def test_stride_4(self):
        p = StridedPattern(seq_len=1024, block_size=64, stride_blocks=4)
        for row in range(p.num_blocks):
            cols = p.get_active_cols(row)
            # All active cols should be multiples of stride_blocks or == row
            for col in cols:
                assert col % 4 == 0 or col == row, \
                    f"Non-stride col {col} in row {row}"

    def test_diagonal_always_active(self):
        p = StridedPattern(seq_len=1024, block_size=64, stride_blocks=8)
        for row in range(p.num_blocks):
            assert row in p.get_active_cols(row)

    def test_causal(self):
        p = StridedPattern(seq_len=512, block_size=64, stride_blocks=4)
        for row in range(p.num_blocks):
            for col in p.get_active_cols(row):
                assert col <= row


# --------------------------------------------------------------------------- #
#  HeterogeneousPattern tests                                                 #
# --------------------------------------------------------------------------- #

class TestHeterogeneousPattern:

    def test_construction(self):
        p = HeterogeneousPattern(
            seq_len=2048, block_size=64,
            local_window_blocks=4, stride_blocks=8, global_blocks=2
        )
        assert p.num_blocks == 32
        assert 0.0 < p.sparsity < 1.0

    def test_local_window_active(self):
        """Local window blocks should be active."""
        p = HeterogeneousPattern(
            seq_len=1024, block_size=64,
            local_window_blocks=3, stride_blocks=100, global_blocks=0
        )
        # Row 10 should have cols 7, 8, 9, 10 active (window=3 + diagonal)
        cols = set(p.get_active_cols(10))
        for c in [8, 9, 10]:  # last 3 + diagonal
            assert c in cols, f"Local block {c} not in row 10"

    def test_stride_active(self):
        """Stride blocks should be active."""
        stride = 4
        p = HeterogeneousPattern(
            seq_len=1024, block_size=64,
            local_window_blocks=1, stride_blocks=stride, global_blocks=0
        )
        for row in range(p.num_blocks):
            cols = set(p.get_active_cols(row))
            for c in range(0, row, stride):
                assert c in cols, f"Stride col {c} not active in row {row}"

    def test_global_prefix_active(self):
        """Global prefix blocks should always be active."""
        global_blocks = 3
        p = HeterogeneousPattern(
            seq_len=2048, block_size=64,
            local_window_blocks=2, stride_blocks=100, global_blocks=global_blocks
        )
        for row in range(global_blocks, p.num_blocks):  # skip first few rows
            cols = set(p.get_active_cols(row))
            for c in range(global_blocks):
                assert c in cols, f"Global block {c} not in row {row}"

    def test_causal(self):
        p = make_local_stride_pattern(seq_len=2048, block_size=64)
        p.validate_causal()  # Should not raise

    def test_diagonal(self):
        p = make_local_stride_pattern(seq_len=2048, block_size=64)
        p.validate_diagonal()  # Should not raise


# --------------------------------------------------------------------------- #
#  CustomPattern tests                                                        #
# --------------------------------------------------------------------------- #

class TestCustomPattern:

    def test_basic_custom(self):
        cp = CustomPattern(seq_len=512, block_size=64)
        cp.add_rule("diag", lambda I, J: I == J)
        compiled = cp.compile()
        # Only diagonal should be active
        for row in range(compiled.num_blocks):
            assert compiled.get_active_cols(row) == [row]

    def test_or_combination(self):
        cp = CustomPattern(seq_len=512, block_size=64)
        cp.add_rule("diag",   lambda I, J: I == J)
        cp.add_rule("prefix", lambda I, J: J == 0)
        compiled = cp.compile()
        # Row 5 should have col 0 and col 5
        cols = set(compiled.get_active_cols(5))
        assert 0 in cols
        assert 5 in cols

    def test_duplicate_rule_error(self):
        cp = CustomPattern(seq_len=512, block_size=64)
        cp.add_rule("rule_a", lambda I, J: True)
        with pytest.raises(ValueError, match="already exists"):
            cp.add_rule("rule_a", lambda I, J: False)

    def test_no_rules_error(self):
        cp = CustomPattern(seq_len=512, block_size=64)
        with pytest.raises(RuntimeError, match="No rules"):
            cp.compile()

    def test_enforce_causal(self):
        """enforce_causal=True should eliminate all J > I blocks."""
        cp = CustomPattern(seq_len=512, block_size=64)
        cp.add_rule("all", lambda I, J: True)
        compiled = cp.compile(enforce_causal=True)
        compiled.validate_causal()  # Should not raise

    def test_method_chaining(self):
        cp = CustomPattern(seq_len=512, block_size=64)
        result = (cp
                  .add_rule("a", lambda I, J: I == J)
                  .add_rule("b", lambda I, J: J == 0))
        assert result is cp

    def test_remove_rule(self):
        cp = CustomPattern(seq_len=512, block_size=64)
        cp.add_rule("a", lambda I, J: True)
        cp.remove_rule("a")
        assert "a" not in cp.list_rules()

    def test_preview(self):
        cp = CustomPattern(seq_len=512, block_size=64)
        cp.add_rule("diag", lambda I, J: I == J)
        preview = cp.preview(max_blocks=4)
        assert "[■]" in preview


# --------------------------------------------------------------------------- #
#  CSR format integrity tests                                                 #
# --------------------------------------------------------------------------- #

class TestCSRFormat:

    def test_row_ptrs_monotone(self):
        p = make_local_stride_pattern(seq_len=1024, block_size=64)
        kf = p.to_kernel_format()
        rp = kf["row_ptrs"]
        for i in range(len(rp) - 1):
            assert rp[i] <= rp[i + 1], f"row_ptrs not monotone at i={i}"

    def test_col_indices_in_bounds(self):
        p = make_local_stride_pattern(seq_len=1024, block_size=64)
        kf = p.to_kernel_format()
        nb = p.num_blocks
        assert (kf["col_indices"] >= 0).all()
        assert (kf["col_indices"] <  nb).all()

    def test_causal_mask_diagonal(self):
        """All diagonal blocks must be marked as causal."""
        p = make_local_stride_pattern(seq_len=1024, block_size=64)
        kf = p.to_kernel_format()
        nb   = p.num_blocks
        rp   = kf["row_ptrs"]
        ci   = kf["col_indices"]
        cm   = kf["causal_mask"]

        for row in range(nb):
            for idx in range(rp[row].item(), rp[row + 1].item()):
                if ci[idx].item() == row:  # diagonal block
                    assert cm[idx].item(), f"Diagonal block at row={row} not marked causal"

    def test_total_active_blocks(self):
        p = make_local_stride_pattern(seq_len=1024, block_size=64)
        kf = p.to_kernel_format()
        assert len(kf["col_indices"]) == p.num_active_blocks
