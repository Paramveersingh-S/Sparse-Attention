"""
sparse_attn/patterns/heterogeneous.py
=======================================
Heterogeneous Local-Stride pattern (the primary production pattern).

Combines three components:
  1. Local Window  — attend to nearest w key blocks (sliding diagonal band)
  2. Global Stride — attend to every s-th block (global landmark tokens)
  3. Prefix Blocks — always attend to first g blocks (BOS, system prompt, etc.)

Rule: Block (I, J) is ACTIVE if ANY of:
  - J == I                           (diagonal, always causal)
  - I - window_blocks < J < I        (local window, causal)
  - J % stride_blocks == 0 and J<=I  (stride landmark)
  - J < global_blocks                (prefix, always attended)

Visual (w=2, s=4, g=1, n_blocks=8):
  Block:  0  1  2  3  4  5  6  7
  Row 0: [■] .  .  .  .  .  .  .
  Row 1: [■][■] .  .  .  .  .  .
  Row 2: [■][■][■] .  .  .  .  .
  Row 3: [■] . [■][■] .  .  .  .   ← stride hit col 0
  Row 4: [■] .  . [■][■] .  .  .
  Row 5: [■] .  .  . [■][■] .  .
  Row 6: [■] .  .  .  . [■][■] .
  Row 7: [■] .  .  .  . [■][■][■]
  ■ = attended (local or stride or prefix)
"""

from __future__ import annotations

import torch
from sparse_attn.patterns.base import SparseBlockPattern


class HeterogeneousPattern(SparseBlockPattern):
    """
    Heterogeneous local+stride+prefix sparse attention pattern.

    This is the core pattern used in production. It balances:
    - Local coherence  (local window)
    - Global context   (stride landmarks)
    - Special tokens   (prefix always attended)

    Parameters
    ----------
    seq_len           : Full sequence length.
    block_size        : Block size (power of 2, ≥16). Default 64.
    local_window_blocks : Number of past blocks in local window. Default 8.
    stride_blocks     : Attend to every stride_blocks-th column. Default 16.
    global_blocks     : Always attend to the first global_blocks blocks. Default 4.
    """

    def __init__(
        self,
        seq_len: int,
        block_size: int = 64,
        local_window_blocks: int = 8,
        stride_blocks: int = 16,
        global_blocks: int = 4,
    ):
        nb = seq_len // block_size
        mask = torch.zeros(nb, nb, dtype=torch.bool)

        for row in range(nb):
            for col in range(row + 1):  # causal: col <= row only
                is_local  = col >= row - local_window_blocks
                is_stride = (stride_blocks > 0) and (col % stride_blocks == 0)
                is_prefix = col < global_blocks
                is_diag   = col == row

                if is_local or is_stride or is_prefix or is_diag:
                    mask[row, col] = True

        super().__init__(seq_len=seq_len, block_size=block_size, block_mask=mask)
        self.local_window_blocks = local_window_blocks
        self.stride_blocks       = stride_blocks
        self.global_blocks       = global_blocks

    def __repr__(self) -> str:
        return (
            f"HeterogeneousPattern("
            f"seq_len={self.seq_len}, "
            f"block_size={self.block_size}, "
            f"local_window_blocks={self.local_window_blocks}, "
            f"stride_blocks={self.stride_blocks}, "
            f"global_blocks={self.global_blocks}, "
            f"sparsity={self.sparsity:.1%})"
        )


def make_local_stride_pattern(
    seq_len: int,
    block_size: int = 64,
    local_window_blocks: int = 8,
    stride_blocks: int = 16,
    global_blocks: int = 4,
) -> HeterogeneousPattern:
    """
    Factory function: create the heterogeneous local-stride pattern.

    This is the primary entry point for creating sparse attention patterns.

    Parameters
    ----------
    seq_len             : Full sequence length (must be divisible by block_size).
    block_size          : Block granularity. Default 64 tokens/block.
    local_window_blocks : Sliding window size in blocks. Default 8 (= 512 tokens).
    stride_blocks       : Global stride interval in blocks. Default 16 (= 1024 tokens).
    global_blocks       : Always-attended prefix blocks. Default 4 (= 256 tokens).

    Returns
    -------
    HeterogeneousPattern with all three components active.

    Example
    -------
    >>> pattern = make_local_stride_pattern(seq_len=32768, block_size=64)
    >>> print(pattern)
    HeterogeneousPattern(seq_len=32768, block_size=64, sparsity=94.3%)

    >>> pattern = make_local_stride_pattern(
    ...     seq_len=131072, block_size=64,
    ...     local_window_blocks=16, stride_blocks=32, global_blocks=8
    ... )
    """
    return HeterogeneousPattern(
        seq_len=seq_len,
        block_size=block_size,
        local_window_blocks=local_window_blocks,
        stride_blocks=stride_blocks,
        global_blocks=global_blocks,
    )
