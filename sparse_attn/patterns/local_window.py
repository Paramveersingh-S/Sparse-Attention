"""
sparse_attn/patterns/local_window.py
=====================================
Causal local-window sparse attention pattern.

Each query block I attends to the nearest w key blocks:
  Active(I, J) = True  iff  I - w <= J <= I  (causal)

Visual (w=3, n_blocks=8):
  Block:  0  1  2  3  4  5  6  7
  Row 0: [■] .  .  .  .  .  .  .
  Row 1: [■][■] .  .  .  .  .  .
  Row 2: [■][■][■] .  .  .  .  .
  Row 3:  . [■][■][■] .  .  .  .
  Row 4:  .  . [■][■][■] .  .  .
  Row 5:  .  .  . [■][■][■] .  .
  Row 6:  .  .  .  . [■][■][■] .
  Row 7:  .  .  .  .  . [■][■][■]
  ■ = attended, . = skipped

FLOP reduction: (w+1)/n_blocks per row (sliding window)
"""

from __future__ import annotations

import torch
from sparse_attn.patterns.base import SparseBlockPattern


class LocalWindowPattern(SparseBlockPattern):
    """
    Causal local-window sparse attention pattern.

    Parameters
    ----------
    seq_len      : Full sequence length.
    block_size   : Block size (power of 2, ≥16).
    window_blocks: Number of past blocks each query block attends to
                   (including itself). E.g., window_blocks=8 → 8×block_size token window.
    """

    def __init__(
        self,
        seq_len: int,
        block_size: int = 64,
        window_blocks: int = 8,
    ):
        nb = seq_len // block_size
        mask = torch.zeros(nb, nb, dtype=torch.bool)

        for row in range(nb):
            # Always attend to diagonal (self)
            col_start = max(0, row - window_blocks + 1)
            col_end   = row + 1  # inclusive → exclusive
            mask[row, col_start:col_end] = True

        super().__init__(seq_len=seq_len, block_size=block_size, block_mask=mask)
        self.window_blocks = window_blocks

    def __repr__(self) -> str:
        return (
            f"LocalWindowPattern("
            f"seq_len={self.seq_len}, "
            f"block_size={self.block_size}, "
            f"window_blocks={self.window_blocks}, "
            f"sparsity={self.sparsity:.1%})"
        )
