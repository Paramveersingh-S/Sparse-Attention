"""
sparse_attn/patterns/strided.py
================================
Strided (global landmark) sparse attention pattern.

Each query block I attends to every s-th block in the past:
  Active(I, J) = True  iff  J % stride_blocks == 0  and  J <= I

These "landmark" blocks provide global context without full O(n²) attention.

Visual (s=4, n_blocks=8):
  Block:  0  1  2  3  4  5  6  7
  Row 0: [■] .  .  .  .  .  .  .   (col 0 % 4 == 0)
  Row 1: [■] .  .  .  .  .  .  .
  Row 2: [■] .  .  .  .  .  .  .
  Row 3: [■] .  .  .  .  .  .  .
  Row 4: [■] .  .  . [■] .  .  .   (col 4 % 4 == 0)
  Row 5: [■] .  .  . [■] .  .  .
  Row 6: [■] .  .  . [■] .  .  .
  Row 7: [■] .  .  . [■] .  .  .
  ■ = attended, . = skipped
"""

from __future__ import annotations

import torch
from sparse_attn.patterns.base import SparseBlockPattern


class StridedPattern(SparseBlockPattern):
    """
    Strided (global landmark) sparse attention pattern.

    Parameters
    ----------
    seq_len       : Full sequence length.
    block_size    : Block size (power of 2, ≥16).
    stride_blocks : Attend to every stride_blocks-th column block.
                    E.g., stride_blocks=16 → attend to blocks 0, 16, 32, ...
    """

    def __init__(
        self,
        seq_len: int,
        block_size: int = 64,
        stride_blocks: int = 16,
    ):
        nb = seq_len // block_size
        mask = torch.zeros(nb, nb, dtype=torch.bool)

        for row in range(nb):
            # Diagonal always active
            mask[row, row] = True
            # Stride columns: J % stride_blocks == 0 and J <= row
            for col in range(0, row, stride_blocks):
                mask[row, col] = True

        super().__init__(seq_len=seq_len, block_size=block_size, block_mask=mask)
        self.stride_blocks = stride_blocks

    def __repr__(self) -> str:
        return (
            f"StridedPattern("
            f"seq_len={self.seq_len}, "
            f"block_size={self.block_size}, "
            f"stride_blocks={self.stride_blocks}, "
            f"sparsity={self.sparsity:.1%})"
        )
