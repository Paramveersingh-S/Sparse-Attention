"""
sparse_attn/patterns/base.py
============================
Core SparseBlockPattern data structure.

Represents a sparse attention pattern as a block-level binary mask stored in
CSR (Compressed Sparse Row) format for efficient kernel consumption.

Attributes
----------
seq_len     : int — full sequence length (must be divisible by block_size)
block_size  : int — size of each square block (power of 2, ≥16)
num_blocks  : int — seq_len // block_size
block_mask  : torch.BoolTensor [num_blocks, num_blocks] — True = attended
diag_blocks : List[int] — block indices that are ON the main diagonal

CSR format (for kernels)
------------------------
col_indices : [num_active_blocks]  — which column blocks are active
row_ptrs    : [num_blocks + 1]     — CSR row pointers
causal_mask : [num_active_blocks]  — bool: is this the diagonal block?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch


@dataclass
class SparseBlockPattern:
    """
    Block-level sparse attention mask with CSR serialization.

    Parameters
    ----------
    seq_len    : Full sequence length. Must be divisible by block_size.
    block_size : Block granularity (power of 2, ≥16).
    block_mask : Boolean tensor [num_blocks, num_blocks]. True = block is attended.
                 Lower-triangular only (causal). If None, starts as all-False.
    """

    seq_len: int
    block_size: int
    block_mask: Optional[torch.BoolTensor] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    #  Post-init validation                                                #
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        if self.seq_len % self.block_size != 0:
            raise ValueError(
                f"seq_len ({self.seq_len}) must be divisible by block_size ({self.block_size})"
            )
        if self.block_size < 16 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(
                f"block_size must be a power of 2 and ≥ 16, got {self.block_size}"
            )
        nb = self.num_blocks
        if self.block_mask is None:
            self.block_mask = torch.zeros(nb, nb, dtype=torch.bool)
        elif self.block_mask.shape != (nb, nb):
            raise ValueError(
                f"block_mask shape {self.block_mask.shape} != expected ({nb}, {nb})"
            )

    # ------------------------------------------------------------------ #
    #  Derived properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def num_blocks(self) -> int:
        return self.seq_len // self.block_size

    @property
    def diag_blocks(self) -> List[int]:
        """Block indices on the main diagonal (always attended, causal)."""
        return list(range(self.num_blocks))

    @property
    def sparsity(self) -> float:
        """Fraction of blocks that are NOT attended (0 = dense, 1 = fully sparse)."""
        total = self.num_blocks * self.num_blocks
        active = int(self.block_mask.sum().item())
        return 1.0 - active / total

    @property
    def density(self) -> float:
        return 1.0 - self.sparsity

    @property
    def num_active_blocks(self) -> int:
        return int(self.block_mask.sum().item())

    # ------------------------------------------------------------------ #
    #  Block access helpers                                                #
    # ------------------------------------------------------------------ #

    def get_active_cols(self, row: int) -> List[int]:
        """Return list of active column block indices for a given query row block."""
        return self.block_mask[row].nonzero(as_tuple=False).squeeze(1).tolist()

    def is_active(self, row: int, col: int) -> bool:
        return bool(self.block_mask[row, col].item())

    # ------------------------------------------------------------------ #
    #  CSR serialization for kernel consumption                            #
    # ------------------------------------------------------------------ #

    def to_kernel_format(self) -> Dict[str, torch.Tensor]:
        """
        Convert block_mask to CSR-like format for kernel consumption.
        Caches the result to avoid Python overhead on repeated calls.

        Returns
        -------
        col_indices : LongTensor [num_active_blocks]
            Column block index for each active (row, col) pair, ordered row-major.
        row_ptrs    : LongTensor [num_blocks + 1]
            CSR row pointers: active blocks for row I are
            col_indices[row_ptrs[I] : row_ptrs[I+1]].
        causal_mask : BoolTensor [num_active_blocks]
            True if this active block is on the main diagonal (row == col).
        """
        if hasattr(self, "_cached_kf") and self._cached_kf is not None:
            return self._cached_kf

        nb = self.num_blocks
        col_indices_list: List[int] = []
        row_ptrs_list: List[int] = [0]
        causal_list: List[bool] = []

        for row in range(nb):
            active_cols = self.get_active_cols(row)
            for col in active_cols:
                col_indices_list.append(col)
                causal_list.append(col == row)
            row_ptrs_list.append(len(col_indices_list))

        col_indices = torch.tensor(col_indices_list, dtype=torch.long)
        row_ptrs    = torch.tensor(row_ptrs_list,    dtype=torch.long)
        causal_mask = torch.tensor(causal_list,      dtype=torch.bool)

        self._cached_kf = {
            "col_indices": col_indices,
            "row_ptrs":    row_ptrs,
            "causal_mask": causal_mask,
            "num_blocks":  nb,
            "block_size":  self.block_size,
        }
        return self._cached_kf

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def validate_causal(self) -> None:
        """
        Assert that no query block attends to a future key block.
        Raises AssertionError on violation.
        """
        for row in range(self.num_blocks):
            for col in self.get_active_cols(row):
                assert col <= row, (
                    f"Causal violation: query block {row} attends to "
                    f"future key block {col}"
                )

    def validate_diagonal(self) -> None:
        """Assert every block attends to itself (minimum causal self-attention)."""
        for row in range(self.num_blocks):
            assert self.is_active(row, row), (
                f"Block {row} does not attend to itself (diagonal must be active)"
            )

    def ascii_art(self, max_blocks: int = 16) -> str:
        """
        Render an ASCII art visualization of the block mask.
        Clips to max_blocks × max_blocks for readability.
        """
        nb = min(self.num_blocks, max_blocks)
        header = "Block: " + "  ".join(f"{i:2d}" for i in range(nb))
        lines = [header]
        for row in range(nb):
            row_str = f"Row {row:2d}: "
            for col in range(nb):
                if col > self.num_blocks - 1:
                    row_str += "  "
                elif self.block_mask[row, col].item():
                    row_str += "[■]"
                else:
                    row_str += " . "
            lines.append(row_str)
        lines.append(f"\nSparsity: {self.sparsity:.1%} | Active blocks: {self.num_active_blocks}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SparseBlockPattern("
            f"seq_len={self.seq_len}, "
            f"block_size={self.block_size}, "
            f"num_blocks={self.num_blocks}, "
            f"sparsity={self.sparsity:.1%}, "
            f"active_blocks={self.num_active_blocks})"
        )
