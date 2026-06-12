"""
sparse_attn/patterns/custom.py
================================
User-defined sparse attention pattern via lambda predicate rules.

Allows defining arbitrary patterns in <10 lines of Python:

    from sparse_attn.patterns import CustomPattern

    pattern = CustomPattern(seq_len=32768, block_size=64)
    pattern.add_rule("local",  lambda I, J: abs(I - J) <= 8 and J <= I)
    pattern.add_rule("stride", lambda I, J: J % 16 == 0 and J <= I)
    pattern.add_rule("prefix", lambda I, J: J < 4)
    compiled = pattern.compile()

Rules are OR-combined: a block is active if ANY rule returns True.
The causal constraint J <= I is NOT automatically enforced — users must add
it to their rules or call .enforce_causal() after compile().
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from sparse_attn.patterns.base import SparseBlockPattern


# Type alias for a rule predicate: (query_block_idx, key_block_idx) → bool
RuleFn = Callable[[int, int], bool]


class CustomPattern:
    """
    User-defined sparse attention pattern via lambda predicate rules.

    Rules are evaluated for all (I, J) block pairs and OR-combined.
    After adding all rules, call .compile() to get a SparseBlockPattern.

    Parameters
    ----------
    seq_len    : Full sequence length.
    block_size : Block granularity (power of 2, ≥16).

    Example
    -------
    >>> pattern = CustomPattern(seq_len=32768, block_size=64)
    >>> pattern.add_rule("local_window",  lambda I, J: abs(I - J) <= 8 and J <= I)
    >>> pattern.add_rule("global_stride", lambda I, J: J % 16 == 0 and J <= I)
    >>> pattern.add_rule("prefix",        lambda I, J: J < 4)
    >>> compiled = pattern.compile()
    >>> print(compiled)
    """

    def __init__(self, seq_len: int, block_size: int = 64):
        self.seq_len    = seq_len
        self.block_size = block_size
        self._rules: Dict[str, RuleFn] = {}

    @property
    def num_blocks(self) -> int:
        return self.seq_len // self.block_size

    def add_rule(self, name: str, rule_fn: RuleFn) -> "CustomPattern":
        """
        Add a named predicate rule.

        Parameters
        ----------
        name    : Descriptive name (used in repr and debugging).
        rule_fn : Callable(I: int, J: int) → bool.
                  I = query block index, J = key block index.
                  Return True to ATTEND to block (I, J).

        Returns self for method chaining.
        """
        if name in self._rules:
            raise ValueError(f"Rule '{name}' already exists. Use remove_rule() first.")
        self._rules[name] = rule_fn
        return self

    def remove_rule(self, name: str) -> "CustomPattern":
        """Remove a named rule."""
        if name not in self._rules:
            raise KeyError(f"Rule '{name}' not found.")
        del self._rules[name]
        return self

    def list_rules(self) -> List[str]:
        """Return names of all registered rules."""
        return list(self._rules.keys())

    def compile(self, enforce_causal: bool = False) -> SparseBlockPattern:
        """
        Evaluate all rules for every (I, J) block pair and return a
        SparseBlockPattern with the resulting block mask.

        Parameters
        ----------
        enforce_causal : If True, automatically mask out all J > I blocks
                         (i.e., enforce causal constraint regardless of rules).

        Returns
        -------
        SparseBlockPattern with block_mask reflecting the union of all rules.
        """
        if not self._rules:
            raise RuntimeError("No rules defined. Call add_rule() before compile().")

        nb = self.num_blocks
        mask = torch.zeros(nb, nb, dtype=torch.bool)

        for row in range(nb):
            for col in range(nb):
                if enforce_causal and col > row:
                    continue  # Hard causal constraint
                for rule_fn in self._rules.values():
                    if rule_fn(row, col):
                        mask[row, col] = True
                        break  # OR-combine: one True is enough

        return SparseBlockPattern(
            seq_len=self.seq_len,
            block_size=self.block_size,
            block_mask=mask,
        )

    def preview(self, max_blocks: int = 12) -> str:
        """Quick preview: compile and render ASCII art."""
        compiled = self.compile()
        return compiled.ascii_art(max_blocks=max_blocks)

    def __repr__(self) -> str:
        rules_str = ", ".join(self._rules.keys()) if self._rules else "none"
        return (
            f"CustomPattern("
            f"seq_len={self.seq_len}, "
            f"block_size={self.block_size}, "
            f"rules=[{rules_str}])"
        )
