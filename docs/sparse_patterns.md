# Sparse Attention Patterns — Visual Guide

## Overview

This guide explains the three sparse attention patterns implemented in this library,
with ASCII art diagrams, sparsity formulas, and guidance on choosing parameters.

---

## Pattern 1: Local Window (`LocalWindowPattern`)

Each query token attends only to the nearest **w** past tokens (local window).

```
Pattern: Local Window (w=3 blocks, causal)
==========================================
Block:  0  1  2  3  4  5  6  7
Row 0: [■] .  .  .  .  .  .  .
Row 1: [■][■] .  .  .  .  .  .
Row 2: [■][■][■] .  .  .  .  .
Row 3:  . [■][■][■] .  .  .  .
Row 4:  .  . [■][■][■] .  .  .
Row 5:  .  .  . [■][■][■] .  .
Row 6:  .  .  .  . [■][■][■] .
Row 7:  .  .  .  .  . [■][■][■]
■ = attended block, . = skipped block

Active blocks per row: min(w+1, row+1) → converges to w+1 = 4 per row
FLOP reduction: (w+1)/n = 4/8 = 50% active → 50% FLOP reduction (n=8, w=3)
```

**Best for**: Local language modeling tasks, short-range dependencies.

**Parameters**:
- `window_blocks=8`: 8 blocks × 64 tokens = 512-token local window
- `window_blocks=16`: 1024-token local window

**Sparsity formula**:
```
Sparsity ≈ 1 - (w+1)/n   (for large n)
At n=2048, w=8: sparsity ≈ 1 - 9/2048 ≈ 99.6%
```

---

## Pattern 2: Strided Global (`StridedPattern`)

Each token attends to every **s**-th block in the past (global landmark tokens).

```
Pattern: Stride-4 Global (s=4)
================================
Block:  0  1  2  3  4  5  6  7
Row 0: [■] .  .  .  .  .  .  .
Row 1: [■] .  .  .  .  .  .  .
Row 2: [■] .  .  .  .  .  .  .
Row 3: [■] .  .  .  .  .  .  .
Row 4: [■] .  .  . [■] .  .  .
Row 5: [■] .  .  . [■] .  .  .
Row 6: [■] .  .  . [■] .  .  .
Row 7: [■] .  .  . [■] .  .  .
■ = attended (stride column), . = skipped
```

**Best for**: Global context aggregation, document-level understanding.

**Parameters**:
- `stride_blocks=16`: attend to every 1024th token landmark
- `stride_blocks=8`: attend to every 512th token landmark

---

## Pattern 3: Heterogeneous Local + Stride (Primary Production Pattern)

The recommended pattern for long-context LLMs. Combines:
- **Local window** (short-range coherence)
- **Global stride** (long-range context)
- **Prefix blocks** (special tokens, system prompt)

```
Pattern: Local (w=2) + Strided (s=4) + Prefix (g=1)
=====================================================
Block:  0  1  2  3  4  5  6  7
Row 0: [■] .  .  .  .  .  .  .
Row 1: [■][■] .  .  .  .  .  .
Row 2: [■][■][■] .  .  .  .  .
Row 3: [■] . [■][■] .  .  .  .   ← stride hit: col 0
Row 4: [■] .  . [■][■] .  .  .
Row 5: [■] .  .  . [■][■] .  .
Row 6: [■] .  .  .  . [■][■] .
Row 7: [■] .  .  .  . [■][■][■]  ← both local and diagonal
■ = attended (local or stride or prefix)

Sparsity: ~75% for this small example (full sequences: 90-99%)
```

**Rule set (block I, J active if ANY true)**:
1. `J == I` (diagonal — always, causal self)
2. `I - w ≤ J < I` (local window)
3. `J % s == 0 AND J ≤ I` (stride landmark)
4. `J < g` (prefix — always attend to first g blocks)

**Recommended parameters for common models**:

| Model Size | Context | `block_size` | `local_window_blocks` | `stride_blocks` | `global_blocks` | Sparsity |
|---|---|---|---|---|---|---|
| 7B  | 32K  | 64  | 8  | 16 | 4  | ~94% |
| 7B  | 64K  | 64  | 8  | 32 | 4  | ~97% |
| 70B | 128K | 128 | 8  | 32 | 8  | ~98% |
| Any | 1M   | 128 | 4  | 64 | 4  | ~99.5% |

---

## Pattern 4: Custom (`CustomPattern`)

Define arbitrary patterns using Python lambda predicates:

```python
from sparse_attn.patterns import CustomPattern

pattern = CustomPattern(seq_len=32768, block_size=64)

# Define rules (OR-combined)
pattern.add_rule("local_window",  lambda I, J: abs(I - J) <= 8 and J <= I)
pattern.add_rule("global_stride", lambda I, J: J % 16 == 0 and J <= I)
pattern.add_rule("prefix",        lambda I, J: J < 4)

compiled = pattern.compile()
print(compiled.ascii_art())
```

**Example: BigBird-style random attention**:
```python
import random
random.seed(42)

random_cols = set(random.sample(range(nb), k=3))
pattern.add_rule("random_global", lambda I, J: J in random_cols and J <= I)
```

**Example: Sliding window + global CLS token**:
```python
pattern.add_rule("cls",    lambda I, J: J == 0)         # CLS always attended
pattern.add_rule("window", lambda I, J: abs(I-J) <= 4 and J <= I)
```

---

## Sparsity → Memory → Context Length

| Sparsity | Memory vs Dense | Context at Dense OOM (A100) |
|---|---|---|
| 75%  | 4×  less  | ~72K tokens  |
| 90%  | 10× less  | ~180K tokens |
| 95%  | 20× less  | ~360K tokens |
| 99%  | 100× less | ~1.8M tokens |
| 99.5%| 200× less | ~3.6M tokens |

*Dense OOM on A100 80GB: ~18K tokens (B=1, H=32, D=128)*

---

## Choosing Parameters

```
Goal: seq_len = 128K, OOM-free on A100 80GB

Dense attention matrix: 128K² × 2B = 32 GB → OOM

With sparsity=97% (local_w=8, stride=32):
  Attention matrix: 32GB × 0.03 = 960 MB → fits

With sparsity=99% (local_w=4, stride=64):
  Attention matrix: 32GB × 0.01 = 320 MB → fits with headroom
```

**Rule of thumb**: `local_window_blocks=8, stride_blocks=16` gives ~94% sparsity at 32K context.
Double `stride_blocks` for each doubling of context length.
