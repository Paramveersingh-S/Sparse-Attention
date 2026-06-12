# SM Utilization Analysis: Chunked-KV Decode

## The Problem: SM Starvation in Sparse Decode

During autoregressive decoding (one new token at a time), the attention computation is:
- **Q**: shape [B, H, 1, D] — one query per batch element
- **K, V**: shape [B, H, S, D] — full KV cache

### Standard Sparse Decode Grid

```
Dense decode:  grid = (B × H) = 1 × 32 = 32 blocks
               → 32 / 108 SMs = 30% SM utilization (A100)

Naive sparse decode:
  Same grid: still 32 blocks
  But with sparse mask: each block skips 90% of KV blocks
  → Compute is 10× less, but SM util is STILL 30%
  → Terrible efficiency: 9× compute reduction but only within 30% of SMs
```

### Why Does This Happen?

The CUDA block grid for standard decode is:
```
dim3 grid(1, num_heads, batch_size);  // (1 query block, H heads, B batches)
```

With `B=1, H=32`: only 32 CUDA blocks → only 32 SMs active.
The other 76 SMs on an A100 sit idle.

---

## The Solution: Chunked-KV Approach

**Key insight**: Divide the KV sequence into fixed-size chunks.
Launch one CUDA block per **(batch, head, KV chunk)** triple.

```
KV sequence: 128K tokens
Chunk size:  512 tokens
Num chunks:  256

Active chunks (10% density): 256 × 0.10 = 26 per (batch, head)

Grid: B × H × active_chunks = 1 × 32 × 26 = 832 blocks
     → 832 / 108 = 7.7× oversubscription → 100% SM utilization!
```

### SM Utilization Formula

```python
def sm_utilization(batch, heads, seq_len, chunk_size, sparsity, num_sms):
    num_chunks    = seq_len // chunk_size
    active_chunks = int(num_chunks * (1 - sparsity))
    total_blocks  = batch * heads * active_chunks
    return min(total_blocks / num_sms, 1.0)  # capped at 100%

# Example configurations:
# B=1, H=32, S=128K, chunk=512, sparsity=0.90, num_sms=108
util = sm_utilization(1, 32, 131072, 512, 0.90, 108)
# num_chunks=256, active=26, total_blocks=832, util=min(7.7, 1.0)=100%
```

---

## Utilization Comparison Table

| Config | Standard Decode | Chunked-KV Decode | Improvement |
|---|---|---|---|
| B=1, H=8,  S=8K,   sparse=90% | 7%  | 100% | 14× |
| B=1, H=32, S=32K,  sparse=90% | 30% | 100% | 3.3× |
| B=1, H=32, S=128K, sparse=90% | 30% | 100% | 3.3× |
| B=4, H=32, S=64K,  sparse=95% | 100%| 100% | 1× (already saturated) |
| B=1, H=1,  S=128K, sparse=90% | <1% | 24%  | 24× |

*SM count: 108 (A100)*

---

## Algorithm: Two-Pass Chunked-KV Decode

```
┌─────────────────────────────────────────────────────┐
│  PASS 1: Partial Attention (one block per chunk)    │
│                                                     │
│  Grid: (num_active_chunks, H, B)                   │
│  Each block:                                        │
│    1. Check sparse mask → early exit if inactive   │
│    2. Load KV[chunk_start:chunk_end]               │
│    3. Compute scores: Q · Kᵀ                       │
│    4. Online softmax over chunk                     │
│    5. Write (O_partial, m_partial, l_partial)      │
│                                                     │
│  Output: partial_O [B, H, num_chunks, D]           │
│          partial_m [B, H, num_chunks]               │
│          partial_l [B, H, num_chunks]               │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│  PASS 2: Merge Partials (one block per B×H)         │
│                                                     │
│  Grid: (H, B, 1)                                   │
│  Each block:                                        │
│    For c in range(num_chunks):                      │
│      m_new = max(m_acc, m_c)                       │
│      l_new = exp(m_acc-m_new)*l_acc                │
│             + exp(m_c-m_new)*l_c                   │
│      O_new = (exp*l_acc*O_acc + exp*l_c*O_c)/l_new│
│    Write final O                                    │
└─────────────────────────────────────────────────────┘
```

### Online Softmax Merge (Mathematical Proof of Correctness)

Let two partial results from disjoint KV ranges be:
```
Partition 1: (O₁, m₁, l₁)  where  m₁ = max(scores₁), l₁ = Σexp(sᵢ-m₁)
Partition 2: (O₂, m₂, l₂)  similarly

Combined:
  m* = max(m₁, m₂)
  l* = exp(m₁-m*) × l₁ + exp(m₂-m*) × l₂
  O* = [exp(m₁-m*) × l₁ × O₁ + exp(m₂-m*) × l₂ × O₂] / l*

This is algebraically equivalent to the full softmax over all KV tokens. ✓
```

---

## Chunk Size Selection

| Chunk Size | SM Util (S=128K) | Cache Pressure | Recommendation |
|---|---|---|---|
| 128  | 100% | Low  | Best for high-sparsity (>95%) |
| 256  | 100% | Low  | Good balance |
| 512  | 100% | Med  | Default (good for 90% sparse) |
| 1024 | 100% | High | Use for large-D (D=256+) |
| 2048 | 100% | Very | Not recommended |

**Rule**: `chunk_size = 512` is the default. Decrease if memory-bound, increase if compute-bound.
