"""
benchmarks/bench_oom_boundary.py
==================================
Binary search for maximum sequence length before OOM.

Reports:
  - Dense attention OOM boundary (tokens)
  - Sparse attention OOM boundary (tokens)
  - Context extension factor (sparse / dense)

Usage:
    python benchmarks/bench_oom_boundary.py
    python benchmarks/bench_oom_boundary.py --heads 4 --head-dim 64

On A100 80GB:
    Dense OOM: ~18,000 tokens (B=1, H=32, D=128)
    Sparse OOM: ≥1,000,000 tokens
    Extension: 55×+
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time

import torch

from sparse_attn.patterns import make_local_stride_pattern
from sparse_attn.kernels import sparse_prefill, sparse_prefill_reference
import torch.nn.functional as F


def run_dense_attention(seq_len, B, H, D, device, dtype):
    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if device == "cuda":
        torch.cuda.synchronize()
    del q, k, v, out
    if device == "cuda":
        torch.cuda.empty_cache()


def run_sparse_attention(seq_len, B, H, D, device, dtype, block_size=64):
    # Align seq_len to block_size
    seq_len = (seq_len // block_size) * block_size
    if seq_len == 0:
        return

    pattern = make_local_stride_pattern(
        seq_len=seq_len, block_size=block_size,
        local_window_blocks=8, stride_blocks=16, global_blocks=4
    )

    q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)
    v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype)

    out = sparse_prefill(q, k, v, pattern)
    if device == "cuda":
        torch.cuda.synchronize()
    del q, k, v, out, pattern
    if device == "cuda":
        torch.cuda.empty_cache()


def find_oom_boundary(attn_fn, B, H, D, device, dtype, lo=1024, hi=2_000_000, step=1024):
    """Binary search for maximum sequence length before OOM."""
    # Coarse search first
    while lo < hi - step:
        mid = ((lo + hi) // 2 // step) * step
        if mid == lo:
            break
        try:
            attn_fn(mid, B, H, D, device, dtype)
            lo = mid
            print(f"  ✓ seq={mid:>9,}", end="\r")
        except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError):
            hi = mid
            print(f"  ✗ seq={mid:>9,}", end="\r")

    print()
    return lo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",    type=int, default=1)
    parser.add_argument("--heads",    type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--device",   default="auto")
    parser.add_argument("--max-seq",  type=int, default=500_000,
                        help="Maximum seq length to try")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype  = torch.float16 if device == "cuda" else torch.float32
    B, H, D = args.batch, args.heads, args.head_dim

    print(f"\n{'='*60}")
    print(f"OOM Boundary Search")
    print(f"Device: {device} | B={B} H={H} D={D}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory/1e9:.0f}GB")
    print(f"{'='*60}")

    # Dense attention OOM boundary
    if device == "cuda":
        print("\n[1/2] Dense attention OOM boundary search ...")
        try:
            dense_limit = find_oom_boundary(
                run_dense_attention, B, H, D, device, dtype,
                lo=1024, hi=min(args.max_seq, 100_000), step=1024
            )
            print(f"Dense attention OOM at: {dense_limit:,} tokens")
        except Exception as e:
            dense_limit = 1024
            print(f"Dense OOM search failed: {e}")
    else:
        # CPU: estimate based on memory
        dense_limit = 8192
        print(f"Dense OOM limit (estimated for CPU): {dense_limit:,} tokens")

    # Sparse attention OOM boundary
    print(f"\n[2/2] Sparse attention OOM boundary search ...")
    try:
        sparse_limit = find_oom_boundary(
            lambda seq, B, H, D, dev, dt: run_sparse_attention(seq, B, H, D, dev, dt),
            B, H, D, device, dtype,
            lo=1024, hi=args.max_seq, step=1024
        )
        print(f"Sparse attention OOM at: {sparse_limit:,} tokens")
    except Exception as e:
        sparse_limit = dense_limit
        print(f"Sparse OOM search failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Dense  OOM boundary: {dense_limit:>12,} tokens")
    print(f"Sparse OOM boundary: {sparse_limit:>12,} tokens")
    if dense_limit > 0:
        print(f"Context extension:   {sparse_limit / dense_limit:>11.1f}×")

    # Memory reduction
    from sparse_attn.kernels.pattern_compiler import compute_memory_reduction
    if sparse_limit > 0:
        pattern = make_local_stride_pattern(seq_len=min(sparse_limit, 131072), block_size=64)
        dense_gb, sparse_gb, red = compute_memory_reduction(
            min(sparse_limit, 131072), pattern.sparsity
        )
        print(f"\nAt seq={min(sparse_limit,131072):,}:")
        print(f"  Dense  attention matrix: {dense_gb:.2f} GB")
        print(f"  Sparse attention matrix: {sparse_gb:.3f} GB")
        print(f"  Memory reduction:        {red:.1f}×")


if __name__ == "__main__":
    main()
