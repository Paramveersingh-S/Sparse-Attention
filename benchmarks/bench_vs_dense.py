"""
benchmarks/bench_vs_dense.py
==============================
Benchmark sparse attention vs dense PyTorch SDPA across multiple configs.

Reports:
  - Speedup vs dense SDPA
  - Memory usage (GB) vs dense
  - SM utilization (%)
  - Correctness: max error on attended positions

Usage:
    python benchmarks/bench_vs_dense.py                  # Full suite
    python benchmarks/bench_vs_dense.py --quick          # Quick (CPU-safe) configs only
    python benchmarks/bench_vs_dense.py --config 0       # Single config

Colab:
    !python benchmarks/bench_vs_dense.py
"""

from __future__ import annotations

import argparse
import math
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from sparse_attn.patterns import make_local_stride_pattern
from sparse_attn.kernels import sparse_prefill, sparse_prefill_reference
from sparse_attn.kernels.pattern_compiler import (
    compute_memory_reduction,
    expected_sm_utilization,
)

# --------------------------------------------------------------------------- #
#  Benchmark Configurations                                                   #
# --------------------------------------------------------------------------- #

CONFIGS = [
    # (batch, heads, seq_len, head_dim, local_window_blocks, stride_blocks, global_blocks)
    # name,         B,  H,  S,      D,   lw, st, gb
    ("4K-75%",      1, 32,  4096,  128,   8, 16,  4),
    ("16K-90%",     1, 32, 16384,  128,  16, 32,  4),
    ("32K-95%",     1, 32, 32768,  128,  16, 32,  4),
    ("128K-99%",    1, 32,131072,  128,   8, 16,  4),
    ("64K-batch4",  4, 32, 65536,  128,  16, 32,  4),
]

# Quick CPU-safe configs (small sequences, 1 head)
QUICK_CONFIGS = [
    ("256-local",   1, 2,  256,  32,  4,  8,  2),
    ("512-hetero",  1, 2,  512,  32,  4,  8,  2),
    ("1K-hetero",   1, 4, 1024,  64,  4,  8,  2),
    ("2K-hetero",   1, 4, 2048,  64,  8, 16,  2),
]


# --------------------------------------------------------------------------- #
#  Dense Reference                                                            #
# --------------------------------------------------------------------------- #

def dense_sdpa(q, k, v, scale=None):
    """PyTorch scaled_dot_product_attention (dense, causal)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)


# --------------------------------------------------------------------------- #
#  Timing Utilities                                                           #
# --------------------------------------------------------------------------- #

def time_fn(fn, *args, warmup=3, repeats=10, **kwargs):
    """Time a function with GPU sync."""
    for _ in range(warmup):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeats

    return elapsed, out


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


# --------------------------------------------------------------------------- #
#  Single Config Benchmark                                                    #
# --------------------------------------------------------------------------- #

def bench_config(name, B, H, S, D, local_window, stride, global_blocks, device, dtype):
    print(f"\n{'='*70}")
    print(f"Config: {name}  | B={B} H={H} S={S:,} D={D} | device={device}")
    print(f"{'='*70}")

    # Determine block size based on GPU capability to prevent SMEM spilling on T4 (Compute 7.5)
    # T4 has 64KB SMEM/SM. bs=64 requires ~80KB -> spills to global memory (10x slowdown).
    # A100 (Compute 8.0) has 164KB SMEM/SM, so bs=64 is fine.
    is_t4 = torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] < 8
    bs = 32 if is_t4 else 64
    
    # Scale block parameters to maintain the exact same token coverage
    scale_factor = 64 // bs
    adj_lw = local_window * scale_factor
    adj_st = stride * scale_factor
    adj_gb = global_blocks * scale_factor

    pattern = make_local_stride_pattern(
        seq_len=S,
        block_size=bs,
        local_window_blocks=adj_lw,
        stride_blocks=adj_st,
        global_blocks=adj_gb,
    )
    sparsity = pattern.sparsity
    print(f"Pattern sparsity: {sparsity:.1%} | Active blocks: {pattern.num_active_blocks}")

    # Create tensors
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    q = torch.randn(B, H, S, D, dtype=dtype, device=device)
    k = torch.randn(B, H, S, D, dtype=dtype, device=device)
    v = torch.randn(B, H, S, D, dtype=dtype, device=device)

    scale = D ** -0.5

    # --- Dense attention ---
    dense_time, dense_out = None, None
    try:
        dense_time, dense_out = time_fn(dense_sdpa, q, k, v, scale)
        dense_mem = get_gpu_memory_mb()
        print(f"Dense SDPA:    {dense_time*1000:8.2f} ms | mem: {dense_mem:.0f} MB")
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"Dense SDPA:    OOM ({type(e).__name__})")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- Sparse attention ---
    try:
        sparse_fn = lambda: sparse_prefill(q, k, v, pattern, softmax_scale=scale)
        sparse_time, sparse_out = time_fn(sparse_fn)
        sparse_mem = get_gpu_memory_mb()
        print(f"Sparse Prefill:{sparse_time*1000:8.2f} ms | mem: {sparse_mem:.0f} MB")

        # Speedup
        if dense_time:
            speedup = dense_time / sparse_time
            print(f"Speedup:       {speedup:.2f}×")

        # Memory reduction
        dense_gb, sparse_gb, reduction = compute_memory_reduction(S, sparsity)
        print(f"Theoretical memory: dense={dense_gb:.1f}GB → sparse={sparse_gb:.3f}GB ({reduction:.1f}×)")

        # SM Utilization
        util = expected_sm_utilization(B, H, S, 512, sparsity)
        print(f"Expected SM util (chunked-KV decode): {util:.0%}")

        # Correctness
        if S <= 16384:
            ref_out = sparse_prefill(q, k, v, pattern, softmax_scale=scale, force_reference=True)
            err = (ref_out.float() - sparse_out.float()).abs().max().item()
            print(f"Max error vs ref: {err:.2e} {'✓' if err < 1e-2 else '✗'}")
        else:
            print("Max error vs ref: Skipped for large S")

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"Sparse:        OOM ({type(e).__name__})")

    return {
        "name": name, "seq_len": S, "sparsity": sparsity,
        "dense_ms": (dense_time or 0) * 1000,
        "sparse_ms": (sparse_time if 'sparse_time' in dir() else 0) * 1000,
    }


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Sparse vs Dense Attention Benchmark")
    parser.add_argument("--quick",  action="store_true", help="Run quick CPU-safe configs")
    parser.add_argument("--config", type=int, default=-1, help="Run single config by index")
    parser.add_argument("--device", default="auto", help="cuda|cpu|auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"\n{'#'*70}")
    print(f"# Sparse Attention vs Dense SDPA Benchmark")
    print(f"# Device: {device} | dtype: {dtype}")
    if torch.cuda.is_available():
        print(f"# GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'#'*70}")

    configs = QUICK_CONFIGS if args.quick else CONFIGS

    if args.config >= 0:
        configs = [configs[args.config]]

    results = []
    for cfg in configs:
        name, B, H, S, D, lw, st, gb = cfg
        result = bench_config(name, B, H, S, D, lw, st, gb, device, dtype)
        results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Config':<15} {'Seq':>8} {'Sparsity':>10} {'Dense ms':>10} {'Sparse ms':>10} {'Speedup':>8}")
    print(f"{'-'*70}")
    for r in results:
        speedup = r["dense_ms"] / r["sparse_ms"] if r["sparse_ms"] > 0 else 0
        print(f"{r['name']:<15} {r['seq_len']:>8,} {r['sparsity']:>9.1%} "
              f"{r['dense_ms']:>10.2f} {r['sparse_ms']:>10.2f} {speedup:>7.2f}×")


if __name__ == "__main__":
    main()
