"""
benchmarks/bench_sparse_prefill.py
=====================================
Detailed prefill kernel benchmark: throughput vs. sequence length.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
from sparse_attn.patterns import make_local_stride_pattern
from sparse_attn.kernels import sparse_prefill

CONFIGS = [
    # (seq_len, block_size, B, H, D, local_w, stride, global_b)
    ( 1024,  64, 1, 8,  64,  4,  8, 2),
    ( 2048,  64, 1, 8,  64,  8, 16, 4),
    ( 4096,  64, 1, 16, 64,  8, 16, 4),
    ( 8192,  64, 1, 16, 64,  8, 16, 4),
    (16384,  64, 1, 32, 64,  8, 16, 4),
    (32768,  64, 1, 32, 128, 8, 16, 4),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"\nSparse Prefill Benchmark | device={device}")
    print(f"{'Seq':>8} {'H':>4} {'D':>4} {'Sparsity':>9} {'Time ms':>9} {'TFLOP/s':>9}")
    print("-" * 55)

    for (S, bs, B, H, D, lw, st, gb) in CONFIGS:
        pattern = make_local_stride_pattern(
            seq_len=S, block_size=bs,
            local_window_blocks=lw, stride_blocks=st, global_blocks=gb
        )

        q = torch.randn(B, H, S, D, dtype=dtype, device=device)
        k = torch.randn(B, H, S, D, dtype=dtype, device=device)
        v = torch.randn(B, H, S, D, dtype=dtype, device=device)

        # Warmup
        for _ in range(3):
            sparse_prefill(q, k, v, pattern)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timed run
        t0 = time.perf_counter()
        REPS = 10
        for _ in range(REPS):
            out = sparse_prefill(q, k, v, pattern)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / REPS * 1000

        # Compute theoretical FLOPs (active blocks only)
        active_flops = pattern.num_active_blocks * bs * bs * D * 2 * B * H  # 2 for matmul
        tflops = active_flops / (elapsed_ms / 1000) / 1e12

        print(f"{S:>8,} {H:>4} {D:>4} {pattern.sparsity:>8.1%} {elapsed_ms:>9.2f} {tflops:>9.3f}")


if __name__ == "__main__":
    main()
