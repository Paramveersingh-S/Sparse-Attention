"""
benchmarks/bench_sparse_decode.py
=====================================
Decode (single-token) benchmark: SM utilization with chunked-KV approach.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch

from sparse_attn.patterns import make_local_stride_pattern
from sparse_attn.kernels import sparse_decode_chunked
from sparse_attn.kernels.sparse_decode import compute_decode_sm_utilization
from sparse_attn.kernels.pattern_compiler import expected_sm_utilization


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    num_sms = 128  # A100
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Approximate SM count from CUDA cores
        num_sms = props.multi_processor_count
        print(f"GPU: {props.name} | SMs: {num_sms}")

    configs = [
        # (B, H, S, D, chunk_size)
        (1,  8,   4096, 64, 256),
        (1,  16,  8192, 64, 256),
        (4,  16, 16384, 64, 512),
        (4,  32, 32768, 64, 512),
    ]

    print(f"\nSparse Decode (Chunked-KV) Benchmark | device={device}")
    print(f"{'B':>3} {'H':>4} {'S':>8} {'D':>4} {'Chunk':>6} {'Sparse%':>8} "
          f"{'Naive SM':>9} {'Chunk SM':>9} {'Time ms':>9}")
    print("-" * 80)

    for (B, H, S, D, chunk_size) in configs:
        pattern = make_local_stride_pattern(
            seq_len=S, block_size=64,
            local_window_blocks=8, stride_blocks=16, global_blocks=4
        )

        q = torch.randn(B, H,  1, D, dtype=dtype, device=device)
        k = torch.randn(B, H,  S, D, dtype=dtype, device=device)
        v = torch.randn(B, H,  S, D, dtype=dtype, device=device)

        # SM utilization stats
        stats       = compute_decode_sm_utilization(B, H, S, chunk_size, pattern, num_sms)
        naive_util  = expected_sm_utilization(B, H, S, chunk_size, 0.0, num_sms)
        chunked_util = min(stats["total_blocks"] / num_sms, 1.0)

        # Warmup
        for _ in range(2):
            sparse_decode_chunked(q, k, v, pattern, chunk_size=chunk_size)
        if device == "cuda":
            torch.cuda.synchronize()

        # Timing
        t0 = time.perf_counter()
        REPS = 5
        for _ in range(REPS):
            out = sparse_decode_chunked(q, k, v, pattern, chunk_size=chunk_size)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / REPS * 1000

        print(f"{B:>3} {H:>4} {S:>8,} {D:>4} {chunk_size:>6} "
              f"{pattern.sparsity:>7.1%} "
              f"{naive_util:>8.0%} {chunked_util:>8.0%} "
              f"{elapsed_ms:>9.2f}")

    print(f"\n  Naive SM: utilization without chunked-KV (just batch×heads)")
    print(f"  Chunk SM: utilization with chunked-KV (batch×heads×active_chunks)")


if __name__ == "__main__":
    main()
