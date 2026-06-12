/**
 * include/sparse_attn.cuh
 * ========================
 * Main CUDA header for sparse attention kernels.
 *
 * Provides:
 *  - SparseAttentionConfig: kernel configuration struct
 *  - launch_sparse_prefill: launch Triton prefill kernel via PyTorch extension
 *  - launch_sparse_decode: launch chunked-KV decode kernel
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 -I include/ sparse_decode_chunked_kv.cu -o sparse_decode
 *
 * Or use the Python extension:
 *   python setup.py build_ext --inplace
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// --------------------------------------------------------------------------- //
//  Configuration                                                               //
// --------------------------------------------------------------------------- //

struct SparseAttentionConfig {
    int batch_size;      // B
    int num_heads;       // H
    int seq_len;         // S (KV sequence length)
    int head_dim;        // D
    int block_size;      // b (block granularity)
    int chunk_size;      // chunk size for decode kernel (default: 512)
    float softmax_scale; // 1/sqrt(D)
    bool causal;         // enforce causal masking
};

// --------------------------------------------------------------------------- //
//  CSR Pattern Format                                                          //
// --------------------------------------------------------------------------- //

struct SparseCSRPattern {
    const int* col_indices;    // [num_active_blocks]   column block indices
    const int* row_ptrs;       // [num_blocks + 1]       CSR row pointers
    const int8_t* causal_flags;// [num_active_blocks]   1 = diagonal block
    int num_blocks;
    int num_active_blocks;
};

// --------------------------------------------------------------------------- //
//  Kernel Declarations                                                         //
// --------------------------------------------------------------------------- //

/**
 * sparse_decode_chunked_kv_kernel
 * --------------------------------
 * Pass 1 of chunked-KV decode: compute partial attention per KV chunk.
 *
 * Grid: (num_chunks, num_heads, batch_size)
 * Block: (32, 1, 1)  — warp-level computation
 *
 * Parameters:
 *   Q             [B, H, 1, D]          query (current decode token)
 *   K, V          [B, H, S, D]          KV cache
 *   chunk_active  [B, H, num_chunks]    0/1 per chunk (sparse mask)
 *   partial_O     [B, H, num_chunks, D] output buffer
 *   partial_m     [B, H, num_chunks]    running max
 *   partial_l     [B, H, num_chunks]    running sum
 */
__global__ void sparse_decode_chunked_kv_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    const int8_t* __restrict__ chunk_active,
    float* __restrict__ partial_O,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int B, int H, int S, int D,
    int chunk_size, int num_chunks,
    float softmax_scale
);

/**
 * sparse_merge_partial_kernel
 * ----------------------------
 * Pass 2: merge partial (O, m, l) results via online softmax merge.
 *
 * Grid: (batch_size, num_heads, 1)
 * Block: (head_dim, 1, 1)
 */
__global__ void sparse_merge_partial_kernel(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    __half* __restrict__ output,
    int B, int H, int num_chunks, int D
);

// --------------------------------------------------------------------------- //
//  Host Launch Functions                                                       //
// --------------------------------------------------------------------------- //

/**
 * launch_sparse_decode
 * ----------------------
 * Host-side launcher for the two-pass chunked-KV decode.
 */
void launch_sparse_decode(
    const __half* Q,
    const __half* K,
    const __half* V,
    const int8_t* chunk_active,
    __half* output,
    const SparseAttentionConfig& cfg,
    cudaStream_t stream = nullptr
);
