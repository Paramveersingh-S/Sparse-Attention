/**
 * include/block_pattern.cuh
 * ==========================
 * Sparse block pattern representation and CSR format for CUDA kernels.
 *
 * The SparseCSRMask struct represents a block-sparse attention pattern
 * in Compressed Sparse Row format for efficient kernel iteration.
 *
 * Usage (host side):
 *   SparseCSRMask mask;
 *   mask.col_indices = dev_col_indices;
 *   mask.row_ptrs    = dev_row_ptrs;
 *   mask.causal_mask = dev_causal_flags;
 *   mask.num_blocks  = N;
 *   launch_kernel<<<grid, block>>>(mask, ...);
 *
 * Usage (device side):
 *   for (int ptr = mask.row_ptrs[query_block];
 *            ptr < mask.row_ptrs[query_block + 1]; ptr++) {
 *       int kv_block   = mask.col_indices[ptr];
 *       bool is_causal = mask.causal_mask[ptr];
 *       ...
 *   }
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// --------------------------------------------------------------------------- //
//  CSR Sparse Block Mask                                                       //
// --------------------------------------------------------------------------- //

struct SparseCSRMask {
    const int*    col_indices;   // [num_active_blocks] — column block index
    const int*    row_ptrs;      // [num_blocks + 1]    — CSR row pointers
    const int8_t* causal_mask;   // [num_active_blocks] — 1 if diagonal block
    int num_blocks;
    int num_active_blocks;
    int block_size;

    /**
     * Device-callable: iterate active KV blocks for a query block.
     * Example:
     *   int start = mask.row_start(q_block);
     *   int end   = mask.row_end(q_block);
     *   for (int i = start; i < end; i++) {
     *       int kv = mask.col_indices[i];
     *       ...
     *   }
     */
    __device__ __forceinline__
    int row_start(int query_block) const {
        return row_ptrs[query_block];
    }

    __device__ __forceinline__
    int row_end(int query_block) const {
        return row_ptrs[query_block + 1];
    }

    __device__ __forceinline__
    bool is_diagonal(int ptr_idx) const {
        return causal_mask[ptr_idx] != 0;
    }

    __device__ __forceinline__
    int kv_block(int ptr_idx) const {
        return col_indices[ptr_idx];
    }
};

// --------------------------------------------------------------------------- //
//  Block pattern statistics (for analysis)                                     //
// --------------------------------------------------------------------------- //

__global__ void compute_pattern_stats_kernel(
    const int* row_ptrs,
    int num_blocks,
    int* active_per_row,     // output: active blocks per query block
    float* avg_sparsity      // output: overall sparsity
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_blocks) return;

    int active = row_ptrs[row + 1] - row_ptrs[row];
    active_per_row[row] = active;

    // Reduction for average (simplified, use proper reduction in production)
    atomicAdd(reinterpret_cast<int*>(avg_sparsity), num_blocks - active);
}
