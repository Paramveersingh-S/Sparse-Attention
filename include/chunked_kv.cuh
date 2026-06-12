/**
 * include/chunked_kv.cuh
 * =======================
 * Chunked KV attention helpers and warp-level primitives.
 *
 * Key Insight: SM Utilization Recovery
 * =====================================
 * Standard sparse decode: grid = B × H → low occupancy
 * Chunked-KV decode:     grid = B × H × active_chunks → high occupancy
 *
 * Online Softmax Merge (Flash-Attention split-KV)
 * ------------------------------------------------
 * For two partials (O1, m1, l1) and (O2, m2, l2):
 *   m_new = max(m1, m2)
 *   l_new = exp(m1 - m_new) * l1 + exp(m2 - m_new) * l2
 *   O_new = (exp(m1-m_new) * l1 * O1 + exp(m2-m_new) * l2 * O2) / l_new
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

// --------------------------------------------------------------------------- //
//  Warp-level Reduction Utilities                                              //
// --------------------------------------------------------------------------- //

__device__ __forceinline__
float warp_reduce_max(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__
float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// --------------------------------------------------------------------------- //
//  Partial Attention Accumulator                                               //
// --------------------------------------------------------------------------- //

struct PartialAttn {
    float m;  // running max
    float l;  // running sum of exp
    float* o; // running weighted sum (pointer to shared/register memory)
    int   D;  // head dimension

    __device__ void init(float* buf, int head_dim) {
        m = -FLT_MAX;
        l = 0.0f;
        o = buf;
        D = head_dim;
        for (int i = 0; i < D; i++) o[i] = 0.0f;
    }

    /**
     * Merge another partial result into this one.
     * Implements the online softmax merge formula.
     */
    __device__ void merge(const PartialAttn& other) {
        float m_new = fmaxf(m, other.m);
        float scale1 = expf(m - m_new);
        float scale2 = expf(other.m - m_new);
        float l_new  = scale1 * l + scale2 * other.l;
        for (int i = 0; i < D; i++) {
            o[i] = (scale1 * l * o[i] + scale2 * other.l * other.o[i]) / fmaxf(l_new, 1e-8f);
        }
        m = m_new;
        l = l_new;
    }

    /**
     * Finalize: normalize by l to get the true output.
     * (Not needed when using the merge formula above, which keeps O normalized.)
     */
    __device__ void finalize() {
        float inv_l = (l > 1e-8f) ? (1.0f / l) : 0.0f;
        for (int i = 0; i < D; i++) o[i] *= inv_l;
    }
};

// --------------------------------------------------------------------------- //
//  Chunked KV Kernel (Pass 1)                                                  //
// --------------------------------------------------------------------------- //

/**
 * One CUDA block per (batch, head, chunk) triple.
 * Early-exit for inactive chunks (sparse pattern check).
 */
__global__ void sparse_decode_chunked_kv_kernel(
    const __half* __restrict__ Q,        // [B, H, 1, D]
    const __half* __restrict__ K,        // [B, H, S, D]
    const __half* __restrict__ V,        // [B, H, S, D]
    const int8_t* __restrict__ chunk_active, // [B, H, num_chunks]
    float* __restrict__ partial_O,       // [B, H, num_chunks, D]
    float* __restrict__ partial_m,       // [B, H, num_chunks]
    float* __restrict__ partial_l,       // [B, H, num_chunks]
    int B, int H, int S, int D,
    int chunk_size, int num_chunks,
    float softmax_scale
) {
    int chunk = blockIdx.x;
    int head  = blockIdx.y;
    int batch = blockIdx.z;

    // Early exit for inactive chunks
    int active_idx = batch * H * num_chunks + head * num_chunks + chunk;
    if (!chunk_active[active_idx]) return;

    int tid  = threadIdx.x;
    int lane = tid % 32;  // warp lane

    // KV range for this chunk
    int kv_start = chunk * chunk_size;
    int kv_end   = min(kv_start + chunk_size, S);
    int cs       = kv_end - kv_start;

    if (cs <= 0) return;

    // Load Q vector (shared across all threads in block)
    // Q: [B, H, 1, D] → offset to (batch, head, 0, :)
    const __half* q_ptr = Q + (batch * H + head) * D;

    // Running accumulators (in registers for each thread handling a subset of D)
    float m_val = -FLT_MAX;
    float l_val = 0.0f;

    // Allocate shared memory for partial output
    extern __shared__ float shmem[];  // [D]
    for (int d = tid; d < D; d += blockDim.x)
        shmem[d] = 0.0f;
    __syncthreads();

    // Compute Q·K^T for each KV token in chunk
    for (int kv_i = kv_start; kv_i < kv_end; kv_i++) {
        // Compute dot product Q·K[kv_i]
        const __half* k_ptr = K + (batch * H + head) * S * D + kv_i * D;
        float score = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
            score += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        // Warp reduce sum
        score = warp_reduce_sum(score);
        score *= softmax_scale;

        // Online softmax update
        float m_new = fmaxf(m_val, score);
        float p     = expf(score - m_new);
        l_val = expf(m_val - m_new) * l_val + p;
        m_val = m_new;

        // Accumulate O
        const __half* v_ptr = V + (batch * H + head) * S * D + kv_i * D;
        for (int d = tid; d < D; d += blockDim.x) {
            // Re-normalize as we go
            atomicAdd(&shmem[d], p * __half2float(v_ptr[d]));
        }
    }
    __syncthreads();

    // Normalize and write output
    int out_base = ((batch * H + head) * num_chunks + chunk);
    if (tid == 0) {
        partial_m[out_base] = m_val;
        partial_l[out_base] = l_val;
    }
    float inv_l = (l_val > 1e-8f) ? (1.0f / l_val) : 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        partial_O[out_base * D + d] = shmem[d] * inv_l;
    }
}


// --------------------------------------------------------------------------- //
//  Merge Kernel (Pass 2)                                                       //
// --------------------------------------------------------------------------- //

/**
 * Merge all chunk partial outputs via online softmax merge.
 * One block per (batch, head); threads handle head_dim elements.
 */
__global__ void sparse_merge_partial_kernel(
    const float* __restrict__ partial_O,  // [B, H, num_chunks, D]
    const float* __restrict__ partial_m,  // [B, H, num_chunks]
    const float* __restrict__ partial_l,  // [B, H, num_chunks]
    __half* __restrict__ output,           // [B, H, 1, D]
    int B, int H, int num_chunks, int D
) {
    int head  = blockIdx.x;
    int batch = blockIdx.y;
    int d     = threadIdx.x;

    if (d >= D) return;

    // Running accumulators
    float m_acc = -FLT_MAX;
    float l_acc = 0.0f;
    float o_acc = 0.0f;

    int base = (batch * H + head) * num_chunks;

    for (int c = 0; c < num_chunks; c++) {
        float l_c = partial_l[base + c];
        if (l_c < 1e-8f) continue;  // empty chunk

        float m_c = partial_m[base + c];
        float o_c = partial_O[(base + c) * D + d];

        float m_new = fmaxf(m_acc, m_c);
        float s1    = expf(m_acc - m_new);
        float s2    = expf(m_c   - m_new);
        float l_new = s1 * l_acc + s2 * l_c;

        o_acc = (s1 * l_acc * o_acc + s2 * l_c * o_c) / fmaxf(l_new, 1e-8f);
        m_acc = m_new;
        l_acc = l_new;
    }

    // Write final output
    int out_idx = (batch * H + head) * D + d;
    output[out_idx] = __float2half(o_acc);
}
