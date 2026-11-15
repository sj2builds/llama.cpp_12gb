// R3-5 step 1: fake-int8 operands for the prefill quality gate. Opt-in only (LLAMA_FAKE_INT8_PREFILL); with the
// variable unset no kernel in this file is launched and the default path is untouched.
//
// Both operands are brought onto the int8 grid in the Hadamard-rotated basis and stored as bare integer codes in
// f16 (|q| <= 127 is exact in f16, and every product q_w*q_x <= 16129 is exact in f32), so an f16 GEMM with f32
// accumulation reproduces an int8 GEMM to f32-sum rounding. The per-row / per-token scales are applied to the
// output afterwards (ggml_cuda_fakeq_epilogue), which is the int8 GEMM epilogue.
//
// Structure per line (one row of the weights, or one column of the activations), as specified in
// dev/int8-prefill-gate.md: two passes over the line, blocks of 128 kept independent.
//   pass 1: rotate each 128-value block in place in the f16 buffer, track the abs-max of the whole line
//   pass 2: read the rotated value back, round it to the integer code, clamp to +-127, store it (still f16)
// With 4 warps per CTA and blocks of 128 values held in 4 registers per lane, no shared memory is needed beyond
// the 4-float abs-max reduction: the warp shuffle does the stages below 32 and the register array the rest.

#include "common.cuh"
#include "convert.cuh"
#include "fakeq.cuh"

// Hadamard-128 of one block held as 4 values per lane (element i lives at reg[i/32] of lane i%32, exactly the
// layout of fwht_cuda in fwht.cu). Stages below the warp width shuffle, the two stages above it are register
// pairs.
__device__ __forceinline__ void fq_hadamard_block(float reg[FQ_BLK / 32], int lane) {
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
        for (int j = 0; j < FQ_BLK / 32; ++j) {
            const float v  = reg[j];
            const float v2 = __shfl_xor_sync(0xFFFFFFFFu, v, h, 32);
            reg[j] = (lane & h) == 0 ? v + v2 : v2 - v;
        }
    }
#pragma unroll
    for (int h = 32; h < FQ_BLK; h <<= 1) {
        const int step = h / 32;
#pragma unroll
        for (int j = 0; j < FQ_BLK / 32; j += 2 * step) {
#pragma unroll
            for (int q = 0; q < step; ++q) {
                const float x = reg[j + q];
                const float y = reg[j + q + step];
                reg[j + q]        = x + y;
                reg[j + q + step] = x - y;
            }
        }
    }
}

// One CTA per line. TPLAIN: read f32 (activations) instead of f16 (weights).
template <bool PLAIN_F32>
__device__ __forceinline__ void fq_line(const void * src_v, half * dst, float * scale_out, int64_t k,
                                       bool quantize) {
    const int64_t line = blockIdx.x;
    const int     t    = threadIdx.x;
    const int     w    = t >> 5;        // warp in the CTA
    const int     lane = t & 31;
    const int64_t nb   = k / FQ_BLK;    // independent 128-value blocks along K

    const float * src_f = (const float *) src_v;
    const half  * src_h = (const half  *) src_v;
    if constexpr (PLAIN_F32) {
        src_f += line * k;
    } else {
        src_h += line * k;
    }
    dst += line * k;

    __shared__ float s_red[FQ_NT / 32];
    __shared__ float s_scale;

    // ---- pass 1: rotate each block in place, accumulate the abs-max of the whole line
    float amax = 0.0f;
    for (int64_t b = w; b < nb; b += FQ_NT / 32) {
        const int64_t off = b * FQ_BLK;
        float reg[FQ_BLK / 32];
#pragma unroll
        for (int j = 0; j < FQ_BLK / 32; ++j) {
            const int64_t i = off + j * 32 + lane;
            if constexpr (PLAIN_F32) {
                reg[j] = src_f[i];
            } else {
                reg[j] = __half2float(src_h[i]);
            }
        }
        fq_hadamard_block(reg, lane);
        constexpr float norm = 0.08838834764831845f;   // 1/sqrt(128)
#pragma unroll
        for (int j = 0; j < FQ_BLK / 32; ++j) {
            reg[j] *= norm;
            dst[off + j * 32 + lane] = __float2half(reg[j]);
            amax = fmaxf(amax, fabsf(reg[j]));
        }
    }

    // ---- abs-max of the line: warp reduce, then the 4 warps of the CTA
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFFu, amax, o, 32));
    }
    if (lane == 0) {
        s_red[w] = amax;
    }
    __syncthreads();
    if (t == 0) {
        float m = s_red[0];
        for (int i = 1; i < FQ_NT / 32; ++i) {
            m = fmaxf(m, s_red[i]);
        }
        s_scale      = m > 0.0f ? m / 127.0f : 1.0f;
        scale_out[line] = s_scale;
    }
    __syncthreads();

    if (!quantize) {
        return;   // rotation only: the f16 buffer keeps the rotated values
    }

    // ---- pass 2: scale + round to the integer code (read back the rotated value)
    const float inv = 1.0f / s_scale;
    for (int64_t b = w; b < nb; b += FQ_NT / 32) {
        const int64_t off = b * FQ_BLK;
#pragma unroll
        for (int j = 0; j < FQ_BLK / 32; ++j) {
            float v = __half2float(dst[off + j * 32 + lane]) * inv;
            int   q = __float2int_rn(v);
            q = q > 127 ? 127 : (q < -127 ? -127 : q);
            dst[off + j * 32 + lane] = __float2half((float) q);
        }
    }
}

__global__ void fq_rows_kernel(half * w, float * scale, int64_t k, bool quantize) {
    fq_line<false>(w, w, scale, k, quantize);
}

__global__ void fq_cols_kernel(const float * x, half * q, float * scale, int64_t k, bool quantize) {
    fq_line<true>(x, q, scale, k, quantize);
}

__global__ void fq_cvt_kernel(const float * __restrict__ x, half * __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half(x[i]);
    }
}

__global__ void fq_epilogue_kernel(float * __restrict__ dst, const float * __restrict__ ws,
                                   const float * __restrict__ xs, int64_t m, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m * n) {
        const int64_t r = i % m;         // row of the weights (contiguous within one column of dst)
        const int64_t c = i / m;         // token
        dst[i] *= ws[r] * xs[c];
    }
}

void ggml_cuda_fakeq_rows_f16(half * w, int64_t nrows, int64_t k, float * scale, bool quantize,
                              cudaStream_t stream) {
    GGML_ASSERT(k % FQ_BLK == 0);
    fq_rows_kernel<<<(unsigned) nrows, FQ_NT, 0, stream>>>(w, scale, k, quantize);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_fakeq_cols_f16(const float * x, half * q, int64_t ncols, int64_t k, float * scale, bool quantize,
                              cudaStream_t stream) {
    GGML_ASSERT(k % FQ_BLK == 0);
    fq_cols_kernel<<<(unsigned) ncols, FQ_NT, 0, stream>>>(x, q, scale, k, quantize);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_fakeq_cvt_f16(const float * x, half * q, int64_t n, cudaStream_t stream) {
    const int64_t nblk = (n + 255) / 256;
    fq_cvt_kernel<<<(unsigned) nblk, 256, 0, stream>>>(x, q, n);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_fakeq_epilogue(float * dst, const float * ws, const float * xs, int64_t m, int64_t n,
                              cudaStream_t stream) {
    const int64_t total = m * n;
    const int64_t nblk  = (total + 255) / 256;
    fq_epilogue_kernel<<<(unsigned) nblk, 256, 0, stream>>>(dst, ws, xs, m, n);
    CUDA_CHECK(cudaGetLastError());
}
