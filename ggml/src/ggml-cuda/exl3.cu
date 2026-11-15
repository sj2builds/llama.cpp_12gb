#include "exl3.cuh"

#include <cuda_fp16.h>

// Trellis decode ported from exllamav3 (MIT license, exllamav3_ext/quant/codebook.cuh and exl3_dq.cuh), mul1 codebook
// only. A lane decodes 8 consecutive trellis positions (lane * 8 .. lane * 8 + 7) of a 16x16 tile straight into the two
// m16n8k16 B fragments of the tile: f0 = columns 0..7, f1 = columns 8..15.

struct exl3_frag_b {
    half2 v[2];
};

static __device__ __forceinline__ half2 exl3_decode_mul1_2(uint32_t x0, uint32_t x1) {
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    const uint32_t acc  = 0x6400u; // 0x6400 -> 1024.0 .. 0x67FF -> 2047.0
    const uint32_t sum0 = __dp4a(x0, 0x01010101u, acc);
    const uint32_t sum1 = __dp4a(x1, 0x01010101u, acc);
    const half2 k_inv  = __half2half2(__ushort_as_half(0x1eee)); //  1/147.7
    const half2 k_bias = __half2half2(__ushort_as_half(0xc931)); // -10.39 = (-1024 - 510) / 147.7
    const half2 h = __halves2half2(__ushort_as_half((unsigned short) sum0), __ushort_as_half((unsigned short) sum1));
    return __hfma2(h, k_inv, k_bias);
}

static __device__ __forceinline__ uint32_t exl3_fshift(const uint32_t b, const uint32_t a, const int shift) {
    const uint64_t merged = ((uint64_t) a << 32) | (uint64_t) b;
    return (uint32_t) (merged >> shift);
}

static __device__ __forceinline__ uint32_t exl3_bfe16(const uint32_t x, const int offset) {
    return (x >> offset) & 0xffffu;
}

template <int bits>
static __device__ __forceinline__ void exl3_dq4(const uint32_t * ptr, const int t_offset, exl3_frag_b & frag) {
    constexpr int W = bits * 8;
    const int b0 = (t_offset + 257) * bits - 16;
    const int b1 = b0 + 3 * bits;
    const int b2 = b1 + 16;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a  = ptr[i0 % W];
    const uint32_t b  = ptr[i2 % W];
    const uint32_t w3 = exl3_fshift(b, a, s2)            & 0xffffu;
    const uint32_t w2 = exl3_fshift(b, a, s2 + bits)     & 0xffffu;
    const uint32_t w1 = exl3_fshift(b, a, s2 + bits * 2) & 0xffffu;
    const uint32_t w0 = exl3_fshift(b, a, s2 + bits * 3) & 0xffffu;
    frag.v[0] = exl3_decode_mul1_2(w0, w1);
    frag.v[1] = exl3_decode_mul1_2(w2, w3);
}

template <int bits>
static __device__ __forceinline__ void exl3_dq2x2(const uint32_t * ptr, const int t_offset, exl3_frag_b & frag) {
    constexpr int W = bits * 8;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int b0 = (t_offset + 2 * i + 257) * bits - 16;
        const int b1 = b0 + bits;
        const int b2 = b1 + 16;
        const int i0 = b0 / 32;
        const int i2 = (b2 - 1) / 32;
        const int s2 = (i2 + 1) * 32 - b2;
        const uint32_t a  = ptr[i0 % W];
        const uint32_t b  = ptr[i2 % W];
        const uint32_t w1 = exl3_fshift(b, a, s2)        & 0xffffu;
        const uint32_t w0 = exl3_fshift(b, a, s2 + bits) & 0xffffu;
        frag.v[i] = exl3_decode_mul1_2(w0, w1);
    }
}

// 3 bits: eight windows span at most two words for every lane offset (the align == 4 case of exllamav3's dq8)
template <int bits>
static __device__ __forceinline__ void exl3_dq8_align4(const uint32_t * ptr, const int t_offset, exl3_frag_b & f0, exl3_frag_b & f1) {
    constexpr int W = bits * 8;
    const int b1 = (t_offset + 257) * bits;
    const int b0 = b1 - 16;
    const int b2 = b1 + bits * 7;
    const int i0 = b0 / 32;
    const int i2 = (b2 - 1) / 32;
    const int s2 = (i2 + 1) * 32 - b2;
    const uint32_t a = ptr[i0 % W];
    const uint32_t b = ptr[i2 % W];
    const uint32_t w7 = exl3_fshift(b, a, s2);
    const uint32_t w6 = w7 >> bits;
    const uint32_t w5 = w6 >> bits;
    const uint32_t w4 = w5 >> bits;
    const uint32_t w3 = exl3_fshift(b, a, s2 + bits * 4);
    const uint32_t w2 = w3 >> bits;
    const uint32_t w1 = w2 >> bits;
    const uint32_t w0 = w1 >> bits;
    f0.v[0] = exl3_decode_mul1_2(w0 & 0xffffu, w1 & 0xffffu);
    f0.v[1] = exl3_decode_mul1_2(w2 & 0xffffu, w3 & 0xffffu);
    f1.v[0] = exl3_decode_mul1_2(w4 & 0xffffu, w5 & 0xffffu);
    f1.v[1] = exl3_decode_mul1_2(w6 & 0xffffu, w7 & 0xffffu);
}

static __device__ __forceinline__ void exl3_dq8_4bits(const uint32_t * ptr, const int t_offset, exl3_frag_b & f0, exl3_frag_b & f1) {
    const uint32_t i1 = t_offset >> 3;
    const uint32_t i0 = (i1 + 31) & 31;
    const uint32_t a  = ptr[i0];
    const uint32_t b  = ptr[i1];
    const uint32_t s  = __funnelshift_r(b, a, 20);
    const uint32_t w7 = b & 0xffffu;
    const uint32_t w6 = exl3_bfe16(b, 4);
    const uint32_t w5 = exl3_bfe16(b, 8);
    const uint32_t w4 = exl3_bfe16(b, 12);
    const uint32_t w3 = exl3_bfe16(b, 16);
    const uint32_t w2 = s & 0xffffu;
    const uint32_t w1 = exl3_bfe16(s, 4);
    const uint32_t w0 = exl3_bfe16(s, 8);
    f0.v[0] = exl3_decode_mul1_2(w0, w1);
    f0.v[1] = exl3_decode_mul1_2(w2, w3);
    f1.v[0] = exl3_decode_mul1_2(w4, w5);
    f1.v[1] = exl3_decode_mul1_2(w6, w7);
}

static __device__ __forceinline__ void exl3_dq8_2bits(const uint32_t * ptr, const int t_offset, exl3_frag_b & f0, exl3_frag_b & f1) {
    const uint32_t i1 = t_offset >> 4;
    const uint32_t i0 = (i1 + 15) & 15;
    const uint32_t a  = ptr[i0];
    uint32_t       b  = ptr[i1];
    b = exl3_fshift(b, a, ((~t_offset) & 8) << 1);
    const uint32_t w7 = b & 0xffffu;
    const uint32_t w6 = exl3_bfe16(b, 2);
    const uint32_t w5 = exl3_bfe16(b, 4);
    const uint32_t w4 = exl3_bfe16(b, 6);
    const uint32_t w3 = exl3_bfe16(b, 8);
    const uint32_t w2 = exl3_bfe16(b, 10);
    const uint32_t w1 = exl3_bfe16(b, 12);
    const uint32_t w0 = exl3_bfe16(b, 14);
    f0.v[0] = exl3_decode_mul1_2(w0, w1);
    f0.v[1] = exl3_decode_mul1_2(w2, w3);
    f1.v[0] = exl3_decode_mul1_2(w4, w5);
    f1.v[1] = exl3_decode_mul1_2(w6, w7);
}

static __device__ __forceinline__ void exl3_dq8_1bit(const uint32_t * ptr, const int t_offset, exl3_frag_b & f0, exl3_frag_b & f1) {
    const uint32_t i1 = t_offset >> 5;
    const uint32_t i0 = (i1 + 7) & 7;
    const uint32_t a  = ptr[i0];
    uint32_t       b  = ptr[i1];
    b = exl3_fshift(b, a, (~t_offset) & 24);
    const uint32_t w7 = b & 0xffffu;
    const uint32_t w6 = exl3_bfe16(b, 1);
    const uint32_t w5 = exl3_bfe16(b, 2);
    const uint32_t w4 = exl3_bfe16(b, 3);
    const uint32_t w3 = exl3_bfe16(b, 4);
    const uint32_t w2 = exl3_bfe16(b, 5);
    const uint32_t w1 = exl3_bfe16(b, 6);
    const uint32_t w0 = exl3_bfe16(b, 7);
    f0.v[0] = exl3_decode_mul1_2(w0, w1);
    f0.v[1] = exl3_decode_mul1_2(w2, w3);
    f1.v[0] = exl3_decode_mul1_2(w4, w5);
    f1.v[1] = exl3_decode_mul1_2(w6, w7);
}

// ptr: the tile's 8 * bits uint32, idx: first trellis position (lane * 8)
template <int bits>
static __device__ __forceinline__ void exl3_dq_tile(const uint32_t * ptr, const int idx, exl3_frag_b & f0, exl3_frag_b & f1) {
    if constexpr (bits == 1) {
        exl3_dq8_1bit(ptr, idx, f0, f1);
    } else if constexpr (bits == 2) {
        exl3_dq8_2bits(ptr, idx, f0, f1);
    } else if constexpr (bits == 3) {
        exl3_dq8_align4<bits>(ptr, idx, f0, f1);
    } else if constexpr (bits == 4) {
        exl3_dq8_4bits(ptr, idx, f0, f1);
    } else if constexpr (bits == 7) {
        exl3_dq2x2<bits>(ptr, idx,     f0);
        exl3_dq2x2<bits>(ptr, idx + 4, f1);
    } else {
        exl3_dq4<bits>(ptr, idx,     f0);
        exl3_dq4<bits>(ptr, idx + 4, f1);
    }
}

// 128-point Sylvester Hadamard across one warp: lane l holds elements 4l .. 4l + 3 (same order as exllamav3)
static __device__ __forceinline__ void exl3_had128(float & h0, float & h1, float & h2, float & h3, const int lane) {
    const float s0 = h0 + h1;
    const float d0 = h0 - h1;
    const float s1 = h2 + h3;
    const float d1 = h2 - h3;
    h0 = s0 + s1;
    h1 = d0 + d1;
    h2 = s0 - s1;
    h3 = d0 - d1;
#pragma unroll
    for (int i = 1; i < 32; i <<= 1) {
        const float p0 = __shfl_xor_sync(0xffffffff, h0, i);
        const float p1 = __shfl_xor_sync(0xffffffff, h1, i);
        const float p2 = __shfl_xor_sync(0xffffffff, h2, i);
        const float p3 = __shfl_xor_sync(0xffffffff, h3, i);
        if (lane & i) {
            h0 = p0 - h0;
            h1 = p1 - h1;
            h2 = p2 - h2;
            h3 = p3 - h3;
        } else {
            h0 = h0 + p0;
            h1 = h1 + p1;
            h2 = h2 + p2;
            h3 = h3 + p3;
        }
    }
}

#define EXL3_HAD_SCALE 0.088388347648f // 1/sqrt(128)

// xh[r, :] = H128(suh * x[r, :]) in f16; grid (K / 128, rows), 32 threads. Row r of src1 is (i1, i2, i3) in ggml order.
static __global__ void exl3_had_in_kernel(
        const char * __restrict__ x, half * __restrict__ xh, const half * __restrict__ suh,
        const int K, const int ne11, const int ne12, const size_t nb11, const size_t nb12, const size_t nb13) {
    const int lane = threadIdx.x;
    const int r    = blockIdx.y;
    const int c0   = blockIdx.x * 128 + lane * 4;

    const int i1  = r % ne11;
    const int i23 = r / ne11;
    const int i2  = i23 % ne12;
    const int i3  = i23 / ne12;
    const float * xr = (const float *) (x + i1 * nb11 + i2 * nb12 + i3 * nb13);

    float h0 = xr[c0 + 0] * __half2float(suh[c0 + 0]);
    float h1 = xr[c0 + 1] * __half2float(suh[c0 + 1]);
    float h2 = xr[c0 + 2] * __half2float(suh[c0 + 2]);
    float h3 = xr[c0 + 3] * __half2float(suh[c0 + 3]);
    exl3_had128(h0, h1, h2, h3, lane);

    half2 * out = (half2 *) (xh + (size_t) r * K + c0);
    out[0] = __floats2half2_rn(h0 * EXL3_HAD_SCALE, h1 * EXL3_HAD_SCALE);
    out[1] = __floats2half2_rn(h2 * EXL3_HAD_SCALE, h3 * EXL3_HAD_SCALE);
}

// dst[r, :] = svh * H128(sum_s part[s, r, :]); the slices are added in a fixed order. part may alias dst (S == 1).
// grid (N / 128, rows), 32 threads.
static __global__ void exl3_had_out_kernel(
        const float * part, float * dst, const half * __restrict__ svh,
        const int S, const size_t slice_stride, const size_t row_in, const size_t row_out) {
    const int lane = threadIdx.x;
    const int r    = blockIdx.y;
    const int c0   = blockIdx.x * 128 + lane * 4;

    float h0 = 0.0f;
    float h1 = 0.0f;
    float h2 = 0.0f;
    float h3 = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float4 p = *(const float4 *) (part + s * slice_stride + r * row_in + c0);
        h0 += p.x;
        h1 += p.y;
        h2 += p.z;
        h3 += p.w;
    }
    exl3_had128(h0, h1, h2, h3, lane);

    float4 o;
    o.x = h0 * EXL3_HAD_SCALE * __half2float(svh[c0 + 0]);
    o.y = h1 * EXL3_HAD_SCALE * __half2float(svh[c0 + 1]);
    o.z = h2 * EXL3_HAD_SCALE * __half2float(svh[c0 + 2]);
    o.w = h3 * EXL3_HAD_SCALE * __half2float(svh[c0 + 3]);
    *(float4 *) (dst + r * row_out + c0) = o;
}

static __device__ __forceinline__ void exl3_mma(const uint32_t (&a)[4], const exl3_frag_b & b, float (&c)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    const uint32_t * bb = reinterpret_cast<const uint32_t *>(&b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(c);
    NO_DEVICE_CODE;
#endif
}

// Small batches (m <= 16 * MT rows of xh): one warp per (32 output columns = 2 tiles, K slice s), 4 warps per block
// along n, MT row tiles of 16 per warp. Writes the f32 partial products of slice s to part[s, 0..m-1, :]. The slice split
// depends only on (K, N, device), and the tensor-core result of a row depends neither on the other rows nor on MT, so
// each row's partials are the same for every m <= EXL3_SMALL_MAX.
template <int bits, int MT>
static __global__ void __launch_bounds__(128) exl3_mm_small_kernel(
        const half * __restrict__ xh, const uint32_t * __restrict__ trellis, float * __restrict__ part,
        const int m, const int K, const int N, const int kt_per_slice) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
    constexpr int TW = bits * 8;          // uint32 per tile
    constexpr int GW = 2 * TW;            // the two adjacent tiles of a warp
    constexpr int NL = (GW + 31) / 32;    // loads per lane
    __shared__ uint32_t sbuf[4][GW];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int ng   = blockIdx.x * 4 + warp;
    const int s    = blockIdx.y;

    const int n_tiles  = N / 16;
    const int kt_total = K / 16;
    const int kt0      = s * kt_per_slice;
    const int kt1      = min(kt0 + kt_per_slice, kt_total);

    const int g = lane >> 2;
    const int t = lane & 3;

    float    acc[MT][4][4] = {};
    uint32_t pf[NL];
    uint32_t a[MT][4];
    uint32_t an[MT][4] = {};

    const uint32_t * tr = trellis + (size_t) 2 * ng * TW;
    const size_t     kt_stride = (size_t) n_tiles * TW;

    // A fragment rows 16 * mt + g (regs 0, 2) and 16 * mt + g + 8 (regs 1, 3), columns k + 2t (+8)
    auto load_a = [&](const int kt, uint32_t (&r)[MT][4]) {
        const int k = kt * 16 + 2 * t;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int row_lo = 16 * mt + g;
            const int row_hi = row_lo + 8;
            const half * x_lo = xh + (size_t) row_lo * K + k;
            const half * x_hi = xh + (size_t) row_hi * K + k;
            r[mt][0] = row_lo < m ? *(const uint32_t *) (x_lo)     : 0u;
            r[mt][1] = row_hi < m ? *(const uint32_t *) (x_hi)     : 0u;
            r[mt][2] = row_lo < m ? *(const uint32_t *) (x_lo + 8) : 0u;
            r[mt][3] = row_hi < m ? *(const uint32_t *) (x_hi + 8) : 0u;
        }
    };
    auto load_w = [&](const int kt) {
        const uint32_t * src = tr + kt * kt_stride;
#pragma unroll
        for (int j = 0; j < NL; ++j) {
            const int i = lane + 32 * j;
            pf[j] = (GW % 32 == 0 || i < GW) ? src[i] : 0u;
        }
    };

    if (kt0 < kt1) {
        load_w(kt0);
        load_a(kt0, a);
    }

    for (int kt = kt0; kt < kt1; ++kt) {
#pragma unroll
        for (int j = 0; j < NL; ++j) {
            const int i = lane + 32 * j;
            if (GW % 32 == 0 || i < GW) {
                sbuf[warp][i] = pf[j];
            }
        }
        __syncwarp();

        if (kt + 1 < kt1) {
            load_w(kt + 1);
            load_a(kt + 1, an);
        }

        exl3_frag_b b00, b01, b10, b11;
        exl3_dq_tile<bits>(sbuf[warp],      lane * 8, b00, b01);
        exl3_dq_tile<bits>(sbuf[warp] + TW, lane * 8, b10, b11);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            exl3_mma(a[mt], b00, acc[mt][0]);
            exl3_mma(a[mt], b01, acc[mt][1]);
            exl3_mma(a[mt], b10, acc[mt][2]);
            exl3_mma(a[mt], b11, acc[mt][3]);
        }
        __syncwarp();

#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                a[mt][j] = an[mt][j];
            }
        }
    }

    float * p = part + (size_t) s * m * N;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row_lo = 16 * mt + g;
        const int row_hi = row_lo + 8;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int col = ng * 32 + (q >> 1) * 16 + (q & 1) * 8 + 2 * t;
            if (row_lo < m) {
                *(float2 *) (p + (size_t) row_lo * N + col) = make_float2(acc[mt][q][0], acc[mt][q][1]);
            }
            if (row_hi < m) {
                *(float2 *) (p + (size_t) row_hi * N + col) = make_float2(acc[mt][q][2], acc[mt][q][3]);
            }
        }
    }
#else
    GGML_UNUSED(xh);
    GGML_UNUSED(trellis);
    GGML_UNUSED(part);
    GGML_UNUSED(m);
    GGML_UNUSED(K);
    GGML_UNUSED(N);
    GGML_UNUSED(kt_per_slice);
    NO_DEVICE_CODE;
#endif
}

// Rotated-basis f16 weight slice w[K, ns] (row-major) of n tiles [n_tile_offset, n_tile_offset + ns / 16), ported from
// exllamav3's reconstruct_tile. grid (ns / 128, K / 16), 256 threads: warp w decodes tile n_tile_offset + 8 * bx + w.
template <int bits>
static __global__ void __launch_bounds__(256) exl3_reconstruct_kernel(
        half * __restrict__ w, const uint32_t * __restrict__ trellis, const int n_tiles_total, const int n_tile_offset) {
    constexpr int TW = bits * 8;
    __shared__ uint32_t s_packed[8][TW];
    __shared__ half2    tile[16][8][8];

    const int t    = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int k    = blockIdx.y;
    const int n    = blockIdx.x * 8;
    const int out_tiles_n = gridDim.x * 8;

    const uint32_t * src = trellis + ((size_t) k * n_tiles_total + n_tile_offset + n) * TW;
    if (t < 16 * bits) {
        ((int4 *) s_packed)[t] = ((const int4 *) src)[t];
    }
    __syncthreads();

    exl3_frag_b f0, f1;
    exl3_dq_tile<bits>(s_packed[warp], lane * 8, f0, f1);

    // tensor-core fragment layout -> row-major tile
    const half2 n0 = __shfl_down_sync(0xffffffff, f0.v[0], 4, 32);
    const half2 n1 = __shfl_down_sync(0xffffffff, f0.v[1], 4, 32);
    const half2 n2 = __shfl_down_sync(0xffffffff, f1.v[0], 4, 32);
    const half2 n3 = __shfl_down_sync(0xffffffff, f1.v[1], 4, 32);

    if (!(lane & 4)) {
        const half2 m0 = __halves2half2(__low2half(f0.v[0]),  __low2half(n0));
        const half2 m1 = __halves2half2(__high2half(f0.v[0]), __high2half(n0));
        const half2 m2 = __halves2half2(__low2half(f0.v[1]),  __low2half(n1));
        const half2 m3 = __halves2half2(__high2half(f0.v[1]), __high2half(n1));
        const half2 m4 = __halves2half2(__low2half(f1.v[0]),  __low2half(n2));
        const half2 m5 = __halves2half2(__high2half(f1.v[0]), __high2half(n2));
        const half2 m6 = __halves2half2(__low2half(f1.v[1]),  __low2half(n3));
        const half2 m7 = __halves2half2(__high2half(f1.v[1]), __high2half(n3));
        const int r0 = (lane % 4) * 2;
        const int r1 = r0 + 1;
        const int r2 = r0 + 8;
        const int r3 = r0 + 9;
        const int c0 = lane / 8;
        const int c1 = c0 + 4;
        tile[r0][warp][c0] = m0;
        tile[r1][warp][c0] = m1;
        tile[r2][warp][c0] = m2;
        tile[r3][warp][c0] = m3;
        tile[r0][warp][c1] = m4;
        tile[r1][warp][c1] = m5;
        tile[r2][warp][c1] = m6;
        tile[r3][warp][c1] = m7;
    }
    __syncthreads();

    const int r = t / 16;
    const int c = t % 16;
    int4 * out = ((int4 *) w) + ((size_t) k * 16 + r) * 2 * out_tiles_n + n * 2 + c;
    *out = ((const int4 *) tile)[t];
}

bool ggml_cuda_mul_mat_exl3_supported(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * suh  = dst->src[2];
    const ggml_tensor * svh  = dst->src[3];

    if (dst->op != GGML_OP_MUL_MAT || ggml_exl3_bits(src0->type) == 0) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE) {
        return false;
    }
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    return src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && K % 128 == 0 && N % 128 == 0 && src1->ne[0] == K &&
        src1->nb[0] == sizeof(float) &&
        suh != nullptr && suh->type == GGML_TYPE_F16 && ggml_nelements(suh) == K && ggml_is_contiguous(suh) &&
        svh != nullptr && svh->type == GGML_TYPE_F16 && ggml_nelements(svh) == N && ggml_is_contiguous(svh);
}

template <int bits>
static void exl3_mul_mat_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const half * suh = (const half *) dst->src[2]->data;
    const half * svh = (const half *) dst->src[3]->data;
    const uint32_t * trellis = (const uint32_t *) src0->data;

    const int K = (int) src0->ne[0];
    const int N = (int) src0->ne[1];
    const int m = (int) (src1->ne[1] * src1->ne[2] * src1->ne[3]);
    if (m == 0) {
        return;
    }
    cudaStream_t stream = ctx.stream();
    float * out = (float *) dst->data;

    static const int small_max = [] {
        const char * e = getenv("LLAMA_EXL3_SMALL_MAX");
        return e ? std::max(0, std::min(EXL3_SMALL_MAX, atoi(e))) : EXL3_SMALL_MAX;
    }();

    ggml_cuda_pool_alloc<half> xh(ctx.pool(), (size_t) m * K);
    exl3_had_in_kernel<<<dim3(K / 128, m), 32, 0, stream>>>(
        (const char *) src1->data, xh.get(), suh, K, (int) src1->ne[1], (int) src1->ne[2], src1->nb[1], src1->nb[2], src1->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    if (m <= small_max) {
        // fixed split of K: enough warps to fill the device, at least 8 k tiles per slice
        const int nsm      = ggml_cuda_info().devices[ctx.device].nsm;
        const int groups   = N / 32;
        const int kt_total = K / 16;
        int S = (nsm * 24 + groups - 1) / groups;
        S = std::max(1, std::min(S, kt_total / 8));
        const int kt_per_slice = (kt_total + S - 1) / S;
        S = (kt_total + kt_per_slice - 1) / kt_per_slice;

        ggml_cuda_pool_alloc<float> part_alloc;
        float * part = out;
        if (S > 1) {
            part = part_alloc.alloc(ctx.pool(), (size_t) S * m * N);
        }
        const dim3 grid(N / 128, S);
        switch ((m + 15) / 16) {
            case 1: exl3_mm_small_kernel<bits, 1><<<grid, 128, 0, stream>>>(xh.get(), trellis, part, m, K, N, kt_per_slice); break;
            case 2: exl3_mm_small_kernel<bits, 2><<<grid, 128, 0, stream>>>(xh.get(), trellis, part, m, K, N, kt_per_slice); break;
            case 3: exl3_mm_small_kernel<bits, 3><<<grid, 128, 0, stream>>>(xh.get(), trellis, part, m, K, N, kt_per_slice); break;
            case 4: exl3_mm_small_kernel<bits, 4><<<grid, 128, 0, stream>>>(xh.get(), trellis, part, m, K, N, kt_per_slice); break;
            default: GGML_ABORT("EXL3 small path: too many rows");
        }
        CUDA_CHECK(cudaGetLastError());
        exl3_had_out_kernel<<<dim3(N / 128, m), 32, 0, stream>>>(part, out, svh, S, (size_t) m * N, (size_t) N, (size_t) N);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // wide batches: f16 weight slices + cuBLAS (f16 inputs, f32 accumulation), then the output transform in place
    const int ns_max = std::min(N, 1024);
    ggml_cuda_pool_alloc<half> wbuf(ctx.pool(), (size_t) K * ns_max);
    cublasHandle_t handle = ctx.cublas_handle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    for (int n0 = 0; n0 < N; n0 += ns_max) {
        const int ns = std::min(ns_max, N - n0);
        exl3_reconstruct_kernel<bits><<<dim3(ns / 128, K / 16), 256, 0, stream>>>(wbuf.get(), trellis, N / 16, n0 / 16);
        CUDA_CHECK(cudaGetLastError());
        // column-major view: C (ns x m, ld N) = W^T (ns x K, ld ns) * Xh^T (K x m, ld K)
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, ns, m, K,
                                  &alpha, wbuf.get(), CUDA_R_16F, ns,
                                          xh.get(),   CUDA_R_16F, K,
                                  &beta,  out + n0,   CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    exl3_had_out_kernel<<<dim3(N / 128, m), 32, 0, stream>>>(out, out, svh, 1, 0, (size_t) N, (size_t) N);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_mul_mat_exl3_supported(dst));
    switch (ggml_exl3_bits(dst->src[0]->type)) {
        case 1: exl3_mul_mat_impl<1>(ctx, dst); break;
        case 2: exl3_mul_mat_impl<2>(ctx, dst); break;
        case 3: exl3_mul_mat_impl<3>(ctx, dst); break;
        case 4: exl3_mul_mat_impl<4>(ctx, dst); break;
        case 5: exl3_mul_mat_impl<5>(ctx, dst); break;
        case 6: exl3_mul_mat_impl<6>(ctx, dst); break;
        case 7: exl3_mul_mat_impl<7>(ctx, dst); break;
        case 8: exl3_mul_mat_impl<8>(ctx, dst); break;
        default: GGML_ABORT("unsupported EXL3 bit width");
    }
}
