#pragma once
// Direct quantized matmul for small batches (opt-in LLAMA_MMQ_IQ_DIRECT=1), up to 16 columns (decode, speculative
// verify, drafter blocks). Types: IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ4_XS, Q4_K, Q2_K (every quantized
// matmul type of the Swift GSQ-RCO IQ3_XXS GGUF, so with this on, all of its small-batch matmuls share one kernel).
//
// Why: ncu on the MMQ kernel (RTX 3060, batch 16) showed the IQ path limited by its block-synchronous structure: every
// 256-weight K step decodes into a shared-memory tile, synchronizes the whole CTA four times and reads the tile back
// for the MMA, at 1-2 CTAs per SM, so decode latency, barriers and the MMA never overlap (DRAM at ~37% of peak).
//
// What: a CTA owns 16*MT rows (MT = 2 for <= 8 columns, 1 for 16: halves the accumulators where two 8-column n-tiles
// are live); its 4 warps split K 4 ways. Each warp streams the raw blocks of its rows for its K steps into a
// double-buffered warp-private shared buffer with cp.async (next step in flight while this one is decoded, no
// registers held, __syncwarp only) and decodes exactly the 4-weight groups its tensor-core A fragments need: no weight
// tile, no ldmatrix, no CTA barrier in the K loop. IQ signs come from a 256-entry byte->mask table (both sign encodings
// use bit k = sign of weight k); grid tables are copied to shared memory where they fit; d / scales / signs are
// decoded once per (row, 32 weights). One-scale-per-32 types use one m16n8k32 MMA per 32 weights, per-16-scale types
// two m16n8k16. Activation fragments and scales are read straight from the q8_1 MMQ layout of the type (D4 / DS4 /
// D2S6) through L1. Min / offset terms: Q4_K and IQ1_S use the per-32 activation sums of the DS4 layout (as MMQ does);
// Q2_K (per-16 mins, D2S6 layout) gets exact per-16 sums of the quantized activations from an extra m16n8k16 MMA
// against an all-ones A. The K slices are added in a fixed order at the end.
//
// Exactness: every output column is computed with the same operations in the same order whatever the number of
// columns (8-column n-tiles, int32 MMA, fixed epilogue in K order, slices 0..KS-1), so results are batch-width
// invariant (spec == greedy). The fp32 order differs from MMQ, so the output is not bit-identical to the default path.
//
// Blocks of 2 mod 4 bytes (all IQ2/IQ3/IQ1 types) are only 2-byte aligned: each row is copied from the 4-byte word at
// or before the block start, so a staged block starts at byte o = 0 or 2 and multi-byte fields are read as 16-bit
// halves. The host only takes this path when a weight row is a multiple of 4 bytes (no read past the tensor end).

#include "common.cuh"

#define MMQ_IQD_KS 4 // K slices = warps per CTA

static __device__ __forceinline__ uint32_t iqd_ld32(const uint8_t * p) { // 2-byte aligned 32-bit read
    return (uint32_t) *(const uint16_t *) p | ((uint32_t) *(const uint16_t *) (p + 2) << 16);
}

// Apply 4 sign bits (bit k = sign of byte k) to 4 grid bytes. Every grid value of the signed IQ types is in 1..127
// (IQ2: 0x08/0x19/0x2b, IQ3_XXS >= 4, IQ3_S odd), so negating a byte as (g ^ 0xFF) + 1 never carries into the next
// byte and a plain 32-bit add does it for all four. The bit spread (nib * 0x00204081: bit k -> bit 8k, no overlapping
// partial products) replaces a lookup in a 256-entry shared mask table (random 64-bit LDS with bank conflicts).
static __device__ __forceinline__ int iqd_sgn4(const int g, const uint32_t nib) {
    const uint32_t y = ((nib & 0x0Fu) * 0x00204081u) & 0x01010101u;
    return (int) (((uint32_t) g ^ (y * 0xFFu)) + y);
}

enum mmq_iqd_ylayout { IQD_Y_D4 = 0, IQD_Y_DS4 = 1, IQD_Y_D2S6 = 2 };
enum mmq_iqd_minmode { IQD_MIN_NONE = 0, IQD_MIN_YSUM32 = 1, IQD_MIN_ONES16 = 2 };

// Per type:
//   bs, table_words (0 = no shared copy), scale16, minmode, ylayout
//   rowc(b)  -> float2 per-block constants (d, dmin) for the staged block at b
//   decode2(b, s, t, rc, grid, smask, lo, hi, sl, sh, ml, mh): for sub-block s (32 weights) and lane-in-quad t,
//     lo = 4-weight group t (first 16 weights), hi = group t + 4 (second 16), sl/sh = x scales of the halves;
//     perm types (one scale per 32) instead return groups 2t / 2t+1 (adjacent bytes, one shared sign byte); the
//     kernel feeds the activation fragments in the same order, and the int8 MMA sum over the 32 is exact, so the
//     result is identical;
//     ml/mh = additive min/offset coefficients of the halves (multiplied with the activation sum of that half / 32)
// Same arithmetic as mmq-load-tiles.cuh for the type.
template <ggml_type type> struct mmq_iqd_traits;

template <> struct mmq_iqd_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int bs = 98, table_words = 256; static constexpr bool scale16 = false; static constexpr bool perm = true; 
    static constexpr int minmode = IQD_MIN_NONE, ylayout = IQD_Y_D4;
    static __device__ __forceinline__ const uint32_t * gtable() { return iq3xxs_grid; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return make_float2(__half2float(*(const half *) b), 0.0f); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        const float d = rc.x;
        const uint8_t * qs  = b + 2;
        const uint32_t  aux = iqd_ld32(qs + 64 + 4*s);
        const uint32_t q2 = *(const uint16_t *) (qs + 8*s + 2*t);  // grid indices of groups 2t, 2t+1
        uint32_t v = (aux >> (7*t)) & 0x7F; v ^= (__popc(v) & 1) << 7; // their shared sign byte
        lo = iqd_sgn4((int) grid[q2 & 0xFF], v);
        hi = iqd_sgn4((int) grid[q2 >> 8],   v >> 4);
        const int ls = aux >> 28;
        sl = sh = (ls*d + d/2)/2;
        ml = mh = 0.0f;
    }
};

template <> struct mmq_iqd_traits<GGML_TYPE_IQ3_S> {
    static constexpr int bs = 110, table_words = 512; static constexpr bool scale16 = false; static constexpr bool perm = true; 
    static constexpr int minmode = IQD_MIN_NONE, ylayout = IQD_Y_D4;
    static __device__ __forceinline__ const uint32_t * gtable() { return iq3s_grid; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return make_float2(__half2float(*(const half *) b), 0.0f); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        const float d = rc.x;
        const uint8_t * qs    = b + 2;
        const int       qh    = b[66 + s];
        const uint8_t * signs = b + 74 + 4*s;
        const uint32_t q2 = *(const uint16_t *) (qs + 8*s + 2*t);  // grid index low bytes of groups 2t, 2t+1
        const uint32_t v = signs[t];                                 // their shared sign byte
        const int j0 = 2*t, j1 = 2*t + 1;
        lo = iqd_sgn4((int) grid[(q2 & 0xFF) | ((qh << (8 - j0)) & 0x100)], v);
        hi = iqd_sgn4((int) grid[(q2 >> 8)   | ((qh << (8 - j1)) & 0x100)], v >> 4);
        const int scb = b[106 + s/2];
        const int ls = 1 + 2*((scb >> (((2*s) << 1) & 0x04)) & 0x0F);
        sl = sh = ls*d;
        ml = mh = 0.0f;
    }
};

template <> struct mmq_iqd_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int bs = 66, table_words = 512; static constexpr bool scale16 = false; static constexpr bool perm = true; 
    static constexpr int minmode = IQD_MIN_NONE, ylayout = IQD_Y_D4;
    static __device__ __forceinline__ const uint32_t * gtable() { return (const uint32_t *) iq2xxs_grid; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return make_float2(__half2float(*(const half *) b), 0.0f); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        const float d = rc.x;
        const uint8_t * qs  = b + 2;
        const uint32_t  aux = iqd_ld32(qs + 8*s + 4);
        const uint2 gp = *(const uint2 *) (grid + 2*qs[8*s + t]);   // both 4-weight halves of grid entry t
        uint32_t v = (aux >> (7*t)) & 0x7F; v ^= (__popc(v) & 1) << 7;
        lo = iqd_sgn4((int) gp.x, v);
        hi = iqd_sgn4((int) gp.y, v >> 4);
        const int ls = aux >> 27 | 1;
        sl = sh = d * ls / 8;
        ml = mh = 0.0f;
    }
};

template <> struct mmq_iqd_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int bs = 74, table_words = 1024; static constexpr bool scale16 = true;  static constexpr bool perm = false;
    static constexpr int minmode = IQD_MIN_NONE, ylayout = IQD_Y_D4;
    static __device__ __forceinline__ const uint32_t * gtable() { return (const uint32_t *) iq2xs_grid; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return make_float2(__half2float(*(const half *) b), 0.0f); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        const float d = rc.x;
        const uint16_t * q2 = (const uint16_t *) (b + 2 + 8*s);
        const int l0 = t >> 1, l1 = 2 + (t >> 1), h = t & 1;
        const int q0 = q2[l0], q1 = q2[l1];
        uint32_t v0 = q0 >> 9; v0 ^= (__popc(v0) & 1) << 7;
        uint32_t v1 = q1 >> 9; v1 ^= (__popc(v1) & 1) << 7;
        lo = iqd_sgn4((int) grid[2*(q0 & 0x1FF) + h], v0 >> (4*h));
        hi = iqd_sgn4((int) grid[2*(q1 & 0x1FF) + h], v1 >> (4*h));
        const int ls = b[66 + s];
        sl = ((ls & 0x0F)*d + d/2)/4;
        sh = ((ls >>   4)*d + d/2)/4;
        ml = mh = 0.0f;
    }
};

template <> struct mmq_iqd_traits<GGML_TYPE_IQ2_S> {
    static constexpr int bs = 82, table_words = 2048; static constexpr bool scale16 = true;  static constexpr bool perm = false;
    static constexpr int minmode = IQD_MIN_NONE, ylayout = IQD_Y_D4;
    static __device__ __forceinline__ const uint32_t * gtable() { return (const uint32_t *) iq2s_grid; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return make_float2(__half2float(*(const half *) b), 0.0f); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        const float d = rc.x;
        const uint8_t * qs    = b + 2 + 4*s;
        const uint8_t * signs = b + 2 + 32 + 4*s;
        const int       qh    = b[66 + s];
        const int l0 = t >> 1, l1 = 2 + (t >> 1), h = t & 1;
        lo = iqd_sgn4((int) grid[2*(qs[l0] | ((qh << (8 - 2*l0)) & 0x300)) + h], (uint32_t) signs[l0] >> (4*h));
        hi = iqd_sgn4((int) grid[2*(qs[l1] | ((qh << (8 - 2*l1)) & 0x300)) + h], (uint32_t) signs[l1] >> (4*h));
        const int ls = b[74 + s];
        sl = ((ls & 0x0F)*d + d/2)/4;
        sh = ((ls >>   4)*d + d/2)/4;
        ml = mh = 0.0f;
    }
};

// IQ1_S: x = d1q*(q + delta), q in {0,1,2} from iq1s_grid_gpu (4-bit fields); the delta term uses the DS4 sum
template <> struct mmq_iqd_traits<GGML_TYPE_IQ1_S> {
    static constexpr int bs = 50, table_words = 0; static constexpr bool scale16 = false; static constexpr bool perm = true;  // 8 KiB grid stays global
    static constexpr int minmode = IQD_MIN_YSUM32, ylayout = IQD_Y_DS4;
    static __device__ __forceinline__ const uint32_t * gtable() { return iq1s_grid_gpu; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return make_float2(__half2float(*(const half *) b), 0.0f); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        GGML_UNUSED(smask);
        const uint8_t * qs = b + 2 + 4*s;
        const int       qh = *(const uint16_t *) (b + 34 + 2*s);
        const uint32_t g = grid[qs[t] | (((qh >> (3*t)) & 0x07) << 8)]; // groups 2t, 2t+1 = both nibble halves
        lo = (int) ( g       & 0x0F0F0F0F);
        hi = (int) ((g >> 4) & 0x0F0F0F0F);
        const float d1q   = rc.x * (((qh >> 11) & 0x0E) + 1);
        const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
        sl = sh = d1q;
        ml = mh = d1q*delta;
    }
};

// IQ4_XS: 4-bit indices into kvalues_iq4nl, one 6-bit scale per 32
template <> struct mmq_iqd_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int bs = 136, table_words = 0; static constexpr bool scale16 = false; static constexpr bool perm = false;
    static constexpr int minmode = IQD_MIN_NONE, ylayout = IQD_Y_D4;
    static __device__ __forceinline__ const uint32_t * gtable() { return nullptr; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return make_float2(__half2float(*(const half *) b), 0.0f); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        GGML_UNUSED(grid); GGML_UNUSED(smask);
        const int q4 = *(const int *) (b + 8 + 16*s + 4*t);       // bytes 4t..4t+3 of the sub-block's 16
        const int2 v = get_int_from_table_16(q4, kvalues_iq4nl);  // low nibbles = weights 0..15, high = 16..31
        lo = v.x;
        hi = v.y;
        const int scales_h = *(const uint16_t *) (b + 2);
        const int ls = ((b[4 + s/2] >> (4*(s % 2))) & 0x0F) | (((scales_h >> (2*s)) & 0x03) << 4);
        sl = sh = rc.x * (ls - 32);
        ml = mh = 0.0f;
    }
};

static __device__ __forceinline__ void iqd_scale_min_k4(const int j, const uint8_t * q, int & d, int & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

// Q4_K: x = d*sc*q - dmin*m per 32; the min term uses the DS4 sum
template <> struct mmq_iqd_traits<GGML_TYPE_Q4_K> {
    static constexpr int bs = 144, table_words = 0; static constexpr bool scale16 = false; static constexpr bool perm = true; 
    static constexpr int minmode = IQD_MIN_YSUM32, ylayout = IQD_Y_DS4;
    static __device__ __forceinline__ const uint32_t * gtable() { return nullptr; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return __half22float2(*(const half2 *) b); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        GGML_UNUSED(grid); GGML_UNUSED(smask);
        const int * qs = (const int *) (b + 16) + 8*(s >> 1);
        const int sh4 = 4*(s & 1);
        lo = (qs[2*t]     >> sh4) & 0x0F0F0F0F;
        hi = (qs[2*t + 1] >> sh4) & 0x0F0F0F0F;
        int sc, m;
        iqd_scale_min_k4(s, b + 4, sc, m);
        sl = sh = rc.x * sc;
        ml = mh = -rc.y * m;
    }
};

// Q2_K: x = d*sc*q - dmin*m per 16; the per-16 activation sums come from an all-ones MMA (D2S6 has none for all 16s)
template <> struct mmq_iqd_traits<GGML_TYPE_Q2_K> {
    static constexpr int bs = 84, table_words = 0; static constexpr bool scale16 = true;  static constexpr bool perm = false;
    static constexpr int minmode = IQD_MIN_ONES16, ylayout = IQD_Y_D2S6;
    static __device__ __forceinline__ const uint32_t * gtable() { return nullptr; }
    static __device__ __forceinline__ float2 rowc(const uint8_t * b) { return __half22float2(*(const half2 *) (b + 80)); }
    static __device__ __forceinline__ void decode2(const uint8_t * b, const int s, const int t, const float2 rc,
            const uint32_t * grid, const uint2 * smask, int & lo, int & hi, float & sl, float & sh, float & ml, float & mh) {
        GGML_UNUSED(grid); GGML_UNUSED(smask);
        const int * qs = (const int *) (b + 16) + 8*(s >> 2);
        const int sh2 = 2*(s & 3);
        lo = (qs[t]     >> sh2) & 0x03030303;
        hi = (qs[t + 4] >> sh2) & 0x03030303;
        const int sc0 = b[2*s + 0], sc1 = b[2*s + 1];
        sl = rc.x * (sc0 & 0x0F);
        sh = rc.x * (sc1 & 0x0F);
        ml = -rc.y * (sc0 >> 4);
        mh = -rc.y * (sc1 >> 4);
    }
};

static constexpr __host__ __device__ bool mmq_iqd_supported(const ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS  || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ1_S   ||
           type == GGML_TYPE_IQ4_XS  || type == GGML_TYPE_Q4_K  || type == GGML_TYPE_Q2_K;
}

static constexpr __host__ __device__ int mmq_iqd_block_bytes(const ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS ?  98 : type == GGML_TYPE_IQ3_S  ? 110 : type == GGML_TYPE_IQ2_XXS ? 66 :
           type == GGML_TYPE_IQ2_XS  ?  74 : type == GGML_TYPE_IQ2_S  ?  82 : type == GGML_TYPE_IQ1_S   ? 50 :
           type == GGML_TYPE_IQ4_XS  ? 136 : type == GGML_TYPE_Q4_K   ? 144 : type == GGML_TYPE_Q2_K    ? 84 : 0;
}


template <ggml_type type>
static constexpr __host__ __device__ int mmq_iqd_words() { // 4-byte words copied per row
    constexpr int bs = mmq_iqd_traits<type>::bs;
    return bs % 4 == 0 ? bs / 4 : (bs + 2 + 3) / 4;          // 2-mod-4 blocks may start 2 bytes into a word
}

template <ggml_type type, int NT, int MT>
static constexpr __host__ __device__ int mmq_iqd_stage_bytes() {
    constexpr int stage = 2 * MMQ_IQD_KS * 16*MT * mmq_iqd_words<type>() * 4;            // double buffered
    constexpr int part  = MMQ_IQD_KS * 32 * MT * NT * 4 * (int) sizeof(float);           // aliases the staging
    return stage > part ? stage : part;
}

template <ggml_type type, int NT, int MT>
static constexpr __host__ __device__ int mmq_iqd_smem_bytes() {
    return mmq_iqd_stage_bytes<type, NT, MT>() + mmq_iqd_traits<type>::table_words * (int) sizeof(uint32_t);
}

static __device__ __forceinline__ void mmq_iqd_mma_k16(int (&c)[4], const int a0, const int a1, const int b0) {
#if defined(TURING_MMA_AVAILABLE)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    GGML_UNUSED_VARS(c, a0, a1, b0);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

static __device__ __forceinline__ void mmq_iqd_mma_k32(int (&c)[4], const int a0, const int a1, const int a2, const int a3,
        const int b0, const int b1) {
#if defined(AMPERE_MMA_AVAILABLE)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    int c0[4], c1[4];
    mmq_iqd_mma_k16(c0, a0, a1, b0);
    mmq_iqd_mma_k16(c1, a2, a3, b1);
    c[0] = c0[0] + c1[0]; c[1] = c0[1] + c1[1]; c[2] = c0[2] + c1[2]; c[3] = c0[3] + c1[3];
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __device__ __forceinline__ void iqd_cp_async4(void * smem_dst, const void * gmem_src) {
#if defined(AMPERE_MMA_AVAILABLE)
    const unsigned int d = (unsigned int) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(d), "l"(gmem_src));
#else
    *(int *) smem_dst = *(const int *) gmem_src;
#endif // defined(AMPERE_MMA_AVAILABLE)
}
static __device__ __forceinline__ void iqd_cp_commit() {
#if defined(AMPERE_MMA_AVAILABLE)
    asm volatile("cp.async.commit_group;");
#endif // defined(AMPERE_MMA_AVAILABLE)
}
template <int N>
static __device__ __forceinline__ void iqd_cp_wait() {
#if defined(AMPERE_MMA_AVAILABLE)
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
#endif // defined(AMPERE_MMA_AVAILABLE)
}

// activation scale (and DS4 sum) of column block p (36 ints) for 32-value sub-block so
template <int ylayout>
static __device__ __forceinline__ float2 iqd_yscale(const int * p, const int so) {
    if constexpr (ylayout == IQD_Y_D4) {
        return make_float2(((const float *) p)[so], 0.0f);
    } else if constexpr (ylayout == IQD_Y_DS4) {
        return __half22float2(((const half2 *) p)[so]);
    } else {
        return make_float2(__half2float(((const half *) p)[so >> 1]), 0.0f); // D2S6: one scale per 64 values
    }
}

template <ggml_type type, int NT, int MT>
__launch_bounds__(MMQ_IQD_KS*32, 4)
static __global__ void mul_mat_iq_direct(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int ncols_dst,
        const int stride_col_dst) {
#if defined(TURING_MMA_AVAILABLE)
    using T = mmq_iqd_traits<type>;
    constexpr int ROWS = 16*MT; // MT 16-row MMA tiles per warp (chosen per launch; results do not depend on it)
    constexpr int bs   = T::bs;
    constexpr int NWR  = mmq_iqd_words<type>();                // words per staged row
    constexpr int SLOT = ROWS*NWR*4;                           // bytes per warp per buffer
    constexpr int NWC  = ROWS*NWR;                             // words copied per warp per K step
    constexpr int YB   = sizeof(block_q8_1_mmq) / sizeof(int); // 36 ints per 128-value activation block

    extern __shared__ int iqd_smem[];
    uint8_t  * stage_all = (uint8_t *) iqd_smem;
    const uint2 * smask  = nullptr; // sign masks are computed (iqd_sgn4); kept in the decode2 signature
    uint32_t * sgrid     = (uint32_t *) (stage_all + mmq_iqd_stage_bytes<type, NT, MT>());
    float    * part      = (float *) stage_all; // after the K loop

    const int lane = threadIdx.x;
    const int ks   = threadIdx.y;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    const uint32_t * grid = T::gtable();
    if constexpr (T::table_words > 0) {
        for (int i = ks*32 + lane; i < T::table_words; i += MMQ_IQD_KS*32) {
            sgrid[i] = grid[i];
        }
        grid = sgrid;
    }
    __syncthreads();

    const int row0 = blockIdx.x*ROWS;
    uint8_t * slot0 = stage_all + (2*ks + 0)*SLOT;
    uint8_t * slot1 = stage_all + (2*ks + 1)*SLOT;

    // Staging addresses: a weight row is a multiple of 4 bytes (host guard), so the word alignment of block kb is the
    // same for every row ((kb*bs) & 3) and a row's staged words start at row_base + ((kb*bs) & ~3). Each lane owns fixed
    // (row phase, word) pairs: RPI rows per pass when a row has < 32 words, and the row pointer advances by whole rows
    // (clamped to the last row only in the edge CTA), instead of a divide, clamp and 64-bit multiply per word.
    // Packed mapping as before (word f = lane + 32*i of the warp's ROWS x NWR staged words, so the shared destination is
    // just slot + 4*f), but the lane's (row, word, source offset) advance incrementally by 32 words per pass.
    constexpr int DR = 32 / NWR, DW = 32 % NWR;         // rows / words advanced per pass of 32 words
    const int  rowbytes = stride_row_x*bs;               // < 2^31 (int stride in blocks)
    const char * xr   = x + (int64_t) row0*rowbytes;
    const int    rlim = nrows_x - 1 - row0;
    const bool   full = rlim >= ROWS - 1;
    const int    lr0  = lane / NWR;
    const int    lw0  = lane - lr0*NWR;

    auto issue = [&](const int kb, uint8_t * slot) {
        const char * base = xr + ((kb*bs) & ~3);
        int r = lr0, w = lw0;
        int soff = lr0*rowbytes + 4*lw0;
        // full unroll for <= 32-word rows; the 34/36-word types (IQ4_XS, Q4_K) measured 20% slower fully unrolled
#pragma unroll (NWR > 32 ? 4 : NWC/32 + 1)
        for (int f0 = 0; f0 < NWC; f0 += 32) {
            if (f0 + 32 <= NWC || f0 + lane < NWC) {
                const int so = full ? soff : min(r, rlim)*rowbytes + 4*w;
                iqd_cp_async4(slot + 4*(f0 + lane), base + so);
            }
            r += DR; w += DW; soff += DR*rowbytes + 4*DW;
            if (w >= NWR) {
                w -= NWR; ++r; soff += rowbytes - 4*NWR;
            }
        }
        iqd_cp_commit();
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;
        }
    }

    int kb = ks;
    if (kb < nkb) {
        issue(kb, slot0);
    }
    int buf = 0;
    for (; kb < nkb; kb += MMQ_IQD_KS) {
        if (kb + MMQ_IQD_KS < nkb) {
            issue(kb + MMQ_IQD_KS, buf ? slot0 : slot1); // next step in flight
            iqd_cp_wait<1>();
        } else {
            iqd_cp_wait<0>();
        }
        __syncwarp();
        const uint8_t * slot = buf ? slot1 : slot0;

        const uint8_t * brow[MT][2];
        float2          rcon[MT][2];
        const int o = (kb*bs) & 3; // 0 or 2, same for every row (row bytes % 4 == 0)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const int r   = mt*16 + hr*8 + g;
                brow[mt][hr] = slot + r*NWR*4 + o;
                rcon[mt][hr] = T::rowc(brow[mt][hr]);
            }
        }

        const int * yk = y + (int64_t) (2*kb)*ncols_y*YB;
        // D4 / DS4: the 4 sub-block scales of a 128-value block are its first 16 bytes; one 128-bit load per column
        // per 4 sub-blocks instead of a scalar load per sub-block (same values)
        constexpr bool yvec = T::ylayout == IQD_Y_D4 || T::ylayout == IQD_Y_DS4;
        int4 yh0[NT], yh1[NT];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int * yb = yk + (s >> 2)*ncols_y*YB; // 128-value activation block
            const int   so = s & 3;                    // its 32-value sub-block
            int    b0[NT], b1[NT];
            float2 yc0[NT], yc1[NT];                   // (scale, DS4 sum) of columns 2t, 2t+1
            if constexpr (yvec) {
                if (so == 0) {
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        yh0[nt] = *(const int4 *) (yb + (nt*8 + 2*t + 0)*YB);
                        yh1[nt] = *(const int4 *) (yb + (nt*8 + 2*t + 1)*YB);
                    }
                }
            } else {
                GGML_UNUSED(yh0); GGML_UNUSED(yh1);
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int * yc = yb + (nt*8 + g)*YB + 4 + so*8;
                if constexpr (T::perm) { // k slots t / t+4 carry groups 2t / 2t+1 (same permutation as decode2)
                    const int2 v = *(const int2 *) (yc + 2*t);
                    b0[nt] = v.x;
                    b1[nt] = v.y;
                } else {
                    b0[nt] = yc[t];
                    b1[nt] = yc[t + 4];
                }
                if constexpr (yvec) {
                    const int w0 = so == 0 ? yh0[nt].x : so == 1 ? yh0[nt].y : so == 2 ? yh0[nt].z : yh0[nt].w;
                    const int w1 = so == 0 ? yh1[nt].x : so == 1 ? yh1[nt].y : so == 2 ? yh1[nt].z : yh1[nt].w;
                    if constexpr (T::ylayout == IQD_Y_D4) {
                        yc0[nt] = make_float2(__int_as_float(w0), 0.0f);
                        yc1[nt] = make_float2(__int_as_float(w1), 0.0f);
                    } else {
                        yc0[nt] = __half22float2(*(const half2 *) &w0);
                        yc1[nt] = __half22float2(*(const half2 *) &w1);
                    }
                } else {
                    yc0[nt] = iqd_yscale<T::ylayout>(yb + (nt*8 + 2*t + 0)*YB, so);
                    yc1[nt] = iqd_yscale<T::ylayout>(yb + (nt*8 + 2*t + 1)*YB, so);
                }
            }
            int o0[NT][2], o1[NT][2]; // all-ones MMA: per-16 activation sums of columns 2t, 2t+1 (Q2_K)
            if constexpr (T::minmode == IQD_MIN_ONES16) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    int c[4], e[4];
                    mmq_iqd_mma_k16(c, 0x01010101, 0x01010101, b0[nt]);
                    mmq_iqd_mma_k16(e, 0x01010101, 0x01010101, b1[nt]);
                    o0[nt][0] = c[0]; o0[nt][1] = c[1];
                    o1[nt][0] = e[0]; o1[nt][1] = e[1];
                }
            } else {
                GGML_UNUSED(o0); GGML_UNUSED(o1);
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                int lo0, hi0, lo1, hi1; float sl0, sh0, sl1, sh1, ml0, mh0, ml1, mh1;
                T::decode2(brow[mt][0], s, t, rcon[mt][0], grid, smask, lo0, hi0, sl0, sh0, ml0, mh0);
                T::decode2(brow[mt][1], s, t, rcon[mt][1], grid, smask, lo1, hi1, sl1, sh1, ml1, mh1);
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const float dB0 = yc0[nt].x, dB1 = yc1[nt].x;
                    if constexpr (!T::scale16) {
                        int c[4];
                        mmq_iqd_mma_k32(c, lo0, lo1, hi0, hi1, b0[nt], b1[nt]);
                        if constexpr (T::minmode == IQD_MIN_YSUM32) {
                            const float sB0 = yc0[nt].y, sB1 = yc1[nt].y;
                            acc[mt][nt][0] += (float) c[0] * sl0 * dB0 + ml0 * sB0;
                            acc[mt][nt][1] += (float) c[1] * sl0 * dB1 + ml0 * sB1;
                            acc[mt][nt][2] += (float) c[2] * sl1 * dB0 + ml1 * sB0;
                            acc[mt][nt][3] += (float) c[3] * sl1 * dB1 + ml1 * sB1;
                        } else {
                            acc[mt][nt][0] += (float) c[0] * sl0 * dB0;
                            acc[mt][nt][1] += (float) c[1] * sl0 * dB1;
                            acc[mt][nt][2] += (float) c[2] * sl1 * dB0;
                            acc[mt][nt][3] += (float) c[3] * sl1 * dB1;
                        }
                    } else {
                        int c[4], e[4];
                        mmq_iqd_mma_k16(c, lo0, lo1, b0[nt]);
                        mmq_iqd_mma_k16(e, hi0, hi1, b1[nt]);
                        if constexpr (T::minmode == IQD_MIN_ONES16) {
                            acc[mt][nt][0] += ((float) c[0] * sl0 + ml0 * (float) o0[nt][0] + (float) e[0] * sh0 + mh0 * (float) o1[nt][0]) * dB0;
                            acc[mt][nt][1] += ((float) c[1] * sl0 + ml0 * (float) o0[nt][1] + (float) e[1] * sh0 + mh0 * (float) o1[nt][1]) * dB1;
                            acc[mt][nt][2] += ((float) c[2] * sl1 + ml1 * (float) o0[nt][0] + (float) e[2] * sh1 + mh1 * (float) o1[nt][0]) * dB0;
                            acc[mt][nt][3] += ((float) c[3] * sl1 + ml1 * (float) o0[nt][1] + (float) e[3] * sh1 + mh1 * (float) o1[nt][1]) * dB1;
                        } else {
                            acc[mt][nt][0] += ((float) c[0] * sl0 + (float) e[0] * sh0) * dB0;
                            acc[mt][nt][1] += ((float) c[1] * sl0 + (float) e[1] * sh0) * dB1;
                            acc[mt][nt][2] += ((float) c[2] * sl1 + (float) e[2] * sh1) * dB0;
                            acc[mt][nt][3] += ((float) c[3] * sl1 + (float) e[3] * sh1) * dB1;
                        }
                    }
                }
            }
        }
        __syncwarp(); // all lanes done with this slot before it is refilled
        buf ^= 1;
    }

    // fixed-order sum of the K slices (partials reuse the staging buffers)
    __syncthreads();
    constexpr int ntiles = MT*NT;
    float * my = part + (ks*32 + lane)*ntiles*4;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                my[(mt*NT + nt)*4 + q] = acc[mt][nt][q];
            }
        }
    }
    __syncthreads();
    for (int tile = ks; tile < ntiles; tile += MMQ_IQD_KS) { // warp ks writes tiles ks, ks + KS, ...
        const int mt = tile / NT;
        const int nt = tile % NT;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < MMQ_IQD_KS; ++k) {
                v += part[(k*32 + lane)*ntiles*4 + tile*4 + q];
            }
            const int row = row0 + mt*16 + g + (q >= 2 ? 8 : 0);
            const int col = nt*8 + 2*t + (q & 1);
            if (row < nrows_x && col < ncols_dst) {
                dst[(int64_t) col*stride_col_dst + row] = v;
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nrows_x, nkb, stride_row_x, ncols_y, ncols_dst, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}


// ---------------------------------------------------------------------------------------------------------------------
// Wide variant for prefill batches (opt-in LLAMA_MMQ_IQ_WIDE=1, MMQ tile width J = 128): same per-type decode, staging and
// per-32 epilogue as mul_mat_iq_direct, but the CTA's 8 warps split ROWS (16 each, 128 per CTA) instead of K, and every
// warp covers 8*NT columns. One decode of a weight fragment feeds NT MMAs; the 8 warps of a CTA read the same
// activation fragments through L1. No cross-warp reduction and no CTA barrier in the K loop. ncu on the MMQ prefill
// kernel (J = 128): 1 CTA/SM at 238-255 registers, tensor pipe 26-35% busy (decode -> barrier -> MMA phases never overlap).
template <ggml_type type, int NW, int MT>
static constexpr __host__ __device__ int mmq_iqw_smem_bytes() {
    return 2 * NW * 16*MT * mmq_iqd_words<type>() * 4 + mmq_iqd_traits<type>::table_words * (int) sizeof(uint32_t);
}

#ifndef MMQ_IQW_MINB
#define MMQ_IQW_MINB 2 // CTAs per SM for the 4-warp shape (3 spills: 381 vs 468 t/s)
#endif
template <ggml_type type, int NT, int NW, int MT>
__launch_bounds__(NW*32, NW == 4 ? MMQ_IQW_MINB : 1)
static __global__ void mul_mat_iq_wide(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int ncols_dst,
        const int stride_col_dst) {
#if defined(TURING_MMA_AVAILABLE)
    using T = mmq_iqd_traits<type>;
    constexpr int ROWS = 16*MT;
    constexpr int bs   = T::bs;
    constexpr int NWR  = mmq_iqd_words<type>();
    constexpr int SLOT = ROWS*NWR*4;
    constexpr int NWC  = ROWS*NWR;
    constexpr int YB   = sizeof(block_q8_1_mmq) / sizeof(int);

    extern __shared__ int iqd_smem[];
    uint8_t  * stage_all = (uint8_t *) iqd_smem;
    uint32_t * sgrid     = (uint32_t *) (stage_all + 2 * NW * SLOT);
    const uint2 * smask  = nullptr;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    const uint32_t * grid = T::gtable();
    if constexpr (T::table_words > 0) {
        for (int i = w*32 + lane; i < T::table_words; i += NW*32) {
            sgrid[i] = grid[i];
        }
        grid = sgrid;
    }
    __syncthreads();

    const int row0 = blockIdx.x*(ROWS*NW) + w*ROWS;
    if (row0 >= nrows_x) {
        return; // no CTA barrier below
    }
    const int col0 = blockIdx.y*(8*NT);
    y   += (int64_t) col0*YB;
    dst += (int64_t) col0*stride_col_dst;
    const int ncols = ncols_dst - col0;

    uint8_t * slot0 = stage_all + (2*w + 0)*SLOT;
    uint8_t * slot1 = stage_all + (2*w + 1)*SLOT;

    constexpr int DR = 32 / NWR, DW = 32 % NWR;
    const int  rowbytes = stride_row_x*bs;
    const char * xr   = x + (int64_t) row0*rowbytes;
    const int    rlim = nrows_x - 1 - row0;
    const bool   full = rlim >= ROWS - 1;
    const int    lr0  = lane / NWR;
    const int    lw0  = lane - lr0*NWR;

    auto issue = [&](const int kb, uint8_t * slot) {
        const char * base = xr + ((kb*bs) & ~3);
        int r = lr0, wd = lw0;
        int soff = lr0*rowbytes + 4*lw0;
#pragma unroll (NWR > 32 ? 4 : NWC/32 + 1)
        for (int f0 = 0; f0 < NWC; f0 += 32) {
            if (f0 + 32 <= NWC || f0 + lane < NWC) {
                const int so = full ? soff : min(r, rlim)*rowbytes + 4*wd;
                iqd_cp_async4(slot + 4*(f0 + lane), base + so);
            }
            r += DR; wd += DW; soff += DR*rowbytes + 4*DW;
            if (wd >= NWR) {
                wd -= NWR; ++r; soff += rowbytes - 4*NWR;
            }
        }
        iqd_cp_commit();
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.0f;
        }
    }

    issue(0, slot0);
    int buf = 0;
    for (int kb = 0; kb < nkb; ++kb) {
        if (kb + 1 < nkb) {
            issue(kb + 1, buf ? slot0 : slot1);
            iqd_cp_wait<1>();
        } else {
            iqd_cp_wait<0>();
        }
        __syncwarp();
        const uint8_t * slot = buf ? slot1 : slot0;
        const int o = (kb*bs) & 3;
        const uint8_t * brow[MT][2];
        float2          rcon[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                brow[mt][hr] = slot + (mt*16 + hr*8 + g)*NWR*4 + o;
                rcon[mt][hr] = T::rowc(brow[mt][hr]);
            }
        }

        const int * yk = y + (int64_t) (2*kb)*ncols_y*YB;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int * yb = yk + (s >> 2)*ncols_y*YB;
            const int   so = s & 3;
            int lo[MT][2], hi[MT][2]; float sl[MT][2], sh[MT][2], ml[MT][2], mh[MT][2];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
                for (int hr = 0; hr < 2; ++hr) {
                    T::decode2(brow[mt][hr], s, t, rcon[mt][hr], grid, smask, lo[mt][hr], hi[mt][hr], sl[mt][hr], sh[mt][hr], ml[mt][hr], mh[mt][hr]);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int * yc = yb + (nt*8 + g)*YB + 4 + so*8;
                int b0, b1;
                if constexpr (T::perm) {
                    const int2 v = *(const int2 *) (yc + 2*t);
                    b0 = v.x;
                    b1 = v.y;
                } else {
                    b0 = yc[t];
                    b1 = yc[t + 4];
                }
                const float2 yc0 = iqd_yscale<T::ylayout>(yb + (nt*8 + 2*t + 0)*YB, so);
                const float2 yc1 = iqd_yscale<T::ylayout>(yb + (nt*8 + 2*t + 1)*YB, so);
                const float dB0 = yc0.x, dB1 = yc1.x;
                int oc[4], oe[4];
                if constexpr (T::minmode == IQD_MIN_ONES16) {
                    mmq_iqd_mma_k16(oc, 0x01010101, 0x01010101, b0);
                    mmq_iqd_mma_k16(oe, 0x01010101, 0x01010101, b1);
                } else {
                    GGML_UNUSED(oc); GGML_UNUSED(oe);
                }
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const float sl0 = sl[mt][0], sl1 = sl[mt][1], sh0 = sh[mt][0], sh1 = sh[mt][1];
                    const float ml0 = ml[mt][0], ml1 = ml[mt][1], mh0 = mh[mt][0], mh1 = mh[mt][1];
                    float * a = acc[mt][nt];
                    if constexpr (!T::scale16) {
                        int c[4];
                        mmq_iqd_mma_k32(c, lo[mt][0], lo[mt][1], hi[mt][0], hi[mt][1], b0, b1);
                        if constexpr (T::minmode == IQD_MIN_YSUM32) {
                            const float sB0 = yc0.y, sB1 = yc1.y;
                            a[0] += (float) c[0] * sl0 * dB0 + ml0 * sB0;
                            a[1] += (float) c[1] * sl0 * dB1 + ml0 * sB1;
                            a[2] += (float) c[2] * sl1 * dB0 + ml1 * sB0;
                            a[3] += (float) c[3] * sl1 * dB1 + ml1 * sB1;
                        } else {
                            a[0] += (float) c[0] * sl0 * dB0;
                            a[1] += (float) c[1] * sl0 * dB1;
                            a[2] += (float) c[2] * sl1 * dB0;
                            a[3] += (float) c[3] * sl1 * dB1;
                        }
                    } else {
                        int c[4], e[4];
                        mmq_iqd_mma_k16(c, lo[mt][0], lo[mt][1], b0);
                        mmq_iqd_mma_k16(e, hi[mt][0], hi[mt][1], b1);
                        if constexpr (T::minmode == IQD_MIN_ONES16) {
                            a[0] += ((float) c[0] * sl0 + ml0 * (float) oc[0] + (float) e[0] * sh0 + mh0 * (float) oe[0]) * dB0;
                            a[1] += ((float) c[1] * sl0 + ml0 * (float) oc[1] + (float) e[1] * sh0 + mh0 * (float) oe[1]) * dB1;
                            a[2] += ((float) c[2] * sl1 + ml1 * (float) oc[0] + (float) e[2] * sh1 + mh1 * (float) oe[0]) * dB0;
                            a[3] += ((float) c[3] * sl1 + ml1 * (float) oc[1] + (float) e[3] * sh1 + mh1 * (float) oe[1]) * dB1;
                        } else {
                            a[0] += ((float) c[0] * sl0 + (float) e[0] * sh0) * dB0;
                            a[1] += ((float) c[1] * sl0 + (float) e[1] * sh0) * dB1;
                            a[2] += ((float) c[2] * sl1 + (float) e[2] * sh1) * dB0;
                            a[3] += ((float) c[3] * sl1 + (float) e[3] * sh1) * dB1;
                        }
                    }
                }
            }
        }
        __syncwarp();
        buf ^= 1;
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = row0 + mt*16 + g + (q >= 2 ? 8 : 0);
                const int col = nt*8 + 2*t + (q & 1);
                if (row < nrows_x && col < ncols) {
                    dst[(int64_t) col*stride_col_dst + row] = acc[mt][nt][q];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nrows_x, nkb, stride_row_x, ncols_y, ncols_dst, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
}

template <ggml_type type, int NT, int NW, int MT>
static void launch_mul_mat_iq_wide_v(const char * x, const int * y, float * dst, const int64_t nrows_x,
        const int64_t ncols_x, const int64_t stride_row_x, const int64_t ncols_y, const int64_t ncols_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    constexpr int smem = mmq_iqw_smem_bytes<type, NW, MT>();
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_iq_wide<type, NT, NW, MT>), smem);
    const dim3 grid((nrows_x + 16*MT*NW - 1) / (16*MT*NW), (ncols_dst + 8*NT - 1) / (8*NT), 1);
    const dim3 block(32, NW, 1);
    mul_mat_iq_wide<type, NT, NW, MT><<<grid, block, smem, stream>>>(
        x, y, dst, (int) nrows_x, (int) (ncols_x / QK_K), (int) stride_row_x, (int) ncols_y, (int) ncols_dst,
        (int) stride_col_dst);
}

// LLAMA_IQW_SHAPE: 1 (default) = 4 warps x 32 rows x 64 columns per CTA (activation fragment reused by 2 row tiles,
// 2 CTAs/SM), 0 = 8 warps x 16 rows x 128 columns. Every output element is computed the same way in both.
template <ggml_type type>
static void launch_mul_mat_iq_wide(const char * x, const int * y, float * dst, const int64_t nrows_x,
        const int64_t ncols_x, const int64_t stride_row_x, const int64_t ncols_y, const int64_t ncols_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    static const int shape = [] { const char * e = getenv("LLAMA_IQW_SHAPE"); return e ? atoi(e) : 1; }();
    if (shape == 0) {
        launch_mul_mat_iq_wide_v<type, 16, 8, 1>(x, y, dst, nrows_x, ncols_x, stride_row_x, ncols_y, ncols_dst, stride_col_dst, stream);
    } else {
        launch_mul_mat_iq_wide_v<type,  8, 4, 2>(x, y, dst, nrows_x, ncols_x, stride_row_x, ncols_y, ncols_dst, stride_col_dst, stream);
    }
}


// ---------------------------------------------------------------------------------------------------------------------
// Tiled prefill kernel (opt-in LLAMA_MMQ_IQ_TILE=1, MMQ tile width J = 128; IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S,
// IQ4_XS). CTA = 4*NWM warps, 64*NWM rows x 128 columns. Warp w decodes 16-row tile w of the next K block from its own
// cp.async-staged raw blocks (the decode2 of the direct kernel) into a fragment-native shared tile (each lane's four A
// registers are 16 contiguous bytes: one conflict-free 128-bit store, one 128-bit load per fragment); warp (wm, wn) then
// runs 4 row tiles x 4 column tiles of m16n8k32 per 32 K (each A fragment feeds 4 MMAs, each B fragment 4), with the
// activation fragments and scales read through L1. Decoded tiles are double buffered: iteration i decodes K block i+1,
// issues the raw copy of i+2 and runs the MMAs of i, with one CTA barrier per K block (MMQ: decode -> barrier -> MMA ->
// barrier, tensor pipe 26-35% busy at J = 128). The fp32 order differs from MMQ (prefill only, n = 128 tiles).
static constexpr __host__ __device__ bool mmq_iqt_supported(const ggml_type type) {
    return type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS  || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ4_XS;
}

template <ggml_type type, int NWM>
struct mmq_iqt_cfg {
    static constexpr int NW         = 4*NWM;
    static constexpr int ROWS       = 64*NWM;
    static constexpr int MTI        = ROWS/16;                                   // == NW
    static constexpr int NWR        = mmq_iqd_words<type>();
    static constexpr int SC         = mmq_iqd_traits<type>::scale16 ? 4 : 2;     // floats per (row tile, s, g)
    static constexpr int DEC_BYTES  = MTI*8*32*16;
    static constexpr int SC_BYTES   = MTI*8*8*SC*4;
    static constexpr int RAW_BYTES  = NW*16*NWR*4;
    static constexpr int GRID_BYTES = mmq_iqd_traits<type>::table_words*4;
    static constexpr int SMEM       = 2*DEC_BYTES + 2*SC_BYTES + RAW_BYTES + GRID_BYTES;
};

template <ggml_type type, int NWM>
__launch_bounds__(128*NWM, 1)
static __global__ void mul_mat_iq_tile(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int ncols_dst,
        const int stride_col_dst) {
#if defined(AMPERE_MMA_AVAILABLE)
    using T = mmq_iqd_traits<type>;
    using C = mmq_iqt_cfg<type, NWM>;
    static_assert(T::minmode == IQD_MIN_NONE && T::ylayout == IQD_Y_D4, "tile kernel: min-free D4 types only");
    constexpr int bs   = T::bs;
    constexpr int NWR  = C::NWR;
    constexpr int NWC  = 16*NWR;
    constexpr int YB   = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr int MTI  = C::MTI;
    constexpr int SC   = C::SC;

    extern __shared__ int iqd_smem[];
    int4     * dec     = (int4 *) iqd_smem;                                              // [2][MTI][8][32]
    float    * scs     = (float *) ((char *) iqd_smem + 2*C::DEC_BYTES);                 // [2][MTI][8][8][SC]
    uint8_t  * raw_all = (uint8_t *) iqd_smem + 2*C::DEC_BYTES + 2*C::SC_BYTES;
    uint32_t * sgrid   = (uint32_t *) (raw_all + C::RAW_BYTES);
    const uint2 * smask = nullptr;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int g    = lane >> 2;
    const int t    = lane & 3;

    const uint32_t * grid = T::gtable();
    if constexpr (T::table_words > 0) {
        for (int i = w*32 + lane; i < T::table_words; i += C::NW*32) {
            sgrid[i] = grid[i];
        }
        grid = sgrid;
    }

    const int row0 = blockIdx.x*C::ROWS;

    // ---- this warp's decode rows: 16-row tile w (clamped to the last row, never out of bounds) -------------------------
    uint8_t * raw = raw_all + w*16*NWR*4;
    constexpr int DR = 32 / NWR, DW = 32 % NWR;
    const int  rowbytes = stride_row_x*bs;
    const int  rd0    = row0 + w*16;
    const int  rfirst = min(rd0, nrows_x - 1);
    const char * xr   = x + (int64_t) rfirst*rowbytes;
    const int  rlim   = nrows_x - 1 - rfirst;
    const bool full   = rd0 + 15 <= nrows_x - 1;
    const int  lr0    = lane / NWR;
    const int  lw0    = lane - lr0*NWR;

    auto issue = [&](const int kb) {
        const char * base = xr + ((kb*bs) & ~3);
        int r = lr0, wd = lw0;
        int soff = lr0*rowbytes + 4*lw0;
#pragma unroll (NWR > 32 ? 4 : NWC/32 + 1)
        for (int f0 = 0; f0 < NWC; f0 += 32) {
            if (f0 + 32 <= NWC || f0 + lane < NWC) {
                const int so = full ? soff : min(r, rlim)*rowbytes + 4*wd;
                iqd_cp_async4(raw + 4*(f0 + lane), base + so);
            }
            r += DR; wd += DW; soff += DR*rowbytes + 4*DW;
            if (wd >= NWR) {
                wd -= NWR; ++r; soff += rowbytes - 4*NWR;
            }
        }
        iqd_cp_commit();
    };

    // decode of one 32-weight sub-block s of the staged K block (rows g and g + 8 of this warp's tile)
    auto decode_s = [&](const uint8_t * brow0, const uint8_t * brow1, const float2 rc0, const float2 rc1,
                        int4 * dw, float * sw, const int s) {
        {
            int lo0, hi0, lo1, hi1; float sl0, sh0, sl1, sh1, ml0, mh0, ml1, mh1;
            T::decode2(brow0, s, t, rc0, grid, smask, lo0, hi0, sl0, sh0, ml0, mh0);
            T::decode2(brow1, s, t, rc1, grid, smask, lo1, hi1, sl1, sh1, ml1, mh1);
            dw[s*32 + lane] = make_int4(lo0, lo1, hi0, hi1);
            if (t == 0) {
                if constexpr (SC == 4) {
                    *(float4 *) (sw + (s*8 + g)*4) = make_float4(sl0, sl1, sh0, sh1);
                } else {
                    *(float2 *) (sw + (s*8 + g)*2) = make_float2(sl0, sl1);
                }
            }
        }
    };
    auto decode = [&](const int kb, const int buf) {
        const int o = (kb*bs) & 3;
        const uint8_t * brow0 = raw + (g + 0)*NWR*4 + o;
        const uint8_t * brow1 = raw + (g + 8)*NWR*4 + o;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            decode_s(brow0, brow1, T::rowc(brow0), T::rowc(brow1), dec + (buf*MTI + w)*8*32, scs + (buf*MTI + w)*8*8*SC, s);
        }
    };

    // ---- MMA role: rows wm*64 .. +63 (4 row tiles), columns wn*32 .. +31 (4 column tiles) ---------------------------
    const int wm = w / 4;
    const int wn = w % 4;
    float acc[4][4][4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.0f;
        }
    }

    auto mma_s = [&](const int kb, const int buf, const int s) {
        const int * yk = y + (int64_t) (2*kb)*ncols_y*YB;
        const int4  * dr = dec + (buf*MTI + wm*4)*8*32;
        const float * sr = scs + (buf*MTI + wm*4)*8*8*SC;
        {
            const int * yb = yk + (s >> 2)*ncols_y*YB;
            const int   so = s & 3;
            int4   A[4];
            float4 S[4];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                A[mi] = dr[(mi*8 + s)*32 + lane];
                if constexpr (SC == 4) {
                    S[mi] = *(const float4 *) (sr + ((mi*8 + s)*8 + g)*4);
                } else {
                    const float2 v = *(const float2 *) (sr + ((mi*8 + s)*8 + g)*2);
                    S[mi] = make_float4(v.x, v.y, v.x, v.y);
                }
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int col = wn*32 + ni*8;
                const int * yc = yb + (col + g)*YB + 4 + so*8;
                int b0, b1;
                if constexpr (T::perm) {
                    const int2 v = *(const int2 *) (yc + 2*t);
                    b0 = v.x;
                    b1 = v.y;
                } else {
                    b0 = yc[t];
                    b1 = yc[t + 4];
                }
                const float dB0 = ((const float *) (yb + (col + 2*t + 0)*YB))[so];
                const float dB1 = ((const float *) (yb + (col + 2*t + 1)*YB))[so];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    float * a = acc[mi][ni];
                    if constexpr (!T::scale16) {
                        int c[4];
                        mmq_iqd_mma_k32(c, A[mi].x, A[mi].y, A[mi].z, A[mi].w, b0, b1);
                        a[0] += (float) c[0] * S[mi].x * dB0;
                        a[1] += (float) c[1] * S[mi].x * dB1;
                        a[2] += (float) c[2] * S[mi].y * dB0;
                        a[3] += (float) c[3] * S[mi].y * dB1;
                    } else {
                        int c[4], e[4];
                        mmq_iqd_mma_k16(c, A[mi].x, A[mi].y, b0);
                        mmq_iqd_mma_k16(e, A[mi].z, A[mi].w, b1);
                        a[0] += ((float) c[0] * S[mi].x + (float) e[0] * S[mi].z) * dB0;
                        a[1] += ((float) c[1] * S[mi].x + (float) e[1] * S[mi].z) * dB1;
                        a[2] += ((float) c[2] * S[mi].y + (float) e[2] * S[mi].w) * dB0;
                        a[3] += ((float) c[3] * S[mi].y + (float) e[3] * S[mi].w) * dB1;
                    }
                }
            }
        }
    };

    // ---- pipeline ------------------------------------------------------------------------------------------------
    issue(0);
    __syncthreads(); // grid table
    iqd_cp_wait<0>();
    __syncwarp();
    decode(0, 0);
    __syncwarp();
    if (nkb > 1) {
        issue(1);
    }
    __syncthreads();
    for (int kb = 0; kb < nkb; ++kb) {
        const int  buf      = kb & 1;
        const bool has_next = kb + 1 < nkb;
        if (has_next) {
            iqd_cp_wait<0>();
            __syncwarp();
        }
        {
            // decode of K block kb + 1 (2 sub-blocks per step) interleaved with the MMAs of sub-blocks 0..3 of kb, so
            // the integer decode and the tensor pipe run side by side instead of in alternating CTA-wide phases
            const int o = ((kb + 1)*bs) & 3;
            const uint8_t * brow0 = raw + (g + 0)*NWR*4 + o;
            const uint8_t * brow1 = raw + (g + 8)*NWR*4 + o;
            float2 rc0 = make_float2(0.0f, 0.0f), rc1 = rc0;
            if (has_next) {
                rc0 = T::rowc(brow0);
                rc1 = T::rowc(brow1);
            }
            int4  * dw = dec + ((buf ^ 1)*MTI + w)*8*32;
            float * sw = scs + ((buf ^ 1)*MTI + w)*8*8*SC;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                if (has_next) {
                    decode_s(brow0, brow1, rc0, rc1, dw, sw, 2*p + 0);
                    decode_s(brow0, brow1, rc0, rc1, dw, sw, 2*p + 1);
                }
                mma_s(kb, buf, p);
            }
        }
        __syncwarp();
        if (kb + 2 < nkb) {
            issue(kb + 2); // the raw buffer is free; the copy runs under the second half and the barrier
        }
#pragma unroll
        for (int p = 4; p < 8; ++p) {
            mma_s(kb, buf, p);
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = row0 + (wm*4 + mi)*16 + g + (q >= 2 ? 8 : 0);
                const int col = wn*32 + ni*8 + 2*t + (q & 1);
                if (row < nrows_x && col < ncols_dst) {
                    dst[(int64_t) col*stride_col_dst + row] = acc[mi][ni][q];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nrows_x, nkb, stride_row_x, ncols_y, ncols_dst, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE)
}

template <ggml_type type, int NWM>
static void launch_mul_mat_iq_tile_v(const char * x, const int * y, float * dst, const int64_t nrows_x,
        const int64_t ncols_x, const int64_t stride_row_x, const int64_t ncols_y, const int64_t ncols_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    using C = mmq_iqt_cfg<type, NWM>;
    static_assert(C::SMEM <= 101376, "tile kernel: shared memory over the sm_86 limit");
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_iq_tile<type, NWM>), C::SMEM);
    const dim3 grid((nrows_x + C::ROWS - 1) / C::ROWS, 1, 1);
    const dim3 block(32, C::NW, 1);
    mul_mat_iq_tile<type, NWM><<<grid, block, C::SMEM, stream>>>(
        x, y, dst, (int) nrows_x, (int) (ncols_x / QK_K), (int) stride_row_x, (int) ncols_y, (int) ncols_dst,
        (int) stride_col_dst);
}

template <ggml_type type>
static void launch_mul_mat_iq_tile(const char * x, const int * y, float * dst, const int64_t nrows_x,
        const int64_t ncols_x, const int64_t stride_row_x, const int64_t ncols_y, const int64_t ncols_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    launch_mul_mat_iq_tile_v<type, 2>(x, y, dst, nrows_x, ncols_x, stride_row_x, ncols_y, ncols_dst, stride_col_dst, stream);
}


// ---------------------------------------------------------------------------------------------------------------------
// Warp-specialized variant of the tiled prefill kernel (LLAMA_MMQ_IQ_TILE=2): 8 consumer warps (2 x 4, 64 rows x 32
// columns each, MMA + per-32 epilogue only) and 4 producer warps (32 rows each: cp.async staging + decode of the next K
// block into the other buffer). One 384-thread barrier per K block; the producers' raw copy for block k + 2 is issued
// right after their decode of k + 1 and lands under the consumers' MMAs. The per-32 epilogue (~12 instructions per MMA)
// leaves the consumer warps little issue slack, so the decode is moved off them entirely.
template <ggml_type type>
struct mmq_iqs_cfg {
    static constexpr int NWC_  = 8;                                      // consumer warps
#ifndef MMQ_IQS_NWP
#define MMQ_IQS_NWP 4
#endif
    static constexpr int NWP   = MMQ_IQS_NWP;                            // producer warps (4: 479 t/s; 2 producers could not keep up: 430)
    static constexpr int ROWS  = 128;
    static constexpr int MTI   = 8;
    static constexpr int NWR   = mmq_iqd_words<type>();
    static constexpr int SC    = mmq_iqd_traits<type>::scale16 ? 4 : 2;
    static constexpr int DEC_BYTES  = MTI*8*32*16;
    static constexpr int SC_BYTES   = MTI*8*8*SC*4;
    static constexpr int PR         = ROWS/NWP;                          // rows per producer
    static constexpr int RAW_BYTES  = NWP*PR*NWR*4;
    static constexpr int GRID_BYTES = mmq_iqd_traits<type>::table_words*4;
    static constexpr int SMEM = 2*DEC_BYTES + 2*SC_BYTES + RAW_BYTES + GRID_BYTES;
};

template <ggml_type type>
__launch_bounds__(32*(8 + MMQ_IQS_NWP), 1)
static __global__ void mul_mat_iq_tile_ws(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int nkb, const int stride_row_x, const int ncols_y, const int ncols_dst,
        const int stride_col_dst) {
#if defined(AMPERE_MMA_AVAILABLE)
    using T = mmq_iqd_traits<type>;
    using C = mmq_iqs_cfg<type>;
    static_assert(T::minmode == IQD_MIN_NONE && T::ylayout == IQD_Y_D4, "tile kernel: min-free D4 types only");
    constexpr int bs  = T::bs;
    constexpr int NWR = C::NWR;
    constexpr int YB  = sizeof(block_q8_1_mmq) / sizeof(int);
    constexpr int MTI = C::MTI;
    constexpr int SC  = C::SC;

    extern __shared__ int iqd_smem[];
    int4     * dec     = (int4 *) iqd_smem;
    float    * scs     = (float *) ((char *) iqd_smem + 2*C::DEC_BYTES);
    uint8_t  * raw_all = (uint8_t *) iqd_smem + 2*C::DEC_BYTES + 2*C::SC_BYTES;
    uint32_t * sgrid   = (uint32_t *) (raw_all + C::RAW_BYTES);
    const uint2 * smask = nullptr;

    const int lane = threadIdx.x;
    const int w    = threadIdx.y;
    const int g    = lane >> 2;
    const int t    = lane & 3;
    const int row0 = blockIdx.x*C::ROWS;

    const uint32_t * grid = T::gtable();
    if constexpr (T::table_words > 0) {
        for (int i = w*32 + lane; i < T::table_words; i += (8 + C::NWP)*32) {
            sgrid[i] = grid[i];
        }
        grid = sgrid;
    }

    if (w >= C::NWC_) {
        // ================================ producer: rows row0 + 32*pw .. +31 ===================================
        const int pw = w - C::NWC_;
        constexpr int PR  = C::PR;
        constexpr int NWC = PR*NWR;
        constexpr int DR = 32 / NWR, DW = 32 % NWR;
        uint8_t * raw = raw_all + pw*PR*NWR*4;
        const int  rowbytes = stride_row_x*bs;
        const int  rd0    = row0 + pw*PR;
        const int  rfirst = min(rd0, nrows_x - 1);
        const char * xr   = x + (int64_t) rfirst*rowbytes;
        const int  rlim   = nrows_x - 1 - rfirst;
        const bool full   = rd0 + PR - 1 <= nrows_x - 1;
        const int  lr0    = lane / NWR;
        const int  lw0    = lane - lr0*NWR;
        auto issue = [&](const int kb) {
            const char * base = xr + ((kb*bs) & ~3);
            int r = lr0, wd = lw0;
            int soff = lr0*rowbytes + 4*lw0;
#pragma unroll 4
            for (int f0 = 0; f0 < NWC; f0 += 32) {
                if (f0 + 32 <= NWC || f0 + lane < NWC) {
                    const int so = full ? soff : min(r, rlim)*rowbytes + 4*wd;
                    iqd_cp_async4(raw + 4*(f0 + lane), base + so);
                }
                r += DR; wd += DW; soff += DR*rowbytes + 4*DW;
                if (wd >= NWR) {
                    wd -= NWR; ++r; soff += rowbytes - 4*NWR;
                }
            }
            iqd_cp_commit();
        };
        auto decode = [&](const int kb, const int buf) {
            const int o = (kb*bs) & 3;
#pragma unroll
            for (int lt = 0; lt < PR/16; ++lt) {
                const int mt = (PR/16)*pw + lt;
                const uint8_t * brow0 = raw + (lt*16 + g + 0)*NWR*4 + o;
                const uint8_t * brow1 = raw + (lt*16 + g + 8)*NWR*4 + o;
                const float2 rc0 = T::rowc(brow0);
                const float2 rc1 = T::rowc(brow1);
                int4  * dw = dec + (buf*MTI + mt)*8*32;
                float * sw = scs + (buf*MTI + mt)*8*8*SC;
#pragma unroll
                for (int s = 0; s < 8; ++s) {
                    int lo0, hi0, lo1, hi1; float sl0, sh0, sl1, sh1, ml0, mh0, ml1, mh1;
                    T::decode2(brow0, s, t, rc0, grid, smask, lo0, hi0, sl0, sh0, ml0, mh0);
                    T::decode2(brow1, s, t, rc1, grid, smask, lo1, hi1, sl1, sh1, ml1, mh1);
                    dw[s*32 + lane] = make_int4(lo0, lo1, hi0, hi1);
                    if (t == 0) {
                        if constexpr (SC == 4) {
                            *(float4 *) (sw + (s*8 + g)*4) = make_float4(sl0, sl1, sh0, sh1);
                        } else {
                            *(float2 *) (sw + (s*8 + g)*2) = make_float2(sl0, sl1);
                        }
                    }
                }
            }
        };
        issue(0);
        __syncthreads(); // grid table
        iqd_cp_wait<0>();
        __syncwarp();
        decode(0, 0);
        __syncwarp();
        if (nkb > 1) {
            issue(1);
        }
        __syncthreads(); // tile 0 ready
        for (int kb = 0; kb < nkb; ++kb) {
            if (kb + 1 < nkb) {
                iqd_cp_wait<0>();
                __syncwarp();
                decode(kb + 1, (kb + 1) & 1);
                __syncwarp();
                if (kb + 2 < nkb) {
                    issue(kb + 2);
                }
            }
            __syncthreads();
        }
        return;
    }

    // ================================ consumer: rows wm*64.., columns wn*32.. ===================================
    const int wm = w / 4;
    const int wn = w % 4;
    float acc[4][4][4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.0f;
        }
    }
    __syncthreads(); // grid table
    __syncthreads(); // tile 0 ready
    for (int kb = 0; kb < nkb; ++kb) {
        const int buf = kb & 1;
        const int * yk = y + (int64_t) (2*kb)*ncols_y*YB;
        const int4  * dr = dec + (buf*MTI + wm*4)*8*32;
        const float * sr = scs + (buf*MTI + wm*4)*8*8*SC;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int * yb = yk + (s >> 2)*ncols_y*YB;
            const int   so = s & 3;
            int4   A[4];
            float4 S[4];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                A[mi] = dr[(mi*8 + s)*32 + lane];
                if constexpr (SC == 4) {
                    S[mi] = *(const float4 *) (sr + ((mi*8 + s)*8 + g)*4);
                } else {
                    const float2 v = *(const float2 *) (sr + ((mi*8 + s)*8 + g)*2);
                    S[mi] = make_float4(v.x, v.y, v.x, v.y);
                }
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int col = wn*32 + ni*8;
                const int * yc = yb + (col + g)*YB + 4 + so*8;
                int b0, b1;
                if constexpr (T::perm) {
                    const int2 v = *(const int2 *) (yc + 2*t);
                    b0 = v.x;
                    b1 = v.y;
                } else {
                    b0 = yc[t];
                    b1 = yc[t + 4];
                }
                const float dB0 = ((const float *) (yb + (col + 2*t + 0)*YB))[so];
                const float dB1 = ((const float *) (yb + (col + 2*t + 1)*YB))[so];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi) {
                    float * a = acc[mi][ni];
                    if constexpr (!T::scale16) {
                        int c[4];
                        mmq_iqd_mma_k32(c, A[mi].x, A[mi].y, A[mi].z, A[mi].w, b0, b1);
                        a[0] += (float) c[0] * S[mi].x * dB0;
                        a[1] += (float) c[1] * S[mi].x * dB1;
                        a[2] += (float) c[2] * S[mi].y * dB0;
                        a[3] += (float) c[3] * S[mi].y * dB1;
                    } else {
                        int c[4], e[4];
                        mmq_iqd_mma_k16(c, A[mi].x, A[mi].y, b0);
                        mmq_iqd_mma_k16(e, A[mi].z, A[mi].w, b1);
                        a[0] += ((float) c[0] * S[mi].x + (float) e[0] * S[mi].z) * dB0;
                        a[1] += ((float) c[1] * S[mi].x + (float) e[1] * S[mi].z) * dB1;
                        a[2] += ((float) c[2] * S[mi].y + (float) e[2] * S[mi].w) * dB0;
                        a[3] += ((float) c[3] * S[mi].y + (float) e[3] * S[mi].w) * dB1;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int row = row0 + (wm*4 + mi)*16 + g + (q >= 2 ? 8 : 0);
                const int col = wn*32 + ni*8 + 2*t + (q & 1);
                if (row < nrows_x && col < ncols_dst) {
                    dst[(int64_t) col*stride_col_dst + row] = acc[mi][ni][q];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nrows_x, nkb, stride_row_x, ncols_y, ncols_dst, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE)
}

template <ggml_type type>
static void launch_mul_mat_iq_tile_ws(const char * x, const int * y, float * dst, const int64_t nrows_x,
        const int64_t ncols_x, const int64_t stride_row_x, const int64_t ncols_y, const int64_t ncols_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    using C = mmq_iqs_cfg<type>;
    static_assert(C::SMEM <= 101376, "tile kernel: shared memory over the sm_86 limit");
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_iq_tile_ws<type>), C::SMEM);
    const dim3 grid((nrows_x + C::ROWS - 1) / C::ROWS, 1, 1);
    const dim3 block(32, 8 + C::NWP, 1);
    mul_mat_iq_tile_ws<type><<<grid, block, C::SMEM, stream>>>(
        x, y, dst, (int) nrows_x, (int) (ncols_x / QK_K), (int) stride_row_x, (int) ncols_y, (int) ncols_dst,
        (int) stride_col_dst);
}

template <ggml_type type, int NT, int MT>
static void launch_mul_mat_iq_direct_mt(const char * x, const int * y, float * dst, const int64_t nrows_x,
        const int64_t ncols_x, const int64_t stride_row_x, const int64_t ncols_y, const int64_t ncols_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    constexpr int smem = mmq_iqd_smem_bytes<type, NT, MT>();
    constexpr int rows = 16*MT;
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_iq_direct<type, NT, MT>), smem);
    const dim3 grid((nrows_x + rows - 1) / rows, 1, 1);
    const dim3 block(32, MMQ_IQD_KS, 1);
    mul_mat_iq_direct<type, NT, MT><<<grid, block, smem, stream>>>(
        x, y, dst, (int) nrows_x, (int) (ncols_x / QK_K), (int) stride_row_x, (int) ncols_y, (int) ncols_dst,
        (int) stride_col_dst);
}

// resident CTAs per SM of one kernel instance (cached per instance)
template <ggml_type type, int NT, int MT>
static int mmq_iqd_blocks_per_sm() {
    static int n = -1;
    if (n < 0) {
        constexpr int smem = mmq_iqd_smem_bytes<type, NT, MT>();
        CUDA_CHECK(cudaFuncSetAttribute(mul_mat_iq_direct<type, NT, MT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mul_mat_iq_direct<type, NT, MT>, MMQ_IQD_KS*32, smem));
        n = std::max(n, 1);
    }
    return n;
}

// Row tile per launch: MT = 2 (32 rows/warp: each activation fragment feeds two 16-row tiles) or MT = 1 (16 rows:
// half the staging, so more resident CTAs, and twice the CTAs for small row counts). LLAMA_IQD_MT=1|2 forces one;
// LLAMA_IQD_MT=auto scores both with (active warps per SM, capped at 16) x (wave efficiency) and takes MT = 1 only when it
// scores 25% higher (MT = 2 reuses the activation fragments). The choice depends on the row count and the type only,
// never on the number of columns, and every column's arithmetic is the same for both tiles: results are identical.
template <ggml_type type, int NT>
static void launch_mul_mat_iq_direct(const char * x, const int * y, float * dst, const int64_t nrows_x,
        const int64_t ncols_x, const int64_t stride_row_x, const int64_t ncols_y, const int64_t ncols_dst,
        const int64_t stride_col_dst, cudaStream_t stream) {
    // default per type (ncu, 2026-11-11): MT = 1 for the wide-row types whose double-buffered 32-row staging limits
    // residency to 2 CTAs/SM (IQ4_XS 66.6 vs 81.8 us, Q2_K 49.1 vs 52.2), MT = 2 elsewhere (IQ1/IQ2 types 2-8% faster)
    constexpr int mt_default = type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q2_K ? 1 : 2;
    static const int force_env = [] { // -1 = unset
        const char * e = getenv("LLAMA_IQD_MT");
        return e == nullptr ? -1 : (strcmp(e, "auto") == 0 ? 0 : atoi(e));
    }();
    const int force = force_env < 0 ? mt_default : force_env;
    int mt = force == 1 || force == 2 ? force : 0;
    if (mt == 0) {
        const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        auto score = [&](const int bps, const int rows) {
            const int64_t blocks = (nrows_x + rows - 1) / rows;
            const int64_t slots  = (int64_t) nsm * bps;
            const int64_t waves  = (blocks + slots - 1) / slots;
            const double  eff    = (double) blocks / (double) (waves * slots);
            const double  warps  = std::min(16.0, (double) bps * MMQ_IQD_KS);
            return eff * warps;
        };
        const double s2 = score(mmq_iqd_blocks_per_sm<type, NT, 2>(), 32);
        const double s1 = score(mmq_iqd_blocks_per_sm<type, NT, 1>(), 16);
        mt = s1 > 1.25 * s2 ? 1 : 2;
    }
    if (mt == 1) {
        launch_mul_mat_iq_direct_mt<type, NT, 1>(x, y, dst, nrows_x, ncols_x, stride_row_x, ncols_y, ncols_dst, stride_col_dst, stream);
    } else {
        launch_mul_mat_iq_direct_mt<type, NT, 2>(x, y, dst, nrows_x, ncols_x, stride_row_x, ncols_y, ncols_dst, stride_col_dst, stream);
    }
}
