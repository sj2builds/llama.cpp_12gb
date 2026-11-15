#include "hot.cuh"
#include "ggml.h"

#include <algorithm>

// q4_0 row of d=256 values: 8 blocks of (half scale + 16 packed nibbles) = 144 B
// Layout must match ggml (ggml/src/ggml-cuda/convert.cu, dequantize_block_q4_0): value p (p < 16) is the LOW
// nibble of qs[p], value p + 16 is the HIGH nibble of qs[p]. The earlier interleaved reading was wrong and
// permuted every key vector inside each 32-value block (R2-5c Stage 0, seq 154).
__device__ static inline float hot_q4_0_get(const char * row, int idx) {
    const int b = idx >> 5;
    const int i = idx & 31;
    const char * blk = row + b * 18;
    const float d = __half2float(*(const __half *) blk);
    const unsigned char byte = (unsigned char) blk[2 + (i & 15)];
    const int nib = (i & 16) ? (byte >> 4) : (byte & 0x0F);
    return d * (float) (nib - 8);
}

static __device__ __forceinline__ int hot_clamp255(int v) {
    return v < 256 ? v : 255;   // lane*8+u is always in [0,255] at blockDim 256; keeps the index provably in range
}

// block-wide max, result valid in EVERY thread (the shuffle-down only leaves it in lane 0)
static __device__ float hot_block_max(float v, float * red, float init) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    }
    if (lane == 0) {
        red[wid] = v;
    }
    __syncthreads();
    const int nw = (int) ((blockDim.x + 31) >> 5);
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < nw) ? red[threadIdx.x] : init;
        for (int o = 16; o > 0; o >>= 1) {
            t = fmaxf(t, __shfl_down_sync(0xffffffffu, t, o));
        }
        if (threadIdx.x == 0) {
            red[0] = t;
        }
    }
    __syncthreads();
    return red[0];
}

static __device__ float hot_block_sum(float v, float * red) {
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
        red[wid] = v;
    }
    __syncthreads();
    const int nw = (int) ((blockDim.x + 31) >> 5);
    if (threadIdx.x < 32) {
        float t = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) {
            t += __shfl_down_sync(0xffffffffu, t, o);
        }
        if (threadIdx.x == 0) {
            red[0] = t;
        }
    }
    __syncthreads();
    return red[0];
}

// one block per (query, query head); threads split the keys between them
template <int MAXJ>
static __global__ void hot_mass_kernel(const float * __restrict__ q, int64_t q_nb1, int64_t q_nb2,
                                       const char  * __restrict__ k, int64_t k_nb1, int64_t k_nb2,
                                       const __half * __restrict__ mask, int64_t mask_nb1,
                                       float scale, int n_q, int n_kv, int n_head_q, int n_head_kv,
                                       int stride, float * __restrict__ mass,
                                       float * __restrict__ mass_max, float * __restrict__ perhead) {
    __shared__ float qs[256];
    __shared__ float red[32];

    const int qi = (blockIdx.x / n_head_q) * stride;
    const int h  = blockIdx.x % n_head_q;
    if (qi >= n_q) {
        return;
    }
    const int hkv = h / (n_head_q / n_head_kv);

    // q_nb1 / q_nb2 are BYTE strides: advance the char pointer, do not add them to a float pointer
    const float * qrow = (const float *) ((const char *) q + (int64_t) qi * q_nb1 + (int64_t) h * q_nb2);
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        qs[t] = qrow[t];
    }
    __syncthreads();

    const char * kbase = k + (int64_t) hkv * k_nb2;
    // mask_nb1 is a BYTE stride as well
    const __half * mrow = (const __half *) ((const char *) mask + (int64_t) qi * mask_nb1);

    float sv[MAXJ];
    int   jv[MAXJ];
    int   cnt = 0;
    float lmax = -INFINITY;

    for (int j = threadIdx.x; j < n_kv; j += blockDim.x) {
        if (__half2float(mrow[j]) < -1.0e4f) {
            continue;
        }
        const char * row = kbase + (int64_t) j * k_nb1;
        float s = 0.0f;
        for (int t = 0; t < 256; ++t) {
            s += qs[t] * hot_q4_0_get(row, t);
        }
        s *= scale;
        if (cnt < MAXJ) {
            sv[cnt] = s;
            jv[cnt] = j;
            ++cnt;
        }
        lmax = fmaxf(lmax, s);
    }

    lmax = hot_block_max(lmax, red, -INFINITY);

    float lsum = 0.0f;
    for (int c = 0; c < cnt; ++c) {
        lsum += __expf(sv[c] - lmax);
    }
    lsum = hot_block_sum(lsum, red);


    if (!(lsum > 0.0f) || !isfinite(lsum)) {
        return;
    }
    const float inv = 1.0f / lsum;
    for (int c = 0; c < cnt; ++c) {
        const int j = jv[c];
        const float p = __expf(sv[c] - lmax) * inv;
        atomicAdd(&mass[j], p);
        if (mass_max != nullptr) {
            // a key is hot if ANY head puts a large probability on it (non-negative floats: integer order works)
            atomicMax((unsigned int *) &mass_max[j], __float_as_uint(p));
        }
        if (perhead != nullptr) {
            atomicAdd(&perhead[(size_t) h * n_kv + j], p);
        }
    }
}

void ggml_cuda_hot_mass(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * mask,
                        float scale, int n_head_q, int n_head_kv, int stride,
                        float * mass, float * mass_max, float * perhead, cudaStream_t stream) {
    const int n_q  = (int) q->ne[1];
    const int n_kv = (int) k->ne[1];
    if (n_kv <= 0 || n_q <= 0 || n_head_q <= 0 || n_head_kv <= 0 || mask == nullptr || mass == nullptr) {
        return;
    }
    const int nq_used = (n_q + stride - 1) / stride;
    const dim3 grid((unsigned) (nq_used * n_head_q));
    hot_mass_kernel<320><<<grid, 256, 0, stream>>>((const float *) q->data, q->nb[1], q->nb[2],
                                                   (const char  *) k->data, k->nb[1], k->nb[2],
                                                   (const __half *) mask->data, mask->nb[1],
                                                   scale, n_q, n_kv, n_head_q, n_head_kv, stride, mass,
                                                   mass_max, perhead);
}

// ---------------------------------------------------------------------------------------------
// Sentinel probe (tasks R2-5b / R2-5c). From the queries of one layer, how much of the answer lives outside
// the resident scratch?  Per (query, head):
//   gap    = max score over out-of-scratch keys - max over in-scratch keys   (the R2-5b max-gap control)
//   argmax = the out-of-scratch key that achieved that max
//   mass   = 1 - sum_scratch exp(s - m) / sum_all exp(s - m)                 (out-of-scratch mass)
//   delta  = ||o_full - o_scratch|| / ||o_full||, o = softmax(.) @ V         (output perturbation)
//   page   = the out-of-scratch mass summed per `page_tokens`-token page     (optional)
// Scratch membership comes from the mask row (causality + keep-set: the D pass) or, when `scratch_tokens` > 0,
// from a recency window of that many keys - the F pass's mask allows every past key, so it needs the window.
//
// Numerics: pass 1 finds the global max score m over the causal prefix, so every weight of pass 2 is
// exp(s - m) <= 1 and no online rescaling is needed; the two value accumulators are therefore plain sums, one
// over the scratch keys and one over the out-of-scratch keys, and o_full/o_scratch follow from them plus the
// two softmax denominators.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ bool hot_in_scratch(const __half * mrow, int j, int p1, int scratch_tokens) {
    if (scratch_tokens > 0) {
        return j > p1 - scratch_tokens;          // recency window: the scratch a draft of S tokens would hold
    }
    return __half2float(mrow[j]) > -1.0e4f;      // mask: causal AND inside the keep-set
}

// one block per (query, query head), blockDim.x = 256 (8 warps): pass 2 lets each warp own one key at a time,
// 32 lanes x 8 value dims = the whole 256-value head, so both the score and the two value accumulators are
// warp-cooperative and no thread needs a private 256-float array.
static __global__ void hot_sentinel_kernel(const float * __restrict__ q, int64_t q_nb1, int64_t q_nb2,
                                           const char  * __restrict__ k, int64_t k_nb1, int64_t k_nb2,
                                           const char  * __restrict__ v, int64_t v_nb1, int64_t v_nb2,
                                           const __half * __restrict__ mask, int64_t mask_nb1,
                                           float scale, int n_q, int n_kv, int n_head_q, int n_head_kv,
                                           int stride, int pos_base, int scratch_tokens,
                                           int page_tokens, int n_page,
                                           float * __restrict__ gap, int * __restrict__ argmax_key,
                                           float * __restrict__ mass_out, float * __restrict__ delta_out,
                                           float * __restrict__ page_out,
                                           float * __restrict__ lse_in_out, float * __restrict__ lse_out_out,
                                           float * __restrict__ onorm_f_out, float * __restrict__ onorm_s_out,
                                           float * __restrict__ dnorm_out) {
    __shared__ float qs[256];
    __shared__ float rin[32], rout[32];
    __shared__ int   rj[32];
    __shared__ float red_a[32], red_b[32];
    __shared__ float sacc_a[8][256];   // per-warp value accumulators, scratch keys
    __shared__ float sacc_b[8][256];   // ... and out-of-scratch keys
    __shared__ float s_inv_all, s_inv_scr;
    __shared__ float s_mall;
    extern __shared__ float sh_page[]; // n_page floats, only when page_out != nullptr

    const int qi = (blockIdx.x / n_head_q) * stride;
    const int h  = blockIdx.x % n_head_q;
    if (qi >= n_q || blockDim.x != 256) {
        return;
    }
    const int nwarps = 8;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int hkv = h / (n_head_q / n_head_kv);
    // q_nb1 / q_nb2 are BYTE strides: advance the char pointer, do not add them to a float pointer
    const float * qrow = (const float *) ((const char *) q + (int64_t) qi * q_nb1 + (int64_t) h * q_nb2);
    for (int t = threadIdx.x; t < 256; t += blockDim.x) {
        qs[t] = qrow[t];
    }
    if (page_out != nullptr) {
        for (int pg = threadIdx.x; pg < n_page; pg += blockDim.x) {
            sh_page[pg] = 0.0f;
        }
    }
    __syncthreads();

    const char * kbase = k + (int64_t) hkv * k_nb2;
    const char * vbase = v != nullptr ? v + (int64_t) hkv * v_nb2 : nullptr;
    // mask_nb1 is a BYTE stride as well
    const __half * mrow = (const __half *) ((const char *) mask + (int64_t) qi * mask_nb1);
    const int p1  = pos_base + qi;
    const int lim = min(p1, n_kv - 1);   // causal prefix this query may attend to (the block's own keys included)
    const int idx = qi * n_head_q + h;

    // ---- pass 1: the two maxima (in / out of scratch) and the out-of-scratch argmax
    float m_in = -INFINITY, m_out = -INFINITY;
    int   j_out = 0;
    for (int j = threadIdx.x; j <= lim; j += blockDim.x) {
        const char * row = kbase + (int64_t) j * k_nb1;
        float s = 0.0f;
        for (int t = 0; t < 256; ++t) {
            s += qs[t] * hot_q4_0_get(row, t);
        }
        s *= scale;
        if (hot_in_scratch(mrow, j, p1, scratch_tokens)) {
            m_in = fmaxf(m_in, s);
        } else if (s > m_out) {
            m_out = s;
            j_out = j;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        m_in = fmaxf(m_in, __shfl_down_sync(0xffffffffu, m_in, o));
        const float mo = __shfl_down_sync(0xffffffffu, m_out, o);
        const int   jo = __shfl_down_sync(0xffffffffu, j_out, o);
        if (mo > m_out) { m_out = mo; j_out = jo; }
    }
    if (lane == 0) { rin[wid] = m_in; rout[wid] = m_out; rj[wid] = j_out; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float bi = -INFINITY, bo = -INFINITY;
        int   bj = 0;
        for (int w = 0; w < nwarps; ++w) {
            bi = fmaxf(bi, rin[w]);
            if (rout[w] > bo) { bo = rout[w]; bj = rj[w]; }
        }
        s_mall = fmaxf(bi, bo);
        gap[idx] = (isinf(bo) || isinf(bi)) ? 0.0f : (bo - bi);
        argmax_key[idx] = bj;
    }
    __syncthreads();

    // ---- pass 2: one sweep over the causal prefix for the softmax sums, both value accumulators and the pages
    const float mall = s_mall;
    float la = 0.0f, ls = 0.0f;
    float va[8], vb[8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) { va[u] = 0.0f; vb[u] = 0.0f; }
    const int t0 = 8 * lane;
    if (!isinf(mall)) {
        for (int j = wid; j <= lim; j += nwarps) {
            const char * krow = kbase + (int64_t) j * k_nb1;
            float s = 0.0f;
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                s += qs[hot_clamp255(t0 + u)] * hot_q4_0_get(krow, hot_clamp255(t0 + u));
            }
            for (int o = 16; o > 0; o >>= 1) {
                s += __shfl_down_sync(0xffffffffu, s, o);
            }
            s = __shfl_sync(0xffffffffu, s, 0) * scale;   // every lane needs the same weight
            const float e = __expf(s - mall);
            la += e;
            if (hot_in_scratch(mrow, j, p1, scratch_tokens)) {
                ls += e;
                if (vbase != nullptr) {
                    const char * vrow = vbase + (int64_t) j * v_nb1;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        va[u] += e * hot_q4_0_get(vrow, hot_clamp255(t0 + u));
                    }
                }
            } else {
                if (vbase != nullptr) {
                    const char * vrow = vbase + (int64_t) j * v_nb1;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        vb[u] += e * hot_q4_0_get(vrow, hot_clamp255(t0 + u));
                    }
                }
                if (page_out != nullptr && page_tokens > 0) {
                    atomicAdd(&sh_page[j / page_tokens], e);
                }
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        la += __shfl_down_sync(0xffffffffu, la, o);
        ls += __shfl_down_sync(0xffffffffu, ls, o);
    }
    if (lane == 0) { red_a[wid] = la; red_b[wid] = ls; }
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        sacc_a[wid][hot_clamp255(t0 + u)] = va[u];
        sacc_b[wid][hot_clamp255(t0 + u)] = vb[u];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float ta = 0.0f, ts = 0.0f;
        for (int w = 0; w < nwarps; ++w) { ta += red_a[w]; ts += red_b[w]; }
        if (mass_out != nullptr) {
            mass_out[idx] = ta > 0.0f ? (1.0f - ts / ta) : 0.0f;
        }
        s_inv_all = ta > 0.0f ? 1.0f / ta : 0.0f;
        s_inv_scr = ts > 0.0f ? 1.0f / ts : 0.0f;
        if (lse_in_out != nullptr) {
            lse_in_out[idx] = ts > 0.0f ? s_mall + __logf(ts) : -1.0e30f;
        }
        if (lse_out_out != nullptr) {
            lse_out_out[idx] = ta > ts ? s_mall + __logf(ta - ts) : -1.0e30f;
        }
    }
    __syncthreads();

    // ---- delta: o_full = (A + B) / S_all and o_scr = A / S_scr, from the accumulators above
    float num = 0.0f, den = 0.0f, dens = 0.0f;
    if (vbase != nullptr) {
        float fa = 0.0f, fb = 0.0f;
        for (int w = 0; w < nwarps; ++w) {
            fa += sacc_a[w][threadIdx.x];
            fb += sacc_b[w][threadIdx.x];
        }
        const float of  = (fa + fb) * s_inv_all;
        const float osc = fa * s_inv_scr;
        const float d   = of - osc;
        num  = d * d;
        den  = of * of;
        dens = osc * osc;
    }
    num = hot_block_sum(num, red_a);
    den = hot_block_sum(den, red_b);
    dens = hot_block_sum(dens, red_a);
    if (threadIdx.x == 0) {
        if (delta_out != nullptr) {
            // no mass outside the scratch -> 0; no scratch at all -> 1 (the whole output is replaced)
            delta_out[idx] = den > 0.0f ? sqrtf(num / den) : (num > 0.0f ? 1.0f : 0.0f);
        }
        if (onorm_f_out != nullptr) {
            onorm_f_out[idx] = sqrtf(den);
        }
        if (onorm_s_out != nullptr) {
            onorm_s_out[idx] = sqrtf(dens);
        }
        if (dnorm_out != nullptr) {
            dnorm_out[idx] = sqrtf(num);
        }
    }
    if (page_out != nullptr && page_tokens > 0) {
        // the page sums above are unnormalised exp(s - m); scale by 1/S_all so a page score is a probability
        // mass and the pages of one (layer, head, query) add up to `mass`
        const float inv_all = s_inv_all;
        float * dst = page_out + (size_t) idx * n_page;
        for (int pg = threadIdx.x; pg < n_page; pg += blockDim.x) {
            if (sh_page[pg] != 0.0f) {
                dst[pg] += sh_page[pg] * inv_all;
            }
        }
    }
}

void ggml_cuda_hot_sentinel(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                            const ggml_tensor * mask, float scale, int n_head_q, int n_head_kv,
                            int stride, int pos_base, int scratch_tokens, int page_tokens, int n_page,
                            float * gap, int * argmax_key, float * mass_out, float * delta_out,
                            float * page_out, float * lse_in, float * lse_out, float * onorm_f,
                            float * onorm_s, float * dnorm, cudaStream_t stream) {
    const int n_q  = (int) q->ne[1];
    if (n_q <= 0 || gap == nullptr) {
        return;
    }
    const int nq_used = (n_q + stride - 1) / stride;
    const size_t shmem = page_out != nullptr ? (size_t) n_page * sizeof(float) : 0;
    hot_sentinel_kernel<<<dim3((unsigned) (nq_used * n_head_q)), 256, shmem, stream>>>(
        (const float *) q->data, q->nb[1], q->nb[2],
        (const char *) k->data, k->nb[1], k->nb[2],
        v != nullptr ? (const char *) v->data : nullptr,
        v != nullptr ? v->nb[1] : 0, v != nullptr ? v->nb[2] : 0,
        (const __half *) mask->data, mask->nb[1], scale, n_q, (int) k->ne[1], n_head_q, n_head_kv,
        stride, pos_base, scratch_tokens, page_tokens, n_page, gap, argmax_key, mass_out, delta_out,
        page_out, lse_in, lse_out, onorm_f, onorm_s, dnorm);
}

// ---------------------------------------------------------------------------------------------
// Fast page heat (KV-scratch production path, LLAMA_SENT_FAST_HEAT): the same page plane the sentinel kernel
// writes (out-of-scratch attention mass per page, normalised by the full softmax denominator of each
// (query, head)), computed page-parallel. The sentinel runs one block per (query, head) and sweeps every key twice
// with scalar q4_0 reads (~30 ms per query at 130K). Here one block takes one (page, KV head): it loads the page's
// raw K rows into shared memory once and loops over the queries (all GQA heads at once), so extra queries cost
// only their arithmetic. A second kernel folds the per-page (max, sums) into masses.
// part[(qi * n_head_q + h) * n_pg + pg] = (page max score, sum exp(s - max) over the page's causal keys,
//                                           the same sum over its out-of-scratch keys, unused)
// ---------------------------------------------------------------------------------------------
#define HOT_PH_MAX_GQA  8
#define HOT_PH_MAX_PAGE 128
#define HOT_PH_ROW      144   // q4_0 bytes of one 256-dim K head row
#define HOT_PH_ROW8     272   // q8_0 bytes of one 256-dim K head row

template <bool KQ8>
static __global__ void hot_page_partial_kernel(const float * __restrict__ q, int64_t q_nb1, int64_t q_nb2,
                                               const char  * __restrict__ k, int64_t k_nb1, int64_t k_nb2,
                                               const __half * __restrict__ mask, int64_t mask_nb1,
                                               float scale, int n_q, int n_kv, int n_head_q, int n_head_kv,
                                               int pos_base, int q_stride, int scratch_tokens, int page_tokens,
                                               int n_pg, float4 * __restrict__ part) {
    __shared__ float qs[HOT_PH_MAX_GQA][256];
    __shared__ float sc[HOT_PH_MAX_GQA][HOT_PH_MAX_PAGE];
    constexpr int ROW = KQ8 ? HOT_PH_ROW8 : HOT_PH_ROW;
    __shared__ __align__(16) unsigned char ks[HOT_PH_MAX_PAGE * ROW];
    const int pg   = blockIdx.x;
    const int hkv  = blockIdx.y;
    const int gqa  = n_head_q / n_head_kv;
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    const int nw   = blockDim.x >> 5;
    const int j0   = pg * page_tokens;
    const int nj   = max(0, min(page_tokens, n_kv - j0));
    const char * kbase = k + (int64_t) hkv * k_nb2;
    // the page's K rows of this head, once (144/272-byte rows at 4-byte aligned strides: copy 32-bit words)
    for (int i = threadIdx.x; i < nj * (ROW / 4); i += blockDim.x) {
        const int r = i / (ROW / 4), w = i % (ROW / 4);
        ((uint32_t *) ks)[i] = ((const uint32_t *) (kbase + (int64_t) (j0 + r) * k_nb1))[w];
    }

    for (int qi = 0; qi < n_q; ++qi) {
        __syncthreads();   // ks ready (first query); qs and sc free again (later queries)
        for (int i = threadIdx.x; i < gqa * 256; i += blockDim.x) {
            const int g = i / 256, t = i % 256;
            qs[g][t] = *(const float *) ((const char *) q + (int64_t) qi * q_nb1 + (int64_t) (hkv * gqa + g) * q_nb2 + t * 4);
        }
        __syncthreads();

        const int p1  = pos_base + qi * q_stride;
        const int lim = min(p1, n_kv - 1);
        // lane owns dims u*32 + lane (u = q4_0 block, in-block value = lane): the qs loads are conflict-free and
        // the K bytes/scales are broadcasts (consecutive dims per lane made every qs load an 8-way bank conflict)
        for (int jj = wid; jj < nj; jj += nw) {
            const int j = j0 + jj;
            float acc[HOT_PH_MAX_GQA];
#pragma unroll
            for (int g = 0; g < HOT_PH_MAX_GQA; ++g) {
                acc[g] = 0.0f;
            }
            if (j <= lim) {
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    float kv;
                    if constexpr (KQ8) {
                        const unsigned char * blk = ks + jj * ROW + u * 34;
                        kv = __half2float(*(const __half *) blk) * (float) (signed char) blk[2 + lane];
                    } else {
                        const unsigned char * blk = ks + jj * ROW + u * 18;
                        const float d = __half2float(*(const __half *) blk);
                        const unsigned char byte = blk[2 + (lane & 15)];
                        kv = d * (float) (((lane & 16) ? (byte >> 4) : (byte & 0x0F)) - 8);
                    }
                    const int t = u * 32 + lane;
#pragma unroll
                    for (int g = 0; g < HOT_PH_MAX_GQA; ++g) {
                        if (g < gqa) {
                            acc[g] += qs[g][t] * kv;
                        }
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < HOT_PH_MAX_GQA; ++g) {
                float sv = acc[g];
                for (int o = 16; o > 0; o >>= 1) {
                    sv += __shfl_xor_sync(0xffffffffu, sv, o);
                }
                if (lane == 0 && g < gqa) {
                    sc[g][jj] = j <= lim ? sv * scale : -INFINITY;
                }
            }
        }
        __syncthreads();

        const __half * mrow = (const __half *) ((const char *) mask + (int64_t) qi * mask_nb1);
        for (int g = wid; g < gqa; g += nw) {
            float m = -INFINITY;
            for (int jj = lane; jj < nj; jj += 32) {
                m = fmaxf(m, sc[g][jj]);
            }
            for (int o = 16; o > 0; o >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            }
            float sa = 0.0f, so = 0.0f;
            if (m != -INFINITY) {
                for (int jj = lane; jj < nj; jj += 32) {
                    const float sv = sc[g][jj];
                    if (sv == -INFINITY) {
                        continue;
                    }
                    const float e = __expf(sv - m);
                    sa += e;
                    if (!hot_in_scratch(mrow, j0 + jj, p1, scratch_tokens)) {
                        so += e;
                    }
                }
            }
            for (int o = 16; o > 0; o >>= 1) {
                sa += __shfl_xor_sync(0xffffffffu, sa, o);
                so += __shfl_xor_sync(0xffffffffu, so, o);
            }
            if (lane == 0) {
                part[((int64_t) qi * n_head_q + hkv * gqa + g) * n_pg + pg] = make_float4(m, sa, so, 0.0f);
            }
        }
    }
}

// one block per (query, head): the max and the denominator over all pages, then each page's mass
static __global__ void hot_page_finalize_kernel(const float4 * __restrict__ part, int n_pg,
                                                float * __restrict__ page_out, int n_page) {
    // two reduction buffers: a thread may still read red_m[0] from the max while warp 0 writes the sum
    __shared__ float red_m[32], red_s[32];
    const float4 * p = part + (int64_t) blockIdx.x * n_pg;
    float m = -INFINITY;
    for (int pg = threadIdx.x; pg < n_pg; pg += blockDim.x) {
        if (p[pg].y > 0.0f) {
            m = fmaxf(m, p[pg].x);
        }
    }
    m = hot_block_max(m, red_m, -INFINITY);
    if (m == -INFINITY) {
        return;
    }
    float tot = 0.0f;
    for (int pg = threadIdx.x; pg < n_pg; pg += blockDim.x) {
        if (p[pg].y > 0.0f) {
            tot += p[pg].y * __expf(p[pg].x - m);
        }
    }
    tot = hot_block_sum(tot, red_s);
    const float inv = tot > 0.0f ? 1.0f / tot : 0.0f;
    float * dst = page_out + (int64_t) blockIdx.x * n_page;
    for (int pg = threadIdx.x; pg < n_pg; pg += blockDim.x) {
        if (p[pg].z > 0.0f) {
            dst[pg] += p[pg].z * __expf(p[pg].x - m) * inv;
        }
    }
}

bool ggml_cuda_hot_page_heat(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * mask,
                             float scale, int n_head_q, int n_head_kv, int pos_base, int q_stride, int scratch_tokens,
                             int page_tokens, int n_page, float * page_out, cudaStream_t stream) {
    const int n_q  = (int) q->ne[1];
    const int n_kv = (int) k->ne[1];
    if (n_q <= 0 || n_kv <= 0 || page_out == nullptr || mask == nullptr || page_tokens <= 0 ||
            page_tokens > HOT_PH_MAX_PAGE || n_head_kv <= 0 || n_head_q % n_head_kv != 0 ||
            n_head_q / n_head_kv > HOT_PH_MAX_GQA || (k->type != GGML_TYPE_Q4_0 && k->type != GGML_TYPE_Q8_0) ||
            q->type != GGML_TYPE_F32 ||
            q->ne[0] != 256 || k->ne[0] != 256 || k->nb[1] % 4 != 0 || k->nb[2] % 4 != 0 ||
            (uintptr_t) k->data % 4 != 0) {
        return false;
    }
    const int n_pg = std::min(n_page, (n_kv + page_tokens - 1) / page_tokens);
    if (n_pg <= 0) {
        return false;
    }
    // grow-only scratch, never freed (a free could race kernels still queued on the non-blocking stream)
    static float4 * part   = nullptr;
    static size_t   part_n = 0;
    const size_t need = (size_t) n_q * n_head_q * n_pg;
    if (need > part_n) {
        void * p = nullptr;
        if (cudaMalloc(&p, need * sizeof(float4)) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        part   = (float4 *) p;
        part_n = need;
    }
    // LLAMA_SENT_HEAT_TIME=1 (diagnostic): time the two kernels with events (syncs the stream)
    static const bool t_heat = getenv("LLAMA_SENT_HEAT_TIME") != nullptr;
    static cudaEvent_t he[3] = {};
    if (t_heat && he[0] == nullptr) {
        for (auto & e : he) {
            (void) cudaEventCreate(&e);
        }
    }
    if (t_heat) {
        (void) cudaEventRecord(he[0], stream);
    }
    auto * kern = k->type == GGML_TYPE_Q8_0 ? hot_page_partial_kernel<true> : hot_page_partial_kernel<false>;
    kern<<<dim3((unsigned) n_pg, (unsigned) n_head_kv), 256, 0, stream>>>(
        (const float *) q->data, q->nb[1], q->nb[2], (const char *) k->data, k->nb[1], k->nb[2],
        (const __half *) mask->data, mask->nb[1], scale, n_q, n_kv, n_head_q, n_head_kv, pos_base,
        std::max(1, q_stride), scratch_tokens, page_tokens, n_pg, part);
    if (t_heat) {
        (void) cudaEventRecord(he[1], stream);
    }
    hot_page_finalize_kernel<<<(unsigned) (n_q * n_head_q), 256, 0, stream>>>(part, n_pg, page_out, n_page);
    if (t_heat) {
        (void) cudaEventRecord(he[2], stream);
        (void) cudaEventSynchronize(he[2]);
        float a = 0.0f, b = 0.0f;
        (void) cudaEventElapsedTime(&a, he[0], he[1]);
        (void) cudaEventElapsedTime(&b, he[1], he[2]);
        fprintf(stderr, "HEATTIME n_q=%d n_pg=%d partial=%.3f finalize=%.3f ms\n", n_q, n_pg, a, b);
    }
    return cudaGetLastError() == cudaSuccess;
}
