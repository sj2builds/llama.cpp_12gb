#pragma once

// Emulation-only hotness probe for the hot-KV scratch study (worker4, reviewer seq 131/133).
// Computes, from the FLASH_ATTN_EXT inputs of one layer, the attention probability mass that the block's queries
// place on every key position, summed over the 24 query heads (K is shared by 6 of them) and over the block's
// queries (optionally every `stride`-th query). It is the same arithmetic as the FA kernel, only the reduction
// differs, so no -fa off pass and no second KV cache is needed.
//
// Everything here is gated by LLAMA_HOT_TRACE / LLAMA_SENT_BLOCK; the default path never calls it.

#include "common.cuh"

struct ggml_tensor;

void ggml_cuda_hot_mass(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * mask,
                        float scale, int n_head_q, int n_head_kv, int stride,
                        float * mass, float * mass_max, float * perhead, cudaStream_t stream);

// R2-5c sentinel probe, per (query, head) of one layer:
//   gap   = max score over out-of-scratch keys - max score over in-scratch keys
//   argmax= the out-of-scratch key that achieved the max
//   mass  = 1 - sum_scratch exp(s - m) / sum_all exp(s - m)      (out-of-scratch attention mass)
//   delta = ||o_full - o_scratch|| / ||o_full||                  (output perturbation, in-kernel)
//   page  = out-of-scratch mass summed per `page_tokens`-token page (optional, page_out != nullptr)
// `v` is optional (delta/page need it); `scratch_tokens` > 0 defines the scratch as the last `scratch_tokens`
// keys instead of taking membership from the mask row (the F pass, whose mask allows every past key).
// seq 205: lse_in/lse_out are the absolute log-sum-exp of the in-scratch and out-of-scratch scores (so
// mass = sigmoid(lse_out - lse_in)), and onorm_f/onorm_s/dnorm are ||o_full||, ||o_scratch|| and
// ||o_full - o_scratch|| (delta above is dnorm/onorm_f; the study also wants dnorm/onorm_s). Any of the
// extra pointers may be null, in which case nothing is written.
void ggml_cuda_hot_sentinel(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                            const ggml_tensor * mask, float scale, int n_head_q, int n_head_kv,
                            int stride, int pos_base, int scratch_tokens, int page_tokens, int n_page,
                            float * gap, int * argmax_key, float * mass_out, float * delta_out,
                            float * page_out, float * lse_in, float * lse_out, float * onorm_f,
                            float * onorm_s, float * dnorm, cudaStream_t stream);

// Page plane only (the fast-heat path): out-of-scratch attention mass per page, added to page_out with the
// sentinel's layout (index (qi * n_head_q + h) * n_page + page). q4_0 or q8_0 K, head size 256, GQA <= 8, pages of
// <= 128 tokens; returns false without launching anything for other shapes.
// q_stride: query qi of the (strided) q / mask views is the node's query qi * q_stride (position pos_base + that)
bool ggml_cuda_hot_page_heat(const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * mask,
                             float scale, int n_head_q, int n_head_kv, int pos_base, int q_stride, int scratch_tokens,
                             int page_tokens, int n_page, float * page_out, cudaStream_t stream);

// One entry point for the whole probe (hot mass + sentinel + dumps) on a single FLASH_ATTN_EXT node. The node loop
// calls it for nodes the staged-KV plan does not own; the plan calls it with src[1]/src[2] pointing at the per-layer
// staging slot, which holds the full used prefix, so staged layers are probed as well.
struct ggml_backend_cuda_context;
void ggml_cuda_hot_probe_node(ggml_backend_cuda_context & ctx, const ggml_tensor * node);
