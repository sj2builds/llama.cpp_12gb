#pragma once

#include "common.cuh"

// ---------------------------------------------------------------------------------------------
// Stripped-model study hooks (dev/stripped-draft-study.md). All of them are opt-in through
// LLAMA_STRIP_* environment variables; with none of them set every function below returns after a
// single load of a static bool and the compute path is unchanged (no extra launch, no extra byte).
//
// The idea being emulated: keep a SUBSET of the full model's weights resident and delete the rest.
// Because a deleted FFN group contributes exactly zero to ffn_down (its 256 activations multiply the
// matching 256-wide block of every ffn_down row), zeroing the 256-row block of the SWIGLU output is
// equivalent to deleting the group - which lets one dense model emulate any stripped variant.
//
//   LLAMA_STRIP_FFN=oracle:<frac>   per token, keep the top-frac groups by ||silu(gate)*up||_2
//   LLAMA_STRIP_FFN_LAYERS="7:0.25,*:0.5"   per-layer override for the oracle fraction
//   LLAMA_STRIP_MASK=<file>         static group masks, one line per tensor:
//                                   blk.N.ffn_down.weight 0,1,3,...   (kept group indices)
//                                   blk.N.attn_output.weight ...      (24 groups of 256)
//                                   blk.N.ssm_out.weight ...          (48 groups of 128, GDN heads)
//   LLAMA_STRIP_VOCAB=<file>        output-head subset: token ids to keep, one per line or comma list
//   LLAMA_STRIP_STATS=<file>        calibration: mean per-group activation norm per layer, rewritten
//                                   periodically (requires GGML_CUDA_DISABLE_GRAPHS=1)
// ---------------------------------------------------------------------------------------------

struct ggml_tensor;
struct ggml_backend_cuda_context;

bool ggml_cuda_strip_enabled();

// Called from the CUDA graph eval loop around the dispatch of one node.
void ggml_cuda_strip_pre (ggml_backend_cuda_context & ctx, ggml_tensor * node);
void ggml_cuda_strip_post(ggml_backend_cuda_context & ctx, ggml_tensor * node);
