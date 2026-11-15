#pragma once

#include "common.cuh"

// Decode attention for q4_0 K / q4_0 V, head size 256, GQA 6, 1-8 query tokens, Ampere mma.sync (opt-in:
// LLAMA_FA_DECODE_Q4=1). CUDA port of the paged-decode design in amd_ref (fattn-paged-decode-kernel.cuh), contiguous
// K/V only.

// true when the env flag is set and dst has a shape the kernel takes
bool ggml_cuda_fattn_decode_q4_supported(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_decode_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Paged variant (KV-scratch D attention): the K/V view of dst holds n_pages * 64 cells; chunk c covers the 64 cells
// starting at cell pages[c].x (device array). It reads them from frame pages[c].y of mirror_k / mirror_v (rows laid
// out like the cache: 64 consecutive rows per frame) when pages[c].y >= 0, else from the view's own buffer at that
// cell; the mask columns of the chunk are pages[c].x .. +63 of the original mask. No shape checks: the caller only
// uses it where ggml_cuda_fattn_decode_q4_supported() holds for the unpaged node.
void ggml_cuda_flash_attn_ext_decode_q4_paged(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const int2 * pages,
                                              int n_pages, const void * mirror_k, const void * mirror_v);
