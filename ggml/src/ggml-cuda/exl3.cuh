#pragma once

#include "common.cuh"

// EXL3 (exllamav3 trellis format, mul1 codebook) matrix multiplication, see dev/exl3-scope.md.
//
// A weight of type GGML_TYPE_EXL3_<K> holds the packed trellis of a K-bit EXL3 tensor: 16x16 tiles, tile index
// (k / 16) * (N / 16) + (n / 16), 16 * K uint16 per tile, bytes exactly as exllamav3 stores them. The matmul node
// carries the two EXL3 sign/scale vectors as extra sources: src[2] = suh (F16, K values, input side), src[3] = svh
// (F16, N values, output side). The product is
//
//     y = svh * H128( H128(suh * x) @ W )        (H128: normalized Sylvester Hadamard over blocks of 128)
//
// Up to EXL3_SMALL_MAX columns of src1 take a tensor-core kernel with a fixed split of K per (K, N) and an ordered
// reduction, so a column's result does not depend on how many columns are in the batch (spec == greedy, and the
// KV-scratch wide checks). Wider batches reconstruct the weight to f16 in column slices and use cuBLAS.
// LLAMA_EXL3_SMALL_MAX=<n> lowers the limit (0 = always cuBLAS).

#define EXL3_SMALL_MAX 64

bool ggml_cuda_mul_mat_exl3_supported(const ggml_tensor * dst);

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
