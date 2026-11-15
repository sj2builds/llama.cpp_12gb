#pragma once

#include "common.cuh"

// R3-5 step 1 (opt-in quality gate only, see dev/int8-prefill-gate.md): fake-quantise the two operands of a
// large prefill matmul onto the int8 grid, in the Hadamard-rotated basis, storing the bare integer codes in an
// f16 buffer. Only called from the LLAMA_FAKE_INT8_PREFILL hook in ggml-cuda.cu; nothing here runs unless
// that variable is set. Decode and verify never take this path.
//
// Rotation: independent blocks of FQ_BLK (= 128) values along K, each rotated by the orthogonal
// Hadamard-128 (1/sqrt(128) normalisation), so no row-sized shared memory is needed: one warp holds one block
// in 4 registers per lane. The same transform on both operands leaves the product unchanged in exact
// arithmetic; it exists so that per-row / per-token int8 quantisation stops being ruined by outliers.

#define FQ_BLK 128
#define FQ_NT  128          // threads per CTA: 4 warps, one line (row or column) per CTA

// Weights: rotate every block of each row in place (f16 -> f16) and write the integer code, plus the per-row
// scale (f32, nrows entries).
// quantize == false: rotation only, no int8 rounding (diagnostic control)
void ggml_cuda_fakeq_rows_f16(half * w, int64_t nrows, int64_t k, float * scale, bool quantize,
                              cudaStream_t stream);

// Activations: f32 column-major (N columns of K contiguous values) -> rotated integer codes in f16, plus the
// per-column scale (f32, ncols entries).
void ggml_cuda_fakeq_cols_f16(const float * x, half * q, int64_t ncols, int64_t k, float * scale, bool quantize,
                              cudaStream_t stream);

// Control path: plain f32 -> f16 conversion, no rotation and no int8 rounding.
void ggml_cuda_fakeq_cvt_f16(const float * x, half * q, int64_t n, cudaStream_t stream);

// dst[m + n*m] *= ws[m] * xs[n]: the scales of the int8 GEMM applied after it (an int8 GEMM epilogue).
void ggml_cuda_fakeq_epilogue(float * dst, const float * ws, const float * xs, int64_t m, int64_t n,
                              cudaStream_t stream);
