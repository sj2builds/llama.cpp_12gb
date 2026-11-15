# IQ matmul for small batches (2026-11-11)

All opt-in; unset = default MMQ path, golden sha 40b3b84a... unchanged.

| env | what | output |
|---|---|---|
| `LLAMA_MMQ_STREAMK_2X=1` | every quant type, J = 8/16: stream-k grid 2 x nsm + `__launch_bounds__` min 2 blocks (ncu: the 1-CTA/SM stream-k launches were 44% of matmul time at ~17% active warps) | new fp32 order, batch invariant |
| `LLAMA_MMQ_IQ_PREFETCH=11` | IQ3_XXS/IQ3_S/IQ2_S/IQ2_XS/IQ2_XXS in MMQ: raw-block + activation register prefetch, shared-memory grid table (IQ3_XXS/IQ3_S/IQ2_XXS), 2 CTAs/SM | new fp32 order, batch invariant |
| `LLAMA_MMQ_IQ_DIRECT=1` | new kernel `mmq-iq-direct.cuh` for the same 5 IQ types at <= 16 columns (2-D products, row bytes % 4 == 0): decode straight into m16n8k32 / m16n8k16 int8 A fragments, raw blocks double-buffered per warp with cp.async, no CTA barrier in the K loop, byte->sign-mask table, 32 rows x 4 K slices per CTA, fixed-order slice sum | new fp32 order, batch invariant |

Measured (llama-bench, Swift IQ3_XXS GGUF, q4 KV, short context, t/s, 10 reps; the GGUF is a mix of 9 quant types,
IQ3_S 22.5% / IQ2_S 20.5% / IQ3_XXS 17.6% / IQ2_XS 10.2% / IQ2_XXS 9.9% of matmul time):

| config | pp1 | pp8 | pp16 |
|---|---|---|---|
| default | 13.04 | 98.4 | 175.0 |
| STREAMK_2X | 13.6 | 102.0 | 178.8 |
| STREAMK_2X + IQ_PREFETCH=11 | 14.46 | 109.4 | 189.3 |
| STREAMK_2X + IQ_DIRECT=1 | **15.36** | **113.3** | 188.6 |

Checks for STREAMK_2X + IQ_DIRECT: test-backend-ops MUL_MAT vs CPU 277/277 for the 5 types; golden spec == no-spec
(40b3b84a..., same text as default over 128 tokens); no register spills (121-128 regs).

Direct-kernel history: v1 (one k16 MMA per 16 weights, per-call scale/sign decode, 16 rows/warp) was slower than
MMQ (8 instr/weight); v2 (sign table, k32, 32 rows/warp, register prefetch) spilled (128-reg cap); v3 replaced the
register prefetch with cp.async double buffering.

Not done: J=16 is only on par with the MMQ variants (two n-tiles double the activation loads and epilogue per decode);
a 16-column specialization (activation fragments shared across warps through shared memory) is the next step.


## Update 2026-11-11 (all quant types, final)

`LLAMA_MMQ_IQ_DIRECT=1` now covers IQ3_XXS, IQ3_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ4_XS, Q4_K, Q2_K, i.e. every
quantized matmul type of the Swift GGUF. Min/offset terms: Q4_K and IQ1_S use the DS4 activation sums (as MMQ), Q2_K
(per-16 mins, D2S6 layout) uses exact per-16 sums of the quantized activations from an all-ones m16n8k16 MMA.
A 16-rows-per-warp variant for 16 columns was tried and reverted (pp16 181.5 vs 190.3).
Launcher: `LLAMA_SPEC_LAZY_SMPL=1` added to serve_q8kv_96k_dspine.bat (host cut 3, ~0.9 ms/step).

| config | pp1 | pp8 | pp16 |
|---|---|---|---|
| default | 13.02 | 98.2 | 174.5 |
| STREAMK_2X + IQ_PREFETCH=11 | 14.56 | 109.3 | 188.9 |
| STREAMK_2X + IQ_DIRECT (all 9 types) | **15.77** | **115.5** | **190.3** |

Checks: test-backend-ops MUL_MAT 550/550 for the 9 types; no register spills (101-128 regs); golden spec == no-spec
(40b3b84a...); DSpine d15 greedy == no-spec (2cd513bc...). DSpine d15 under T=1 sampling still differs from no-spec
(dff71b39 vs 3d58ecb1): with every matmul width-invariant the remaining width dependence is in a non-matmul op at
16-token batches (attention split-K q4 decode path or the GDN recurrence) - next to find. No-spec decode at T=1:
16.0 t/s (was 13.4).


## Update 2026-11-11 (afternoon): staging address math

Profile (ncu `--set full`, pp8, 12 direct-kernel launches): the kernel is issue-bound at 50-68% issue utilisation and
16-31% warps active (occupancy capped at 2-4 CTAs/SM by shared memory and the 128-register cap). Whole-step split at
short context (DSpine d7): 78 ms GPU per step, 64 ms of it in `mul_mat_iq_direct` (~160 GB/s on 9.7 GiB); drafter
~8 ms GPU (5 layers 3.8 ms, its 248K-vocab IQ4_XS lm_head 3.0 ms, top-k 0.12 ms); target lm_head 2.8 ms.

SASS per MMA (512 weights): IQ3_S 118 instructions, IQ4_XS 113, IQ2_XXS 94, IQ1_S 90, IQ2_XS/IQ2_S 67, Q2_K 36. About a
third of it was the cp.async staging loop: per 4-byte word a divide by the words-per-row, a row clamp and a 64-bit
row*stride multiply. Rewritten: row bytes are a multiple of 4 (host guard, now also checks stride_row_x*bs and x
alignment), so the word offset of block kb is the same for every row; each lane keeps the packed word mapping
(f = lane + 32*i, shared destination slot + 4f) and advances (row, word, source offset) incrementally; the clamp only in
the edge CTA. Full unroll except for the 34/36-word types (IQ4_XS, Q4_K: unroll 4, fully unrolled they were 20% slower).
A first row-outer variant (one row per pass) left lanes idle and was 20-30% slower; reverted.

| | old | new |
|---|---|---|
| 12 profiled direct launches (sum) | 1571 us | 1492 us (-5%) |
| instructions (IQ2_XS / IQ2_XXS / IQ4_XS kernel) | 23.6 / 16.5 / 7.0 M | 18.9 / 12.7 / 6.3 M |
| server step, d7, short code prompt (elastic KV, prod launcher) | 86.4 ms (79.9 t/s) | 83.4 ms (82.8 t/s) |

Same text (2cd513bc7763); golden spec == no-spec 40b3b84a; test-backend-ops MUL_MAT OK; no spills.
Same server with LLAMA_MMQ_IQ_DIRECT=0: 99.4 ms; all-resident + CUDA graphs (24K ctx): 84.3 ms (old kernel).

## Update 2026-11-11 (evening): no shared sign table, vector activation scales

After the staging fix the IQ kernels were bound by the MIO / LSU pipe (`mio_throttle` top stall, LSU 60-84% busy) and
about half of all shared-memory wavefronts were bank-conflict replays of random-index lookups. The 256-entry
byte -> sign-mask table (one random 64-bit LDS per decode, two for IQ2_XS / IQ2_S) is gone: `iqd_sgn4` spreads 4 sign
bits to the byte LSBs with one multiply (nib * 0x00204081, no overlapping partial products) and negates with
`(g ^ (y * 0xFF)) + y`, which cannot carry between bytes because every signed grid value is 1..127 (also replaces the
emulated `__vsub4`). 2 KiB less shared memory per CTA. The D4 / DS4 activation scales of a 128-value block are read
with one 128-bit load per column instead of a scalar load per sub-block. Decoded values and the fp32 order are unchanged.

| kernel (same 12 launches) | morning | v3 staging | now |
|---|---|---|---|
| IQ2_XXS (544) | 167.4 | 161.4 | **128.4** us |
| IQ2_S (544) | 225.9 | 214.2 | **177.3** |
| IQ2_XS (160) | 194.0 | 182.6 | **155.2** |
| IQ3_S (160) | 79.9 | 76.9 | **66.9** |
| sum of 12 | 1571 | 1492 | **1306** (-17%) |

Server (prod launcher config, short code prompt): d7 step 79.3 -> **74.5 ms** (86.8 -> 92.5 t/s); dynamic d15/adapt 7:
94.1 -> **102.5 t/s**. Same text (2cd513bc7763); golden spec == no-spec 40b3b84a; test-backend-ops MUL_MAT OK.

Per-type tile height (same evening): one ncu run with LLAMA_IQD_MT=1 for every type. IQ4_XS 81.8 -> 66.6 us and
79.4 -> 69.5 us (its 34-word double-buffered 32-row staging allowed only 2 CTAs/SM), Q2_K 52.2 -> 49.1, IQ3_S even,
IQ1_S / IQ2_* 2-8% slower. Default now MT = 1 for IQ4_XS, Q4_K, Q2_K and MT = 2 for the rest (LLAMA_IQD_MT still
overrides; results do not depend on MT). Server d7 step 74.5 -> **72.4 ms (95.1 t/s)**, drafter 8.6 -> 8.0 ms (its
lm_head is IQ4_XS); same text, golden 40b3b84a, MUL_MAT tests OK.

## Prefill (2026-11-11, late)

Prefill (ubatch 128) is 79% `mul_mat_q` at J = 128: 1 CTA (8 warps) per SM at 238-255 registers, 58-70 KB shared,
tensor pipe 26-35% busy (`math_pipe_throttle`, decode -> barrier -> MMA phases). pp2048: ub 128 496 t/s, ub 256 535,
ub 512 544. f16 dequant + cuBLAS (GGML_CUDA_FORCE_CUBLAS=1) lands at the same speed (494 / 542).

Tried, none faster than MMQ, all left opt-in / off:
- `LLAMA_MMQ_IQ_PREFETCH_PREFILL=1|9` (register prefetch / shared grid at J = 128): 478 / 483 t/s (255 registers).
- `LLAMA_MMQ_IQ_WIDE=1` (mmq-iq-direct.cuh `mul_mat_iq_wide`: the direct kernel's decode with warps splitting rows,
  activation fragments through L1): `LLAMA_IQW_SHAPE=0` (8 warps x 16 rows x 128 cols) 449 t/s, shape 1 (4 warps x 32 rows
  x 64 cols, default) 468 t/s; ncu: tensor 24-39%, 8 warps/SM, long_scoreboard on the activation loads. Capping at 3
  CTAs/SM (170 registers) spills: 381 t/s. MUL_MAT tests pass. Beating MMQ needs activations shared through shared
  memory (ldmatrix) with more warps resident - not done.

Shipped: `-ub 256` in both launchers: server 29K prompt 393 -> 414 t/s (+5%), peak VRAM +58 MiB (11,679 MiB); an 86K
prompt runs at 314 t/s with peak 11,669 MiB, no OOM.

## Prefill kernel rewrite attempt (2026-11-11, night) - did not beat MMQ

Goal 1.5x prefill. Built and tested (all correct: PPL 4 x 512 chunks 2.5048 vs MMQ 2.5029; all opt-in, off by default):

| kernel (env) | design | pp2048 ub128 (MMQ 494-496) |
|---|---|---|
| `LLAMA_MMQ_IQ_TILE=1` | 8 warps, 128x128 CTA tile, warp 64x32 (A/B fragment reuse 4x), fragment-native decoded shared tile (128-bit STS/LDS), double-buffered decode, 1 barrier/K block | 463 (decode then MMA per warp), 419 (decode interleaved into the MMA loop) |
| `LLAMA_MMQ_IQ_TILE=2` | warp-specialized: 8 consumer warps (MMA + epilogue only) + 4 producer warps (cp.async + decode of K block k+1) | **479** (2 producers: 430) |
| `LLAMA_MMQ_IQ_WIDE=1` | direct-kernel decode with row-split warps, activations via L1 | 449 / 468 |

Per type (ws vs MMQ, us per call): IQ2_XXS 1.08x, IQ3_S 1.05x, IQ3_XXS 1.00x, IQ4_XS 0.95x, IQ2_XS 0.86x, IQ2_S 0.82x
(the per-16-scale types need two k16 MMAs and ~20 epilogue instructions per 32 K).

Why it stops near MMQ: with q8_1 activations (one scale per 32) and per-32 weight scales, every m16n8k32 needs ~12
epilogue instructions (convert, scale product, FMA for 4 outputs) against a tensor-core budget of 16 issue slots per
IMMA per scheduler; with decode, loads and the MMA itself the consumer is issue-bound at ~14-19 instructions per MMA and
only 2 warps/scheduler fit (168-255 registers). Measured IMMA peak on this card: 101.7 TOPS (microbenchmark); the
kernels reach ~40% of it, MMQ ~30-35%. Also tried: replacing the int32->fp32 conversion with the 1.5*2^23 magic
accumulator (bit-identical): no gain (the SASS conversion is I2FP, not the 1/8-rate I2F), reverted.
Going further would need a coarser activation scale (one per 128: the int accumulator can then run over 4 sub-blocks),
which changes the numerics (same family as the shelved int8 route) - not done.

