# Elastic staged KV + long-context findings (2026-11-11)

Opt-in: `LLAMA_KV_STAGED_LAYERS=16 LLAMA_KV_STAGE_ELASTIC=<pool MiB>` (no LLAMA_KV_GPU_BUDGET_MIB). Default staging
unchanged.

## Why

With `LLAMA_KV_GPU_BUDGET_MIB`, "resident" layers are allocated at full 96K capacity (q8: 204 MiB each) and staged
layers re-copy their whole used prefix over PCIe every step through 2 rotating full-capacity slots. At 15-30K the
used KV of all 16 layers would fit in the budget, yet 13 layers were re-copied per step; at 59K the step is
PCIe-bound (13 x 128 MB = 1.67 GB/step).

## What (ggml-cuda/staged-kv.cuh, ggml-cuda.cu, src/llama-kv-cache.cpp)

- Slots are sized to the used prefix (+4096 cells headroom, 2048-cell steps, re-sized as the context grows),
  zero-initialised, up to 32 slots; the pool holds as many as fit.
- The first n_sticky layers own a slot that keeps their prefix across evals (new rows arrive through the
  intercepted SET_ROWS, which already writes host + slot): no DMA. The rest rotate through 2 slots as before.
- `LLAMA_KV_STAGED_LAYERS` is skipped (with a log line) for caches that cannot be staged (the drafter's SWA cache),
  instead of failing.
- Also: per-call `getenv` in the staged write/attention/prefetch paths cached.

## Measured (production launcher config, q8 KV, 96K, DSpine d7, new kernels, codebook on host; real prefills)

| context | budget staging (1050 MiB) | elastic (pool 1050 MiB) | all-resident (24K ctx, graphs) |
|---|---|---|---|
| 15K | 98.7 ms/step | **94.5** (+4.4% t/s) | 91.2 |
| 30K | 107.3 ms (24.7 t/s) | **99.2** (26.8 t/s, +8.5%) | - |
| 59K | 156.1 ms (15.0 t/s) | **152.4** (15.3 t/s, +2.5%) | - |

Same generated text in every pair (d7 is exact); peak VRAM at 59K 11.59 GB.

Also measured today:
- DSpine codebook to host RAM (`-otd dspine_codebook=CPU`): +1.1 ms/step at 8K, frees 258 MiB; with the budget raised
  800 -> 1050 MiB: 59K 13.3 -> 15.0 t/s (175.5 -> 156.1 ms).
- Drafter attention window 8K -> 4K / 2K: drafter 13.5 -> 11.2 / 10.1 ms but 3.15 -> 2.93 / 2.88 tokens/step:
  slower (34.7 -> 33.0 / 32.7 t/s). Keep 8K.
- Remaining staged cost at 15K (94.5 vs 91.2): staged evals never use CUDA graphs (slot events outlive an eval), so
  the host launches ~4600 nodes per step; ~3 ms of GPU idle.
- At long context the step is bounded by PCIe bytes of the rotating layers (~10 ms per 128 MB layer vs ~5 ms compute);
  only more VRAM for KV or fewer bytes per token would move it.

## CUDA graphs for sticky-only evals (opt-in `LLAMA_KV_STAGE_GRAPHS=1`, now in both launchers)

An elastic eval in which every layer owns a valid sticky slot (same layer, current host generation, full-prefix mode)
and no prefetch is outstanding needs no staging events: SET_ROWS writes host + slot, attention reads the slot, nothing
is copied. Such evals now go through the normal CUDA-graph path (capture / replay). Per-slot `pending` flag (set by
prefetch, cleared by the first compute-stream wait); in graph mode write/attention skip the ready waits and the consumed
record; `begin()` records `consumed` on the compute stream before any later prefetch (a replayed graph records none).
Any rotating slot, hot/pack/range staging, GGML_KV_STAGE_PROF, LLAMA_HOT_STAGED, LLAMA_KV_STAGE_JOIN or
LLAMA_KV_STAGE_TRACE -> the old event path. A slot resize already clears the graphs.

| (d7, prod launcher, new matmul staging) | graphs off | graphs on |
|---|---|---|
| short code prompt, step | 83.4 ms (82.8 t/s) | **79.3 ms (86.8 t/s)**; host launch 42 -> 4.5 ms |
| 15K prompt, 128 tokens | 40.3 t/s | 41.5 t/s |
| ~31K prompt (pool full, rotating slots: graphs off by design) | 29.9 t/s | 29.8 t/s |

Same text in every pair (short 2cd513bc7763, 15K c379c73fb066, 31K f8f589bf03b2). Dynamic d15/adapt 7 with graphs:
94.1 t/s on the short code prompt, same greedy text.
