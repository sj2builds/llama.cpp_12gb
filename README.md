# Qwen3.8-27B on an RTX 3060 12 GiB: a llama.cpp fork

27B model, **96K context, q8 KV cache**, on one 12 GiB GPU, using the IQ3_XXS weights and a speculative drafter. Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at base `547208dca`.

## Top 5 changes

1. **Direct IQ matmul kernels.** Small batches read the quantized weights directly: IQ kernels +15-26%.
2. **DSpine drafter, 7 or 15 tokens per step.** Draft length is picked per step from measured speed.
3. **Elastic staged KV.** q8 KV is kept in pinned host RAM and streamed through a GPU pool, so 96K fits.
4. **CUDA graphs for staged steps.** Host launch time per step drops from 42 to 4.5 ms.
5. **Exact verification.** Replay rollback plus batch-invariant matmuls, so greedy output equals plain decoding.

## Real-world performance (q8 KV, 96K server)

| Workload | Context | Decode | Prompt |
|---|---|---|---|
| Code generation, thinking off | 21-30K | **93 t/s** | — |
| Coding agent (thinking, tool calls), 19 requests | 15-50K | **41 t/s** | 351 t/s |
| Long prompt prefill | 29K / 86K | — | 414 / 314 t/s |

## Run (96K, q8 KV, DSpine)

```bat
set GGML_CUDA_PREFER_MMQ=1
set LLAMA_FA_DECODE_Q4=1
set LLAMA_FA_DECODE_Q4_M3=1
set LLAMA_FA_Q4_SPLIT_TOKENS=2048
set LLAMA_FA_VEC_QUANT_KV=1
set LLAMA_FA_VEC_MAX_NQ=128
set LLAMA_RS_REPLAY=1
set LLAMA_MMVQ_BATCH_INVARIANT=1
set LLAMA_MMVQ_UNPACK_ONCE=0
set LLAMA_KV_HOST_ROWCOPY=1
set LLAMA_KV_BUDGET_MIN_RESIDENT=0
set LLAMA_KV_STAGED_LAYERS=16
set LLAMA_KV_STAGE_ELASTIC=1050
set LLAMA_KV_STAGE_GRAPHS=1
set LLAMA_DFLASH_ADAPT=7
set LLAMA_SPEC_LAZY_SMPL=1
set LLAMA_MMQ_STREAMK_2X=1
set LLAMA_MMQ_IQ_DIRECT=1

llama-server -m Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf -c 98304 -ngl 99 -fa on ^
  -ctk q8_0 -ctv q8_0 -b 512 -ub 256 -np 1 --fit off --no-context-shift --jinja ^
  -ot "^output\.weight$=CUDA0" ^
  --spec-type draft-dflash --spec-draft-model Qwen3.8-27B-DSpine-b16-IQ2_XXS-MIX.gguf ^
  --spec-draft-ngl 99 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 ^
  --spec-draft-n-max 15 --spec-draft-n-ubatch 16 -otd dspine_codebook=CPU ^
  --reasoning-budget 8192 --port 8080
```

For output that also matches plain decoding under sampling (T>0), use `--spec-draft-n-max 7` and drop `LLAMA_DFLASH_ADAPT`.

Measurements, design notes and build scripts: [`rtx3060/`](rtx3060/).
