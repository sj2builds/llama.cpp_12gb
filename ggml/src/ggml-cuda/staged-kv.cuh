// Opt-in, layer-granular host KV with depth-N device staging slots.
// Included by ggml-cuda.cu after the ordinary CUDA buffer-type implementation.
//
// Pipeline (stream mode; staged decode never reaches CUDA-graph replay because
// KV op shapes grow every token):
//   - canonical KV lives in pinned host RAM (cudaHostAlloc Mapped|Portable)
//   - N slots x per-layer capacity, two non-blocking copy streams, prefetching
//     is prefix-sized (only the used rows, not the allocated capacity)
//   - prefetch(i) into slot i%N waits ONLY on that slot's persistent "consumed"
//     event (recorded after the flash-attention that last read the slot), so the
//     copy engine runs continuously ACROSS token boundaries; there is no
//     per-token fork/join that would quiesce and restart the pipeline
//   - each layer's K and V are one contiguous copy each, on two copy streams
//   - flash attention / SET_ROWS wait on both copy-done events of the slot
//   - GGML_KV_STAGE_PROF=1 prints per-layer copy and attention device times

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

// Opt-in GGML_OP_PROF=1 (diagnostic): the main stream's GPU timeline of each eval, split by what it spent the time
// on. Events are recorded between nodes and read once at the end of the eval (no sync inside it), so overlap is
// unchanged. An interval is labelled by the mark that ends it: GAP = the GPU waited for the host to launch the next
// node (host cost not hidden), KVWAIT = a staged layer waited for its slot copy (PCIe not hidden), the rest are the
// kernels themselves. Copies that finished in time never show up: they were hidden. Only valid with CUDA graphs off.
enum { OPP_GAP, OPP_MM, OPP_ATTN, OPP_KVWAIT, OPP_PROBE, OPP_GDN, OPP_KVWRITE, OPP_OTHER, OPP_N };
struct ggml_cuda_op_prof {
    const bool on = getenv("GGML_OP_PROF") != nullptr;
    std::vector<cudaEvent_t> ev;
    std::vector<int> cls;
    size_t n = 0;
    long long tok = -1;
    void mark(cudaStream_t s, int c) {
        if (!on) return;
        if (n == ev.size()) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreate(&e));
            ev.push_back(e);
            cls.push_back(0);
        }
        CUDA_CHECK(cudaEventRecord(ev[n], s));
        cls[n++] = c;
    }
    void note(const ggml_tensor * node) {
        if (on && tok < 0 && node->op == GGML_OP_MUL_MAT && node->src[1] != nullptr) tok = node->src[1]->ne[1];
    }
    void flush(cudaStream_t s, int n_nodes) {
        if (!on) return;
        if (n >= 2) {
            CUDA_CHECK(cudaStreamSynchronize(s));
            double t[OPP_N] = {};
            for (size_t i = 1; i < n; ++i) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, ev[i - 1], ev[i]));
                t[cls[i]] += ms;
            }
            float span = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&span, ev[0], ev[n - 1]));
            fprintf(stderr, "OPPROF tok=%lld nodes=%d span=%.2f gap=%.2f mm=%.2f attn=%.2f kvwait=%.2f probe=%.2f "
                    "gdn=%.2f kvwrite=%.2f other=%.2f\n", tok, n_nodes, span, t[OPP_GAP], t[OPP_MM], t[OPP_ATTN],
                    t[OPP_KVWAIT], t[OPP_PROBE], t[OPP_GDN], t[OPP_KVWRITE], t[OPP_OTHER]);
        }
        n = 0;
        tok = -1;
    }
};
static ggml_cuda_op_prof g_op_prof;
static int ggml_cuda_op_prof_class(const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:         return OPP_MM;
        case GGML_OP_FLASH_ATTN_EXT:     return OPP_ATTN;
        case GGML_OP_SET_ROWS:           return OPP_KVWRITE;
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_SSM_CONV:           return OPP_GDN;
        default:                         return OPP_OTHER;
    }
}

// KV-scratch single-context mode (opt-in LLAMA_KV_RANGE_STAGING=1): llama-kv-cache publishes, per ubatch,
// LLAMA_KV_STAGE_CELLS = "lo:hi,..." cell runs that a stripped (scratch) pass can attend to. prefetch() then
// copies only those rows of a staged layer into its slot. The other rows of the slot keep whatever finite data
// they held (zeroed once at allocation) and are masked to -inf by the same strip mask, so they get zero weight.
static bool ggml_cuda_kv_range_staging_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_KV_RANGE_STAGING");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return on;
}

// KV-scratch hot rows (opt-in LLAMA_KV_HOT_MIB=<MiB>): a device mirror of chosen COMMITTED pages of every staged
// layer, so the rows both passes read most (D's scratch: recent window + heat pages) never cross PCIe again.
// llama-kv-cache publishes LLAMA_KV_HOT_CELLS = "lo:hi,..." cell runs (from LLAMA_KV_HOT_RANGES positions below
// LLAMA_KV_HOT_COMMIT, in priority order). A slot is then filled from the mirror (device to device) where a page is
// mirrored and from the pinned host copy elsewhere: long host runs by DMA, the rest by one gather kernel per
// layer and stream that reads the mapped host memory directly. Committed rows are never written again, so the
// mirror stays equal to the host copy; anything that rewrites the host prefix (restore, clear) bumps the host
// generation and drops the mirror.
struct ggml_cuda_kv_hot_entry {
    uint32_t src_row;
    uint32_t dst_row;
    uint32_t n_rows;
    uint32_t from_mirror;   // 0: src is the mapped host layer, 1: src is the layer's mirror, 2: the cold landing buffer
};

static bool ggml_cuda_kv_hot_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_KV_HOT_MIB");
        return e != nullptr && atoi(e) > 0;
    }();
    return on;
}

// Cold-page pack (opt-in LLAMA_KV_COLD_PAGES=<pages>, with hot rows and range staging): committed scratch pages the
// mirror does not hold would otherwise reach the slot through the gather kernel reading mapped host memory, which keeps
// SMs busy for the whole PCIe transfer and slows D's matmuls (~21 ms per D step at 180K). Instead they are packed once,
// when the hot cells change, into contiguous pinned host frames ([layer][frame][HOT_PAGE rows]); every range-staged
// eval then sends one DMA per layer and stream into a small device landing buffer, and the gather kernel scatters the
// pages into the slot device to device. Committed rows are never rewritten, so the packed copy stays exact.
static size_t ggml_cuda_kv_cold_pages() {
    static const size_t n = [] {
        const char * e = getenv("LLAMA_KV_COLD_PAGES");
        return e != nullptr && atoi(e) > 0 ? (size_t) atoi(e) : (size_t) 0;
    }();
    return n;
}

// one block per entry; rows are copied as 16-byte words when everything is aligned (q4_0 rows: 576 B)
static __global__ void ggml_cuda_kv_hot_gather(const ggml_cuda_kv_hot_entry * __restrict__ entries,
                                               const char * __restrict__ host, const char * __restrict__ mirror,
                                               char * __restrict__ dst, const uint32_t row_bytes,
                                               const char * __restrict__ cold) {
    const ggml_cuda_kv_hot_entry e = entries[blockIdx.x];
    const char * src = (e.from_mirror == 2 ? cold : e.from_mirror ? mirror : host) + (size_t) e.src_row * row_bytes;
    char * d = dst + (size_t) e.dst_row * row_bytes;
    const size_t nb = (size_t) e.n_rows * row_bytes;
    if ((((uintptr_t) src) | ((uintptr_t) d) | nb) % 16 == 0) {
        const int4 * s4 = (const int4 *) src;
        int4 * d4 = (int4 *) d;
        for (size_t i = threadIdx.x; i < nb / 16; i += blockDim.x) {
            d4[i] = s4[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < nb; i += blockDim.x) {
            d[i] = src[i];
        }
    }
}

// Packed scratch attention (opt-in LLAMA_KV_PACK=1, range staging only): the strip mask only zeroes the weights of
// out-of-scratch keys, so the attention kernel still walks the whole n_kv view. With packing, a staged layer's
// attention gathers the scratch rows of its slot (and the matching mask columns) into a contiguous buffer and
// attends over that instead: D's attention costs its scratch size, not the context. The mask carries causality,
// so the result is the same attention up to summation order (D's drafts do not need to be bit-exact).
static bool ggml_cuda_kv_pack_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_KV_PACK");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return on;
}

// Paged scratch attention (opt-in LLAMA_KV_PAGED_D=1, range staging + hot rows): D's attention reads its scratch
// page by page straight from where each page lives: a mirrored page from its mirror frame, everything else (pages
// not mirrored, D's own new rows) from the slot, which the range staging fills for exactly those cells. No slot
// copy of mirrored pages and no packed copy: D's per-step KV traffic is only what is not resident. The mask carries
// causality and the strip, so this is the same attention up to summation order (D need not be bit-exact).
static bool ggml_cuda_kv_paged_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_KV_PAGED_D");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return on;
}

// packed mask column j = source column colmap[j] (UINT32_MAX: padding, masked out)
static __global__ void ggml_cuda_kv_pack_mask(const uint32_t * __restrict__ colmap, const char * __restrict__ src,
                                              const size_t src_nb1, half * __restrict__ dst, const uint32_t n_cols) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_cols) {
        return;
    }
    const uint32_t c = colmap[j];
    dst[(size_t) blockIdx.y * n_cols + j] = c == UINT32_MAX ? __float2half(-INFINITY) :
        *(const half *) (src + (size_t) blockIdx.y * src_nb1 + (size_t) c * sizeof(half));
}

// KV writes (opt-in LLAMA_KV_HOST_ROWCOPY=1): quantize the new rows into the device slot, then copy those finished
// rows to the pinned host copy with coalesced 16-byte writes. SET_ROWS straight into mapped host memory issues
// scattered 18-byte block writes over PCIe (19 ms for a 33-token check at 16 layers). Same quantizer, same bytes.
static bool ggml_cuda_kv_host_rowcopy_enabled() {
    static const bool on = [] {
        const char * e = getenv("LLAMA_KV_HOST_ROWCOPY");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    return on;
}

// one block per written row: host row idx[b] = slot row idx[b]
template <typename I>
static __global__ void ggml_cuda_kv_rows_to_host(const I * __restrict__ idx, const char * __restrict__ slot,
                                                 char * __restrict__ host, const size_t row_bytes, const size_t nb1) {
    const int64_t r = (int64_t) idx[blockIdx.x];
    const char * src = slot + r * nb1;
    char * dst = host + r * nb1;
    if ((((uintptr_t) src) | ((uintptr_t) dst) | row_bytes) % 16 == 0) {
        for (size_t i = threadIdx.x; i < row_bytes / 16; i += blockDim.x) {
            ((int4 *) dst)[i] = ((const int4 *) src)[i];
        }
    } else {
        for (size_t i = threadIdx.x; i < row_bytes; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

static const char * ggml_cuda_kv_host_name(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_cuda_buffer_type_context *>(buft->context)->name.c_str();
}
static ggml_backend_buffer_t ggml_cuda_kv_host_alloc(ggml_backend_buffer_type_t buft, size_t n);
static bool ggml_cuda_is_kv_host(ggml_backend_buffer_type_t buft) {
    // Do not use get_name identity: MSVC /OPT:ICF can fold its body with the
    // ordinary CUDA name getter, making every device buffer appear host-backed.
    return buft && buft->iface.alloc_buffer == ggml_cuda_kv_host_alloc;
}
struct ggml_cuda_kv_host_buffer {
    int device;
    void * host;
    void * base;
};
static void ggml_cuda_kv_host_free(ggml_backend_buffer_t b) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFreeHost(p->host));
    delete p;
}
static void * ggml_cuda_kv_host_base(ggml_backend_buffer_t b) {
    return static_cast<ggml_cuda_kv_host_buffer *>(b->context)->base;
}
static char * ggml_cuda_kv_host_ptr(ggml_backend_buffer_t b, const void * addr) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    return (char *) p->host + ((const char *) addr - (const char *) p->base);
}
// Everything that changes the canonical pinned prefix (restore, clear, plain set)
// bumps this, invalidating any slot that was prefetched for the next token.
static uint64_t ggml_cuda_kv_host_generation = 1;

static void ggml_cuda_kv_host_set(ggml_backend_buffer_t b, ggml_tensor * t, const void * src, size_t off, size_t n) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(ggml_cuda_kv_host_ptr(b, t->data) + off, src, n);
    ggml_cuda_kv_host_generation++;
}
static void ggml_cuda_kv_host_get(ggml_backend_buffer_t b, const ggml_tensor * t, void * dst, size_t off, size_t n) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(dst, ggml_cuda_kv_host_ptr(b, t->data) + off, n);
}
static void ggml_cuda_kv_host_memset(ggml_backend_buffer_t b, ggml_tensor * t, uint8_t value, size_t off, size_t n) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memset(ggml_cuda_kv_host_ptr(b, t->data) + off, value, n);
}
// Async backend IO is allowed to complete synchronously. State snapshots/restore
// must never issue a device-only memcpy against a mapped canonical host buffer.
static bool ggml_cuda_kv_host_io(ggml_backend_cuda_context & ctx, ggml_backend_buffer_t b,
                                const ggml_tensor * t, void * data, size_t offset,
                                size_t size, size_t count, size_t tensor_stride,
                                size_t data_stride, bool write) {
    if (!ggml_cuda_is_kv_host(b->buft)) return false;
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    char * host = ggml_cuda_kv_host_ptr(b, t->data) + offset;
    for (size_t i = 0; i < count; ++i) {
        if (write) memcpy(host + i * tensor_stride, (const char *) data + i * data_stride, size);
        else       memcpy((char *) data + i * data_stride, host + i * tensor_stride, size);
    }
    if (write) ggml_cuda_kv_host_generation++;
    return true;
}
static void ggml_cuda_kv_host_clear(ggml_backend_buffer_t b, uint8_t value) {
    auto * p = static_cast<ggml_cuda_kv_host_buffer *>(b->context);
    ggml_cuda_set_device(p->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    memset(p->host, value, b->size);
    ggml_cuda_kv_host_generation++;
}
static ggml_backend_buffer_t ggml_cuda_kv_host_alloc(ggml_backend_buffer_type_t buft, size_t n) {
    auto * type = static_cast<ggml_backend_cuda_buffer_type_context *>(buft->context);
    ggml_cuda_set_device(type->device);
    void * host = nullptr;
    cudaError_t err = cudaHostAlloc(&host, std::max<size_t>(n, 1), cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("staged KV: failed to pin %.1f MiB: %s\n", n / 1048576.0, cudaGetErrorString(err));
        (void) cudaGetLastError();
        return nullptr;
    }
    void * base = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&base, host, 0));
    const ggml_backend_buffer_i iface = {
        ggml_cuda_kv_host_free, ggml_cuda_kv_host_base, nullptr,
        ggml_cuda_kv_host_memset, ggml_cuda_kv_host_set, ggml_cuda_kv_host_get,
        nullptr, nullptr, nullptr, ggml_cuda_kv_host_clear, nullptr,
    };
    return ggml_backend_buffer_init(buft, iface, new ggml_cuda_kv_host_buffer{type->device, host, base}, n);
}
static size_t ggml_cuda_kv_host_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) {
    return ggml_nbytes(t);
}
// Registry extension: same signature as ggml_backend_dev_buffer_type(). Not a generic
// host buffer: scheduler must keep SET_ROWS + FLASH_ATTN on the owning CUDA device.
static ggml_backend_buffer_type_t ggml_backend_cuda_kv_host_buffer_type(ggml_backend_dev_t dev) {
    static std::mutex mutex;
    static std::map<ggml_backend_dev_t, std::unique_ptr<ggml_backend_buffer_type>> types;
    std::lock_guard<std::mutex> lock(mutex);
    auto & type = types[dev];
    if (!type) {
        auto * original = static_cast<ggml_backend_cuda_buffer_type_context *>(ggml_backend_dev_buffer_type(dev)->context);
        type.reset(new ggml_backend_buffer_type{
            {ggml_cuda_kv_host_name, ggml_cuda_kv_host_alloc, nullptr, ggml_backend_cuda_buffer_type_get_alignment,
             nullptr, ggml_cuda_kv_host_alloc_size, nullptr, nullptr}, dev,
            new ggml_backend_cuda_buffer_type_context{original->device, original->name + "_KV_StagedHost"}});
    }
    return type.get();
}

struct ggml_cuda_kv_staging {
    int device;
    int n_slots = 0;
    cudaStream_t copy[2] = { nullptr, nullptr };
    cudaEvent_t join[2] = {};       // [stream]: copy-stream tail, joined at the end of an eval
    static constexpr int MAX_SLOTS = 32;
    cudaEvent_t ready[2][MAX_SLOTS] = {};   // [stream][slot]: chunk copies done
    cudaEvent_t consumed[MAX_SLOTS] = {};   // [slot]: last flash attention read done (persistent across evals)
    void * k[MAX_SLOTS] = {};
    void * v[MAX_SLOTS] = {};
    size_t cap_k = 0, cap_v = 0;
    // What slot holds right now, and which canonical-host generation it came from.
    // Lets the next eval reuse a slot that was already staged at the end of the
    // previous eval instead of paying for the copy again.
    int    slot_layer[MAX_SLOTS] = {};   // set to -1 in the constructor
    uint64_t slot_gen[MAX_SLOTS] = {};
    // which staging mode filled the slot: 0 = the full prefix, otherwise a hash of the cell runs (range staging)
    uint64_t slot_mode[MAX_SLOTS] = {};
    // opt-in LLAMA_KV_STAGE_GRAPHS=1 (elastic only): an eval in which every staged layer owns a valid sticky slot and
    // no prefetch is outstanding needs no events at all (no DMA; SET_ROWS writes host + slot, attention reads the
    // slot), so it may be captured / replayed as a CUDA graph like an all-resident eval.
    bool pending[MAX_SLOTS] = {};   // prefetched, not yet waited on by the compute stream
    bool graph_mode = false;        // set by graph_compute for the current eval

    // ---- opt-in elastic slots (LLAMA_KV_STAGE_ELASTIC=<pool MiB>, use with LLAMA_KV_STAGED_LAYERS = all layers):
    // slots are sized to the used prefix (+ headroom, re-sized as the context grows) instead of a layer's full
    // capacity, and the pool holds as many of them as fit. The first n_sticky staged layers own a slot that keeps
    // their prefix across evals (new rows reach it through the intercepted SET_ROWS, as for any slot), so they cost no
    // DMA at all; the remaining layers rotate through n_rot slots as before. At short context every layer is sticky
    // (no PCIe traffic), at long context as many as fit stay, instead of a few full-capacity resident layers.
    bool   elastic    = false;
    size_t pool_bytes = 0;
    int    n_sticky   = 0;
    int    n_rot      = 0;
    size_t cap_rows   = 0;
    int slot_of(size_t i) const {
        if (!elastic) return (int) (i % (size_t) n_slots);
        return i < (size_t) n_sticky ? (int) i : n_sticky + (int) ((i - (size_t) n_sticky) % (size_t) n_rot);
    }

    // ---- KV-scratch hot rows (see ggml_cuda_kv_hot_enabled): page frames, the same frame index in every layer
    static constexpr size_t HOT_PAGE = 64;          // cells per mirrored page
    bool   hot_failed = false;
    int    hot_layers = 0;                          // staged layers the mirror holds (plan order)
    std::vector<const void *> hot_layer_k;          // K root data of each mirrored layer (identity check)
    size_t hot_k_row = 0, hot_v_row = 0;
    size_t hot_frames = 0;                          // frames per layer
    char * hot_k = nullptr;                         // [layer][frame][HOT_PAGE rows]
    char * hot_v = nullptr;
    std::vector<int32_t> hot_frame_of_page;         // cell page -> frame, -1 = not mirrored
    std::vector<int64_t> hot_page_of_frame;         // frame -> cell page, -1 = free
    std::vector<int32_t> hot_free;
    size_t   hot_mapped = 0;
    uint64_t hot_gen = 0;                           // host generation the mirror content belongs to
    std::string hot_key;                            // LLAMA_KV_HOT_CELLS the frames were assigned for
    cudaEvent_t hot_ev = nullptr;                   // main-stream point a mirror fill waits for
    cudaEvent_t hot_up_ev = nullptr;                // entry upload done (copy[0])
    cudaEvent_t hot_tail_ev = nullptr;              // copy[1] tail before an upload
    cudaEvent_t hot_fill_ev[2] = {};                // page fills done (copy[0] K, copy[1] V): paged readers wait
    ggml_cuda_kv_hot_entry * hot_e_host = nullptr;  // pinned upload buffer
    ggml_cuda_kv_hot_entry * hot_e_dev = nullptr;
    size_t hot_e_cap = 0;

    // ---- cold-page pack (see ggml_cuda_kv_cold_pages): frames of committed pages the mirror does not hold
    bool   cold_failed = false;
    size_t cold_frames = 0;                         // frames per layer
    size_t cold_used = 0;                           // highest used frame + 1: the rows each layer's DMA sends
    char * cold_host_k = nullptr;                   // pinned [layer][frame][HOT_PAGE rows]
    char * cold_host_v = nullptr;
    char * cold_dev_k = nullptr;                    // landing buffer for one layer (copy[0] K / copy[1] V in order)
    char * cold_dev_v = nullptr;
    std::vector<int32_t> cold_frame_of_page;        // cell page -> frame, -1 = not packed
    std::vector<int64_t> cold_page_of_frame;        // frame -> cell page, -1 = free
    bool cold_alloc(size_t n_layers, size_t k_row, size_t v_row) {
        if (cold_failed) return false;
        if (cold_host_k != nullptr) return true;
        cold_frames = ggml_cuda_kv_cold_pages();
        const size_t fk = HOT_PAGE * k_row, fv = HOT_PAGE * v_row;
        cudaError_t err = cudaMallocHost(&cold_host_k, n_layers * cold_frames * fk);
        if (err == cudaSuccess) err = cudaMallocHost(&cold_host_v, n_layers * cold_frames * fv);
        if (err == cudaSuccess) err = cudaMalloc(&cold_dev_k, cold_frames * fk);
        if (err == cudaSuccess) err = cudaMalloc(&cold_dev_v, cold_frames * fv);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            if (cold_host_k) CUDA_CHECK(cudaFreeHost(cold_host_k));
            if (cold_host_v) CUDA_CHECK(cudaFreeHost(cold_host_v));
            if (cold_dev_k)  CUDA_CHECK(cudaFree(cold_dev_k));
            cold_host_k = cold_host_v = cold_dev_k = cold_dev_v = nullptr;
            cold_failed = true;
            GGML_LOG_ERROR("staged KV: cold-page pack of %zu pages does not fit; using the gather kernel\n", cold_frames);
            return false;
        }
        cold_page_of_frame.assign(cold_frames, -1);
        fprintf(stderr, "[hot] cold-page pack %zu frames: %.2f MiB pinned host, %.2f MiB device landing\n", cold_frames,
                (double) (n_layers * cold_frames * (fk + fv)) / 1048576.0, (double) (cold_frames * (fk + fv)) / 1048576.0);
        return true;
    }
    int32_t cold_frame(size_t page) const {
        return page < cold_frame_of_page.size() ? cold_frame_of_page[page] : -1;
    }
    // pack the pages lying fully inside `runs` that the mirror does not hold (priority order, up to cold_frames);
    // pages still wanted keep their frame, new ones take the lowest free frames. kh/kv: host K/V base of each layer.
    void cold_assign(const std::vector<std::pair<size_t, size_t>> & runs, const std::vector<const char *> & kh,
                     const std::vector<const char *> & vh, size_t k_row, size_t v_row, cudaStream_t main) {
        std::vector<size_t> want;
        std::vector<char> seen;
        for (const auto & r : runs) {
            for (size_t p = (r.first + HOT_PAGE - 1) / HOT_PAGE; (p + 1) * HOT_PAGE <= r.second && want.size() < cold_frames; ++p) {
                if (hot_frame(p) >= 0) continue;
                if (p >= seen.size()) seen.resize(p + 1, 0);
                if (seen[p]) continue;
                seen[p] = 1;
                want.push_back(p);
            }
        }
        for (size_t f = 0; f < cold_frames; ++f) {
            const int64_t p = cold_page_of_frame[f];
            if (p >= 0 && ((size_t) p >= seen.size() || !seen[(size_t) p])) {
                cold_frame_of_page[(size_t) p] = -1;
                cold_page_of_frame[f] = -1;
            }
        }
        std::vector<std::pair<size_t, int32_t>> fresh;   // (page, frame) to pack
        size_t next_free = 0;
        for (const size_t p : want) {
            if (cold_frame(p) >= 0) continue;
            while (next_free < cold_frames && cold_page_of_frame[next_free] >= 0) ++next_free;
            if (next_free == cold_frames) break;
            if (p >= cold_frame_of_page.size()) cold_frame_of_page.resize(p + 1, -1);
            cold_frame_of_page[p] = (int32_t) next_free;
            cold_page_of_frame[next_free] = (int64_t) p;
            fresh.emplace_back(p, (int32_t) next_free);
        }
        cold_used = 0;
        for (size_t f = 0; f < cold_frames; ++f) {
            if (cold_page_of_frame[f] >= 0) cold_used = f + 1;
        }
        if (fresh.empty()) return;
        // the host rows of committed pages were written by earlier evals (main stream, copy streams), and earlier
        // DMAs may still read the frames being replaced: drain all three streams before the CPU copy
        CUDA_CHECK(cudaStreamSynchronize(main));
        CUDA_CHECK(cudaStreamSynchronize(copy[0]));
        CUDA_CHECK(cudaStreamSynchronize(copy[1]));
        const size_t fk = HOT_PAGE * k_row, fv = HOT_PAGE * v_row;
        for (size_t li = 0; li < kh.size(); ++li) {
            for (const auto & pf : fresh) {
                memcpy(cold_host_k + (li * cold_frames + (size_t) pf.second) * fk, kh[li] + pf.first * fk, fk);
                memcpy(cold_host_v + (li * cold_frames + (size_t) pf.second) * fv, vh[li] + pf.first * fv, fv);
            }
        }
    }

    // ---- packed scratch attention (see ggml_cuda_kv_pack_enabled): [gather entries | column map], uploaded on the
    // main stream once per eval
    char * pack_host = nullptr;
    char * pack_dev = nullptr;
    size_t pack_cap = 0;
    cudaEvent_t pack_ev = nullptr;   // the last upload has left pack_host
    void pages_upload(const std::vector<int2> & pages, cudaStream_t stream) {
        const std::vector<ggml_cuda_kv_hot_entry> none;
        std::vector<uint32_t> raw(pages.size() * 2);
        memcpy(raw.data(), pages.data(), pages.size() * sizeof(int2));
        pack_upload(none, raw, stream);
    }
    void pack_upload(const std::vector<ggml_cuda_kv_hot_entry> & entries, const std::vector<uint32_t> & colmap,
                     cudaStream_t stream) {
        const size_t be = entries.size() * sizeof(ggml_cuda_kv_hot_entry);
        const size_t bytes = be + colmap.size() * sizeof(uint32_t);
        if (pack_ev == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&pack_ev, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(pack_ev, stream));
        }
        if (bytes > pack_cap) {
            CUDA_CHECK(cudaDeviceSynchronize());
            if (pack_host) CUDA_CHECK(cudaFreeHost(pack_host));
            if (pack_dev)  CUDA_CHECK(cudaFree(pack_dev));
            pack_cap = std::max(bytes, 2 * pack_cap);
            CUDA_CHECK(cudaMallocHost(&pack_host, pack_cap));
            CUDA_CHECK(cudaMalloc(&pack_dev, pack_cap));
        }
        CUDA_CHECK(cudaEventSynchronize(pack_ev));
        memcpy(pack_host, entries.data(), be);
        memcpy(pack_host + be, colmap.data(), colmap.size() * sizeof(uint32_t));
        CUDA_CHECK(cudaMemcpyAsync(pack_dev, pack_host, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(pack_ev, stream));
    }

    // allocate the mirror once, sized from LLAMA_KV_HOT_MIB for n_layers staged layers
    bool hot_alloc(const std::vector<const void *> & layer_k, size_t k_row, size_t v_row) {
        if (hot_failed) return false;
        if (hot_k != nullptr) return k_row == hot_k_row && v_row == hot_v_row;
        const size_t mib = (size_t) atoi(getenv("LLAMA_KV_HOT_MIB"));
        const size_t per_frame = layer_k.size() * HOT_PAGE * (k_row + v_row);
        hot_frames = per_frame > 0 ? mib * 1048576 / per_frame : 0;
        if (hot_frames == 0) {
            hot_failed = true;
            return false;
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaError_t err = cudaMalloc(&hot_k, layer_k.size() * hot_frames * HOT_PAGE * k_row);
        if (err == cudaSuccess) err = cudaMalloc(&hot_v, layer_k.size() * hot_frames * HOT_PAGE * v_row);
        if (err != cudaSuccess) {
            (void) cudaGetLastError();
            if (hot_k) CUDA_CHECK(cudaFree(hot_k));
            hot_k = hot_v = nullptr;
            hot_failed = true;
            size_t free_b = 0, total_b = 0;
            (void) cudaMemGetInfo(&free_b, &total_b);
            GGML_LOG_ERROR("staged KV: hot-row mirror of %zu MiB does not fit in VRAM (%.0f MiB free); hot rows disabled\n",
                           mib, (double) free_b / 1048576.0);
            return false;
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&hot_ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hot_up_ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hot_tail_ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hot_fill_ev[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hot_fill_ev[1], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(hot_up_ev, copy[0]));
        hot_layers = (int) layer_k.size();
        hot_layer_k = layer_k;
        hot_k_row = k_row;
        hot_v_row = v_row;
        hot_page_of_frame.assign(hot_frames, -1);
        hot_free.clear();
        for (size_t f = hot_frames; f-- > 0;) hot_free.push_back((int32_t) f);
        // stderr like the probe's [sent] lines: harnesses run with ggml INFO logging off
        fprintf(stderr, "[hot] staged KV mirror %zu frames x %zu cells x %d layers = %.2f MiB\n",
                hot_frames, HOT_PAGE, hot_layers,
                (double) (hot_layers * hot_frames * HOT_PAGE * (k_row + v_row)) / 1048576.0);
        return true;
    }
    void hot_reset() {
        cold_frame_of_page.clear();
        cold_page_of_frame.assign(cold_frames, -1);
        cold_used = 0;
        hot_frame_of_page.clear();
        hot_page_of_frame.assign(hot_frames, -1);
        hot_free.clear();
        for (size_t f = hot_frames; f-- > 0;) hot_free.push_back((int32_t) f);
        hot_mapped = 0;
        hot_key.clear();
    }
    int32_t hot_frame(size_t page) const {
        return page < hot_frame_of_page.size() ? hot_frame_of_page[page] : -1;
    }
    // map the pages lying fully inside `runs` (priority order, up to hot_frames of them); frames of pages no longer
    // wanted are freed first. New pages are appended to `fill` as host -> mirror copies.
    void hot_assign(const std::vector<std::pair<size_t, size_t>> & runs, std::vector<ggml_cuda_kv_hot_entry> & fill) {
        std::vector<size_t> want;
        std::vector<uint8_t> seen;
        for (const auto & r : runs) {
            for (size_t p = (r.first + HOT_PAGE - 1) / HOT_PAGE; (p + 1) * HOT_PAGE <= r.second && want.size() < hot_frames; ++p) {
                if (p >= seen.size()) seen.resize(p + 1, 0);
                if (seen[p]) continue;
                seen[p] = 1;
                want.push_back(p);
            }
        }
        for (size_t f = 0; f < hot_frames; ++f) {
            const int64_t p = hot_page_of_frame[f];
            if (p >= 0 && ((size_t) p >= seen.size() || !seen[(size_t) p])) {
                hot_frame_of_page[(size_t) p] = -1;
                hot_page_of_frame[f] = -1;
                hot_free.push_back((int32_t) f);
                --hot_mapped;
            }
        }
        for (const size_t p : want) {
            if (hot_frame(p) >= 0) continue;
            if (hot_free.empty()) break;
            const int32_t f = hot_free.back();
            hot_free.pop_back();
            if (p >= hot_frame_of_page.size()) hot_frame_of_page.resize(p + 1, -1);
            hot_frame_of_page[p] = f;
            hot_page_of_frame[(size_t) f] = (int64_t) p;
            ++hot_mapped;
            fill.push_back({ (uint32_t) (p * HOT_PAGE), (uint32_t) ((size_t) f * HOT_PAGE), (uint32_t) HOT_PAGE, 0u });
        }
    }
    // upload this eval's gather entries; everything still reading the previous list has finished first
    void hot_upload(const std::vector<ggml_cuda_kv_hot_entry> & all) {
        if (all.empty()) return;
        if (all.size() > hot_e_cap) {
            CUDA_CHECK(cudaDeviceSynchronize());
            if (hot_e_host) CUDA_CHECK(cudaFreeHost(hot_e_host));
            if (hot_e_dev)  CUDA_CHECK(cudaFree(hot_e_dev));
            hot_e_cap = std::max(all.size(), 2 * hot_e_cap);
            CUDA_CHECK(cudaMallocHost(&hot_e_host, hot_e_cap * sizeof(ggml_cuda_kv_hot_entry)));
            CUDA_CHECK(cudaMalloc(&hot_e_dev, hot_e_cap * sizeof(ggml_cuda_kv_hot_entry)));
        }
        CUDA_CHECK(cudaEventSynchronize(hot_up_ev));   // the previous upload has left the pinned buffer
        memcpy(hot_e_host, all.data(), all.size() * sizeof(ggml_cuda_kv_hot_entry));
        CUDA_CHECK(cudaEventRecord(hot_tail_ev, copy[1]));
        CUDA_CHECK(cudaStreamWaitEvent(copy[0], hot_tail_ev, 0));
        CUDA_CHECK(cudaMemcpyAsync(hot_e_dev, hot_e_host, all.size() * sizeof(ggml_cuda_kv_hot_entry),
                                   cudaMemcpyHostToDevice, copy[0]));
        CUDA_CHECK(cudaEventRecord(hot_up_ev, copy[0]));
        CUDA_CHECK(cudaStreamWaitEvent(copy[1], hot_up_ev, 0));
    }

    // TEMP diagnostic: GGML_KV_STAGE_PROF=1 dumps per-layer copy/wait/attention times.
    bool prof = false;
    int  prof_evals = 0;
    std::vector<std::array<cudaEvent_t, 5>> pev;

    void ensure_prof(size_t n) {
        if (!prof) return;
        while (pev.size() < n) {
            std::array<cudaEvent_t, 5> e{};
            for (auto & ev : e) {
                CUDA_CHECK(cudaEventCreate(&ev));
            }
            pev.push_back(e);
        }
        if (!pref) {
            CUDA_CHECK(cudaEventCreate(&pref));
        }
    }
    cudaEvent_t pref = nullptr;
    void dump_prof(ggml_backend_cuda_context & ctx) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(copy[0]));
        CUDA_CHECK(cudaStreamSynchronize(copy[1]));
        double s_dma = 0, s_gap = 0, s_attn = 0;
        float span = 0;
        std::string rows;
        for (size_t i = 0; i < pev.size(); ++i) {
            float cs = 0, ce = 0, ve = 0, as = 0, ae = 0;
            CUDA_CHECK(cudaEventElapsedTime(&cs, pref, pev[i][0]));
            CUDA_CHECK(cudaEventElapsedTime(&ce, pref, pev[i][1]));
            CUDA_CHECK(cudaEventElapsedTime(&ve, pref, pev[i][2]));
            CUDA_CHECK(cudaEventElapsedTime(&as, pref, pev[i][3]));
            CUDA_CHECK(cudaEventElapsedTime(&ae, pref, pev[i][4]));
            const float copy_end = std::max(ce, ve);
            const float gap = as - copy_end;
            s_dma += copy_end - cs; s_gap += gap; s_attn += ae - as;
            span = std::max(span, ae);
            char buf[200];
            snprintf(buf, sizeof(buf), "  L%zu dma[%7.2f ->%7.2f] attn[%7.2f ->%7.2f] gap %+6.2f\n",
                     i, cs, copy_end, as, ae, gap);
            rows += buf;
        }
        fprintf(stderr, "staged KV profile eval %d: span %.2f ms, dma %.2f, idle-gap %.2f, attn %.2f\n",
                      prof_evals, span, s_dma, s_gap, s_attn);
        if (prof_evals % 32 == 0) {
            fprintf(stderr, "%s", rows.c_str());
        }
        prof_evals++;
    }

    explicit ggml_cuda_kv_staging(int dev) : device(dev) {
        prof = getenv("GGML_KV_STAGE_PROF") != nullptr;

        const char * env = getenv("LLAMA_KV_STAGE_SLOTS");
        // one slot is allowed: it lowers VRAM but serialises copy and attention
        n_slots = env && env[0] ? std::max(1, std::min(8, std::atoi(env))) : 2;
        for (int s = 0; s < MAX_SLOTS; ++s) {
            slot_layer[s] = -1;
        }
        if (const char * el = getenv("LLAMA_KV_STAGE_ELASTIC"); el && atoi(el) > 0) {
            elastic    = true;
            pool_bytes = (size_t) atoi(el) * 1024 * 1024;
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy[0], cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy[1], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&join[0], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&join[1], cudaEventDisableTiming));
        for (int s = 0; s < MAX_SLOTS; ++s) {
            CUDA_CHECK(cudaEventCreateWithFlags(&consumed[s], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(consumed[s], copy[0]));  // pre-signal: first evals may copy at once
        }
        for (int st = 0; st < 2; ++st) {
            for (int s = 0; s < MAX_SLOTS; ++s) {
                CUDA_CHECK(cudaEventCreateWithFlags(&ready[st][s], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventRecord(ready[st][s], copy[st]));
            }
        }
    }
    ~ggml_cuda_kv_staging() {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int s = 0; s < MAX_SLOTS; ++s) {
            CUDA_CHECK(cudaFree(k[s])); CUDA_CHECK(cudaFree(v[s]));
            CUDA_CHECK(cudaEventDestroy(consumed[s]));
        }
        for (int st = 0; st < 2; ++st) {
            for (int s = 0; s < MAX_SLOTS; ++s) {
                CUDA_CHECK(cudaEventDestroy(ready[st][s]));
            }
            CUDA_CHECK(cudaEventDestroy(join[st]));
            CUDA_CHECK(cudaStreamDestroy(copy[st]));
        }
        if (hot_k)       CUDA_CHECK(cudaFree(hot_k));
        if (hot_v)       CUDA_CHECK(cudaFree(hot_v));
        if (hot_e_dev)   CUDA_CHECK(cudaFree(hot_e_dev));
        if (hot_e_host)  CUDA_CHECK(cudaFreeHost(hot_e_host));
        if (hot_ev)      CUDA_CHECK(cudaEventDestroy(hot_ev));
        if (hot_up_ev)   CUDA_CHECK(cudaEventDestroy(hot_up_ev));
        if (hot_tail_ev) CUDA_CHECK(cudaEventDestroy(hot_tail_ev));
        for (auto & e : hot_fill_ev) {
            if (e) CUDA_CHECK(cudaEventDestroy(e));
        }
        if (pack_dev)    CUDA_CHECK(cudaFree(pack_dev));
        if (pack_host)   CUDA_CHECK(cudaFreeHost(pack_host));
        if (cold_dev_k)  CUDA_CHECK(cudaFree(cold_dev_k));
        if (cold_dev_v)  CUDA_CHECK(cudaFree(cold_dev_v));
        if (cold_host_k) CUDA_CHECK(cudaFreeHost(cold_host_k));
        if (cold_host_v) CUDA_CHECK(cudaFreeHost(cold_host_v));
        if (pack_ev)     CUDA_CHECK(cudaEventDestroy(pack_ev));
    }
    // elastic: (re)size the slots for `rows` used cells of `nlayers` staged layers; returns true when reallocated
    bool reserve_elastic(size_t nlayers, size_t row_k, size_t row_v, size_t rows) {
        if (cap_rows > 0 && rows <= cap_rows) return false;
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int s = 0; s < n_slots; ++s) {
            CUDA_CHECK(cudaFree(k[s])); CUDA_CHECK(cudaFree(v[s]));
            k[s] = v[s] = nullptr;
            slot_layer[s] = -1; slot_gen[s] = 0; slot_mode[s] = 0;
        }
        constexpr size_t HEAD = 4096, STEP = 2048;  // headroom so a growing context re-sizes rarely
        cap_rows = ((rows + HEAD + STEP - 1) / STEP) * STEP;
        cap_k = cap_rows * row_k; cap_v = cap_rows * row_v;
        const size_t per = cap_k + cap_v;
        const size_t fit = per > 0 ? pool_bytes / per : 0;
        if (fit >= nlayers) {
            n_sticky = (int) nlayers; n_rot = 0;
        } else {
            n_rot = 2;
            n_sticky = (int) std::min<size_t>(fit > 2 ? fit - 2 : 0, nlayers > 2 ? nlayers - 2 : 0);
        }
        n_slots = std::min(MAX_SLOTS, n_sticky + n_rot);
        n_sticky = n_slots - n_rot;
        for (int s = 0; s < n_slots; ++s) {
            cudaError_t err = cudaMalloc(&k[s], cap_k);
            if (err == cudaSuccess) err = cudaMalloc(&v[s], cap_v);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_ABORT("staged KV (elastic): slot %d of %d (%.2f MiB each) does not fit in VRAM", s, n_slots, (double) per / 1048576.0);
            }
            CUDA_CHECK(cudaMemset(k[s], 0, cap_k)); // never-written cells must read as finite zeros
            CUDA_CHECK(cudaMemset(v[s], 0, cap_v));
        }
        GGML_LOG_INFO("staged KV (elastic): %zu cells/slot, %d sticky + %d rotating slots of %.2f MiB = %.2f MiB (pool %.2f MiB, %zu layers)\n",
                      cap_rows, n_sticky, n_rot, (double) per / 1048576.0, (double) n_slots * per / 1048576.0,
                      (double) pool_bytes / 1048576.0, nlayers);
        return true;
    }
    bool reserve(size_t nk, size_t nv) {
        if (nk <= cap_k && nv <= cap_v) return false;
        CUDA_CHECK(cudaDeviceSynchronize());
        cap_k = std::max(cap_k, nk); cap_v = std::max(cap_v, nv);
        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        GGML_LOG_INFO("staged KV: %d slots x %.2f MiB = %.2f MiB; %.2f MiB dedicated VRAM free of %.2f MiB\n",
                      n_slots, (double) (cap_k + cap_v) / 1048576.0,
                      (double) n_slots * (cap_k + cap_v) / 1048576.0,
                      (double) free_b / 1048576.0, (double) total_b / 1048576.0);
        for (int s = 0; s < n_slots; ++s) {
            CUDA_CHECK(cudaFree(k[s])); CUDA_CHECK(cudaFree(v[s]));
            cudaError_t err = cudaMalloc(&k[s], cap_k);
            if (err == cudaSuccess) err = cudaMalloc(&v[s], cap_v);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("staged KV: staging slots do not fit in VRAM: need %.2f MiB, %.2f MiB free of %.2f MiB. "
                               "Lower LLAMA_KV_STAGE_SLOTS/context or stage more layers. "
                               "Failing instead of letting the driver spill to host memory.\n",
                               (double) n_slots * (cap_k + cap_v) / 1048576.0,
                               (double) free_b / 1048576.0, (double) total_b / 1048576.0);
                GGML_ABORT("staged KV staging slots do not fit in VRAM");
            }
            if (ggml_cuda_kv_range_staging_enabled()) {
                // range staging leaves rows it did not copy untouched: make sure they are finite from the start
                CUDA_CHECK(cudaMemset(k[s], 0, cap_k));
                CUDA_CHECK(cudaMemset(v[s], 0, cap_v));
            }
            slot_layer[s] = -1;
            slot_gen[s] = 0;
            slot_mode[s] = 0;
        }
        GGML_LOG_INFO("staged KV: %d device layer slots = %.2f MiB, pinned host backing; prefix-only DMA\n",
                      n_slots, (double) n_slots * (cap_k + cap_v) / 1048576.0);
        return true;
    }
};
static const ggml_tensor * ggml_cuda_kv_root(const ggml_tensor * t) {
    while (t->view_src) t = t->view_src;
    return t;
}
static bool ggml_cuda_kv_tensor(const ggml_tensor * t) {
    return t && t->buffer && ggml_cuda_is_kv_host(t->buffer->buft);
}
struct ggml_cuda_kv_plan {
    struct layer {
        ggml_tensor * attention;
        const ggml_tensor * k;
        const ggml_tensor * v;
        size_t nk, nv;
    };
    std::vector<layer> layers;
    ggml_cuda_kv_staging * staging = nullptr;
    // range staging for this eval (empty = full prefix); see ggml_cuda_kv_range_staging_enabled()
    std::vector<std::pair<size_t, size_t>> cell_runs;
    uint64_t mode = 0;
    // hot rows (ggml_cuda_kv_hot_enabled): this eval's slot recipe, the same for every mirrored layer
    bool   hot_on = false;
    size_t hot_rows = 0;                                // cells of the layer view the recipe is for
    std::vector<std::pair<size_t, size_t>> hot_dma;     // host runs copied by DMA
    size_t hot_n_fill = 0;                              // device entries [0, n_fill): host -> mirror page fills
    size_t hot_n_gather = 0;                            // device entries [n_fill, n_fill + n_gather): slot gather
    bool   cold_live = false;                           // this eval's recipe reads the cold landing buffer
    // shorter host runs go through the gather kernel; LLAMA_KV_HOT_DMA_MIN_ROWS overrides (e.g. 64: every run by DMA)
    const size_t HOT_DMA_MIN_ROWS = [] { const char * e = getenv("LLAMA_KV_HOT_DMA_MIN_ROWS"); return e ? (size_t) atoi(e) : (size_t) 512; }();
    static constexpr size_t HOT_ENTRY_ROWS   = 64;      // rows per gather block

    static std::vector<std::pair<size_t, size_t>> parse_runs(const char * e) {
        std::vector<std::pair<size_t, size_t>> runs;
        const char * p = e;
        while (*p != '\0') {
            char * end = nullptr;
            const long long lo = strtoll(p, &end, 10);
            if (end == p || *end != ':') break;
            p = end + 1;
            const long long hi = strtoll(p, &end, 10);
            if (end == p) break;
            p = end;
            if (lo >= 0 && hi > lo) runs.emplace_back((size_t) lo, (size_t) hi);
            while (*p == ',' || *p == ' ') ++p;
        }
        return runs;
    }
    // Assign mirror frames for this eval's hot cells, fill new pages, and build the slot recipe for the rows this
    // eval stages (the full prefix, or the range-staging cell runs).
    void hot_begin(ggml_backend_cuda_context & ctx) {
        hot_on = false;
        paged_want = false;
        paged_on = false;
        paged_n = 0;
        hot_dma.clear();
        hot_n_fill = hot_n_gather = 0;
        cold_live = false;
        if (!ggml_cuda_kv_hot_enabled()) return;
        const char * e = getenv("LLAMA_KV_HOT_CELLS");
        if (e == nullptr || e[0] == '\0') return;
        auto & st = *staging;
        const size_t k_row = layers[0].k->nb[1];
        const size_t v_row = layers[0].v->nb[1];
        std::vector<const void *> layer_k;
        for (const auto & l : layers) layer_k.push_back(l.k->data);
        if (!st.hot_alloc(layer_k, k_row, v_row)) return;
        // the mirror belongs to the graph it was sized for: another set of staged layers gets no hot rows
        // (a partial fill would leave frames of the missing layers stale)
        if (layer_k != st.hot_layer_k) return;
        for (const auto & l : layers) {
            if (l.k->nb[1] != k_row || l.v->nb[1] != v_row || l.nk != layers[0].nk || l.nv != layers[0].nv) return;
        }
        if (st.hot_gen != ggml_cuda_kv_host_generation) {
            st.hot_reset();
            st.hot_gen = ggml_cuda_kv_host_generation;
        }
        std::vector<ggml_cuda_kv_hot_entry> entries;
        if (st.hot_key != e) {
            st.hot_key = e;
            const auto runs = parse_runs(e);
            st.hot_assign(runs, entries);
            if (ggml_cuda_kv_cold_pages() > 0 && st.cold_alloc(layers.size(), k_row, v_row)) {
                std::vector<const char *> kh, vh;
                for (const auto & l : layers) {
                    kh.push_back(ggml_cuda_kv_host_ptr(l.k->buffer, l.k->data));
                    vh.push_back(ggml_cuda_kv_host_ptr(l.v->buffer, l.v->data));
                }
                st.cold_assign(runs, kh, vh, k_row, v_row, ctx.stream());
            }
        }
        hot_n_fill = entries.size();
        if (st.hot_mapped == 0) return;

        // slot recipe: walk the staged rows, mirrored pages from the mirror, the rest from host. A paged D eval
        // reads mirrored pages in place, so they are not copied into the slot at all.
        paged_want = ggml_cuda_kv_paged_enabled() && mode != 0 && !cell_runs.empty();
        hot_rows = k_row > 0 ? layers[0].nk / k_row : 0;
        std::vector<std::pair<size_t, size_t>> want;
        if (mode == 0) {
            want.emplace_back(0, hot_rows);
        } else {
            for (const auto & r : cell_runs) {
                const size_t lo = std::min(r.first, hot_rows), hi = std::min(r.second, hot_rows);
                if (hi > lo) want.emplace_back(lo, hi);
            }
        }
        const size_t P = ggml_cuda_kv_staging::HOT_PAGE;
        auto add_entries = [&](size_t src, size_t dst, size_t n, uint32_t from_mirror) {
            for (size_t o = 0; o < n; o += HOT_ENTRY_ROWS) {
                const size_t m = std::min(HOT_ENTRY_ROWS, n - o);
                entries.push_back({ (uint32_t) (src + o), (uint32_t) (dst + o), (uint32_t) m, from_mirror });
            }
        };
        // current segment: kind (0 host, 1 mirror), source and destination start, length
        int seg_kind = -1;
        size_t seg_src = 0, seg_dst = 0, seg_n = 0;
        const bool use_cold = mode != 0 && st.cold_used > 0;
        auto flush = [&]() {
            if (seg_kind == 1) {
                if (!paged_want) add_entries(seg_src, seg_dst, seg_n, 1u);
            } else if (seg_kind == 2) {
                add_entries(seg_src, seg_dst, seg_n, 2u);   // the slot must hold them, paged or not
                cold_live = true;
            } else if (seg_kind == 0) {
                if (seg_n >= HOT_DMA_MIN_ROWS) hot_dma.emplace_back(seg_dst, seg_dst + seg_n);
                else add_entries(seg_dst, seg_dst, seg_n, 0u);
            }
            seg_kind = -1;
            seg_n = 0;
        };
        auto walk = [&]() {
            for (const auto & r : want) {
                size_t row = r.first;
                while (row < r.second) {
                    const size_t page = row / P;
                    const size_t end  = std::min(r.second, (page + 1) * P);
                    const int32_t f   = st.hot_frame(page);
                    const int32_t cf  = use_cold && f < 0 ? st.cold_frame(page) : -1;
                    const int kind    = f >= 0 ? 1 : cf >= 0 ? 2 : 0;
                    const size_t src  = f >= 0 ? (size_t) f * P + (row - page * P)
                                      : cf >= 0 ? (size_t) cf * P + (row - page * P) : row;
                    if (kind != seg_kind || src != seg_src + seg_n || row != seg_dst + seg_n) {
                        flush();
                        seg_kind = kind;
                        seg_src = src;
                        seg_dst = row;
                    }
                    seg_n += end - row;
                    row = end;
                }
            }
            flush();
        };
        walk();
        hot_n_gather = entries.size() - hot_n_fill;
        st.hot_upload(entries);
        if (paged_want && hot_rows % P == 0) {
            // every 64-cell page that holds a scratch cell, ascending: (first cell, mirror frame or -1)
            std::vector<int2> pages;
            for (const auto & r : want) {
                for (size_t page = r.first / P; page * P < r.second; ++page) {
                    if (!pages.empty() && (size_t) pages.back().x >= page * P) continue;
                    pages.push_back(make_int2((int) (page * P), st.hot_frame(page)));
                }
            }
            if (!pages.empty()) {
                st.pages_upload(pages, ctx.stream());
                paged_n = (int) pages.size();
                paged_on = true;
                paged_pages = pages;
            }
        }
        if (paged_want && !paged_on) {
            // no page list: the slot must hold the mirrored pages after all (plain range staging)
            paged_want = false;
            hot_n_gather = 0;
            hot_dma.clear();
            entries.resize(hot_n_fill);
            seg_kind = -1;
            seg_n = 0;
            cold_live = false;
            walk();
            hot_n_gather = entries.size() - hot_n_fill;
            st.hot_upload(entries);
        }

        if (hot_n_fill > 0) {
            // new pages: their host rows were written by earlier work on the main stream
            CUDA_CHECK(cudaEventRecord(st.hot_ev, ctx.stream()));
            CUDA_CHECK(cudaStreamWaitEvent(st.copy[0], st.hot_ev, 0));
            CUDA_CHECK(cudaStreamWaitEvent(st.copy[1], st.hot_ev, 0));
            for (int li = 0; li < st.hot_layers; ++li) {
                const auto & l = layers[(size_t) li];
                ggml_cuda_kv_hot_gather<<<(unsigned) hot_n_fill, 256, 0, st.copy[0]>>>(st.hot_e_dev,
                    (const char *) l.k->data, nullptr, st.hot_k + (size_t) li * st.hot_frames * P * k_row, (uint32_t) k_row, nullptr);
                ggml_cuda_kv_hot_gather<<<(unsigned) hot_n_fill, 256, 0, st.copy[1]>>>(st.hot_e_dev,
                    (const char *) l.v->data, nullptr, st.hot_v + (size_t) li * st.hot_frames * P * v_row, (uint32_t) v_row, nullptr);
            }
            CUDA_CHECK(cudaGetLastError());
            if (paged_on) {
                // paged attention reads the frames on the main stream, not through a slot whose ready event
                // would order it after the fill: wait for the fills explicitly
                CUDA_CHECK(cudaEventRecord(st.hot_fill_ev[0], st.copy[0]));
                CUDA_CHECK(cudaEventRecord(st.hot_fill_ev[1], st.copy[1]));
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), st.hot_fill_ev[0], 0));
                CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), st.hot_fill_ev[1], 0));
            }
        }
        hot_on = true;
    }
    // paged scratch attention (ggml_cuda_kv_paged_enabled): this eval's page list, the same for every staged layer
    bool   paged_want = false;   // range-staged eval with paging enabled: mirrored pages are not copied to the slot
    bool   paged_on = false;     // page list uploaded for this eval
    int    paged_n = 0;
    std::vector<int2> paged_pages;   // host copy (LLAMA_KV_PAGED_VERIFY)
    // packed scratch attention (ggml_cuda_kv_pack_enabled): this eval's recipe, the same for every staged layer
    bool   pack_on = false;
    size_t pack_view_rows = 0;   // cells of the layer view the recipe is for
    size_t pack_used = 0;        // packed scratch rows
    size_t pack_rows = 0;        // packed rows incl. padding (multiple of 256: FA KV stride)
    size_t pack_n_entries = 0;
    void pack_begin(ggml_backend_cuda_context & ctx) {
        pack_on = false;
        if (!ggml_cuda_kv_pack_enabled() || paged_on || mode == 0 || cell_runs.empty()) return;
        const size_t k_row = layers[0].k->nb[1];
        pack_view_rows = k_row > 0 ? layers[0].nk / k_row : 0;
        std::vector<ggml_cuda_kv_hot_entry> entries;
        std::vector<uint32_t> colmap;
        for (const auto & r : cell_runs) {
            const size_t lo = std::min(r.first, pack_view_rows), hi = std::min(r.second, pack_view_rows);
            for (size_t o = lo; o < hi; o += HOT_ENTRY_ROWS) {
                const size_t m = std::min(HOT_ENTRY_ROWS, hi - o);
                entries.push_back({ (uint32_t) o, (uint32_t) colmap.size(), (uint32_t) m, 1u });   // src = the slot
                for (size_t c = o; c < o + m; ++c) colmap.push_back((uint32_t) c);
            }
        }
        pack_used = colmap.size();
        pack_rows = (pack_used + 255) / 256 * 256;
        if (pack_used == 0 || pack_rows >= pack_view_rows) return;   // nothing to gain
        colmap.resize(pack_rows, UINT32_MAX);
        pack_n_entries = entries.size();
        staging->pack_upload(entries, colmap, ctx.stream());
        pack_on = true;
    }
    // attention over the packed scratch rows of slot s; false = not applicable, run the full view
    bool pack_attention(ggml_backend_cuda_context & ctx, size_t i, int s, ggml_tensor * node) {
        const auto & l = layers[i];
        const ggml_tensor * mask = node->src[3];
        if (!pack_on || mask == nullptr || mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1 ||
                l.nk / l.k->nb[1] != pack_view_rows || l.nv / l.v->nb[1] != pack_view_rows ||
                node->src[1]->nb[1] != l.k->nb[1] || node->src[2]->nb[1] != l.v->nb[1] ||
                node->src[1]->data != l.k->data || node->src[2]->data != l.v->data) {
            return false;
        }
        cudaStream_t stream = ctx.stream();
        const size_t k_row = l.k->nb[1], v_row = l.v->nb[1];
        const int64_t n_mrow = mask->ne[1];
        ggml_cuda_pool_alloc<char> pk(ctx.pool(), pack_rows * k_row);
        ggml_cuda_pool_alloc<char> pv(ctx.pool(), pack_rows * v_row);
        ggml_cuda_pool_alloc<half> pm(ctx.pool(), pack_rows * (size_t) n_mrow);
        if (pack_rows > pack_used) {
            // padding rows are masked out, but must be finite (0 * NaN would poison the sums)
            CUDA_CHECK(cudaMemsetAsync(pk.get() + pack_used * k_row, 0, (pack_rows - pack_used) * k_row, stream));
            CUDA_CHECK(cudaMemsetAsync(pv.get() + pack_used * v_row, 0, (pack_rows - pack_used) * v_row, stream));
        }
        const auto * entries = (const ggml_cuda_kv_hot_entry *) staging->pack_dev;
        const auto * colmap  = (const uint32_t *) (staging->pack_dev + pack_n_entries * sizeof(ggml_cuda_kv_hot_entry));
        ggml_cuda_kv_hot_gather<<<(unsigned) pack_n_entries, 256, 0, stream>>>(entries, nullptr,
            (const char *) staging->k[s], pk.get(), (uint32_t) k_row, nullptr);
        ggml_cuda_kv_hot_gather<<<(unsigned) pack_n_entries, 256, 0, stream>>>(entries, nullptr,
            (const char *) staging->v[s], pv.get(), (uint32_t) v_row, nullptr);
        ggml_cuda_kv_pack_mask<<<dim3((unsigned) ((pack_rows + 255) / 256), (unsigned) n_mrow), 256, 0, stream>>>(
            colmap, (const char *) mask->data, mask->nb[1], pm.get(), (uint32_t) pack_rows);
        CUDA_CHECK(cudaGetLastError());

        ggml_tensor k = *node->src[1], v = *node->src[2], m = *mask, out = *node;
        k.ne[1] = (int64_t) pack_rows;
        k.data  = pk.get();
        k.nb[3] = k.nb[1] * k.ne[1];
        v.ne[1] = (int64_t) pack_rows;
        v.data  = pv.get();
        v.nb[3] = v.nb[1] * v.ne[1];
        m.ne[0] = (int64_t) pack_rows;
        m.data  = pm.get();
        m.nb[1] = pack_rows * sizeof(half);
        m.nb[2] = m.nb[1] * m.ne[1];
        m.nb[3] = m.nb[2] * m.ne[2];
        out.src[1] = &k; out.src[2] = &v; out.src[3] = &m;
        ggml_cuda_flash_attn_ext(ctx, &out);
        return true;
    }
    // Opt-in LLAMA_KV_PAGED_VERIFY=N: for the first N paged layer evals, rebuild the same pages in the same order as
    // one contiguous copy (mirror frames / slot rows + the matching mask columns), run the ordinary kernel on it
    // and compare with the paged output. Same data, same chunk order, same split partition: must be bit-identical.
    void paged_verify(ggml_backend_cuda_context & ctx, size_t i, int s, const ggml_tensor * node, const ggml_tensor & out,
                      const char * mk, const char * mv) {
        static int left = [] {
            const char * e = getenv("LLAMA_KV_PAGED_VERIFY");
            return e != nullptr ? atoi(e) : 0;
        }();
        if (left <= 0) return;
        --left;
        cudaStream_t stream = ctx.stream();
        const auto & l = layers[i];
        const size_t P = ggml_cuda_kv_staging::HOT_PAGE;
        const size_t k_row = l.k->nb[1], v_row = l.v->nb[1];
        const size_t rows = (size_t) paged_n * P;
        const ggml_tensor * mask = node->src[3];
        const int64_t n_mrow = mask->ne[1];
        // gather entries: from the mirror frame (from_mirror=1) or the slot (from_mirror=0, passed as "host")
        std::vector<ggml_cuda_kv_hot_entry> e;
        std::vector<uint32_t> colmap;
        for (int j = 0; j < paged_n; ++j) {
            const int2 pg = paged_pages[(size_t) j];
            e.push_back({ (uint32_t) (pg.y >= 0 ? (size_t) pg.y * P : (size_t) pg.x), (uint32_t) (j * P), (uint32_t) P,
                          pg.y >= 0 ? 1u : 0u });
            for (size_t r = 0; r < P; ++r) colmap.push_back((uint32_t) (pg.x + r));
        }
        ggml_cuda_pool_alloc<char> buf(ctx.pool(), e.size() * sizeof(e[0]) + colmap.size() * 4);
        CUDA_CHECK(cudaMemcpyAsync(buf.get(), e.data(), e.size() * sizeof(e[0]), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(buf.get() + e.size() * sizeof(e[0]), colmap.data(), colmap.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        ggml_cuda_pool_alloc<char> pk(ctx.pool(), rows * k_row), pv(ctx.pool(), rows * v_row);
        ggml_cuda_pool_alloc<half> pm(ctx.pool(), rows * (size_t) n_mrow);
        const auto * ed = (const ggml_cuda_kv_hot_entry *) buf.get();
        ggml_cuda_kv_hot_gather<<<(unsigned) e.size(), 256, 0, stream>>>(ed, (const char *) staging->k[s], mk, pk.get(), (uint32_t) k_row, nullptr);
        ggml_cuda_kv_hot_gather<<<(unsigned) e.size(), 256, 0, stream>>>(ed, (const char *) staging->v[s], mv, pv.get(), (uint32_t) v_row, nullptr);
        ggml_cuda_kv_pack_mask<<<dim3((unsigned) ((rows + 255) / 256), (unsigned) n_mrow), 256, 0, stream>>>(
            (const uint32_t *) (buf.get() + e.size() * sizeof(e[0])), (const char *) mask->data, mask->nb[1], pm.get(),
            (uint32_t) rows);
        ggml_tensor k = *node->src[1], v = *node->src[2], m = *mask, ref = out;
        k.ne[1] = (int64_t) rows; k.data = pk.get(); k.nb[3] = k.nb[1] * k.ne[1];
        v.ne[1] = (int64_t) rows; v.data = pv.get(); v.nb[3] = v.nb[1] * v.ne[1];
        m.ne[0] = (int64_t) rows; m.data = pm.get(); m.nb[1] = rows * sizeof(half); m.nb[2] = m.nb[1] * m.ne[1]; m.nb[3] = m.nb[2];
        const size_t nbytes = ggml_nbytes(&out);
        ggml_cuda_pool_alloc<char> ro(ctx.pool(), nbytes);
        ref.data = ro.get();
        ref.src[1] = &k; ref.src[2] = &v; ref.src[3] = &m;
        ggml_cuda_flash_attn_ext_decode_q4(ctx, &ref);
        std::vector<float> a(nbytes / 4), b(nbytes / 4);
        CUDA_CHECK(cudaMemcpyAsync(a.data(), out.data, nbytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), ro.get(), nbytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        size_t ndiff = 0;
        float maxd = 0.0f, maxa = 0.0f;
        for (size_t x = 0; x < a.size(); ++x) {
            if (a[x] != b[x]) ++ndiff;
            maxd = std::max(maxd, std::fabs(a[x] - b[x]));
            maxa = std::max(maxa, std::fabs(b[x]));
        }
        fprintf(stderr, "[paged-verify] layer %zu pages %d mirrored %d: %zu of %zu values differ, max|diff| %.3g (max|ref| %.3g)\n",
                i, paged_n, (int) std::count_if(paged_pages.begin(), paged_pages.end(), [](const int2 & q) { return q.y >= 0; }),
                ndiff, a.size(), maxd, maxa);
    }
    bool hot_layer(size_t i) const {
        const auto & l = layers[i];
        return hot_on && (int) i < staging->hot_layers && staging->hot_layer_k[i] == l.k->data &&
               l.k->nb[1] == staging->hot_k_row && l.v->nb[1] == staging->hot_v_row &&
               l.nk / l.k->nb[1] == hot_rows && l.nv / l.v->nb[1] == hot_rows;
    }
    void read_cell_runs() {
        cell_runs.clear();
        mode = 0;
        if (!ggml_cuda_kv_range_staging_enabled()) return;
        const char * e = getenv("LLAMA_KV_STAGE_CELLS");
        if (e == nullptr || e[0] == '\0') return;
        const std::string s(e);
        mode = (uint64_t) std::hash<std::string>{}(s) | 1u;   // never 0 = "full prefix"
        const char * p = s.c_str();
        while (*p != '\0') {
            char * end = nullptr;
            const long long lo = strtoll(p, &end, 10);
            if (end == p || *end != ':') break;
            p = end + 1;
            const long long hi = strtoll(p, &end, 10);
            if (end == p) break;
            p = end;
            if (lo >= 0 && hi > lo) cell_runs.emplace_back((size_t) lo, (size_t) hi);
            while (*p == ',' || *p == ' ') ++p;
        }
    }
    explicit ggml_cuda_kv_plan(ggml_cgraph * graph) {
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * node = graph->nodes[i];
            if (node->op != GGML_OP_FLASH_ATTN_EXT || !ggml_cuda_kv_tensor(node->src[1])) continue;
            GGML_ASSERT(ggml_cuda_kv_tensor(node->src[2]));
            const auto * k = ggml_cuda_kv_root(node->src[1]);
            const auto * v = ggml_cuda_kv_root(node->src[2]);
            // Flash input: [head_dim, cells, heads, sequences]. Only one sequence/stream.
            GGML_ASSERT(node->src[1]->ne[3] == 1 && k->ne[2] == 1 && v->ne[2] == 1);
            GGML_ASSERT(k->nb[1] == ggml_row_size(k->type, k->ne[0]));
            GGML_ASSERT(v->nb[1] == ggml_row_size(v->type, v->ne[0]));
            layers.push_back({node, k, v, k->nb[1] * size_t(node->src[1]->ne[1]),
                                            v->nb[1] * size_t(node->src[2]->ne[1])});
        }
    }
    // Copy layer i's used prefix into slot (i % n_slots): K on copy[0], V on
    // copy[1]. Waits ONLY on the slot's persistent consumed event, i.e. the last
    // flash attention that read this slot in ANY earlier eval, so the DMA starts
    // as soon as the previous token released the slot.
    void prefetch(size_t i) {
        const auto & l = layers[i]; const int s = staging->slot_of(i);
        static const bool trace = getenv("LLAMA_KV_STAGE_TRACE") != nullptr;
        if (trace) { fprintf(stderr, "kv prefetch %zu %s K=%zu V=%zu\n", i, l.k->name, l.nk, l.nv); fflush(stderr); }
        const char * src_k = ggml_cuda_kv_host_ptr(l.k->buffer, l.k->data);
        const char * src_v = ggml_cuda_kv_host_ptr(l.v->buffer, l.v->data);
        for (int st = 0; st < 2; ++st) {
            CUDA_CHECK(cudaStreamWaitEvent(staging->copy[st], staging->consumed[s], 0));
        }
        if (staging->prof) {
            CUDA_CHECK(cudaEventRecord(staging->pev[i][0], staging->copy[0]));
        }
        if (hot_layer(i)) {
            // hot rows: long host runs by DMA, mirrored pages and short host runs by the gather kernel
            const size_t k_row = l.k->nb[1];
            const size_t v_row = l.v->nb[1];
            for (const auto & r : hot_dma) {
                CUDA_CHECK(cudaMemcpyAsync((char *) staging->k[s] + r.first * k_row, src_k + r.first * k_row,
                                           (r.second - r.first) * k_row, cudaMemcpyHostToDevice, staging->copy[0]));
                CUDA_CHECK(cudaMemcpyAsync((char *) staging->v[s] + r.first * v_row, src_v + r.first * v_row,
                                           (r.second - r.first) * v_row, cudaMemcpyHostToDevice, staging->copy[1]));
            }
            if (hot_n_gather > 0) {
                const size_t P = ggml_cuda_kv_staging::HOT_PAGE;
                if (cold_live) {
                    // this layer's packed cold frames, one DMA per stream; the scatter below follows in stream order,
                    // and the next layer's DMA into the same landing buffer queues behind it
                    const size_t nf = staging->cold_used, fr = staging->cold_frames;
                    CUDA_CHECK(cudaMemcpyAsync(staging->cold_dev_k, staging->cold_host_k + i * fr * P * k_row,
                                               nf * P * k_row, cudaMemcpyHostToDevice, staging->copy[0]));
                    CUDA_CHECK(cudaMemcpyAsync(staging->cold_dev_v, staging->cold_host_v + i * fr * P * v_row,
                                               nf * P * v_row, cudaMemcpyHostToDevice, staging->copy[1]));
                }
                ggml_cuda_kv_hot_gather<<<(unsigned) hot_n_gather, 256, 0, staging->copy[0]>>>(
                    staging->hot_e_dev + hot_n_fill, (const char *) l.k->data,
                    staging->hot_k + i * staging->hot_frames * P * k_row, (char *) staging->k[s], (uint32_t) k_row,
                    staging->cold_dev_k);
                ggml_cuda_kv_hot_gather<<<(unsigned) hot_n_gather, 256, 0, staging->copy[1]>>>(
                    staging->hot_e_dev + hot_n_fill, (const char *) l.v->data,
                    staging->hot_v + i * staging->hot_frames * P * v_row, (char *) staging->v[s], (uint32_t) v_row,
                    staging->cold_dev_v);
                CUDA_CHECK(cudaGetLastError());
            }
        } else if (mode == 0) {
            // One contiguous copy per stream: chunking was measured not to change
            // throughput (13.4 GB/s for 1 to 16 chunks), so keep the simple form.
            CUDA_CHECK(cudaMemcpyAsync(staging->k[s], src_k, l.nk, cudaMemcpyHostToDevice, staging->copy[0]));
            CUDA_CHECK(cudaMemcpyAsync(staging->v[s], src_v, l.nv, cudaMemcpyHostToDevice, staging->copy[1]));
        } else {
            // range staging: only the cell runs this (stripped) pass can attend to
            const size_t k_row = l.k->nb[1];
            const size_t v_row = l.v->nb[1];
            const size_t rows = k_row > 0 ? l.nk / k_row : 0;
            for (const auto & r : cell_runs) {
                const size_t lo = std::min(r.first, rows);
                const size_t hi = std::min(r.second, rows);
                if (hi <= lo) continue;
                CUDA_CHECK(cudaMemcpyAsync((char *) staging->k[s] + lo * k_row, src_k + lo * k_row, (hi - lo) * k_row,
                                           cudaMemcpyHostToDevice, staging->copy[0]));
                CUDA_CHECK(cudaMemcpyAsync((char *) staging->v[s] + lo * v_row, src_v + lo * v_row, (hi - lo) * v_row,
                                           cudaMemcpyHostToDevice, staging->copy[1]));
            }
        }
        CUDA_CHECK(cudaEventRecord(staging->ready[0][s], staging->copy[0]));
        CUDA_CHECK(cudaEventRecord(staging->ready[1][s], staging->copy[1]));
        if (staging->prof) {
            CUDA_CHECK(cudaEventRecord(staging->pev[i][1], staging->copy[0]));
            CUDA_CHECK(cudaEventRecord(staging->pev[i][2], staging->copy[1]));
        }
        staging->slot_layer[s] = (int) i;
        staging->pending[s] = true;
        staging->slot_gen[s] = ggml_cuda_kv_host_generation;
        staging->slot_mode[s] = mode;
    }
    // LLAMA_KV_STAGE_GRAPHS=1: true when this eval needs no staging events (see ggml_cuda_kv_staging::graph_mode)
    bool graph_ok(const ggml_cuda_kv_staging * st) const {
        static const bool on = [] { const char * e = getenv("LLAMA_KV_STAGE_GRAPHS"); return e != nullptr && atoi(e) != 0; }();
        static const bool extras = getenv("GGML_KV_STAGE_PROF") != nullptr || getenv("LLAMA_HOT_STAGED") != nullptr ||
                                   getenv("LLAMA_KV_STAGE_JOIN") != nullptr || getenv("LLAMA_KV_STAGE_TRACE") != nullptr;
        if (!on || extras || st == nullptr || !st->elastic || st->n_rot != 0 || st->prof || layers.empty() ||
                (int) layers.size() > st->n_sticky || ggml_cuda_kv_hot_enabled() || ggml_cuda_kv_range_staging_enabled() ||
                ggml_cuda_kv_pack_enabled()) {
            return false;
        }
        for (size_t i = 0; i < layers.size(); ++i) {
            const int s = st->slot_of(i);
            if (st->pending[s] || st->slot_layer[s] != (int) i || st->slot_gen[s] != ggml_cuda_kv_host_generation ||
                    st->slot_mode[s] != 0) {
                return false;
            }
        }
        return true;
    }
    void begin(ggml_backend_cuda_context & ctx) {
        if (layers.empty()) return;
        staging = ctx.kv_staging.get();
        read_cell_runs();
        hot_begin(ctx);
        pack_begin(ctx);
        // Prefetch the slots this eval starts with. A slot already staged for the
        // same layer at the end of the previous eval is reused: the rows written
        // since then are applied by SET_ROWS, so the copy is still valid.
        staging->ensure_prof(layers.size());
        if (staging->prof) {
            CUDA_CHECK(cudaEventRecord(staging->pref, ctx.stream()));
        }
        const int n = std::min<size_t>(staging->n_slots, layers.size());
        for (size_t i = 0; i < (size_t) n; ++i) {
            const int s = staging->slot_of(i);
            // reuse only a slot filled in the same mode: a range-staged slot is not a full prefix, and vice versa
            // a full prefix is a superset of any range set but the cheaper rule is enough here
            if (staging->slot_layer[s] == (int) i && staging->slot_gen[s] == ggml_cuda_kv_host_generation &&
                    staging->slot_mode[s] == mode) continue;
            if (staging->elastic) {
                // a replayed graph records no consumed event: order this copy after all earlier compute work
                CUDA_CHECK(cudaEventRecord(staging->consumed[s], ctx.stream()));
            }
            prefetch(i);
        }
    }
    // Intercept writes: wait for the prefetched OLD prefix, update canonical pinned
    // storage using CUDA's existing quantizer, then update the device slot identically.
    // This avoids racing prefetch with host writes, and preserves arbitrary row indices,
    // speculative overwrite/reject, and prefix reuse without a second KV position ledger.
    bool write(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
        if (node->op != GGML_OP_SET_ROWS || !ggml_cuda_kv_tensor(node)) return false;
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & l = layers[i];
            const auto * root = ggml_cuda_kv_root(node);
            const bool is_k = root->data == l.k->data;
            if (!is_k && root->data != l.v->data) continue;
            const int s = staging->slot_of(i);
            static const bool trace_w = getenv("LLAMA_KV_STAGE_TRACE") != nullptr;
            if (trace_w) fprintf(stderr, "kv write %zu %s\n", i, node->name);
            if (!staging->graph_mode) {
                for (int st = 0; st < 2; ++st) {
                    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->ready[st][s], 0));
                }
                staging->pending[s] = false;
            }
            g_op_prof.mark(ctx.stream(), OPP_KVWAIT);
            ggml_tensor staged = *node;
            staged.data = (char *) (is_k ? staging->k[s] : staging->v[s]) +
                          ((char *) node->data - (char *) root->data);
            const ggml_tensor * idx = node->src[1];
            const bool rowcopy = ggml_cuda_kv_host_rowcopy_enabled() && node->src[0]->ne[2] == 1 &&
                    node->src[0]->ne[3] == 1 && idx->ne[1] == 1 && idx->ne[2] == 1 && ggml_is_contiguous(idx) &&
                    idx->ne[0] > 0 && (idx->type == GGML_TYPE_I64 || idx->type == GGML_TYPE_I32);
            if (rowcopy) {
                ggml_cuda_op_set_rows(ctx, &staged);
                const size_t row_bytes = ggml_row_size(node->type, node->ne[0]);
                if (idx->type == GGML_TYPE_I64) {
                    ggml_cuda_kv_rows_to_host<int64_t><<<(unsigned) idx->ne[0], 64, 0, ctx.stream()>>>(
                        (const int64_t *) idx->data, (const char *) staged.data, (char *) node->data, row_bytes, node->nb[1]);
                } else {
                    ggml_cuda_kv_rows_to_host<int32_t><<<(unsigned) idx->ne[0], 64, 0, ctx.stream()>>>(
                        (const int32_t *) idx->data, (const char *) staged.data, (char *) node->data, row_bytes, node->nb[1]);
                }
                CUDA_CHECK(cudaGetLastError());
            } else {
                ggml_cuda_op_set_rows(ctx, node); // small, coalesced writes into mapped pinned RAM
                ggml_cuda_op_set_rows(ctx, &staged);
            }
            g_op_prof.mark(ctx.stream(), OPP_KVWRITE);
            return true;
        }
        GGML_ABORT("staged KV write has no attention consumer in this CUDA graph");
    }
    bool attention(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto & l = layers[i];
            if (node != l.attention) continue;
            static const bool trace_a = getenv("LLAMA_KV_STAGE_TRACE") != nullptr;
            if (trace_a) fprintf(stderr, "kv attention %zu\n", i);
            const int s = staging->slot_of(i);
            if (!staging->graph_mode) {
                for (int st = 0; st < 2; ++st) {
                    CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->ready[st][s], 0));
                }
                staging->pending[s] = false;
            }
            g_op_prof.mark(ctx.stream(), OPP_KVWAIT);
            ggml_tensor k = *node->src[1], v = *node->src[2], out = *node;
            k.data = (char *) staging->k[s] + ((char *) k.data - (char *) l.k->data);
            v.data = (char *) staging->v[s] + ((char *) v.data - (char *) l.v->data);
            out.src[1] = &k; out.src[2] = &v;
            if (staging->prof) {
                CUDA_CHECK(cudaEventRecord(staging->pev[i][3], ctx.stream()));
            }
            if (paged_on && hot_layer(i) && ggml_cuda_fattn_decode_q4_supported(ctx.device, node)) {
                const size_t P = ggml_cuda_kv_staging::HOT_PAGE;
                const char * mk = staging->hot_k + i * staging->hot_frames * P * l.k->nb[1];
                const char * mv = staging->hot_v + i * staging->hot_frames * P * l.v->nb[1];
                ggml_cuda_flash_attn_ext_decode_q4_paged(ctx, &out, (const int2 *) staging->pack_dev, paged_n, mk, mv);
                paged_verify(ctx, i, s, node, out, mk, mv);
            } else if (!pack_attention(ctx, i, s, node)) {
                ggml_cuda_flash_attn_ext(ctx, &out);
            }
            g_op_prof.mark(ctx.stream(), OPP_ATTN);
            // Probe only after F's attention work has been queued. `out` carries the staged K/V slot,
            // so the side kernel still measures the exact cache consumed by this layer.
            static const bool hot_staged = getenv("LLAMA_HOT_STAGED") != nullptr;
            if (hot_staged) {
                ggml_cuda_hot_probe_node(ctx, &out);
            }
            g_op_prof.mark(ctx.stream(), OPP_PROBE);
            if (staging->prof) {
                CUDA_CHECK(cudaEventRecord(staging->pev[i][4], ctx.stream()));
            }
            // Release the slot for the next use (next eval, or a later layer of
            // this eval) only after ALL attention kernels have consumed it.
            if (!staging->graph_mode) {
                CUDA_CHECK(cudaEventRecord(staging->consumed[s], ctx.stream()));
            }
            if (staging->elastic) {
                // sticky layers keep their slot; a rotating slot is refilled for its next user in this eval, or
                // for its first user of the next eval
                if (i >= (size_t) staging->n_sticky && staging->n_rot > 0) {
                    const size_t nr = (size_t) staging->n_rot;
                    if (i + nr < layers.size()) {
                        prefetch(i + nr);
                    } else {
                        prefetch((size_t) staging->n_sticky + (i - (size_t) staging->n_sticky) % nr);
                    }
                }
                return true;
            }
            const size_t ns = (size_t) staging->n_slots;
            if (i + ns < layers.size()) {
                prefetch(i + ns);
            } else if (i >= ns) {
                // Last user of this slot in this eval: stage the layer that will use
                // it first in the NEXT eval. This gives the copy a whole layer-gap to
                // run in; otherwise every token would stall one full prefix copy at
                // its leading staged attention.
                prefetch(i % ns);
            }
            return true;
        }
        return false;
    }
    // Opt-in A/B knob. Normally NOT joined: a slot staged for the next eval must be
    // free to keep copying past the end of this eval, which is what removes the
    // start-of-token stall.
    void end(ggml_backend_cuda_context & ctx) {
        if (staging && staging->prof) {
            staging->dump_prof(ctx);
        }
        static const bool join = getenv("LLAMA_KV_STAGE_JOIN") != nullptr;
        if (!staging || !join) return;
        CUDA_CHECK(cudaEventRecord(staging->join[0], staging->copy[0]));
        CUDA_CHECK(cudaEventRecord(staging->join[1], staging->copy[1]));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->join[0], 0));
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), staging->join[1], 0));
    }
};
