#include "strip.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------------------------
// Stripped-model study hooks. See strip.cuh for the environment variables and the equivalence
// argument (zeroing the 256-row block of the SWIGLU output == deleting that FFN group).
//
// Nothing here runs unless a LLAMA_STRIP_* variable is set on the command line.
// ---------------------------------------------------------------------------------------------

static constexpr int STRIP_N_FFN_GROUPS = 68;   // 17408 / 256
static constexpr int STRIP_N_ATTN_HEADS = 24;   // attn_output in-blocks of 256
static constexpr int STRIP_N_GDN_HEADS  = 48;   // ssm_out     in-blocks of 128
static constexpr int STRIP_MAX_LAYERS   = 128;

// tensor kinds, from the weight tensor name of the matmul we intercept
enum strip_kind { STRIP_FFN = 0, STRIP_ATTN = 1, STRIP_GDN = 2, STRIP_KIND_N = 3 };

static int strip_gsize(int kind) {
    return kind == STRIP_GDN ? 128 : 256;
}

static int strip_ngroups(int kind) {
    return kind == STRIP_FFN ? STRIP_N_FFN_GROUPS : (kind == STRIP_ATTN ? STRIP_N_ATTN_HEADS : STRIP_N_GDN_HEADS);
}

// ---------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------

// Static mask: zero group g of every token column whose bit is 0. grid = (n_tok, n_groups).
template <int GSIZE>
static __global__ void strip_zero_group_kernel(float * __restrict__ x, int n_groups,
                                               unsigned long long w0, unsigned long long w1) {
    const int g = blockIdx.y;
    const bool keep = g < 64 ? ((w0 >> g) & 1ULL) != 0 : ((w1 >> (g - 64)) & 1ULL) != 0;
    if (keep) {
        return;
    }
    float * col = x + (int64_t) blockIdx.x * n_groups * GSIZE + (int64_t) g * GSIZE;
    for (int i = threadIdx.x; i < GSIZE; i += blockDim.x) {
        col[i] = 0.0f;
    }
}

// Oracle: one block per token column. Per-group L2 norm of the activation block, keep the top-k groups
// under the deterministic order (norm desc, group index asc), zero the rest. When stats != nullptr the
// norms are also accumulated for calibration.
template <int GSIZE>
static __global__ void strip_oracle_kernel(float * __restrict__ x, int n_groups, int keep_k,
                                           float * __restrict__ stats, int stats_off) {
    extern __shared__ float sh[];      // [n_groups] norms, then [n_groups] keep flags
    float * norms = sh;
    float * keeps = sh + n_groups;

    float * col = x + (int64_t) blockIdx.x * n_groups * GSIZE;

    const int wid  = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarp = (int) (blockDim.x >> 5);

    for (int g = wid; g < n_groups; g += nwarp) {
        float s = 0.0f;
        for (int i = lane; i < GSIZE; i += 32) {
            const float v = col[(int64_t) g * GSIZE + i];
            s += v * v;
        }
        s += __shfl_down_sync(0xffffffffu, s, 16);
        s += __shfl_down_sync(0xffffffffu, s, 8);
        s += __shfl_down_sync(0xffffffffu, s, 4);
        s += __shfl_down_sync(0xffffffffu, s, 2);
        s += __shfl_down_sync(0xffffffffu, s, 1);
        if (lane == 0) {
            norms[g] = sqrtf(s);
        }
    }
    __syncthreads();

    if (stats != nullptr) {
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            atomicAdd(&stats[stats_off + g], norms[g]);
        }
        if (threadIdx.x == 0) {
            atomicAdd(&stats[STRIP_MAX_LAYERS * STRIP_N_FFN_GROUPS + stats_off / STRIP_N_FFN_GROUPS], 1.0f);
        }
    }

    // rank: number of groups strictly better than g (ties broken by lower index)
    for (int g = wid; g < n_groups; g += nwarp) {
        if (lane == 0) {
            const float mine = norms[g];
            int rank = 0;
            for (int j = 0; j < n_groups; ++j) {
                const float o = norms[j];
                if (o > mine || (o == mine && j < g)) {
                    ++rank;
                }
            }
            keeps[g] = rank < keep_k ? 1.0f : 0.0f;
        }
    }
    __syncthreads();

    for (int g = 0; g < n_groups; ++g) {
        if (keeps[g] != 0.0f) {
            continue;
        }
        float * blk = col + (int64_t) g * GSIZE;
        for (int i = threadIdx.x; i < GSIZE; i += blockDim.x) {
            blk[i] = 0.0f;
        }
    }
}

// Vocabulary subset on the logits: logits is [n_vocab, n_tok], one row per token.
static __global__ void strip_vocab_kernel(float * __restrict__ logits, int n_vocab,
                                          const unsigned char * __restrict__ keep) {
    const int64_t base = (int64_t) blockIdx.y * n_vocab;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n_vocab; v += gridDim.x * blockDim.x) {
        if (keep[v] == 0) {
            logits[base + v] = -1.0e30f;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// configuration (parsed once, on first use)
// ---------------------------------------------------------------------------------------------

struct strip_mask_ent {
    std::array<unsigned long long, 2> w = { 0, 0 };
};

struct strip_cfg {
    bool enabled = false;

    bool      ffn_oracle = false;
    int       ffn_keep_default = -1;                        // group count, -1 = unused
    std::unordered_map<int, int> ffn_keep_layer;            // layer -> group count

    std::unordered_map<int, strip_mask_ent> masks[STRIP_KIND_N];   // layer -> bitmask

    std::vector<unsigned char> vocab_keep;                  // empty = off
    unsigned char * vocab_keep_dev = nullptr;

    bool        stats = false;
    std::string stats_path;
    float *     stats_dev = nullptr;
    float *     stats_host = nullptr;
    int         stats_calls = 0;
};

static strip_cfg g_cfg;
static bool     g_cfg_parsed = false;

static void strip_stats_flush(cudaStream_t stream) {
    if (g_cfg.stats_dev == nullptr || g_cfg.stats_host == nullptr) {
        return;
    }
    // never inside a capture
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
        return;
    }
    if (cudaMemcpy(g_cfg.stats_host, g_cfg.stats_dev,
                   sizeof(float) * (STRIP_MAX_LAYERS * STRIP_N_FFN_GROUPS + STRIP_MAX_LAYERS),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void) cudaGetLastError();
        return;
    }
    FILE * f = fopen(g_cfg.stats_path.c_str(), "w");
    if (f == nullptr) {
        return;
    }
    for (int l = 0; l < STRIP_MAX_LAYERS; ++l) {
        const float cnt = g_cfg.stats_host[STRIP_MAX_LAYERS * STRIP_N_FFN_GROUPS + l];
        if (cnt <= 0.0f) {
            continue;
        }
        fprintf(f, "%d", l);
        for (int g = 0; g < STRIP_N_FFN_GROUPS; ++g) {
            fprintf(f, " %.6g", g_cfg.stats_host[l * STRIP_N_FFN_GROUPS + g] / cnt);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// "blk.<N>.<suffix>" -> kind + layer
static bool strip_parse_name(const char * name, int * kind, int * layer) {
    if (name == nullptr || strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    char * end = nullptr;
    const long il = strtol(name + 4, &end, 10);
    if (end == name + 4 || il < 0 || il >= STRIP_MAX_LAYERS) {
        return false;
    }
    if (strcmp(end, ".ffn_down.weight") == 0) {
        *kind = STRIP_FFN;
    } else if (strcmp(end, ".attn_output.weight") == 0) {
        *kind = STRIP_ATTN;
    } else if (strcmp(end, ".ssm_out.weight") == 0) {
        *kind = STRIP_GDN;
    } else {
        return false;
    }
    *layer = (int) il;
    return true;
}

static void strip_set_bit(strip_mask_ent & e, int g) {
    if (g < 64) {
        e.w[0] |= 1ULL << g;
    } else {
        e.w[1] |= 1ULL << (g - 64);
    }
}

static void strip_parse_mask_file(const char * path) {
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        fprintf(stderr, "STRIP: cannot open mask file '%s'\n", path);
        fflush(stderr);
        return;
    }
    char line[8192];
    int n_tensors = 0, n_groups_total = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        char * p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '#' || *p == '\n' || *p == '\0') {
            continue;
        }
        char name[256] = {0};
        if (sscanf(p, "%255s", name) != 1) {
            continue;
        }
        int kind = 0, layer = 0;
        if (!strip_parse_name(name, &kind, &layer)) {
            fprintf(stderr, "STRIP: mask file line ignored (unknown tensor '%s')\n", name);
            fflush(stderr);
            continue;
        }
        char * q = strchr(p, ' ');
        if (q == nullptr) {
            continue;
        }
        strip_mask_ent & e = g_cfg.masks[kind][layer];
        const int ng = strip_ngroups(kind);
        int g = -1;
        for (++q; ; ++q) {
            if (*q >= '0' && *q <= '9') {
                if (g < 0) g = 0;
                g = g * 10 + (*q - '0');
            } else {
                if (g >= 0) {
                    if (g < ng) {
                        strip_set_bit(e, g);
                        ++n_groups_total;
                    }
                    g = -1;
                }
                if (*q == '\0' || *q == '\n') {
                    break;
                }
            }
        }
        ++n_tensors;
    }
    fclose(f);
    fprintf(stderr, "STRIP: mask file '%s': %d tensors, %d group bits set\n", path, n_tensors, n_groups_total);
    fflush(stderr);
}

// parse "N" or "N,M,K" into the vector
static void strip_parse_list(const char * s, std::vector<int> & out) {
    while (s != nullptr && *s != '\0') {
        while (*s != '\0' && (*s < '0' || *s > '9')) ++s;
        if (*s == '\0') break;
        out.push_back((int) strtol(s, (char **) &s, 10));
    }
}

static void strip_parse_vocab_file(const char * path, int n_vocab) {
    FILE * f = fopen(path, "r");
    if (f == nullptr) {
        fprintf(stderr, "STRIP: cannot open vocab file '%s'\n", path);
        fflush(stderr);
        return;
    }
    std::vector<int> ids;
    char line[65536];
    while (fgets(line, sizeof(line), f) != nullptr) {
        char * p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '#' || *p == '\n' || *p == '\0') {
            continue;
        }
        strip_parse_list(line, ids);
    }
    fclose(f);
    g_cfg.vocab_keep.assign((size_t) n_vocab, 0);
    int n_kept = 0;
    for (int id : ids) {
        if (id >= 0 && id < n_vocab && g_cfg.vocab_keep[(size_t) id] == 0) {
            g_cfg.vocab_keep[(size_t) id] = 1;
            ++n_kept;
        }
    }
    fprintf(stderr, "STRIP: vocab file '%s': keep %d of %d tokens (%.2f%%)\n",
            path, n_kept, n_vocab, 100.0 * n_kept / n_vocab);
    fflush(stderr);
}

// fraction (0..1] -> group count; an integer >= 2 is taken as a group count
static int strip_keep_from_frac(double frac, int n_groups) {
    if (frac >= 2.0) {
        int k = (int) (frac + 0.5);
        return k < 1 ? 1 : (k > n_groups ? n_groups : k);
    }
    int k = (int) (frac * n_groups + 0.5);
    return k < 1 ? 1 : (k > n_groups ? n_groups : k);
}

static void strip_parse_cfg() {
    g_cfg_parsed = true;

    const char * ffn = getenv("LLAMA_STRIP_FFN");
    if (ffn != nullptr && ffn[0] != '\0') {
        const char * colon = strchr(ffn, ':');
        if (colon != nullptr && strncmp(ffn, "oracle:", 7) == 0) {
            g_cfg.ffn_oracle = true;
            g_cfg.ffn_keep_default = strip_keep_from_frac(atof(colon + 1), STRIP_N_FFN_GROUPS);
        } else {
            fprintf(stderr, "STRIP: LLAMA_STRIP_FFN='%s' not understood (want oracle:<frac>)\n", ffn);
            fflush(stderr);
        }
    }

    const char * per_layer = getenv("LLAMA_STRIP_FFN_LAYERS");
    if (per_layer != nullptr && per_layer[0] != '\0') {
        std::string s(per_layer);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            if (comma == std::string::npos) comma = s.size();
            const std::string item = s.substr(pos, comma - pos);
            const size_t c = item.find(':');
            if (c != std::string::npos) {
                const std::string k = item.substr(0, c);
                const int n = strip_keep_from_frac(atof(item.c_str() + c + 1), STRIP_N_FFN_GROUPS);
                if (k == "*") {
                    g_cfg.ffn_keep_default = n;
                } else {
                    g_cfg.ffn_keep_layer[(int) atoi(k.c_str())] = n;
                }
            }
            pos = comma + 1;
        }
    }

    const char * mask = getenv("LLAMA_STRIP_MASK");
    if (mask != nullptr && mask[0] != '\0') {
        strip_parse_mask_file(mask);
    }

    const char * stats = getenv("LLAMA_STRIP_STATS");
    if (stats != nullptr && stats[0] != '\0') {
        g_cfg.stats = true;
        g_cfg.stats_path = stats;
        const size_t n = (size_t) (STRIP_MAX_LAYERS * STRIP_N_FFN_GROUPS + STRIP_MAX_LAYERS);
        if (cudaMalloc(&g_cfg.stats_dev, sizeof(float) * n) == cudaSuccess) {
            CUDA_CHECK(cudaMemset(g_cfg.stats_dev, 0, sizeof(float) * n));
            g_cfg.stats_host = (float *) calloc(n, sizeof(float));
        } else {
            (void) cudaGetLastError();
            g_cfg.stats = false;
        }
    }

    // LLAMA_STRIP_VOCAB is parsed lazily (the vocabulary size comes from the logits tensor)
    g_cfg.enabled = g_cfg.ffn_oracle || g_cfg.stats || !g_cfg.masks[0].empty() || !g_cfg.masks[1].empty() ||
                    !g_cfg.masks[2].empty() || getenv("LLAMA_STRIP_VOCAB") != nullptr;

    if (g_cfg.enabled) {
        fprintf(stderr, "STRIP: enabled: ffn_oracle=%d keep_default=%d keep_layers=%zu masks ffn=%zu attn=%zu gdn=%zu "
                        "vocab=%d stats=%d\n",
                (int) g_cfg.ffn_oracle, g_cfg.ffn_keep_default, g_cfg.ffn_keep_layer.size(),
                g_cfg.masks[0].size(), g_cfg.masks[1].size(), g_cfg.masks[2].size(),
                (int) (getenv("LLAMA_STRIP_VOCAB") != nullptr), (int) g_cfg.stats);
        fflush(stderr);
    }
}

bool ggml_cuda_strip_enabled() {
    if (!g_cfg_parsed) {
        strip_parse_cfg();
    }
    if (!g_cfg.enabled) {
        return false;
    }
    // Runtime off switch, re-read on every call so a harness can toggle the hooks between passes
    // (tests/strip_faithful.cpp calls _putenv_s("LLAMA_STRIP_OFF", ...)). One getenv per node, and only
    // when a hook is configured at all.
    const char * off = getenv("LLAMA_STRIP_OFF");
    return off == nullptr || off[0] == '\0' || off[0] == '0';
}

// The main model's weight buffer, so the draft model's tensors (same names, same shapes: DFlash also
// has blk.N.ffn_down with n_ff 17408) are left alone.
static const void * g_strip_target_buffer = nullptr;

static bool strip_is_target(const ggml_tensor * weight) {
    if (g_strip_target_buffer == nullptr) {
        return true;   // not identified yet: the first model loaded is the target
    }
    return (const void *) weight->buffer == g_strip_target_buffer;
}

void ggml_cuda_strip_pre(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
    if (!ggml_cuda_strip_enabled() || node->op != GGML_OP_MUL_MAT) {
        return;
    }
    ggml_tensor * w = node->src[0];
    ggml_tensor * x = node->src[1];
    if (w == nullptr || x == nullptr) {
        return;
    }

    // remember the target model's weight buffer (blk.63 only exists in the 64-layer target)
    if (g_strip_target_buffer == nullptr && strcmp(w->name, "blk.63.ffn_down.weight") == 0) {
        g_strip_target_buffer = (const void *) w->buffer;
    }

    int kind = 0, layer = 0;
    if (!strip_parse_name(w->name, &kind, &layer) || !strip_is_target(w)) {
        return;
    }
    if (x->type != GGML_TYPE_F32 || x->nb[0] != (int64_t) sizeof(float) ||
        x->nb[1] != x->ne[0] * (int64_t) sizeof(float) ||
        x->nb[2] != x->nb[1] * x->ne[1] || x->nb[3] != x->nb[2] * x->ne[2]) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "STRIP: %s input is not a contiguous f32 tensor, hook skipped\n", w->name);
            fflush(stderr);
        }
        return;
    }
    const int gsize = strip_gsize(kind);
    const int ngroups = strip_ngroups(kind);
    if (x->ne[0] != (int64_t) gsize * ngroups) {
        static bool warned2 = false;
        if (!warned2) {
            warned2 = true;
            fprintf(stderr, "STRIP: %s input ne0=%lld does not match %d groups x %d, hook skipped\n",
                    w->name, (long long) x->ne[0], ngroups, gsize);
            fflush(stderr);
        }
        return;
    }
    const int64_t n_tok = x->ne[1] * x->ne[2] * x->ne[3];
    cudaStream_t stream = ctx.stream();

    // static mask
    auto it = g_cfg.masks[kind].find(layer);
    if (it != g_cfg.masks[kind].end()) {
        const dim3 grid((unsigned) n_tok, (unsigned) ngroups);
        if (gsize == 256) {
            strip_zero_group_kernel<256><<<grid, 256, 0, stream>>>((float *) x->data, ngroups,
                                                                   it->second.w[0], it->second.w[1]);
        } else {
            strip_zero_group_kernel<128><<<grid, 128, 0, stream>>>((float *) x->data, ngroups,
                                                                   it->second.w[0], it->second.w[1]);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // per-token oracle over the FFN groups
    if (kind == STRIP_FFN && (g_cfg.ffn_oracle || g_cfg.stats)) {
        int keep = g_cfg.ffn_keep_default;
        auto itl = g_cfg.ffn_keep_layer.find(layer);
        if (itl != g_cfg.ffn_keep_layer.end()) {
            keep = itl->second;
        }
        if (keep < 1) {
            keep = STRIP_N_FFN_GROUPS;
        }
        float * stats = (g_cfg.stats && layer < STRIP_MAX_LAYERS) ? g_cfg.stats_dev : nullptr;
        const int stats_off = layer * STRIP_N_FFN_GROUPS;
        const size_t smem = sizeof(float) * 2 * STRIP_N_FFN_GROUPS;
        strip_oracle_kernel<256><<<(unsigned) n_tok, 256, smem, stream>>>((float *) x->data, ngroups, keep,
                                                                         stats, stats_off);
        CUDA_CHECK(cudaGetLastError());
        if (stats != nullptr && ++g_cfg.stats_calls >= 64) {
            g_cfg.stats_calls = 0;
            CUDA_CHECK(cudaStreamSynchronize(stream));
            strip_stats_flush(stream);
        }
    }
}

void ggml_cuda_strip_post(ggml_backend_cuda_context & ctx, ggml_tensor * node) {
    if (!ggml_cuda_strip_enabled() || node->op != GGML_OP_MUL_MAT) {
        return;
    }
    ggml_tensor * w = node->src[0];
    if (w == nullptr || strcmp(w->name, "output.weight") != 0) {
        return;
    }
    if (!strip_is_target(w)) {
        return;
    }
    if (node->type != GGML_TYPE_F32 || node->nb[1] != node->ne[0] * (int64_t) sizeof(float)) {
        return;
    }
    const char * vpath = getenv("LLAMA_STRIP_VOCAB");
    if (vpath == nullptr || vpath[0] == '\0') {
        return;
    }
    const int n_vocab = (int) node->ne[0];
    if (g_cfg.vocab_keep_dev == nullptr) {
        strip_parse_vocab_file(vpath, n_vocab);
        if (g_cfg.vocab_keep.empty()) {
            return;
        }
        // detect a stream capture: allocation must happen outside it (the first eval is never captured)
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(ctx.stream(), &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            fprintf(stderr, "STRIP: vocab mask allocation inside a capture, mask not applied\n");
            fflush(stderr);
            return;
        }
        if (cudaMalloc(&g_cfg.vocab_keep_dev, g_cfg.vocab_keep.size()) != cudaSuccess) {
            (void) cudaGetLastError();
            fprintf(stderr, "STRIP: cannot allocate the vocabulary mask (%zu bytes)\n", g_cfg.vocab_keep.size());
            fflush(stderr);
            return;
        }
        CUDA_CHECK(cudaMemcpy(g_cfg.vocab_keep_dev, g_cfg.vocab_keep.data(), g_cfg.vocab_keep.size(),
                              cudaMemcpyHostToDevice));
    }
    if ((int) g_cfg.vocab_keep.size() != n_vocab) {
        return;
    }
    const int64_t n_tok = node->ne[1] * node->ne[2] * node->ne[3];
    const int threads = 256;
    const int blocks = (n_vocab + threads * 8 - 1) / (threads * 8);
    strip_vocab_kernel<<<dim3((unsigned) blocks, (unsigned) n_tok), threads, 0, ctx.stream()>>>(
        (float *) node->data, n_vocab, g_cfg.vocab_keep_dev);
    CUDA_CHECK(cudaGetLastError());
}
