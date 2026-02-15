#include "llama-prefill-stream.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-tb-transport.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <future>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <vector>

#if defined(_POSIX_ADVISORY_INFO) || defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

template <typename F>
class llama_scope_exit {
  public:
    explicit llama_scope_exit(F && fn) : fn(std::forward<F>(fn)) {}
    ~llama_scope_exit() { fn(); }

  private:
    F fn;
};

struct llama_prefill_host_cache_span {
    uint8_t * addr = nullptr;
    size_t    len  = 0;
};

struct llama_prefill_host_cache_layer_plan {
    std::vector<llama_prefill_host_cache_span> spans;
    size_t                                     source_bytes = 0;
    size_t                                     advised_bytes = 0;
};

struct llama_prefill_host_cache_runtime {
    bool                                       enabled = false;
    int                                        window_layers = 0;
    size_t                                     window_bytes = 0;
    int                                        prefetch_distance = 1;
    bool                                       advice_supported = true;
    std::vector<llama_prefill_host_cache_layer_plan> layer_plans;
    std::vector<bool>                          prefetched;
    std::vector<bool>                          evicted;
    int                                        prefetched_layers = 0;
    size_t                                     prefetched_bytes = 0;
    int                                        evicted_layers = 0;
    size_t                                     evicted_bytes = 0;
    int                                        lookups = 0;
    int                                        hits = 0;
    int                                        misses = 0;
    float                                      storage_wait_ms = 0.0f;
};

static size_t llama_prefill_host_cache_page_size() {
#if defined(_POSIX_ADVISORY_INFO) || defined(__APPLE__) || defined(__linux__)
    static size_t page = 0;
    if (page == 0) {
        long v = sysconf(_SC_PAGESIZE);
        page = v > 0 ? (size_t) v : (size_t) 4096;
    }
    return page;
#else
    return (size_t) 4096;
#endif
}

static bool llama_prefill_host_cache_advise_span(const llama_prefill_host_cache_span & span, int advice, std::string * err) {
#if defined(_POSIX_ADVISORY_INFO) || defined(__APPLE__) || defined(__linux__)
    if (!span.addr || span.len == 0) {
        return true;
    }
    const size_t    page = llama_prefill_host_cache_page_size();
    const uintptr_t beg  = (uintptr_t) span.addr;
    const uintptr_t end  = beg + span.len;
    const uintptr_t aligned_beg = beg & ~(uintptr_t) (page - 1);
    const uintptr_t aligned_end = (end + page - 1) & ~(uintptr_t) (page - 1);
    if (aligned_end <= aligned_beg) {
        return true;
    }
    const int rc = posix_madvise((void *) aligned_beg, aligned_end - aligned_beg, advice);
    if (rc != 0) {
        if (err) {
            *err = format("posix_madvise failed rc=%d advice=%d addr=%p len=%zu", rc, advice, (void *) aligned_beg,
                          (size_t) (aligned_end - aligned_beg));
        }
        return false;
    }
    return true;
#else
    GGML_UNUSED(span);
    GGML_UNUSED(advice);
    if (err) {
        *err = "posix_madvise unavailable on this platform";
    }
    return false;
#endif
}

static void llama_prefill_build_host_cache_plans(const std::vector<layer_tensor_info> & layers,
                                                 llama_prefill_host_cache_runtime &      cache) {
    cache.layer_plans.clear();
    cache.layer_plans.resize(layers.size());

    for (size_t il = 0; il < layers.size(); ++il) {
        auto & plan = cache.layer_plans[il];
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        ranges.reserve(layers[il].entries.size());
        for (const auto & entry : layers[il].entries) {
            if (!entry.orig_data || entry.nbytes == 0) {
                continue;
            }
            const uintptr_t beg = (uintptr_t) entry.orig_data;
            const uintptr_t end = beg + entry.nbytes;
            if (end > beg) {
                ranges.emplace_back(beg, end);
                plan.source_bytes += entry.nbytes;
            }
        }
        if (ranges.empty()) {
            continue;
        }

        std::sort(ranges.begin(), ranges.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
        std::vector<std::pair<uintptr_t, uintptr_t>> merged;
        merged.reserve(ranges.size());
        for (const auto & r : ranges) {
            if (merged.empty() || r.first > merged.back().second) {
                merged.push_back(r);
                continue;
            }
            merged.back().second = std::max(merged.back().second, r.second);
        }

        plan.spans.reserve(merged.size());
        for (const auto & r : merged) {
            if (r.second <= r.first) {
                continue;
            }
            const size_t len = r.second - r.first;
            plan.spans.push_back({ (uint8_t *) r.first, len });
            plan.advised_bytes += len;
        }
    }
}

static int llama_prefill_host_cache_window_end(const llama_prefill_host_cache_runtime & cache, int anchor_layer, int n_layer) {
    if (anchor_layer >= n_layer) {
        return n_layer - 1;
    }

    int end = n_layer - 1;
    if (cache.window_layers > 0) {
        end = std::min(end, anchor_layer + cache.window_layers - 1);
    }

    if (cache.window_bytes > 0 && anchor_layer < n_layer) {
        size_t budget = 0;
        int    budget_end = anchor_layer - 1;
        for (int il = anchor_layer; il <= end && il < n_layer; ++il) {
            const size_t layer_bytes = il < (int) cache.layer_plans.size() ? cache.layer_plans[il].advised_bytes : 0;
            if (budget > 0 && budget + layer_bytes > cache.window_bytes) {
                break;
            }
            budget += layer_bytes;
            budget_end = il;
            if (budget >= cache.window_bytes) {
                break;
            }
        }
        if (budget_end >= anchor_layer) {
            end = std::min(end, budget_end);
        } else {
            end = anchor_layer;
        }
    }

    return end;
}

static void llama_prefill_host_cache_prefetch_layer(llama_prefill_host_cache_runtime & cache, int il) {
    if (il < 0 || il >= (int) cache.layer_plans.size()) {
        return;
    }
    if (cache.prefetched[il] && !cache.evicted[il]) {
        return;
    }

    auto & plan = cache.layer_plans[il];
    for (const auto & span : plan.spans) {
        std::string err;
#if defined(POSIX_MADV_WILLNEED)
        const bool ok = llama_prefill_host_cache_advise_span(span, POSIX_MADV_WILLNEED, &err);
#else
        const bool ok = llama_prefill_host_cache_advise_span(span, MADV_WILLNEED, &err);
#endif
        if (!ok) {
            cache.advice_supported = false;
            LLAMA_LOG_WARN("%s: S3 host cache prefetch disabled after advise failure: %s\n", __func__,
                           err.empty() ? "<unknown>" : err.c_str());
            return;
        }
    }

    cache.prefetched[il] = true;
    cache.evicted[il]    = false;
    cache.prefetched_layers += 1;
    cache.prefetched_bytes += plan.advised_bytes;
}

static void llama_prefill_host_cache_evict_layer(llama_prefill_host_cache_runtime & cache, int il) {
    if (il < 0 || il >= (int) cache.layer_plans.size()) {
        return;
    }
    if (cache.evicted[il] || !cache.prefetched[il]) {
        return;
    }

    auto & plan = cache.layer_plans[il];
    for (const auto & span : plan.spans) {
        std::string err;
#if defined(POSIX_MADV_DONTNEED)
        const bool ok = llama_prefill_host_cache_advise_span(span, POSIX_MADV_DONTNEED, &err);
#else
        const bool ok = llama_prefill_host_cache_advise_span(span, MADV_DONTNEED, &err);
#endif
        if (!ok) {
            cache.advice_supported = false;
            LLAMA_LOG_WARN("%s: S3 host cache eviction disabled after advise failure: %s\n", __func__,
                           err.empty() ? "<unknown>" : err.c_str());
            return;
        }
    }

    cache.evicted[il] = true;
    cache.evicted_layers += 1;
    cache.evicted_bytes += plan.advised_bytes;
}

//
// Streaming Prefill — Phase 2.1: Pointer-Rerouted 2-Buffer GPU Rotation
//
// Architecture:
//   1. Model loaded with -ngl 0 → all weights in CPU heap RAM
//   2. Allocate 2 GPU buffers sized to hold the largest layer
//   3. Temporarily remap all weight tensor->buffer/data to GPU buffer
//      → ggml_backend_sched sees GPU-backed tensors → routes compute to Metal
//   4. Restore original CPU pointers after graph allocation
//   5. Per-layer callbacks memcpy weights from CPU heap → active GPU buffer
//   6. Ping-pong between GPU buffers A and B across layers
//
// Memory: model_heap(~70GB) + 2*max_layer(~2.2GB) + KV+sys(~15GB) ≈ 87GB
// No Triple-Allocation Trap. No SIGKILL on 128GB machines.
//

//
// Collect all non-null tensor pointers from a llama_layer
//
static void collect_tensors_from_layer(const llama_layer & layer, std::vector<tensor_remap_entry> & out) {
// Macro to avoid repeating the null check + push_back pattern 100+ times
#define PUSH_IF(field)                            \
    if (layer.field) {                            \
        tensor_remap_entry e;                     \
        e.tensor      = layer.field;              \
        e.orig_buffer = layer.field->buffer;      \
        e.orig_data   = layer.field->data;        \
        e.nbytes      = ggml_nbytes(layer.field); \
        out.push_back(e);                         \
    }

    // Attention normalization
    PUSH_IF(attn_norm)
    PUSH_IF(attn_norm_b)
    PUSH_IF(attn_norm_2)
    PUSH_IF(attn_norm_2_b)
    PUSH_IF(attn_q_norm)
    PUSH_IF(attn_q_norm_b)
    PUSH_IF(attn_k_norm)
    PUSH_IF(attn_k_norm_b)
    PUSH_IF(attn_out_norm)
    PUSH_IF(attn_out_norm_b)
    PUSH_IF(attn_q_a_norm)
    PUSH_IF(attn_kv_a_norm)
    PUSH_IF(attn_sub_norm)
    PUSH_IF(attn_post_norm)
    PUSH_IF(ffn_sub_norm)
    PUSH_IF(attn_norm_cross)
    PUSH_IF(attn_norm_enc)

    // Attention weights
    PUSH_IF(wq)
    PUSH_IF(wk)
    PUSH_IF(wv)
    PUSH_IF(wo)
    PUSH_IF(wqkv)
    PUSH_IF(wq_a)
    PUSH_IF(wq_b)
    PUSH_IF(wkv_a_mqa)
    PUSH_IF(wkv_b)
    PUSH_IF(wk_b)
    PUSH_IF(wv_b)
    PUSH_IF(wq_cross)
    PUSH_IF(wk_cross)
    PUSH_IF(wv_cross)
    PUSH_IF(wo_cross)
    PUSH_IF(wq_enc)
    PUSH_IF(wk_enc)
    PUSH_IF(wv_enc)
    PUSH_IF(wo_enc)
    PUSH_IF(wqkv_gate)

    // Attention biases
    PUSH_IF(bq)
    PUSH_IF(bk)
    PUSH_IF(bv)
    PUSH_IF(bo)
    PUSH_IF(bqkv)

    // Relative position bias
    PUSH_IF(attn_rel_b)
    PUSH_IF(attn_rel_b_enc)
    PUSH_IF(attn_rel_b_cross)

    // FFN normalization
    PUSH_IF(ffn_norm)
    PUSH_IF(ffn_norm_b)
    PUSH_IF(ffn_post_norm)
    PUSH_IF(layer_out_norm)
    PUSH_IF(layer_out_norm_b)
    PUSH_IF(ffn_norm_exps)
    PUSH_IF(ffn_norm_enc)

    // FFN weights
    PUSH_IF(ffn_gate)
    PUSH_IF(ffn_down)
    PUSH_IF(ffn_up)
    PUSH_IF(ffn_gate_enc)
    PUSH_IF(ffn_down_enc)
    PUSH_IF(ffn_up_enc)

    // MoE
    PUSH_IF(ffn_gate_inp)
    PUSH_IF(ffn_gate_exps)
    PUSH_IF(ffn_down_exps)
    PUSH_IF(ffn_up_exps)
    PUSH_IF(ffn_gate_inp_b)
    PUSH_IF(ffn_gate_exps_b)
    PUSH_IF(ffn_down_exps_b)
    PUSH_IF(ffn_up_exps_b)

    // Shared expert
    PUSH_IF(ffn_gate_inp_shexp)
    PUSH_IF(ffn_gate_shexp)
    PUSH_IF(ffn_down_shexp)
    PUSH_IF(ffn_up_shexp)

    // FFN biases
    PUSH_IF(ffn_gate_b)
    PUSH_IF(ffn_down_b)
    PUSH_IF(ffn_up_b)
    PUSH_IF(ffn_act)
    PUSH_IF(ffn_exp_probs_b)

    // SSM / Mamba
    PUSH_IF(ssm_in)
    PUSH_IF(ssm_x)
    PUSH_IF(ssm_dt)
    PUSH_IF(ssm_out)
    PUSH_IF(ssm_conv1d)
    PUSH_IF(ssm_a)
    PUSH_IF(ssm_d)
    PUSH_IF(ssm_conv1d_b)
    PUSH_IF(ssm_dt_b)

    // Rope factors
    PUSH_IF(rope_long)
    PUSH_IF(rope_short)
    PUSH_IF(rope_freqs)

    // Bitnet scales
    PUSH_IF(wq_scale)
    PUSH_IF(wk_scale)
    PUSH_IF(wv_scale)
    PUSH_IF(wo_scale)
    PUSH_IF(ffn_gate_scale)
    PUSH_IF(ffn_up_scale)
    PUSH_IF(ffn_down_scale)

    // OpenAI-MOE
    PUSH_IF(attn_sinks)

#undef PUSH_IF
}

//
// Layer Weight Size Calculation
//
size_t llama_layer_weight_bytes(const llama_model * model, int layer_idx) {
    if (!model || layer_idx < 0 || (size_t) layer_idx >= model->layers.size()) {
        return 0;
    }

    std::vector<tensor_remap_entry> entries;
    collect_tensors_from_layer(model->layers[layer_idx], entries);

    size_t total = 0;
    for (const auto & e : entries) {
        total += e.nbytes;
    }
    return total;
}

//
// Collect all per-layer tensor pointers and compute max layer size
//
void prefill_collect_layer_tensors(const llama_model *              model,
                                   std::vector<layer_tensor_info> & out_layers,
                                   size_t &                         out_max_layer_bytes) {
    const int n_layer = model->hparams.n_layer;
    out_layers.resize(n_layer);
    out_max_layer_bytes = 0;

    for (int il = 0; il < n_layer; ++il) {
        auto & info = out_layers[il];
        info.entries.clear();
        info.total_bytes = 0;

        collect_tensors_from_layer(model->layers[il], info.entries);

        for (const auto & e : info.entries) {
            info.total_bytes += e.nbytes;
        }

        if (info.total_bytes > out_max_layer_bytes) {
            out_max_layer_bytes = info.total_bytes;
        }
    }

    std::unordered_map<ggml_tensor *, int> first_seen_layer;
    first_seen_layer.reserve(n_layer * 128);
    int duplicate_count = 0;
    std::vector<std::string> duplicate_samples;
    duplicate_samples.reserve(8);

    for (int il = 0; il < n_layer; ++il) {
        for (const auto & e : out_layers[il].entries) {
            auto it = first_seen_layer.find(e.tensor);
            if (it == first_seen_layer.end()) {
                first_seen_layer.emplace(e.tensor, il);
                continue;
            }
            duplicate_count++;
            if ((int) duplicate_samples.size() < 8) {
                const char * tname = ggml_get_name(e.tensor);
                duplicate_samples.push_back(
                    format("%s first=L%d now=L%d", tname ? tname : "(unnamed)", it->second, il));
            }
        }
    }

    LLAMA_LOG_INFO("%s: collected %d layers, max layer = %.2f MB, total tensors per layer = %zu..%zu\n", __func__,
                   n_layer, out_max_layer_bytes / (1024.0 * 1024.0),
                   out_layers.empty() ? 0 : out_layers[0].entries.size(),
                   out_layers.empty() ? 0 : out_layers[n_layer - 1].entries.size());
    if (duplicate_count > 0) {
        std::string sample_joined;
        for (size_t i = 0; i < duplicate_samples.size(); ++i) {
            if (!sample_joined.empty()) {
                sample_joined += " | ";
            }
            sample_joined += duplicate_samples[i];
        }
        LLAMA_LOG_WARN("%s: detected %d cross-layer duplicate tensor references; overlap safety may be impacted%s%s\n",
                       __func__, duplicate_count, sample_joined.empty() ? "" : " (samples: ",
                       sample_joined.empty() ? "" : sample_joined.c_str());
        if (!sample_joined.empty()) {
            LLAMA_LOG_WARN("%s: duplicate sample details end\n", __func__);
        }
    }
}

//
// Allocate 2 GPU buffers for the largest layer
//
bool prefill_allocate_gpu_buffers(ggml_backend_t          backend,
                                  size_t                  max_layer_bytes,
                                  ggml_backend_buffer_t * out_buf_a,
                                  ggml_backend_buffer_t * out_buf_b) {
    // Get the GPU buffer type from the backend directly
    // (ik_llama does not have the device abstraction layer)
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (!buft) {
        LLAMA_LOG_ERROR("%s: backend has no default buffer type\n", __func__);
        return false;
    }

    // Add 10% padding for alignment
    size_t alloc_size = max_layer_bytes + max_layer_bytes / 10;

    *out_buf_a = ggml_backend_buft_alloc_buffer(buft, alloc_size);
    if (!*out_buf_a) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU buffer A (%.2f MB)\n", __func__, alloc_size / (1024.0 * 1024.0));
        return false;
    }

    *out_buf_b = ggml_backend_buft_alloc_buffer(buft, alloc_size);
    if (!*out_buf_b) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU buffer B (%.2f MB)\n", __func__, alloc_size / (1024.0 * 1024.0));
        ggml_backend_buffer_free(*out_buf_a);
        *out_buf_a = nullptr;
        return false;
    }

    // Mark the rotating buffers as weights so the scheduler preferentially
    // assigns weight-consuming ops to the backend that owns these buffers.
    ggml_backend_buffer_set_usage(*out_buf_a, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(*out_buf_b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    LLAMA_LOG_INFO("%s: allocated 2 GPU buffers: %.2f MB each (%.2f MB total)\n", __func__,
                   alloc_size / (1024.0 * 1024.0), 2.0 * alloc_size / (1024.0 * 1024.0));

    return true;
}

//
// Temporarily remap all weight tensors to point at a GPU buffer
//
// This is the core "trick": the ggml_backend_sched looks at tensor->buffer
// to decide which backend gets the compute. By setting all weight buffers
// to the GPU buffer, the scheduler will route all matmuls to Metal.
//
// The tensor->data field is set to a valid offset within the GPU buffer
// so that the scheduler can compute buffer sizes. The actual data content
// doesn't matter at this point — real data is uploaded per-layer later.
//
void prefill_remap_tensors_to_gpu(std::vector<layer_tensor_info> & layers, ggml_backend_buffer_t gpu_buf) {
    void * gpu_base = ggml_backend_buffer_get_base(gpu_buf);
    size_t gpu_size = ggml_backend_buffer_get_size(gpu_buf);

    int total_remapped = 0;

    for (auto & info : layers) {
        size_t offset = 0;
        for (auto & entry : info.entries) {
            // Save original pointers (already done in collect, but re-save in case
            // this is called after a previous remap/restore cycle)
            entry.orig_buffer = entry.tensor->buffer;
            entry.orig_data   = entry.tensor->data;

            // Remap to GPU
            entry.tensor->buffer = gpu_buf;

            // Set data to a valid offset within the GPU buffer
            // We wrap around if a single layer exceeds the buffer — this is fine
            // because the scheduler only inspects the buffer type, not the data.
            size_t data_offset = offset % gpu_size;
            entry.tensor->data = (char *) gpu_base + data_offset;

            offset += entry.nbytes;
            total_remapped++;
        }
    }

    LLAMA_LOG_INFO("%s: remapped %d tensors across %zu layers to GPU buffer\n", __func__, total_remapped,
                   layers.size());
}

//
// Restore original CPU buffer/data pointers on all tensors
//
void prefill_restore_tensor_pointers(std::vector<layer_tensor_info> & layers) {
    for (auto & info : layers) {
        for (auto & entry : info.entries) {
            entry.tensor->buffer = entry.orig_buffer;
            entry.tensor->data   = entry.orig_data;
        }
    }
}

//
// Upload one layer's weights from CPU heap into a GPU buffer
//
// This is the hot path during the per-layer forward pass.
// For each tensor in the layer:
//   1. Bind tensor->buffer/data to packed offsets in gpu_buf
//   2. Upload tensor bytes via backend transfer API
//
// Using ggml_backend_tensor_set() keeps this backend-agnostic:
// - Metal (Apple unified memory)
// - CUDA (host->device copy)
// - Vulkan/OpenCL/SYCL backends that implement set_tensor
//
static float prefill_upload_layer(std::vector<tensor_remap_entry> & layer_entries,
                                  ggml_backend_buffer_t             gpu_buf,
                                  size_t                            slab_bytes,
                                  int                               layer_idx,
                                  bool                              trace_swap,
                                  bool                              remote_weight_relay) {
    auto t_start = std::chrono::high_resolution_clock::now();

    void * gpu_base = ggml_backend_buffer_get_base(gpu_buf);
    size_t offset   = 0;
    const size_t chunk_bytes = slab_bytes == 0 ? std::numeric_limits<size_t>::max() : slab_bytes;
    if (trace_swap) {
        LLAMA_LOG_INFO("%s: [SWAP_TRACE] upload begin layer %d buf=%p entries=%zu\n",
                       __func__, layer_idx, (void *) gpu_buf, layer_entries.size());
    }

    for (auto & entry : layer_entries) {
        // Redirect tensor to read from GPU buffer
        entry.tensor->buffer = gpu_buf;
        entry.tensor->data   = (char *) gpu_base + offset;

        // Copy weight data from CPU heap to device buffer. When slab_bytes is set,
        // upload in fixed chunks so staging/upload behavior can be tuned at runtime.
        size_t copied = 0;
        while (copied < entry.nbytes) {
            const size_t n = std::min(chunk_bytes, entry.nbytes - copied);
            const uint8_t *src = (const uint8_t *) entry.orig_data + copied;
            if (remote_weight_relay) {
                std::vector<uint8_t> remote_chunk;
                std::string fetch_error;
                if (llama_tb_transport_fetch_weight_chunk(src, n, remote_chunk, &fetch_error) &&
                    remote_chunk.size() == n) {
                    ggml_backend_tensor_set(entry.tensor, remote_chunk.data(), copied, n);
                } else {
                    static bool warned_remote_fetch_fallback = false;
                    if (!warned_remote_fetch_fallback) {
                        warned_remote_fetch_fallback = true;
                        LLAMA_LOG_WARN("%s: remote weight fetch failed, falling back to local chunk copy: %s\n",
                                       __func__, fetch_error.empty() ? "<unknown>" : fetch_error.c_str());
                    }
                    ggml_backend_tensor_set(entry.tensor, src, copied, n);
                }
            } else {
                ggml_backend_tensor_set(entry.tensor, src, copied, n);
            }
            copied += n;
        }

        offset += entry.nbytes;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    const float upload_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    if (trace_swap) {
        LLAMA_LOG_INFO("%s: [SWAP_TRACE] upload end layer %d buf=%p bytes=%.2fMB time=%.2fms\n",
                       __func__, layer_idx, (void *) gpu_buf, offset / (1024.0 * 1024.0), upload_ms);
    }
    return upload_ms;
}

//
// Streaming Availability Check
//
bool llama_prefill_stream_available(const llama_context * ctx) {
    GGML_UNUSED(ctx);
    return true;
}

//
// Main Streaming Prefill Entry Point — Phase 2.1: Pointer-Rerouted
//
// This function is called from decode_streaming(). It:
//   1. Collects all per-layer tensor pointers
//   2. Allocates 2 GPU buffers for the largest layer
//   3. Remaps all tensors → GPU, builds graph, restores CPU pointers
//   4. Per-layer callbacks upload weights and ping-pong GPU buffers
//
int llama_prefill_stream(llama_context *                     ctx,
                         const llama_batch &                 batch,
                         const llama_prefill_stream_params & params,
                         llama_prefill_stream_result *       result) {
    if (!result) {
        return -1;
    }

    result->layer_metrics.clear();
    result->used_streaming = false;
    result->host_cache_enabled = false;
    result->host_cache_advice_supported = true;
    result->host_prefetch_layers = 0;
    result->host_prefetch_bytes = 0;
    result->host_evict_layers = 0;
    result->host_evict_bytes = 0;
    result->host_cache_lookups = 0;
    result->host_cache_hits = 0;
    result->host_cache_misses = 0;
    result->host_cache_hit_rate = 0.0f;
    result->host_storage_wait_ms = 0.0f;

    LLAMA_LOG_INFO("%s: [DIAG] entering streaming prefill\n", __func__);

    auto t_total_start = std::chrono::high_resolution_clock::now();

    const llama_model * model = llama_get_model(ctx);
    if (!model) {
        LLAMA_LOG_ERROR("%s: [DIAG] llama_get_model returned null!\n", __func__);
        return -1;
    }
    const int n_layer = model->hparams.n_layer;

    LLAMA_LOG_INFO("%s: [DIAG] model has %d layers\n", __func__, n_layer);

    const bool enable_overlap = params.enable_overlap;
    bool remote_weight_relay = params.remote_weight_relay;
    if (remote_weight_relay && !llama_tb_transport_enabled()) {
        if (params.remote_weight_hybrid) {
            LLAMA_LOG_WARN("%s: remote weight relay requested but transport is unavailable; using local weight uploads\n", __func__);
            remote_weight_relay = false;
        } else {
            LLAMA_LOG_WARN("%s: remote weight relay requested without active transport; using loopback relay copy path\n", __func__);
        }
    }
    const char * swap_trace_env = std::getenv("LLAMA_PREFILL_SWAP_TRACE");
    const bool trace_swap = swap_trace_env != nullptr && swap_trace_env[0] != '\0' && std::strcmp(swap_trace_env, "0") != 0;
    const size_t slab_bytes = params.slab_bytes;
    int prefetch_distance = std::max(1, params.prefetch_distance);
    const int host_cache_prefetch_distance = std::max(1, params.host_cache_prefetch_distance);
    llama_prefill_host_cache_runtime host_cache;
    host_cache.window_layers = std::max(0, params.host_cache_window_layers);
    host_cache.window_bytes  = params.host_cache_window_bytes;
    host_cache.prefetch_distance = host_cache_prefetch_distance;
    host_cache.enabled = host_cache.window_layers > 0 || host_cache.window_bytes > 0;
    result->host_cache_enabled = host_cache.enabled;
    if (params.n_buffers != 2) {
        LLAMA_LOG_WARN("%s: only 2-buffer ping-pong is currently implemented (requested %d, using 2)\n", __func__,
                       params.n_buffers);
    }
    if (prefetch_distance != 1) {
        LLAMA_LOG_WARN("%s: prefetch_distance=%d requested, clamping to 1 for 2-buffer ping-pong\n", __func__,
                       prefetch_distance);
        prefetch_distance = 1;
    }

    // ====================================================================
    // Step 1: Collect per-layer tensor info and compute max layer size
    // ====================================================================

    std::vector<layer_tensor_info> layer_infos;
    size_t                         max_layer_bytes = 0;
    prefill_collect_layer_tensors(model, layer_infos, max_layer_bytes);

    LLAMA_LOG_INFO("%s: [DIAG] tensor collection done, max_layer=%.2f MB, layers=%zu\n", __func__,
                   max_layer_bytes / (1024.0 * 1024.0), layer_infos.size());
    if (host_cache.enabled) {
        llama_prefill_build_host_cache_plans(layer_infos, host_cache);
        host_cache.prefetched.assign(n_layer, false);
        host_cache.evicted.assign(n_layer, false);
        LLAMA_LOG_INFO("%s: S3 host moving-window active (layers=%d, bytes=%.2fMB, prefetch_distance=%d)\n",
                       __func__, host_cache.window_layers, host_cache.window_bytes / (1024.0 * 1024.0),
                       host_cache.prefetch_distance);
        const int initial_anchor = std::min(n_layer, std::max(0, host_cache.prefetch_distance - 1));
        if (initial_anchor < n_layer) {
            const int initial_end = llama_prefill_host_cache_window_end(host_cache, initial_anchor, n_layer);
            for (int il = initial_anchor; il <= initial_end; ++il) {
                if (!host_cache.advice_supported) {
                    break;
                }
                llama_prefill_host_cache_prefetch_layer(host_cache, il);
            }
        }
    }

    // ====================================================================
    // Step 2: Allocate 2 GPU buffers sized for the largest layer
    // ====================================================================

    // Get the first GPU backend from the context
    ggml_backend_t gpu_backend = nullptr;
    {
        ggml_backend_sched_t sched = ctx->sched;
        if (!sched) {
            LLAMA_LOG_ERROR("%s: [DIAG] ctx->sched is null!\n", __func__);
            return -1;
        }
        // Try to find a non-CPU backend (Metal/CUDA)
        int n_backends = ggml_backend_sched_get_n_backends(sched);
        LLAMA_LOG_INFO("%s: [DIAG] scheduler has %d backends\n", __func__, n_backends);
        for (int i = 0; i < n_backends; ++i) {
            ggml_backend_t b    = ggml_backend_sched_get_backend(sched, i);
            const char *   name = ggml_backend_name(b);
            LLAMA_LOG_INFO("%s: [DIAG]   backend[%d] = %s\n", __func__, i, name ? name : "(null)");
            if (name && strstr(name, "CPU") == nullptr && strstr(name, "cpu") == nullptr) {
                gpu_backend = b;
                break;
            }
        }
        if (!gpu_backend && n_backends > 0) {
            // Fallback: use backend 0 even if it's CPU (will fail on buffer alloc)
            gpu_backend = ggml_backend_sched_get_backend(sched, 0);
            LLAMA_LOG_WARN("%s: [DIAG] no non-CPU backend found, using backend[0]\n", __func__);
        }
    }

    if (!gpu_backend) {
        LLAMA_LOG_ERROR("%s: no GPU backend available\n", __func__);
        return -1;
    }
    LLAMA_LOG_INFO("%s: [DIAG] using backend: %s\n", __func__, ggml_backend_name(gpu_backend));

    ggml_backend_buffer_t gpu_buf_a = nullptr;
    ggml_backend_buffer_t gpu_buf_b = nullptr;

    if (!prefill_allocate_gpu_buffers(gpu_backend, max_layer_bytes, &gpu_buf_a, &gpu_buf_b)) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU buffers\n", __func__);
        return -1;
    }

    LLAMA_LOG_INFO("%s: [DIAG] GPU buffers allocated successfully\n", __func__);

    // ====================================================================
    // Step 3: Remap → Build Graph → Restore
    //
    // We temporarily make all weight tensors look GPU-resident so the
    // scheduler routes compute to Metal. The callbacks then keep each layer's
    // pointers redirected to the active ping-pong buffer.
    // ====================================================================

    prefill_remap_tensors_to_gpu(layer_infos, gpu_buf_a);
    LLAMA_LOG_INFO("%s: [DIAG] tensor remap complete\n", __func__);

    std::vector<float> upload_times(n_layer, 0.0f);
    std::vector<float> stall_times(n_layer, 0.0f);
    std::vector<float> compute_times(n_layer, 0.0f);

    struct pending_upload_t {
        int               layer_idx = -1;
        bool              active    = false;
        std::future<float> fut;
    };

    pending_upload_t pending_upload;
    std::vector<bool> layer_loaded(n_layer, false);
    auto elapsed_ms = [&]() -> float {
        return std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t_total_start).count();
    };

    bool remap_applied       = true;
    bool callbacks_registered = false;
    const bool saved_streaming_flag = ctx->prefill_streaming;
    const char * force_single_env = std::getenv("LLAMA_PREFILL_FORCE_SINGLE_BUFFER");
    const bool force_single_buffer =
        force_single_env != nullptr && force_single_env[0] != '\0' && std::strcmp(force_single_env, "0") != 0;

    const auto cleanup = llama_scope_exit([&]() {
        if (callbacks_registered) {
            ctx->pre_layer_cb = nullptr;
            ctx->post_layer_cb = nullptr;
        }
        if (pending_upload.active) {
            pending_upload.fut.wait();
            pending_upload.active = false;
        }
        if (remap_applied) {
            prefill_restore_tensor_pointers(layer_infos);
        }
        if (gpu_buf_a) {
            ggml_backend_buffer_free(gpu_buf_a);
        }
        if (gpu_buf_b) {
            ggml_backend_buffer_free(gpu_buf_b);
        }
        ctx->prefill_streaming = saved_streaming_flag;
    });

    auto get_buf_for_layer = [&](int il) -> ggml_backend_buffer_t {
        if (force_single_buffer) {
            return gpu_buf_a;
        }
        return (il % 2 == 0) ? gpu_buf_a : gpu_buf_b;
    };

    if (force_single_buffer) {
        LLAMA_LOG_WARN("%s: LLAMA_PREFILL_FORCE_SINGLE_BUFFER enabled; all layers will upload into buffer A\n", __func__);
    }

    auto upload_sync = [&](int il) {
        if (trace_swap) {
            LLAMA_LOG_INFO("%s: [SWAP_TRACE] sync upload layer %d start @ %.2fms\n", __func__, il, elapsed_ms());
        }
        upload_times[il] = prefill_upload_layer(layer_infos[il].entries, get_buf_for_layer(il), slab_bytes, il, trace_swap,
                                                remote_weight_relay);
        layer_loaded[il] = true;
        if (trace_swap) {
            LLAMA_LOG_INFO("%s: [SWAP_TRACE] sync upload layer %d done @ %.2fms\n", __func__, il, elapsed_ms());
        }
    };

    auto schedule_async_upload = [&](int il) {
        if (il < 0 || il >= n_layer || layer_loaded[il]) {
            return;
        }
        if (trace_swap) {
            LLAMA_LOG_INFO("%s: [SWAP_TRACE] async schedule layer %d @ %.2fms\n", __func__, il, elapsed_ms());
        }
        pending_upload.layer_idx = il;
        pending_upload.active    = true;
        pending_upload.fut       = std::async(
            std::launch::async, [&layer_infos, &get_buf_for_layer, il, slab_bytes, trace_swap, remote_weight_relay]() {
            return prefill_upload_layer(layer_infos[il].entries, get_buf_for_layer(il), slab_bytes, il, trace_swap,
                                        remote_weight_relay);
        });
    };

    auto schedule_next_for_layer = [&](int il, const char * where) {
        const int next_il = il + prefetch_distance;
        if (!enable_overlap || next_il >= n_layer || layer_loaded[next_il]) {
            return;
        }
        if (pending_upload.active) {
            if (trace_swap) {
                LLAMA_LOG_INFO("%s: [SWAP_TRACE] async skip next=%d at %s (pending layer %d) @ %.2fms\n",
                               __func__, next_il, where ? where : "unknown", pending_upload.layer_idx, elapsed_ms());
            }
            return;
        }
        schedule_async_upload(next_il);
    };

    auto slide_host_cache_window = [&](int il) {
        if (!host_cache.enabled || !host_cache.advice_supported) {
            return;
        }
        const int anchor = il + host_cache.prefetch_distance;
        if (anchor < n_layer) {
            const int end = llama_prefill_host_cache_window_end(host_cache, anchor, n_layer);
            for (int want = anchor; want <= end; ++want) {
                if (!host_cache.advice_supported) {
                    break;
                }
                llama_prefill_host_cache_prefetch_layer(host_cache, want);
            }
        }
        const int evict_upto = std::min(n_layer, std::max(0, anchor));
        for (int old = 0; old < evict_upto; ++old) {
            if (!host_cache.advice_supported) {
                break;
            }
            llama_prefill_host_cache_evict_layer(host_cache, old);
        }
    };

    auto pre_layer_cb = [&](int il, int /*n_layer_total*/) {
        if (host_cache.enabled) {
            const bool prefetched = il >= 0 && il < (int) host_cache.prefetched.size() ? host_cache.prefetched[il] : false;
            const bool evicted = il >= 0 && il < (int) host_cache.evicted.size() ? host_cache.evicted[il] : false;
            const bool resident = prefetched && !evicted;
            host_cache.lookups += 1;
            if (resident) {
                host_cache.hits += 1;
            } else {
                host_cache.misses += 1;
            }
        }

        if (enable_overlap && pending_upload.active && pending_upload.layer_idx == il) {
            auto t_wait_start = std::chrono::high_resolution_clock::now();
            if (trace_swap) {
                LLAMA_LOG_INFO("%s: [SWAP_TRACE] pre-layer wait begin layer %d @ %.2fms\n", __func__, il, elapsed_ms());
            }
            pending_upload.fut.wait();
            auto t_wait_end = std::chrono::high_resolution_clock::now();
            stall_times[il] = std::chrono::duration<float, std::milli>(t_wait_end - t_wait_start).count();
            upload_times[il]  = pending_upload.fut.get();
            pending_upload.active = false;
            layer_loaded[il]      = true;
            host_cache.storage_wait_ms += stall_times[il];
            if (trace_swap) {
                LLAMA_LOG_INFO("%s: [SWAP_TRACE] pre-layer wait end layer %d stall=%.2fms @ %.2fms\n",
                               __func__, il, stall_times[il], elapsed_ms());
            }
            // Launch upload of the following layer while this layer computes.
            schedule_next_for_layer(il, "pre_layer_cb(wait-path)");
            slide_host_cache_window(il);
            return;
        }

        upload_sync(il);
        stall_times[il] = 0.0f;
        // Launch upload of the following layer while this layer computes.
        schedule_next_for_layer(il, "pre_layer_cb(sync-path)");
        slide_host_cache_window(il);
    };

    auto post_layer_cb = [&](int il, int /*n_layer_total*/) {
        // Intentionally keep post-layer callback side-effect free for overlap.
        // Scheduling the next upload in pre_layer_cb gives the async worker
        // the full compute window of the current layer.
        GGML_UNUSED(il);
    };

    // Register callbacks in the context
    ctx->pre_layer_cb = pre_layer_cb;
    ctx->post_layer_cb = post_layer_cb;
    callbacks_registered = true;
    LLAMA_LOG_INFO("%s: [DIAG] callbacks registered, calling llama_decode...\n", __func__);

    auto t_compute_start = std::chrono::high_resolution_clock::now();

    // Temporarily disable recursive streaming dispatch while we run the
    // orchestrated decode path with per-layer callbacks.
    ctx->prefill_streaming = false;
    llama_batch batch_copy = batch;  // llama_decode takes by value
    LLAMA_LOG_INFO("%s: [DIAG] about to call llama_decode with n_tokens=%d (overlap=%s)\n", __func__,
                   batch_copy.n_tokens, enable_overlap ? "on" : "off");
    int ret = llama_decode(ctx, batch_copy);
    LLAMA_LOG_INFO("%s: [DIAG] llama_decode returned %d\n", __func__, ret);

    auto  t_compute_end = std::chrono::high_resolution_clock::now();
    float compute_ms    = std::chrono::duration<float, std::milli>(t_compute_end - t_compute_start).count();

    if (pending_upload.active) {
        const int il = pending_upload.layer_idx;
        if (il >= 0 && il < n_layer) {
            upload_times[il] = pending_upload.fut.get();
            layer_loaded[il] = true;
        } else {
            pending_upload.fut.wait();
        }
        pending_upload.active = false;
    }

    if (ret < 0) {
        LLAMA_LOG_WARN("%s: streaming path failed (ret=%d), falling back to standard decode\n", __func__, ret);

        // Disable callbacks and restore tensor pointers before fallback decode.
        ctx->pre_layer_cb = nullptr;
        ctx->post_layer_cb = nullptr;
        callbacks_registered = false;
        prefill_restore_tensor_pointers(layer_infos);
        remap_applied = false;
        ctx->prefill_streaming = false;

        llama_batch fallback_batch = batch;
        ret = llama_decode(ctx, fallback_batch);

        result->layer_metrics.clear();
        result->used_streaming = false;
        result->n_tokens_processed = batch.n_tokens;
        result->total_time_ms =
            std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
        result->total_tok_s = (result->total_time_ms > 0) ? batch.n_tokens / (result->total_time_ms / 1000.0f) : 0.0f;
        return ret;
    }

    compute_times = ctx->last_layer_compute_times_ms;
    if ((int) compute_times.size() != n_layer) {
        compute_times.assign(n_layer, 0.0f);
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();

    // ====================================================================
    // Step 5: Build metrics and report
    // ====================================================================

    size_t total_bytes     = 0;
    float  total_upload_ms = 0;
    for (int il = 0; il < n_layer; ++il) {
        llama_layer_metrics m;
        m.layer_idx       = il;
        m.weight_bytes    = layer_infos[il].total_bytes;
        m.compute_time_ms = compute_times[il];
        m.load_time_ms    = upload_times[il];
        m.stall_time_ms   = stall_times[il];
        result->layer_metrics.push_back(m);
        total_bytes += m.weight_bytes;
        total_upload_ms += upload_times[il];
    }

    result->total_time_ms      = std::chrono::duration<float, std::milli>(t_total_end - t_total_start).count();
    result->n_tokens_processed = batch.n_tokens;
    result->total_tok_s    = (result->total_time_ms > 0) ? batch.n_tokens / (result->total_time_ms / 1000.0f) : 0.0f;
    result->used_streaming = true;
    result->host_cache_enabled = host_cache.enabled;
    result->host_cache_advice_supported = host_cache.advice_supported;
    result->host_prefetch_layers = host_cache.prefetched_layers;
    result->host_prefetch_bytes  = host_cache.prefetched_bytes;
    result->host_evict_layers = host_cache.evicted_layers;
    result->host_evict_bytes  = host_cache.evicted_bytes;
    result->host_cache_lookups = host_cache.lookups;
    result->host_cache_hits = host_cache.hits;
    result->host_cache_misses = host_cache.misses;
    result->host_cache_hit_rate = host_cache.lookups > 0
        ? (100.0f * ((float) host_cache.hits / (float) host_cache.lookups))
        : 0.0f;
    result->host_storage_wait_ms = host_cache.storage_wait_ms;

    // Summary
    float upload_bw = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_upload_ms / 1000.0);

    LLAMA_LOG_INFO("%s: === Phase 2.1 Pointer-Rerouted Summary ===\n", __func__);
    LLAMA_LOG_INFO("%s: layers: %d, weight data: %.2f GB\n", __func__, n_layer,
                   total_bytes / (1024.0 * 1024.0 * 1024.0));
    LLAMA_LOG_INFO("%s: total upload: %.2f ms (%.2f GB/s effective)\n", __func__, total_upload_ms, upload_bw);
    LLAMA_LOG_INFO("%s: forward pass: %.2f ms\n", __func__, compute_ms);
    LLAMA_LOG_INFO("%s: total: %d tokens, %.2f ms, %.2f tok/s\n", __func__, result->n_tokens_processed,
                   result->total_time_ms, result->total_tok_s);
    if (host_cache.enabled) {
        LLAMA_LOG_INFO("%s: S3 host moving-window activity: prefetched=%d layers (%.2f MB), evicted=%d layers (%.2f MB), lookups=%d hits=%d misses=%d hit_rate=%.1f%% storage_wait_ms=%.2f advise_supported=%s\n",
                       __func__, host_cache.prefetched_layers, host_cache.prefetched_bytes / (1024.0 * 1024.0),
                       host_cache.evicted_layers, host_cache.evicted_bytes / (1024.0 * 1024.0),
                       host_cache.lookups, host_cache.hits, host_cache.misses,
                       host_cache.lookups > 0 ? (100.0f * ((float) host_cache.hits / (float) host_cache.lookups)) : 0.0f,
                       host_cache.storage_wait_ms,
                       host_cache.advice_supported ? "true" : "false");
    }

    if (params.telemetry) {
        LLAMA_LOG_INFO("%s: per-layer upload times:\n", __func__);
        for (int il = 0; il < n_layer; ++il) {
            LLAMA_LOG_INFO("  layer %3d: upload=%.2fms bytes=%.2fMB\n", il, upload_times[il],
                           layer_infos[il].total_bytes / (1024.0 * 1024.0));
        }
    }

    return ret;
}
