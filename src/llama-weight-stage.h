#pragma once

#include "llama-context.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

// Weight staging for layer-major streaming prefill.
// Manages double-buffered GPU staging buffers for models that don't fit
// in VRAM. Each layer's weights are uploaded to a GPU buffer just before
// compute, and the buffer is recycled after compute completes.
//
// Compatible with graph split multi-GPU: when split_mode_graph is active,
// each device gets its own pair of staging buffers.

struct llama_weight_stage_config {
    size_t  buffer_size     = 0;    // per-buffer size (0 = auto from max layer)
    int     n_buffers       = 2;    // double-buffer by default
    int     prefetch_distance = 1;  // upload N+prefetch while computing N
};

struct llama_weight_stage_stats {
    int     layers_staged   = 0;
    double  total_upload_ms = 0.0;
    double  total_compute_ms = 0.0;
    double  overlap_ms      = 0.0;  // time compute overlapped with upload
};

// Per-device staging state
struct llama_weight_stage_device {
    ggml_backend_buffer_t bufs[2] = {nullptr, nullptr};
    void *                ptrs[2] = {nullptr, nullptr};
    size_t                buf_size = 0;
    int                   active_buf = 0;  // which buffer is currently in use

    void * get_active()  const { return ptrs[active_buf]; }
    void * get_staging() const { return ptrs[1 - active_buf]; }
    void   swap()              { active_buf = 1 - active_buf; }
};

// Weight staging manager
struct llama_weight_stage {
    llama_weight_stage_config config;
    std::vector<llama_weight_stage_device> devices;
    llama_weight_stage_stats stats;

    // Initialize staging buffers for the given backends.
    // max_layer_bytes: largest layer weight size across all layers.
    bool init(const std::vector<ggml_backend_t> & backends,
              ggml_backend_t backend_cpu,
              size_t max_layer_bytes);

    // Free all staging buffers.
    void free();

    // Get pre/post layer callbacks that manage weight staging.
    // upload_fn(il, device_id, dst_ptr, dst_size) should upload layer il's
    // weights for device_id to the provided GPU buffer.
    using upload_fn_t = std::function<void(int il, int device_id, void * dst, size_t size)>;

    layer_callback_fn make_pre_cb(upload_fn_t upload_fn);
    layer_callback_fn make_post_cb();

    ~llama_weight_stage() { free(); }
};
