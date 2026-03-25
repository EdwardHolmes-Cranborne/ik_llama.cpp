#pragma once

// Weight streaming engine for layer-major prefill.
//
// Enables -ngl 0 with GPU compute by streaming weights from CPU/mmap
// to GPU layer-by-layer through ping-pong staging buffers.
//
// Architecture:
//   1. Model loaded with -ngl 0: all weights in CPU heap/mmap
//   2. Allocate 2 GPU buffers sized to hold the largest layer
//   3. Temporarily remap all weight tensor->buffer/data to GPU buffer
//      so ggml_backend_sched routes compute to CUDA
//   4. Restore original CPU pointers after graph allocation
//   5. Per-layer callbacks memcpy weights from CPU -> active GPU buffer
//   6. Ping-pong between GPU buffers A and B across layers
//
// DirectStorage integration: when DS is available (Windows + NVMe),
// weights are read directly from NVMe to GPU via D3D12, bypassing CPU.
// This is a future extension; the initial implementation uses CPU->GPU memcpy.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Forward declarations
struct llama_model;
struct llama_context;
struct llama_layer;

// Callback type for per-layer notifications
using layer_callback_fn = std::function<void(int il, int n_layer)>;

// ============================================================================
// Tensor remap entry — tracks original and remapped tensor state
// ============================================================================

struct wf_tensor_remap_entry {
    struct ggml_tensor *       tensor      = nullptr;
    ggml_backend_buffer_t      orig_buffer = nullptr;
    void *                     orig_data   = nullptr;
    size_t                     nbytes      = 0;
};

// Per-layer tensor collection
struct wf_layer_tensor_info {
    std::vector<wf_tensor_remap_entry> entries;
    size_t total_bytes = 0;
};

// ============================================================================
// Weight streaming context
// ============================================================================

struct llama_weight_stream_config {
    int     n_buffers       = 2;     // ping-pong GPU buffers
    bool    trace           = false; // verbose logging
    bool    use_dstorage    = false; // future: DirectStorage path
};

struct llama_weight_stream_stats {
    int     layers_staged      = 0;
    double  total_upload_ms    = 0.0;
    double  total_stall_ms     = 0.0;
    double  total_compute_ms   = 0.0;
    size_t  bytes_uploaded     = 0;
};

struct llama_weight_stream {
    llama_weight_stream_config config;
    llama_weight_stream_stats  stats;

    // Per-layer tensor info (populated at init)
    std::vector<wf_layer_tensor_info> layer_infos;
    size_t max_layer_bytes = 0;

    // GPU staging buffers (ping-pong)
    ggml_backend_buffer_t gpu_buf[2] = {nullptr, nullptr};
    void *                gpu_ptr[2] = {nullptr, nullptr};
    size_t                gpu_buf_size = 0;
    int                   active_buf = 0;

    // The GPU backend that owns the staging buffers
    ggml_backend_t        gpu_backend = nullptr;

    // ---- Lifecycle ----

    // Collect all per-layer tensors from the model and compute sizes.
    void collect_layer_tensors(const llama_model * model);

    // Allocate GPU staging buffers on the given backend.
    bool allocate_gpu_buffers(ggml_backend_t backend);

    // Temporarily remap ALL layer tensors to point at the GPU buffer.
    // This makes ggml_backend_sched route compute to GPU.
    // Call this BEFORE ggml_backend_sched_alloc_graph().
    void remap_tensors_to_gpu();

    // Restore ALL layer tensors to their original CPU pointers.
    // Call this AFTER ggml_backend_sched_alloc_graph().
    void restore_tensors_to_cpu();

    // Upload layer `il` weights from CPU -> active GPU buffer.
    // Remaps tensor pointers to the GPU buffer for compute.
    void upload_layer(int il);

    // Restore layer `il` tensors and swap active buffer.
    void finish_layer(int il);

    // Get pre/post layer callbacks for use with layer-major decode.
    layer_callback_fn make_pre_layer_cb();
    layer_callback_fn make_post_layer_cb();

    // Free GPU buffers.
    void free_buffers();

    ~llama_weight_stream() { free_buffers(); }
};

// ============================================================================
// Utility: collect all non-null tensor pointers from an ik_llama layer
// ============================================================================

void wf_collect_tensors_from_layer(const llama_layer & layer,
                                   std::vector<wf_tensor_remap_entry> & out);

// Compute the weight size of a single layer
size_t wf_layer_weight_bytes(const llama_model * model, int layer_idx);
