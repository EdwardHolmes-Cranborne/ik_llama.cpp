#include "llama-weight-stage.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

bool llama_weight_stage::init(
    const std::vector<ggml_backend_t> & backends,
    ggml_backend_t backend_cpu,
    size_t max_layer_bytes) {

    free();
    stats = {};

    size_t buf_size = config.buffer_size > 0 ? config.buffer_size : max_layer_bytes;
    if (buf_size == 0) return false;

    // Create staging buffers for each non-CPU backend
    for (auto * backend : backends) {
        if (backend == backend_cpu) continue;

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        if (!buft) continue;

        llama_weight_stage_device dev = {};
        dev.buf_size = buf_size;

        for (int i = 0; i < config.n_buffers && i < 2; i++) {
            dev.bufs[i] = ggml_backend_buft_alloc_buffer(buft, buf_size);
            if (!dev.bufs[i]) {
                // Clean up on failure
                for (int j = 0; j < i; j++) {
                    ggml_backend_buffer_free(dev.bufs[j]);
                    dev.bufs[j] = nullptr;
                }
                continue;
            }
            dev.ptrs[i] = ggml_backend_buffer_get_base(dev.bufs[i]);
        }

        if (dev.bufs[0]) {
            devices.push_back(dev);
        }
    }

    return !devices.empty();
}

void llama_weight_stage::free() {
    for (auto & dev : devices) {
        for (int i = 0; i < 2; i++) {
            if (dev.bufs[i]) {
                ggml_backend_buffer_free(dev.bufs[i]);
                dev.bufs[i] = nullptr;
                dev.ptrs[i] = nullptr;
            }
        }
    }
    devices.clear();
}

layer_callback_fn llama_weight_stage::make_pre_cb(upload_fn_t upload_fn) {
    return [this, upload_fn](int il, int n_layer) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Upload current layer weights to active buffer on each device
        for (int d = 0; d < (int)devices.size(); d++) {
            auto & dev = devices[d];
            upload_fn(il, d, dev.get_active(), dev.buf_size);
        }

        // Prefetch next layer to staging buffer (overlap with compute)
        int next_il = il + config.prefetch_distance;
        if (next_il < n_layer) {
            for (int d = 0; d < (int)devices.size(); d++) {
                auto & dev = devices[d];
                upload_fn(next_il, d, dev.get_staging(), dev.buf_size);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats.total_upload_ms += ms;
        stats.layers_staged++;
    };
}

layer_callback_fn llama_weight_stage::make_post_cb() {
    return [this](int il, int /*n_layer*/) {
        (void)il;
        // Swap active/staging buffers on all devices so next layer's
        // prefetched weights become active
        for (auto & dev : devices) {
            dev.swap();
        }
    };
}
