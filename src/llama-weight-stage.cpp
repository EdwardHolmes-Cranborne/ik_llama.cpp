#include "llama-weight-stage.h"
#include "llama-model.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

// ============================================================================
// Collect all non-null tensor pointers from an ik_llama layer
// ============================================================================

void wf_collect_tensors_from_layer(const llama_layer & layer,
                                   std::vector<wf_tensor_remap_entry> & out) {
#define PUSH_IF(field)                                     \
    if (layer.field) {                                     \
        wf_tensor_remap_entry e;                           \
        e.tensor      = layer.field;                       \
        e.orig_buffer = layer.field->buffer;               \
        e.orig_data   = layer.field->data;                 \
        if ((uintptr_t)e.orig_data <= 1) e.orig_data = nullptr; \
        e.nbytes      = ggml_nbytes(layer.field);          \
        out.push_back(e);                                  \
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
    PUSH_IF(wqkv_gate)

    // Attention weights
    PUSH_IF(wq)
    PUSH_IF(wk)
    PUSH_IF(wv)
    PUSH_IF(wo)
    PUSH_IF(wqkv)
    PUSH_IF(wqk)
    PUSH_IF(wkv)
    PUSH_IF(wq_a)
    PUSH_IF(wq_b)
    PUSH_IF(wkv_a_mqa)
    PUSH_IF(wkq_a_mqa)
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

    // Attention biases
    PUSH_IF(bq)
    PUSH_IF(bk)
    PUSH_IF(bv)
    PUSH_IF(bo)
    PUSH_IF(bqkv)
    PUSH_IF(bqk)
    PUSH_IF(bkv)

    // Attention sinks
    PUSH_IF(attn_sinks)

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
    PUSH_IF(ffn_up_gate_exps)

    // MoE biases
    PUSH_IF(ffn_gate_inp_b)
    PUSH_IF(ffn_gate_exps_b)
    PUSH_IF(ffn_down_exps_b)
    PUSH_IF(ffn_up_exps_b)
    PUSH_IF(ffn_up_gate_exps_b)
    PUSH_IF(ffn_gate_exps_b_dup)
    PUSH_IF(ffn_down_exps_b_dup)
    PUSH_IF(ffn_up_exps_b_dup)

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
    PUSH_IF(ssm_norm)
    PUSH_IF(ssm_conv1d)
    PUSH_IF(ssm_a)
    PUSH_IF(ssm_d)
    PUSH_IF(ssm_conv1d_b)
    PUSH_IF(ssm_dt_b)
    PUSH_IF(ssm_beta_alpha)
    PUSH_IF(ssm_beta)
    PUSH_IF(ssm_alpha)

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

    // DSA indexer (GLM-5)
    PUSH_IF(indexer_k_norm)
    PUSH_IF(indexer_k_norm_b)
    PUSH_IF(indexer_proj)
    PUSH_IF(indexer_attn_k)
    PUSH_IF(indexer_attn_q_b)

#undef PUSH_IF
}

size_t wf_layer_weight_bytes(const llama_model * model, int layer_idx) {
    if (!model || layer_idx < 0 || (size_t)layer_idx >= model->layers.size()) {
        return 0;
    }
    std::vector<wf_tensor_remap_entry> entries;
    wf_collect_tensors_from_layer(model->layers[layer_idx], entries);
    size_t total = 0;
    for (const auto & e : entries) total += e.nbytes;
    return total;
}

// ============================================================================
// Weight stream implementation
// ============================================================================

void llama_weight_stream::collect_layer_tensors(const llama_model * model) {
    const int n_layer = model->hparams.n_layer;
    layer_infos.resize(n_layer);
    max_layer_bytes = 0;

    for (int il = 0; il < n_layer; ++il) {
        auto & info = layer_infos[il];
        info.entries.clear();
        info.total_bytes = 0;

        wf_collect_tensors_from_layer(model->layers[il], info.entries);

        for (const auto & e : info.entries) {
            info.total_bytes += e.nbytes;
        }
        if (info.total_bytes > max_layer_bytes) {
            max_layer_bytes = info.total_bytes;
        }
    }

    fprintf(stderr, "[weight-stream] collected %d layers, max layer = %.2f MB, "
            "tensors/layer = %zu..%zu\n",
            n_layer,
            max_layer_bytes / (1024.0 * 1024.0),
            layer_infos.empty() ? (size_t)0 : layer_infos[0].entries.size(),
            layer_infos.empty() ? (size_t)0 : layer_infos[n_layer - 1].entries.size());
}

bool llama_weight_stream::allocate_gpu_buffers(ggml_backend_t backend) {
    free_buffers();
    gpu_backend = backend;

    if (max_layer_bytes == 0) {
        fprintf(stderr, "[weight-stream] ERROR: max_layer_bytes == 0, call collect_layer_tensors first\n");
        return false;
    }

    // Get GPU buffer type from backend
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    if (!buft) {
        fprintf(stderr, "[weight-stream] ERROR: backend has no buffer type\n");
        return false;
    }

    // Add 10% padding for alignment
    gpu_buf_size = max_layer_bytes + max_layer_bytes / 10;
    // Round up to 64KB for D3D12/alignment
    gpu_buf_size = (gpu_buf_size + 65535) & ~(size_t)65535;

    int n_alloc = std::min(config.n_buffers, 2);
    for (int i = 0; i < n_alloc; i++) {
        gpu_buf[i] = ggml_backend_buft_alloc_buffer(buft, gpu_buf_size);
        if (!gpu_buf[i]) {
            fprintf(stderr, "[weight-stream] ERROR: failed to allocate GPU buffer %d (%.2f MB)\n",
                    i, gpu_buf_size / (1024.0 * 1024.0));
            free_buffers();
            return false;
        }
        gpu_ptr[i] = ggml_backend_buffer_get_base(gpu_buf[i]);

        // Mark as weight buffer so scheduler routes matmuls to this backend
        ggml_backend_buffer_set_usage(gpu_buf[i], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    fprintf(stderr, "[weight-stream] allocated %d GPU buffers: %.2f MB each (%.2f MB total)\n",
            n_alloc, gpu_buf_size / (1024.0 * 1024.0),
            n_alloc * gpu_buf_size / (1024.0 * 1024.0));

    return true;
}

void llama_weight_stream::remap_tensors_to_gpu() {
    if (!gpu_buf[0]) return;

    int total_remapped = 0;
    for (auto & info : layer_infos) {
        size_t offset = 0;
        for (auto & entry : info.entries) {
            // Save original pointers
            entry.orig_buffer = entry.tensor->buffer;
            entry.orig_data   = entry.tensor->data;
            if ((uintptr_t)entry.orig_data <= 1) entry.orig_data = nullptr;

            // Remap to GPU buffer (wrap around for scheduler inspection)
            entry.tensor->buffer = gpu_buf[0];
            entry.tensor->data = (char *)gpu_ptr[0] + (offset % gpu_buf_size);
            offset += entry.nbytes;
            total_remapped++;
        }
    }

    if (config.trace) {
        fprintf(stderr, "[weight-stream] remapped %d tensors to GPU buffer for scheduler\n",
                total_remapped);
    }
}

void llama_weight_stream::restore_tensors_to_cpu() {
    for (auto & info : layer_infos) {
        for (auto & entry : info.entries) {
            entry.tensor->buffer = entry.orig_buffer;
            entry.tensor->data   = entry.orig_data;
        }
    }
}

void llama_weight_stream::upload_layer(int il) {
    if (il < 0 || il >= (int)layer_infos.size()) return;

    auto & info = layer_infos[il];
    void * dst_base = gpu_ptr[active_buf];
    ggml_backend_buffer_t dst_buf = gpu_buf[active_buf];

    auto t0 = std::chrono::high_resolution_clock::now();

    // Copy each tensor from CPU (mmap) to GPU staging buffer, packed contiguously.
    // We remap tensor->buffer and tensor->data to the GPU buffer FIRST,
    // then use ggml_backend_tensor_set to upload from CPU source.
    // This way ggml_backend_tensor_set knows the destination is on GPU.
    size_t offset = 0;
    for (auto & entry : info.entries) {
        void * src = entry.orig_data;
        void * dst = (char *)dst_base + offset;

        // Remap tensor to GPU buffer for compute
        entry.tensor->buffer = dst_buf;
        entry.tensor->data = dst;

        if (src && entry.nbytes > 0) {
            // ggml_backend_tensor_set: copies from host src → tensor's backend (GPU)
            ggml_backend_tensor_set(entry.tensor, src, 0, entry.nbytes);
        }

        offset += entry.nbytes;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    stats.total_upload_ms += ms;
    stats.bytes_uploaded += info.total_bytes;
    stats.layers_staged++;

    if (config.trace) {
        fprintf(stderr, "[weight-stream] upload layer %d: %.2f MB in %.2f ms (%.1f GB/s)\n",
                il, info.total_bytes / (1024.0 * 1024.0), ms,
                ms > 0 ? (info.total_bytes / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0) : 0.0);
    }
}

void llama_weight_stream::finish_layer(int il) {
    if (il < 0 || il >= (int)layer_infos.size()) return;

    // Restore tensor pointers to CPU (so next graph build finds them on CPU)
    auto & info = layer_infos[il];
    for (auto & entry : info.entries) {
        entry.tensor->buffer = entry.orig_buffer;
        entry.tensor->data   = entry.orig_data;
    }

    // Swap active buffer for next layer
    active_buf = 1 - active_buf;
}

layer_callback_fn llama_weight_stream::make_pre_layer_cb() {
    return [this](int il, int /*n_layer*/) {
        upload_layer(il);
    };
}

layer_callback_fn llama_weight_stream::make_post_layer_cb() {
    return [this](int il, int /*n_layer*/) {
        finish_layer(il);
    };
}

void llama_weight_stream::free_buffers() {
    for (int i = 0; i < 2; i++) {
        if (gpu_buf[i]) {
            ggml_backend_buffer_free(gpu_buf[i]);
            gpu_buf[i] = nullptr;
            gpu_ptr[i] = nullptr;
        }
    }
    active_buf = 0;
    gpu_backend = nullptr;
}
