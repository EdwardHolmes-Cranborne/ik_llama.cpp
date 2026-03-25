#include "llama-kv-gather.h"
#include "llama-impl.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

bool llama_kv_gather_layer(
    const llama_kv_cache & kv,
    int il,
    int tensor_kind,
    int token_count,
    uint8_t * dst,
    size_t dst_size) {

    ggml_tensor * tensor = nullptr;
    if (tensor_kind == 0) {
        if (il >= (int)kv.k_l.size()) return false;
        tensor = kv.k_l[il];
    } else {
        if (il >= (int)kv.v_l.size()) return false;
        tensor = kv.v_l[il];
    }

    if (!tensor) {
        // MLA models may have nullptr V tensors
        if (tensor_kind == 1) {
            memset(dst, 0, dst_size);
            return true;
        }
        return false;
    }

    // Check if tensor is split across devices
    auto * split = (const ggml_split_tensor_t *)tensor->extra;
    if (split && split->n_device > 1) {
        // Multi-device: gather from each split
        size_t offset = 0;
        for (int d = 0; d < split->n_device; d++) {
            ggml_tensor * split_tensor = split->splits[d];
            if (!split_tensor) continue;

            // Calculate bytes for this split's contribution
            size_t split_row_bytes = ggml_row_size(split_tensor->type, split_tensor->ne[0]);
            size_t split_bytes = split_row_bytes;
            for (int dim = 1; dim < ggml_n_dims(split_tensor); dim++) {
                // Only count up to token_count rows
                if (dim == ggml_n_dims(split_tensor) - 1) {
                    split_bytes *= std::min((int64_t)token_count, split_tensor->ne[dim]);
                } else {
                    split_bytes *= split_tensor->ne[dim];
                }
            }

            if (offset + split_bytes > dst_size) {
                split_bytes = dst_size - offset;
            }

            ggml_backend_tensor_get(split_tensor, dst + offset, 0, split_bytes);
            offset += split_bytes;
        }
        return true;
    }

    // Single device: direct read
    size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    size_t bytes = row_bytes;
    for (int dim = 1; dim < ggml_n_dims(tensor); dim++) {
        if (dim == ggml_n_dims(tensor) - 1) {
            bytes *= std::min((int64_t)token_count, tensor->ne[dim]);
        } else {
            bytes *= tensor->ne[dim];
        }
    }
    bytes = std::min(bytes, dst_size);

    ggml_backend_tensor_get(tensor, dst, 0, bytes);
    return true;
}

bool llama_kv_gather(
    llama_context * ctx,
    int token_count,
    std::vector<uint8_t> & payload_out,
    llama_kv_gather_result * result) {

    if (result) *result = {};
    payload_out.clear();

    const llama_kv_cache & kv = ctx->kv_self;
    const int n_layer = (int)kv.k_l.size();

    if (n_layer == 0 || token_count <= 0) {
        return true;
    }

    // First pass: calculate total size needed
    size_t total = 0;
    int max_devices = 1;

    for (int il = 0; il < n_layer; il++) {
        if (kv.k_l[il]) {
            auto * split = (const ggml_split_tensor_t *)kv.k_l[il]->extra;
            if (split && split->n_device > max_devices) max_devices = split->n_device;

            size_t row_bytes = ggml_row_size(kv.k_l[il]->type, kv.k_l[il]->ne[0]);
            // K: [n_embd_k, n_kv] — we want token_count rows
            total += row_bytes * token_count;
        }

        if (il < (int)kv.v_l.size() && kv.v_l[il]) {
            size_t row_bytes = ggml_row_size(kv.v_l[il]->type, kv.v_l[il]->ne[0]);
            total += row_bytes * token_count;
        }
    }

    payload_out.resize(total, 0);

    // Second pass: gather data
    size_t offset = 0;
    for (int il = 0; il < n_layer; il++) {
        // K
        if (kv.k_l[il]) {
            size_t row_bytes = ggml_row_size(kv.k_l[il]->type, kv.k_l[il]->ne[0]);
            size_t k_bytes = row_bytes * token_count;
            llama_kv_gather_layer(kv, il, 0, token_count,
                                  payload_out.data() + offset, k_bytes);
            offset += k_bytes;
        }

        // V
        if (il < (int)kv.v_l.size() && kv.v_l[il]) {
            size_t row_bytes = ggml_row_size(kv.v_l[il]->type, kv.v_l[il]->ne[0]);
            size_t v_bytes = row_bytes * token_count;
            llama_kv_gather_layer(kv, il, 1, token_count,
                                  payload_out.data() + offset, v_bytes);
            offset += v_bytes;
        }
    }

    if (result) {
        result->total_bytes = offset;
        result->n_layers    = n_layer;
        result->n_devices   = max_devices;
    }

    return true;
}
