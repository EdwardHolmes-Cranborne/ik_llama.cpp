#pragma once

#include "llama.h"
#include "llama-context.h"

#include <cstdint>
#include <vector>

// Gather KV cache data from potentially split (multi-GPU) tensors into
// a contiguous buffer suitable for artifact serialization.
//
// When split_mode_graph is active, each layer's K/V tensors may be
// distributed across multiple devices via ggml_split_tensor_t. This
// function reads from each device's split tensor and concatenates into
// a single payload.
//
// When KV is not split (single GPU), this is equivalent to a direct
// ggml_backend_tensor_get.

struct llama_kv_gather_result {
    size_t total_bytes = 0;
    int    n_layers    = 0;
    int    n_devices   = 0;  // max devices any layer was split across
};

// Gather all KV data for token_count tokens into payload_out.
// Returns false on failure.
bool llama_kv_gather(
    llama_context * ctx,
    int token_count,
    std::vector<uint8_t> & payload_out,
    llama_kv_gather_result * result);

// Gather a single layer's K (or V) tensor data into dst.
// split_idx: if >= 0, gather only this split; if -1, gather all and concatenate.
bool llama_kv_gather_layer(
    const llama_kv_cache & kv,
    int il,
    int tensor_kind,  // 0=K, 1=V
    int token_count,
    uint8_t * dst,
    size_t dst_size);
