#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct kv_scatter_entry {
    uint32_t layer_idx;
    uint32_t tensor_kind;     // 0 = K, 1 = V
    size_t   artifact_offset;
    size_t   tensor_offset;
    size_t   bytes;
};

struct kv_scatter_map {
    std::vector<kv_scatter_entry> entries;
    size_t total_payload_bytes = 0;
};

struct kv_import_layer_info {
    uint32_t cache_idx;
    uint32_t n_embd_k_gqa;
    uint32_t n_embd_v_gqa;
};

struct kv_import_model_info {
    uint32_t n_layers    = 0;
    uint32_t kv_size     = 0;
    bool     v_trans     = false;
    bool     is_mla      = false;
    size_t   type_k_size = 0;
    size_t   type_v_size = 0;
    uint64_t model_fingerprint = 0;
    std::vector<kv_import_layer_info> layers;
};

struct kv_import_artifact_info {
    uint16_t format_major      = 0;
    uint32_t n_layers          = 0;
    uint32_t token_count       = 0;
    bool     v_trans           = false;
    bool     is_mla            = false;
    uint64_t model_fingerprint = 0;
};

bool kv_scatter_map_build(const kv_import_model_info & info,
                          uint32_t token_count,
                          kv_scatter_map & map_out);

bool kv_import_validate(const kv_import_model_info &    model,
                        const kv_import_artifact_info & artifact,
                        bool allow_fingerprint_mismatch,
                        std::string * error);
