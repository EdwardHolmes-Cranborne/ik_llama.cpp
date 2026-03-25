#include "llama-kv-import.h"

#include <cstdio>
#include <cstring>

bool kv_scatter_map_build(const kv_import_model_info & info,
                          uint32_t token_count,
                          kv_scatter_map & map_out) {
    map_out = {};

    if (token_count > 0 && token_count >= info.kv_size) {
        return false;
    }
    if (token_count == 0) {
        return true;
    }
    if (info.layers.size() != info.n_layers) {
        return false;
    }

    size_t artifact_cursor = 0;

    for (uint32_t il = 0; il < info.n_layers; il++) {
        const auto & layer = info.layers[il];

        // K entry
        {
            kv_scatter_entry e = {};
            e.layer_idx       = layer.cache_idx;
            e.tensor_kind     = 0;
            e.artifact_offset = artifact_cursor;
            e.tensor_offset   = 0;
            size_t k_row_bytes = (size_t)layer.n_embd_k_gqa * info.type_k_size;
            e.bytes = (size_t)token_count * k_row_bytes;
            map_out.entries.push_back(e);
            artifact_cursor += e.bytes;
        }

        // V entry (skip for MLA)
        if (!info.is_mla) {
            size_t v_row_bytes = (size_t)layer.n_embd_v_gqa * info.type_v_size;

            if (!info.v_trans) {
                kv_scatter_entry e = {};
                e.layer_idx       = layer.cache_idx;
                e.tensor_kind     = 1;
                e.artifact_offset = artifact_cursor;
                e.tensor_offset   = 0;
                e.bytes           = (size_t)token_count * v_row_bytes;
                map_out.entries.push_back(e);
                artifact_cursor += e.bytes;
            } else {
                for (uint32_t pos = 0; pos < token_count; pos++) {
                    kv_scatter_entry e = {};
                    e.layer_idx       = layer.cache_idx;
                    e.tensor_kind     = 1;
                    e.artifact_offset = artifact_cursor + (size_t)pos * v_row_bytes;
                    e.tensor_offset   = (size_t)pos * v_row_bytes;
                    e.bytes           = v_row_bytes;
                    map_out.entries.push_back(e);
                }
                artifact_cursor += (size_t)token_count * v_row_bytes;
            }
        }
    }

    map_out.total_payload_bytes = artifact_cursor;
    return true;
}

bool kv_import_validate(const kv_import_model_info &    model,
                        const kv_import_artifact_info & artifact,
                        bool allow_fingerprint_mismatch,
                        std::string * error) {
    auto set_err = [&](const std::string & msg) {
        if (error) *error = msg;
        return false;
    };

    if (artifact.format_major != 1) {
        return set_err("unsupported artifact format_major: " + std::to_string(artifact.format_major));
    }
    if (artifact.n_layers != model.n_layers) {
        return set_err("layer count mismatch: artifact=" + std::to_string(artifact.n_layers)
                       + " model=" + std::to_string(model.n_layers));
    }
    if (artifact.token_count >= model.kv_size) {
        return set_err("token_count >= kv_size — no room for decode");
    }
    if (artifact.v_trans != model.v_trans) {
        return set_err("v_trans mismatch");
    }
    if (artifact.is_mla != model.is_mla) {
        return set_err("is_mla mismatch");
    }
    if (artifact.model_fingerprint != 0 && model.model_fingerprint != 0) {
        if (artifact.model_fingerprint != model.model_fingerprint && !allow_fingerprint_mismatch) {
            return set_err("model fingerprint mismatch");
        }
    }
    return true;
}
