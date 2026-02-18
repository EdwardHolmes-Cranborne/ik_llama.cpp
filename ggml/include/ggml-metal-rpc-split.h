#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Create a split buffer type that distributes tensor rows across multiple
// devices (Metal + RPC) for graph split mode (tensor parallelism).
//
// device_bufts: array of per-device buffer types (e.g., [metal_buft, rpc_buft])
// n_devices:    number of devices
// tensor_split: cumulative split ratios (same as model.splits)
//
// Returns a singleton buffer type that can be used as buft_matrix in
// llama_default_buffer_type_split().
GGML_API ggml_backend_buffer_type_t ggml_backend_metal_rpc_split_buffer_type(
    ggml_backend_buffer_type_t *device_bufts, int n_devices,
    const float *tensor_split);

GGML_API bool
ggml_backend_buffer_is_metal_rpc_split(ggml_backend_buffer_t buffer);

#ifdef __cplusplus
}
#endif
