// llama-ane-dispatch.h — Phase 3+4: ANE FFN dispatch for GPU+ANE split prefill
//
// Manages per-layer CoreML FFN kernels that run on the ANE in parallel with
// GPU computation. During prefill, the FFN MLP is split along the intermediate
// dimension: GPU handles [0, gpu_inter) and ANE handles [gpu_inter, n_ff).
// Results are summed after both complete.
//
// Build with -DGGML_ANE=ON

#pragma once

#include "llama-hparams.h"

#include <stddef.h>
#include <stdbool.h>

struct llama_layer;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// Opaque dispatch context (one per llama_context when ANE prefill is active)
typedef struct ane_dispatch_ctx * ane_dispatch_ctx_t;

// Initialize an ANE dispatch context.
// split_ratio: fraction of intermediate dim routed to ANE (e.g. 0.5)
// cache_dir:   directory for caching compiled CoreML models (NULL = temp)
// python_path: path to python3 with coremltools (NULL = auto-detect)
ane_dispatch_ctx_t ane_dispatch_init(
    const struct llama_hparams * hparams,
    float split_ratio,
    const char * cache_dir,
    const char * python_path);

// Compile fused FFN kernels for all layers.
// Extracts ANE-portion weights, generates CoreML models, caches them.
// Also pre-computes GPU-half FP16 weights.
// seq_len: maximum sequence length for this prefill batch.
// Returns true on success.
bool ane_dispatch_compile_kernels(
    ane_dispatch_ctx_t ctx,
    const struct llama_layer * layers,
    int n_layers,
    int seq_len);

// Dispatch ANE FFN computation asynchronously for layer il.
// ffn_inp_f16: pointer to FP16 input tensor data, row-major [seq_len, hidden_dim]
// seq_len: current sequence length
// The function returns immediately; use ane_dispatch_ffn_sync() to wait.
void ane_dispatch_ffn_async(
    ane_dispatch_ctx_t ctx,
    int il,
    const void * ffn_inp_f16,
    int seq_len);

// Wait for the ANE FFN result for layer il.
// out_f16: buffer to receive FP16 output, row-major [seq_len, hidden_dim]
// nbytes:  size of out_f16 buffer
// Returns wall-clock wait time in milliseconds.
float ane_dispatch_ffn_sync(
    ane_dispatch_ctx_t ctx,
    int il,
    void * out_f16,
    size_t nbytes);

// Get the GPU-half intermediate dimension (for tensor shape modification).
int ane_dispatch_gpu_inter_dim(ane_dispatch_ctx_t ctx);

// Get the ANE-half intermediate dimension.
int ane_dispatch_ane_inter_dim(ane_dispatch_ctx_t ctx);

// Get the pre-computed GPU-half FP16 weights for a layer's FFN tensor.
// tensor_name: "ffn_gate", "ffn_up", or "ffn_down"
// Returns pointer to FP16 data, or NULL if not available.
// The pointer is valid until ane_dispatch_free().
const void * ane_dispatch_gpu_weight(
    ane_dispatch_ctx_t ctx,
    int il,
    const char * tensor_name);

// Get the byte size of a GPU-half weight tensor.
size_t ane_dispatch_gpu_weight_bytes(
    ane_dispatch_ctx_t ctx,
    int il,
    const char * tensor_name);

// Get the ggml_type of a GPU-half weight tensor.
// tensor_name: "ffn_gate", "ffn_up", or "ffn_down"
// Returns original quant type (zero-copy) or GGML_TYPE_F16 (fallback).
int ane_dispatch_gpu_weight_type(ane_dispatch_ctx_t ctx, int il, const char * tensor_name);

// Check whether a layer should use ANE split (false for MoE layers).
bool ane_dispatch_layer_active(ane_dispatch_ctx_t ctx, int il);

// Check if the existing context can handle the given seq_len.
// Returns true if kernels are compiled and seq_len matches.
bool ane_dispatch_is_ready(ane_dispatch_ctx_t ctx, int seq_len);

// Validate ANE output for layer 0 against CPU reference.
// Requires ANE_VALIDATE=1 env var to be set before compile_kernels.
// normed_input_f32: row-major [seq_len, hidden], post-RMS-norm FP32 input
// ane_output_f16:   row-major [seq_len, hidden], FP16 output from ANE
// Returns true if max absolute error < 0.1.
bool ane_dispatch_validate_layer0(
    ane_dispatch_ctx_t ctx,
    const float * normed_input_f32,
    const void * ane_output_f16,
    int seq_len);

// Free the dispatch context and all per-layer state.
void ane_dispatch_free(ane_dispatch_ctx_t ctx);

#ifdef __cplusplus
}
#endif
