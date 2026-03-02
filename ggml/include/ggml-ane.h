// ggml-ane.h — Apple Neural Engine backend for ggml
//
// Uses public CoreML API to dispatch conv-based matmuls to the ANE.
// Designed as a co-processor alongside Metal GPU for split-graph
// prefill acceleration on M3/M4 Apple Silicon.
//
// Models are FP16 MLProgram compiled via coremltools (opset_version=iOS17).
// Requires macOS 14+ with Apple Silicon.
// Build with -DGGML_ANE=ON

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// Backend API (standard ggml pattern)
// =========================================================================

// Check if ANE is available (Apple Silicon with Neural Engine)
GGML_API bool ggml_backend_ane_supported(void);

// Initialize the ANE backend (stub — ANE runs as co-processor side-channel)
GGML_API ggml_backend_t ggml_backend_ane_init(void);

// Check if a backend instance is the ANE backend
GGML_API bool ggml_backend_is_ane(ggml_backend_t backend);

// Get the ANE buffer type (stub)
GGML_API ggml_backend_buffer_type_t ggml_backend_ane_buffer_type(void);

// =========================================================================
// ANE kernel management — public CoreML API
// =========================================================================

// Opaque handle to a loaded ANE model
typedef struct ggml_ane_kernel * ggml_ane_kernel_t;

// Load a pre-compiled .mlmodelc from disk with ANE compute units.
// input_name: name of the model's input feature (e.g., "x")
// compute_units: 0=CPU, 1=GPU, 2=ANE, 3=ALL (default: 2=ANE)
GGML_API ggml_ane_kernel_t ggml_ane_load(const char * modelc_path,
                                           const char * input_name,
                                           int compute_units);

// Generate a model with baked weights using coremltools, compile, and load.
// python_path: path to python3.12 (NULL = auto-detect)
// op_type: "conv" for single conv, "qkv" for fused Q/K/V, "ffn" for fused FFN
// in_ch, out_ch, spatial: dimensions
// weight_f32: row-major float32 weights [out_ch, in_ch] (or multiple concatenated)
// weight_nbytes: total bytes of weight data
// cache_dir: directory for caching compiled models (NULL = temp dir)
// Returns loaded kernel or NULL on failure.
GGML_API ggml_ane_kernel_t ggml_ane_compile(
    const char * python_path,
    const char * op_type,
    int in_ch, int out_ch, int spatial,
    const float * weight_f32, size_t weight_nbytes,
    const char * cache_dir);

// Write FP16 data to the model's input.
// data points to __fp16 array in channel-first layout [1, C, 1, S].
GGML_API void ggml_ane_write_input(ggml_ane_kernel_t k, int idx,
                                    const void * data, size_t nbytes);

// Read FP16 data from the model's output.
// data points to __fp16 buffer in channel-first layout [1, C, 1, S].
GGML_API void ggml_ane_read_output(ggml_ane_kernel_t k, int idx,
                                    void * data, size_t nbytes);

// Execute the model on ANE (synchronous)
GGML_API bool ggml_ane_eval(ggml_ane_kernel_t k);

// Read output data after eval. idx selects output (0-based).
// Copies FP16 data into caller's buffer.
GGML_API void ggml_ane_read_output_after_eval(ggml_ane_kernel_t k, int idx,
                                                void * data, size_t nbytes);

// Free a loaded kernel
GGML_API void ggml_ane_free(ggml_ane_kernel_t k);

// Get number of outputs from a loaded kernel
GGML_API int ggml_ane_n_outputs(ggml_ane_kernel_t k);

// Get the output name at given index
GGML_API const char * ggml_ane_output_name(ggml_ane_kernel_t k, int idx);

// =========================================================================
// Weight blob builders (used by Python model generator)
// =========================================================================

// Build FP16 weight blob with coremltools-compatible header.
// weights_f32: row-major [out_ch, in_ch] float32
// Returns malloc'd blob (caller must free), sets *out_nbytes
GGML_API void * ggml_ane_build_weight_blob(const float * weights_f32,
                                            int out_ch, int in_ch,
                                            size_t * out_nbytes);

// Free a MIL string or weight blob
GGML_API void ggml_ane_free_string(char * s);

// =========================================================================
// Benchmarking / characterization (Phase 0)
// =========================================================================

// Measure peak TFLOPS using a loaded kernel.
// channels, spatial: dimensions for FLOP calculation (2*C*C*S per eval).
// Fills input with small values, runs warmup + n_iters, returns TFLOPS.
GGML_API float ggml_ane_measure_tflops(ggml_ane_kernel_t k,
                                         int channels, int spatial,
                                         int n_iters);

// Measure dispatch latency (mean ms over n_iters) using a loaded kernel.
// Runs warmup + n_iters evals, returns mean ms/eval.
GGML_API float ggml_ane_measure_dispatch_latency(ggml_ane_kernel_t k,
                                                   int n_iters);

// Test Metal+ANE concurrent execution using a loaded ANE kernel.
// Runs Metal busy-work shader concurrently with ANE eval.
// Returns wall_time / max(metal_time, ane_time); < 1.1 means concurrent.
GGML_API float ggml_ane_test_concurrent_metal(ggml_ane_kernel_t k);

#ifdef __cplusplus
}
#endif
