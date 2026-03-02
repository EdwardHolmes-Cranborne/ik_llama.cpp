// llama-ane-weights.h — Phase 2: GGUF → ANE weight splitting and conversion
//
// Converts GGUF quantized weights to ANE-compatible formats (INT8 or FP16).
// Supports splitting tensors along a dimension for GPU+ANE tensor parallel.
//
// Conversion pipeline:
//   GGUF tensor (any quant type)
//     → to_float() dequantize to FP32
//     → slice along split dimension
//     → convert to FP16 or INT8 (per-channel absmax scaling)
//
// Build with -DGGML_ANE=ON

#pragma once

#include "ggml.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// Type mapping: GGUF → ANE target format
// =========================================================================

// ANE target format
enum ane_target_fmt {
    ANE_FMT_FP16  = 0,   // FP16 — used for F32/F16/BF16 source or when INT8 unavailable
    ANE_FMT_INT8  = 1,   // INT8 with per-channel absmax scaling — optimal for ANE
};

// Map a GGUF quantization type to the optimal ANE target format.
// Returns ANE_FMT_INT8 for all quantized types (Q4_0, Q8_0, IQ*, etc.)
// Returns ANE_FMT_FP16 for float types (F32, F16, BF16)
// Returns ANE_FMT_FP16 for unknown types as safe fallback.
enum ane_target_fmt ane_target_type(enum ggml_type src_type);

// Get the human-readable name for an ANE target format.
const char * ane_fmt_name(enum ane_target_fmt fmt);

// =========================================================================
// Weight conversion: GGUF → FP32 → ANE format
// =========================================================================

// Result of a weight conversion.
struct ane_converted_weight {
    void *           data;         // Converted weight data (FP16 or INT8). Caller must free().
    size_t           nbytes;       // Size of data in bytes
    float *          scales;       // Per-channel absmax scales (only for INT8). Caller must free(). NULL for FP16.
    int              n_scales;     // Number of scale values (= out_channels for INT8)
    enum ane_target_fmt fmt;       // Format of the converted data
    int64_t          ne[2];        // Shape: [out_dim, in_dim] after split
};

// Dequantize a GGUF tensor to FP32.
// The tensor data must be accessible (CPU memory).
// Returns malloc'd FP32 buffer with ggml_nelements(tensor) floats, or NULL on failure.
// Caller must free() the returned buffer.
float * ane_dequantize_tensor(const struct ggml_tensor * tensor);

// Convert a GGUF tensor to ANE format.
// Dequantizes to FP32, then converts to FP16 or INT8 (per ane_target_type).
// The full tensor is converted (no splitting).
// Returns filled ane_converted_weight, or .data=NULL on failure.
struct ane_converted_weight ane_convert_tensor(const struct ggml_tensor * tensor);

// =========================================================================
// Splitting: extract a sub-range along a dimension
// =========================================================================

// Split a GGUF tensor along dimension `dim`, taking elements [split_start, split_start+split_count).
// Then convert the split portion to ANE format.
//
// For weight matrices [out_dim, in_dim]:
//   dim=0: split along out_dim (column split for QKV heads, FFN intermediate)
//   dim=1: split along in_dim (row split for down projection)
//
// Returns filled ane_converted_weight with the split portion, or .data=NULL on failure.
struct ane_converted_weight ane_split_and_convert(
    const struct ggml_tensor * tensor,
    int dim,
    int64_t split_start,
    int64_t split_count);

// =========================================================================
// INT8 quantization helpers
// =========================================================================

// Quantize FP32 data to INT8 with per-channel absmax scaling.
// data_f32: row-major [n_rows, n_cols] FP32 input
// out_int8: pre-allocated [n_rows * n_cols] INT8 output
// out_scales: pre-allocated [n_rows] FP32 scale output
// Each row is quantized independently: int8_val = round(fp32_val / scale * 127)
void ane_quantize_int8_per_channel(
    const float * data_f32,
    int8_t * out_int8,
    float * out_scales,
    int64_t n_rows,
    int64_t n_cols);

// Convert FP32 data to FP16 (using hardware conversion if available).
// data_f32: [n_elements] FP32 input
// out_fp16: pre-allocated [n_elements] FP16 output
void ane_convert_f32_to_fp16(
    const float * data_f32,
    void * out_fp16,
    int64_t n_elements);

// =========================================================================
// Cache hashing
// =========================================================================

// Compute a hash of the GGUF file for cache invalidation.
// Reads the first 64 bytes + file size to create a fast fingerprint.
// Returns 0 on failure.
uint64_t ane_compute_model_hash(const char * gguf_path);

#ifdef __cplusplus
}
#endif
