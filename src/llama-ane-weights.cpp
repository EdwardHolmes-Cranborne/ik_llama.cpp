// llama-ane-weights.cpp — Phase 2: GGUF → ANE weight splitting and conversion
//
// Converts GGUF quantized weights to ANE-compatible formats (INT8 or FP16).

#include "llama-ane-weights.h"
#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>

// =========================================================================
// Type mapping
// =========================================================================

enum ane_target_fmt ane_target_type(enum ggml_type src_type) {
    switch (src_type) {
        // Float types → FP16 (no requantization needed, just truncate)
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return ANE_FMT_FP16;

        // All quantized types → INT8 (dequant to FP32, then per-channel INT8)
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_Q6_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_I2_S:
            return ANE_FMT_INT8;

        default:
            // Unknown type — safe fallback to FP16 via dequant→FP16
            return ANE_FMT_FP16;
    }
}

const char * ane_fmt_name(enum ane_target_fmt fmt) {
    switch (fmt) {
        case ANE_FMT_FP16: return "FP16";
        case ANE_FMT_INT8: return "INT8";
        default:           return "UNKNOWN";
    }
}

// =========================================================================
// Dequantization
// =========================================================================

float * ane_dequantize_tensor(const struct ggml_tensor * tensor) {
    if (!tensor || !tensor->data) return NULL;

    const int64_t n_elements = ggml_nelements(tensor);
    if (n_elements <= 0) return NULL;

    float * fp32_buf = (float *)malloc(n_elements * sizeof(float));
    if (!fp32_buf) return NULL;

    if (tensor->type == GGML_TYPE_F32) {
        // Already FP32 — just copy
        memcpy(fp32_buf, tensor->data, n_elements * sizeof(float));
        return fp32_buf;
    }

    // Get the dequantization function for this type
    ggml_type_traits_t traits = ggml_internal_get_type_traits(tensor->type);
    if (!traits.to_float) {
        fprintf(stderr, "[ANE] No to_float() for type %d (%s)\n",
                tensor->type, ggml_type_name(tensor->type));
        free(fp32_buf);
        return NULL;
    }

    // Dequantize: quantized data → FP32
    traits.to_float(tensor->data, fp32_buf, n_elements);
    return fp32_buf;
}

// =========================================================================
// INT8 quantization (per-channel absmax)
// =========================================================================

void ane_quantize_int8_per_channel(
    const float * data_f32,
    int8_t * out_int8,
    float * out_scales,
    int64_t n_rows,
    int64_t n_cols)
{
    for (int64_t r = 0; r < n_rows; r++) {
        const float * row = data_f32 + r * n_cols;

        // Find absmax for this row (channel)
        float amax = 0.0f;
        for (int64_t c = 0; c < n_cols; c++) {
            float av = fabsf(row[c]);
            if (av > amax) amax = av;
        }

        // Scale: maps [-amax, +amax] → [-127, +127]
        float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        out_scales[r] = scale;

        // Quantize
        float inv_scale = 1.0f / scale;
        int8_t * out_row = out_int8 + r * n_cols;
        for (int64_t c = 0; c < n_cols; c++) {
            float v = row[c] * inv_scale;
            // Round and clamp to [-127, 127]
            int iv = (int)roundf(v);
            if (iv > 127)  iv = 127;
            if (iv < -127) iv = -127;
            out_row[c] = (int8_t)iv;
        }
    }
}

// =========================================================================
// FP32 → FP16 conversion
// =========================================================================

void ane_convert_f32_to_fp16(
    const float * data_f32,
    void * out_fp16,
    int64_t n_elements)
{
    // Use ggml's FP32→FP16 conversion
    ggml_fp32_to_fp16_row(data_f32, (ggml_fp16_t *)out_fp16, n_elements);
}

// =========================================================================
// Full tensor conversion (no split)
// =========================================================================

struct ane_converted_weight ane_convert_tensor(const struct ggml_tensor * tensor) {
    struct ane_converted_weight result = {};

    if (!tensor || !tensor->data) return result;

    // Determine shape: treat as 2D [ne[1], ne[0]] (rows × cols)
    int64_t n_rows = tensor->ne[1] > 0 ? tensor->ne[1] : 1;
    int64_t n_cols = tensor->ne[0];
    int64_t n_elements = n_rows * n_cols;

    // Dequantize to FP32
    float * fp32_buf = ane_dequantize_tensor(tensor);
    if (!fp32_buf) return result;

    // Determine target format
    enum ane_target_fmt fmt = ane_target_type(tensor->type);

    if (fmt == ANE_FMT_INT8) {
        // FP32 → INT8 per-channel
        int8_t * int8_data = (int8_t *)malloc(n_elements * sizeof(int8_t));
        float * scales = (float *)malloc(n_rows * sizeof(float));
        if (!int8_data || !scales) {
            free(fp32_buf);
            free(int8_data);
            free(scales);
            return result;
        }

        ane_quantize_int8_per_channel(fp32_buf, int8_data, scales, n_rows, n_cols);

        result.data     = int8_data;
        result.nbytes   = n_elements * sizeof(int8_t);
        result.scales   = scales;
        result.n_scales = (int)n_rows;
        result.fmt      = ANE_FMT_INT8;
        result.ne[0]    = n_cols;
        result.ne[1]    = n_rows;
    } else {
        // FP32 → FP16
        ggml_fp16_t * fp16_data = (ggml_fp16_t *)malloc(n_elements * sizeof(ggml_fp16_t));
        if (!fp16_data) {
            free(fp32_buf);
            return result;
        }

        ane_convert_f32_to_fp16(fp32_buf, fp16_data, n_elements);

        result.data     = fp16_data;
        result.nbytes   = n_elements * sizeof(ggml_fp16_t);
        result.scales   = NULL;
        result.n_scales = 0;
        result.fmt      = ANE_FMT_FP16;
        result.ne[0]    = n_cols;
        result.ne[1]    = n_rows;
    }

    free(fp32_buf);
    return result;
}

// =========================================================================
// Split + convert
// =========================================================================

struct ane_converted_weight ane_split_and_convert(
    const struct ggml_tensor * tensor,
    int dim,
    int64_t split_start,
    int64_t split_count)
{
    struct ane_converted_weight result = {};

    if (!tensor || !tensor->data) return result;
    if (dim < 0 || dim > 1) return result;

    int64_t ne0 = tensor->ne[0];  // cols (inner dim)
    int64_t ne1 = tensor->ne[1] > 0 ? tensor->ne[1] : 1;  // rows (outer dim)

    // Validate split range
    int64_t dim_size = (dim == 0) ? ne0 : ne1;
    if (split_start < 0 || split_start + split_count > dim_size) {
        fprintf(stderr, "[ANE] Split out of range: dim=%d size=%lld start=%lld count=%lld\n",
                dim, (long long)dim_size, (long long)split_start, (long long)split_count);
        return result;
    }

    // Dequantize the full tensor to FP32
    float * fp32_full = ane_dequantize_tensor(tensor);
    if (!fp32_full) return result;

    // Extract the split portion
    int64_t out_rows, out_cols;
    float * fp32_split = NULL;

    if (dim == 0) {
        // Split along columns (ne[0]): each row gets a subset of columns
        out_rows = ne1;
        out_cols = split_count;
        fp32_split = (float *)malloc(out_rows * out_cols * sizeof(float));
        if (!fp32_split) { free(fp32_full); return result; }

        for (int64_t r = 0; r < ne1; r++) {
            const float * src_row = fp32_full + r * ne0 + split_start;
            float * dst_row = fp32_split + r * out_cols;
            memcpy(dst_row, src_row, out_cols * sizeof(float));
        }
    } else {
        // Split along rows (ne[1]): take a subset of rows
        out_rows = split_count;
        out_cols = ne0;
        fp32_split = (float *)malloc(out_rows * out_cols * sizeof(float));
        if (!fp32_split) { free(fp32_full); return result; }

        const float * src = fp32_full + split_start * ne0;
        memcpy(fp32_split, src, out_rows * out_cols * sizeof(float));
    }

    free(fp32_full);

    // Convert the split data to ANE format
    enum ane_target_fmt fmt = ane_target_type(tensor->type);
    int64_t n_elements = out_rows * out_cols;

    if (fmt == ANE_FMT_INT8) {
        int8_t * int8_data = (int8_t *)malloc(n_elements * sizeof(int8_t));
        float * scales = (float *)malloc(out_rows * sizeof(float));
        if (!int8_data || !scales) {
            free(fp32_split);
            free(int8_data);
            free(scales);
            return result;
        }

        ane_quantize_int8_per_channel(fp32_split, int8_data, scales, out_rows, out_cols);

        result.data     = int8_data;
        result.nbytes   = n_elements * sizeof(int8_t);
        result.scales   = scales;
        result.n_scales = (int)out_rows;
        result.fmt      = ANE_FMT_INT8;
    } else {
        ggml_fp16_t * fp16_data = (ggml_fp16_t *)malloc(n_elements * sizeof(ggml_fp16_t));
        if (!fp16_data) {
            free(fp32_split);
            return result;
        }

        ane_convert_f32_to_fp16(fp32_split, fp16_data, n_elements);

        result.data     = fp16_data;
        result.nbytes   = n_elements * sizeof(ggml_fp16_t);
        result.scales   = NULL;
        result.n_scales = 0;
        result.fmt      = ANE_FMT_FP16;
    }

    result.ne[0] = out_cols;
    result.ne[1] = out_rows;

    free(fp32_split);
    return result;
}

// =========================================================================
// Cache hashing
// =========================================================================

uint64_t ane_compute_model_hash(const char * gguf_path) {
    if (!gguf_path) return 0;

    FILE * f = fopen(gguf_path, "rb");
    if (!f) return 0;

    // Read first 64 bytes (GGUF header)
    uint8_t header[64];
    size_t n_read = fread(header, 1, 64, f);
    if (n_read < 64) {
        fclose(f);
        return 0;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);

    // FNV-1a hash of header + file size
    uint64_t hash = 14695981039346656037ULL; // FNV offset basis
    for (size_t i = 0; i < 64; i++) {
        hash ^= (uint64_t)header[i];
        hash *= 1099511628211ULL; // FNV prime
    }
    // Mix in file size
    for (int i = 0; i < 8; i++) {
        hash ^= (uint64_t)((file_size >> (i * 8)) & 0xFF);
        hash *= 1099511628211ULL;
    }

    return hash;
}
