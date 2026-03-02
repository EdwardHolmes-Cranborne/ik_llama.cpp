// test-ane-weights.cpp — Phase 2 unit tests for GGUF → ANE weight conversion
//
// Tests: type mapping, dequantization, INT8 quantization, FP16 conversion,
//        split+convert, round-trip accuracy, cache hashing.

#include "llama-ane-weights.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_BEGIN(name) printf("TEST: %s\n", name)
#define TEST_PASS(name) do { printf("  PASS: %s\n\n", name); g_tests_passed++; } while(0)
#define TEST_FAIL(name, ...) do { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n\n"); g_tests_failed++; } while(0)

#define TEST_ASSERT(cond, name, ...) do { \
    if (!(cond)) { TEST_FAIL(name, __VA_ARGS__); return; } \
} while(0)

// ===========================================================================
// Test: Type mapping
// ===========================================================================
static void test_type_mapping(void) {
    TEST_BEGIN("type_mapping");

    // Float types → FP16
    TEST_ASSERT(ane_target_type(GGML_TYPE_F32)  == ANE_FMT_FP16, "type_mapping", "F32 should map to FP16");
    TEST_ASSERT(ane_target_type(GGML_TYPE_F16)  == ANE_FMT_FP16, "type_mapping", "F16 should map to FP16");
    TEST_ASSERT(ane_target_type(GGML_TYPE_BF16) == ANE_FMT_FP16, "type_mapping", "BF16 should map to FP16");

    // Quantized types → INT8
    TEST_ASSERT(ane_target_type(GGML_TYPE_Q4_0) == ANE_FMT_INT8, "type_mapping", "Q4_0 should map to INT8");
    TEST_ASSERT(ane_target_type(GGML_TYPE_Q8_0) == ANE_FMT_INT8, "type_mapping", "Q8_0 should map to INT8");
    TEST_ASSERT(ane_target_type(GGML_TYPE_Q4_K) == ANE_FMT_INT8, "type_mapping", "Q4_K should map to INT8");
    TEST_ASSERT(ane_target_type(GGML_TYPE_Q6_K) == ANE_FMT_INT8, "type_mapping", "Q6_K should map to INT8");
    TEST_ASSERT(ane_target_type(GGML_TYPE_IQ4_XS) == ANE_FMT_INT8, "type_mapping", "IQ4_XS should map to INT8");
    TEST_ASSERT(ane_target_type(GGML_TYPE_I2_S) == ANE_FMT_INT8, "type_mapping", "I2_S should map to INT8");

    // Format names
    TEST_ASSERT(strcmp(ane_fmt_name(ANE_FMT_FP16), "FP16") == 0, "type_mapping", "FP16 name mismatch");
    TEST_ASSERT(strcmp(ane_fmt_name(ANE_FMT_INT8), "INT8") == 0, "type_mapping", "INT8 name mismatch");

    TEST_PASS("type_mapping");
}

// ===========================================================================
// Test: INT8 per-channel quantization
// ===========================================================================
static void test_int8_quantization(void) {
    TEST_BEGIN("int8_quantization");

    const int n_rows = 4;
    const int n_cols = 8;
    float data[32] = {
        // Row 0: max=1.0
         0.5f,  1.0f, -0.5f, -1.0f,  0.25f, -0.25f,  0.0f,  0.75f,
        // Row 1: max=2.0
         2.0f, -2.0f,  1.0f, -1.0f,  0.5f,  -0.5f,  0.0f,  1.5f,
        // Row 2: all zeros
         0.0f,  0.0f,  0.0f,  0.0f,  0.0f,   0.0f,  0.0f,  0.0f,
        // Row 3: uniform small
         0.01f, 0.01f, 0.01f, 0.01f, 0.01f,  0.01f, 0.01f, 0.01f,
    };

    int8_t int8_out[32];
    float scales[4];

    ane_quantize_int8_per_channel(data, int8_out, scales, n_rows, n_cols);

    // Row 0: scale = 1.0/127 ≈ 0.00787
    printf("  Row 0 scale: %.6f (expected ~0.00787)\n", scales[0]);
    TEST_ASSERT(fabsf(scales[0] - 1.0f/127.0f) < 0.001f, "int8_quantization",
                "Row 0 scale wrong: %.6f", scales[0]);

    // Row 0, col 1 (value 1.0) should map to 127
    printf("  Row 0 col 1: %d (expected 127)\n", int8_out[1]);
    TEST_ASSERT(int8_out[1] == 127, "int8_quantization", "1.0 should map to 127, got %d", int8_out[1]);

    // Row 0, col 3 (value -1.0) should map to -127
    TEST_ASSERT(int8_out[3] == -127, "int8_quantization", "-1.0 should map to -127, got %d", int8_out[3]);

    // Row 2: all zeros, scale should be 1.0 (fallback)
    printf("  Row 2 scale: %.6f (expected 1.0 fallback)\n", scales[2]);
    TEST_ASSERT(scales[2] == 1.0f, "int8_quantization", "Zero row scale should be 1.0");

    // Round-trip accuracy: dequant and check max error
    float max_err = 0;
    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
            float original = data[r * n_cols + c];
            float recovered = (float)int8_out[r * n_cols + c] * scales[r];
            float err = fabsf(original - recovered);
            if (err > max_err) max_err = err;
        }
    }
    printf("  Round-trip max error: %.6f\n", max_err);
    TEST_ASSERT(max_err < 0.02f, "int8_quantization", "Round-trip error too large: %.6f", max_err);

    TEST_PASS("int8_quantization");
}

// ===========================================================================
// Test: FP16 conversion
// ===========================================================================
static void test_fp16_conversion(void) {
    TEST_BEGIN("fp16_conversion");

    const int n = 16;
    float input[16];
    for (int i = 0; i < n; i++) input[i] = (float)i * 0.1f - 0.8f;

    ggml_fp16_t output[16];
    ane_convert_f32_to_fp16(input, output, n);

    float max_err = 0;
    for (int i = 0; i < n; i++) {
        float recovered = ggml_fp16_to_fp32(output[i]);
        float err = fabsf(input[i] - recovered);
        if (err > max_err) max_err = err;
    }

    printf("  FP16 round-trip max error: %.6f\n", max_err);
    TEST_ASSERT(max_err < 0.001f, "fp16_conversion", "Error too large: %.6f", max_err);

    TEST_PASS("fp16_conversion");
}

// ===========================================================================
// Test: F32 tensor dequantization (identity — should just copy)
// ===========================================================================
static void test_dequantize_f32(void) {
    TEST_BEGIN("dequantize_f32");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != NULL, "dequantize_f32", "ggml_init failed");

    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    float * data = (float *)t->data;
    for (int i = 0; i < 32; i++) data[i] = (float)i * 0.1f;

    float * result = ane_dequantize_tensor(t);
    TEST_ASSERT(result != NULL, "dequantize_f32", "Dequantize returned NULL");

    float max_err = 0;
    for (int i = 0; i < 32; i++) {
        float err = fabsf(result[i] - data[i]);
        if (err > max_err) max_err = err;
    }

    printf("  F32 dequant max error: %.6f (should be 0)\n", max_err);
    TEST_ASSERT(max_err == 0.0f, "dequantize_f32", "F32 dequant should be exact copy");

    free(result);
    ggml_free(ctx);
    TEST_PASS("dequantize_f32");
}

// ===========================================================================
// Test: Full tensor conversion (F32 → FP16)
// ===========================================================================
static void test_convert_f32_tensor(void) {
    TEST_BEGIN("convert_f32_tensor");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    float * data = (float *)t->data;
    for (int i = 0; i < 128; i++) data[i] = (float)i * 0.01f - 0.64f;

    struct ane_converted_weight w = ane_convert_tensor(t);
    TEST_ASSERT(w.data != NULL, "convert_f32_tensor", "Conversion returned NULL");
    TEST_ASSERT(w.fmt == ANE_FMT_FP16, "convert_f32_tensor", "F32 should convert to FP16");
    TEST_ASSERT(w.ne[0] == 16, "convert_f32_tensor", "ne[0] should be 16, got %lld", (long long)w.ne[0]);
    TEST_ASSERT(w.ne[1] == 8, "convert_f32_tensor", "ne[1] should be 8, got %lld", (long long)w.ne[1]);
    TEST_ASSERT(w.nbytes == 128 * sizeof(ggml_fp16_t), "convert_f32_tensor", "Wrong nbytes");
    TEST_ASSERT(w.scales == NULL, "convert_f32_tensor", "FP16 should have no scales");

    // Verify accuracy
    ggml_fp16_t * fp16 = (ggml_fp16_t *)w.data;
    float max_err = 0;
    for (int i = 0; i < 128; i++) {
        float recovered = ggml_fp16_to_fp32(fp16[i]);
        float err = fabsf(data[i] - recovered);
        if (err > max_err) max_err = err;
    }
    printf("  F32→FP16 max error: %.6f\n", max_err);
    TEST_ASSERT(max_err < 0.001f, "convert_f32_tensor", "Error too large");

    free(w.data);
    ggml_free(ctx);
    TEST_PASS("convert_f32_tensor");
}

// ===========================================================================
// Test: Split along dim=0 (columns)
// ===========================================================================
static void test_split_dim0(void) {
    TEST_BEGIN("split_dim0");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 4 rows × 8 cols, F32
    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    float * data = (float *)t->data;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            data[r * 8 + c] = (float)(r * 10 + c);

    // Split: take columns [2, 6) → 4 columns
    struct ane_converted_weight w = ane_split_and_convert(t, 0, 2, 4);
    TEST_ASSERT(w.data != NULL, "split_dim0", "Split returned NULL");
    TEST_ASSERT(w.ne[0] == 4, "split_dim0", "ne[0] should be 4, got %lld", (long long)w.ne[0]);
    TEST_ASSERT(w.ne[1] == 4, "split_dim0", "ne[1] should be 4, got %lld", (long long)w.ne[1]);
    TEST_ASSERT(w.fmt == ANE_FMT_FP16, "split_dim0", "F32 source should give FP16");

    // Verify values: row 0 should have [2, 3, 4, 5], row 1 [12, 13, 14, 15], etc.
    ggml_fp16_t * fp16 = (ggml_fp16_t *)w.data;
    float max_err = 0;
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            float expected = (float)(r * 10 + (c + 2));
            float got = ggml_fp16_to_fp32(fp16[r * 4 + c]);
            float err = fabsf(got - expected);
            if (err > max_err) max_err = err;
        }
    }
    printf("  Split dim0 max error: %.6f\n", max_err);
    TEST_ASSERT(max_err < 0.01f, "split_dim0", "Values wrong after split");

    free(w.data);
    ggml_free(ctx);
    TEST_PASS("split_dim0");
}

// ===========================================================================
// Test: Split along dim=1 (rows)
// ===========================================================================
static void test_split_dim1(void) {
    TEST_BEGIN("split_dim1");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // 4 rows × 8 cols, F32
    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    float * data = (float *)t->data;
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 8; c++)
            data[r * 8 + c] = (float)(r * 10 + c);

    // Split: take rows [1, 3) → 2 rows
    struct ane_converted_weight w = ane_split_and_convert(t, 1, 1, 2);
    TEST_ASSERT(w.data != NULL, "split_dim1", "Split returned NULL");
    TEST_ASSERT(w.ne[0] == 8, "split_dim1", "ne[0] should be 8, got %lld", (long long)w.ne[0]);
    TEST_ASSERT(w.ne[1] == 2, "split_dim1", "ne[1] should be 2, got %lld", (long long)w.ne[1]);

    // Verify: row 0 should be [10,11,12,...,17], row 1 should be [20,21,...,27]
    ggml_fp16_t * fp16 = (ggml_fp16_t *)w.data;
    float max_err = 0;
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 8; c++) {
            float expected = (float)((r + 1) * 10 + c);
            float got = ggml_fp16_to_fp32(fp16[r * 8 + c]);
            float err = fabsf(got - expected);
            if (err > max_err) max_err = err;
        }
    }
    printf("  Split dim1 max error: %.6f\n", max_err);
    TEST_ASSERT(max_err < 0.01f, "split_dim1", "Values wrong after split");

    free(w.data);
    ggml_free(ctx);
    TEST_PASS("split_dim1");
}

// ===========================================================================
// Test: Split preserves full tensor when taking entire range
// ===========================================================================
static void test_split_full_range(void) {
    TEST_BEGIN("split_full_range");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16, 8);
    float * data = (float *)t->data;
    for (int i = 0; i < 128; i++) data[i] = (float)i * 0.1f;

    // Full conversion
    struct ane_converted_weight w_full = ane_convert_tensor(t);
    // Split with full range
    struct ane_converted_weight w_split = ane_split_and_convert(t, 0, 0, 16);

    TEST_ASSERT(w_full.data != NULL && w_split.data != NULL, "split_full_range", "Conversion failed");
    TEST_ASSERT(w_full.nbytes == w_split.nbytes, "split_full_range", "Size mismatch");
    TEST_ASSERT(memcmp(w_full.data, w_split.data, w_full.nbytes) == 0, "split_full_range",
                "Full-range split should equal full convert");

    free(w_full.data);
    free(w_split.data);
    ggml_free(ctx);
    TEST_PASS("split_full_range");
}

// ===========================================================================
// Test: GPU + ANE halves reconstruct original (additive split)
// ===========================================================================
static void test_split_reconstruction(void) {
    TEST_BEGIN("split_reconstruction");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    int cols = 16, rows = 4;
    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    float * data = (float *)t->data;
    for (int i = 0; i < rows * cols; i++) data[i] = (float)i * 0.05f;

    // Split at column 10: GPU gets [0,10), ANE gets [10,16)
    int split_at = 10;
    struct ane_converted_weight gpu_half = ane_split_and_convert(t, 0, 0, split_at);
    struct ane_converted_weight ane_half = ane_split_and_convert(t, 0, split_at, cols - split_at);

    TEST_ASSERT(gpu_half.data && ane_half.data, "split_reconstruction", "Split failed");
    TEST_ASSERT(gpu_half.ne[0] == split_at, "split_reconstruction", "GPU half wrong cols");
    TEST_ASSERT(ane_half.ne[0] == cols - split_at, "split_reconstruction", "ANE half wrong cols");

    // Reconstruct by concatenating columns and check against original
    float max_err = 0;
    ggml_fp16_t * gpu_fp16 = (ggml_fp16_t *)gpu_half.data;
    ggml_fp16_t * ane_fp16 = (ggml_fp16_t *)ane_half.data;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float original = data[r * cols + c];
            float recovered;
            if (c < split_at) {
                recovered = ggml_fp16_to_fp32(gpu_fp16[r * split_at + c]);
            } else {
                recovered = ggml_fp16_to_fp32(ane_fp16[r * (cols - split_at) + (c - split_at)]);
            }
            float err = fabsf(original - recovered);
            if (err > max_err) max_err = err;
        }
    }

    printf("  Reconstruction max error: %.6f\n", max_err);
    TEST_ASSERT(max_err < 0.001f, "split_reconstruction", "Halves don't reconstruct original");

    free(gpu_half.data);
    free(ane_half.data);
    ggml_free(ctx);
    TEST_PASS("split_reconstruction");
}

// ===========================================================================
// Test: Out of range split returns NULL
// ===========================================================================
static void test_split_bounds_check(void) {
    TEST_BEGIN("split_bounds_check");

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);

    // Out of range: start + count > dim_size
    struct ane_converted_weight w = ane_split_and_convert(t, 0, 6, 4);
    TEST_ASSERT(w.data == NULL, "split_bounds_check", "Should fail for out-of-range split");

    // Invalid dim
    w = ane_split_and_convert(t, 2, 0, 1);
    TEST_ASSERT(w.data == NULL, "split_bounds_check", "Should fail for dim=2");

    // NULL tensor
    w = ane_split_and_convert(NULL, 0, 0, 1);
    TEST_ASSERT(w.data == NULL, "split_bounds_check", "Should fail for NULL tensor");

    ggml_free(ctx);
    TEST_PASS("split_bounds_check");
}

// ===========================================================================
// Test: Cache hash
// ===========================================================================
static void test_cache_hash(void) {
    TEST_BEGIN("cache_hash");

    // Non-existent file should return 0
    uint64_t h = ane_compute_model_hash("/nonexistent/file.gguf");
    TEST_ASSERT(h == 0, "cache_hash", "Non-existent file should hash to 0");

    // NULL path should return 0
    h = ane_compute_model_hash(NULL);
    TEST_ASSERT(h == 0, "cache_hash", "NULL path should hash to 0");

    // Create a temp file and hash it
    const char * tmp = "/tmp/ane_test_hash.bin";
    FILE * f = fopen(tmp, "wb");
    if (f) {
        uint8_t buf[128];
        for (int i = 0; i < 128; i++) buf[i] = (uint8_t)(i * 7);
        fwrite(buf, 1, 128, f);
        fclose(f);

        uint64_t h1 = ane_compute_model_hash(tmp);
        uint64_t h2 = ane_compute_model_hash(tmp);
        TEST_ASSERT(h1 != 0, "cache_hash", "Valid file should have non-zero hash");
        TEST_ASSERT(h1 == h2, "cache_hash", "Same file should hash identically");
        printf("  Hash: 0x%016llx\n", (unsigned long long)h1);

        // Modify file → different hash
        f = fopen(tmp, "wb");
        buf[0] = 0xFF;
        fwrite(buf, 1, 128, f);
        fclose(f);

        uint64_t h3 = ane_compute_model_hash(tmp);
        TEST_ASSERT(h3 != h1, "cache_hash", "Modified file should have different hash");

        remove(tmp);
    } else {
        printf("  Warning: couldn't create temp file, skipping file hash test\n");
    }

    TEST_PASS("cache_hash");
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("=== ANE Weight Conversion Tests (Phase 2) ===\n\n");

    test_type_mapping();
    test_int8_quantization();
    test_fp16_conversion();
    test_dequantize_f32();
    test_convert_f32_tensor();
    test_split_dim0();
    test_split_dim1();
    test_split_full_range();
    test_split_reconstruction();
    test_split_bounds_check();
    test_cache_hash();

    printf("=== Results: %d passed, %d failed ===\n",
           g_tests_passed, g_tests_failed);
    return g_tests_failed > 0 ? 1 : 0;
}
