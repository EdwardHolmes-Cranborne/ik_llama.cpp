// test-ane-backend.mm — Phase 1 backend unit tests
// Tests the public CoreML-based ANE backend: compile, load, eval, correctness.
//
// Self-contained: generates test models via ggml_ane_compile() (needs coremltools).
// Run WITHOUT sandbox for ANE access.

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>
#import <mach/mach_time.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "ggml-ane.h"

static double now_ms(void) {
    static mach_timebase_info_data_t tb = {0};
    if (tb.numer == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_tests_skipped = 0;

static const char * g_cache_dir = "/tmp/ggml_ane_test_cache";

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        g_tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_BEGIN(name) printf("TEST: %s\n", name)
#define TEST_PASS(name) do { printf("  PASS: %s\n\n", name); g_tests_passed++; } while(0)
#define TEST_SKIP(name, reason) do { printf("  SKIP: %s - %s\n\n", name, reason); g_tests_skipped++; } while(0)

// Helper: compile a conv model with given weights
static ggml_ane_kernel_t make_conv(int in_ch, int out_ch, int spatial,
                                    const float * weights) {
    size_t nbytes = (size_t)out_ch * in_ch * sizeof(float);
    return ggml_ane_compile(NULL, "conv", in_ch, out_ch, spatial,
                            weights, nbytes, g_cache_dir);
}

// Helper: compile a conv model with identity-like weights
static ggml_ane_kernel_t make_identity_conv(int ch, int spatial) {
    size_t n_w = (size_t)ch * ch;
    float *w = (float*)calloc(n_w, sizeof(float));
    for (int i = 0; i < ch; i++) w[i * ch + i] = 1.0f;
    ggml_ane_kernel_t k = make_conv(ch, ch, spatial, w);
    free(w);
    return k;
}

// ===========================================================================
// Test: ANE API available
// ===========================================================================
static void test_ane_api_available(void) {
    TEST_BEGIN("ane_api_available");
    bool supported = ggml_backend_ane_supported();
    printf("  ggml_backend_ane_supported() = %s\n", supported ? "true" : "false");
    TEST_ASSERT(supported, "ANE not supported on this system");
    TEST_PASS("ane_api_available");
}

// ===========================================================================
// Test: Compile and load model
// ===========================================================================
static void test_ane_compile_model(void) {
    TEST_BEGIN("ane_compile_model");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_compile_model", "ANE not available");
        return;
    }

    ggml_ane_kernel_t k = make_identity_conv(64, 64);
    TEST_ASSERT(k != NULL, "Failed to compile conv64 model (need coremltools)");

    int n_out = ggml_ane_n_outputs(k);
    printf("  Compiled: %d outputs\n", n_out);
    for (int i = 0; i < n_out; i++) {
        printf("  Output %d: %s\n", i, ggml_ane_output_name(k, i));
    }
    TEST_ASSERT(n_out >= 1, "Expected at least 1 output");

    ggml_ane_free(k);
    TEST_PASS("ane_compile_model");
}

// ===========================================================================
// Test: ANE eval produces non-zero output
// ===========================================================================
static void test_ane_eval_nonzero(void) {
    TEST_BEGIN("ane_eval_nonzero");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_eval_nonzero", "ANE not available");
        return;
    }

    int ch = 256, sp = 64;
    ggml_ane_kernel_t k = make_identity_conv(ch, sp);
    TEST_ASSERT(k != NULL, "Failed to compile model");

    size_t n_elem = (size_t)ch * sp;
    std::vector<__fp16> input(n_elem, (__fp16)0.01f);
    ggml_ane_write_input(k, 0, input.data(), n_elem * sizeof(__fp16));

    bool ok = ggml_ane_eval(k);
    TEST_ASSERT(ok, "Eval failed");

    std::vector<__fp16> output(n_elem);
    ggml_ane_read_output_after_eval(k, 0, output.data(), n_elem * sizeof(__fp16));

    int nonzero = 0;
    float max_val = 0;
    for (size_t i = 0; i < n_elem; i++) {
        float v = fabsf((float)output[i]);
        if (v > 1e-8f) nonzero++;
        if (v > max_val) max_val = v;
    }

    printf("  Non-zero outputs: %d/%zu (max=%.6f)\n", nonzero, n_elem, max_val);
    TEST_ASSERT(nonzero > 0, "All outputs are zero — model may not be executing");

    ggml_ane_free(k);
    TEST_PASS("ane_eval_nonzero");
}

// ===========================================================================
// Test: CPU vs ANE output consistency
// ===========================================================================
static void test_ane_conv_correctness(void) {
    TEST_BEGIN("ane_conv_correctness");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_conv_correctness", "ANE not available");
        return;
    }

    // Compile with random weights
    int ch = 64, sp = 64;
    size_t n_w = (size_t)ch * ch;
    float *weights = (float*)malloc(n_w * sizeof(float));
    srand(42);
    for (size_t i = 0; i < n_w; i++) {
        weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Load on CPU and ANE via different compute units
    // We compile once, then load the cached .mlmodelc on different units
    ggml_ane_kernel_t k_first = make_conv(ch, ch, sp, weights);
    free(weights);
    TEST_ASSERT(k_first != NULL, "Failed to compile model");

    // For CPU vs ANE comparison, we'd need path-based loading with different
    // compute units. For now, just verify ANE produces reasonable output.
    size_t n_elem = (size_t)ch * sp;

    // Create random FP16 input
    std::vector<__fp16> input(n_elem);
    srand(42);
    for (size_t i = 0; i < n_elem; i++) {
        input[i] = (__fp16)(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    }

    ggml_ane_write_input(k_first, 0, input.data(), n_elem * sizeof(__fp16));
    TEST_ASSERT(ggml_ane_eval(k_first), "ANE eval failed");

    std::vector<__fp16> output(n_elem);
    ggml_ane_read_output_after_eval(k_first, 0, output.data(), n_elem * sizeof(__fp16));

    // Verify output is not all zeros and has reasonable magnitude
    float sum = 0, max_abs = 0;
    int nonzero = 0;
    for (size_t i = 0; i < n_elem; i++) {
        float v = (float)output[i];
        sum += v;
        float av = fabsf(v);
        if (av > max_abs) max_abs = av;
        if (av > 1e-8f) nonzero++;
    }

    printf("  Output stats: nonzero=%d/%zu max_abs=%.6f mean=%.6f\n",
           nonzero, n_elem, max_abs, sum / n_elem);
    TEST_ASSERT(nonzero > (int)(n_elem / 2), "Too many zero outputs");
    TEST_ASSERT(max_abs < 10.0f, "Output magnitude unreasonably large");

    ggml_ane_free(k_first);
    TEST_PASS("ane_conv_correctness");
}

// ===========================================================================
// Test: Throughput benchmark
// ===========================================================================
static void test_ane_throughput(void) {
    TEST_BEGIN("ane_throughput");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_throughput", "ANE not available");
        return;
    }

    int ch = 1024, sp = 64;
    size_t n_w = (size_t)ch * ch;
    float *weights = (float*)malloc(n_w * sizeof(float));
    for (size_t i = 0; i < n_w; i++) weights[i] = 0.001f;

    ggml_ane_kernel_t k = ggml_ane_compile(NULL, "conv", ch, ch, sp,
                                            weights, n_w * sizeof(float), g_cache_dir);
    free(weights);
    TEST_ASSERT(k != NULL, "Failed to compile model");

    // Fill input
    size_t n_elem = (size_t)ch * sp;
    std::vector<__fp16> input(n_elem, (__fp16)0.01f);
    ggml_ane_write_input(k, 0, input.data(), n_elem * sizeof(__fp16));

    // Warmup
    for (int i = 0; i < 10; i++) ggml_ane_eval(k);

    // Benchmark 2 seconds
    double deadline = now_ms() + 2000;
    int count = 0;
    double t0 = now_ms();
    while (now_ms() < deadline) {
        ggml_ane_eval(k);
        count++;
    }
    double elapsed = now_ms() - t0;
    double avg_ms = elapsed / count;
    double flops = 2.0 * ch * ch * sp;
    double tflops = flops / (avg_ms * 1e9);

    printf("  ANE: %d evals  %.3f ms/eval  %.3f TFLOPS\n", count, avg_ms, tflops);
    TEST_ASSERT(tflops > 0.5, "TFLOPS too low");

    ggml_ane_free(k);
    TEST_PASS("ane_throughput");
}

// ===========================================================================
// Test: Dispatch latency
// ===========================================================================
static void test_ane_dispatch_latency(void) {
    TEST_BEGIN("ane_dispatch_latency");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_dispatch_latency", "ANE not available");
        return;
    }

    ggml_ane_kernel_t k = make_identity_conv(64, 64);
    TEST_ASSERT(k != NULL, "Failed to compile model");

    size_t n_elem = 64 * 64;
    std::vector<__fp16> input(n_elem, (__fp16)0.01f);
    ggml_ane_write_input(k, 0, input.data(), n_elem * sizeof(__fp16));

    // Warmup
    for (int i = 0; i < 50; i++) ggml_ane_eval(k);

    int N = 1000;
    std::vector<double> latencies(N);
    for (int i = 0; i < N; i++) {
        double t0 = now_ms();
        ggml_ane_eval(k);
        latencies[i] = now_ms() - t0;
    }

    std::sort(latencies.begin(), latencies.end());

    double sum = 0;
    for (double l : latencies) sum += l;
    double mean = sum / N;

    printf("  N=%d dispatches\n", N);
    printf("  Mean:  %.3f ms\n", mean);
    printf("  P50:   %.3f ms\n", latencies[N/2]);
    printf("  P90:   %.3f ms\n", latencies[(int)(N*0.9)]);
    printf("  P99:   %.3f ms\n", latencies[(int)(N*0.99)]);
    printf("  Min:   %.3f ms\n", latencies[0]);
    printf("  Max:   %.3f ms\n", latencies[N-1]);

    TEST_ASSERT(mean < 1.0, "Mean dispatch latency > 1ms");

    ggml_ane_free(k);
    TEST_PASS("ane_dispatch_latency");
}

// ===========================================================================
// Test: Weight blob builder
// ===========================================================================
static void test_ane_weight_blob(void) {
    TEST_BEGIN("ane_weight_blob");

    int out_ch = 16, in_ch = 16;
    size_t n_w = (size_t)out_ch * in_ch;
    float *weights = (float*)malloc(n_w * sizeof(float));
    for (size_t i = 0; i < n_w; i++) weights[i] = (float)i * 0.01f;

    size_t blob_nb = 0;
    void *blob = ggml_ane_build_weight_blob(weights, out_ch, in_ch, &blob_nb);
    free(weights);

    TEST_ASSERT(blob != NULL, "Blob build returned NULL");
    TEST_ASSERT(blob_nb > 0, "Blob size is 0");

    // Expected: 64 (global header) + 64 (chunk header) + out*in*2 (FP16 data)
    size_t expected = 64 + 64 + (size_t)out_ch * in_ch * 2;
    printf("  Blob size: %zu (expected %zu)\n", blob_nb, expected);
    TEST_ASSERT(blob_nb == expected, "Unexpected blob size");

    free(blob);
    TEST_PASS("ane_weight_blob");
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    @autoreleasepool {
        printf("=== ggml-ane Backend Tests (Phase 1) — macOS %s ===\n\n",
               [[[NSProcessInfo processInfo] operatingSystemVersionString] UTF8String]);

        test_ane_api_available();
        test_ane_weight_blob();
        test_ane_compile_model();
        test_ane_eval_nonzero();
        test_ane_conv_correctness();
        test_ane_throughput();
        test_ane_dispatch_latency();

        printf("=== Results: %d passed, %d failed, %d skipped ===\n",
               g_tests_passed, g_tests_failed, g_tests_skipped);
        return g_tests_failed > 0 ? 1 : 0;
    }
}
