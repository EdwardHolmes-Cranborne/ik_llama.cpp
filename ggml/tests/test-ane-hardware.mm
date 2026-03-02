// test-ane-hardware.mm — Phase 0: M3 ANE Hardware Characterization
// Tests API availability, compile/eval via coremltools, TFLOPS measurement,
// dispatch latency, Metal+ANE concurrency.
//
// Requires: macOS 14+, Apple Silicon, Python 3.12+ with coremltools installed.
// Run WITHOUT sandbox for ANE access.

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <math.h>
#import <string.h>

#include "ggml-ane.h"

static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_tests_skipped = 0;

#define TEST_BEGIN(name) do { \
    printf("\n=== TEST: %s ===\n", name); \
} while(0)

#define TEST_PASS(name) do { \
    printf("  PASS: %s\n", name); \
    g_tests_passed++; \
} while(0)

#define TEST_FAIL(name, ...) do { \
    printf("  FAIL: %s - ", name); \
    printf(__VA_ARGS__); \
    printf("\n"); \
    g_tests_failed++; \
} while(0)

#define TEST_SKIP(name, reason) do { \
    printf("  SKIP: %s - %s\n", name, reason); \
    g_tests_skipped++; \
} while(0)

// ===========================================================================
// Helper: compile a small test conv model via coremltools
// ===========================================================================
static ggml_ane_kernel_t compile_test_model(int in_ch, int out_ch, int spatial,
                                             const char * cache_dir) {
    size_t n_w = (size_t)out_ch * in_ch;
    float *weights = (float*)calloc(n_w, sizeof(float));
    // Identity-like weights (diagonal = 1) for correctness tests
    for (int i = 0; i < (in_ch < out_ch ? in_ch : out_ch); i++) {
        weights[i * in_ch + i] = 1.0f;
    }

    ggml_ane_kernel_t k = ggml_ane_compile(
        NULL,         // auto-detect python
        "conv",
        in_ch, out_ch, spatial,
        weights, n_w * sizeof(float),
        cache_dir);

    free(weights);
    return k;
}

// ===========================================================================
// Test: ane_api_available
// ===========================================================================
static void test_ane_api_available(void) {
    TEST_BEGIN("ane_api_available");

    bool avail = ggml_backend_ane_supported();
    printf("  ggml_backend_ane_supported() = %s\n", avail ? "true" : "false");
    if (avail) {
        TEST_PASS("ane_api_available");
    } else {
        TEST_FAIL("ane_api_available",
                  "ANE not available on this system (need macOS 14+ Apple Silicon)");
    }
}

// ===========================================================================
// Test: ane_compile_and_eval
// ===========================================================================
static void test_ane_compile_and_eval(const char * cache_dir) {
    TEST_BEGIN("ane_compile_and_eval");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_compile_and_eval", "ANE not available");
        return;
    }

    int ch = 64, sp = 64;
    ggml_ane_kernel_t k = compile_test_model(ch, ch, sp, cache_dir);
    if (!k) {
        TEST_FAIL("ane_compile_and_eval",
                  "Model compile failed (is coremltools installed? python3 -c 'import coremltools')");
        return;
    }

    // Create input: channel c gets value (c+1)*0.01
    size_t n_elem = (size_t)ch * sp;
    __fp16 *input = (__fp16*)malloc(n_elem * sizeof(__fp16));
    for (int c = 0; c < ch; c++) {
        for (int s = 0; s < sp; s++) {
            input[c * sp + s] = (__fp16)((float)(c + 1) * 0.01f);
        }
    }
    ggml_ane_write_input(k, 0, input, n_elem * sizeof(__fp16));

    bool ok = ggml_ane_eval(k);
    if (!ok) {
        TEST_FAIL("ane_compile_and_eval", "ANE eval failed");
        free(input);
        ggml_ane_free(k);
        return;
    }

    // Read output — with identity weights, output should approximate input
    __fp16 *output = (__fp16*)malloc(n_elem * sizeof(__fp16));
    ggml_ane_read_output_after_eval(k, 0, output, n_elem * sizeof(__fp16));

    float max_err = 0;
    for (int c = 0; c < ch; c++) {
        for (int s = 0; s < sp; s++) {
            float expected = (float)(c + 1) * 0.01f;
            float got = (float)output[c * sp + s];
            float err = fabsf(got - expected);
            if (err > max_err) max_err = err;
        }
    }

    free(input);
    free(output);
    ggml_ane_free(k);

    printf("  Identity matmul max error: %.6f\n", max_err);
    if (max_err < 0.01f) {
        TEST_PASS("ane_compile_and_eval");
    } else {
        TEST_FAIL("ane_compile_and_eval", "max error %.6f > 0.01", max_err);
    }
}

// ===========================================================================
// Test: ane_peak_tflops
// ===========================================================================
static void test_ane_peak_tflops(const char * cache_dir) {
    TEST_BEGIN("ane_peak_tflops");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_peak_tflops", "ANE not available");
        return;
    }

    // Test a large conv for peak throughput
    int ch = 4096, sp = 64;
    size_t n_w = (size_t)ch * ch;
    float *weights = (float*)malloc(n_w * sizeof(float));
    for (size_t i = 0; i < n_w; i++) weights[i] = 0.001f;

    ggml_ane_kernel_t k = ggml_ane_compile(NULL, "conv", ch, ch, sp,
                                            weights, n_w * sizeof(float), cache_dir);
    free(weights);

    if (!k) {
        TEST_SKIP("ane_peak_tflops", "Model compile failed (coremltools not available)");
        return;
    }

    float tflops = ggml_ane_measure_tflops(k, ch, sp, 50);
    printf("  ch=%d sp=%d -> %.3f TFLOPS\n", ch, sp, tflops);

    ggml_ane_free(k);

    if (tflops > 1.0f) {
        TEST_PASS("ane_peak_tflops");
    } else {
        TEST_FAIL("ane_peak_tflops", "peak %.3f TFLOPS too low (expected > 1.0)", tflops);
    }
}

// ===========================================================================
// Test: ane_dispatch_latency
// ===========================================================================
static void test_ane_dispatch_latency(const char * cache_dir) {
    TEST_BEGIN("ane_dispatch_latency");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_dispatch_latency", "ANE not available");
        return;
    }

    // Use a small model for latency measurement
    int ch = 64, sp = 64;
    ggml_ane_kernel_t k = compile_test_model(ch, ch, sp, cache_dir);
    if (!k) {
        TEST_SKIP("ane_dispatch_latency", "Model compile failed");
        return;
    }

    float mean_ms = ggml_ane_measure_dispatch_latency(k, 1000);
    ggml_ane_free(k);

    printf("  Mean dispatch latency: %.4f ms (over 1000 iterations)\n", mean_ms);

    if (mean_ms < 0) {
        TEST_FAIL("ane_dispatch_latency", "measurement failed");
    } else if (mean_ms < 0.5f) {
        TEST_PASS("ane_dispatch_latency");
    } else {
        TEST_FAIL("ane_dispatch_latency", "%.4f ms > 0.5 ms target", mean_ms);
    }
}

// ===========================================================================
// Test: ane_concurrent_metal [CRITICAL GO/NO-GO]
// ===========================================================================
static void test_ane_concurrent_metal(const char * cache_dir) {
    TEST_BEGIN("ane_concurrent_metal [GO/NO-GO]");

    if (!ggml_backend_ane_supported()) {
        TEST_SKIP("ane_concurrent_metal", "ANE not available");
        return;
    }

    // Use a medium-sized model so ANE has real work to do
    int ch = 512, sp = 64;
    size_t n_w = (size_t)ch * ch;
    float *weights = (float*)malloc(n_w * sizeof(float));
    for (size_t i = 0; i < n_w; i++) weights[i] = 0.001f;

    ggml_ane_kernel_t k = ggml_ane_compile(NULL, "conv", ch, ch, sp,
                                            weights, n_w * sizeof(float), cache_dir);
    free(weights);

    if (!k) {
        TEST_SKIP("ane_concurrent_metal", "Model compile failed");
        return;
    }

    float ratio = ggml_ane_test_concurrent_metal(k);
    ggml_ane_free(k);

    if (ratio < 0) {
        TEST_FAIL("ane_concurrent_metal", "test failed to run");
    } else if (ratio < 1.1f) {
        printf("  CONCURRENT: wall_time/max(metal,ane) = %.3f < 1.1\n", ratio);
        printf("  *** GO: Metal+ANE can execute concurrently! ***\n");
        TEST_PASS("ane_concurrent_metal");
    } else {
        printf("  SERIALIZED: wall_time/max(metal,ane) = %.3f >= 1.1\n", ratio);
        printf("  *** REVISE: ANE suitable only for idle-window work ***\n");
        TEST_FAIL("ane_concurrent_metal",
                  "Metal+ANE appear serialized (ratio=%.3f)", ratio);
    }
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=================================================\n");
        printf("M3 ANE Hardware Characterization Test Suite\n");
        printf("=================================================\n");

        bool quick = false;
        const char *cache_dir = "/tmp/ggml_ane_test_cache";
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--quick") == 0) quick = true;
        }

        // Always run API check
        test_ane_api_available();
        test_ane_compile_and_eval(cache_dir);
        test_ane_dispatch_latency(cache_dir);
        test_ane_concurrent_metal(cache_dir);

        if (!quick) {
            test_ane_peak_tflops(cache_dir);
        } else {
            printf("\n[--quick mode: skipping TFLOPS sweep]\n");
        }

        printf("\n=================================================\n");
        printf("Results: %d passed, %d failed, %d skipped\n",
               g_tests_passed, g_tests_failed, g_tests_skipped);
        printf("=================================================\n");

        return g_tests_failed > 0 ? 1 : 0;
    }
}
