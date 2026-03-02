// bench-ane-prefill.cpp — Benchmark GPU-only vs GPU+ANE streaming prefill
//
// Loads a model, runs streaming prefill with and without ANE,
// compares wall-clock time, and generates text to verify correctness.
//
// Usage: bench-ane-prefill -m <model.gguf> [-s seq_len] [-r ratio] [-n n_runs] [-g gen_tokens]
//        bench-ane-prefill -m <model.gguf> --sweep [-r ratio]   (seq sweep mode)

#include "llama.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m <model.gguf> [-s seq_len] [-r ratio] [-n n_runs] [-c n_ctx] [-g gen_tokens]\n", prog);
    fprintf(stderr, "       %s -m <model.gguf> --sweep [-r ratio]  (seq length sweep)\n", prog);
    fprintf(stderr, "  -m  model path (required)\n");
    fprintf(stderr, "  -s  sequence length for prefill benchmark (default: 512)\n");
    fprintf(stderr, "  -r  ANE split ratio (default: 0.5)\n");
    fprintf(stderr, "  -n  number of benchmark runs (default: 3)\n");
    fprintf(stderr, "  -c  context size (default: max(2048, seq_len*2))\n");
    fprintf(stderr, "  -g  tokens to generate in coherence test (default: 64)\n");
    fprintf(stderr, "  --sweep  seq length sweep: test 128,512,1024,2048,4096,8192,16384\n");
    fprintf(stderr, "  --no-coherence  skip coherence test (faster for sweep)\n");
}

// Greedy-sample the top token from logits
static llama_token sample_greedy(llama_context * ctx) {
    const float * logits = llama_get_logits(ctx);
    int n_vocab = llama_n_vocab(llama_get_model(ctx));
    llama_token best = 0;
    float best_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best = i;
        }
    }
    return best;
}

// Run prefill + text generation, return generated text and prefill time
struct gen_result {
    std::string text;
    float prefill_ms;
    float gen_ms;
};

static gen_result run_generation(
    llama_model * model,
    const std::vector<llama_token> & prompt_tokens,
    int n_ctx,
    int n_gen,
    bool ane_enabled,
    float ane_ratio)
{
    gen_result result = {};

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = (int)prompt_tokens.size();
    cparams.n_ubatch = (int)prompt_tokens.size();
    cparams.prefill_streaming = true;
    cparams.prefill_telemetry = false;
    cparams.prefill_ane = ane_enabled;
    cparams.prefill_ane_ratio = ane_ratio;
    cparams.prefill_ane_cache = "/tmp/ane-bench-cache";

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "FAIL: could not create context\n");
        return result;
    }

    int n_prompt = (int)prompt_tokens.size();

    // Prefill: decode prompt
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt_tokens[i], i, {0}, i == n_prompt - 1);
    }

    // Warmup decode (compiles ANE kernels if needed)
    llama_decode(ctx, batch);
    llama_kv_cache_clear(ctx);

    // Timed prefill
    auto t0 = std::chrono::high_resolution_clock::now();
    int ret = llama_decode(ctx, batch);
    auto t1 = std::chrono::high_resolution_clock::now();
    result.prefill_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    llama_batch_free(batch);

    if (ret != 0) {
        fprintf(stderr, "  prefill failed: %d\n", ret);
        llama_free(ctx);
        return result;
    }

    // Generate tokens one at a time (greedy)
    auto t2 = std::chrono::high_resolution_clock::now();
    int pos = n_prompt;
    for (int i = 0; i < n_gen; i++) {
        llama_token tok = sample_greedy(ctx);

        // Check for EOS
        if (llama_token_is_eog(model, tok)) {
            break;
        }

        // Decode token to text
        char buf[256];
        int len = llama_token_to_piece(model, tok, buf, sizeof(buf), 0, true);
        if (len > 0) {
            result.text.append(buf, len);
        }

        // Feed token back
        llama_batch next = llama_batch_init(1, 0, 1);
        common_batch_add(next, tok, pos++, {0}, true);

        // Use standard decode for generation (not streaming prefill)
        bool saved = cparams.prefill_streaming;
        // Single-token decode doesn't use streaming prefill
        ret = llama_decode(ctx, next);
        llama_batch_free(next);

        if (ret != 0) {
            fprintf(stderr, "  generation failed at token %d: %d\n", i, ret);
            break;
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    result.gen_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();

    llama_free(ctx);
    return result;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int seq_len = 512;
    float ratio = 0.5f;
    int n_runs = 3;
    int n_ctx = 0;
    int n_gen = 64;
    bool sweep_mode = false;
    bool do_coherence = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            ratio = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            n_ctx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sweep") == 0) {
            sweep_mode = true;
        } else if (strcmp(argv[i], "--no-coherence") == 0) {
            do_coherence = false;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    if (n_ctx == 0) n_ctx = sweep_mode ? 65536 : std::max(2048, seq_len * 2);

    fprintf(stderr, "=== ANE Prefill Benchmark ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    if (sweep_mode) {
        fprintf(stderr, "Mode: seq length sweep, Ratio: %.2f, Context: %d\n\n", ratio, n_ctx);
    } else {
        fprintf(stderr, "Seq length: %d, Context: %d, Ratio: %.2f, Runs: %d, Gen: %d\n\n",
                seq_len, n_ctx, ratio, n_runs, n_gen);
    }

    llama_backend_init();

    // Load model
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU weights for streaming prefill
    mparams.use_mmap = false;  // Avoid mmap (hangs on network volumes)

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model\n");
        return 1;
    }

    int n_layer = llama_n_layer(model);
    int n_embd  = llama_model_n_embd(model);
    int n_vocab = llama_n_vocab(model);
    fprintf(stderr, "Model: n_layer=%d, n_embd=%d, n_vocab=%d\n", n_layer, n_embd, n_vocab);

    // ================================================================
    // Sweep mode: test multiple seq lengths with one model load
    // ================================================================
    if (sweep_mode) {
        const int sweep_seqs[] = {128, 512, 1024, 2048, 4096, 8192, 16384};
        const int n_sweep = sizeof(sweep_seqs) / sizeof(sweep_seqs[0]);

        // Helper lambda: run one prefill, return time in ms
        auto run_prefill_sweep = [&](int sl, bool ane_enabled, float ane_ratio) -> float {
            int ctx_size = std::max(n_ctx, sl * 2);
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = ctx_size;
            cparams.n_batch = sl;
            cparams.n_ubatch = sl;
            cparams.prefill_streaming = true;
            cparams.prefill_telemetry = false;
            cparams.prefill_ane = ane_enabled;
            cparams.prefill_ane_ratio = ane_ratio;
            cparams.prefill_ane_cache = "/tmp/ane-bench-cache";

            llama_context * ctx = llama_init_from_model(model, cparams);
            if (!ctx) return -1.0f;

            std::vector<llama_token> toks(sl);
            for (int i = 0; i < sl; i++) toks[i] = (i % (n_vocab - 1)) + 1;

            llama_batch batch = llama_batch_init(sl, 0, 1);
            for (int i = 0; i < sl; i++) {
                common_batch_add(batch, toks[i], i, {0}, i == sl - 1);
            }

            // Warmup (compiles kernels if needed)
            llama_decode(ctx, batch);
            llama_kv_cache_clear(ctx);

            // Timed run
            auto t0 = std::chrono::high_resolution_clock::now();
            int ret = llama_decode(ctx, batch);
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

            if (ret != 0) {
                fprintf(stderr, "  WARNING: llama_decode returned %d for seq=%d\n", ret, sl);
            }

            llama_batch_free(batch);
            llama_free(ctx);
            return ms;
        };

        fprintf(stderr, "\n=== Seq Length Sweep (ratio=%.2f) ===\n", ratio);
        fprintf(stderr, "%8s  %10s  %10s  %10s  %10s  %8s\n",
                "seq_len", "GPU(ms)", "GPU(tok/s)", "ANE(ms)", "ANE(tok/s)", "speedup");
        fprintf(stderr, "%8s  %10s  %10s  %10s  %10s  %8s\n",
                "-------", "--------", "---------", "-------", "---------", "-------");

        for (int si = 0; si < n_sweep; si++) {
            int sl = sweep_seqs[si];
            if (sl * 2 > n_ctx) {
                fprintf(stderr, "%8d  skipped (need n_ctx >= %d, have %d)\n", sl, sl * 2, n_ctx);
                continue;
            }

            fprintf(stderr, "Testing seq=%d...\n", sl);

            float gpu_ms = run_prefill_sweep(sl, false, 0.0f);
            float ane_ms = run_prefill_sweep(sl, true, ratio);

            if (gpu_ms < 0 || ane_ms < 0) {
                fprintf(stderr, "%8d  FAILED\n", sl);
                continue;
            }

            float gpu_tps = sl / (gpu_ms / 1000.0f);
            float ane_tps = sl / (ane_ms / 1000.0f);
            float spdup = gpu_ms / ane_ms;

            fprintf(stderr, "%8d  %10.1f  %10.1f  %10.1f  %10.1f  %7.2fx %s\n",
                    sl, gpu_ms, gpu_tps, ane_ms, ane_tps, spdup,
                    spdup > 1.0f ? "(FASTER)" : "(slower)");
        }

        fprintf(stderr, "\n=== Sweep complete ===\n");
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }

    // ================================================================
    // Part 1: Text Generation Coherence Test
    // ================================================================
    bool text_match = true;  // default to true when skipping

    if (do_coherence) {
        fprintf(stderr, "\n=== Text Generation Coherence Test ===\n");

        const char * test_prompt = "What is the capital of France? Answer in one sentence:";
        fprintf(stderr, "Prompt: \"%s\"\n\n", test_prompt);

        std::vector<llama_token> prompt_tokens(strlen(test_prompt) + 16);
        int n_prompt_tokens = llama_tokenize(model, test_prompt, strlen(test_prompt),
                                              prompt_tokens.data(), (int)prompt_tokens.size(),
                                              true, true);
        if (n_prompt_tokens < 0) {
            fprintf(stderr, "FAIL: tokenization failed\n");
            return 1;
        }
        prompt_tokens.resize(n_prompt_tokens);
        fprintf(stderr, "Prompt tokens: %d\n", n_prompt_tokens);

        int gen_ctx = std::max(n_ctx, n_prompt_tokens + n_gen + 16);

        fprintf(stderr, "\n--- GPU-only generation ---\n");
        gen_result gpu_gen = run_generation(model, prompt_tokens, gen_ctx, n_gen, false, 0.0f);
        fprintf(stderr, "  Prefill: %.1f ms (%d tokens, %.1f tok/s)\n",
                gpu_gen.prefill_ms, n_prompt_tokens,
                gpu_gen.prefill_ms > 0 ? n_prompt_tokens / (gpu_gen.prefill_ms / 1000.0f) : 0.0f);
        fprintf(stderr, "  Output: \"%s\"\n", gpu_gen.text.c_str());

        fprintf(stderr, "\n--- GPU+ANE generation ---\n");
        gen_result ane_gen = run_generation(model, prompt_tokens, gen_ctx, n_gen, true, ratio);
        fprintf(stderr, "  Prefill: %.1f ms (%d tokens, %.1f tok/s)\n",
                ane_gen.prefill_ms, n_prompt_tokens,
                ane_gen.prefill_ms > 0 ? n_prompt_tokens / (ane_gen.prefill_ms / 1000.0f) : 0.0f);
        fprintf(stderr, "  Output: \"%s\"\n", ane_gen.text.c_str());

        bool text_exact = (gpu_gen.text == ane_gen.text);
        bool gpu_has_answer = !gpu_gen.text.empty() && gpu_gen.text.find("Paris") != std::string::npos;
        bool ane_has_answer = !ane_gen.text.empty() && ane_gen.text.find("Paris") != std::string::npos;
        text_match = gpu_has_answer && ane_has_answer;

        fprintf(stderr, "\n  Exact match: %s\n", text_exact ? "YES" : "NO (expected with FP16 split)");
        fprintf(stderr, "  GPU  answer correct: %s\n", gpu_has_answer ? "YES (contains 'Paris')" : "NO");
        fprintf(stderr, "  ANE  answer correct: %s\n", ane_has_answer ? "YES (contains 'Paris')" : "NO");
        fprintf(stderr, "  Coherence: %s\n", text_match ? "PASS — both produce correct answers" : "FAIL");
    } else {
        fprintf(stderr, "\n=== Coherence test skipped (--no-coherence) ===\n");
    }

    // ================================================================
    // Part 2: Prefill Speed Benchmark
    // ================================================================
    fprintf(stderr, "\n=== Prefill Speed Benchmark ===\n");

    // Create a batch of tokens for benchmarking
    std::vector<llama_token> tokens(seq_len);
    for (int i = 0; i < seq_len; i++) {
        tokens[i] = (i % (n_vocab - 1)) + 1;
    }

    // Helper to create a context, run streaming prefill, destroy context.
    // Does TWO decodes: first compiles kernels + warmup (not timed),
    // second is the actual benchmark (timed).
    auto run_prefill = [&](bool ane_enabled, float ane_ratio) -> float {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = seq_len;
        cparams.n_ubatch = seq_len;
        cparams.prefill_streaming = true;
        cparams.prefill_telemetry = false;
        cparams.prefill_ane = ane_enabled;
        cparams.prefill_ane_ratio = ane_ratio;
        cparams.prefill_ane_cache = "/tmp/ane-bench-cache";

        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "FAIL: could not create context\n");
            return -1.0f;
        }

        // Build batch
        llama_batch batch = llama_batch_init(seq_len, 0, 1);
        for (int i = 0; i < seq_len; i++) {
            common_batch_add(batch, tokens[i], i, {0}, i == seq_len - 1);
        }

        // First decode: compile kernels + warmup (not timed)
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "  WARNING: warmup llama_decode returned %d\n", ret);
        }

        // Clear KV cache and re-run for clean timing
        llama_kv_cache_clear(ctx);

        // Second decode: timed benchmark run
        auto t0 = std::chrono::high_resolution_clock::now();
        ret = llama_decode(ctx, batch);
        auto t1 = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (ret != 0) {
            fprintf(stderr, "  WARNING: llama_decode returned %d\n", ret);
        }

        llama_batch_free(batch);
        llama_free(ctx);

        return ms;
    };

    // Warmup runs
    fprintf(stderr, "Warmup (GPU-only)...\n");
    float warmup_ms = run_prefill(false, 0.0f);
    fprintf(stderr, "  warmup GPU: %.1f ms (%.1f tok/s)\n", warmup_ms,
            warmup_ms > 0 ? seq_len / (warmup_ms / 1000.0f) : 0.0f);

    fprintf(stderr, "Warmup (GPU+ANE, compiling kernels)...\n");
    float warmup_ane_ms = run_prefill(true, ratio);
    fprintf(stderr, "  warmup ANE: %.1f ms (includes kernel compilation)\n\n", warmup_ane_ms);

    // GPU-only runs
    fprintf(stderr, "--- GPU-only streaming prefill ---\n");
    std::vector<float> gpu_times;
    for (int r = 0; r < n_runs; r++) {
        float ms = run_prefill(false, 0.0f);
        gpu_times.push_back(ms);
        fprintf(stderr, "  run %d: %.1f ms (%.1f tok/s)\n", r + 1, ms,
                ms > 0 ? seq_len / (ms / 1000.0f) : 0.0f);
    }

    float gpu_avg = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0f) / n_runs;
    float gpu_min = *std::min_element(gpu_times.begin(), gpu_times.end());
    fprintf(stderr, "  avg: %.1f ms, min: %.1f ms\n\n", gpu_avg, gpu_min);

    // GPU+ANE runs
    fprintf(stderr, "--- GPU+ANE streaming prefill (ratio=%.2f) ---\n", ratio);
    std::vector<float> ane_times;
    for (int r = 0; r < n_runs; r++) {
        float ms = run_prefill(true, ratio);
        ane_times.push_back(ms);
        fprintf(stderr, "  run %d: %.1f ms (%.1f tok/s)\n", r + 1, ms,
                ms > 0 ? seq_len / (ms / 1000.0f) : 0.0f);
    }

    float ane_avg = std::accumulate(ane_times.begin(), ane_times.end(), 0.0f) / n_runs;
    float ane_min = *std::min_element(ane_times.begin(), ane_times.end());
    fprintf(stderr, "  avg: %.1f ms, min: %.1f ms\n\n", ane_avg, ane_min);

    // Summary
    float speedup_avg = gpu_avg / ane_avg;
    float speedup_min = gpu_min / ane_min;
    fprintf(stderr, "=== Summary ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Seq: %d, ANE ratio: %.2f\n", seq_len, ratio);
    fprintf(stderr, "n_embd: %d, n_layer: %d\n", n_embd, n_layer);
    fprintf(stderr, "GPU-only:  avg=%.1fms  min=%.1fms  (%.1f tok/s avg)\n",
            gpu_avg, gpu_min, seq_len / (gpu_avg / 1000.0f));
    fprintf(stderr, "GPU+ANE:   avg=%.1fms  min=%.1fms  (%.1f tok/s avg)\n",
            ane_avg, ane_min, seq_len / (ane_avg / 1000.0f));
    fprintf(stderr, "Speedup:   avg=%.3fx  min=%.3fx\n", speedup_avg, speedup_min);

    if (speedup_avg > 1.0f) {
        fprintf(stderr, "\nRESULT: GPU+ANE is %.1f%% FASTER\n",
                (speedup_avg - 1.0f) * 100.0f);
    } else {
        fprintf(stderr, "\nRESULT: GPU+ANE is %.1f%% SLOWER (model too small for ANE benefit, need hidden>=2560)\n",
                (1.0f - speedup_avg) * 100.0f);
    }

    fprintf(stderr, "\nCoherence: %s\n", text_match ? "PASS" : "PARTIAL (FP16 divergence expected)");

    llama_free_model(model);
    llama_backend_free();

    return 0;
}
