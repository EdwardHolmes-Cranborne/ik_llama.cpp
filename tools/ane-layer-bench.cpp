// ane-layer-bench.cpp — Benchmark ANE throughput for real model dimensions
//
// Reads model dimensions from a GGUF file, compiles CoreML conv shards
// for each projection type at real dimensions, and benchmarks ANE throughput.
// Reports per-projection, per-layer, and full-model performance estimates.
//
// Supports dense, MoE, and hybrid-attention (SSM + MoE) architectures
// such as Qwen 3.5 35B-A3B.
//
// Usage:
//   ane-layer-bench -m model.gguf [--seq 512] [--ratio 0.5]
//   ane-layer-bench --hidden 4096 --heads 32 --kv-heads 8 --intermediate 11008 --layers 32
//
// Build: cmake -DGGML_ANE=ON && cmake --build . --target ane-layer-bench
// Run WITHOUT sandbox for ANE access.

#include "ggml.h"
#include "ggml-ane.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mach/mach_time.h>

static double bench_time_ms(void) {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.numer == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

// ===========================================================================
// Model dimensions
// ===========================================================================

struct model_dims {
    int hidden_dim;
    int n_heads;
    int n_kv_heads;
    int intermediate_dim;
    int n_layers;
    int head_dim;
    char arch[64];

    // MoE fields
    int expert_count;
    int expert_used_count;
    int expert_ffn_dim;
    int shared_expert_ffn_dim;
    int ssm_inner_size;
    int full_attn_interval;
    int key_length;
    int value_length;
    bool is_moe;
};

// ===========================================================================
// GGUF dimension reader
// ===========================================================================

static bool read_gguf_dims(const char * path, model_dims * dims) {
    struct gguf_init_params params = { false, NULL };
    struct gguf_context * ctx = gguf_init_from_file(path, params);
    if (!ctx) {
        fprintf(stderr, "Failed to open GGUF: %s\n", path);
        return false;
    }

    // Read architecture name
    int arch_key = gguf_find_key(ctx, "general.architecture");
    const char * arch = "llama";
    if (arch_key >= 0) {
        arch = gguf_get_val_str(ctx, arch_key);
    }
    snprintf(dims->arch, sizeof(dims->arch), "%s", arch);

    // Helper to read arch-prefixed u32 keys
    char key[256];
    auto get_u32 = [&](const char * suffix, int def) -> int {
        snprintf(key, sizeof(key), "%s.%s", arch, suffix);
        int id = gguf_find_key(ctx, key);
        return (id >= 0) ? (int)gguf_get_val_u32(ctx, id) : def;
    };

    // Standard fields
    dims->hidden_dim       = get_u32("embedding_length", 0);
    dims->n_heads          = get_u32("attention.head_count", 0);
    dims->n_kv_heads       = get_u32("attention.head_count_kv", dims->n_heads);
    dims->intermediate_dim = get_u32("feed_forward_length", 0);
    dims->n_layers         = get_u32("block_count", 0);

    // MoE fields
    dims->expert_count          = get_u32("expert_count", 0);
    dims->expert_used_count     = get_u32("expert_used_count", 0);
    dims->expert_ffn_dim        = get_u32("expert_feed_forward_length", 0);
    dims->shared_expert_ffn_dim = get_u32("expert_shared_feed_forward_length", 0);
    dims->ssm_inner_size        = get_u32("ssm.inner_size", 0);
    dims->full_attn_interval    = get_u32("full_attention_interval", 0);

    // Head dimensions (may be explicit or derived)
    if (dims->hidden_dim > 0 && dims->n_heads > 0) {
        dims->head_dim = dims->hidden_dim / dims->n_heads;
    } else {
        dims->head_dim = 128;
    }
    dims->key_length   = get_u32("attention.key_length", dims->head_dim);
    dims->value_length = get_u32("attention.value_length", dims->head_dim);

    // Detect MoE
    dims->is_moe = dims->expert_count > 0;

    // Fallback: if feed_forward_length missing, use expert_feed_forward_length
    if (dims->intermediate_dim == 0 && dims->expert_ffn_dim > 0) {
        dims->intermediate_dim = dims->expert_ffn_dim;
    }

    gguf_free(ctx);

    // Validation
    if (dims->hidden_dim == 0 || dims->n_heads == 0 || dims->n_layers == 0) {
        fprintf(stderr, "Could not read basic model dimensions from GGUF\n");
        fprintf(stderr, "  arch=%s hidden=%d heads=%d layers=%d\n",
                dims->arch, dims->hidden_dim, dims->n_heads, dims->n_layers);
        return false;
    }
    if (!dims->is_moe && dims->intermediate_dim == 0) {
        fprintf(stderr, "Could not read feed_forward_length from GGUF (arch=%s)\n", dims->arch);
        return false;
    }
    if (dims->is_moe && dims->expert_ffn_dim == 0) {
        fprintf(stderr, "MoE model but expert_feed_forward_length not found (arch=%s)\n", dims->arch);
        return false;
    }

    return true;
}

// ===========================================================================
// Projection definition
// ===========================================================================

#define MAX_PROJS 16

struct projection {
    const char * name;
    int in_ch;
    int out_ch;
    double gflops_per_token;  // 2 * in * out / 1e9
};

// ===========================================================================
// Main
// ===========================================================================

int main(int argc, char ** argv) {
    // Parse arguments
    const char * model_path = NULL;
    model_dims dims = {};
    int seq_len = 512;
    float ratio = 0.5f;
    const char * cache_dir = "/tmp/ane_layer_bench_cache";
    int bench_iters = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            dims.hidden_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) {
            dims.n_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) {
            dims.n_kv_heads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--intermediate") == 0 && i + 1 < argc) {
            dims.intermediate_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            dims.n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ratio") == 0 && i + 1 < argc) {
            ratio = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            bench_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cache") == 0 && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: ane-layer-bench -m model.gguf [options]\n"
                   "       ane-layer-bench --hidden N --heads N --intermediate N --layers N [options]\n"
                   "\nOptions:\n"
                   "  -m <path>           GGUF model file (auto-detect dimensions)\n"
                   "  --hidden <N>        Hidden dimension\n"
                   "  --heads <N>         Attention head count\n"
                   "  --kv-heads <N>      KV head count (default: same as heads)\n"
                   "  --intermediate <N>  FFN intermediate dimension\n"
                   "  --layers <N>        Number of layers\n"
                   "  --seq <N>           Sequence length (default: 512)\n"
                   "  --ratio <F>         ANE split ratio (default: 0.5)\n"
                   "  --iters <N>         Benchmark iterations (default: 100)\n"
                   "  --cache <dir>       Cache dir for compiled shards\n");
            return 0;
        }
    }

    // Read dimensions from GGUF if provided
    if (model_path) {
        if (!read_gguf_dims(model_path, &dims)) {
            return 1;
        }
    }

    // Apply defaults
    if (dims.n_kv_heads == 0) dims.n_kv_heads = dims.n_heads;
    if (dims.hidden_dim > 0 && dims.n_heads > 0) {
        dims.head_dim = dims.hidden_dim / dims.n_heads;
    }
    if (dims.key_length == 0)   dims.key_length   = dims.head_dim;
    if (dims.value_length == 0) dims.value_length = dims.head_dim;

    // Validate
    if (dims.hidden_dim == 0 || dims.n_heads == 0 || dims.n_layers == 0) {
        fprintf(stderr, "Error: Missing model dimensions. Provide -m model.gguf or --hidden/--heads/--layers.\n");
        return 1;
    }
    if (!dims.is_moe && dims.intermediate_dim == 0) {
        fprintf(stderr, "Error: Missing intermediate dimension for dense model.\n");
        return 1;
    }

    // Check ANE
    if (!ggml_backend_ane_supported()) {
        fprintf(stderr, "Error: ANE not available. Run on Apple Silicon macOS 14+ without sandbox.\n");
        return 1;
    }

    // ===================================================================
    // Print model info
    // ===================================================================
    printf("================================================================\n");
    printf("ANE Layer Benchmark\n");
    printf("================================================================\n");
    if (model_path) printf("Model:         %s\n", model_path);
    printf("Architecture:  %s%s\n", dims.arch[0] ? dims.arch : "unknown",
           dims.is_moe ? " (MoE)" : "");
    printf("Hidden dim:    %d\n", dims.hidden_dim);
    printf("Heads:         %d (KV: %d), head_dim: %d\n",
           dims.n_heads, dims.n_kv_heads, dims.head_dim);
    if (dims.key_length != dims.head_dim || dims.value_length != dims.head_dim) {
        printf("Key length:    %d, Value length: %d\n", dims.key_length, dims.value_length);
    }
    if (dims.is_moe) {
        printf("Experts:       %d total, %d active per token\n",
               dims.expert_count, dims.expert_used_count);
        printf("Expert FFN:    %d\n", dims.expert_ffn_dim);
        if (dims.shared_expert_ffn_dim > 0) {
            printf("Shared FFN:    %d\n", dims.shared_expert_ffn_dim);
        }
        if (dims.ssm_inner_size > 0) {
            printf("SSM inner:     %d\n", dims.ssm_inner_size);
        }
        if (dims.full_attn_interval > 0) {
            int n_full = dims.n_layers / dims.full_attn_interval;
            int n_linear = dims.n_layers - n_full;
            printf("Attention:     %d linear + %d full (interval=%d)\n",
                   n_linear, n_full, dims.full_attn_interval);
        }
    } else {
        printf("Intermediate:  %d\n", dims.intermediate_dim);
    }
    printf("Layers:        %d\n", dims.n_layers);
    printf("Seq length:    %d\n", seq_len);
    printf("ANE ratio:     %.2f\n", ratio);
    printf("Bench iters:   %d\n", bench_iters);
    printf("================================================================\n\n");

    // ===================================================================
    // Build projection list
    // ===================================================================
    projection projs[MAX_PROJS] = {};
    int n_projs = 0;

    if (dims.is_moe && dims.ssm_inner_size > 0) {
        // ==============================================================
        // MoE + hybrid attention (e.g., Qwen 3.5 35B-A3B)
        //
        // Projection indices:
        //   0=Lin QKV, 1=Lin Z, 2=Lin out
        //   3=Attn Q, 4=Attn K, 5=Attn V, 6=Attn O
        //   7=Expert gate, 8=Expert up, 9=Expert down
        //   10=Shared gate, 11=Shared up, 12=Shared down
        // ==============================================================
        int H   = dims.hidden_dim;
        int ssm = dims.ssm_inner_size;
        int n_h = dims.n_heads;
        int n_kv = dims.n_kv_heads;
        int kl  = dims.key_length;
        int vl  = dims.value_length;
        int eff = dims.expert_ffn_dim;
        int sff = dims.shared_expert_ffn_dim > 0 ? dims.shared_expert_ffn_dim : eff;

        // Linear attention projections (ANE half via ratio)
        projs[n_projs++] = {"Lin QKV",      H, (int)(ssm * 2 * ratio), 0};
        projs[n_projs++] = {"Lin Z",        H, (int)(ssm * ratio),     0};
        projs[n_projs++] = {"Lin out",      (int)(ssm * ratio), H,     0};

        // Full attention projections (ANE half via ratio)
        projs[n_projs++] = {"Attn Q",       H, (int)(n_h * kl * ratio),  0};
        projs[n_projs++] = {"Attn K",       H, (int)(n_kv * kl * ratio), 0};
        projs[n_projs++] = {"Attn V",       H, (int)(n_kv * vl * ratio), 0};
        projs[n_projs++] = {"Attn O",       (int)(n_h * vl * ratio), H,  0};

        // Per-expert MLP — no ratio split, experts are small
        projs[n_projs++] = {"Expert gate",   H, eff, 0};
        projs[n_projs++] = {"Expert up",     H, eff, 0};
        projs[n_projs++] = {"Expert down",   eff, H, 0};

        // Shared expert — no ratio split, same size as per-expert
        projs[n_projs++] = {"Shared gate",   H, sff, 0};
        projs[n_projs++] = {"Shared up",     H, sff, 0};
        projs[n_projs++] = {"Shared down",   sff, H, 0};

    } else if (dims.is_moe) {
        // ==============================================================
        // Standard MoE (no SSM, e.g., Mixtral)
        // ==============================================================
        int H   = dims.hidden_dim;
        int eff = dims.expert_ffn_dim;
        int sff = dims.shared_expert_ffn_dim > 0 ? dims.shared_expert_ffn_dim : eff;

        // Standard attention (ANE half via ratio)
        int q_out  = (int)(dims.n_heads * dims.head_dim * ratio);
        int kv_out = (int)(dims.n_kv_heads * dims.head_dim * ratio);
        if (kv_out < dims.head_dim) kv_out = dims.head_dim;

        projs[n_projs++] = {"Q projection", H, q_out,  0};
        projs[n_projs++] = {"K projection", H, kv_out, 0};
        projs[n_projs++] = {"V projection", H, kv_out, 0};
        projs[n_projs++] = {"O projection", q_out, H,  0};

        // Per-expert MLP — no ratio, experts are small
        projs[n_projs++] = {"Expert gate",  H, eff, 0};
        projs[n_projs++] = {"Expert up",    H, eff, 0};
        projs[n_projs++] = {"Expert down",  eff, H, 0};

        // Shared expert (if present)
        if (dims.shared_expert_ffn_dim > 0) {
            projs[n_projs++] = {"Shared gate",  H, sff, 0};
            projs[n_projs++] = {"Shared up",    H, sff, 0};
            projs[n_projs++] = {"Shared down",  sff, H, 0};
        }

    } else {
        // ==============================================================
        // Dense model (original path)
        // ==============================================================
        int q_out  = (int)(dims.n_heads * dims.head_dim * ratio);
        int kv_out = (int)(dims.n_kv_heads * dims.head_dim * ratio);
        if (kv_out < dims.head_dim) kv_out = dims.head_dim;

        int gate_out = (int)(dims.intermediate_dim * ratio);
        int up_out   = (int)(dims.intermediate_dim * ratio);
        int down_in  = (int)(dims.intermediate_dim * ratio);

        projs[n_projs++] = {"Q projection",  dims.hidden_dim, q_out,    0};
        projs[n_projs++] = {"K projection",  dims.hidden_dim, kv_out,   0};
        projs[n_projs++] = {"V projection",  dims.hidden_dim, kv_out,   0};
        projs[n_projs++] = {"FFN gate",      dims.hidden_dim, gate_out, 0};
        projs[n_projs++] = {"FFN up",        dims.hidden_dim, up_out,   0};
        projs[n_projs++] = {"FFN down",      down_in, dims.hidden_dim,  0};
    }

    // Compute GFLOPS per projection per token
    for (int i = 0; i < n_projs; i++) {
        projs[i].gflops_per_token = 2.0 * projs[i].in_ch * projs[i].out_ch / 1e9;
    }

    // ===================================================================
    // Compile CoreML shards for each unique (in_ch, out_ch)
    // ===================================================================
    printf("Compiling CoreML shards (coremltools)...\n");
    ggml_ane_kernel_t kernels[MAX_PROJS] = {};

    for (int i = 0; i < n_projs; i++) {
        // Reuse kernel if identical shape already compiled
        bool found = false;
        for (int j = 0; j < i; j++) {
            if (projs[j].in_ch == projs[i].in_ch && projs[j].out_ch == projs[i].out_ch) {
                kernels[i] = kernels[j];
                found = true;
                printf("  [%2d/%d] %-15s [%4d -> %4d] sp=%d: (reusing shard %d)\n",
                       i + 1, n_projs, projs[i].name,
                       projs[i].in_ch, projs[i].out_ch, seq_len, j + 1);
                break;
            }
        }
        if (found) continue;

        printf("  [%2d/%d] %-15s [%4d -> %4d] sp=%d: compiling...",
               i + 1, n_projs, projs[i].name,
               projs[i].in_ch, projs[i].out_ch, seq_len);
        fflush(stdout);

        // Generate synthetic weights (throughput is independent of values)
        size_t n_w = (size_t)projs[i].out_ch * projs[i].in_ch;
        float * weights = (float *)malloc(n_w * sizeof(float));
        for (size_t w = 0; w < n_w; w++) weights[w] = 0.001f * ((float)(w % 1000) / 1000.0f);

        kernels[i] = ggml_ane_compile(
            NULL, "conv",
            projs[i].in_ch, projs[i].out_ch, seq_len,
            weights, n_w * sizeof(float),
            cache_dir);
        free(weights);

        if (!kernels[i]) {
            printf(" FAILED\n");
            fprintf(stderr, "  Error: Could not compile shard. Is coremltools installed?\n");
            fprintf(stderr, "  Try: pip3 install coremltools\n");
            return 1;
        }
        printf(" OK\n");
    }
    printf("\n");

    // ===================================================================
    // Benchmark each projection
    // ===================================================================
    printf("Benchmarking ANE throughput (seq_len=%d, %d iters)...\n\n", seq_len, bench_iters);
    printf("  %-17s | %6s %6s | %8s | %9s | %8s\n",
           "Projection", "in_ch", "out_ch", "GFLOPS", "ANE ms", "TFLOPS");
    printf("  %-17s-+-%6s-%6s-+-%8s-+-%9s-+-%8s\n",
           "-----------------", "------", "------", "--------", "---------", "--------");

    double measured_ms[MAX_PROJS] = {};
    double measured_tflops[MAX_PROJS] = {};
    bool   proj_ok[MAX_PROJS] = {};

    for (int i = 0; i < n_projs; i++) {
        if (!kernels[i]) {
            printf("  %-17s | %6d %6d | %8.3f | %9s | %8s\n",
                   projs[i].name, projs[i].in_ch, projs[i].out_ch,
                   projs[i].gflops_per_token * seq_len, "SKIP", "SKIP");
            continue;
        }

        // Fill input
        size_t n_in = (size_t)projs[i].in_ch * seq_len;
        __fp16 * input_buf = (__fp16 *)malloc(n_in * sizeof(__fp16));
        for (size_t j = 0; j < n_in; j++) input_buf[j] = (__fp16)0.01f;
        ggml_ane_write_input(kernels[i], 0, input_buf, n_in * sizeof(__fp16));
        free(input_buf);

        // Warmup
        for (int w = 0; w < 15; w++) ggml_ane_eval(kernels[i]);

        // Benchmark
        double t0 = bench_time_ms();
        for (int it = 0; it < bench_iters; it++) {
            ggml_ane_eval(kernels[i]);
        }
        double elapsed = bench_time_ms() - t0;

        double ms_per_eval = elapsed / bench_iters;
        double gflops = projs[i].gflops_per_token * seq_len;
        double tflops = gflops / ms_per_eval;  // GFLOPS / ms = TFLOPS

        measured_ms[i] = ms_per_eval;
        measured_tflops[i] = tflops;
        proj_ok[i] = true;

        printf("  %-17s | %6d %6d | %8.3f | %9.3f | %8.3f\n",
               projs[i].name, projs[i].in_ch, projs[i].out_ch,
               gflops, ms_per_eval, tflops);
    }

    printf("  %-17s-+-%6s-%6s-+-%8s-+-%9s-+-%8s\n",
           "-----------------", "------", "------", "--------", "---------", "--------");

    // Summary row
    double total_ane_ms = 0, total_gflops = 0;
    for (int i = 0; i < n_projs; i++) {
        if (proj_ok[i]) {
            total_ane_ms += measured_ms[i];
            total_gflops += projs[i].gflops_per_token * seq_len;
        }
    }
    double avg_tflops = (total_ane_ms > 0) ? total_gflops / total_ane_ms : 0;
    printf("  %-17s | %6s %6s | %8.3f | %9.3f | %8.3f\n",
           "TOTAL", "", "", total_gflops, total_ane_ms, avg_tflops);
    printf("\n");

    // ===================================================================
    // Per-Layer & Full-Model Estimates
    // ===================================================================
    double gpu_tflops_est = 8.0;  // Conservative M3 Max Metal estimate

    if (dims.is_moe && dims.ssm_inner_size > 0) {
        // ==============================================================
        // MoE + Hybrid attention (Qwen 3.5 style)
        //
        // Layout: 30 linear-attention layers + 10 full-attention layers
        //         All 40 layers have MoE MLP
        //
        // Strategy:
        //   - Attention projections: split GPU/ANE with ratio
        //   - Expert projections: run entirely on ANE
        //   - Shared expert: run on GPU concurrently with ANE experts
        // ==============================================================
        int n_full_attn   = dims.full_attn_interval > 0
                            ? dims.n_layers / dims.full_attn_interval : 0;
        int n_linear_attn = dims.n_layers - n_full_attn;
        int n_active      = dims.expert_used_count > 0 ? dims.expert_used_count : 8;
        int sff           = dims.shared_expert_ffn_dim > 0
                            ? dims.shared_expert_ffn_dim : dims.expert_ffn_dim;

        printf("================================================================\n");
        printf("Per-Layer Estimate (MoE + hybrid attn, seq_len=%d)\n", seq_len);
        printf("================================================================\n\n");

        // --- MoE MLP phase ---
        // ANE: runs N active experts sequentially (gate + up + down each)
        double expert_ane_ms = 0;
        if (proj_ok[7])  expert_ane_ms += measured_ms[7];   // gate
        if (proj_ok[8])  expert_ane_ms += measured_ms[8];   // up
        if (proj_ok[9])  expert_ane_ms += measured_ms[9];   // down
        expert_ane_ms *= n_active;

        // GPU: runs shared expert concurrently
        double shared_gflops = (2.0 * dims.hidden_dim * sff * 2 +     // gate + up
                                2.0 * sff * dims.hidden_dim)           // down
                               * (double)seq_len / 1e9;
        double shared_gpu_ms = shared_gflops / gpu_tflops_est;

        // Shared expert ANE time (for reporting)
        double shared_ane_ms = 0;
        if (proj_ok[10]) shared_ane_ms += measured_ms[10];
        if (proj_ok[11]) shared_ane_ms += measured_ms[11];
        if (proj_ok[12]) shared_ane_ms += measured_ms[12];

        // MoE phase: ANE experts || GPU shared expert
        double moe_phase_ms = fmax(expert_ane_ms, shared_gpu_ms) + 0.05;

        // GPU-only MoE: all on GPU
        double expert_full_gflops = n_active *
            (2.0 * dims.hidden_dim * dims.expert_ffn_dim * 2 +
             2.0 * dims.expert_ffn_dim * dims.hidden_dim) * (double)seq_len / 1e9;
        double moe_gpu_only_ms = (expert_full_gflops + shared_gflops) / gpu_tflops_est;

        // --- Linear attention layer (30 layers) ---
        // Full projection FLOPs (before split)
        double lin_qkv_full = 2.0 * dims.hidden_dim * (dims.ssm_inner_size * 2.0) * seq_len / 1e9;
        double lin_z_full   = 2.0 * dims.hidden_dim * (double)dims.ssm_inner_size * seq_len / 1e9;
        double lin_out_full = 2.0 * (double)dims.ssm_inner_size * dims.hidden_dim * seq_len / 1e9;
        double lin_proj_full = lin_qkv_full + lin_z_full + lin_out_full;

        // ANE half time (measured)
        double lin_ane_ms = 0;
        if (proj_ok[0]) lin_ane_ms += measured_ms[0];
        if (proj_ok[1]) lin_ane_ms += measured_ms[1];
        if (proj_ok[2]) lin_ane_ms += measured_ms[2];

        // GPU half time
        double lin_gpu_half_ms = lin_proj_full * (1.0 - ratio) / gpu_tflops_est;
        double lin_gpu_only_ms = lin_proj_full / gpu_tflops_est;

        // Projection phase: GPU || ANE concurrent
        double lin_proj_ms = fmax(lin_gpu_half_ms, lin_ane_ms) + 0.1;

        // SSM/conv compute (GPU only, lightweight)
        double lin_compute_ms = lin_proj_full * 0.05 / gpu_tflops_est;

        // Total linear-attention layer
        double lin_layer_ane_ms = lin_proj_ms + lin_compute_ms + moe_phase_ms;
        double lin_layer_gpu_ms = lin_gpu_only_ms + lin_compute_ms + moe_gpu_only_ms;

        // --- Full attention layer (10 layers) ---
        double attn_q_full = 2.0 * dims.hidden_dim * (double)(dims.n_heads * dims.key_length)    * seq_len / 1e9;
        double attn_k_full = 2.0 * dims.hidden_dim * (double)(dims.n_kv_heads * dims.key_length) * seq_len / 1e9;
        double attn_v_full = 2.0 * dims.hidden_dim * (double)(dims.n_kv_heads * dims.value_length)* seq_len / 1e9;
        double attn_o_full = 2.0 * (double)(dims.n_heads * dims.value_length) * dims.hidden_dim  * seq_len / 1e9;
        double attn_proj_full = attn_q_full + attn_k_full + attn_v_full + attn_o_full;

        // ANE half time (measured)
        double attn_ane_ms = 0;
        if (proj_ok[3]) attn_ane_ms += measured_ms[3];
        if (proj_ok[4]) attn_ane_ms += measured_ms[4];
        if (proj_ok[5]) attn_ane_ms += measured_ms[5];
        if (proj_ok[6]) attn_ane_ms += measured_ms[6];

        // GPU half time
        double attn_gpu_half_ms = attn_proj_full * (1.0 - ratio) / gpu_tflops_est;
        double attn_gpu_only_ms = attn_proj_full / gpu_tflops_est;

        // Projection phase: GPU || ANE
        double attn_proj_ms = fmax(attn_gpu_half_ms, attn_ane_ms) + 0.1;

        // Attention computation: QK^T + softmax + V (GPU only)
        double attn_compute_gflops = 2.0 * dims.n_heads *
            (double)seq_len * seq_len * dims.key_length / 1e9;
        double attn_compute_ms = attn_compute_gflops / gpu_tflops_est;

        // Total full-attention layer
        double full_layer_ane_ms = attn_proj_ms + attn_compute_ms + moe_phase_ms;
        double full_layer_gpu_ms = attn_gpu_only_ms + attn_compute_ms + moe_gpu_only_ms;

        // --- Print breakdown ---
        printf("  Component                | GPU-only ms | GPU+ANE ms | Savings\n");
        printf("  -------------------------|-------------|------------|--------\n");

        printf("  Linear attn projs (×%d)  | %9.3f   | %8.3f    | %.1f%%\n",
               n_linear_attn, lin_gpu_only_ms, lin_proj_ms,
               (1.0 - lin_proj_ms / lin_gpu_only_ms) * 100);

        printf("  Full attn projs (×%d)    | %9.3f   | %8.3f    | %.1f%%\n",
               n_full_attn, attn_gpu_only_ms, attn_proj_ms,
               (1.0 - attn_proj_ms / attn_gpu_only_ms) * 100);

        printf("  Attn compute (GPU)       | %9.3f   | %8.3f    | 0%%\n",
               attn_compute_ms, attn_compute_ms);

        printf("  MoE MLP (%d of %d exp)   | %9.3f   | %8.3f    | %.1f%%\n",
               n_active, dims.expert_count, moe_gpu_only_ms, moe_phase_ms,
               (1.0 - moe_phase_ms / moe_gpu_only_ms) * 100);

        printf("    ANE: %d×expert          |             | %8.3f    |\n",
               n_active, expert_ane_ms);
        printf("    GPU: shared expert      |             | %8.3f    |\n",
               shared_gpu_ms);

        printf("  -------------------------|-------------|------------|--------\n");

        printf("  Linear attn layer        | %9.3f   | %8.3f    | %.1f%%\n",
               lin_layer_gpu_ms, lin_layer_ane_ms,
               (1.0 - lin_layer_ane_ms / lin_layer_gpu_ms) * 100);

        printf("  Full attn layer          | %9.3f   | %8.3f    | %.1f%%\n",
               full_layer_gpu_ms, full_layer_ane_ms,
               (1.0 - full_layer_ane_ms / full_layer_gpu_ms) * 100);

        // Weighted average
        double avg_layer_gpu = ((double)n_linear_attn * lin_layer_gpu_ms +
                                (double)n_full_attn   * full_layer_gpu_ms) / dims.n_layers;
        double avg_layer_ane = ((double)n_linear_attn * lin_layer_ane_ms +
                                (double)n_full_attn   * full_layer_ane_ms) / dims.n_layers;

        printf("  Weighted avg layer       | %9.3f   | %8.3f    | %.1f%%\n",
               avg_layer_gpu, avg_layer_ane,
               (1.0 - avg_layer_ane / avg_layer_gpu) * 100);

        // Full model
        printf("\n");
        printf("================================================================\n");
        printf("Full Model Estimate (%d layers, seq_len=%d)\n", dims.n_layers, seq_len);
        printf("================================================================\n");

        double full_gpu_ms  = avg_layer_gpu * dims.n_layers;
        double full_both_ms = avg_layer_ane * dims.n_layers;

        printf("  GPU-only:   %8.1f ms  (%6.0f tok/s)\n",
               full_gpu_ms, seq_len / (full_gpu_ms / 1000.0));
        printf("  GPU+ANE:    %8.1f ms  (%6.0f tok/s)\n",
               full_both_ms, seq_len / (full_both_ms / 1000.0));
        printf("  Speedup:    %.2fx\n", full_gpu_ms / full_both_ms);
        printf("\n");

        // Utilization summary
        double ane_work_per_lin  = lin_ane_ms + expert_ane_ms;
        double ane_work_per_full = attn_ane_ms + expert_ane_ms;
        double total_ane_work = n_linear_attn * ane_work_per_lin +
                                n_full_attn   * ane_work_per_full;

        printf("  MoE: %d active experts × (gate+up+down) per layer\n", n_active);
        printf("  ANE time on experts:   %.3f ms/layer\n", expert_ane_ms);
        printf("  GPU shared expert:     %.3f ms/layer (concurrent)\n", shared_gpu_ms);
        printf("  ANE idle during attn:  %.3f ms (full-attn layers only)\n", attn_compute_ms);
        printf("  Total ANE work:        %.1f ms / %.1f ms wall (%.1f%% util)\n",
               total_ane_work, full_both_ms,
               total_ane_work / full_both_ms * 100);
        printf("\n");

    } else if (dims.is_moe) {
        // ==============================================================
        // Standard MoE estimate (no SSM)
        // ==============================================================
        int n_active = dims.expert_used_count > 0 ? dims.expert_used_count : 8;
        int sff      = dims.shared_expert_ffn_dim > 0
                       ? dims.shared_expert_ffn_dim : dims.expert_ffn_dim;
        bool has_shared = dims.shared_expert_ffn_dim > 0;

        printf("================================================================\n");
        printf("Per-Layer Estimate (MoE, seq_len=%d)\n", seq_len);
        printf("================================================================\n\n");

        // Attention projections: indices 0-3 (Q, K, V, O)
        double attn_full_gflops = 0;
        attn_full_gflops += 2.0 * dims.hidden_dim * (double)(dims.n_heads * dims.head_dim)    * seq_len / 1e9;
        attn_full_gflops += 2.0 * dims.hidden_dim * (double)(dims.n_kv_heads * dims.head_dim) * seq_len / 1e9 * 2; // K+V
        attn_full_gflops += 2.0 * (double)(dims.n_heads * dims.head_dim) * dims.hidden_dim    * seq_len / 1e9;

        double attn_ane_ms = 0;
        for (int i = 0; i < 4 && i < n_projs; i++) {
            if (proj_ok[i]) attn_ane_ms += measured_ms[i];
        }
        double attn_gpu_half_ms = attn_full_gflops * (1.0 - ratio) / gpu_tflops_est;
        double attn_gpu_only_ms = attn_full_gflops / gpu_tflops_est;
        double attn_proj_ms     = fmax(attn_gpu_half_ms, attn_ane_ms) + 0.1;

        // Attention compute
        double attn_compute_gflops = 2.0 * dims.n_heads *
            (double)seq_len * seq_len * dims.head_dim / 1e9;
        double attn_compute_ms = attn_compute_gflops / gpu_tflops_est;

        // Expert projections: indices 4-6
        double expert_ane_ms = 0;
        if (proj_ok[4]) expert_ane_ms += measured_ms[4];
        if (proj_ok[5]) expert_ane_ms += measured_ms[5];
        if (proj_ok[6]) expert_ane_ms += measured_ms[6];
        expert_ane_ms *= n_active;

        double expert_full_gflops = n_active *
            (2.0 * dims.hidden_dim * dims.expert_ffn_dim * 2 +
             2.0 * dims.expert_ffn_dim * dims.hidden_dim) * (double)seq_len / 1e9;

        double shared_gflops = 0, shared_gpu_ms = 0;
        if (has_shared) {
            shared_gflops = (2.0 * dims.hidden_dim * sff * 2 +
                             2.0 * sff * dims.hidden_dim) * (double)seq_len / 1e9;
            shared_gpu_ms = shared_gflops / gpu_tflops_est;
        }

        double moe_phase_ms = fmax(expert_ane_ms, shared_gpu_ms) + 0.05;
        double moe_gpu_only_ms = (expert_full_gflops + shared_gflops) / gpu_tflops_est;

        double layer_ane_ms = attn_proj_ms + attn_compute_ms + moe_phase_ms;
        double layer_gpu_ms = attn_gpu_only_ms + attn_compute_ms + moe_gpu_only_ms;

        printf("  Component              | GPU-only ms | GPU+ANE ms | Savings\n");
        printf("  -----------------------|-------------|------------|--------\n");
        printf("  Attention projs        | %9.3f   | %8.3f    | %.1f%%\n",
               attn_gpu_only_ms, attn_proj_ms,
               (1.0 - attn_proj_ms / attn_gpu_only_ms) * 100);
        printf("  Attn compute (GPU)     | %9.3f   | %8.3f    | 0%%\n",
               attn_compute_ms, attn_compute_ms);
        printf("  MoE MLP (%d experts)   | %9.3f   | %8.3f    | %.1f%%\n",
               n_active, moe_gpu_only_ms, moe_phase_ms,
               (1.0 - moe_phase_ms / moe_gpu_only_ms) * 100);
        printf("  -----------------------|-------------|------------|--------\n");
        printf("  TOTAL/layer            | %9.3f   | %8.3f    | %.1f%%\n",
               layer_gpu_ms, layer_ane_ms,
               (1.0 - layer_ane_ms / layer_gpu_ms) * 100);

        printf("\n");
        printf("================================================================\n");
        printf("Full Model Estimate (%d layers, seq_len=%d)\n", dims.n_layers, seq_len);
        printf("================================================================\n");

        double full_gpu_ms  = layer_gpu_ms * dims.n_layers;
        double full_both_ms = layer_ane_ms * dims.n_layers;

        printf("  GPU-only:   %8.1f ms  (%6.0f tok/s)\n",
               full_gpu_ms, seq_len / (full_gpu_ms / 1000.0));
        printf("  GPU+ANE:    %8.1f ms  (%6.0f tok/s)\n",
               full_both_ms, seq_len / (full_both_ms / 1000.0));
        printf("  Speedup:    %.2fx\n", full_gpu_ms / full_both_ms);
        printf("\n");

    } else {
        // ==============================================================
        // Dense model estimate (original path)
        // ==============================================================
        printf("================================================================\n");
        printf("Per-Layer Estimate (seq_len=%d)\n", seq_len);
        printf("================================================================\n");

        // GPU-only: all projections on GPU
        double gpu_only_gflops = 0;
        for (int i = 0; i < n_projs; i++) {
            double full_in, full_out;
            if (i < 3) {
                full_in  = dims.hidden_dim;
                full_out = (i == 0) ? dims.n_heads * dims.head_dim
                                    : dims.n_kv_heads * dims.head_dim;
            } else {
                full_in  = (i < 5) ? dims.hidden_dim : dims.intermediate_dim;
                full_out = (i < 5) ? dims.intermediate_dim : dims.hidden_dim;
            }
            gpu_only_gflops += 2.0 * full_in * full_out * seq_len / 1e9;
        }
        double gpu_only_ms = gpu_only_gflops / gpu_tflops_est;

        // QKV phase: GPU || ANE concurrent
        double qkv_gpu_gflops = 0;
        for (int i = 0; i < 3; i++) {
            double full = 2.0 * dims.hidden_dim *
                (i == 0 ? dims.n_heads * dims.head_dim : dims.n_kv_heads * dims.head_dim) *
                (double)seq_len / 1e9;
            qkv_gpu_gflops += full * (1.0 - ratio);
        }
        double qkv_gpu_ms = qkv_gpu_gflops / gpu_tflops_est;
        double qkv_ane_ms = 0;
        for (int i = 0; i < 3; i++) {
            if (proj_ok[i]) qkv_ane_ms += measured_ms[i];
        }
        double qkv_ms = fmax(qkv_gpu_ms, qkv_ane_ms) + 0.1;

        // Attention (GPU only)
        double attn_gflops = 2.0 * dims.n_heads * dims.head_dim *
            (double)seq_len * seq_len / 1e9;
        double o_proj_gflops = 2.0 * dims.hidden_dim * (double)dims.hidden_dim * seq_len / 1e9;
        double attn_ms = (attn_gflops + o_proj_gflops) / gpu_tflops_est;

        // MLP phase: GPU || ANE concurrent
        double mlp_gpu_gflops = 0;
        for (int i = 3; i < n_projs; i++) {
            double full_in  = (i < 5) ? dims.hidden_dim : dims.intermediate_dim;
            double full_out = (i < 5) ? dims.intermediate_dim : dims.hidden_dim;
            mlp_gpu_gflops += 2.0 * full_in * full_out * seq_len / 1e9 * (1.0 - ratio);
        }
        double mlp_gpu_ms = mlp_gpu_gflops / gpu_tflops_est;
        double mlp_ane_ms = 0;
        for (int i = 3; i < n_projs; i++) {
            if (proj_ok[i]) mlp_ane_ms += measured_ms[i];
        }
        double mlp_ms = fmax(mlp_gpu_ms, mlp_ane_ms) + 0.1;

        double total_gpu_ane_ms = qkv_ms + attn_ms + mlp_ms;

        printf("\n");
        printf("  Phase              | GPU-only ms | GPU+ANE ms | Savings\n");
        printf("  -------------------|-------------|------------|--------\n");
        printf("  QKV projections    | %9.2f   | %8.2f    | %.1f%%\n",
               qkv_gpu_gflops / (1.0 - ratio) / gpu_tflops_est,
               qkv_ms,
               (1.0 - qkv_ms / (qkv_gpu_gflops / (1.0 - ratio) / gpu_tflops_est)) * 100);
        printf("  Attention+O        | %9.2f   | %8.2f    | 0%% (GPU-only)\n",
               attn_ms, attn_ms);
        printf("  MLP (gate+up+down) | %9.2f   | %8.2f    | %.1f%%\n",
               mlp_gpu_gflops / (1.0 - ratio) / gpu_tflops_est,
               mlp_ms,
               (1.0 - mlp_ms / (mlp_gpu_gflops / (1.0 - ratio) / gpu_tflops_est)) * 100);
        printf("  -------------------|-------------|------------|--------\n");
        printf("  TOTAL/layer        | %9.2f   | %8.2f    | %.1f%%\n",
               gpu_only_ms + attn_ms, total_gpu_ane_ms,
               (1.0 - total_gpu_ane_ms / (gpu_only_ms + attn_ms)) * 100);

        // Full model
        printf("\n");
        printf("================================================================\n");
        printf("Full Model Estimate (%d layers, seq_len=%d)\n", dims.n_layers, seq_len);
        printf("================================================================\n");

        double full_gpu_ms  = (gpu_only_ms + attn_ms) * dims.n_layers;
        double full_both_ms = total_gpu_ane_ms * dims.n_layers;

        printf("  GPU-only:   %8.1f ms  (%6.0f tok/s)\n",
               full_gpu_ms, seq_len / (full_gpu_ms / 1000.0));
        printf("  GPU+ANE:    %8.1f ms  (%6.0f tok/s)\n",
               full_both_ms, seq_len / (full_both_ms / 1000.0));
        printf("  Speedup:    %.2fx\n", full_gpu_ms / full_both_ms);
        printf("\n");

        printf("ANE idle during attention: %.2f ms/layer (%.1f%% of layer time)\n",
               attn_ms, attn_ms / total_gpu_ane_ms * 100);
        printf("Effective ANE utilization: %.1f%%\n",
               (1.0 - attn_ms / total_gpu_ane_ms) * 100);
        printf("\n");
    }

    // Cleanup (don't double-free shared kernels)
    for (int i = n_projs - 1; i >= 0; i--) {
        bool shared = false;
        for (int j = 0; j < i; j++) {
            if (kernels[j] == kernels[i]) { shared = true; break; }
        }
        if (!shared && kernels[i]) ggml_ane_free(kernels[i]);
    }

    return 0;
}
