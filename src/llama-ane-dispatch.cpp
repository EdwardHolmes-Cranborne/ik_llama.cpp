// llama-ane-dispatch.cpp — Phase 3+4: ANE FFN dispatch implementation
//
// Manages per-layer fused FFN CoreML kernels on the ANE.
// During prefill, splits FFN MLP between GPU (reduced-dim FP16) and ANE
// (complementary portion baked into CoreML conv model).

#include "llama-ane-dispatch.h"
#include "llama-ane-weights.h"
#include "llama-model.h"
#include "ggml-ane.h"
#include "ggml.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#endif

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

static double ane_dispatch_time_ms(void) {
#ifdef __APPLE__
    static mach_timebase_info_data_t tb;
    static bool tb_init = false;
    if (!tb_init) { mach_timebase_info(&tb); tb_init = true; }
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
#else
    return 0.0;
#endif
}

// ---------------------------------------------------------------------------
// Per-layer state
// ---------------------------------------------------------------------------

struct ane_layer_state {
    // ANE kernel (fused gate+up+silu+mul+down)
    ggml_ane_kernel_t kernel = nullptr;

    // Whether this layer uses ANE split (false for MoE, skip layers, etc.)
    bool active = false;

    // GPU-half FP16 weights (pre-computed during compile)
    void *  gpu_gate_f16 = nullptr;  // [gpu_inter, hidden] FP16
    size_t  gpu_gate_bytes = 0;
    void *  gpu_up_f16   = nullptr;  // [gpu_inter, hidden] FP16
    size_t  gpu_up_bytes = 0;
    void *  gpu_down_f16 = nullptr;  // [hidden, gpu_inter] FP16
    size_t  gpu_down_bytes = 0;

    // Async dispatch state
#ifdef __APPLE__
    dispatch_group_t  group = nullptr;
    dispatch_queue_t  queue = nullptr;
#endif
    bool pending = false;
    int actual_seq = 0;  // actual seq_len for current dispatch (may be < compiled_seq)

    // Transpose buffers (reused per dispatch)
    void * inp_transposed  = nullptr;  // channel-first [1, C, 1, S] FP16
    void * out_transposed  = nullptr;  // channel-first [1, C, 1, S] FP16
    size_t inp_trans_bytes = 0;
    size_t out_trans_bytes = 0;

    // Validation weights (layer 0 only, when ANE_VALIDATE=1)
    float * val_gate = nullptr;  // [ane_inter, hidden] FP32
    float * val_up   = nullptr;  // [ane_inter, hidden] FP32
    float * val_down = nullptr;  // [hidden, ane_inter] FP32
};

// ---------------------------------------------------------------------------
// Dispatch context
// ---------------------------------------------------------------------------

// Minimum spatial (seq_len) for CoreML/ANE compilation.
// Small spatial dims cause zeroed outputs on ANE hardware due to tile alignment.
static constexpr int ANE_MIN_SPATIAL = 64;

struct ane_dispatch_ctx {
    int hidden_dim     = 0;
    int full_inter_dim = 0;
    int gpu_inter_dim  = 0;
    int ane_inter_dim  = 0;
    float split_ratio  = 0.5f;
    int n_layers       = 0;
    int seq_len        = 0;      // requested seq_len
    int compiled_seq   = 0;      // padded seq_len used for kernel compilation

    std::string cache_dir;
    std::string python_path;

    std::vector<ane_layer_state> layers;
};

// ---------------------------------------------------------------------------
// Transpose helpers: row-major [S,C] ↔ channel-first [1,C,1,S]
// ---------------------------------------------------------------------------

// Row-major FP16 [S, C] → channel-first FP16 [1, C, 1, spatial_stride]
// S_actual: number of rows with real data
// spatial_stride: stride between channels (may be > S_actual for padding)
static void transpose_to_channel_first_fp16(const void * src, void * dst,
                                             int S_actual, int C, int spatial_stride) {
    const uint16_t * s = (const uint16_t *)src;
    uint16_t * d = (uint16_t *)dst;
    // src[s*C + c] → dst[c*spatial_stride + s]
    for (int seq = 0; seq < S_actual; seq++) {
        for (int ch = 0; ch < C; ch++) {
            d[ch * spatial_stride + seq] = s[seq * C + ch];
        }
    }
}

// Channel-first FP16 [1, C, 1, spatial_stride] → row-major FP16 [S, C]
// S_actual: number of rows to extract
// spatial_stride: stride between channels in source
static void transpose_from_channel_first_fp16(const void * src, void * dst,
                                               int S_actual, int C, int spatial_stride) {
    const uint16_t * s = (const uint16_t *)src;
    uint16_t * d = (uint16_t *)dst;
    // src[c*spatial_stride + s] → dst[s*C + c]
    for (int seq = 0; seq < S_actual; seq++) {
        for (int ch = 0; ch < C; ch++) {
            d[seq * C + ch] = s[ch * spatial_stride + seq];
        }
    }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

ane_dispatch_ctx_t ane_dispatch_init(
    const struct llama_hparams * hparams,
    float split_ratio,
    const char * cache_dir,
    const char * python_path)
{
    if (!hparams || split_ratio <= 0.0f || split_ratio >= 1.0f) return nullptr;

    auto * ctx = new ane_dispatch_ctx();

    ctx->hidden_dim = (int)hparams->n_embd;
    // Use first layer's n_ff as reference (most models have uniform FFN dims)
    ctx->full_inter_dim = (int)hparams->n_ff_arr[0];
    ctx->split_ratio = split_ratio;

    // Compute split dimensions (round ANE portion down to multiple of 16 for ANE alignment)
    int ane_raw = (int)(ctx->full_inter_dim * split_ratio);
    ctx->ane_inter_dim = (ane_raw / 16) * 16;
    if (ctx->ane_inter_dim < 16) ctx->ane_inter_dim = 16;

    // Cap ane_inter_dim to keep per-layer CoreML model weight size under ~150MB.
    // Total baked weights: 3 * ane_inter * hidden * sizeof(FP16) = 6 * ane_inter * hidden.
    // Limit: 150 * 1024 * 1024 bytes → ane_inter_max = 150MB / (6 * hidden)
    const int64_t max_model_bytes = 150LL * 1024 * 1024;
    int ane_inter_max = (int)(max_model_bytes / (6LL * ctx->hidden_dim));
    ane_inter_max = (ane_inter_max / 16) * 16;  // round down to multiple of 16
    if (ane_inter_max < 16) ane_inter_max = 16;

    if (ctx->ane_inter_dim > ane_inter_max) {
        fprintf(stderr, "[ANE-dispatch] auto-capping ane_inter from %d to %d (model weight limit ~150MB)\n",
                ctx->ane_inter_dim, ane_inter_max);
        ctx->ane_inter_dim = ane_inter_max;
    }

    ctx->gpu_inter_dim = ctx->full_inter_dim - ctx->ane_inter_dim;

    ctx->n_layers = (int)hparams->n_layer;

    if (cache_dir)   ctx->cache_dir = cache_dir;
    if (python_path) ctx->python_path = python_path;

    fprintf(stderr, "[ANE-dispatch] init: hidden=%d full_inter=%d gpu_inter=%d ane_inter=%d ratio=%.2f\n",
            ctx->hidden_dim, ctx->full_inter_dim, ctx->gpu_inter_dim, ctx->ane_inter_dim,
            (float)ctx->ane_inter_dim / ctx->full_inter_dim);

    return ctx;
}

// ---------------------------------------------------------------------------
// Compile kernels
// ---------------------------------------------------------------------------

bool ane_dispatch_compile_kernels(
    ane_dispatch_ctx_t ctx,
    const struct llama_layer * layers,
    int n_layers,
    int seq_len)
{
    if (!ctx || !layers || n_layers <= 0 || seq_len <= 0) return false;

    ctx->seq_len = seq_len;
    // Pad spatial dimension to ANE minimum to avoid tile alignment issues
    ctx->compiled_seq = std::max(seq_len, ANE_MIN_SPATIAL);
    if (ctx->compiled_seq != seq_len) {
        fprintf(stderr, "[ANE-dispatch] padding spatial from %d to %d (ANE minimum)\n",
                seq_len, ctx->compiled_seq);
    }
    ctx->layers.resize(n_layers);

    const int hidden = ctx->hidden_dim;
    const int ane_inter = ctx->ane_inter_dim;
    const int gpu_inter = ctx->gpu_inter_dim;

    const char * py = ctx->python_path.empty() ? nullptr : ctx->python_path.c_str();
    const char * cache = ctx->cache_dir.empty() ? nullptr : ctx->cache_dir.c_str();

    for (int il = 0; il < n_layers; il++) {
        auto & ls = ctx->layers[il];
        const llama_layer & layer = layers[il];

        // Skip MoE layers (they have ffn_gate_inp)
        if (layer.ffn_gate_inp != nullptr) {
            ls.active = false;
            fprintf(stderr, "[ANE-dispatch] layer %d: MoE, skipping\n", il);
            continue;
        }

        // Need gate, up, down tensors
        if (!layer.ffn_gate || !layer.ffn_up || !layer.ffn_down) {
            ls.active = false;
            fprintf(stderr, "[ANE-dispatch] layer %d: missing FFN tensors, skipping\n", il);
            continue;
        }

        // Check that layer's FFN dimension matches expected
        int layer_inter = (int)layer.ffn_gate->ne[1];  // gate is [hidden, inter] → ne[1] = inter
        if (layer_inter != ctx->full_inter_dim) {
            ls.active = false;
            fprintf(stderr, "[ANE-dispatch] layer %d: inter_dim mismatch (%d vs %d), skipping\n",
                    il, layer_inter, ctx->full_inter_dim);
            continue;
        }

        ls.active = true;

        // --- Extract and compile ANE weights ---

        // Dequantize full tensors to FP32
        float * gate_full = ane_dequantize_tensor(layer.ffn_gate);
        float * up_full   = ane_dequantize_tensor(layer.ffn_up);
        float * down_full = ane_dequantize_tensor(layer.ffn_down);

        if (!gate_full || !up_full || !down_full) {
            free(gate_full); free(up_full); free(down_full);
            ls.active = false;
            fprintf(stderr, "[ANE-dispatch] layer %d: dequantize failed\n", il);
            continue;
        }

        // gate [inter_dim, hidden_dim] — ANE gets rows [gpu_inter, full_inter)
        // Extract ANE portion: rows gpu_inter..full_inter
        float * ane_gate = (float *)malloc(ane_inter * hidden * sizeof(float));
        float * ane_up   = (float *)malloc(ane_inter * hidden * sizeof(float));
        float * ane_down = (float *)malloc(hidden * ane_inter * sizeof(float));

        if (!ane_gate || !ane_up || !ane_down) {
            free(gate_full); free(up_full); free(down_full);
            free(ane_gate); free(ane_up); free(ane_down);
            ls.active = false;
            continue;
        }

        // gate/up: row-major [inter, hidden] → ANE portion is last ane_inter rows
        memcpy(ane_gate, gate_full + (size_t)gpu_inter * hidden, (size_t)ane_inter * hidden * sizeof(float));
        memcpy(ane_up,   up_full   + (size_t)gpu_inter * hidden, (size_t)ane_inter * hidden * sizeof(float));

        // down: row-major [hidden, inter] → ANE portion is last ane_inter columns
        for (int r = 0; r < hidden; r++) {
            memcpy(ane_down + (size_t)r * ane_inter,
                   down_full + (size_t)r * ctx->full_inter_dim + gpu_inter,
                   ane_inter * sizeof(float));
        }

        // Compile fused FFN kernel for this layer (per-layer cache dir for unique weights)
        std::string layer_cache;
        const char * layer_cache_ptr = cache;
        if (cache) {
            char buf[32];
            snprintf(buf, sizeof(buf), "/layer_%d", il);
            layer_cache = std::string(cache) + buf;
            layer_cache_ptr = layer_cache.c_str();
        }
        ls.kernel = ggml_ane_compile_ffn(py, hidden, ane_inter, ctx->compiled_seq,
                                          ane_gate, ane_up, ane_down, layer_cache_ptr);

        // Keep validation weights for layer 0 when ANE_VALIDATE=1
        static const bool do_validate = (getenv("ANE_VALIDATE") != nullptr);
        if (do_validate && il == 0) {
            ls.val_gate = (float *)malloc((size_t)ane_inter * hidden * sizeof(float));
            ls.val_up   = (float *)malloc((size_t)ane_inter * hidden * sizeof(float));
            ls.val_down = (float *)malloc((size_t)hidden * ane_inter * sizeof(float));
            memcpy(ls.val_gate, ane_gate, (size_t)ane_inter * hidden * sizeof(float));
            memcpy(ls.val_up,   ane_up,   (size_t)ane_inter * hidden * sizeof(float));
            memcpy(ls.val_down, ane_down, (size_t)hidden * ane_inter * sizeof(float));
            fprintf(stderr, "[ANE-validate] saved layer 0 ANE-half weights for CPU reference\n");
        }

        free(ane_gate); free(ane_up); free(ane_down);

        if (!ls.kernel) {
            fprintf(stderr, "[ANE-dispatch] layer %d: FFN kernel compile failed\n", il);
            ls.active = false;
            free(gate_full); free(up_full); free(down_full);
            continue;
        }

        // --- Pre-compute GPU-half FP16 weights ---

        // gate GPU half: first gpu_inter rows → [gpu_inter, hidden] FP16
        ls.gpu_gate_bytes = (size_t)gpu_inter * hidden * sizeof(ggml_fp16_t);
        ls.gpu_gate_f16 = malloc(ls.gpu_gate_bytes);
        ane_convert_f32_to_fp16(gate_full, ls.gpu_gate_f16, (int64_t)gpu_inter * hidden);

        // up GPU half: first gpu_inter rows → [gpu_inter, hidden] FP16
        ls.gpu_up_bytes = (size_t)gpu_inter * hidden * sizeof(ggml_fp16_t);
        ls.gpu_up_f16 = malloc(ls.gpu_up_bytes);
        ane_convert_f32_to_fp16(up_full, ls.gpu_up_f16, (int64_t)gpu_inter * hidden);

        // down GPU half: first gpu_inter columns → [hidden, gpu_inter] FP16
        float * gpu_down_f32 = (float *)malloc((size_t)hidden * gpu_inter * sizeof(float));
        for (int r = 0; r < hidden; r++) {
            memcpy(gpu_down_f32 + (size_t)r * gpu_inter,
                   down_full + (size_t)r * ctx->full_inter_dim,
                   gpu_inter * sizeof(float));
        }
        ls.gpu_down_bytes = (size_t)hidden * gpu_inter * sizeof(ggml_fp16_t);
        ls.gpu_down_f16 = malloc(ls.gpu_down_bytes);
        ane_convert_f32_to_fp16(gpu_down_f32, ls.gpu_down_f16, (int64_t)hidden * gpu_inter);
        free(gpu_down_f32);

        free(gate_full); free(up_full); free(down_full);

        // --- Allocate transpose buffers ---
        ls.inp_trans_bytes = (size_t)hidden * ctx->compiled_seq * sizeof(uint16_t);
        ls.out_trans_bytes = (size_t)hidden * ctx->compiled_seq * sizeof(uint16_t);
        ls.inp_transposed = malloc(ls.inp_trans_bytes);
        ls.out_transposed = malloc(ls.out_trans_bytes);

#ifdef __APPLE__
        // Create per-layer dispatch group and serial queue
        ls.group = dispatch_group_create();
        char qname[64];
        snprintf(qname, sizeof(qname), "com.ggml.ane.layer.%d", il);
        ls.queue = dispatch_queue_create(qname, DISPATCH_QUEUE_SERIAL);
#endif

        fprintf(stderr, "[ANE-dispatch] layer %d: compiled (gpu_inter=%d, ane_inter=%d)\n",
                il, gpu_inter, ane_inter);
    }

    return true;
}

// ---------------------------------------------------------------------------
// Async dispatch
// ---------------------------------------------------------------------------

void ane_dispatch_ffn_async(
    ane_dispatch_ctx_t ctx,
    int il,
    const void * ffn_inp_f16,
    int seq_len)
{
    if (!ctx || il < 0 || il >= ctx->n_layers) return;
    auto & ls = ctx->layers[il];
    if (!ls.active || !ls.kernel || !ffn_inp_f16) return;

    const int hidden = ctx->hidden_dim;
    const int S = seq_len;
    const int CS = ctx->compiled_seq;  // padded spatial dimension
    ls.actual_seq = S;

    // Zero the transpose buffer first (padding for spatial positions beyond S)
    memset(ls.inp_transposed, 0, ls.inp_trans_bytes);

    // Transpose input: row-major [S, hidden] → channel-first [1, hidden, 1, CS]
    // Only fills positions 0..S-1; positions S..CS-1 remain zero
    transpose_to_channel_first_fp16(ffn_inp_f16, ls.inp_transposed, S, hidden, CS);

    // Write full compiled_seq buffer to ANE kernel input
    size_t inp_bytes = (size_t)hidden * CS * sizeof(uint16_t);
    ggml_ane_write_input(ls.kernel, 0, ls.inp_transposed, inp_bytes);

#ifdef __APPLE__
    ls.pending = true;
    dispatch_group_enter(ls.group);

    // Capture kernel pointer for the block
    ggml_ane_kernel_t k = ls.kernel;
    dispatch_group_t g = ls.group;

    dispatch_async(ls.queue, ^{
        ggml_ane_eval(k);
        dispatch_group_leave(g);
    });
#else
    // Fallback: synchronous eval
    ggml_ane_eval(ls.kernel);
    ls.pending = true;
#endif
}

// ---------------------------------------------------------------------------
// Sync + read output
// ---------------------------------------------------------------------------

float ane_dispatch_ffn_sync(
    ane_dispatch_ctx_t ctx,
    int il,
    void * out_f16,
    size_t nbytes)
{
    if (!ctx || il < 0 || il >= ctx->n_layers) return 0.0f;
    auto & ls = ctx->layers[il];
    if (!ls.active || !ls.pending) return 0.0f;

    double t0 = ane_dispatch_time_ms();

#ifdef __APPLE__
    dispatch_group_wait(ls.group, DISPATCH_TIME_FOREVER);
#endif

    ls.pending = false;

    double t1 = ane_dispatch_time_ms();

    // Read ANE output (channel-first [1, hidden, 1, compiled_seq])
    size_t out_ch_bytes = (size_t)ctx->hidden_dim * ctx->compiled_seq * sizeof(uint16_t);
    ggml_ane_read_output_after_eval(ls.kernel, 0, ls.out_transposed, out_ch_bytes);
    (void)nbytes;

    // Transpose back only the actual_seq positions: channel-first → row-major [S, hidden]
    const int S = ls.actual_seq;
    transpose_from_channel_first_fp16(ls.out_transposed, out_f16, S, ctx->hidden_dim, ctx->compiled_seq);

    return (float)(t1 - t0);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int ane_dispatch_gpu_inter_dim(ane_dispatch_ctx_t ctx) {
    return ctx ? ctx->gpu_inter_dim : 0;
}

int ane_dispatch_ane_inter_dim(ane_dispatch_ctx_t ctx) {
    return ctx ? ctx->ane_inter_dim : 0;
}

bool ane_dispatch_layer_active(ane_dispatch_ctx_t ctx, int il) {
    if (!ctx || il < 0 || il >= ctx->n_layers) return false;
    return ctx->layers[il].active;
}

bool ane_dispatch_is_ready(ane_dispatch_ctx_t ctx, int seq_len) {
    if (!ctx || ctx->n_layers <= 0 || ctx->layers.empty()) return false;
    // Kernels compiled for padded spatial can handle any seq_len that pads to the same value
    int padded = std::max(seq_len, ANE_MIN_SPATIAL);
    return ctx->compiled_seq == padded;
}

const void * ane_dispatch_gpu_weight(
    ane_dispatch_ctx_t ctx,
    int il,
    const char * tensor_name)
{
    if (!ctx || il < 0 || il >= ctx->n_layers || !tensor_name) return nullptr;
    const auto & ls = ctx->layers[il];
    if (!ls.active) return nullptr;

    if (strcmp(tensor_name, "ffn_gate") == 0) return ls.gpu_gate_f16;
    if (strcmp(tensor_name, "ffn_up")   == 0) return ls.gpu_up_f16;
    if (strcmp(tensor_name, "ffn_down") == 0) return ls.gpu_down_f16;
    return nullptr;
}

size_t ane_dispatch_gpu_weight_bytes(
    ane_dispatch_ctx_t ctx,
    int il,
    const char * tensor_name)
{
    if (!ctx || il < 0 || il >= ctx->n_layers || !tensor_name) return 0;
    const auto & ls = ctx->layers[il];
    if (!ls.active) return 0;

    if (strcmp(tensor_name, "ffn_gate") == 0) return ls.gpu_gate_bytes;
    if (strcmp(tensor_name, "ffn_up")   == 0) return ls.gpu_up_bytes;
    if (strcmp(tensor_name, "ffn_down") == 0) return ls.gpu_down_bytes;
    return 0;
}

// ---------------------------------------------------------------------------
// Validation: CPU reference vs ANE output for layer 0
// ---------------------------------------------------------------------------

static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

bool ane_dispatch_validate_layer0(
    ane_dispatch_ctx_t ctx,
    const float * normed_input_f32,  // row-major [S, hidden], post-RMS-norm
    const void * ane_output_f16,     // row-major [S, hidden], FP16 from ANE
    int seq_len)
{
    if (!ctx || ctx->layers.empty()) return false;
    const auto & ls = ctx->layers[0];
    if (!ls.val_gate || !ls.val_up || !ls.val_down) {
        fprintf(stderr, "[ANE-validate] no validation weights (set ANE_VALIDATE=1)\n");
        return false;
    }

    const int H = ctx->hidden_dim;
    const int A = ctx->ane_inter_dim;
    const int S = seq_len;

    // Allocate temporaries for CPU reference: gate_out[S,A], up_out[S,A], act[S,A], out[S,H]
    std::vector<float> gate_out(S * A), up_out(S * A), act(S * A), ref_out(S * H, 0.0f);

    // gate_out = input @ gate^T : [S,H] x [A,H]^T = [S,A]
    // up_out   = input @ up^T   : [S,H] x [A,H]^T = [S,A]
    for (int s = 0; s < S; s++) {
        const float * inp_row = normed_input_f32 + s * H;
        for (int a = 0; a < A; a++) {
            float g = 0.0f, u = 0.0f;
            const float * gw = ls.val_gate + a * H;  // gate row a: [H]
            const float * uw = ls.val_up   + a * H;  // up row a: [H]
            for (int h = 0; h < H; h++) {
                g += inp_row[h] * gw[h];
                u += inp_row[h] * uw[h];
            }
            gate_out[s * A + a] = g;
            up_out[s * A + a]   = u;
        }
    }

    // act = SiLU(gate_out) * up_out
    for (int i = 0; i < S * A; i++) {
        act[i] = silu_f(gate_out[i]) * up_out[i];
    }

    // ref_out = act @ down^T : [S,A] x [H,A]^T = [S,H]
    // down is [H, A] row-major: down[h, a]
    for (int s = 0; s < S; s++) {
        for (int h = 0; h < H; h++) {
            float sum = 0.0f;
            const float * dw = ls.val_down + h * A;  // down row h: [A]
            const float * ar = act.data() + s * A;
            for (int a = 0; a < A; a++) {
                sum += ar[a] * dw[a];
            }
            ref_out[s * H + h] = sum;
        }
    }

    // Compare ref_out vs ane_output_f16
    const uint16_t * ane_f16 = (const uint16_t *)ane_output_f16;
    float max_abs_err = 0.0f;
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    int max_err_idx = 0;
    for (int i = 0; i < S * H; i++) {
        float ane_val = ggml_fp16_to_fp32(ane_f16[i]);
        float ref_val = ref_out[i];
        float err = fabsf(ane_val - ref_val);
        if (err > max_abs_err) {
            max_abs_err = err;
            max_err_idx = i;
        }
        sum_sq_err += (double)(ane_val - ref_val) * (ane_val - ref_val);
        sum_sq_ref += (double)ref_val * ref_val;
    }
    float rmse = (float)sqrt(sum_sq_err / (S * H));
    float ref_rms = (float)sqrt(sum_sq_ref / (S * H));
    float rel_err = ref_rms > 0 ? rmse / ref_rms : 0.0f;

    int s_idx = max_err_idx / H;
    int h_idx = max_err_idx % H;
    float ane_at_max = ggml_fp16_to_fp32(ane_f16[max_err_idx]);
    float ref_at_max = ref_out[max_err_idx];

    fprintf(stderr, "[ANE-validate] layer 0 CPU ref vs ANE:\n");
    fprintf(stderr, "  max_abs_err = %.6f at [%d,%d] (ane=%.6f ref=%.6f)\n",
            max_abs_err, s_idx, h_idx, ane_at_max, ref_at_max);
    fprintf(stderr, "  RMSE = %.6f, ref_RMS = %.6f, relative = %.4f%%\n",
            rmse, ref_rms, rel_err * 100.0f);
    fprintf(stderr, "  verdict: %s\n",
            max_abs_err < 0.1f ? "PASS (< 0.1)" :
            max_abs_err < 1.0f ? "MARGINAL (< 1.0)" : "FAIL (>= 1.0)");

    return max_abs_err < 0.1f;
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void ane_dispatch_free(ane_dispatch_ctx_t ctx) {
    if (!ctx) return;

    for (auto & ls : ctx->layers) {
        if (ls.kernel) {
            ggml_ane_free(ls.kernel);
            ls.kernel = nullptr;
        }
        free(ls.gpu_gate_f16);
        free(ls.gpu_up_f16);
        free(ls.gpu_down_f16);
        free(ls.inp_transposed);
        free(ls.out_transposed);
        free(ls.val_gate);
        free(ls.val_up);
        free(ls.val_down);
        ls.gpu_gate_f16 = nullptr;
        ls.gpu_up_f16   = nullptr;
        ls.gpu_down_f16 = nullptr;
        ls.inp_transposed = nullptr;
        ls.out_transposed = nullptr;
        ls.val_gate = nullptr;
        ls.val_up   = nullptr;
        ls.val_down = nullptr;

#ifdef __APPLE__
        if (ls.group) dispatch_release(ls.group);
        if (ls.queue) dispatch_release(ls.queue);
        ls.group = nullptr;
        ls.queue = nullptr;
#endif
    }

    delete ctx;
}
