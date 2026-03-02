// test-ane-dispatch.cpp — Phase 3: ANE FFN dispatch correctness test
//
// Loads a real GGUF model, compiles an FFN kernel for one layer,
// runs with known input, and compares vs CPU reference:
//   output = SiLU(x @ gate) * (x @ up) @ down
// Expects max abs error < 0.01.
//
// Usage: test-ane-dispatch -m /path/to/model.gguf [-l layer_idx] [-s seq_len]

#include "llama.h"
#include "llama-ane-dispatch.h"
#include "llama-ane-weights.h"
#include "llama-model.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s -m <model.gguf> [-l layer] [-s seq_len] [-r ratio]\n", prog);
}

// SiLU activation: x * sigmoid(x)
static float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// CPU reference FFN: output[S,H] = SiLU(inp[S,H] @ gate[I,H]^T) * (inp[S,H] @ up[I,H]^T) @ down[H,I]^T
// gate: [inter, hidden], up: [inter, hidden], down: [hidden, inter]
// inp: [seq, hidden], out: [seq, hidden]
static void cpu_ffn_reference(
    const float * inp,       // [seq, hidden]
    const float * gate_w,    // [inter, hidden]
    const float * up_w,      // [inter, hidden]
    const float * down_w,    // [hidden, inter]
    float * out,             // [seq, hidden]
    int seq, int hidden, int inter)
{
    // Temp buffers for intermediate values
    std::vector<float> gate_out(seq * inter);
    std::vector<float> up_out(seq * inter);
    std::vector<float> act_out(seq * inter);

    // gate_out[s,i] = sum_h(inp[s,h] * gate_w[i,h])
    for (int s = 0; s < seq; s++) {
        for (int i = 0; i < inter; i++) {
            float sum = 0.0f;
            for (int h = 0; h < hidden; h++) {
                sum += inp[s * hidden + h] * gate_w[i * hidden + h];
            }
            gate_out[s * inter + i] = sum;
        }
    }

    // up_out[s,i] = sum_h(inp[s,h] * up_w[i,h])
    for (int s = 0; s < seq; s++) {
        for (int i = 0; i < inter; i++) {
            float sum = 0.0f;
            for (int h = 0; h < hidden; h++) {
                sum += inp[s * hidden + h] * up_w[i * hidden + h];
            }
            up_out[s * inter + i] = sum;
        }
    }

    // act_out = SiLU(gate_out) * up_out
    for (int k = 0; k < seq * inter; k++) {
        act_out[k] = silu_f32(gate_out[k]) * up_out[k];
    }

    // out[s,h] = sum_i(act_out[s,i] * down_w[h,i])
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < hidden; h++) {
            float sum = 0.0f;
            for (int i = 0; i < inter; i++) {
                sum += act_out[s * inter + i] * down_w[h * inter + i];
            }
            out[s * hidden + h] = sum;
        }
    }
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int layer_idx = 0;
    int seq_len = 32;
    float ratio = 0.5f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            layer_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            ratio = atof(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!model_path) {
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "=== ANE Dispatch Correctness Test ===\n");
    fprintf(stderr, "Model: %s\n", model_path);
    fprintf(stderr, "Layer: %d, Seq: %d, Ratio: %.2f\n", layer_idx, seq_len, ratio);

    // Load model
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU only for weight access

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "FAIL: could not load model\n");
        return 1;
    }

    const auto & hparams = model->hparams;
    const int n_layer = (int)hparams.n_layer;
    const int hidden = (int)hparams.n_embd;
    const int full_inter = (int)hparams.n_ff_arr[0];

    fprintf(stderr, "Model: n_layer=%d, n_embd=%d, n_ff=%d\n", n_layer, hidden, full_inter);

    if (layer_idx < 0 || layer_idx >= n_layer) {
        fprintf(stderr, "FAIL: layer %d out of range [0, %d)\n", layer_idx, n_layer);
        llama_free_model(model);
        return 1;
    }

    const llama_layer & layer = model->layers[layer_idx];
    if (!layer.ffn_gate || !layer.ffn_up || !layer.ffn_down) {
        fprintf(stderr, "FAIL: layer %d missing FFN tensors\n", layer_idx);
        llama_free_model(model);
        return 1;
    }

    // Initialize ANE dispatch
    ane_dispatch_ctx_t ctx = ane_dispatch_init(&hparams, ratio, nullptr, nullptr);
    if (!ctx) {
        fprintf(stderr, "FAIL: ane_dispatch_init failed\n");
        llama_free_model(model);
        return 1;
    }

    int ane_inter = ane_dispatch_ane_inter_dim(ctx);
    int gpu_inter = ane_dispatch_gpu_inter_dim(ctx);
    fprintf(stderr, "Split: gpu_inter=%d, ane_inter=%d\n", gpu_inter, ane_inter);

    // Compile kernel for just this one layer
    fprintf(stderr, "Compiling FFN kernel...\n");
    if (!ane_dispatch_compile_kernels(ctx, model->layers.data(), n_layer, seq_len)) {
        fprintf(stderr, "FAIL: kernel compilation failed\n");
        ane_dispatch_free(ctx);
        llama_free_model(model);
        return 1;
    }

    if (!ane_dispatch_layer_active(ctx, layer_idx)) {
        fprintf(stderr, "FAIL: layer %d not active after compile\n", layer_idx);
        ane_dispatch_free(ctx);
        llama_free_model(model);
        return 1;
    }

    // Dequantize full weights for CPU reference
    float * gate_full = ane_dequantize_tensor(layer.ffn_gate);
    float * up_full   = ane_dequantize_tensor(layer.ffn_up);
    float * down_full = ane_dequantize_tensor(layer.ffn_down);

    if (!gate_full || !up_full || !down_full) {
        fprintf(stderr, "FAIL: dequantize failed\n");
        free(gate_full); free(up_full); free(down_full);
        ane_dispatch_free(ctx);
        llama_free_model(model);
        return 1;
    }

    // Generate random FP32 input
    std::vector<float> inp_f32(seq_len * hidden);
    srand(42);
    for (int i = 0; i < seq_len * hidden; i++) {
        inp_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // --- CPU reference (ANE portion only) ---
    // Extract ANE-portion weights
    float * ane_gate = (float *)malloc(ane_inter * hidden * sizeof(float));
    float * ane_up   = (float *)malloc(ane_inter * hidden * sizeof(float));
    float * ane_down = (float *)malloc(hidden * ane_inter * sizeof(float));

    memcpy(ane_gate, gate_full + (size_t)gpu_inter * hidden, (size_t)ane_inter * hidden * sizeof(float));
    memcpy(ane_up,   up_full   + (size_t)gpu_inter * hidden, (size_t)ane_inter * hidden * sizeof(float));
    for (int r = 0; r < hidden; r++) {
        memcpy(ane_down + (size_t)r * ane_inter,
               down_full + (size_t)r * full_inter + gpu_inter,
               ane_inter * sizeof(float));
    }

    std::vector<float> ref_out(seq_len * hidden, 0.0f);
    cpu_ffn_reference(inp_f32.data(), ane_gate, ane_up, ane_down, ref_out.data(),
                      seq_len, hidden, ane_inter);

    free(ane_gate); free(ane_up); free(ane_down);
    free(gate_full); free(up_full); free(down_full);

    // --- ANE computation ---
    // Convert input to FP16
    std::vector<uint16_t> inp_f16(seq_len * hidden);
    for (int i = 0; i < seq_len * hidden; i++) {
        inp_f16[i] = ggml_fp32_to_fp16(inp_f32[i]);
    }

    // Dispatch async and sync
    ane_dispatch_ffn_async(ctx, layer_idx, inp_f16.data(), seq_len);
    std::vector<uint16_t> ane_out_f16(seq_len * hidden, 0);
    float wait_ms = ane_dispatch_ffn_sync(ctx, layer_idx, ane_out_f16.data(),
                                           ane_out_f16.size() * sizeof(uint16_t));

    fprintf(stderr, "ANE wait: %.2f ms\n", wait_ms);

    // Convert ANE output to FP32
    std::vector<float> ane_out_f32(seq_len * hidden);
    for (int i = 0; i < seq_len * hidden; i++) {
        ane_out_f32[i] = ggml_fp16_to_fp32(ane_out_f16[i]);
    }

    // --- Compare ---
    float max_abs_err = 0.0f;
    float sum_abs_err = 0.0f;
    int n_elements = seq_len * hidden;

    for (int i = 0; i < n_elements; i++) {
        float err = fabsf(ane_out_f32[i] - ref_out[i]);
        if (err > max_abs_err) max_abs_err = err;
        sum_abs_err += err;
    }

    float mean_abs_err = sum_abs_err / n_elements;

    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Max abs error:  %.6f\n", max_abs_err);
    fprintf(stderr, "Mean abs error: %.6f\n", mean_abs_err);

    // Print first few values for debugging
    fprintf(stderr, "\nFirst 5 values (ref vs ane):\n");
    for (int i = 0; i < 5 && i < n_elements; i++) {
        fprintf(stderr, "  [%d] ref=%.6f ane=%.6f err=%.6f\n",
                i, ref_out[i], ane_out_f32[i], fabsf(ref_out[i] - ane_out_f32[i]));
    }

    bool pass = max_abs_err < 0.01f;
    fprintf(stderr, "\n%s (max_abs_err=%.6f, threshold=0.01)\n",
            pass ? "PASS" : "FAIL", max_abs_err);

    ane_dispatch_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return pass ? 0 : 1;
}
