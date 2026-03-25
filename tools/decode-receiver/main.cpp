// decode-receiver — standalone Mac-side daemon for hybrid RTX+Mac inference
//
// Loads a model, listens for KV artifacts over TCP (or reads from file),
// imports the KV cache, runs autoregressive decode, and streams tokens
// back to the RTX host via the token stream protocol.
//
// Usage:
//   # Network mode (listens for KV from RTX):
//   ./decode-receiver -m model.gguf --kv-port 9100 --token-stream-port 9101
//
//   # File mode (testing):
//   ./decode-receiver -m model.gguf --kv-artifact kv_cache.bin

#include "llama.h"

#include "llama-kv-artifact.h"
#include "llama-kv-import.h"
#include "llama-kv-receiver.h"
#include "llama-token-stream.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct receiver_params {
    std::string model_path;
    std::string kv_artifact_path;  // file mode
    std::string kv_host       = "0.0.0.0";
    int         kv_port       = 9100;
    std::string ts_host       = "0.0.0.0";
    int         ts_port       = 9101;
    int         n_ctx         = 8192;
    int         n_gpu_layers  = 99;
    int         n_predict     = 2048;
    float       temp          = 0.7f;
    bool        mlock         = false;
    bool        one_shot      = false;
};

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model PATH           Model file (GGUF)\n");
    fprintf(stderr, "  --kv-artifact PATH         File mode: read KV from file\n");
    fprintf(stderr, "  --kv-host HOST             Network mode: bind host (default: 0.0.0.0)\n");
    fprintf(stderr, "  --kv-port PORT             Network mode: listen port (default: 9100)\n");
    fprintf(stderr, "  --token-stream-host HOST   Token relay host (default: 0.0.0.0)\n");
    fprintf(stderr, "  --token-stream-port PORT   Token relay port (default: 9101)\n");
    fprintf(stderr, "  -c, --ctx-size N           Context size (default: 8192)\n");
    fprintf(stderr, "  -ngl, --n-gpu-layers N     GPU layers (default: 99)\n");
    fprintf(stderr, "  -n, --predict N            Max decode tokens (default: 2048)\n");
    fprintf(stderr, "  --temp F                   Temperature (default: 0.7)\n");
    fprintf(stderr, "  --mlock                    Lock model in memory\n");
    fprintf(stderr, "  --one-shot                 Exit after first decode\n");
}

static bool parse_args(int argc, char ** argv, receiver_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };

        if (arg == "-m" || arg == "--model") {
            const char * v = next(); if (!v) return false;
            params.model_path = v;
        } else if (arg == "--kv-artifact") {
            const char * v = next(); if (!v) return false;
            params.kv_artifact_path = v;
        } else if (arg == "--kv-host") {
            const char * v = next(); if (!v) return false;
            params.kv_host = v;
        } else if (arg == "--kv-port") {
            const char * v = next(); if (!v) return false;
            params.kv_port = std::atoi(v);
        } else if (arg == "--token-stream-host") {
            const char * v = next(); if (!v) return false;
            params.ts_host = v;
        } else if (arg == "--token-stream-port") {
            const char * v = next(); if (!v) return false;
            params.ts_port = std::atoi(v);
        } else if (arg == "-c" || arg == "--ctx-size") {
            const char * v = next(); if (!v) return false;
            params.n_ctx = std::atoi(v);
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            const char * v = next(); if (!v) return false;
            params.n_gpu_layers = std::atoi(v);
        } else if (arg == "-n" || arg == "--predict") {
            const char * v = next(); if (!v) return false;
            params.n_predict = std::atoi(v);
        } else if (arg == "--temp") {
            const char * v = next(); if (!v) return false;
            params.temp = (float)std::atof(v);
        } else if (arg == "--mlock") {
            params.mlock = true;
        } else if (arg == "--one-shot") {
            params.one_shot = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return !params.model_path.empty();
}

int main(int argc, char ** argv) {
    receiver_params params;
    if (!parse_args(argc, argv, params)) {
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    fprintf(stderr, "Loading model: %s\n", params.model_path.c_str());
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers;
    mparams.use_mlock = params.mlock;

    llama_model * model = llama_model_load_from_file(params.model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = params.n_ctx;
    cparams.n_batch = params.n_ctx;
    cparams.flash_attn = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_free_model(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    fprintf(stderr, "Model loaded: %d layers, n_ctx=%d\n",
            llama_n_layer(model), params.n_ctx);

    // Session loop
    bool keep_running = true;
    while (keep_running) {
        // Step 1: Receive KV artifact
        std::vector<uint8_t> artifact_data;
        std::string err;

        if (!params.kv_artifact_path.empty()) {
            // File mode: read raw file bytes (header + payload)
            fprintf(stderr, "Reading KV artifact from file: %s\n", params.kv_artifact_path.c_str());
            std::ifstream ifs(params.kv_artifact_path, std::ios::binary | std::ios::ate);
            if (!ifs.is_open()) {
                fprintf(stderr, "Failed to open artifact file: %s\n", params.kv_artifact_path.c_str());
                break;
            }
            size_t file_size = (size_t)ifs.tellg();
            ifs.seekg(0);
            artifact_data.resize(file_size);
            ifs.read(reinterpret_cast<char *>(artifact_data.data()), (std::streamsize)file_size);
            fprintf(stderr, "Read %zu bytes from artifact file\n", file_size);
        } else {
            // Network mode
            fprintf(stderr, "Listening for KV artifact on %s:%d...\n",
                    params.kv_host.c_str(), params.kv_port);

            llama_kv_receiver_config rcfg;
            rcfg.bind_host = params.kv_host;
            rcfg.port      = params.kv_port;

            llama_kv_receiver_result rresult;
            if (!llama_kv_receiver_accept_artifact(rcfg, artifact_data, &rresult, &err)) {
                fprintf(stderr, "Failed to receive artifact: %s\n", err.c_str());
                if (params.one_shot) break;
                continue;
            }
            fprintf(stderr, "Received %zu bytes in %.1f ms (%.2f Gbps), %u chunks\n",
                    (size_t)rresult.bytes_received, rresult.receive_ms,
                    rresult.throughput_gbps, rresult.chunks_received);
        }

        // Step 2: Parse artifact header from received data
        llama_kv_artifact_metadata meta;
        llama_kv_artifact_summary summary;
        std::vector<uint8_t> payload;
        if (!llama_kv_artifact_read_mem(artifact_data.data(), artifact_data.size(),
                                         payload, &meta, &summary, &err)) {
            fprintf(stderr, "Failed to parse artifact: %s\n", err.c_str());
            if (params.one_shot) break;
            continue;
        }

        fprintf(stderr, "Artifact parsed: %u tokens, %u layers, %zu bytes payload\n",
                meta.token_count, meta.n_layers, payload.size());

        // Step 3: Import KV state into context
        // The payload is a serialized llama state (from llama_state_get_data)
        llama_kv_cache_clear(ctx);

        if (!payload.empty()) {
            size_t loaded = llama_state_set_data(ctx, payload.data(), payload.size());
            if (loaded == 0) {
                fprintf(stderr, "Warning: failed to restore KV state from artifact payload\n");
            } else {
                fprintf(stderr, "KV state restored: %zu bytes loaded\n", loaded);
            }
        }

        // Step 4: Autoregressive decode
        fprintf(stderr, "Starting decode (max %d tokens, temp=%.2f)...\n",
                params.n_predict, params.temp);

        // Start token stream server
        llama_token_stream_server ts_server;
        bool ts_ok = ts_server.start(params.ts_host, params.ts_port);
        if (ts_ok) {
            fprintf(stderr, "Token stream server on %s:%d, waiting for client...\n",
                    params.ts_host.c_str(), ts_server.bound_port());
            ts_ok = ts_server.wait_for_client(30000);
            if (ts_ok) {
                llama_token_handshake hs;
                hs.version = 1;
                hs.status = "ready";
                hs.token_count = meta.token_count;
                ts_server.send_handshake(hs);
            }
        }

        // Simple greedy decode loop (placeholder — production would use sampling)
        int tokens_generated = 0;
        llama_token last_token = llama_vocab_bos(vocab);
        auto t_decode_start = std::chrono::steady_clock::now();

        for (int i = 0; i < params.n_predict; i++) {
            llama_batch batch = llama_batch_init(1, 0, 1);
            batch.n_tokens = 1;
            batch.token[0]     = last_token;
            batch.pos[0]       = (int32_t)(meta.token_count + i);
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = 1;

            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "Decode failed at token %d\n", i);
                llama_batch_free(batch);
                break;
            }

            const float * logits = llama_get_logits_ith(ctx, 0);
            if (!logits) {
                llama_batch_free(batch);
                break;
            }

            // Greedy argmax
            llama_token best = 0;
            float best_val = logits[0];
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > best_val) { best_val = logits[v]; best = v; }
            }

            last_token = best;
            tokens_generated++;

            // Get token text
            char buf[256];
            int n = llama_token_to_piece(model, best, buf, sizeof(buf), 0, true);
            std::string text(buf, n > 0 ? n : 0);

            // Stream token
            if (ts_ok) {
                llama_token_msg msg;
                msg.token_id = best;
                msg.text = text;
                msg.pos = (int32_t)(meta.token_count + i);
                ts_server.enqueue(msg);
            }

            // Print to stderr
            fprintf(stderr, "%s", text.c_str());

            // Check EOS
            if (best == llama_vocab_eos(vocab)) {
                break;
            }

            llama_batch_free(batch);
        }

        auto t_decode_end = std::chrono::steady_clock::now();
        double decode_ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(
            t_decode_end - t_decode_start).count() / 1000.0;
        double tok_s = decode_ms > 0 ? (tokens_generated / (decode_ms / 1000.0)) : 0;

        fprintf(stderr, "\n\nDecode complete: %d tokens in %.1f ms (%.1f tok/s)\n",
                tokens_generated, decode_ms, tok_s);

        // Send done
        if (ts_ok) {
            ts_server.send_done(tokens_generated, tok_s);
        }
        ts_server.stop();

        // Session loop control
        if (params.one_shot || !params.kv_artifact_path.empty()) {
            keep_running = false;
        } else {
            fprintf(stderr, "\n--- Session complete, waiting for next KV artifact ---\n\n");
            llama_kv_cache_clear(ctx);
        }
    }

    llama_free(ctx);
    llama_free_model(model);

    return 0;
}
