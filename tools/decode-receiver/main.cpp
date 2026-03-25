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
#include "common.h"
#include "sampling.h"

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
    float       temp          = 0.8f;
    int         top_k         = 40;
    float       top_p         = 0.95f;
    float       min_p         = 0.05f;
    int         seed          = -1;
    bool        mlock         = false;
    bool        one_shot      = false;

    // Debug features
    std::string debug_prompt;       // if set: do local prefill, compare KV with remote
    std::string debug_dump_kv;      // if set: dump received KV state to this file
    bool        debug_kv_compare = false; // compare local vs remote KV checksums
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
    fprintf(stderr, "\nDebug:\n");
    fprintf(stderr, "  --debug-prompt PROMPT       Local prefill + KV compare with remote\n");
    fprintf(stderr, "  --debug-dump-kv PATH        Dump received KV state to file\n");
    fprintf(stderr, "  --debug-kv-compare          Print per-layer KV checksums\n");
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
        } else if (arg == "--top-k") {
            const char * v = next(); if (!v) return false;
            params.top_k = std::atoi(v);
        } else if (arg == "--top-p") {
            const char * v = next(); if (!v) return false;
            params.top_p = (float)std::atof(v);
        } else if (arg == "--min-p") {
            const char * v = next(); if (!v) return false;
            params.min_p = (float)std::atof(v);
        } else if (arg == "--seed" || arg == "-s") {
            const char * v = next(); if (!v) return false;
            params.seed = std::atoi(v);
        } else if (arg == "--mlock") {
            params.mlock = true;
        } else if (arg == "--one-shot") {
            params.one_shot = true;
        } else if (arg == "--debug-prompt") {
            const char * v = next(); if (!v) return false;
            params.debug_prompt = v;
            params.debug_kv_compare = true;
        } else if (arg == "--debug-dump-kv") {
            const char * v = next(); if (!v) return false;
            params.debug_dump_kv = v;
        } else if (arg == "--debug-kv-compare") {
            params.debug_kv_compare = true;
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

        // Debug: dump received KV to file
        if (!params.debug_dump_kv.empty() && !payload.empty()) {
            std::ofstream dump(params.debug_dump_kv, std::ios::binary);
            if (dump.is_open()) {
                dump.write(reinterpret_cast<const char *>(payload.data()), (std::streamsize)payload.size());
                fprintf(stderr, "[debug] Dumped received KV state to %s (%zu bytes)\n",
                        params.debug_dump_kv.c_str(), payload.size());
            }
        }

        // Debug: local prefill for KV comparison
        std::vector<uint8_t> local_kv_state;
        if (!params.debug_prompt.empty()) {
            fprintf(stderr, "\n[debug] === Local prefill for KV comparison ===\n");
            fprintf(stderr, "[debug] Prompt: \"%s\"\n", params.debug_prompt.c_str());

            // Tokenize the debug prompt
            std::vector<llama_token> tokens(params.debug_prompt.size() + 32);
            int n_tokens = llama_tokenize(
                model, params.debug_prompt.c_str(), (int)params.debug_prompt.size(),
                tokens.data(), (int)tokens.size(), true, true);
            if (n_tokens < 0) {
                tokens.resize(-n_tokens);
                n_tokens = llama_tokenize(
                    model, params.debug_prompt.c_str(), (int)params.debug_prompt.size(),
                    tokens.data(), (int)tokens.size(), true, true);
            }
            tokens.resize(n_tokens);
            fprintf(stderr, "[debug] Tokenized: %d tokens\n", n_tokens);

            // Local prefill
            llama_kv_cache_clear(ctx);
            llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens, 0, 0);
            batch.logits = nullptr;  // don't need logits for comparison
            int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                fprintf(stderr, "[debug] Local prefill failed: rc=%d\n", rc);
            } else {
                fprintf(stderr, "[debug] Local prefill succeeded: %d tokens\n", n_tokens);

                // Serialize local KV state
                size_t state_size = llama_state_get_size(ctx);
                local_kv_state.resize(state_size);
                size_t written = llama_state_get_data(ctx, local_kv_state.data(), state_size);
                local_kv_state.resize(written);
                fprintf(stderr, "[debug] Local KV state: %zu bytes\n", written);

                // Compare sizes
                fprintf(stderr, "[debug] Remote KV state: %zu bytes\n", payload.size());
                fprintf(stderr, "[debug] Size match: %s\n",
                        local_kv_state.size() == payload.size() ? "YES" : "NO");

                // Compare content (find first difference)
                if (!local_kv_state.empty() && !payload.empty()) {
                    size_t min_size = std::min(local_kv_state.size(), payload.size());
                    size_t first_diff = min_size;
                    int diff_count = 0;
                    for (size_t i = 0; i < min_size; i++) {
                        if (local_kv_state[i] != payload[i]) {
                            if (first_diff == min_size) first_diff = i;
                            diff_count++;
                        }
                    }
                    if (diff_count == 0 && local_kv_state.size() == payload.size()) {
                        fprintf(stderr, "[debug] KV states are IDENTICAL\n");
                    } else {
                        fprintf(stderr, "[debug] KV states DIFFER: %d bytes different, first diff at offset %zu\n",
                                diff_count, first_diff);
                        // Print hex dump around first difference
                        if (first_diff < min_size) {
                            size_t dump_start = first_diff > 16 ? first_diff - 16 : 0;
                            size_t dump_end = std::min(first_diff + 48, min_size);
                            fprintf(stderr, "[debug] Local  @%zu:", dump_start);
                            for (size_t i = dump_start; i < dump_end; i++) {
                                if (i == first_diff) fprintf(stderr, " [");
                                fprintf(stderr, "%02x", local_kv_state[i]);
                                if (i == first_diff) fprintf(stderr, "]");
                            }
                            fprintf(stderr, "\n[debug] Remote @%zu:", dump_start);
                            for (size_t i = dump_start; i < dump_end; i++) {
                                if (i == first_diff) fprintf(stderr, " [");
                                fprintf(stderr, "%02x", payload[i]);
                                if (i == first_diff) fprintf(stderr, "]");
                            }
                            fprintf(stderr, "\n");
                        }
                    }

                    // Per-64KB block checksums for coarse comparison
                    if (params.debug_kv_compare) {
                        const size_t block_size = 64 * 1024;
                        fprintf(stderr, "[debug] Per-64KB block comparison (first 20 blocks):\n");
                        int blocks_shown = 0;
                        for (size_t off = 0; off < min_size && blocks_shown < 20; off += block_size, blocks_shown++) {
                            size_t len = std::min(block_size, min_size - off);
                            uint32_t local_sum = 0, remote_sum = 0;
                            for (size_t i = 0; i < len; i++) {
                                local_sum += local_kv_state[off + i];
                                remote_sum += payload[off + i];
                            }
                            const char * status = (local_sum == remote_sum) ? "MATCH" : "DIFF";
                            fprintf(stderr, "[debug]   block %3d (@%8zu): local_sum=%10u remote_sum=%10u %s\n",
                                    blocks_shown, off, local_sum, remote_sum, status);
                        }
                    }
                }
            }

            // Dump local KV for external comparison
            if (!params.debug_dump_kv.empty() && !local_kv_state.empty()) {
                std::string local_path = params.debug_dump_kv + ".local";
                std::ofstream dump(local_path, std::ios::binary);
                if (dump.is_open()) {
                    dump.write(reinterpret_cast<const char *>(local_kv_state.data()),
                               (std::streamsize)local_kv_state.size());
                    fprintf(stderr, "[debug] Dumped local KV state to %s (%zu bytes)\n",
                            local_path.c_str(), local_kv_state.size());
                }
            }

            fprintf(stderr, "[debug] === End KV comparison ===\n\n");
        }

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

        // Initialize sampler with proper temperature, top-k, top-p, min-p
        common_params_sampling sparams;
        sparams.temp     = params.temp;
        sparams.top_k    = params.top_k;
        sparams.top_p    = params.top_p;
        sparams.min_p    = params.min_p;
        sparams.seed     = (uint32_t)params.seed;

        struct common_sampler * smpl = common_sampler_init(model, sparams);
        if (!smpl) {
            fprintf(stderr, "Failed to initialize sampler\n");
            break;
        }

        fprintf(stderr, "Sampling: temp=%.2f, top_k=%d, top_p=%.2f, min_p=%.2f, seed=%d\n",
                params.temp, params.top_k, params.top_p, params.min_p, params.seed);

        // Decode loop with proper sampling (matches llama-cli behavior)
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

            // Sample with temperature, top-k, top-p, min-p
            llama_token id = common_sampler_sample_legacy(smpl, ctx, nullptr, -1);
            common_sampler_accept(smpl, ctx, id, true);

            last_token = id;
            tokens_generated++;

            // Get token text
            char buf[256];
            int n = llama_token_to_piece(model, id, buf, sizeof(buf), 0, true);
            std::string text(buf, n > 0 ? n : 0);

            // Stream token
            if (ts_ok) {
                llama_token_msg msg;
                msg.token_id = id;
                msg.text = text;
                msg.pos = (int32_t)(meta.token_count + i);
                ts_server.enqueue(msg);
            }

            // Print to stderr
            fprintf(stderr, "%s", text.c_str());

            // Check EOS
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }

            llama_batch_free(batch);
        }

        common_sampler_free(smpl);

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
