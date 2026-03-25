#include "llama-decode-handoff.h"
#include "llama-kv-artifact.h"
#include "llama-kv-import.h"
#include "llama-tb-transport.h"
#include "llama-token-stream.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

// ============================================================================
// KV cache export: read KV tensors from context into a flat payload
// ============================================================================

bool llama_kv_cache_export_payload(llama_context * ctx,
                                   int token_count,
                                   std::vector<uint8_t> & payload_out,
                                   llama_kv_artifact_metadata & meta_out) {
    const llama_model * model = llama_get_model(ctx);
    const int n_layer = llama_n_layer(model);

    // Build metadata
    meta_out = {};
    meta_out.format_major = 1;
    meta_out.format_minor = 1;
    meta_out.n_layers     = (uint32_t)n_layer;
    meta_out.n_ctx        = llama_n_ctx(ctx);
    meta_out.token_count  = (uint32_t)token_count;

    // Read KV cache data from context
    // This reads from the backend tensors — works for both GPU and CPU KV
    size_t state_size = llama_state_get_size(ctx);
    if (state_size == 0) {
        payload_out.clear();
        return true;
    }

    // Use llama's built-in state serialization for portability
    // This captures the full KV cache state including metadata
    payload_out.resize(state_size);
    size_t written = llama_state_get_data(ctx, payload_out.data(), state_size);
    if (written == 0) {
        payload_out.clear();
        return false;
    }
    payload_out.resize(written);

    return true;
}

// ============================================================================
// Executor
// ============================================================================

llama_decode_executor::llama_decode_executor(const llama_decode_handoff_runtime & runtime)
    : runtime_(runtime) {
    kv_sync_.batch_size = runtime.kv_sync_batch_size;
}

llama_decode_executor::~llama_decode_executor() = default;

bool llama_decode_executor::publish_kv_artifact(
    llama_context * ctx,
    int token_count,
    llama_decode_publish_diag * diag,
    std::string * error) {

    if (diag) *diag = {};

    // Export KV cache to payload
    std::vector<uint8_t> payload;
    llama_kv_artifact_metadata meta;
    if (!llama_kv_cache_export_payload(ctx, token_count, payload, meta)) {
        if (error) *error = "failed to export KV cache";
        return false;
    }

    // Build artifact with header, send over TCP using TBP1 framing.
    // The receiver reassembles TBP1 frames and parses with llama_kv_artifact_read_mem.
    {
        llama_kv_artifact_summary summary;
        std::vector<uint8_t> artifact_buf;

        // Write artifact to temp file (unique name for thread-safety), read back
        std::string tmp_path = "kv_handoff_" + runtime_.session_id + "_" +
                               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin";
        if (!llama_kv_artifact_write(tmp_path, payload, meta, &summary, error)) {
            return false;
        }
        // Read back the written file into memory
        std::ifstream ifs(tmp_path, std::ios::binary | std::ios::ate);
        if (!ifs.is_open()) {
            if (error) *error = "failed to read back temp artifact";
            std::remove(tmp_path.c_str());
            return false;
        }
        size_t file_size = (size_t)ifs.tellg();
        ifs.seekg(0);
        artifact_buf.resize(file_size);
        ifs.read(reinterpret_cast<char *>(artifact_buf.data()), (std::streamsize)file_size);
        ifs.close();
        std::remove(tmp_path.c_str());

        // Send artifact bytes over TBP1
        llama_tb_transfer_options topts;
        topts.kv_host     = runtime_.kv_host;
        topts.kv_port     = runtime_.kv_port;
        topts.chunk_bytes = runtime_.kv_chunk_bytes;
        topts.session_id  = runtime_.session_id;

        llama_tb_transfer_result tresult;
        if (!llama_tb_transport_send_artifact(artifact_buf.data(), artifact_buf.size(),
                                              topts, &tresult, error)) {
            return false;
        }

        if (diag) {
            diag->bytes_sent      = tresult.bytes_sent;
            diag->chunks_sent     = tresult.chunks_sent;
            diag->transfer_ms     = tresult.transfer_ms;
            diag->throughput_gbps = tresult.throughput_gbps;
        }

        fprintf(stderr, "KV artifact published: %zu bytes in %.1f ms (%.2f Gbps)\n",
                (size_t)tresult.bytes_sent, tresult.transfer_ms, tresult.throughput_gbps);
    }

    return true;
}

bool llama_decode_executor::relay_tokens(
    llama_context * ctx,
    llama_decode_relay_result * result,
    std::string * error) {

    if (result) *result = {};

    // Connect to token stream on the Mac
    llama_token_stream_client client;
    if (!client.connect(runtime_.kv_host, runtime_.token_stream_port, 10, 1000)) {
        if (error) *error = "failed to connect to token stream at " +
                            runtime_.kv_host + ":" + std::to_string(runtime_.token_stream_port);
        return false;
    }

    // Read handshake
    llama_token_handshake hs;
    if (!client.read_handshake(hs, 30000)) {
        if (error) *error = "token stream handshake timeout";
        client.disconnect();
        return false;
    }

    if (hs.status != "ready") {
        if (error) *error = "token stream not ready: " + hs.status;
        client.disconnect();
        return false;
    }

    fprintf(stderr, "Token stream connected: version=%d, tokens_in_kv=%u\n",
            hs.version, hs.token_count);

    // Read tokens and sync KV
    int tokens_received = 0;
    int tokens_synced = 0;
    double decode_tok_s = 0.0;

    while (true) {
        llama_token_msg msg;
        if (!client.read_msg(msg, runtime_.token_read_timeout_ms)) {
            if (error) *error = "token stream read timeout";
            break;
        }

        if (msg.done) {
            decode_tok_s = msg.decode_tok_s;
            fprintf(stderr, "\nRemote decode complete: %d tokens at %.1f tok/s\n",
                    msg.tokens_generated, msg.decode_tok_s);

            // Flush any remaining tokens
            if (!kv_sync_.is_empty()) {
                int remaining = kv_sync_.pending_count();
                if (kv_sync_.flush(ctx)) {
                    tokens_synced += remaining;
                }
            }
            break;
        }

        tokens_received++;
        kv_sync_.push(msg.token_id);

        // Print token as received
        fprintf(stderr, "%s", msg.text.c_str());

        // Batch sync when threshold reached
        if (kv_sync_.pending_count() >= kv_sync_.batch_size) {
            int before = kv_sync_.pending_count();
            if (kv_sync_.flush(ctx)) {
                tokens_synced += before;
            }
        }
    }

    client.disconnect();

    if (result) {
        result->tokens_received = tokens_received;
        result->tokens_synced   = tokens_synced;
        result->decode_tok_s    = decode_tok_s;
    }

    return true;
}
