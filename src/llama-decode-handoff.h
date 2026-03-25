#pragma once

#include "llama.h"
#include "llama-kv-artifact.h"
#include "llama-tb-transport.h"
#include "llama-token-stream.h"
#include "llama-kv-sync.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Runtime configuration for hybrid prefill→decode handoff.
struct llama_decode_handoff_runtime {
    // KV transport
    std::string kv_host;
    int32_t     kv_port              = 9100;
    size_t      kv_chunk_bytes       = 4 * 1024 * 1024;

    // Token stream
    std::string token_stream_host    = "0.0.0.0";
    int32_t     token_stream_port    = 9101;

    // KV sync
    int32_t     kv_sync_batch_size   = 512;

    // Timeouts
    int32_t     token_read_timeout_ms = 60000;

    // Session
    std::string session_id;
};

// Diagnostics from a handoff publish operation.
struct llama_decode_publish_diag {
    uint64_t bytes_sent       = 0;
    uint32_t chunks_sent      = 0;
    double   transfer_ms      = 0.0;
    double   throughput_gbps  = 0.0;
};

// Diagnostics from a token relay operation.
struct llama_decode_relay_result {
    int32_t tokens_received = 0;
    int32_t tokens_synced   = 0;
    double  decode_tok_s    = 0.0;
};

// Executor interface: orchestrates the handoff lifecycle.
struct llama_decode_executor {
    llama_decode_executor(const llama_decode_handoff_runtime & runtime);
    ~llama_decode_executor();

    // Serialize KV cache from context and send to remote receiver.
    bool publish_kv_artifact(llama_context * ctx,
                             int token_count,
                             llama_decode_publish_diag * diag,
                             std::string * error);

    // Connect to remote token stream, relay tokens back, sync KV locally.
    bool relay_tokens(llama_context * ctx,
                      llama_decode_relay_result * result,
                      std::string * error);

private:
    llama_decode_handoff_runtime runtime_;
    llama_kv_sync                kv_sync_;
};

// Build a KV artifact payload from context's KV cache.
// Writes the raw KV data (no file header) suitable for transport.
bool llama_kv_cache_export_payload(llama_context * ctx,
                                   int token_count,
                                   std::vector<uint8_t> & payload_out,
                                   llama_kv_artifact_metadata & meta_out);
