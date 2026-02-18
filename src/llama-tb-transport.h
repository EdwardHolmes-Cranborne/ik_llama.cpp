#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_tb_transfer_options {
    std::string session_dir;
    std::string session_id;
    std::string endpoint;
    std::string transport_mode;
    std::string kv_host;
    int32_t     kv_port = 0;
    std::string kv_bind_addrs;
    std::string kv_peer_addrs;
    std::string kv_balance;
    bool        transport_fallback = false;
    std::string execution_mode;
    size_t      chunk_bytes = 4 * 1024 * 1024;
    size_t      max_inflight_bytes = 256 * 1024 * 1024;
    int32_t     stream_count = 1;
    int32_t     socket_send_buf = 0;
    int32_t     socket_recv_buf = 0;
    bool        progressive = false;
    int32_t     remote_nodes = 1;
    int32_t     expected_gpu_layers = 0;
    int32_t     expected_remote_layers = 0;
    std::string layer_map;
    std::string remote_ranges;
    std::string remote_failover_policy;
    std::string handoff_session_id;
    std::string topology_epoch;
    uint32_t    artifact_crc32 = 0;
    std::string remote_node_descriptors_json;
    std::string prefill_handoff_v2_json;
    int32_t     dispatch_hop = 0;
};

struct llama_tb_transfer_result {
    std::string session_path;
    std::string transport_mode = "disabled";
    std::string transport_backend = "filesystem";
    uint64_t    bytes_sent = 0;
    uint32_t    chunks_sent = 0;
    bool        progressive = false;
    int32_t     stream_count = 1;
    int32_t     interface_count = 0;
    uint32_t    retransmit_chunks = 0;
    uint64_t    window_stalls_ms = 0;
    double      transfer_ms = 0.0;
    double      throughput_gbps = 0.0;
    uint64_t    first_chunk_send_unix_us = 0;
    uint64_t    last_chunk_send_unix_us = 0;
};

bool llama_tb_transport_enabled();

bool llama_tb_transport_send_artifact(const std::string &          artifact_path,
                                      const llama_tb_transfer_options & options,
                                      llama_tb_transfer_result *    result,
                                      std::string *                 error);

// Phase 6 helper: copies through an explicit host staging hop to emulate remote slab relay.
void llama_tb_transport_weight_relay_copy(const uint8_t *src, uint8_t *dst, size_t size);

// Phase 6 helper: obtain a weight chunk from the configured remote source mode.
// Returns true and fills dst on success.
bool llama_tb_transport_fetch_weight_chunk(const uint8_t * src, size_t size, std::vector<uint8_t> & dst, std::string * error);
