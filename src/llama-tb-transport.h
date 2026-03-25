#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct llama_tb_transfer_options {
    std::string session_id;
    std::string kv_host;
    int32_t     kv_port = 9100;
    size_t      chunk_bytes = 4 * 1024 * 1024;
    size_t      max_inflight_bytes = 256 * 1024 * 1024;
    int32_t     socket_send_buf = 0;
    int32_t     socket_recv_buf = 0;
    bool        progressive = false;
};

struct llama_tb_transfer_result {
    uint64_t bytes_sent       = 0;
    uint32_t chunks_sent      = 0;
    double   transfer_ms      = 0.0;
    double   throughput_gbps  = 0.0;
};

// Send a KV artifact (raw bytes) to a remote KV receiver over TCP using TBP1 protocol.
bool llama_tb_transport_send_artifact(
    const uint8_t *                     data,
    size_t                              size,
    const llama_tb_transfer_options &   options,
    llama_tb_transfer_result *          result,
    std::string *                       error);

// Send a KV artifact file to a remote KV receiver over TCP using TBP1 protocol.
bool llama_tb_transport_send_artifact_file(
    const std::string &                 path,
    const llama_tb_transfer_options &   options,
    llama_tb_transfer_result *          result,
    std::string *                       error);
