#pragma once

#include "llama.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

enum llama_decode_executor_kind {
    LLAMA_DECODE_EXECUTOR_LOCAL_CPU = 0,
    LLAMA_DECODE_EXECUTOR_LOCAL_GPU = 1,
    LLAMA_DECODE_EXECUTOR_LOCAL_HYBRID = 2,
    LLAMA_DECODE_EXECUTOR_SPLIT_THUNDERBOLT = 3,
};

struct llama_decode_layer_map_entry {
    std::string node;
    int32_t     layer_start = 0;  // inclusive
    int32_t     layer_end   = 0;  // exclusive
};

struct llama_decode_handoff_runtime {
    int32_t n_layers = 0;

    bool offload_kqv     = false;
    bool has_gpu_backend = false;
    int32_t model_gpu_layers = 0;

    int32_t decode_mode        = LLAMA_PREFILL_DECODE_MODE_AUTO;
    int32_t gpu_layers_hint    = -1;  // -1 = auto
    int32_t remote_layers_hint = 0;   // 0 = no remote split
    int32_t remote_nodes_hint  = 1;
    std::string remote_ranges_hint;   // explicit per-node ranges: node:start-end,node:start-end
    std::string remote_failover_policy = "reroute"; // reroute|local|fail

    int32_t transport_mode = LLAMA_PREFILL_TRANSPORT_MODE_DISABLED;
    int32_t execution_mode = LLAMA_PREFILL_EXECUTION_MODE_COUPLED;
    int32_t tb_chunk_bytes = 4 * 1024 * 1024;
    std::string transport_session_dir;
    std::string kv_transport;
    std::string tb_direct_endpoint;
    std::string kv_host;
    int32_t kv_port = 0;
    int32_t kv_streams = 0;
    int32_t kv_stream_chunk_bytes = 0;
    int32_t kv_max_inflight_bytes = 0;
    int32_t kv_socket_send_buf = 0;
    int32_t kv_socket_recv_buf = 0;
    std::string kv_bind_addrs;
    std::string kv_peer_addrs;
    std::string kv_balance;
    bool kv_transport_fallback = false;

    bool transport_required = false;
};

struct llama_decode_handoff_plan {
    int32_t requested_mode = LLAMA_PREFILL_DECODE_MODE_AUTO;
    int32_t resolved_mode  = LLAMA_PREFILL_DECODE_MODE_AUTO;

    llama_decode_executor_kind executor_kind = LLAMA_DECODE_EXECUTOR_LOCAL_CPU;

    bool use_transport      = false;
    bool transport_required = false;
    bool fallback_applied   = false;
    int32_t remote_nodes    = 1;
    int32_t expected_gpu_layers = 0;
    int32_t expected_remote_layers = 0;
    int32_t transport_mode  = LLAMA_PREFILL_TRANSPORT_MODE_DISABLED;
    int32_t execution_mode  = LLAMA_PREFILL_EXECUTION_MODE_COUPLED;
    std::string remote_ranges;
    std::string remote_failover_policy = "reroute";
    int32_t tb_chunk_bytes  = 4 * 1024 * 1024;
    std::string transport_session_dir;
    std::string kv_transport;
    std::string tb_direct_endpoint;
    std::string kv_host;
    int32_t kv_port = 0;
    int32_t kv_streams = 0;
    int32_t kv_stream_chunk_bytes = 0;
    int32_t kv_max_inflight_bytes = 0;
    int32_t kv_socket_send_buf = 0;
    int32_t kv_socket_recv_buf = 0;
    std::string kv_bind_addrs;
    std::string kv_peer_addrs;
    std::string kv_balance;
    bool kv_transport_fallback = false;

    std::string fallback_reason;
    std::vector<llama_decode_layer_map_entry> layer_map;
};

struct llama_decode_publish_diag {
    bool        transport_used = false;
    std::string transport_backend = "disabled";
    bool        progressive = false;
    uint64_t    bytes_sent = 0;
    uint32_t    chunks_sent = 0;
    int32_t     stream_count = 1;
    int32_t     interface_count = 0;
    uint32_t    retransmit_chunks = 0;
    uint64_t    window_stalls_ms = 0;
    double      transfer_ms = 0.0;
    double      throughput_gbps = 0.0;
    uint64_t    first_chunk_send_unix_us = 0;
    uint64_t    last_chunk_send_unix_us = 0;
};

class llama_decode_executor_i {
  public:
    virtual ~llama_decode_executor_i() = default;

    virtual const char * name() const = 0;
    virtual bool         available() const = 0;
    virtual bool begin_session(const llama_decode_handoff_plan & plan, std::string & status) = 0;
    virtual bool publish_kv_artifact(const std::string & artifact_path, const llama_decode_handoff_plan & plan, std::string & status,
                                     llama_decode_publish_diag * diag = nullptr) = 0;
};

llama_decode_handoff_plan llama_decode_handoff_build_plan(const llama_decode_handoff_runtime & runtime);

std::unique_ptr<llama_decode_executor_i>
llama_decode_executor_create(const llama_decode_handoff_plan & plan, std::string * fallback_reason);

const char * llama_decode_executor_kind_name(llama_decode_executor_kind kind);
const char * llama_prefill_decode_mode_name(int32_t mode);

std::string llama_decode_handoff_plan_to_string(const llama_decode_handoff_plan & plan);
