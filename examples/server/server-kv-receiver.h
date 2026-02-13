#pragma once

#include "common.h"
#include "server-queue.h"

#include <memory>
#include <string>
#include <vector>

struct kv_receiver_config {
    bool enabled = false;
    std::string transport_mode = "disabled"; // auto|rdma|tcp|mixed|disabled
    bool transport_fallback = false;

    std::string host;
    int32_t port = -1;

    int32_t slot_id = 0;
    std::string output_dir;
    bool dry_run = false;

    bool ack_enabled = true;
    bool nack_on_crc_bad = true;

    int32_t max_connections = 32;
    int32_t idle_timeout_sec = 30;
    int32_t socket_send_buf = 0;
    int32_t socket_recv_buf = 0;
};

struct kv_receiver_runtime {
    bool enabled = false;
    bool running = false;

    std::string requested_transport_mode = "disabled";
    std::string resolved_transport_mode = "disabled";

    std::string bind_host;
    int32_t bind_port = -1;

    std::string output_dir;
};

struct kv_receiver_session_stats {
    uint64_t session_id = 0;
    uint64_t bytes_received = 0;
    uint64_t chunks_received = 0;
    uint64_t bad_crc_chunks = 0;
    uint64_t expected_chunks = 0;
    int32_t expected_streams = 0;
    int32_t seen_streams = 0;
    int32_t done_streams = 0;
    bool finalized = false;
    bool validation_ok = false;
    bool restore_enqueued = false;
    std::string artifact_path;
    std::string validation_error;
    std::string last_error;
    uint64_t first_frame_unix_us = 0;
    uint64_t last_frame_unix_us = 0;
};

struct kv_receiver_stats {
    bool running = false;
    std::string resolved_transport_mode;
    std::string bind_host;
    int32_t bind_port = -1;
    bool dry_run = false;

    uint64_t connections_accepted = 0;
    uint64_t connections_rejected = 0;
    uint64_t frames_total = 0;
    uint64_t frames_bad = 0;
    uint64_t ack_sent = 0;
    uint64_t nack_sent = 0;
    uint64_t artifacts_reassembled = 0;
    uint64_t artifacts_validated = 0;
    uint64_t restore_tasks_enqueued = 0;
    uint64_t restore_tasks_skipped_dry_run = 0;

    std::vector<kv_receiver_session_stats> sessions;
};

// Resolve and normalize runtime KV receiver config from gpt_params.
kv_receiver_config kv_receiver_config_from_params(const gpt_params & params);

class kv_receiver_service {
public:
    kv_receiver_service(server_queue & queue_tasks, server_response & queue_results);
    ~kv_receiver_service();

    kv_receiver_service(const kv_receiver_service &) = delete;
    kv_receiver_service & operator=(const kv_receiver_service &) = delete;

    bool start(const kv_receiver_config & config, std::string * error);
    void stop();

    kv_receiver_runtime runtime() const;
    kv_receiver_stats stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
