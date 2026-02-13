#pragma once

#include "common.h"
#include "server-queue.h"

#include <memory>
#include <string>

struct kv_receiver_config {
    bool enabled = false;
    std::string transport_mode = "disabled"; // auto|rdma|tcp|mixed|disabled
    bool transport_fallback = false;

    std::string host;
    int32_t port = -1;

    int32_t slot_id = 0;
    std::string output_dir;

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

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
