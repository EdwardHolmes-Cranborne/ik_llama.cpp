#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct llama_token_msg {
    int32_t     token_id        = 0;
    std::string text;
    int32_t     pos             = 0;
    int64_t     timestamp_us    = 0;
    bool        done            = false;
    int32_t     tokens_generated = 0;
    double      decode_tok_s    = 0.0;
    int32_t     dropped_count   = 0;
};

struct llama_token_handshake {
    int32_t     version           = 1;
    std::string status;
    uint64_t    model_fingerprint = 0;
    uint32_t    token_count       = 0;
};

std::string llama_token_msg_to_json(const llama_token_msg & msg);
bool        llama_token_msg_from_json(const std::string & json, llama_token_msg & msg);

std::string llama_token_handshake_to_json(const llama_token_handshake & hs);
bool        llama_token_handshake_from_json(const std::string & json, llama_token_handshake & hs);

struct llama_token_ring {
    static constexpr size_t CAPACITY = 16384;

    llama_token_ring();
    bool push(const llama_token_msg & msg);
    bool pop(llama_token_msg & msg);
    size_t  size() const;
    int32_t total_dropped() const;

private:
    std::vector<llama_token_msg> buf_;
    size_t  head_    = 0;
    size_t  tail_    = 0;
    size_t  count_   = 0;
    int32_t dropped_ = 0;
    mutable std::mutex mtx_;
};

struct llama_token_stream_server {
    bool start(const std::string & host, int port);
    bool wait_for_client(int timeout_ms = 30000);
    bool send_handshake(const llama_token_handshake & hs);
    bool enqueue(const llama_token_msg & msg);
    bool send_done(int32_t tokens_generated, double decode_tok_s);
    void stop();
    int  bound_port() const;

private:
#ifdef _WIN32
    uintptr_t listen_fd_ = ~(uintptr_t)0;
    uintptr_t client_fd_ = ~(uintptr_t)0;
#else
    int listen_fd_ = -1;
    int client_fd_ = -1;
#endif
    int port_ = 0;
    std::atomic<bool> running_{false};
    std::atomic<bool> done_queued_{false};
    llama_token_ring  ring_;
    std::thread       sender_;

    void sender_loop();
    bool send_line(const std::string & line);
};

struct llama_token_stream_client {
    bool connect(const std::string & host, int port,
                 int retries = 5, int retry_ms = 500);
    bool read_handshake(llama_token_handshake & hs, int timeout_ms = 10000);
    bool read_msg(llama_token_msg & msg, int timeout_ms = 60000);
    void disconnect();

private:
#ifdef _WIN32
    uintptr_t fd_ = ~(uintptr_t)0;
#else
    int fd_ = -1;
#endif
    std::string buf_;

    bool read_line(std::string & line, int timeout_ms);
};
