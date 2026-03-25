#include "llama-token-stream.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Platform socket includes
// ---------------------------------------------------------------------------

#ifndef _WIN32
#  include <arpa/inet.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define SOCKET_CLOSE(fd) ::close(fd)
#  define SOCKET_INVALID   (-1)
typedef int socket_fd_t;
#else
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
#  define SOCKET_CLOSE(fd) ::closesocket(fd)
#  define SOCKET_INVALID   ((socket_fd_t)INVALID_SOCKET)
typedef SOCKET socket_fd_t;

static bool winsock_ensure_init() {
    static bool done = false;
    if (!done) {
        WSADATA wsa;
        done = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    }
    return done;
}
#endif

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static std::string json_unescape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"':  out += '"';  i++; break;
                case '\\': out += '\\'; i++; break;
                case 'n':  out += '\n'; i++; break;
                case 'r':  out += '\r'; i++; break;
                case 't':  out += '\t'; i++; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

static std::string json_get_str(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\":\"";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    std::string val;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '\\' && i + 1 < json.size()) {
            val += json[i]; val += json[i + 1]; i++;
        } else if (json[i] == '"') {
            break;
        } else {
            val += json[i];
        }
    }
    return json_unescape(val);
}

static bool json_get_int(const std::string & json, const std::string & key, int64_t & out) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    char * end = nullptr;
    long long val = strtoll(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;
    out = (int64_t)val;
    return true;
}

static bool json_get_double(const std::string & json, const std::string & key, double & out) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    char * end = nullptr;
    double val = strtod(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return false;
    out = val;
    return true;
}

static bool json_get_bool(const std::string & json, const std::string & key, bool & out) {
    std::string needle = "\"" + key + "\":";
    auto pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (json.compare(pos, 4, "true") == 0)  { out = true;  return true; }
    if (json.compare(pos, 5, "false") == 0) { out = false; return true; }
    return false;
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

std::string llama_token_msg_to_json(const llama_token_msg & msg) {
    std::ostringstream ss;
    if (msg.done) {
        ss << "{\"done\":true"
           << ",\"tokens_generated\":" << msg.tokens_generated
           << ",\"decode_tok_s\":" << msg.decode_tok_s
           << "}";
    } else {
        ss << "{\"token_id\":" << msg.token_id
           << ",\"text\":\"" << json_escape(msg.text) << "\""
           << ",\"pos\":" << msg.pos
           << ",\"timestamp_us\":" << msg.timestamp_us
           << ",\"dropped_count\":" << msg.dropped_count
           << "}";
    }
    return ss.str();
}

bool llama_token_msg_from_json(const std::string & json, llama_token_msg & msg) {
    msg = {};
    bool is_done = false;
    if (json_get_bool(json, "done", is_done) && is_done) {
        msg.done = true;
        int64_t tg = 0;
        if (json_get_int(json, "tokens_generated", tg)) msg.tokens_generated = (int32_t)tg;
        json_get_double(json, "decode_tok_s", msg.decode_tok_s);
        return true;
    }
    int64_t v = 0;
    if (json_get_int(json, "token_id", v)) msg.token_id = (int32_t)v;
    msg.text = json_get_str(json, "text");
    if (json_get_int(json, "pos", v)) msg.pos = (int32_t)v;
    if (json_get_int(json, "timestamp_us", v)) msg.timestamp_us = v;
    if (json_get_int(json, "dropped_count", v)) msg.dropped_count = (int32_t)v;
    return true;
}

std::string llama_token_handshake_to_json(const llama_token_handshake & hs) {
    std::ostringstream ss;
    ss << "{\"version\":" << hs.version
       << ",\"status\":\"" << json_escape(hs.status) << "\""
       << ",\"model_fingerprint\":" << hs.model_fingerprint
       << ",\"token_count\":" << hs.token_count
       << "}";
    return ss.str();
}

bool llama_token_handshake_from_json(const std::string & json, llama_token_handshake & hs) {
    hs = {};
    int64_t v = 0;
    if (json_get_int(json, "version", v)) hs.version = (int32_t)v;
    hs.status = json_get_str(json, "status");
    if (json_get_int(json, "model_fingerprint", v)) hs.model_fingerprint = (uint64_t)v;
    if (json_get_int(json, "token_count", v)) hs.token_count = (uint32_t)v;
    return hs.version > 0;
}

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------

llama_token_ring::llama_token_ring() : buf_(CAPACITY) {}

bool llama_token_ring::push(const llama_token_msg & msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    bool overwrite = false;
    if (count_ == CAPACITY) {
        tail_ = (tail_ + 1) % CAPACITY;
        count_--;
        dropped_++;
        overwrite = true;
    }
    buf_[head_] = msg;
    head_ = (head_ + 1) % CAPACITY;
    count_++;
    return !overwrite;
}

bool llama_token_ring::pop(llama_token_msg & msg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (count_ == 0) return false;
    msg = std::move(buf_[tail_]);
    tail_ = (tail_ + 1) % CAPACITY;
    count_--;
    return true;
}

size_t llama_token_ring::size() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return count_;
}

int32_t llama_token_ring::total_dropped() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return dropped_;
}

// ---------------------------------------------------------------------------
// Socket helpers
// ---------------------------------------------------------------------------

static socket_fd_t create_listen_socket(const std::string & host, int port) {
#ifdef _WIN32
    if (!winsock_ensure_init()) return SOCKET_INVALID;
#endif
    socket_fd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCKET_INVALID) return SOCKET_INVALID;

#ifndef _WIN32
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#else
    const char opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        SOCKET_CLOSE(fd); return SOCKET_INVALID;
    }
    if (listen(fd, 1) != 0) {
        SOCKET_CLOSE(fd); return SOCKET_INVALID;
    }
    return fd;
}

static int get_bound_port(socket_fd_t fd) {
    struct sockaddr_in addr = {};
    socklen_t len = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &len) != 0) return -1;
    return ntohs(addr.sin_port);
}

static socket_fd_t accept_with_timeout(socket_fd_t listen_fd, int timeout_ms) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(listen_fd, &rfds);
    struct timeval tv = {};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifndef _WIN32
    int ret = select((int)listen_fd + 1, &rfds, nullptr, nullptr, &tv);
#else
    int ret = select(0, &rfds, nullptr, nullptr, &tv);
#endif
    if (ret <= 0) return SOCKET_INVALID;
    return accept(listen_fd, nullptr, nullptr);
}

static socket_fd_t connect_to(const std::string & host, int port) {
#ifdef _WIN32
    if (!winsock_ensure_init()) return SOCKET_INVALID;
#endif
    socket_fd_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCKET_INVALID) return SOCKET_INVALID;

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (::connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        SOCKET_CLOSE(fd); return SOCKET_INVALID;
    }

#ifndef _WIN32
    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#else
    const char opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif
    return fd;
}

static bool send_all(socket_fd_t fd, const void * data, size_t len) {
    const char * p = (const char *)data;
    size_t sent = 0;
    while (sent < len) {
#ifndef _WIN32
        ssize_t n = ::send(fd, p + sent, len - sent, MSG_NOSIGNAL);
#else
        int n = ::send(fd, p + sent, (int)(len - sent), 0);
#endif
        if (n <= 0) return false;
        sent += (size_t)n;
    }
    return true;
}

static bool recv_with_timeout(socket_fd_t fd, void * buf, size_t len, int timeout_ms, size_t & bytes_read) {
    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);
    struct timeval tv = {};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
#ifndef _WIN32
    int ret = select((int)fd + 1, &rfds, nullptr, nullptr, &tv);
#else
    int ret = select(0, &rfds, nullptr, nullptr, &tv);
#endif
    if (ret <= 0) return false;
#ifndef _WIN32
    ssize_t n = ::recv(fd, buf, len, 0);
#else
    int n = ::recv(fd, (char *)buf, (int)len, 0);
#endif
    if (n <= 0) return false;
    bytes_read = (size_t)n;
    return true;
}

// ---------------------------------------------------------------------------
// Token stream server
// ---------------------------------------------------------------------------

bool llama_token_stream_server::start(const std::string & host, int port) {
    listen_fd_ = create_listen_socket(host, port);
    if (listen_fd_ == SOCKET_INVALID) return false;
    port_ = get_bound_port(listen_fd_);
    running_ = true;
    return true;
}

bool llama_token_stream_server::wait_for_client(int timeout_ms) {
    client_fd_ = accept_with_timeout(listen_fd_, timeout_ms);
    if (client_fd_ == SOCKET_INVALID) return false;
#ifndef _WIN32
    int opt = 1;
    setsockopt(client_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#else
    const char opt = 1;
    setsockopt(client_fd_, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
#endif
    // Start sender thread immediately so tokens stream in real-time
    if (!sender_.joinable()) {
        sender_ = std::thread(&llama_token_stream_server::sender_loop, this);
    }
    return true;
}

bool llama_token_stream_server::send_handshake(const llama_token_handshake & hs) {
    return send_line(llama_token_handshake_to_json(hs));
}

bool llama_token_stream_server::enqueue(const llama_token_msg & msg) {
    ring_.push(msg);
    return true;
}

bool llama_token_stream_server::send_done(int32_t tokens_generated, double decode_tok_s) {
    llama_token_msg done = {};
    done.done = true;
    done.tokens_generated = tokens_generated;
    done.decode_tok_s = decode_tok_s;
    ring_.push(done);
    done_queued_ = true;

    // Wait for sender thread to drain the ring (including the done message)
    if (sender_.joinable()) {
        sender_.join();
    }
    return true;
}

void llama_token_stream_server::stop() {
    running_ = false;
    if (sender_.joinable()) sender_.join();
    if (client_fd_ != SOCKET_INVALID) { SOCKET_CLOSE(client_fd_); client_fd_ = SOCKET_INVALID; }
    if (listen_fd_ != SOCKET_INVALID) { SOCKET_CLOSE(listen_fd_); listen_fd_ = SOCKET_INVALID; }
}

int llama_token_stream_server::bound_port() const { return port_; }

void llama_token_stream_server::sender_loop() {
    while (running_) {
        llama_token_msg msg = {};
        if (ring_.pop(msg)) {
            std::string line = llama_token_msg_to_json(msg);
            if (!send_line(line)) break;
            if (msg.done) break;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

bool llama_token_stream_server::send_line(const std::string & line) {
    if (client_fd_ == SOCKET_INVALID) return false;
    std::string data = line + "\n";
    return send_all(client_fd_, data.data(), data.size());
}

// ---------------------------------------------------------------------------
// Token stream client
// ---------------------------------------------------------------------------

bool llama_token_stream_client::connect(const std::string & host, int port,
                                         int retries, int retry_ms) {
    for (int attempt = 0; attempt < retries; attempt++) {
        fd_ = connect_to(host, port);
        if (fd_ != SOCKET_INVALID) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_ms));
    }
    return false;
}

bool llama_token_stream_client::read_handshake(llama_token_handshake & hs, int timeout_ms) {
    std::string line;
    if (!read_line(line, timeout_ms)) return false;
    return llama_token_handshake_from_json(line, hs);
}

bool llama_token_stream_client::read_msg(llama_token_msg & msg, int timeout_ms) {
    std::string line;
    if (!read_line(line, timeout_ms)) return false;
    return llama_token_msg_from_json(line, msg);
}

void llama_token_stream_client::disconnect() {
    if (fd_ != SOCKET_INVALID) { SOCKET_CLOSE(fd_); fd_ = SOCKET_INVALID; }
    buf_.clear();
}

bool llama_token_stream_client::read_line(std::string & line, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (true) {
        auto nl = buf_.find('\n');
        if (nl != std::string::npos) {
            line = buf_.substr(0, nl);
            buf_.erase(0, nl + 1);
            return true;
        }
        auto now = std::chrono::steady_clock::now();
        int remaining = (int)std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (remaining <= 0) return false;

        char tmp[4096];
        size_t n = 0;
        if (!recv_with_timeout(fd_, tmp, sizeof(tmp), remaining, n)) return false;
        buf_.append(tmp, n);
    }
}
