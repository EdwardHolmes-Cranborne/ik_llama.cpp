#include "server-kv-receiver.h"

#include "server-task.h"

#include "log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef _WIN32
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

#define KVR_INF(fmt, ...) LOG_INF("kvr  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define KVR_WRN(fmt, ...) LOG_WRN("kvr  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define KVR_ERR(fmt, ...) LOG_ERR("kvr  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define KVR_DBG(fmt, ...) LOG_DBG("kvr  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

namespace {

constexpr uint32_t TBP_MAGIC = 0x54425031u; // "TBP1"
constexpr uint16_t TBP_PROTO_MAJOR = 1;
constexpr uint16_t TBP_PROTO_MINOR = 0;
constexpr uint32_t TBP_HEADER_SIZE = 52;
constexpr uint32_t TBP_MAX_PAYLOAD_BYTES = 256u * 1024u * 1024u;

enum tbp_msg_type : uint16_t {
    TBP_MSG_HELLO            = 1,
    TBP_MSG_SESSION_START    = 3,
    TBP_MSG_KV_SEGMENT_BEGIN = 7,
    TBP_MSG_KV_CHUNK         = 8,
    TBP_MSG_KV_SEGMENT_END   = 9,
    TBP_MSG_KV_ACK           = 10,
    TBP_MSG_KV_DONE          = 12,
};

struct tbp_frame {
    uint16_t msg_type = 0;
    uint16_t flags = 0;
    uint64_t session_id = 0;
    uint64_t stream_id = 0;
    uint64_t seq_no = 0;
    uint32_t payload_crc = 0;
    std::vector<uint8_t> payload;
};

struct session_state {
    uint64_t session_id = 0;

    fs::path session_dir;
    fs::path chunks_dir;
    fs::path artifact_path;

    std::mutex mutex;
    int expected_streams = 0;
    std::set<uint64_t> seen_stream_ids;
    std::set<uint64_t> done_stream_ids;

    uint64_t bytes_received = 0;
    uint64_t chunks_received = 0;
    uint64_t bad_crc_chunks = 0;

    bool finalized = false;
};

std::string env_string(const char * name, const std::string & fallback = "") {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return fallback;
    }
    return std::string(v);
}

int32_t env_i32(const char * name, int32_t fallback) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return fallback;
    }
    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

std::string normalize_transport_mode(const std::string & mode_raw) {
    std::string mode = lower_copy(string_strip(mode_raw));

    if (mode.empty()) {
        return "disabled";
    }
    if (mode == "tb-direct") {
        return "rdma";
    }
    if (mode == "tb-ethernet" || mode == "ethernet") {
        return "tcp";
    }

    if (mode == "auto" || mode == "rdma" || mode == "tcp" || mode == "mixed" || mode == "disabled") {
        return mode;
    }

    return "";
}

std::vector<std::string> mode_order_for_receiver(const std::string & mode, bool fallback) {
    if (mode == "auto" || mode == "mixed") {
        return {"rdma", "tcp"};
    }
    if (mode == "rdma") {
        return fallback ? std::vector<std::string>{"rdma", "tcp"} : std::vector<std::string>{"rdma"};
    }
    if (mode == "tcp") {
        return fallback ? std::vector<std::string>{"tcp", "rdma"} : std::vector<std::string>{"tcp"};
    }
    if (mode == "disabled") {
        return {"disabled"};
    }
    return {};
}

std::string default_bind_host_for_mode(const std::string & mode) {
    if (mode == "rdma") {
        return env_string("LLAMA_PREFILL_KV_RDMA_HOST", env_string("LLAMA_PREFILL_KV_HOST", "0.0.0.0"));
    }
    if (mode == "tcp") {
        return env_string("LLAMA_PREFILL_KV_HOST", "0.0.0.0");
    }
    return env_string("LLAMA_PREFILL_KV_RDMA_HOST", env_string("LLAMA_PREFILL_KV_HOST", "0.0.0.0"));
}

int32_t default_bind_port_for_mode(const std::string & mode) {
    if (mode == "rdma") {
        return env_i32("LLAMA_PREFILL_KV_RDMA_PORT", env_i32("LLAMA_PREFILL_KV_PORT", 19001));
    }
    if (mode == "tcp") {
        return env_i32("LLAMA_PREFILL_KV_PORT", 19001);
    }
    return env_i32("LLAMA_PREFILL_KV_RDMA_PORT", env_i32("LLAMA_PREFILL_KV_PORT", 19001));
}

std::string default_output_dir(const kv_receiver_config & config) {
    if (!config.output_dir.empty()) {
        return config.output_dir;
    }
    const std::string env_out = env_string("LLAMA_KV_RECV_OUTPUT_DIR", "");
    if (!env_out.empty()) {
        return env_out;
    }
    return "/tmp/ik_kv_handoff";
}

std::unordered_map<std::string, std::string> parse_sc_kv(const std::vector<uint8_t> & payload) {
    std::unordered_map<std::string, std::string> out;
    if (payload.empty()) {
        return out;
    }

    std::string text(payload.begin(), payload.end());
    std::vector<std::string> parts = string_split(text, ';');
    for (const std::string & p : parts) {
        const size_t eq = p.find('=');
        if (eq == std::string::npos) {
            continue;
        }
        std::string key = string_strip(p.substr(0, eq));
        std::string value = string_strip(p.substr(eq + 1));
        if (!key.empty()) {
            out[key] = value;
        }
    }
    return out;
}

uint32_t crc32_u32(const uint8_t * data, size_t size) {
    static uint32_t table[256];
    static bool init = false;

    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                crc = (crc & 1u) ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
            }
            table[i] = crc;
        }
        init = true;
    }

    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

void append_u16_le(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back((uint8_t) ((v >> 0) & 0xFFu));
    out.push_back((uint8_t) ((v >> 8) & 0xFFu));
}

void append_u32_le(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t) ((v >> 0) & 0xFFu));
    out.push_back((uint8_t) ((v >> 8) & 0xFFu));
    out.push_back((uint8_t) ((v >> 16) & 0xFFu));
    out.push_back((uint8_t) ((v >> 24) & 0xFFu));
}

void append_u64_le(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back((uint8_t) ((v >> (8 * i)) & 0xFFu));
    }
}

void patch_u32_le(std::vector<uint8_t> & out, size_t offset, uint32_t v) {
    if (offset + 4 > out.size()) {
        return;
    }
    out[offset + 0] = (uint8_t) ((v >> 0) & 0xFFu);
    out[offset + 1] = (uint8_t) ((v >> 8) & 0xFFu);
    out[offset + 2] = (uint8_t) ((v >> 16) & 0xFFu);
    out[offset + 3] = (uint8_t) ((v >> 24) & 0xFFu);
}

uint16_t read_u16_le(const uint8_t * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

uint32_t read_u32_le(const uint8_t * p) {
    return (uint32_t) p[0] |
           ((uint32_t) p[1] << 8) |
           ((uint32_t) p[2] << 16) |
           ((uint32_t) p[3] << 24);
}

uint64_t read_u64_le(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= ((uint64_t) p[i]) << (8 * i);
    }
    return v;
}

std::string seq_chunk_name(uint64_t seq_no) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "seq_%020llu.bin", (unsigned long long) seq_no);
    return std::string(buf);
}

#ifndef _WIN32

void close_fd(int fd) {
    if (fd >= 0) {
        ::close(fd);
    }
}

bool set_non_blocking(int fd, std::string * error) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        if (error) {
            *error = "fcntl(F_GETFL) failed: " + std::string(std::strerror(errno));
        }
        return false;
    }
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        if (error) {
            *error = "fcntl(F_SETFL) failed: " + std::string(std::strerror(errno));
        }
        return false;
    }
    return true;
}

void configure_accepted_socket(int fd, const kv_receiver_config & config) {
    int one = 1;
    (void) ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    if (config.socket_send_buf > 0) {
        (void) ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &config.socket_send_buf, sizeof(config.socket_send_buf));
    }
    if (config.socket_recv_buf > 0) {
        (void) ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &config.socket_recv_buf, sizeof(config.socket_recv_buf));
    }
}

int create_listen_socket(const std::string & host, int32_t port, int32_t max_connections, std::string * error) {
    if (port <= 0 || port > 65535) {
        if (error) {
            *error = "invalid receiver port: " + std::to_string(port);
        }
        return -1;
    }

    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    const std::string port_str = std::to_string(port);
    struct addrinfo * addrs = nullptr;
    const char * bind_host = host.empty() ? nullptr : host.c_str();
    int gai_rc = ::getaddrinfo(bind_host, port_str.c_str(), &hints, &addrs);
    if (gai_rc != 0) {
        if (error) {
            *error = "getaddrinfo failed for host='" + host + "' port='" + port_str + "': " + std::string(gai_strerror(gai_rc));
        }
        return -1;
    }

    int listen_fd = -1;
    for (addrinfo * ai = addrs; ai != nullptr; ai = ai->ai_next) {
        int fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }

        int one = 1;
        (void) ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

        if (::bind(fd, ai->ai_addr, ai->ai_addrlen) != 0) {
            close_fd(fd);
            continue;
        }

        const int backlog = std::max<int32_t>(8, max_connections);
        if (::listen(fd, backlog) != 0) {
            close_fd(fd);
            continue;
        }

        std::string nb_err;
        if (!set_non_blocking(fd, &nb_err)) {
            close_fd(fd);
            if (error) {
                *error = nb_err;
            }
            ::freeaddrinfo(addrs);
            return -1;
        }

        listen_fd = fd;
        break;
    }

    ::freeaddrinfo(addrs);

    if (listen_fd < 0 && error && error->empty()) {
        *error = "failed to bind/listen on host='" + host + "' port='" + std::to_string(port) + "'";
    }

    return listen_fd;
}

bool send_all_bytes(int fd, const uint8_t * data, size_t size, std::string * error) {
    size_t off = 0;
    while (off < size) {
        const ssize_t n = ::send(fd, data + off, size - off, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (error) {
                *error = "send failed: " + std::string(std::strerror(errno));
            }
            return false;
        }
        if (n == 0) {
            if (error) {
                *error = "send returned 0";
            }
            return false;
        }
        off += (size_t) n;
    }
    return true;
}

bool recv_all_bytes_timeout(int fd,
                            uint8_t * dst,
                            size_t size,
                            int timeout_ms,
                            const std::atomic<bool> & stop_requested,
                            std::string * error) {
    size_t off = 0;
    while (off < size) {
        if (stop_requested.load()) {
            if (error) {
                *error = "receiver stopping";
            }
            return false;
        }

        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);

        struct timeval tv = {};
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        const int sel = ::select(fd + 1, &rfds, nullptr, nullptr, &tv);
        if (sel < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (error) {
                *error = "select failed: " + std::string(std::strerror(errno));
            }
            return false;
        }
        if (sel == 0) {
            if (error) {
                *error = "receive timeout";
            }
            return false;
        }

        const ssize_t n = ::recv(fd, dst + off, size - off, 0);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            if (error) {
                *error = "recv failed: " + std::string(std::strerror(errno));
            }
            return false;
        }
        if (n == 0) {
            if (error) {
                *error = "peer closed connection";
            }
            return false;
        }

        off += (size_t) n;
    }

    return true;
}

bool send_tbp_frame(int fd,
                    uint16_t msg_type,
                    uint16_t flags,
                    uint64_t session_id,
                    uint64_t stream_id,
                    uint64_t seq_no,
                    const uint8_t * payload,
                    size_t payload_bytes,
                    std::string * error) {
    const uint32_t payload_crc = (payload_bytes > 0 && payload != nullptr)
        ? crc32_u32(payload, payload_bytes)
        : 0u;

    std::vector<uint8_t> hdr;
    hdr.reserve(TBP_HEADER_SIZE);
    append_u32_le(hdr, TBP_MAGIC);
    append_u16_le(hdr, TBP_PROTO_MAJOR);
    append_u16_le(hdr, TBP_PROTO_MINOR);
    append_u16_le(hdr, msg_type);
    append_u16_le(hdr, flags);
    append_u32_le(hdr, TBP_HEADER_SIZE);
    append_u32_le(hdr, (uint32_t) payload_bytes);
    append_u64_le(hdr, session_id);
    append_u64_le(hdr, stream_id);
    append_u64_le(hdr, seq_no);
    const size_t hdr_crc_offset = hdr.size();
    append_u32_le(hdr, 0u); // placeholder
    append_u32_le(hdr, payload_crc);
    const uint32_t hdr_crc = crc32_u32(hdr.data(), hdr.size());
    patch_u32_le(hdr, hdr_crc_offset, hdr_crc);

    if (!send_all_bytes(fd, hdr.data(), hdr.size(), error)) {
        return false;
    }
    if (payload_bytes > 0 && payload != nullptr) {
        if (!send_all_bytes(fd, payload, payload_bytes, error)) {
            return false;
        }
    }

    return true;
}

bool recv_tbp_frame(int fd,
                    int timeout_ms,
                    const std::atomic<bool> & stop_requested,
                    tbp_frame * out,
                    std::string * error) {
    std::array<uint8_t, TBP_HEADER_SIZE> hdr = {};

    if (!recv_all_bytes_timeout(fd, hdr.data(), hdr.size(), timeout_ms, stop_requested, error)) {
        return false;
    }

    const uint32_t magic = read_u32_le(hdr.data() + 0);
    const uint16_t maj = read_u16_le(hdr.data() + 4);
    const uint16_t min = read_u16_le(hdr.data() + 6);
    const uint16_t msg_type = read_u16_le(hdr.data() + 8);
    const uint16_t flags = read_u16_le(hdr.data() + 10);
    const uint32_t header_bytes = read_u32_le(hdr.data() + 12);
    const uint32_t payload_bytes = read_u32_le(hdr.data() + 16);
    const uint64_t session_id = read_u64_le(hdr.data() + 20);
    const uint64_t stream_id = read_u64_le(hdr.data() + 28);
    const uint64_t seq_no = read_u64_le(hdr.data() + 36);
    const uint32_t header_crc_wire = read_u32_le(hdr.data() + 44);
    const uint32_t payload_crc = read_u32_le(hdr.data() + 48);

    if (magic != TBP_MAGIC) {
        if (error) {
            *error = "invalid TBP magic";
        }
        return false;
    }
    if (maj != TBP_PROTO_MAJOR || min != TBP_PROTO_MINOR) {
        if (error) {
            *error = "unsupported TBP protocol version";
        }
        return false;
    }
    if (header_bytes != TBP_HEADER_SIZE) {
        if (error) {
            *error = "invalid TBP header size";
        }
        return false;
    }
    if (payload_bytes > TBP_MAX_PAYLOAD_BYTES) {
        if (error) {
            *error = "TBP payload exceeds max supported size";
        }
        return false;
    }

    std::array<uint8_t, TBP_HEADER_SIZE> verify = hdr;
    verify[44] = 0;
    verify[45] = 0;
    verify[46] = 0;
    verify[47] = 0;
    const uint32_t header_crc_calc = crc32_u32(verify.data(), verify.size());
    if (header_crc_calc != header_crc_wire) {
        if (error) {
            *error = "TBP header CRC mismatch";
        }
        return false;
    }

    out->msg_type = msg_type;
    out->flags = flags;
    out->session_id = session_id;
    out->stream_id = stream_id;
    out->seq_no = seq_no;
    out->payload_crc = payload_crc;
    out->payload.clear();

    if (payload_bytes > 0) {
        out->payload.resize(payload_bytes);
        if (!recv_all_bytes_timeout(fd, out->payload.data(), payload_bytes, timeout_ms, stop_requested, error)) {
            return false;
        }
    }

    return true;
}

#endif // !_WIN32

} // namespace

struct kv_receiver_service::impl {
    explicit impl(server_queue & tasks, server_response & results)
        : queue_tasks(tasks), queue_results(results) {}

    server_queue & queue_tasks;
    server_response & queue_results;

    kv_receiver_config config = {};

    mutable std::mutex runtime_mutex;
    kv_receiver_runtime runtime_state = {};

    std::atomic<bool> stop_requested{false};

#ifndef _WIN32
    int listen_fd = -1;
    std::thread accept_thread;

    struct worker_entry {
        std::thread worker;
        std::shared_ptr<std::atomic<bool>> done;
    };

    std::mutex workers_mutex;
    std::vector<worker_entry> workers;

    std::mutex conn_mutex;
    std::set<int> active_connections;

    std::mutex sessions_mutex;
    std::unordered_map<uint64_t, std::shared_ptr<session_state>> sessions;
#endif

    bool start(const kv_receiver_config & cfg, std::string * error) {
        (void) queue_results;

        const std::string normalized_mode = normalize_transport_mode(cfg.transport_mode);
        if (normalized_mode.empty()) {
            if (error) {
                *error = "invalid kv transport mode '" + cfg.transport_mode + "' (allowed: auto,rdma,tcp,mixed,disabled; aliases: tb-direct,tb-ethernet,ethernet)";
            }
            return false;
        }
        if (cfg.slot_id < 0) {
            if (error) {
                *error = "invalid kv receiver slot id: " + std::to_string(cfg.slot_id);
            }
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(runtime_mutex);
            runtime_state = kv_receiver_runtime{};
            runtime_state.enabled = cfg.enabled;
            runtime_state.requested_transport_mode = normalized_mode;
        }

        if (!cfg.enabled || normalized_mode == "disabled") {
            std::lock_guard<std::mutex> lock(runtime_mutex);
            runtime_state.running = false;
            runtime_state.resolved_transport_mode = "disabled";
            runtime_state.bind_host.clear();
            runtime_state.bind_port = -1;
            runtime_state.output_dir.clear();
            return true;
        }

#ifdef _WIN32
        if (error) {
            *error = "KV receiver currently supports only unix-like platforms";
        }
        return false;
#else
        config = cfg;
        config.transport_mode = normalized_mode;
        config.max_connections = std::max<int32_t>(1, config.max_connections);
        config.idle_timeout_sec = std::max<int32_t>(1, config.idle_timeout_sec);
        config.output_dir = default_output_dir(config);

        std::error_code ec;
        fs::create_directories(config.output_dir, ec);
        if (ec) {
            if (error) {
                *error = "failed to create kv receiver output dir '" + config.output_dir + "': " + ec.message();
            }
            return false;
        }

        std::vector<std::string> modes = mode_order_for_receiver(normalized_mode, config.transport_fallback);
        if (modes.empty()) {
            if (error) {
                *error = "failed to resolve kv receiver mode order";
            }
            return false;
        }

        std::string bind_host;
        int32_t bind_port = -1;
        std::string bind_error;

        for (const std::string & mode : modes) {
            bind_host = config.host.empty() ? default_bind_host_for_mode(mode) : config.host;
            bind_port = config.port > 0 ? config.port : default_bind_port_for_mode(mode);

            std::string mode_error;
            int fd = create_listen_socket(bind_host, bind_port, config.max_connections, &mode_error);
            if (fd >= 0) {
                listen_fd = fd;
                std::lock_guard<std::mutex> lock(runtime_mutex);
                runtime_state.running = true;
                runtime_state.resolved_transport_mode = mode;
                runtime_state.bind_host = bind_host;
                runtime_state.bind_port = bind_port;
                runtime_state.output_dir = config.output_dir;
                break;
            }

            bind_error = mode_error;
            KVR_WRN("listen setup failed for mode=%s host=%s port=%d err=%s\n",
                    mode.c_str(), bind_host.c_str(), bind_port, bind_error.c_str());
        }

        if (listen_fd < 0) {
            if (error) {
                *error = bind_error.empty() ? "failed to start kv receiver" : bind_error;
            }
            return false;
        }

        stop_requested.store(false);
        accept_thread = std::thread([this]() { accept_loop(); });

        KVR_INF("KV receiver started mode=%s host=%s port=%d output=%s\n",
                runtime_state.resolved_transport_mode.c_str(),
                runtime_state.bind_host.c_str(),
                runtime_state.bind_port,
                runtime_state.output_dir.c_str());

        return true;
#endif
    }

    void stop() {
#ifdef _WIN32
        std::lock_guard<std::mutex> lock(runtime_mutex);
        runtime_state.running = false;
#else
        stop_requested.store(true);

        int fd_to_close = -1;
        {
            fd_to_close = listen_fd;
            listen_fd = -1;
        }
        if (fd_to_close >= 0) {
            close_fd(fd_to_close);
        }

        close_active_connections();

        if (accept_thread.joinable()) {
            accept_thread.join();
        }

        {
            std::lock_guard<std::mutex> lock(workers_mutex);
            for (auto & entry : workers) {
                if (entry.worker.joinable()) {
                    entry.worker.join();
                }
            }
            workers.clear();
        }

        {
            std::lock_guard<std::mutex> lock(runtime_mutex);
            runtime_state.running = false;
        }
#endif
    }

    kv_receiver_runtime runtime() const {
        std::lock_guard<std::mutex> lock(runtime_mutex);
        return runtime_state;
    }

#ifndef _WIN32
    void release_connection_fd(int fd) {
        bool should_close = false;
        {
            std::lock_guard<std::mutex> lock(conn_mutex);
            auto it = active_connections.find(fd);
            if (it != active_connections.end()) {
                active_connections.erase(it);
                should_close = true;
            }
        }
        if (should_close) {
            close_fd(fd);
        }
    }

    void close_active_connections() {
        std::vector<int> fds;
        {
            std::lock_guard<std::mutex> lock(conn_mutex);
            fds.assign(active_connections.begin(), active_connections.end());
            active_connections.clear();
        }

        for (int fd : fds) {
            ::shutdown(fd, SHUT_RDWR);
            close_fd(fd);
        }
    }

    void reap_worker_threads() {
        std::lock_guard<std::mutex> lock(workers_mutex);
        auto it = workers.begin();
        while (it != workers.end()) {
            if (it->done && it->done->load()) {
                if (it->worker.joinable()) {
                    it->worker.join();
                }
                it = workers.erase(it);
            } else {
                ++it;
            }
        }
    }

    void accept_loop() {
        while (!stop_requested.load()) {
            reap_worker_threads();

            const int current_listen_fd = listen_fd;
            if (current_listen_fd < 0) {
                break;
            }

            fd_set rfds;
            FD_ZERO(&rfds);
            FD_SET(current_listen_fd, &rfds);

            struct timeval tv = {};
            tv.tv_sec = 0;
            tv.tv_usec = 200 * 1000;

            const int sel = ::select(current_listen_fd + 1, &rfds, nullptr, nullptr, &tv);
            if (sel < 0) {
                if (errno == EINTR) {
                    continue;
                }
                if (!stop_requested.load()) {
                    KVR_WRN("accept select failed: %s\n", std::strerror(errno));
                }
                break;
            }
            if (sel == 0) {
                continue;
            }

            while (!stop_requested.load()) {
                struct sockaddr_storage addr = {};
                socklen_t len = sizeof(addr);
                const int fd = ::accept(current_listen_fd, reinterpret_cast<struct sockaddr *>(&addr), &len);
                if (fd < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                        break;
                    }
                    if (!stop_requested.load()) {
                        KVR_WRN("accept failed: %s\n", std::strerror(errno));
                    }
                    break;
                }

                bool accepted = false;
                {
                    std::lock_guard<std::mutex> lock(conn_mutex);
                    if ((int32_t) active_connections.size() < std::max<int32_t>(1, config.max_connections)) {
                        active_connections.insert(fd);
                        accepted = true;
                    }
                }

                if (!accepted) {
                    KVR_WRN("dropping connection: max connections reached (%d)\n", config.max_connections);
                    close_fd(fd);
                    continue;
                }

                configure_accepted_socket(fd, config);

                auto done = std::make_shared<std::atomic<bool>>(false);
                std::lock_guard<std::mutex> lock(workers_mutex);
                workers.push_back(worker_entry{
                    std::thread([this, fd, done]() {
                        connection_loop(fd);
                        done->store(true);
                    }),
                    done,
                });
            }
        }
    }

    std::shared_ptr<session_state> get_or_create_session(uint64_t session_id) {
        std::lock_guard<std::mutex> lock(sessions_mutex);
        auto it = sessions.find(session_id);
        if (it != sessions.end()) {
            return it->second;
        }

        std::shared_ptr<session_state> session = std::make_shared<session_state>();
        session->session_id = session_id;
        session->session_dir = fs::path(config.output_dir) / ("session_" + std::to_string(session_id));
        session->chunks_dir = session->session_dir / "chunks";
        session->artifact_path = session->session_dir / ("kv_artifact_" + std::to_string(session_id) + ".kva");

        std::error_code ec;
        fs::create_directories(session->chunks_dir, ec);
        if (ec) {
            KVR_WRN("failed to create session chunk dir sid=%" PRIu64 " err=%s\n", session_id, ec.message().c_str());
        }

        sessions[session_id] = session;
        return session;
    }

    bool persist_chunk(const std::shared_ptr<session_state> & session, uint64_t seq_no, const std::vector<uint8_t> & payload, std::string * error) {
        std::error_code ec;
        fs::create_directories(session->chunks_dir, ec);
        if (ec) {
            if (error) {
                *error = "failed to create chunk dir: " + ec.message();
            }
            return false;
        }

        const fs::path chunk_path = session->chunks_dir / seq_chunk_name(seq_no);
        std::ofstream ofs(chunk_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            if (error) {
                *error = "failed to open chunk file for write: " + chunk_path.string();
            }
            return false;
        }

        if (!payload.empty()) {
            ofs.write(reinterpret_cast<const char *>(payload.data()), (std::streamsize) payload.size());
        }

        if (!ofs.good()) {
            if (error) {
                *error = "failed to write chunk payload: " + chunk_path.string();
            }
            return false;
        }

        return true;
    }

    bool reassemble_session_artifact(const std::shared_ptr<session_state> & session, std::string * error) {
        std::vector<fs::path> chunk_files;
        std::error_code ec;

        for (const auto & entry : fs::directory_iterator(session->chunks_dir, ec)) {
            if (ec) {
                if (error) {
                    *error = "failed to iterate chunk dir: " + ec.message();
                }
                return false;
            }
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind("seq_", 0) == 0 && entry.path().extension() == ".bin") {
                chunk_files.push_back(entry.path());
            }
        }

        std::sort(chunk_files.begin(), chunk_files.end());
        if (chunk_files.empty()) {
            if (error) {
                *error = "no chunk files available for session";
            }
            return false;
        }

        const fs::path temp_path = session->artifact_path.string() + ".tmp";
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            if (error) {
                *error = "failed to open artifact temp file for write: " + temp_path.string();
            }
            return false;
        }

        uint64_t written = 0;
        for (const fs::path & chunk : chunk_files) {
            std::ifstream in(chunk, std::ios::binary);
            if (!in.is_open()) {
                if (error) {
                    *error = "failed to open chunk during reassembly: " + chunk.string();
                }
                return false;
            }

            out << in.rdbuf();
            if (!out.good()) {
                if (error) {
                    *error = "failed writing reassembled artifact";
                }
                return false;
            }

            written += (uint64_t) fs::file_size(chunk, ec);
            if (ec) {
                ec.clear();
            }
        }

        out.close();
        if (!out.good()) {
            if (error) {
                *error = "failed finalizing reassembled artifact";
            }
            return false;
        }

        fs::rename(temp_path, session->artifact_path, ec);
        if (ec) {
            if (error) {
                *error = "failed to commit artifact file: " + ec.message();
            }
            return false;
        }

        KVR_INF("reassembled artifact sid=%" PRIu64 " bytes=%" PRIu64 " path=%s\n",
                session->session_id,
                written,
                session->artifact_path.c_str());
        return true;
    }

    void enqueue_slot_restore(const std::shared_ptr<session_state> & session) {
        server_task task;
        task.type = SERVER_TASK_TYPE_SLOT_RESTORE;
        task.data = {
            {"id_slot", config.slot_id},
            {"filename", session->artifact_path.filename().string()},
            {"filepath", session->artifact_path.string()},
        };

        const int id_task = queue_tasks.post(std::move(task));
        KVR_INF("queued slot restore sid=%" PRIu64 " slot=%d task_id=%d\n",
                session->session_id, config.slot_id, id_task);
    }

    void try_finalize_session(const std::shared_ptr<session_state> & session) {
        bool should_finalize = false;
        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (session->finalized) {
                return;
            }

            const size_t expected = session->expected_streams > 0
                ? (size_t) session->expected_streams
                : std::max<size_t>(1, session->seen_stream_ids.size());

            if (session->done_stream_ids.size() < expected) {
                return;
            }

            session->finalized = true;
            should_finalize = true;
        }

        if (!should_finalize) {
            return;
        }

        std::string err;
        if (!reassemble_session_artifact(session, &err)) {
            KVR_WRN("session finalize failed sid=%" PRIu64 " err=%s\n", session->session_id, err.c_str());
            return;
        }

        enqueue_slot_restore(session);
    }

    void process_frame(int fd, const tbp_frame & frame) {
        std::shared_ptr<session_state> session = get_or_create_session(frame.session_id);

        {
            std::lock_guard<std::mutex> lock(session->mutex);
            if (frame.stream_id > 0) {
                session->seen_stream_ids.insert(frame.stream_id);
            }
        }

        switch (frame.msg_type) {
            case TBP_MSG_HELLO:
                break;
            case TBP_MSG_SESSION_START:
            {
                const auto kv = parse_sc_kv(frame.payload);
                auto it = kv.find("streams");
                if (it != kv.end()) {
                    try {
                        const int streams = std::stoi(it->second);
                        std::lock_guard<std::mutex> lock(session->mutex);
                        session->expected_streams = std::max(session->expected_streams, streams);
                    } catch (...) {
                    }
                }
            } break;
            case TBP_MSG_KV_SEGMENT_BEGIN:
                break;
            case TBP_MSG_KV_CHUNK:
            {
                const uint32_t calc_crc = frame.payload.empty() ? 0u : crc32_u32(frame.payload.data(), frame.payload.size());
                const bool crc_ok = calc_crc == frame.payload_crc;

                {
                    std::lock_guard<std::mutex> lock(session->mutex);
                    if (crc_ok) {
                        session->chunks_received += 1;
                        session->bytes_received += frame.payload.size();
                    } else {
                        session->bad_crc_chunks += 1;
                    }
                }

                if (crc_ok) {
                    std::string write_err;
                    if (!persist_chunk(session, frame.seq_no, frame.payload, &write_err)) {
                        KVR_WRN("persist chunk failed sid=%" PRIu64 " seq=%" PRIu64 " err=%s\n",
                                frame.session_id, frame.seq_no, write_err.c_str());
                    }
                }

                if (config.ack_enabled) {
                    const std::string ack_payload = (!crc_ok && config.nack_on_crc_bad) ? "nack=crc" : "ack=1";
                    std::string ack_err;
                    if (!send_tbp_frame(fd,
                                        TBP_MSG_KV_ACK,
                                        0,
                                        frame.session_id,
                                        frame.stream_id,
                                        frame.seq_no,
                                        reinterpret_cast<const uint8_t *>(ack_payload.data()),
                                        ack_payload.size(),
                                        &ack_err)) {
                        KVR_WRN("failed sending ACK sid=%" PRIu64 " stream=%" PRIu64 " seq=%" PRIu64 " err=%s\n",
                                frame.session_id, frame.stream_id, frame.seq_no, ack_err.c_str());
                    }
                }
            } break;
            case TBP_MSG_KV_SEGMENT_END:
                break;
            case TBP_MSG_KV_DONE:
            {
                {
                    std::lock_guard<std::mutex> lock(session->mutex);
                    session->done_stream_ids.insert(frame.stream_id);
                }
                try_finalize_session(session);
            } break;
            default:
                KVR_DBG("ignored unknown TBP frame type=%u sid=%" PRIu64 "\n", frame.msg_type, frame.session_id);
                break;
        }
    }

    void connection_loop(int fd) {
        for (;;) {
            if (stop_requested.load()) {
                break;
            }

            tbp_frame frame = {};
            std::string recv_err;
            const int timeout_ms = std::max<int32_t>(1, config.idle_timeout_sec) * 1000;
            if (!recv_tbp_frame(fd, timeout_ms, stop_requested, &frame, &recv_err)) {
                if (!stop_requested.load() && recv_err != "peer closed connection" && recv_err != "receiver stopping") {
                    KVR_DBG("connection frame receive stopped fd=%d err=%s\n", fd, recv_err.c_str());
                }
                break;
            }

            process_frame(fd, frame);
        }

        release_connection_fd(fd);
    }

#endif
};

kv_receiver_config kv_receiver_config_from_params(const gpt_params & params) {
    kv_receiver_config cfg = {};
    cfg.enabled = params.kv_receiver_enable;
    cfg.transport_mode = params.kv_transport;
    cfg.transport_fallback = params.kv_transport_fallback;

    cfg.host = params.kv_receiver_host;
    cfg.port = params.kv_receiver_port;

    cfg.slot_id = params.kv_receiver_slot_id;
    cfg.output_dir = params.kv_receiver_output_dir;

    cfg.ack_enabled = params.kv_receiver_ack;
    cfg.nack_on_crc_bad = params.kv_receiver_nack_on_crc_bad;

    cfg.max_connections = params.kv_receiver_max_connections;
    cfg.idle_timeout_sec = params.kv_receiver_idle_timeout_sec;
    cfg.socket_send_buf = params.kv_receiver_socket_send_buf;
    cfg.socket_recv_buf = params.kv_receiver_socket_recv_buf;

    if (cfg.output_dir.empty() && !params.slot_save_path.empty()) {
        cfg.output_dir = params.slot_save_path;
    }

    // Convenience: enabling receiver without explicit mode defaults to auto.
    if (cfg.enabled) {
        const std::string normalized = normalize_transport_mode(cfg.transport_mode);
        if (normalized == "disabled") {
            cfg.transport_mode = "auto";
        }
    }

    return cfg;
}

kv_receiver_service::kv_receiver_service(server_queue & queue_tasks, server_response & queue_results)
    : pimpl(std::make_unique<impl>(queue_tasks, queue_results)) {}

kv_receiver_service::~kv_receiver_service() {
    stop();
}

bool kv_receiver_service::start(const kv_receiver_config & config, std::string * error) {
    return pimpl->start(config, error);
}

void kv_receiver_service::stop() {
    pimpl->stop();
}

kv_receiver_runtime kv_receiver_service::runtime() const {
    return pimpl->runtime();
}
