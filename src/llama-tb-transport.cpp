#include "llama-tb-transport.h"
#include "llama-kv-artifact.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#ifndef _WIN32
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
typedef int tb_socket_t;
static constexpr tb_socket_t TB_SOCKET_INVALID = -1;
#  define TB_SOCKET_CLOSE(fd) ::close(fd)
#else
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "Ws2_32.lib")
typedef SOCKET tb_socket_t;
static const tb_socket_t TB_SOCKET_INVALID = INVALID_SOCKET;
#  define TB_SOCKET_CLOSE(fd) ::closesocket(fd)

static bool tb_winsock_init() {
    static bool done = false;
    if (!done) {
        WSADATA wsa;
        done = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
    }
    return done;
}
#endif

namespace {

constexpr uint16_t TBP_PROTO_MAJOR = 1;
constexpr uint16_t TBP_PROTO_MINOR = 0;
constexpr uint32_t TBP_MAGIC       = 0x54425031u;

enum tbp_msg_type : uint16_t {
    TBP_MSG_HELLO            = 1,
    TBP_MSG_SESSION_START    = 3,
    TBP_MSG_KV_SEGMENT_BEGIN = 7,
    TBP_MSG_KV_CHUNK         = 8,
    TBP_MSG_KV_SEGMENT_END   = 9,
    TBP_MSG_KV_DONE          = 12,
};

void append_u16_le(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back((uint8_t)((v >> 0) & 0xFF));
    out.push_back((uint8_t)((v >> 8) & 0xFF));
}
void append_u32_le(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t)((v >> 0) & 0xFF));
    out.push_back((uint8_t)((v >> 8) & 0xFF));
    out.push_back((uint8_t)((v >> 16) & 0xFF));
    out.push_back((uint8_t)((v >> 24) & 0xFF));
}
void append_u64_le(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back((uint8_t)((v >> (8 * i)) & 0xFF));
}
void patch_u32_le(std::vector<uint8_t> & out, size_t offset, uint32_t v) {
    if (offset + 4 > out.size()) return;
    out[offset + 0] = (uint8_t)((v >> 0) & 0xFF);
    out[offset + 1] = (uint8_t)((v >> 8) & 0xFF);
    out[offset + 2] = (uint8_t)((v >> 16) & 0xFF);
    out[offset + 3] = (uint8_t)((v >> 24) & 0xFF);
}

uint64_t hash_session_id(const std::string & sid) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : sid) { h ^= (uint64_t)c; h *= 1099511628211ull; }
    return h;
}

void set_error(std::string * error, const std::string & msg) {
    if (error) *error = msg;
}

bool send_all_bytes(tb_socket_t fd, const uint8_t * data, size_t size, std::string * error) {
    size_t off = 0;
    while (off < size) {
#ifndef _WIN32
        ssize_t n = ::send(fd, data + off, size - off, 0);
#else
        int n = ::send(fd, (const char *)(data + off), (int)(size - off), 0);
#endif
        if (n <= 0) { set_error(error, "transport send failed"); return false; }
        off += (size_t)n;
    }
    return true;
}

bool send_tbp_frame(tb_socket_t fd, uint16_t msg_type, uint64_t session_id, uint64_t stream_id, uint64_t seq_no,
                    const uint8_t * payload, size_t payload_bytes, std::string * error) {
    const uint32_t payload_crc = (payload_bytes > 0 && payload) ?
        llama_kv_artifact_crc32(payload, payload_bytes) : 0u;

    std::vector<uint8_t> hdr;
    hdr.reserve(52);
    append_u32_le(hdr, TBP_MAGIC);
    append_u16_le(hdr, TBP_PROTO_MAJOR);
    append_u16_le(hdr, TBP_PROTO_MINOR);
    append_u16_le(hdr, msg_type);
    append_u16_le(hdr, 0);  // flags
    append_u32_le(hdr, 52u); // header size
    append_u32_le(hdr, (uint32_t)payload_bytes);
    append_u64_le(hdr, session_id);
    append_u64_le(hdr, stream_id);
    append_u64_le(hdr, seq_no);
    size_t hdr_crc_offset = hdr.size();
    append_u32_le(hdr, 0u);  // header CRC placeholder
    append_u32_le(hdr, payload_crc);
    uint32_t hdr_crc = llama_kv_artifact_crc32(hdr.data(), hdr.size());
    patch_u32_le(hdr, hdr_crc_offset, hdr_crc);

    if (!send_all_bytes(fd, hdr.data(), hdr.size(), error)) return false;
    if (payload_bytes > 0 && payload) {
        if (!send_all_bytes(fd, payload, payload_bytes, error)) return false;
    }
    return true;
}

tb_socket_t connect_endpoint(const std::string & host, int port,
                             int sock_send_buf, int sock_recv_buf,
                             std::string * error) {
#ifdef _WIN32
    if (!tb_winsock_init()) { set_error(error, "winsock init failed"); return TB_SOCKET_INVALID; }
#endif
    std::string port_str = std::to_string(port);
    struct addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo * ai = nullptr;
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &ai) != 0) {
        set_error(error, "getaddrinfo failed for " + host + ":" + port_str);
        return TB_SOCKET_INVALID;
    }

    tb_socket_t fd = TB_SOCKET_INVALID;
    for (struct addrinfo * cur = ai; cur; cur = cur->ai_next) {
        fd = ::socket(cur->ai_family, cur->ai_socktype, cur->ai_protocol);
        if (fd == TB_SOCKET_INVALID) continue;

#ifndef _WIN32
        if (sock_send_buf > 0) setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sock_send_buf, sizeof(sock_send_buf));
        if (sock_recv_buf > 0) setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sock_recv_buf, sizeof(sock_recv_buf));
#else
        if (sock_send_buf > 0) setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (const char *)&sock_send_buf, sizeof(sock_send_buf));
        if (sock_recv_buf > 0) setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (const char *)&sock_recv_buf, sizeof(sock_recv_buf));
#endif
        if (::connect(fd, cur->ai_addr, (int)cur->ai_addrlen) == 0) break;
        TB_SOCKET_CLOSE(fd);
        fd = TB_SOCKET_INVALID;
    }
    freeaddrinfo(ai);

    if (fd == TB_SOCKET_INVALID) {
        set_error(error, "failed to connect to " + host + ":" + port_str);
    }
    return fd;
}

}  // namespace

bool llama_tb_transport_send_artifact(
    const uint8_t *                   data,
    size_t                            size,
    const llama_tb_transfer_options & options,
    llama_tb_transfer_result *        result,
    std::string *                     error) {

    if (result) *result = {};

    if (options.kv_host.empty() || options.kv_port <= 0) {
        set_error(error, "kv_host and kv_port must be set");
        return false;
    }

    tb_socket_t fd = connect_endpoint(options.kv_host, options.kv_port,
                                       options.socket_send_buf, options.socket_recv_buf, error);
    if (fd == TB_SOCKET_INVALID) return false;

    auto cleanup = [&]() { TB_SOCKET_CLOSE(fd); };

    const std::string sid = options.session_id.empty() ? "session_default" : options.session_id;
    const uint64_t wire_sid = hash_session_id(sid);
    uint64_t seq = 0;

    auto send_str = [&](uint16_t msg_type, const std::string & payload) -> bool {
        return send_tbp_frame(fd, msg_type, wire_sid, 1, seq++,
                              reinterpret_cast<const uint8_t *>(payload.data()), payload.size(), error);
    };

    auto transfer_t0 = std::chrono::steady_clock::now();

    // HELLO
    if (!send_str(TBP_MSG_HELLO, "node_role=prefill-server")) { cleanup(); return false; }

    // SESSION_START
    std::ostringstream ss;
    ss << "mode=" << (options.progressive ? "progressive" : "bulk")
       << ";bytes=" << size;
    if (!send_str(TBP_MSG_SESSION_START, ss.str())) { cleanup(); return false; }

    // SEG_BEGIN
    const size_t chunk_bytes = std::max<size_t>(1, options.chunk_bytes);
    std::ostringstream seg;
    seg << "segment_id=0;payload_bytes=" << size << ";chunk_bytes_nominal=" << chunk_bytes;
    if (!send_str(TBP_MSG_KV_SEGMENT_BEGIN, seg.str())) { cleanup(); return false; }

    // Send chunks
    uint32_t chunks_sent = 0;
    size_t offset = 0;
    while (offset < size) {
        size_t this_chunk = std::min(chunk_bytes, size - offset);
        if (!send_tbp_frame(fd, TBP_MSG_KV_CHUNK, wire_sid, 1, seq++,
                            data + offset, this_chunk, error)) {
            cleanup(); return false;
        }
        offset += this_chunk;
        chunks_sent++;
    }

    // SEG_END
    if (!send_str(TBP_MSG_KV_SEGMENT_END, "")) { cleanup(); return false; }

    // KV_DONE
    if (!send_str(TBP_MSG_KV_DONE, "")) { cleanup(); return false; }

    cleanup();

    auto transfer_t1 = std::chrono::steady_clock::now();
    double ms = (double)std::chrono::duration_cast<std::chrono::microseconds>(transfer_t1 - transfer_t0).count() / 1000.0;

    if (result) {
        result->bytes_sent      = size;
        result->chunks_sent     = chunks_sent;
        result->transfer_ms     = ms;
        result->throughput_gbps = ms > 0 ? ((double)size * 8.0) / (ms * 1e6) : 0.0;
    }
    return true;
}

bool llama_tb_transport_send_artifact_file(
    const std::string &               path,
    const llama_tb_transfer_options & options,
    llama_tb_transfer_result *        result,
    std::string *                     error) {

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        set_error(error, "failed to open artifact file: " + path);
        return false;
    }
    size_t size = (size_t)ifs.tellg();
    ifs.seekg(0);
    std::vector<uint8_t> data(size);
    ifs.read(reinterpret_cast<char *>(data.data()), (std::streamsize)size);
    if (!ifs) {
        set_error(error, "failed to read artifact file: " + path);
        return false;
    }
    return llama_tb_transport_send_artifact(data.data(), data.size(), options, result, error);
}
