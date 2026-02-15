#include "llama-kv-artifact.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace {

constexpr std::array<char, 8> LLAMA_KV_ARTIFACT_MAGIC = { 'K', 'V', 'A', 'R', 'T', 'I', 'F', '1' };

struct kv_artifact_header_v1 {
    char     magic[8];
    uint16_t format_major;
    uint16_t format_minor;
    uint32_t n_layers;
    uint32_t n_ctx;
    uint32_t token_count;
    uint16_t type_k;
    uint16_t type_v;
    uint32_t flags;
    uint64_t payload_bytes;
    uint32_t payload_crc32;
};

template <typename T>
bool read_exact(std::ifstream &ifs, T &value) {
    ifs.read(reinterpret_cast<char *>(&value), sizeof(T));
    return (bool) ifs;
}

template <typename T>
bool write_exact(std::ofstream &ofs, const T &value) {
    ofs.write(reinterpret_cast<const char *>(&value), sizeof(T));
    return (bool) ofs;
}

void set_error(std::string *error, std::string msg) {
    if (error) {
        *error = std::move(msg);
    }
}

}  // namespace

uint32_t llama_kv_artifact_crc32(const uint8_t *data, size_t size) {
    static uint32_t table[256];
    static bool table_init = false;
    if (!table_init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1u)) : (c >> 1u);
            }
            table[i] = c;
        }
        table_init = true;
    }

    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
    }
    return crc ^ 0xFFFFFFFFu;
}

bool llama_kv_artifact_write(const std::string &               path,
                             const std::vector<uint8_t> &      payload,
                             const llama_kv_artifact_metadata &meta,
                             llama_kv_artifact_summary *       summary,
                             std::string *                     error) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        set_error(error, "failed to open kv artifact for write: " + path);
        return false;
    }

    kv_artifact_header_v1 hdr = {};
    std::memcpy(hdr.magic, LLAMA_KV_ARTIFACT_MAGIC.data(), LLAMA_KV_ARTIFACT_MAGIC.size());
    hdr.format_major = meta.format_major;
    hdr.format_minor = meta.format_minor;
    hdr.n_layers     = meta.n_layers;
    hdr.n_ctx        = meta.n_ctx;
    hdr.token_count  = meta.token_count;
    hdr.type_k       = meta.type_k;
    hdr.type_v       = meta.type_v;
    hdr.flags        = meta.flags;
    hdr.payload_bytes = payload.size();
    hdr.payload_crc32 = payload.empty() ? 0u : llama_kv_artifact_crc32(payload.data(), payload.size());

    if (!write_exact(ofs, hdr)) {
        set_error(error, "failed to write kv artifact header: " + path);
        return false;
    }

    if (!payload.empty()) {
        ofs.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
        if (!ofs) {
            set_error(error, "failed to write kv artifact payload: " + path);
            return false;
        }
    }

    if (summary) {
        summary->payload_bytes = hdr.payload_bytes;
        summary->payload_crc32 = hdr.payload_crc32;
    }
    return true;
}

bool llama_kv_artifact_read(const std::string &         path,
                            std::vector<uint8_t> &      payload_out,
                            llama_kv_artifact_metadata *meta_out,
                            llama_kv_artifact_summary * summary_out,
                            std::string *               error) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        set_error(error, "failed to open kv artifact for read: " + path);
        return false;
    }

    kv_artifact_header_v1 hdr = {};
    if (!read_exact(ifs, hdr)) {
        set_error(error, "failed to read kv artifact header: " + path);
        return false;
    }

    if (!std::equal(std::begin(hdr.magic), std::end(hdr.magic), LLAMA_KV_ARTIFACT_MAGIC.begin())) {
        set_error(error, "invalid kv artifact magic: " + path);
        return false;
    }

    payload_out.resize(static_cast<size_t>(hdr.payload_bytes));
    if (!payload_out.empty()) {
        ifs.read(reinterpret_cast<char *>(payload_out.data()), static_cast<std::streamsize>(payload_out.size()));
        if (!ifs) {
            set_error(error, "failed to read kv artifact payload: " + path);
            return false;
        }
    }

    const uint32_t crc = payload_out.empty() ? 0u : llama_kv_artifact_crc32(payload_out.data(), payload_out.size());
    if (crc != hdr.payload_crc32) {
        set_error(error, "kv artifact crc mismatch: expected " + std::to_string(hdr.payload_crc32) +
                             " got " + std::to_string(crc));
        return false;
    }

    if (meta_out) {
        meta_out->format_major = hdr.format_major;
        meta_out->format_minor = hdr.format_minor;
        meta_out->n_layers     = hdr.n_layers;
        meta_out->n_ctx        = hdr.n_ctx;
        meta_out->token_count  = hdr.token_count;
        meta_out->type_k       = hdr.type_k;
        meta_out->type_v       = hdr.type_v;
        meta_out->flags        = hdr.flags;
    }
    if (summary_out) {
        summary_out->payload_bytes = hdr.payload_bytes;
        summary_out->payload_crc32 = hdr.payload_crc32;
    }
    return true;
}
