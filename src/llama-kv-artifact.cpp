#include "llama-kv-artifact.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>

namespace {

constexpr std::array<char, 8> LLAMA_KV_ARTIFACT_MAGIC = { 'K', 'V', 'A', 'R', 'T', 'I', 'F', '1' };

#pragma pack(push, 1)
struct kv_artifact_header_v1_base {
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

struct kv_artifact_header_v1_ext {
    int32_t  last_token_id;
    uint8_t  v_trans;
    uint8_t  is_mla;
    uint16_t reserved_0;
    uint64_t model_fingerprint;
};
#pragma pack(pop)

static_assert(sizeof(kv_artifact_header_v1_base) == 44, "base header must be 44 bytes");
static_assert(sizeof(kv_artifact_header_v1_ext)  == 16, "ext header must be 16 bytes");

static constexpr size_t HEADER_BASE_SIZE = 44;
static constexpr size_t HEADER_EXT_SIZE  = 16;
static constexpr size_t HEADER_V1_1_SIZE = HEADER_BASE_SIZE + HEADER_EXT_SIZE;

template <typename T>
bool read_exact(std::ifstream & ifs, T & value) {
    ifs.read(reinterpret_cast<char *>(&value), sizeof(T));
    return (bool)ifs;
}

template <typename T>
bool write_exact(std::ofstream & ofs, const T & value) {
    ofs.write(reinterpret_cast<const char *>(&value), sizeof(T));
    return (bool)ofs;
}

void set_error(std::string * error, std::string msg) {
    if (error) *error = std::move(msg);
}

}  // namespace

uint32_t llama_kv_artifact_crc32(const uint8_t * data, size_t size) {
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
                             const llama_kv_artifact_metadata & meta,
                             llama_kv_artifact_summary *        summary,
                             std::string *                      error) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
        set_error(error, "failed to open kv artifact for write: " + path);
        return false;
    }

    kv_artifact_header_v1_base base = {};
    std::memcpy(base.magic, LLAMA_KV_ARTIFACT_MAGIC.data(), LLAMA_KV_ARTIFACT_MAGIC.size());
    base.format_major  = meta.format_major;
    base.format_minor  = meta.format_minor;
    base.n_layers      = meta.n_layers;
    base.n_ctx         = meta.n_ctx;
    base.token_count   = meta.token_count;
    base.type_k        = meta.type_k;
    base.type_v        = meta.type_v;
    base.flags         = meta.flags;
    base.payload_bytes = payload.size();
    base.payload_crc32 = payload.empty() ? 0u : llama_kv_artifact_crc32(payload.data(), payload.size());

    if (!write_exact(ofs, base)) {
        set_error(error, "failed to write kv artifact base header: " + path);
        return false;
    }

    if (meta.format_minor >= 1) {
        kv_artifact_header_v1_ext ext = {};
        ext.last_token_id     = meta.last_token_id;
        ext.v_trans           = meta.v_trans;
        ext.is_mla            = meta.is_mla;
        ext.reserved_0        = meta.reserved_0;
        ext.model_fingerprint = meta.model_fingerprint;
        if (!write_exact(ofs, ext)) {
            set_error(error, "failed to write kv artifact ext header: " + path);
            return false;
        }
    }

    if (!payload.empty()) {
        ofs.write(reinterpret_cast<const char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
        if (!ofs) {
            set_error(error, "failed to write kv artifact payload: " + path);
            return false;
        }
    }

    if (summary) {
        summary->payload_bytes = base.payload_bytes;
        summary->payload_crc32 = base.payload_crc32;
    }
    return true;
}

bool llama_kv_artifact_read(const std::string &          path,
                            std::vector<uint8_t> &       payload_out,
                            llama_kv_artifact_metadata * meta_out,
                            llama_kv_artifact_summary *  summary_out,
                            std::string *                error) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        set_error(error, "failed to open kv artifact for read: " + path);
        return false;
    }

    kv_artifact_header_v1_base base = {};
    if (!read_exact(ifs, base)) {
        set_error(error, "failed to read kv artifact header: " + path);
        return false;
    }

    if (!std::equal(std::begin(base.magic), std::end(base.magic), LLAMA_KV_ARTIFACT_MAGIC.begin())) {
        set_error(error, "invalid kv artifact magic: " + path);
        return false;
    }

    kv_artifact_header_v1_ext ext = {};
    ext.last_token_id = -1;
    if (base.format_minor >= 1) {
        if (!read_exact(ifs, ext)) {
            set_error(error, "failed to read kv artifact ext header: " + path);
            return false;
        }
    }

    payload_out.resize(static_cast<size_t>(base.payload_bytes));
    if (!payload_out.empty()) {
        ifs.read(reinterpret_cast<char *>(payload_out.data()), static_cast<std::streamsize>(payload_out.size()));
        if (!ifs) {
            set_error(error, "failed to read kv artifact payload: " + path);
            return false;
        }
    }

    const uint32_t crc = payload_out.empty() ? 0u : llama_kv_artifact_crc32(payload_out.data(), payload_out.size());
    if (crc != base.payload_crc32) {
        set_error(error, "kv artifact crc mismatch: expected " + std::to_string(base.payload_crc32) +
                             " got " + std::to_string(crc));
        return false;
    }

    if (meta_out) {
        meta_out->format_major      = base.format_major;
        meta_out->format_minor      = base.format_minor;
        meta_out->n_layers          = base.n_layers;
        meta_out->n_ctx             = base.n_ctx;
        meta_out->token_count       = base.token_count;
        meta_out->type_k            = base.type_k;
        meta_out->type_v            = base.type_v;
        meta_out->flags             = base.flags;
        meta_out->last_token_id     = ext.last_token_id;
        meta_out->v_trans           = ext.v_trans;
        meta_out->is_mla            = ext.is_mla;
        meta_out->reserved_0        = ext.reserved_0;
        meta_out->model_fingerprint = ext.model_fingerprint;
    }
    if (summary_out) {
        summary_out->payload_bytes = base.payload_bytes;
        summary_out->payload_crc32 = base.payload_crc32;
    }
    return true;
}

bool llama_kv_artifact_read_mem(const uint8_t *              data,
                                size_t                       size,
                                std::vector<uint8_t> &       payload_out,
                                llama_kv_artifact_metadata * meta_out,
                                llama_kv_artifact_summary *  summary_out,
                                std::string *                error) {
    if (size < HEADER_BASE_SIZE) {
        set_error(error, "kv artifact buffer too small for header");
        return false;
    }

    kv_artifact_header_v1_base base = {};
    std::memcpy(&base, data, HEADER_BASE_SIZE);

    if (!std::equal(std::begin(base.magic), std::end(base.magic), LLAMA_KV_ARTIFACT_MAGIC.begin())) {
        set_error(error, "invalid kv artifact magic in memory buffer");
        return false;
    }

    size_t hdr_size = HEADER_BASE_SIZE;
    kv_artifact_header_v1_ext ext = {};
    ext.last_token_id = -1;
    if (base.format_minor >= 1) {
        if (size < HEADER_V1_1_SIZE) {
            set_error(error, "kv artifact buffer too small for v1.1 header");
            return false;
        }
        std::memcpy(&ext, data + HEADER_BASE_SIZE, HEADER_EXT_SIZE);
        hdr_size = HEADER_V1_1_SIZE;
    }

    const size_t payload_bytes = static_cast<size_t>(base.payload_bytes);
    if (size < hdr_size + payload_bytes) {
        set_error(error, "kv artifact buffer too small for payload");
        return false;
    }

    const uint8_t * payload_ptr = data + hdr_size;
    payload_out.assign(payload_ptr, payload_ptr + payload_bytes);

    const uint32_t crc = payload_bytes > 0 ? llama_kv_artifact_crc32(payload_out.data(), payload_out.size()) : 0u;
    if (crc != base.payload_crc32) {
        set_error(error, "kv artifact crc mismatch: expected " + std::to_string(base.payload_crc32) +
                             " got " + std::to_string(crc));
        return false;
    }

    if (meta_out) {
        meta_out->format_major      = base.format_major;
        meta_out->format_minor      = base.format_minor;
        meta_out->n_layers          = base.n_layers;
        meta_out->n_ctx             = base.n_ctx;
        meta_out->token_count       = base.token_count;
        meta_out->type_k            = base.type_k;
        meta_out->type_v            = base.type_v;
        meta_out->flags             = base.flags;
        meta_out->last_token_id     = ext.last_token_id;
        meta_out->v_trans           = ext.v_trans;
        meta_out->is_mla            = ext.is_mla;
        meta_out->reserved_0        = ext.reserved_0;
        meta_out->model_fingerprint = ext.model_fingerprint;
    }
    if (summary_out) {
        summary_out->payload_bytes = base.payload_bytes;
        summary_out->payload_crc32 = base.payload_crc32;
    }
    return true;
}
