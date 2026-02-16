//
// KV Bridge - Main Compatibility Implementation
// Copyright (C) 2025 Iwan Kawrakow / ik_llama contributors
// MIT license
// SPDX-License-Identifier: MIT
//

#include "ik-kv-compat.h"
#include "ik-kv-compat-types.h"

#include "llama.h"
#include <random>
#include "llama-context.h"
#include "llama-model.h"

#include <string.h>
#include <stdio.h>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>

namespace {

namespace fs = std::filesystem;

constexpr size_t IK_KV_FINGERPRINT_SIZE = 32;
constexpr std::array<char, 8> RTX_KVARTIF1_MAGIC = { 'K', 'V', 'A', 'R', 'T', 'I', 'F', '1' };
constexpr std::array<char, 8> IK_PLAN_CACHE_MAGIC = { 'I', 'K', 'K', 'V', 'C', 'A', 'C', '1' };
constexpr uint16_t IK_PLAN_CACHE_VERSION_MAJOR = 1;
constexpr uint16_t IK_PLAN_CACHE_VERSION_MINOR = 0;
constexpr size_t IK_PLAN_CACHE_HEADER_SIZE = 8 + 2 + 2 + 4 + 4 + 4 + IK_KV_PLAN_KEY_SIZE;

constexpr size_t RTX_HDR_OFF_FORMAT_MAJOR = 8;
constexpr size_t RTX_HDR_OFF_FORMAT_MINOR = 10;
constexpr size_t RTX_HDR_OFF_N_LAYERS     = 12;
constexpr size_t RTX_HDR_OFF_N_CTX        = 16;
constexpr size_t RTX_HDR_OFF_TOKEN_COUNT  = 20;
constexpr size_t RTX_HDR_OFF_TYPE_K       = 24;
constexpr size_t RTX_HDR_OFF_TYPE_V       = 26;
constexpr size_t RTX_HDR_OFF_FLAGS        = 28;
constexpr size_t RTX_HDR_OFF_PAYLOAD_SIZE = 32;
constexpr size_t RTX_HDR_OFF_PAYLOAD_CRC  = 40;

constexpr size_t RTX_HDR_FIELDS_SIZE      = 44; // Logical field size.
constexpr size_t RTX_HDR_PADDED_SIZE      = 48; // Typical C struct serialization size.

constexpr size_t HDR_RESERVED_TOKEN_COUNT = 0;
constexpr size_t HDR_RESERVED_SOURCE_FMT  = 4;

struct kv_artifact_view_t {
    ik_kva_header_t header = {};
    const uint8_t * payload = nullptr;
    size_t payload_size = 0;
};

static inline uint16_t rd_u16_le(const uint8_t * p) {
    return (uint16_t) (p[0] | ((uint16_t) p[1] << 8));
}

static inline uint32_t rd_u32_le(const uint8_t * p) {
    return (uint32_t) (p[0] |
                       ((uint32_t) p[1] << 8) |
                       ((uint32_t) p[2] << 16) |
                       ((uint32_t) p[3] << 24));
}

static inline uint64_t rd_u64_le(const uint8_t * p) {
    return (uint64_t) rd_u32_le(p) | ((uint64_t) rd_u32_le(p + 4) << 32);
}

static inline void wr_u32_le(uint8_t * p, uint32_t v) {
    p[0] = (uint8_t) (v & 0xFFu);
    p[1] = (uint8_t) ((v >> 8) & 0xFFu);
    p[2] = (uint8_t) ((v >> 16) & 0xFFu);
    p[3] = (uint8_t) ((v >> 24) & 0xFFu);
}

static inline void append_u8(std::vector<uint8_t> & out, uint8_t v) {
    out.push_back(v);
}

static inline void append_u16(std::vector<uint8_t> & out, uint16_t v) {
    out.push_back((uint8_t) (v & 0xFFu));
    out.push_back((uint8_t) ((v >> 8) & 0xFFu));
}

static inline void append_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back((uint8_t) ((v >> (8 * i)) & 0xFFu));
    }
}

static bool is_all_zero_fingerprint(const uint8_t fp[IK_KV_FINGERPRINT_SIZE]) {
    for (size_t i = 0; i < IK_KV_FINGERPRINT_SIZE; ++i) {
        if (fp[i] != 0) {
            return false;
        }
    }
    return true;
}

static uint64_t fnv1a64(const uint8_t * data, size_t size) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t fnv1a64_seeded(uint64_t seed, const uint8_t * data, size_t size) {
    uint64_t h = seed;
    for (size_t i = 0; i < size; ++i) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void append_u32(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t) (v & 0xFFu));
    out.push_back((uint8_t) ((v >> 8) & 0xFFu));
    out.push_back((uint8_t) ((v >> 16) & 0xFFu));
    out.push_back((uint8_t) ((v >> 24) & 0xFFu));
}

static void append_bytes(std::vector<uint8_t> & out, const uint8_t * data, size_t size) {
    if (!data || size == 0) {
        return;
    }
    out.insert(out.end(), data, data + size);
}

static uint32_t crc32_ieee(const uint8_t * data, size_t size) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1u)) : (c >> 1u);
            }
            table[i] = c;
        }
        init = true;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i) {
        crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8u);
    }
    return crc ^ 0xFFFFFFFFu;
}

static uint32_t saturating_u32(uint64_t v) {
    return (uint32_t) std::min<uint64_t>(v, std::numeric_limits<uint32_t>::max());
}

static std::string key_to_hex(const ik_kv_compat_plan_key_t & key) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.resize(IK_KV_PLAN_KEY_SIZE * 2);
    for (size_t i = 0; i < IK_KV_PLAN_KEY_SIZE; ++i) {
        out[2 * i + 0] = hex[(key.data[i] >> 4) & 0x0Fu];
        out[2 * i + 1] = hex[key.data[i] & 0x0Fu];
    }
    return out;
}

static std::string get_plan_cache_dir();
static fs::path plan_cache_path_for_key(const ik_kv_compat_plan_key_t & key);

static bool load_file_binary(const fs::path & path, std::vector<uint8_t> & out) {
    out.clear();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff end = ifs.tellg();
    if (end < 0) {
        return false;
    }
    ifs.seekg(0, std::ios::beg);
    out.resize((size_t) end);
    if (!out.empty()) {
        ifs.read((char *) out.data(), (std::streamsize) out.size());
        if (!ifs.good() && !ifs.eof()) {
            out.clear();
            return false;
        }
    }
    return true;
}

static bool write_file_binary_atomic(const fs::path & path, const uint8_t * data, size_t size) {
    if (!data || size == 0) {
        return false;
    }

    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            return false;
        }
        ofs.write((const char *) data, (std::streamsize) size);
        if (!ofs.good()) {
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

static bool map_ggml_type_to_ik(uint16_t ggml_type, uint8_t * out_type) {
    if (!out_type) {
        return false;
    }
    switch (ggml_type) {
        case GGML_TYPE_F32:    *out_type = IK_KV_TYPE_F32;    return true;
        case GGML_TYPE_F16:    *out_type = IK_KV_TYPE_F16;    return true;
        case GGML_TYPE_BF16:   *out_type = IK_KV_TYPE_BF16;   return true;
        case GGML_TYPE_Q4_0:   *out_type = IK_KV_TYPE_Q4_0;   return true;
        case GGML_TYPE_Q4_1:   *out_type = IK_KV_TYPE_Q4_1;   return true;
        case GGML_TYPE_Q5_0:   *out_type = IK_KV_TYPE_Q5_0;   return true;
        case GGML_TYPE_Q5_1:   *out_type = IK_KV_TYPE_Q5_1;   return true;
        case GGML_TYPE_Q8_0:   *out_type = IK_KV_TYPE_Q8_0;   return true;
        case GGML_TYPE_Q8_1:   *out_type = IK_KV_TYPE_Q8_1;   return true;
        case GGML_TYPE_IQ4_NL: *out_type = IK_KV_TYPE_IQ4_NL; return true;
        default: return false;
    }
}

static bool parse_artifact_view(
        const uint8_t * artifact_data,
        size_t artifact_size,
        kv_artifact_view_t * view_out,
        ik_kv_compat_reject_reason_t * reject_reason) {
    if (!artifact_data || !view_out) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return false;
    }

    memset(view_out, 0, sizeof(*view_out));

    // RTX artifact: "KVARTIF1"
    if (artifact_size >= RTX_HDR_FIELDS_SIZE &&
            memcmp(artifact_data, RTX_KVARTIF1_MAGIC.data(), RTX_KVARTIF1_MAGIC.size()) == 0) {
        const uint16_t format_major = rd_u16_le(artifact_data + RTX_HDR_OFF_FORMAT_MAJOR);
        const uint16_t format_minor = rd_u16_le(artifact_data + RTX_HDR_OFF_FORMAT_MINOR);
        const uint32_t n_layers     = rd_u32_le(artifact_data + RTX_HDR_OFF_N_LAYERS);
        const uint32_t n_ctx        = rd_u32_le(artifact_data + RTX_HDR_OFF_N_CTX);
        const uint32_t token_count  = rd_u32_le(artifact_data + RTX_HDR_OFF_TOKEN_COUNT);
        const uint16_t type_k_ggml  = rd_u16_le(artifact_data + RTX_HDR_OFF_TYPE_K);
        const uint16_t type_v_ggml  = rd_u16_le(artifact_data + RTX_HDR_OFF_TYPE_V);
        GGML_UNUSED(rd_u32_le(artifact_data + RTX_HDR_OFF_FLAGS));
        const uint64_t payload_size = rd_u64_le(artifact_data + RTX_HDR_OFF_PAYLOAD_SIZE);
        const uint32_t payload_crc  = rd_u32_le(artifact_data + RTX_HDR_OFF_PAYLOAD_CRC);

        uint8_t type_k = 0;
        uint8_t type_v = 0;
        if (!map_ggml_type_to_ik(type_k_ggml, &type_k) || !map_ggml_type_to_ik(type_v_ggml, &type_v)) {
            if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_INCOMPATIBLE_PROFILE;
            return false;
        }

        size_t payload_offset = 0;
        if (artifact_size >= RTX_HDR_FIELDS_SIZE && artifact_size - RTX_HDR_FIELDS_SIZE == payload_size) {
            payload_offset = RTX_HDR_FIELDS_SIZE;
        } else if (artifact_size >= RTX_HDR_PADDED_SIZE && artifact_size - RTX_HDR_PADDED_SIZE == payload_size) {
            payload_offset = RTX_HDR_PADDED_SIZE;
        } else {
            if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
            return false;
        }

        view_out->payload      = artifact_data + payload_offset;
        view_out->payload_size = (size_t) payload_size;

        auto & hdr = view_out->header;
        memset(&hdr, 0, sizeof(hdr));
        hdr.magic        = IK_KVA_MAGIC;
        hdr.format_major = format_major;
        hdr.format_minor = format_minor;
        hdr.n_layers     = n_layers;
        hdr.n_ctx        = n_ctx;
        hdr.n_head_kv    = 0;
        hdr.type_k       = type_k;
        hdr.type_v       = type_v;
        hdr.v_trans      = 0xFFu; // Unknown until payload parse.
        hdr.n_stream     = 0;
        hdr.payload_size = payload_size;
        hdr.payload_crc  = payload_crc;
        wr_u32_le(hdr.reserved + HDR_RESERVED_TOKEN_COUNT, token_count);
        hdr.reserved[HDR_RESERVED_SOURCE_FMT] = (uint8_t) IK_KV_SOURCE_FORMAT_RTX_KVARTIF1;

        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return true;
    }

    // ik artifact with inline ik_kva_header_t.
    if (artifact_size < sizeof(ik_kva_header_t)) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return false;
    }

    memcpy(&view_out->header, artifact_data, sizeof(ik_kva_header_t));

    if (view_out->header.magic != IK_KVA_MAGIC) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_MAGIC_MISMATCH;
        return false;
    }

    if (view_out->header.format_major > IK_KVA_FORMAT_MAJOR) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_VERSION_UNSUPPORTED;
        return false;
    }

    const size_t payload_offset = sizeof(ik_kva_header_t);
    const uint64_t payload_size = view_out->header.payload_size;
    if (artifact_size < payload_offset) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return false;
    }

    view_out->payload      = artifact_data + payload_offset;
    view_out->payload_size = std::min<size_t>((size_t) payload_size, artifact_size - payload_offset);

    // Preserve producer metadata but ensure source format marker is set.
    if (view_out->header.reserved[HDR_RESERVED_SOURCE_FMT] == 0) {
        view_out->header.reserved[HDR_RESERVED_SOURCE_FMT] = (uint8_t) IK_KV_SOURCE_FORMAT_IK_KVA;
    }

    if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    return true;
}

static bool parse_rtx_payload_shape(
        const uint8_t * payload,
        size_t payload_size,
        uint32_t * n_stream_out,
        uint8_t * v_trans_out) {
    if (!payload || payload_size < 4 || !n_stream_out || !v_trans_out) {
        return false;
    }

    auto mul_size_overflow = [](size_t a, size_t b, size_t * out) -> bool {
        if (!out) {
            return true;
        }
        if (a != 0 && b > SIZE_MAX / a) {
            return true;
        }
        *out = a * b;
        return false;
    };

    struct rtx_stream_view_t {
        uint32_t cell_count = 0;
        size_t stream_end = 0;
        uint8_t v_trans = 0xFFu;
    };

    auto parse_stream = [&](size_t stream_off, rtx_stream_view_t * out) -> bool {
        if (!out || stream_off + sizeof(uint32_t) > payload_size) {
            return false;
        }

        size_t off = stream_off;
        const uint32_t cell_count = rd_u32_le(payload + off);
        off += sizeof(uint32_t);

        for (uint32_t i = 0; i < cell_count; ++i) {
            if (off + sizeof(llama_pos) + sizeof(uint32_t) > payload_size) {
                return false;
            }
            off += sizeof(llama_pos);
            const uint32_t n_seq_id = rd_u32_le(payload + off);
            off += sizeof(uint32_t);

            size_t seq_bytes = 0;
            if (mul_size_overflow((size_t) n_seq_id, sizeof(llama_seq_id), &seq_bytes)) {
                return false;
            }
            if (off + seq_bytes > payload_size) {
                return false;
            }
            off += seq_bytes;
        }

        if (cell_count == 0) {
            out->cell_count = 0;
            out->stream_end = off;
            out->v_trans = 0xFFu;
            return true;
        }

        if (off + 8 > payload_size) {
            return false;
        }

        const uint32_t v_state = rd_u32_le(payload + off);
        off += sizeof(uint32_t);
        const uint32_t n_layer = rd_u32_le(payload + off);
        off += sizeof(uint32_t);

        uint8_t v_trans = 0xFFu;
        if (v_state == 0) {
            v_trans = 0;
        } else if (v_state == 1) {
            v_trans = 1;
        } else if (v_state != 2) {
            return false;
        }

        for (uint32_t il = 0; il < n_layer; ++il) {
            if (off + sizeof(uint32_t) + sizeof(uint64_t) > payload_size) {
                return false;
            }
            off += sizeof(uint32_t); // type
            const uint64_t row_size = rd_u64_le(payload + off);
            off += sizeof(uint64_t);

            size_t bytes = 0;
            if (mul_size_overflow((size_t) cell_count, (size_t) row_size, &bytes)) {
                return false;
            }
            if (off + bytes > payload_size) {
                return false;
            }
            off += bytes;
        }

        if (v_state == 0) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                if (off + sizeof(uint32_t) + sizeof(uint64_t) > payload_size) {
                    return false;
                }
                off += sizeof(uint32_t); // type
                const uint64_t row_size = rd_u64_le(payload + off);
                off += sizeof(uint64_t);

                size_t bytes = 0;
                if (mul_size_overflow((size_t) cell_count, (size_t) row_size, &bytes)) {
                    return false;
                }
                if (off + bytes > payload_size) {
                    return false;
                }
                off += bytes;
            }
        } else if (v_state == 1) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                if (off + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t) > payload_size) {
                    return false;
                }
                off += sizeof(uint32_t); // type
                const uint32_t v_size_el = rd_u32_le(payload + off);
                off += sizeof(uint32_t);
                const uint32_t n_embd_v_gqa = rd_u32_le(payload + off);
                off += sizeof(uint32_t);

                size_t bytes = 0;
                size_t per_row = 0;
                if (mul_size_overflow((size_t) cell_count, (size_t) v_size_el, &per_row) ||
                        mul_size_overflow(per_row, (size_t) n_embd_v_gqa, &bytes)) {
                    return false;
                }
                if (off + bytes > payload_size) {
                    return false;
                }
                off += bytes;
            }
        }

        out->cell_count = cell_count;
        out->stream_end = off;
        out->v_trans = v_trans;
        return true;
    };

    const uint32_t n_stream_raw = rd_u32_le(payload);
    if (n_stream_raw == 0) {
        return false;
    }

    size_t off = sizeof(uint32_t);
    uint32_t non_empty_streams = 0;
    uint8_t v_trans = 0xFFu;

    for (uint32_t s = 0; s < n_stream_raw; ++s) {
        rtx_stream_view_t stream = {};
        if (!parse_stream(off, &stream)) {
            return false;
        }
        if (stream.cell_count > 0) {
            ++non_empty_streams;
            if (v_trans == 0xFFu) {
                v_trans = stream.v_trans;
            } else if (v_trans <= 1 && stream.v_trans <= 1 && v_trans != stream.v_trans) {
                // Mixed V transpose modes across streams; treat as unknown for strict checks.
                v_trans = 0xFFu;
            }
        }
        off = stream.stream_end;
    }

    if (off != payload_size) {
        return false;
    }

    // Bridge strict-v1 supports effectively single-stream data.
    *n_stream_out = non_empty_streams <= 1 ? 1u : n_stream_raw;
    *v_trans_out = v_trans;
    return true;
}

static ik_kv_compat_convert_result_t convert_rtx_stream_block_to_ik(
        const uint8_t * src,
        size_t src_size,
        uint8_t * dst,
        size_t dst_size,
        size_t * written_out,
        ik_kv_compat_reject_reason_t * reject_reason) {
    if (!src || !dst || !written_out) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    *written_out = 0;

    if (src_size < sizeof(uint32_t)) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
    }

    size_t src_off = 0;
    const uint32_t cell_count = rd_u32_le(src + src_off);
    src_off += sizeof(uint32_t);

    size_t dst_off = 0;
    if (dst_size < sizeof(uint32_t)) {
        return IK_KV_COMPAT_CONVERT_ERR_SIZE;
    }
    memcpy(dst + dst_off, &cell_count, sizeof(cell_count));
    dst_off += sizeof(cell_count);

    for (uint32_t i = 0; i < cell_count; ++i) {
        if (src_off + sizeof(llama_pos) + sizeof(uint32_t) > src_size) {
            if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
            return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
        }

        const llama_pos pos = (llama_pos) rd_u32_le(src + src_off);
        src_off += sizeof(llama_pos);

        const uint32_t n_seq_id = rd_u32_le(src + src_off);
        src_off += sizeof(uint32_t);

        const size_t seq_bytes = (size_t) n_seq_id * sizeof(llama_seq_id);
        if (src_off + seq_bytes > src_size) {
            if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
            return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
        }

        if (dst_off + sizeof(llama_pos) + sizeof(uint32_t) > dst_size) {
            return IK_KV_COMPAT_CONVERT_ERR_SIZE;
        }

        memcpy(dst + dst_off, &pos, sizeof(pos));
        dst_off += sizeof(pos);

        const uint32_t dst_n_seq_id = 0;
        memcpy(dst + dst_off, &dst_n_seq_id, sizeof(dst_n_seq_id));
        dst_off += sizeof(dst_n_seq_id);

        src_off += seq_bytes;
    }

    const size_t trailing = src_size - src_off;
    if (dst_off + trailing > dst_size) {
        return IK_KV_COMPAT_CONVERT_ERR_SIZE;
    }

    memcpy(dst + dst_off, src + src_off, trailing);
    dst_off += trailing;

    *written_out = dst_off;
    if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    return IK_KV_COMPAT_CONVERT_OK;
}

static ik_kv_compat_convert_result_t convert_rtx_seq_blob_to_ik(
        const uint8_t * src,
        size_t src_size,
        uint8_t * dst,
        size_t dst_size,
        size_t * written_out,
        ik_kv_compat_reject_reason_t * reject_reason) {
    if (!src || !dst || !written_out) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    *written_out = 0;

    if (src_size < sizeof(uint32_t)) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
    }

    auto mul_size_overflow = [](size_t a, size_t b, size_t * out) -> bool {
        if (!out) {
            return true;
        }
        if (a != 0 && b > SIZE_MAX / a) {
            return true;
        }
        *out = a * b;
        return false;
    };

    struct rtx_stream_extent_t {
        uint32_t cell_count = 0;
        size_t begin = 0;
        size_t end = 0;
    };

    auto parse_stream_extent = [&](size_t begin_off, rtx_stream_extent_t * out) -> bool {
        if (!out || begin_off + sizeof(uint32_t) > src_size) {
            return false;
        }

        size_t off = begin_off;
        const uint32_t cell_count = rd_u32_le(src + off);
        off += sizeof(uint32_t);

        for (uint32_t i = 0; i < cell_count; ++i) {
            if (off + sizeof(llama_pos) + sizeof(uint32_t) > src_size) {
                return false;
            }
            off += sizeof(llama_pos);

            const uint32_t n_seq_id = rd_u32_le(src + off);
            off += sizeof(uint32_t);

            size_t seq_bytes = 0;
            if (mul_size_overflow((size_t) n_seq_id, sizeof(llama_seq_id), &seq_bytes)) {
                return false;
            }
            if (off + seq_bytes > src_size) {
                return false;
            }
            off += seq_bytes;
        }

        if (cell_count > 0) {
            if (off + sizeof(uint32_t) + sizeof(uint32_t) > src_size) {
                return false;
            }

            const uint32_t v_state = rd_u32_le(src + off);
            off += sizeof(uint32_t);
            const uint32_t n_layer = rd_u32_le(src + off);
            off += sizeof(uint32_t);

            for (uint32_t il = 0; il < n_layer; ++il) {
                if (off + sizeof(uint32_t) + sizeof(uint64_t) > src_size) {
                    return false;
                }
                off += sizeof(uint32_t); // type
                const uint64_t row_size = rd_u64_le(src + off);
                off += sizeof(uint64_t);

                size_t bytes = 0;
                if (mul_size_overflow((size_t) cell_count, (size_t) row_size, &bytes)) {
                    return false;
                }
                if (off + bytes > src_size) {
                    return false;
                }
                off += bytes;
            }

            if (v_state == 0) {
                for (uint32_t il = 0; il < n_layer; ++il) {
                    if (off + sizeof(uint32_t) + sizeof(uint64_t) > src_size) {
                        return false;
                    }
                    off += sizeof(uint32_t); // type
                    const uint64_t row_size = rd_u64_le(src + off);
                    off += sizeof(uint64_t);

                    size_t bytes = 0;
                    if (mul_size_overflow((size_t) cell_count, (size_t) row_size, &bytes)) {
                        return false;
                    }
                    if (off + bytes > src_size) {
                        return false;
                    }
                    off += bytes;
                }
            } else if (v_state == 1) {
                for (uint32_t il = 0; il < n_layer; ++il) {
                    if (off + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t) > src_size) {
                        return false;
                    }
                    off += sizeof(uint32_t); // type
                    const uint32_t v_size_el = rd_u32_le(src + off);
                    off += sizeof(uint32_t);
                    const uint32_t n_embd_v_gqa = rd_u32_le(src + off);
                    off += sizeof(uint32_t);

                    size_t bytes = 0;
                    size_t per_row = 0;
                    if (mul_size_overflow((size_t) cell_count, (size_t) v_size_el, &per_row) ||
                            mul_size_overflow(per_row, (size_t) n_embd_v_gqa, &bytes)) {
                        return false;
                    }
                    if (off + bytes > src_size) {
                        return false;
                    }
                    off += bytes;
                }
            } else if (v_state != 2) {
                return false;
            }
        }

        out->cell_count = cell_count;
        out->begin = begin_off;
        out->end = off;
        return true;
    };

    const uint32_t n_stream = rd_u32_le(src);
    if (n_stream == 0) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
    }

    size_t off = sizeof(uint32_t);
    rtx_stream_extent_t selected = {};
    bool has_selected = false;
    uint32_t non_empty_count = 0;

    for (uint32_t s = 0; s < n_stream; ++s) {
        rtx_stream_extent_t cur = {};
        if (!parse_stream_extent(off, &cur)) {
            if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
            return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
        }
        if (cur.cell_count > 0) {
            ++non_empty_count;
            if (!has_selected) {
                selected = cur;
                has_selected = true;
            }
        }
        off = cur.end;
    }

    if (off != src_size) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
    }

    if (non_empty_count == 0 || !has_selected) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_PARTIAL_UNSUPPORTED;
        return IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH;
    }
    if (non_empty_count > 1) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_N_STREAM_UNSUPPORTED;
        return IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH;
    }

    return convert_rtx_stream_block_to_ik(
        src + selected.begin,
        selected.end - selected.begin,
        dst,
        dst_size,
        written_out,
        reject_reason);
}

struct ik_stream_layer_layout_t {
    size_t k_offset = 0;
    size_t k_size = 0;
    size_t v_offset = 0;
    size_t v_size = 0;
    uint32_t k_row_size = 0;
    uint32_t v_row_size = 0;
};

struct ik_stream_layout_t {
    uint32_t n_layers = 0;
    uint32_t cell_count = 0;
    uint8_t v_trans = 0xFFu;
    size_t total_size = 0;
    std::vector<ik_stream_layer_layout_t> layers;
};

static bool mul_size_overflow(size_t a, size_t b, size_t * out) {
    if (!out) {
        return true;
    }
    if (a != 0 && b > SIZE_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static bool parse_ik_stream_layout(
        const uint8_t * payload,
        size_t payload_size,
        ik_stream_layout_t * layout_out) {
    if (!payload || !layout_out || payload_size < sizeof(uint32_t)) {
        return false;
    }

    *layout_out = {};

    size_t off = 0;
    const uint32_t cell_count = rd_u32_le(payload + off);
    off += sizeof(uint32_t);

    for (uint32_t i = 0; i < cell_count; ++i) {
        if (off + sizeof(llama_pos) + sizeof(uint32_t) > payload_size) {
            return false;
        }
        off += sizeof(llama_pos);

        const uint32_t n_seq_id = rd_u32_le(payload + off);
        off += sizeof(uint32_t);

        size_t seq_bytes = 0;
        if (mul_size_overflow((size_t) n_seq_id, sizeof(llama_seq_id), &seq_bytes)) {
            return false;
        }
        if (off + seq_bytes > payload_size) {
            return false;
        }
        off += seq_bytes;
    }

    layout_out->cell_count = cell_count;
    if (cell_count == 0) {
        if (off != payload_size) {
            return false;
        }
        layout_out->n_layers = 0;
        layout_out->v_trans = 0xFFu;
        layout_out->total_size = off;
        return true;
    }

    if (off + sizeof(uint32_t) + sizeof(uint32_t) > payload_size) {
        return false;
    }
    const uint32_t v_state = rd_u32_le(payload + off);
    off += sizeof(uint32_t);
    const uint32_t n_layer = rd_u32_le(payload + off);
    off += sizeof(uint32_t);
    if (n_layer > IK_KV_MAX_LAYERS) {
        return false;
    }

    layout_out->layers.assign(n_layer, ik_stream_layer_layout_t{});
    for (uint32_t il = 0; il < n_layer; ++il) {
        ik_stream_layer_layout_t & layer = layout_out->layers[il];
        layer.k_offset = off;

        if (off + sizeof(uint32_t) + sizeof(uint64_t) > payload_size) {
            return false;
        }
        off += sizeof(uint32_t); // type
        const uint64_t row_size = rd_u64_le(payload + off);
        off += sizeof(uint64_t);

        size_t bytes = 0;
        if (mul_size_overflow((size_t) cell_count, (size_t) row_size, &bytes)) {
            return false;
        }
        if (off + bytes > payload_size) {
            return false;
        }
        off += bytes;

        layer.k_size = sizeof(uint32_t) + sizeof(uint64_t) + bytes;
        layer.k_row_size = saturating_u32(row_size);
    }

    if (v_state == 0) {
        layout_out->v_trans = 0;
        for (uint32_t il = 0; il < n_layer; ++il) {
            ik_stream_layer_layout_t & layer = layout_out->layers[il];
            layer.v_offset = off;

            if (off + sizeof(uint32_t) + sizeof(uint64_t) > payload_size) {
                return false;
            }
            off += sizeof(uint32_t); // type
            const uint64_t row_size = rd_u64_le(payload + off);
            off += sizeof(uint64_t);

            size_t bytes = 0;
            if (mul_size_overflow((size_t) cell_count, (size_t) row_size, &bytes)) {
                return false;
            }
            if (off + bytes > payload_size) {
                return false;
            }
            off += bytes;

            layer.v_size = sizeof(uint32_t) + sizeof(uint64_t) + bytes;
            layer.v_row_size = saturating_u32(row_size);
        }
    } else if (v_state == 1) {
        layout_out->v_trans = 1;
        for (uint32_t il = 0; il < n_layer; ++il) {
            ik_stream_layer_layout_t & layer = layout_out->layers[il];
            layer.v_offset = off;

            if (off + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t) > payload_size) {
                return false;
            }
            off += sizeof(uint32_t); // type
            const uint32_t v_size_el = rd_u32_le(payload + off);
            off += sizeof(uint32_t);
            const uint32_t n_embd_v_gqa = rd_u32_le(payload + off);
            off += sizeof(uint32_t);

            size_t per_row = 0;
            size_t bytes = 0;
            if (mul_size_overflow((size_t) v_size_el, (size_t) n_embd_v_gqa, &per_row) ||
                    mul_size_overflow((size_t) cell_count, per_row, &bytes)) {
                return false;
            }
            if (off + bytes > payload_size) {
                return false;
            }
            off += bytes;

            layer.v_size = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint32_t) + bytes;
            layer.v_row_size = saturating_u32(per_row);
        }
    } else if (v_state == 2) {
        layout_out->v_trans = 0xFFu;
    } else {
        return false;
    }

    if (off != payload_size) {
        return false;
    }

    layout_out->n_layers = n_layer;
    layout_out->total_size = off;
    return true;
}

static void populate_plan_mappings_from_layout(
        const ik_stream_layout_t & layout,
        ik_kv_compat_plan_t * plan_out,
        uint8_t needs_v_trans) {
    if (!plan_out) {
        return;
    }

    if (plan_out->n_layers == 0) {
        plan_out->n_layers = std::min<uint32_t>((uint32_t) layout.layers.size(), IK_KV_MAX_LAYERS);
    }
    const uint32_t n_layers = std::min<uint32_t>(plan_out->n_layers, (uint32_t) layout.layers.size());

    for (uint32_t il = 0; il < n_layers; ++il) {
        const ik_stream_layer_layout_t & src = layout.layers[il];
        ik_kv_layer_mapping_t & dst = plan_out->layer_mappings[il];
        memset(&dst, 0, sizeof(dst));
        dst.layer_idx = il;
        dst.k_row_size = saturating_u32(src.k_size);
        dst.v_row_size = saturating_u32(src.v_size);
        dst.k_offset = saturating_u32(src.k_offset);
        dst.v_offset = saturating_u32(src.v_offset);
        dst.dst_k_offset = saturating_u32(src.k_offset);
        dst.dst_v_offset = saturating_u32(src.v_offset);
        dst.needs_scatter = 0;
        dst.needs_v_trans = needs_v_trans;
    }
}

static bool analyze_source_payload_layout(
        const ik_kv_source_descriptor_t * src,
        ik_stream_layout_t * layout_out,
        size_t * dst_size_out,
        ik_kv_compat_reject_reason_t * reject_reason) {
    if (!src || !layout_out || !dst_size_out || !src->payload) {
        return false;
    }

    *layout_out = {};
    *dst_size_out = 0;
    if (reject_reason) {
        *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    }

    if (src->source_format == IK_KV_SOURCE_FORMAT_IK_KVA) {
        if (!parse_ik_stream_layout(src->payload, (size_t) src->payload_size, layout_out)) {
            if (reject_reason) {
                *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
            }
            return false;
        }
        *dst_size_out = layout_out->total_size;
        return true;
    }

    if (src->source_format == IK_KV_SOURCE_FORMAT_RTX_KVARTIF1) {
        std::vector<uint8_t> converted((size_t) src->payload_size);
        size_t bytes_written = 0;
        ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
        const ik_kv_compat_convert_result_t rc = convert_rtx_seq_blob_to_ik(
            src->payload,
            (size_t) src->payload_size,
            converted.data(),
            converted.size(),
            &bytes_written,
            &reject);
        if (rc != IK_KV_COMPAT_CONVERT_OK) {
            if (reject_reason) {
                *reject_reason = reject;
            }
            return false;
        }
        if (!parse_ik_stream_layout(converted.data(), bytes_written, layout_out)) {
            if (reject_reason) {
                *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
            }
            return false;
        }
        *dst_size_out = bytes_written;
        return true;
    }

    return false;
}

} // namespace

//
// Internal state
//

static struct {
    bool initialized;
    ik_kv_bridge_config_t config;
    ik_kv_bridge_metrics_t last_metrics;
    bool telemetry_enabled;
} g_kv_bridge = {};

namespace {

static const char * bridge_mode_name(ik_kv_bridge_mode_t mode) {
    switch (mode) {
        case IK_KV_BRIDGE_MODE_OFF: return "off";
        case IK_KV_BRIDGE_MODE_RELAXED: return "relaxed";
        case IK_KV_BRIDGE_MODE_STRICT:
        default: return "strict";
    }
}

static std::string get_plan_cache_dir() {
    if (g_kv_bridge.config.plan_cache_dir && g_kv_bridge.config.plan_cache_dir[0] != '\0') {
        return std::string(g_kv_bridge.config.plan_cache_dir);
    }
    const char * env_dir = std::getenv("IK_KV_BRIDGE_PLAN_CACHE_DIR");
    if (env_dir && env_dir[0] != '\0') {
        return std::string(env_dir);
    }
#if defined(_WIN32)
    const char * home = std::getenv("USERPROFILE");
#else
    const char * home = std::getenv("HOME");
#endif
    if (home && home[0] != '\0') {
        return (fs::path(home) / ".ik_llama_kv_plan_cache").string();
    }
    return std::string(".ik_llama_kv_plan_cache");
}

static fs::path plan_cache_path_for_key(const ik_kv_compat_plan_key_t & key) {
    const std::string cache_dir = get_plan_cache_dir();
    if (cache_dir.empty()) {
        return fs::path();
    }
    return fs::path(cache_dir) / ("plan_" + key_to_hex(key) + ".ikvc");
}

static void bridge_log_metrics(
        const ik_kv_bridge_metrics_t & metrics,
        ik_kv_compat_convert_result_t rc,
        ik_kv_compat_reject_reason_t reject,
        bool fallback_used) {
    if (!g_kv_bridge.telemetry_enabled) {
        return;
    }
    fprintf(stderr,
            "ik-kv-bridge: mode=%s rc=%s reject=%s status=%u cache_hit=%s "
            "bytes_in=%llu bytes_out=%llu convert_us=%llu plan_key=%016llx fallback=%s\n",
            bridge_mode_name((ik_kv_bridge_mode_t) metrics.mode),
            ik_kv_compat_result_str(rc),
            ik_kv_compat_reject_str(reject),
            (unsigned) metrics.status,
            metrics.plan_cache_hit ? "true" : "false",
            (unsigned long long) metrics.bytes_in,
            (unsigned long long) metrics.bytes_out,
            (unsigned long long) metrics.convert_us,
            (unsigned long long) metrics.plan_key,
            fallback_used ? "true" : "false");
}

} // namespace

//
// Initialization and configuration
//

bool ik_kv_bridge_init(void) {
    if (g_kv_bridge.initialized) {
        return true;  // Already initialized
    }
    
    memset(&g_kv_bridge, 0, sizeof(g_kv_bridge));
    g_kv_bridge.initialized = true;
    g_kv_bridge.config.mode = IK_KV_BRIDGE_MODE_STRICT;  // Default to strict
    g_kv_bridge.telemetry_enabled = true;
    
    return true;
}

void ik_kv_bridge_shutdown(void) {
    if (!g_kv_bridge.initialized) {
        return;
    }
    
    // Clear any cached plans
    ik_kv_plan_cache_clear();
    
    memset(&g_kv_bridge, 0, sizeof(g_kv_bridge));
}

void ik_kv_bridge_set_config(const ik_kv_bridge_config_t * config) {
    if (!config) return;
    g_kv_bridge.config = *config;
}

void ik_kv_bridge_get_config(ik_kv_bridge_config_t * config) {
    if (!config) return;
    *config = g_kv_bridge.config;
}

//
// Source parsing (KVB-002)
//

ik_kv_compat_convert_result_t ik_kv_source_parse_kva_header(
    const uint8_t * artifact_data,
    size_t artifact_size,
    ik_kva_header_t * header_out,
    ik_kv_compat_reject_reason_t * reject_reason
) {
    if (!artifact_data || !header_out) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }

    kv_artifact_view_t view = {};
    if (!parse_artifact_view(artifact_data, artifact_size, &view, reject_reason)) {
        return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
    }

    *header_out = view.header;

    if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    return IK_KV_COMPAT_CONVERT_OK;
}

ik_kv_compat_convert_result_t ik_kv_source_parse_prefill_seq_state(
    const ik_kva_header_t * header,
    const uint8_t * payload_data,
    size_t payload_size,
    ik_kv_source_descriptor_t * desc_out,
    ik_kv_compat_reject_reason_t * reject_reason
) {
    if (!header || !payload_data || !desc_out) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    
    // Validate payload size matches header
    if (payload_size < header->payload_size) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
        return IK_KV_COMPAT_CONVERT_ERR_SIZE;
    }

    // Build descriptor (non-owning reference to payload)
    memset(desc_out, 0, sizeof(*desc_out));
    desc_out->n_layers = header->n_layers;
    desc_out->n_ctx = header->n_ctx;
    desc_out->n_head_kv = header->n_head_kv;
    desc_out->n_embd_head = 0;
    desc_out->token_count = rd_u32_le(header->reserved + HDR_RESERVED_TOKEN_COUNT);
    desc_out->type_k = header->type_k;
    desc_out->type_v = header->type_v;
    desc_out->v_trans = header->v_trans;
    desc_out->n_stream = header->n_stream;
    desc_out->source_format = header->reserved[HDR_RESERVED_SOURCE_FMT];
    if (desc_out->source_format == IK_KV_SOURCE_FORMAT_UNKNOWN) {
        desc_out->source_format = IK_KV_SOURCE_FORMAT_IK_KVA;
    }
    desc_out->payload_size = header->payload_size;
    desc_out->payload = payload_data;
    memcpy(desc_out->model_fingerprint, header->model_fingerprint, 32);

    // For RTX artifacts, infer stream count and V transpose from payload shape.
    if (desc_out->source_format == IK_KV_SOURCE_FORMAT_RTX_KVARTIF1) {
        uint32_t n_stream = 0;
        uint8_t v_trans = 0xFFu;
        if (!parse_rtx_payload_shape(payload_data, (size_t) header->payload_size, &n_stream, &v_trans)) {
            if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_HEADER_MALFORMED;
            return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
        }
        desc_out->n_stream = (uint8_t) std::min<uint32_t>(n_stream, 255u);
        if (v_trans <= 1) {
            desc_out->v_trans = v_trans;
        }
    }

    if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    return IK_KV_COMPAT_CONVERT_OK;
}

bool ik_kv_source_validate_payload(
    const ik_kva_header_t * header,
    const uint8_t * payload_data,
    size_t payload_size
) {
    if (!header || !payload_data) {
        return false;
    }
    
    // Check size
    if (payload_size < header->payload_size) {
        return false;
    }
    
    if (header->payload_size > payload_size) {
        return false;
    }

    if (header->payload_crc == 0 && header->payload_size == 0) {
        return true;
    }

    if (header->payload_crc == 0) {
        // Older artifacts may omit checksum.
        return true;
    }

    const uint32_t crc = crc32_ieee(payload_data, (size_t) header->payload_size);
    return crc == header->payload_crc;
}

//
// Destination introspection (KVB-003)
//

ik_kv_compat_convert_result_t ik_kv_dest_introspect_from_ctx(
    struct llama_context * ctx,
    ik_kv_dest_descriptor_t * desc_out
) {
    if (!ctx || !desc_out) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }

    memset(desc_out, 0, sizeof(*desc_out));

    const llama_model & model = ctx->model;
    const llama_hparams & hp = model.hparams;

    desc_out->n_layers    = hp.n_layer;
    desc_out->n_ctx       = llama_n_ctx(ctx);
    desc_out->n_head_kv   = hp.n_head_kv(0);
    desc_out->n_embd_head = hp.n_embd_head_k;
    desc_out->n_stream    = 1;
    desc_out->v_trans     = ctx->kv_self.v_trans ? 1 : 0;

    if (!map_ggml_type_to_ik((uint16_t) ctx->kv_self.type_k, &desc_out->type_k) ||
            !map_ggml_type_to_ik((uint16_t) ctx->kv_self.type_v, &desc_out->type_v)) {
        return IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH;
    }

    std::vector<uint8_t> fp_data;
    fp_data.reserve(128);
    append_u32(fp_data, (uint32_t) model.arch);
    append_u32(fp_data, hp.n_layer);
    append_u32(fp_data, hp.n_ctx_train);
    append_u32(fp_data, hp.n_embd);
    append_u32(fp_data, hp.n_head_kv(0));
    append_u32(fp_data, hp.n_embd_head_k);
    append_u32(fp_data, hp.n_embd_head_v);
    append_u32(fp_data, (uint32_t) desc_out->type_k);
    append_u32(fp_data, (uint32_t) desc_out->type_v);
    append_u32(fp_data, (uint32_t) desc_out->v_trans);

    const uint64_t h0 = fnv1a64(fp_data.data(), fp_data.size());
    const uint64_t h1 = fnv1a64((const uint8_t *) &ctx->cparams, sizeof(ctx->cparams));

    for (int i = 0; i < 4; ++i) {
        const uint64_t hv = (i & 1) ? (h1 ^ (h0 + (uint64_t) i * 0x9E3779B97F4A7C15ull))
                                    : (h0 ^ (h1 + (uint64_t) i * 0xD6E8FEB86659FD93ull));
        memcpy(desc_out->model_fingerprint + i * 8, &hv, sizeof(hv));
    }

    return IK_KV_COMPAT_CONVERT_OK;
}

ik_kv_compat_convert_result_t ik_kv_dest_introspect_from_model(
    const char * model_path,
    ik_kv_dest_descriptor_t * desc_out
) {
    if (!model_path || !desc_out) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }

    memset(desc_out, 0, sizeof(*desc_out));
    llama_model_params model_params = llama_model_default_params();
    model_params.vocab_only = true;

    struct llama_model * model = llama_load_model_from_file(model_path, model_params);
    if (!model) {
        return IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH;
    }

    const llama_hparams & hp = model->hparams;
    desc_out->n_layers = hp.n_layer;
    desc_out->n_ctx = (uint32_t) std::max(0, llama_n_ctx_train(model));
    desc_out->n_head_kv = hp.n_head_kv(0);
    desc_out->n_embd_head = hp.n_embd_head_k;
    desc_out->n_stream = 1;
    desc_out->v_trans = 0;
    // Runtime context controls KV cache dtype; precompute path assumes default f16 cache.
    desc_out->type_k = IK_KV_TYPE_F16;
    desc_out->type_v = IK_KV_TYPE_F16;

    const ik_kv_compat_convert_result_t fp_rc = ik_kv_model_fingerprint_build(
        model_path, desc_out->model_fingerprint);
    llama_free_model(model);
    if (fp_rc != IK_KV_COMPAT_CONVERT_OK) {
        return fp_rc;
    }

    return IK_KV_COMPAT_CONVERT_OK;
}

//
// Compatibility plan key and fingerprinting (KVB-004)
//

ik_kv_compat_convert_result_t ik_kv_compat_plan_key_build(
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst,
    ik_kv_compat_plan_key_t * key_out
) {
    if (!src || !dst || !key_out) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    std::vector<uint8_t> material;
    material.reserve(256);

    append_bytes(material, (const uint8_t *) "IK_KV_PLAN_KEY_V1", 16);

    append_u32(material, src->n_layers);
    append_u32(material, src->n_ctx);
    append_u32(material, src->n_head_kv);
    append_u32(material, src->n_embd_head);
    append_u64(material, src->payload_size);
    append_u8(material, src->type_k);
    append_u8(material, src->type_v);
    append_u8(material, src->v_trans);
    append_u8(material, src->n_stream);
    append_u8(material, src->source_format);
    append_bytes(material, src->model_fingerprint, IK_KV_FINGERPRINT_SIZE);

    append_u32(material, dst->n_layers);
    append_u32(material, dst->n_ctx);
    append_u32(material, dst->n_head_kv);
    append_u32(material, dst->n_embd_head);
    append_u8(material, dst->type_k);
    append_u8(material, dst->type_v);
    append_u8(material, dst->v_trans);
    append_u8(material, dst->n_stream);
    append_bytes(material, dst->model_fingerprint, IK_KV_FINGERPRINT_SIZE);

    static const uint64_t salts[8] = {
        0x13c6ef372fe94a71ull, 0x64b3f4d215ec8a09ull,
        0xb4894f7d03e2d6c5ull, 0x5a47ac20df18b6f3ull,
        0x39d1e7b4548cf20dull, 0x8f25c43e18db7691ull,
        0xc4f86123ab4d95e7ull, 0x7be2da0195c4f863ull,
    };

    memset(key_out, 0, sizeof(*key_out));
    for (size_t i = 0; i < 8; ++i) {
        const uint64_t seed = 1469598103934665603ull ^ salts[i];
        const uint64_t h = fnv1a64_seeded(seed, material.data(), material.size());
        for (int j = 0; j < 8; ++j) {
            key_out->data[i * 8 + j] = (uint8_t) ((h >> (8 * j)) & 0xFFu);
        }
    }

    return IK_KV_COMPAT_CONVERT_OK;
}

ik_kv_compat_convert_result_t ik_kv_model_fingerprint_build(
    const char * model_path,
    uint8_t fingerprint[32]
) {
    if (!model_path || !fingerprint) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }

    const uint8_t * bytes = (const uint8_t *) model_path;
    const size_t len = strlen(model_path);
    const uint64_t h0 = fnv1a64(bytes, len);
    const uint64_t h1 = fnv1a64((const uint8_t *) &len, sizeof(len));
    for (int i = 0; i < 4; ++i) {
        const uint64_t hv = (i & 1) ? (h1 ^ (h0 + (uint64_t) i * 0x9E3779B97F4A7C15ull))
                                    : (h0 ^ (h1 + (uint64_t) i * 0xD6E8FEB86659FD93ull));
        memcpy(fingerprint + i * 8, &hv, sizeof(hv));
    }

    return IK_KV_COMPAT_CONVERT_OK;
}

int ik_kv_fingerprint_compare(
    const uint8_t fp1[32],
    const uint8_t fp2[32]
) {
    if (!fp1 || !fp2) {
        return -1;  // Error
    }
    return memcmp(fp1, fp2, 32);
}

//
// Plan building (KVB-005)
//

ik_kv_compat_convert_result_t ik_kv_compat_plan_build_strict_v1(
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst,
    ik_kv_compat_plan_t * plan_out
) {
    if (!src || !dst || !plan_out) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    
    memset(plan_out, 0, sizeof(*plan_out));
    
    // Build plan key first
    ik_kv_compat_convert_result_t result = ik_kv_compat_plan_key_build(src, dst, &plan_out->key);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        return result;
    }
    
    // Strict v1 compatibility checks:
    
    // 1. Check model fingerprint when both sides provide one.
    if (!is_all_zero_fingerprint(src->model_fingerprint) &&
            !is_all_zero_fingerprint(dst->model_fingerprint) &&
            ik_kv_fingerprint_compare(src->model_fingerprint, dst->model_fingerprint) != 0) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_MODEL_FINGERPRINT;
        return IK_KV_COMPAT_CONVERT_OK;  // Plan built, but incompatible
    }
    
    // 2. Check type_k
    if (src->type_k != dst->type_k) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_TYPE_K_MISMATCH;
        return IK_KV_COMPAT_CONVERT_OK;
    }
    
    // 3. Check type_v
    if (src->type_v != dst->type_v) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_TYPE_V_MISMATCH;
        return IK_KV_COMPAT_CONVERT_OK;
    }
    
    // 4. Check v_trans
    if (src->v_trans <= 1 && dst->v_trans <= 1 && src->v_trans != dst->v_trans) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_VTRANS_MISMATCH;
        return IK_KV_COMPAT_CONVERT_OK;
    }
    
    // 5. Check n_stream (must be 1 in strict mode)
    if (src->n_stream != 1 || dst->n_stream != 1) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_N_STREAM_UNSUPPORTED;
        return IK_KV_COMPAT_CONVERT_OK;
    }
    
    // 6. Check n_layers
    if (src->n_layers != 0 && dst->n_layers != 0 && src->n_layers != dst->n_layers) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_N_LAYER_MISMATCH;
        return IK_KV_COMPAT_CONVERT_OK;
    }
    
    // 7. Check n_ctx
    if (src->n_ctx != 0 && dst->n_ctx != 0 && src->n_ctx != dst->n_ctx) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_N_CTX_MISMATCH;
        return IK_KV_COMPAT_CONVERT_OK;
    }
    
    // 8. Check n_head_kv
    if (src->n_head_kv != 0 && dst->n_head_kv != 0 && src->n_head_kv != dst->n_head_kv) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_N_HEAD_MISMATCH;
        return IK_KV_COMPAT_CONVERT_OK;
    }
    
    // All checks passed - compatible
    plan_out->is_compatible = true;
    plan_out->reject_reason = IK_KV_COMPAT_REJECT_NONE;
    plan_out->n_layers = src->n_layers ? src->n_layers : dst->n_layers;
    if (plan_out->n_layers > IK_KV_MAX_LAYERS) {
        plan_out->is_compatible = false;
        plan_out->reject_reason = IK_KV_COMPAT_REJECT_N_LAYER_MISMATCH;
        return IK_KV_COMPAT_CONVERT_OK;
    }

    const uint64_t src_size = src->payload_size;
    uint64_t dst_size = (uint64_t) ik_kv_convert_get_output_size(src, dst);

    const uint8_t needs_v_trans =
        (src->v_trans <= 1 && dst->v_trans <= 1 && src->v_trans != dst->v_trans) ? 1u : 0u;

    bool mapped_from_layout = false;
    if (src->payload && src_size > 0) {
        ik_stream_layout_t layout = {};
        size_t analyzed_dst_size = 0;
        ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
        if (analyze_source_payload_layout(src, &layout, &analyzed_dst_size, &reject)) {
            dst_size = (uint64_t) analyzed_dst_size;
            populate_plan_mappings_from_layout(layout, plan_out, needs_v_trans);
            mapped_from_layout = !layout.layers.empty();
        } else if (src->source_format == IK_KV_SOURCE_FORMAT_RTX_KVARTIF1 &&
                   reject != IK_KV_COMPAT_REJECT_NONE) {
            plan_out->is_compatible = false;
            plan_out->reject_reason = reject;
            return IK_KV_COMPAT_CONVERT_OK;
        }
    }

    if (!mapped_from_layout) {
        for (uint32_t il = 0; il < plan_out->n_layers; ++il) {
            const uint64_t src_begin = (src_size * il) / std::max<uint32_t>(1, plan_out->n_layers);
            const uint64_t src_end   = (src_size * (il + 1)) / std::max<uint32_t>(1, plan_out->n_layers);
            const uint64_t dst_begin = (dst_size * il) / std::max<uint32_t>(1, plan_out->n_layers);

            ik_kv_layer_mapping_t & m = plan_out->layer_mappings[il];
            memset(&m, 0, sizeof(m));
            m.layer_idx = il;
            m.k_row_size = saturating_u32(src_end - src_begin);
            m.v_row_size = saturating_u32(src_end - src_begin);
            m.k_offset = saturating_u32(src_begin);
            m.v_offset = saturating_u32(src_begin);
            m.dst_k_offset = saturating_u32(dst_begin);
            m.dst_v_offset = saturating_u32(dst_begin);
            m.needs_scatter = 0;
            m.needs_v_trans = needs_v_trans;
        }
    }

    plan_out->total_src_size = saturating_u32(src_size);
    plan_out->total_dst_size = saturating_u32(dst_size);
    
    return IK_KV_COMPAT_CONVERT_OK;
}

bool ik_kv_compat_plan_validate(
    const ik_kv_compat_plan_t * plan,
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst
) {
    if (!plan || !src || !dst) {
        return false;
    }
    
    // Verify key matches current source/destination
    ik_kv_compat_plan_key_t current_key;
    if (ik_kv_compat_plan_key_build(src, dst, &current_key) != IK_KV_COMPAT_CONVERT_OK) {
        return false;
    }
    
    if (memcmp(&plan->key, &current_key, sizeof(current_key)) != 0) {
        return false;
    }
    
    return plan->is_compatible;
}

//
// Plan cache (KVB-006)
//

ik_kv_compat_convert_result_t ik_kv_plan_cache_load(
    const ik_kv_compat_plan_key_t * key,
    ik_kv_compat_plan_t * plan_out
) {
    if (!key || !plan_out) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    memset(plan_out, 0, sizeof(*plan_out));

    const fs::path path = plan_cache_path_for_key(*key);
    if (path.empty()) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }

    std::vector<uint8_t> blob;
    if (!load_file_binary(path, blob)) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }

    if (blob.size() < IK_PLAN_CACHE_HEADER_SIZE + sizeof(ik_kv_compat_plan_t)) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }

    const uint8_t * p = blob.data();
    if (memcmp(p, IK_PLAN_CACHE_MAGIC.data(), IK_PLAN_CACHE_MAGIC.size()) != 0) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }
    p += IK_PLAN_CACHE_MAGIC.size();

    const uint16_t ver_major = rd_u16_le(p);
    p += 2;
    const uint16_t ver_minor = rd_u16_le(p);
    p += 2;
    const uint32_t header_size = rd_u32_le(p);
    p += 4;
    const uint32_t payload_size = rd_u32_le(p);
    p += 4;
    const uint32_t payload_crc = rd_u32_le(p);
    p += 4;

    if (ver_major != IK_PLAN_CACHE_VERSION_MAJOR ||
            ver_minor != IK_PLAN_CACHE_VERSION_MINOR ||
            header_size != IK_PLAN_CACHE_HEADER_SIZE ||
            payload_size != sizeof(ik_kv_compat_plan_t)) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }

    if (memcmp(p, key->data, IK_KV_PLAN_KEY_SIZE) != 0) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }
    p += IK_KV_PLAN_KEY_SIZE;

    if (blob.size() != (size_t) header_size + payload_size) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }

    const uint8_t * payload = blob.data() + header_size;
    if (crc32_ieee(payload, payload_size) != payload_crc) {
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }

    memcpy(plan_out, payload, sizeof(*plan_out));
    if (memcmp(plan_out->key.data, key->data, IK_KV_PLAN_KEY_SIZE) != 0) {
        memset(plan_out, 0, sizeof(*plan_out));
        return IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD;
    }

    return IK_KV_COMPAT_CONVERT_OK;
}

ik_kv_compat_convert_result_t ik_kv_plan_cache_store(
    const ik_kv_compat_plan_key_t * key,
    const ik_kv_compat_plan_t * plan
) {
    if (!key || !plan) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    const fs::path path = plan_cache_path_for_key(*key);
    if (path.empty()) {
        return IK_KV_COMPAT_CONVERT_OK;
    }

    if (memcmp(plan->key.data, key->data, IK_KV_PLAN_KEY_SIZE) != 0) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }

    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
        return IK_KV_COMPAT_CONVERT_ERR_INTERNAL;
    }

    const uint8_t * payload = (const uint8_t *) plan;
    const uint32_t payload_size = (uint32_t) sizeof(ik_kv_compat_plan_t);
    const uint32_t payload_crc = crc32_ieee(payload, payload_size);

    std::vector<uint8_t> blob;
    blob.reserve(IK_PLAN_CACHE_HEADER_SIZE + payload_size);
    append_bytes(blob, (const uint8_t *) IK_PLAN_CACHE_MAGIC.data(), IK_PLAN_CACHE_MAGIC.size());
    append_u16(blob, IK_PLAN_CACHE_VERSION_MAJOR);
    append_u16(blob, IK_PLAN_CACHE_VERSION_MINOR);
    append_u32(blob, IK_PLAN_CACHE_HEADER_SIZE);
    append_u32(blob, payload_size);
    append_u32(blob, payload_crc);
    append_bytes(blob, key->data, IK_KV_PLAN_KEY_SIZE);
    append_bytes(blob, payload, payload_size);

    if (!write_file_binary_atomic(path, blob.data(), blob.size())) {
        return IK_KV_COMPAT_CONVERT_ERR_INTERNAL;
    }

    return IK_KV_COMPAT_CONVERT_OK;
}

ik_kv_compat_convert_result_t ik_kv_plan_cache_invalidate(
    const ik_kv_compat_plan_key_t * key
) {
    if (!key) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    const fs::path path = plan_cache_path_for_key(*key);
    if (path.empty()) {
        return IK_KV_COMPAT_CONVERT_OK;
    }

    std::error_code ec;
    fs::remove(path, ec);
    if (ec) {
        return IK_KV_COMPAT_CONVERT_ERR_INTERNAL;
    }

    return IK_KV_COMPAT_CONVERT_OK;
}

void ik_kv_plan_cache_clear(void) {
    const std::string cache_dir = get_plan_cache_dir();
    if (cache_dir.empty()) {
        return;
    }

    std::error_code ec;
    const fs::path root(cache_dir);
    if (!fs::exists(root, ec) || ec) {
        return;
    }

    fs::directory_iterator it(root, ec);
    if (ec) {
        return;
    }
    const fs::directory_iterator end;

    for (; it != end; it.increment(ec)) {
        if (ec) {
            return;
        }
        if (!it->is_regular_file()) {
            continue;
        }
        const fs::path path = it->path();
        const std::string name = path.filename().string();
        if (name.rfind("plan_", 0) == 0 && path.extension() == ".ikvc") {
            std::error_code rm_ec;
            fs::remove(path, rm_ec);
        }
    }
}

//
// Conversion runtime (KVB-007)
//

static ik_kv_compat_convert_result_t convert_ik_seq_blob_with_plan(
    const ik_kv_source_descriptor_t * src,
    const ik_kv_compat_plan_t * plan,
    uint8_t * output_buf,
    size_t output_size,
    size_t * bytes_written,
    ik_kv_compat_reject_reason_t * reject_reason
) {
    if (!src || !plan || !output_buf || !bytes_written) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    *bytes_written = 0;
    if (reject_reason) {
        *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    }

    const size_t src_size = (size_t) src->payload_size;
    if (output_size < src_size) {
        return IK_KV_COMPAT_CONVERT_ERR_SIZE;
    }

    memcpy(output_buf, src->payload, src_size);
    size_t max_dst_end = src_size;
    bool used_mapping = false;

    const uint32_t n_layers = std::min<uint32_t>(plan->n_layers, IK_KV_MAX_LAYERS);
    for (uint32_t il = 0; il < n_layers; ++il) {
        const ik_kv_layer_mapping_t & m = plan->layer_mappings[il];

        auto copy_section = [&](uint32_t src_off, uint32_t dst_off, uint32_t bytes) -> bool {
            if (bytes == 0) {
                return true;
            }
            const size_t src_offset = (size_t) src_off;
            const size_t dst_offset = (size_t) dst_off;
            const size_t n_bytes = (size_t) bytes;
            if (src_offset + n_bytes > src_size || dst_offset + n_bytes > output_size) {
                return false;
            }
            memcpy(output_buf + dst_offset, src->payload + src_offset, n_bytes);
            max_dst_end = std::max(max_dst_end, dst_offset + n_bytes);
            used_mapping = true;
            return true;
        };

        if (!copy_section(m.k_offset, m.dst_k_offset, m.k_row_size) ||
                !copy_section(m.v_offset, m.dst_v_offset, m.v_row_size)) {
            if (reject_reason) {
                *reject_reason = IK_KV_COMPAT_REJECT_INCOMPATIBLE_PROFILE;
            }
            return IK_KV_COMPAT_CONVERT_ERR_CONVERT;
        }
    }

    *bytes_written = used_mapping ? max_dst_end : src_size;
    return IK_KV_COMPAT_CONVERT_OK;
}

ik_kv_compat_convert_result_t ik_kv_convert_prefill_to_ik_seq_blob(
    ik_kv_convert_ctx_t * ctx
) {
    if (!ctx) {
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    
    if (!ctx->src || !ctx->dst || !ctx->plan) {
        ctx->result = IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
        ctx->reject = IK_KV_COMPAT_REJECT_NONE;
        return ctx->result;
    }
    
    // Check compatibility
    if (!ctx->plan->is_compatible) {
        ctx->result = IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH;
        ctx->reject = ctx->plan->reject_reason;
        return ctx->result;
    }
    
    const size_t required_output = ctx->plan->total_dst_size > 0
        ? (size_t) ctx->plan->total_dst_size
        : ik_kv_convert_get_output_size(ctx->src, ctx->dst);

    // Check output buffer
    if (!ctx->output_buf || ctx->output_size < required_output) {
        ctx->result = IK_KV_COMPAT_CONVERT_ERR_SIZE;
        ctx->reject = IK_KV_COMPAT_REJECT_NONE;
        return ctx->result;
    }

    if (ctx->src->source_format == IK_KV_SOURCE_FORMAT_RTX_KVARTIF1) {
        ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
        const ik_kv_compat_convert_result_t rc = convert_rtx_seq_blob_to_ik(
            ctx->src->payload,
            (size_t) ctx->src->payload_size,
            ctx->output_buf,
            ctx->output_size,
            &ctx->bytes_written,
            &reject);
        if (rc != IK_KV_COMPAT_CONVERT_OK) {
            ctx->result = rc;
            ctx->reject = reject;
            return ctx->result;
        }
    } else {
        ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
        const ik_kv_compat_convert_result_t rc = convert_ik_seq_blob_with_plan(
            ctx->src,
            ctx->plan,
            ctx->output_buf,
            ctx->output_size,
            &ctx->bytes_written,
            &reject);
        if (rc != IK_KV_COMPAT_CONVERT_OK) {
            ctx->result = rc;
            ctx->reject = reject;
            return ctx->result;
        }
    }

    ctx->result = IK_KV_COMPAT_CONVERT_OK;
    ctx->reject = IK_KV_COMPAT_REJECT_NONE;

    return IK_KV_COMPAT_CONVERT_OK;
}

size_t ik_kv_convert_get_output_size(
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst
) {
    if (!src || !dst) {
        return 0;
    }
    GGML_UNUSED(dst);

    if (src->source_format == IK_KV_SOURCE_FORMAT_RTX_KVARTIF1 && src->payload && src->payload_size > 0) {
        ik_stream_layout_t layout = {};
        size_t dst_size = 0;
        ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
        if (analyze_source_payload_layout(src, &layout, &dst_size, &reject)) {
            return dst_size;
        }
    }

    // Fallback: conversion output never exceeds source payload size.
    return (size_t) src->payload_size;
}

//
// Decode import hook (KVB-008)
//

ik_kv_compat_convert_result_t ik_kv_import_into_context(
    struct llama_context * ctx,
    const uint8_t * artifact_data,
    size_t artifact_size,
    int32_t dest_seq_id,
    ik_kv_compat_reject_reason_t * reject_reason
) {
    if (!ctx || !artifact_data) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }

    if (!g_kv_bridge.initialized) {
        ik_kv_bridge_init();
    }

    const auto t0 = std::chrono::steady_clock::now();
    ik_kv_bridge_metrics_t metrics = {};
    metrics.mode = (uint8_t) g_kv_bridge.config.mode;
    metrics.status = 2;
    metrics.bytes_in = artifact_size;
    bool fallback_used = false;

    auto finalize = [&](ik_kv_compat_convert_result_t rc,
                        ik_kv_compat_reject_reason_t reject) -> ik_kv_compat_convert_result_t {
        const auto t1 = std::chrono::steady_clock::now();
        metrics.convert_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        metrics.reject_reason = (uint8_t) reject;
        if (rc == IK_KV_COMPAT_CONVERT_OK) {
            metrics.status = 0;
        } else if (reject != IK_KV_COMPAT_REJECT_NONE || rc == IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH) {
            metrics.status = 1;
        } else {
            metrics.status = 2;
        }
        g_kv_bridge.last_metrics = metrics;
        bridge_log_metrics(metrics, rc, reject, fallback_used);
        if (reject_reason) {
            *reject_reason = reject;
        }
        return rc;
    };

    auto try_fallback_payload_import = [&](const uint8_t * payload, size_t payload_size) -> bool {
        if (g_kv_bridge.config.no_fallback || !payload || payload_size == 0) {
            return false;
        }
        fallback_used = true;
        metrics.bytes_out = (uint64_t) payload_size;
        if (g_kv_bridge.config.dry_run) {
            return true;
        }
        const size_t n_read = llama_state_seq_set_data(ctx, payload, payload_size, dest_seq_id);
        return n_read == payload_size;
    };

    // Check bridge mode
    if (g_kv_bridge.config.mode == IK_KV_BRIDGE_MODE_OFF) {
        kv_artifact_view_t off_view = {};
        ik_kv_compat_reject_reason_t off_reject = IK_KV_COMPAT_REJECT_NONE;
        if (!parse_artifact_view(artifact_data, artifact_size, &off_view, &off_reject)) {
            return finalize(IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE, off_reject);
        }
        if (!ik_kv_source_validate_payload(&off_view.header, off_view.payload, off_view.payload_size)) {
            return finalize(IK_KV_COMPAT_CONVERT_ERR_CHECKSUM, IK_KV_COMPAT_REJECT_PAYLOAD_CRC_MISMATCH);
        }
        if (try_fallback_payload_import(off_view.payload, off_view.payload_size)) {
            return finalize(IK_KV_COMPAT_CONVERT_OK, IK_KV_COMPAT_REJECT_NONE);
        }
        return finalize(IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG, IK_KV_COMPAT_REJECT_NONE);
    }

    kv_artifact_view_t artifact_view = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
    if (!parse_artifact_view(artifact_data, artifact_size, &artifact_view, &reject)) {
        return finalize(IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE, reject);
    }

    if (!ik_kv_source_validate_payload(&artifact_view.header, artifact_view.payload, artifact_view.payload_size)) {
        return finalize(IK_KV_COMPAT_CONVERT_ERR_CHECKSUM, IK_KV_COMPAT_REJECT_PAYLOAD_CRC_MISMATCH);
    }

    // Parse source descriptor.
    ik_kv_source_descriptor_t src_desc = {};
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_prefill_seq_state(
        &artifact_view.header,
        artifact_view.payload,
        artifact_view.payload_size,
        &src_desc,
        &reject);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        return finalize(result, reject);
    }
    metrics.bytes_in = (uint64_t) src_desc.payload_size;

    // Introspect destination.
    ik_kv_dest_descriptor_t dst_desc = {};
    result = ik_kv_dest_introspect_from_ctx(ctx, &dst_desc);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        return finalize(result, IK_KV_COMPAT_REJECT_NONE);
    }

    ik_kv_compat_plan_key_t plan_key = {};
    result = ik_kv_compat_plan_key_build(&src_desc, &dst_desc, &plan_key);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        return finalize(result, IK_KV_COMPAT_REJECT_NONE);
    }
    metrics.plan_key = fnv1a64(plan_key.data, sizeof(plan_key.data));

    ik_kv_compat_plan_t plan = {};
    bool plan_loaded = false;

    const ik_kv_compat_convert_result_t cache_rc = ik_kv_plan_cache_load(&plan_key, &plan);
    if (cache_rc == IK_KV_COMPAT_CONVERT_OK) {
        if (ik_kv_compat_plan_validate(&plan, &src_desc, &dst_desc)) {
            plan_loaded = true;
            metrics.plan_cache_hit = true;
        } else {
            (void) ik_kv_plan_cache_invalidate(&plan_key);
            memset(&plan, 0, sizeof(plan));
        }
    }

    if (!plan_loaded) {
        result = ik_kv_compat_plan_build_strict_v1(&src_desc, &dst_desc, &plan);
        if (result != IK_KV_COMPAT_CONVERT_OK) {
            return finalize(result, IK_KV_COMPAT_REJECT_NONE);
        }
        if (!plan.is_compatible &&
            g_kv_bridge.config.mode == IK_KV_BRIDGE_MODE_RELAXED &&
            g_kv_bridge.config.allow_vtrans_convert &&
            plan.reject_reason == IK_KV_COMPAT_REJECT_VTRANS_MISMATCH) {
            ik_kv_source_descriptor_t src_relaxed = src_desc;
            src_relaxed.v_trans = dst_desc.v_trans;
            ik_kv_compat_plan_t relaxed_plan = {};
            const ik_kv_compat_convert_result_t relaxed_rc =
                ik_kv_compat_plan_build_strict_v1(&src_relaxed, &dst_desc, &relaxed_plan);
            if (relaxed_rc == IK_KV_COMPAT_CONVERT_OK && relaxed_plan.is_compatible) {
                plan = relaxed_plan;
                for (uint32_t il = 0; il < std::min<uint32_t>(plan.n_layers, IK_KV_MAX_LAYERS); ++il) {
                    plan.layer_mappings[il].needs_v_trans = 1;
                }
            }
        }
        (void) ik_kv_plan_cache_store(&plan_key, &plan);
    }

    if (!plan.is_compatible) {
        if (try_fallback_payload_import(src_desc.payload, (size_t) src_desc.payload_size)) {
            return finalize(IK_KV_COMPAT_CONVERT_OK, IK_KV_COMPAT_REJECT_NONE);
        }
        return finalize(IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH, plan.reject_reason);
    }

    // Convert source payload to destination sequence-state blob.
    const size_t converted_size = ik_kv_convert_get_output_size(&src_desc, &dst_desc);
    if (converted_size == 0) {
        if (try_fallback_payload_import(src_desc.payload, (size_t) src_desc.payload_size)) {
            return finalize(IK_KV_COMPAT_CONVERT_OK, IK_KV_COMPAT_REJECT_NONE);
        }
        return finalize(IK_KV_COMPAT_CONVERT_ERR_SIZE, IK_KV_COMPAT_REJECT_INCOMPATIBLE_PROFILE);
    }
    std::vector<uint8_t> converted(converted_size);
    ik_kv_convert_ctx_t cvt = {};
    cvt.src = &src_desc;
    cvt.dst = &dst_desc;
    cvt.plan = &plan;
    cvt.output_buf = converted.data();
    cvt.output_size = converted.size();

    result = ik_kv_convert_prefill_to_ik_seq_blob(&cvt);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        if (try_fallback_payload_import(src_desc.payload, (size_t) src_desc.payload_size)) {
            return finalize(IK_KV_COMPAT_CONVERT_OK, IK_KV_COMPAT_REJECT_NONE);
        }
        return finalize(result, cvt.reject);
    }
    metrics.bytes_out = (uint64_t) cvt.bytes_written;

    if (g_kv_bridge.config.dry_run) {
        return finalize(IK_KV_COMPAT_CONVERT_OK, IK_KV_COMPAT_REJECT_NONE);
    }

    // Import converted state into destination sequence.
    const size_t n_read = llama_state_seq_set_data(ctx, converted.data(), cvt.bytes_written, dest_seq_id);
    if (n_read != cvt.bytes_written) {
        if (try_fallback_payload_import(src_desc.payload, (size_t) src_desc.payload_size)) {
            return finalize(IK_KV_COMPAT_CONVERT_OK, IK_KV_COMPAT_REJECT_NONE);
        }
        return finalize(IK_KV_COMPAT_CONVERT_ERR_CONVERT, IK_KV_COMPAT_REJECT_INCOMPATIBLE_PROFILE);
    }

    return finalize(IK_KV_COMPAT_CONVERT_OK, IK_KV_COMPAT_REJECT_NONE);
}

ik_kv_compat_convert_result_t ik_kv_import_into_context_with_plan(
    struct llama_context * ctx,
    const ik_kv_compat_plan_t * plan,
    const uint8_t * payload_data,
    size_t payload_size,
    int32_t dest_seq_id,
    ik_kv_compat_reject_reason_t * reject_reason
) {
    if (!ctx || !plan || !payload_data) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    if (!plan->is_compatible) {
        if (reject_reason) *reject_reason = plan->reject_reason;
        return IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH;
    }

    const size_t n_read = llama_state_seq_set_data(ctx, payload_data, payload_size, dest_seq_id);
    if (n_read != payload_size) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_INCOMPATIBLE_PROFILE;
        return IK_KV_COMPAT_CONVERT_ERR_CONVERT;
    }

    if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    return IK_KV_COMPAT_CONVERT_OK;
}

//
// Telemetry (KVB-010)
//

void ik_kv_bridge_get_last_metrics(ik_kv_bridge_metrics_t * metrics) {
    if (!metrics) return;
    *metrics = g_kv_bridge.last_metrics;
}

void ik_kv_bridge_reset_metrics(void) {
    memset(&g_kv_bridge.last_metrics, 0, sizeof(g_kv_bridge.last_metrics));
}

void ik_kv_bridge_set_telemetry_enabled(bool enabled) {
    g_kv_bridge.telemetry_enabled = enabled;
}

//
// Debug utilities (KVB-012)
//

void ik_kv_bridge_inspect_artifact(const uint8_t * artifact_data, size_t artifact_size) {
    kv_artifact_view_t view = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
    if (!parse_artifact_view(artifact_data, artifact_size, &view, &reject)) {
        printf("Invalid artifact: parse failed (%s)\n", ik_kv_compat_reject_str(reject));
        return;
    }

    const ik_kva_header_t * header = &view.header;

    printf("=== KVA Artifact Header ===\n");
    printf("Magic:         0x%08X (expected: 0x%08X)\n", 
           header->magic, IK_KVA_MAGIC);
    printf("Version:       %u.%u\n", 
           header->format_major, header->format_minor);
    printf("Layers:        %u\n", header->n_layers);
    printf("Context:       %u\n", header->n_ctx);
    printf("KV Heads:      %u\n", header->n_head_kv);
    printf("Type K/V:      %u / %u\n", header->type_k, header->type_v);
    printf("V Transpose:   %u\n", header->v_trans);
    printf("N Streams:     %u\n", header->n_stream);
    printf("Payload Size:  %lu bytes\n", (unsigned long)header->payload_size);
    printf("Payload CRC:   0x%08X\n", header->payload_crc);
    printf("Source Format: %u\n", header->reserved[HDR_RESERVED_SOURCE_FMT]);
    
    printf("Fingerprint:   ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", header->model_fingerprint[i]);
    }
    printf("\n");
}

void ik_kv_bridge_print_plan(const ik_kv_compat_plan_t * plan) {
    if (!plan) {
        printf("NULL plan\n");
        return;
    }
    
    printf("=== KV Compat Plan ===\n");
    printf("Compatible:    %s\n", plan->is_compatible ? "YES" : "NO");
    printf("Reject Reason: %s\n", ik_kv_compat_reject_str(plan->reject_reason));
    printf("Layers:        %u\n", plan->n_layers);
    printf("Src Size:      %u bytes\n", plan->total_src_size);
    printf("Dst Size:      %u bytes\n", plan->total_dst_size);
    
    printf("Plan Key:      ");
    for (int i = 0; i < 16; i++) {
        printf("%02x", plan->key.data[i]);
    }
    printf("...\n");
}

void ik_kv_bridge_print_source_descriptor(const ik_kv_source_descriptor_t * desc) {
    if (!desc) {
        printf("NULL source descriptor\n");
        return;
    }
    
    printf("=== Source Descriptor ===\n");
    printf("Layers:        %u\n", desc->n_layers);
    printf("Context:       %u\n", desc->n_ctx);
    printf("KV Heads:      %u\n", desc->n_head_kv);
    printf("Emb per Head:  %u\n", desc->n_embd_head);
    printf("Type K/V:      %u / %u\n", desc->type_k, desc->type_v);
    printf("V Transpose:   %u\n", desc->v_trans);
    printf("N Streams:     %u\n", desc->n_stream);
    printf("Token Count:   %u\n", desc->token_count);
    printf("Source Format: %u\n", desc->source_format);
    printf("Payload Size:  %lu bytes\n", (unsigned long)desc->payload_size);
}

void ik_kv_bridge_print_dest_descriptor(const ik_kv_dest_descriptor_t * desc) {
    if (!desc) {
        printf("NULL dest descriptor\n");
        return;
    }
    
    printf("=== Dest Descriptor ===\n");
    printf("Layers:        %u\n", desc->n_layers);
    printf("Context:       %u\n", desc->n_ctx);
    printf("KV Heads:      %u\n", desc->n_head_kv);
    printf("Emb per Head:  %u\n", desc->n_embd_head);
    printf("Type K/V:      %u / %u\n", desc->type_k, desc->type_v);
    printf("V Transpose:   %u\n", desc->v_trans);
    printf("N Streams:     %u\n", desc->n_stream);
}

ik_kv_compat_convert_result_t ik_kv_bridge_validate_only(
    struct llama_context * ctx,
    const uint8_t * artifact_data,
    size_t artifact_size,
    ik_kv_compat_reject_reason_t * reject_reason
) {
    if (!ctx || !artifact_data) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG;
    }
    
    kv_artifact_view_t artifact_view = {};
    if (!parse_artifact_view(artifact_data, artifact_size, &artifact_view, reject_reason)) {
        return IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE;
    }
    if (!ik_kv_source_validate_payload(&artifact_view.header, artifact_view.payload, artifact_view.payload_size)) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_PAYLOAD_CRC_MISMATCH;
        return IK_KV_COMPAT_CONVERT_ERR_CHECKSUM;
    }

    ik_kv_source_descriptor_t src_desc;
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_prefill_seq_state(
        &artifact_view.header,
        artifact_view.payload,
        artifact_view.payload_size,
        &src_desc,
        reject_reason);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        return result;
    }
    
    ik_kv_dest_descriptor_t dst_desc;
    result = ik_kv_dest_introspect_from_ctx(ctx, &dst_desc);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return result;
    }
    
    ik_kv_compat_plan_t plan;
    result = ik_kv_compat_plan_build_strict_v1(&src_desc, &dst_desc, &plan);
    if (result != IK_KV_COMPAT_CONVERT_OK) {
        if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
        return result;
    }

    if (!plan.is_compatible &&
        g_kv_bridge.config.mode == IK_KV_BRIDGE_MODE_RELAXED &&
        g_kv_bridge.config.allow_vtrans_convert &&
        plan.reject_reason == IK_KV_COMPAT_REJECT_VTRANS_MISMATCH) {
        ik_kv_source_descriptor_t src_relaxed = src_desc;
        src_relaxed.v_trans = dst_desc.v_trans;
        ik_kv_compat_plan_t relaxed_plan = {};
        const ik_kv_compat_convert_result_t relaxed_rc =
            ik_kv_compat_plan_build_strict_v1(&src_relaxed, &dst_desc, &relaxed_plan);
        if (relaxed_rc == IK_KV_COMPAT_CONVERT_OK && relaxed_plan.is_compatible) {
            plan = relaxed_plan;
        }
    }
    
    if (!plan.is_compatible) {
        if (reject_reason) *reject_reason = plan.reject_reason;
        return IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH;
    }
    
    if (reject_reason) *reject_reason = IK_KV_COMPAT_REJECT_NONE;
    return IK_KV_COMPAT_CONVERT_OK;
}
