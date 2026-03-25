#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct llama_kv_artifact_metadata {
    uint16_t format_major = 1;
    uint16_t format_minor = 0;
    uint32_t n_layers = 0;
    uint32_t n_ctx = 0;
    uint32_t token_count = 0;
    uint16_t type_k = 0;
    uint16_t type_v = 0;
    uint32_t flags = 0;
    // --- v1.1 extension (format_minor >= 1) ---
    int32_t  last_token_id     = -1;
    uint8_t  v_trans           = 0;
    uint8_t  is_mla            = 0;
    uint16_t reserved_0        = 0;
    uint64_t model_fingerprint = 0;
};

struct llama_kv_artifact_summary {
    uint64_t payload_bytes = 0;
    uint32_t payload_crc32 = 0;
};

constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_PROGRESSIVE = 1u << 0;
constexpr uint32_t LLAMA_KV_ARTIFACT_FLAG_TRANSPORTED = 1u << 1;

bool llama_kv_artifact_write(const std::string &               path,
                             const std::vector<uint8_t> &      payload,
                             const llama_kv_artifact_metadata & meta,
                             llama_kv_artifact_summary *        summary,
                             std::string *                      error);

bool llama_kv_artifact_read(const std::string &          path,
                            std::vector<uint8_t> &       payload_out,
                            llama_kv_artifact_metadata * meta_out,
                            llama_kv_artifact_summary *  summary_out,
                            std::string *                error);

uint32_t llama_kv_artifact_crc32(const uint8_t * data, size_t size);

bool llama_kv_artifact_read_mem(const uint8_t *              data,
                                size_t                       size,
                                std::vector<uint8_t> &       payload_out,
                                llama_kv_artifact_metadata * meta_out,
                                llama_kv_artifact_summary *  summary_out,
                                std::string *                error);
