//
// KV Bridge Parser Unit Tests (KVB-UT-001, KVB-UT-010, KVB-UT-011, KVB-UT-012)
// Copyright (C) 2025 Iwan Kawrakow / ik_llama contributors
// MIT license
// SPDX-License-Identifier: MIT
//

#include "kv-bridge/ik-kv-compat.h"
#include "kv-bridge/ik-kv-compat-types.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <stdint.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

// Test counters
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("Running %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
    tests_passed++; \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAILED at line %d: %d != %d\n", __LINE__, (int)(a), (int)(b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        printf("FAILED at line %d: expected true\n", __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))

#define ASSERT_STR_EQ(a, b) do { \
    if (strcmp((a), (b)) != 0) { \
        printf("FAILED at line %d: \"%s\" != \"%s\"\n", __LINE__, (a), (b)); \
        tests_failed++; \
        return; \
    } \
} while(0)

static void append_u32_le(std::vector<uint8_t> & out, uint32_t v) {
    out.push_back((uint8_t) (v & 0xFFu));
    out.push_back((uint8_t) ((v >> 8) & 0xFFu));
    out.push_back((uint8_t) ((v >> 16) & 0xFFu));
    out.push_back((uint8_t) ((v >> 24) & 0xFFu));
}

static void append_u64_le(std::vector<uint8_t> & out, uint64_t v) {
    append_u32_le(out, (uint32_t) (v & 0xFFFFFFFFull));
    append_u32_le(out, (uint32_t) ((v >> 32) & 0xFFFFFFFFull));
}

static void write_u16_le(std::vector<uint8_t> & out, size_t off, uint16_t v) {
    out[off + 0] = (uint8_t) (v & 0xFFu);
    out[off + 1] = (uint8_t) ((v >> 8) & 0xFFu);
}

static void write_u32_le(std::vector<uint8_t> & out, size_t off, uint32_t v) {
    out[off + 0] = (uint8_t) (v & 0xFFu);
    out[off + 1] = (uint8_t) ((v >> 8) & 0xFFu);
    out[off + 2] = (uint8_t) ((v >> 16) & 0xFFu);
    out[off + 3] = (uint8_t) ((v >> 24) & 0xFFu);
}

static void write_u64_le(std::vector<uint8_t> & out, size_t off, uint64_t v) {
    write_u32_le(out, off + 0, (uint32_t) (v & 0xFFFFFFFFull));
    write_u32_le(out, off + 4, (uint32_t) ((v >> 32) & 0xFFFFFFFFull));
}

static uint32_t read_u32_le(const uint8_t * p) {
    return (uint32_t) (p[0] |
        ((uint32_t) p[1] << 8) |
        ((uint32_t) p[2] << 16) |
        ((uint32_t) p[3] << 24));
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

static std::vector<uint8_t> build_rtx_payload_single_stream(void) {
    std::vector<uint8_t> payload;

    // n_stream, cell_count
    append_u32_le(payload, 1);
    append_u32_le(payload, 1);

    // one metadata cell: pos=0, n_seq_id=1, seq_id=0
    append_u32_le(payload, 0);
    append_u32_le(payload, 1);
    append_u32_le(payload, 0);

    // data header: v_state=0 (not transposed), n_layer=1
    append_u32_le(payload, 0);
    append_u32_le(payload, 1);

    // layer 0 K header + 2-byte row payload
    append_u32_le(payload, 1); // GGML_TYPE_F16
    append_u64_le(payload, 2);
    payload.push_back(0x11);
    payload.push_back(0x22);

    // layer 0 V header + 2-byte row payload
    append_u32_le(payload, 1); // GGML_TYPE_F16
    append_u64_le(payload, 2);
    payload.push_back(0x33);
    payload.push_back(0x44);

    return payload;
}

static std::vector<uint8_t> build_rtx_payload_two_streams_one_active(void) {
    const std::vector<uint8_t> single = build_rtx_payload_single_stream();
    std::vector<uint8_t> payload;

    append_u32_le(payload, 2); // n_stream
    payload.insert(payload.end(), single.begin() + 4, single.end()); // stream 0 block
    append_u32_le(payload, 0); // stream 1 cell_count

    return payload;
}

static std::vector<uint8_t> build_rtx_payload_two_streams_two_active(void) {
    const std::vector<uint8_t> single = build_rtx_payload_single_stream();
    std::vector<uint8_t> payload;

    append_u32_le(payload, 2); // n_stream
    payload.insert(payload.end(), single.begin() + 4, single.end()); // stream 0 block
    payload.insert(payload.end(), single.begin() + 4, single.end()); // stream 1 block

    return payload;
}

static std::vector<uint8_t> build_rtx_artifact(size_t header_size, const std::vector<uint8_t> & payload, uint32_t token_count) {
    std::vector<uint8_t> artifact(header_size + payload.size(), 0);

    const char magic[8] = { 'K', 'V', 'A', 'R', 'T', 'I', 'F', '1' };
    memcpy(artifact.data(), magic, 8);

    write_u16_le(artifact, 8, 1);   // format_major
    write_u16_le(artifact, 10, 0);  // format_minor
    write_u32_le(artifact, 12, 32); // n_layers
    write_u32_le(artifact, 16, 4096); // n_ctx
    write_u32_le(artifact, 20, token_count);
    write_u16_le(artifact, 24, 1);  // type_k GGML_TYPE_F16
    write_u16_le(artifact, 26, 1);  // type_v GGML_TYPE_F16
    write_u32_le(artifact, 28, 0);  // flags
    write_u64_le(artifact, 32, payload.size());
    write_u32_le(artifact, 40, crc32_ieee(payload.data(), payload.size()));

    memcpy(artifact.data() + header_size, payload.data(), payload.size());
    return artifact;
}

static std::string make_test_cache_dir(const char * tag) {
    const auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::string("/tmp/ik_kv_plan_cache_") + tag + "_" + std::to_string((long long) ts);
}

static void build_compatible_src_dst(
        ik_kv_source_descriptor_t & src,
        ik_kv_dest_descriptor_t & dst) {
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));

    src.n_layers = dst.n_layers = 32;
    src.n_ctx = dst.n_ctx = 4096;
    src.n_head_kv = dst.n_head_kv = 32;
    src.n_embd_head = dst.n_embd_head = 128;
    src.type_k = dst.type_k = IK_KV_TYPE_F16;
    src.type_v = dst.type_v = IK_KV_TYPE_F16;
    src.v_trans = dst.v_trans = 0;
    src.n_stream = dst.n_stream = 1;
    src.source_format = IK_KV_SOURCE_FORMAT_IK_KVA;
    src.payload_size = 1024;
    memset(src.model_fingerprint, 0xAB, sizeof(src.model_fingerprint));
    memcpy(dst.model_fingerprint, src.model_fingerprint, sizeof(dst.model_fingerprint));
}

//
// KVB-UT-001: Module compiles, API symbols link
//

TEST(kvb_ut_001_module_symbols_link) {
    // Test that all API symbols are available
    ASSERT_TRUE(ik_kv_bridge_init() == true);
    
    ik_kv_bridge_config_t config;
    ik_kv_bridge_get_config(&config);
    ASSERT_TRUE(config.mode == IK_KV_BRIDGE_MODE_STRICT);  // Default
    
    // Test string conversion functions
    ASSERT_STR_EQ(ik_kv_compat_result_str(IK_KV_COMPAT_CONVERT_OK), "OK");
    ASSERT_STR_EQ(ik_kv_compat_reject_str(IK_KV_COMPAT_REJECT_NONE), "NONE");
    
    // Test size function
    ASSERT_TRUE(ik_kva_header_size() == sizeof(ik_kva_header_t));
    
    ik_kv_bridge_shutdown();
    tests_passed++;  // Count this test
}

//
// KVB-UT-010: Valid .kva header parse
//

TEST(kvb_ut_010_valid_kva_header_parse) {
    ik_kv_bridge_init();
    
    // Create a valid header
    ik_kva_header_t valid_header;
    memset(&valid_header, 0, sizeof(valid_header));
    valid_header.magic = IK_KVA_MAGIC;
    valid_header.format_major = IK_KVA_FORMAT_MAJOR;
    valid_header.format_minor = IK_KVA_FORMAT_MINOR;
    valid_header.n_layers = 32;
    valid_header.n_ctx = 4096;
    valid_header.n_head_kv = 32;
    valid_header.type_k = IK_KV_TYPE_F16;
    valid_header.type_v = IK_KV_TYPE_F16;
    valid_header.v_trans = 0;
    valid_header.n_stream = 1;
    valid_header.payload_size = 1024 * 1024;
    
    uint8_t artifact_data[sizeof(ik_kva_header_t) + 100];
    memcpy(artifact_data, &valid_header, sizeof(valid_header));
    
    ik_kva_header_t parsed_header;
    ik_kv_compat_reject_reason_t reject;
    
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact_data, sizeof(artifact_data), &parsed_header, &reject);
    
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(reject, IK_KV_COMPAT_REJECT_NONE);
    ASSERT_EQ(parsed_header.magic, IK_KVA_MAGIC);
    ASSERT_EQ(parsed_header.n_layers, 32);
    ASSERT_EQ(parsed_header.n_ctx, 4096);
    ASSERT_EQ(parsed_header.n_head_kv, 32);
    
    ik_kv_bridge_shutdown();
    tests_passed++;
}

//
// KVB-UT-011: Invalid magic/version/header size reject
//

TEST(kvb_ut_011_invalid_header_reject) {
    ik_kv_bridge_init();
    
    // Test invalid magic
    ik_kva_header_t bad_magic_header;
    memset(&bad_magic_header, 0, sizeof(bad_magic_header));
    bad_magic_header.magic = 0xDEADBEEF;  // Wrong magic
    bad_magic_header.format_major = 1;
    
    ik_kva_header_t parsed_header;
    ik_kv_compat_reject_reason_t reject;
    
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        (uint8_t*)&bad_magic_header, sizeof(bad_magic_header), &parsed_header, &reject);
    
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE);
    ASSERT_EQ(reject, IK_KV_COMPAT_REJECT_MAGIC_MISMATCH);
    
    // Test unsupported version
    ik_kva_header_t bad_version_header;
    memset(&bad_version_header, 0, sizeof(bad_version_header));
    bad_version_header.magic = IK_KVA_MAGIC;
    bad_version_header.format_major = 99;  // Future version
    
    result = ik_kv_source_parse_kva_header(
        (uint8_t*)&bad_version_header, sizeof(bad_version_header), &parsed_header, &reject);
    
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE);
    ASSERT_EQ(reject, IK_KV_COMPAT_REJECT_VERSION_UNSUPPORTED);
    
    // Test too small
    uint8_t small_data[10];
    result = ik_kv_source_parse_kva_header(
        small_data, sizeof(small_data), &parsed_header, &reject);
    
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE);
    ASSERT_EQ(reject, IK_KV_COMPAT_REJECT_HEADER_MALFORMED);
    
    ik_kv_bridge_shutdown();
    tests_passed++;
}

//
// KVB-UT-012: Payload CRC mismatch reject
//

TEST(kvb_ut_012_payload_crc_mismatch_reject) {
    ik_kv_bridge_init();
    
    // Create header with CRC that doesn't match payload
    ik_kva_header_t header;
    memset(&header, 0, sizeof(header));
    header.magic = IK_KVA_MAGIC;
    header.format_major = IK_KVA_FORMAT_MAJOR;
    header.n_layers = 32;
    header.n_ctx = 4096;
    header.payload_size = 100;
    header.payload_crc = 0xDEADBEEF;  // Wrong CRC
    
    uint8_t payload[100];
    memset(payload, 0xAB, sizeof(payload));
    
    // Validation should fail
    bool valid = ik_kv_source_validate_payload(&header, payload, sizeof(payload));
    ASSERT_FALSE(valid);
    
    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_013_valid_rtx_kvartif1_header_parse) {
    ik_kv_bridge_init();

    const std::vector<uint8_t> payload = build_rtx_payload_single_stream();
    const uint32_t token_count = 123;
    const std::vector<uint8_t> artifact = build_rtx_artifact(48, payload, token_count);

    ik_kva_header_t parsed_header;
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_UNKNOWN;

    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact.data(), artifact.size(), &parsed_header, &reject);

    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(reject, IK_KV_COMPAT_REJECT_NONE);
    ASSERT_EQ(parsed_header.n_layers, 32);
    ASSERT_EQ(parsed_header.n_ctx, 4096);
    ASSERT_EQ(parsed_header.type_k, IK_KV_TYPE_F16);
    ASSERT_EQ(parsed_header.type_v, IK_KV_TYPE_F16);

    ASSERT_TRUE(ik_kv_source_validate_payload(&parsed_header, artifact.data() + 48, payload.size()));

    ik_kv_source_descriptor_t src_desc;
    result = ik_kv_source_parse_prefill_seq_state(
        &parsed_header, artifact.data() + 48, payload.size(), &src_desc, &reject);

    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(src_desc.source_format, IK_KV_SOURCE_FORMAT_RTX_KVARTIF1);
    ASSERT_EQ(src_desc.n_stream, 1);
    ASSERT_EQ(src_desc.v_trans, 0);
    ASSERT_EQ(src_desc.token_count, token_count);

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_014_rtx_convert_to_ik_seq_blob_rewrites_meta) {
    ik_kv_bridge_init();

    const std::vector<uint8_t> payload = build_rtx_payload_single_stream();
    const std::vector<uint8_t> artifact = build_rtx_artifact(48, payload, /*token_count=*/7);

    ik_kva_header_t header = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_UNKNOWN;
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact.data(), artifact.size(), &header, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_source_descriptor_t src = {};
    result = ik_kv_source_parse_prefill_seq_state(
        &header, artifact.data() + 48, payload.size(), &src, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(src.source_format, IK_KV_SOURCE_FORMAT_RTX_KVARTIF1);

    ik_kv_dest_descriptor_t dst = {};
    dst.n_layers = src.n_layers;
    dst.n_ctx = src.n_ctx;
    dst.n_head_kv = src.n_head_kv;
    dst.n_embd_head = src.n_embd_head;
    dst.type_k = src.type_k;
    dst.type_v = src.type_v;
    dst.v_trans = 0;
    dst.n_stream = 1;

    ik_kv_compat_plan_t plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(plan.is_compatible);

    std::vector<uint8_t> out(payload.size(), 0);
    ik_kv_convert_ctx_t cvt = {};
    cvt.src = &src;
    cvt.dst = &dst;
    cvt.plan = &plan;
    cvt.output_buf = out.data();
    cvt.output_size = out.size();

    result = ik_kv_convert_prefill_to_ik_seq_blob(&cvt);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(cvt.bytes_written, payload.size() - 8);

    // Expect: [cell_count][pos][n_seq_id=0] + trailing data copied unchanged.
    ASSERT_EQ(read_u32_le(out.data() + 0), 1);
    ASSERT_EQ(read_u32_le(out.data() + 4), 0);
    ASSERT_EQ(read_u32_le(out.data() + 8), 0);
    ASSERT_EQ(read_u32_le(out.data() + 12), 0); // v_state
    ASSERT_EQ(read_u32_le(out.data() + 16), 1); // n_layer

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_015_rtx_vstate_unknown_does_not_force_vtrans) {
    ik_kv_bridge_init();

    std::vector<uint8_t> payload;
    append_u32_le(payload, 1); // n_stream
    append_u32_le(payload, 1); // cell_count
    append_u32_le(payload, 0); // pos
    append_u32_le(payload, 1); // n_seq_id
    append_u32_le(payload, 0); // seq_id
    append_u32_le(payload, 2); // v_state=2 (no V cache)
    append_u32_le(payload, 1); // n_layer
    append_u32_le(payload, 1); // K type F16
    append_u64_le(payload, 2); // K row bytes
    payload.push_back(0x11);
    payload.push_back(0x22);

    const std::vector<uint8_t> artifact = build_rtx_artifact(48, payload, /*token_count=*/9);

    ik_kva_header_t header = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_UNKNOWN;
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact.data(), artifact.size(), &header, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_source_descriptor_t src = {};
    result = ik_kv_source_parse_prefill_seq_state(
        &header, artifact.data() + 48, payload.size(), &src, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(src.v_trans, 0xFF);

    ik_kv_dest_descriptor_t dst = {};
    dst.n_layers = src.n_layers;
    dst.n_ctx = src.n_ctx;
    dst.type_k = src.type_k;
    dst.type_v = src.type_v;
    dst.v_trans = 1; // opposite value should be ignored when src is unknown
    dst.n_stream = 1;

    ik_kv_compat_plan_t plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(plan.is_compatible);

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_016_rtx_multistream_single_active_supported) {
    ik_kv_bridge_init();

    const std::vector<uint8_t> payload = build_rtx_payload_two_streams_one_active();
    const std::vector<uint8_t> artifact = build_rtx_artifact(48, payload, /*token_count=*/11);

    ik_kva_header_t header = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_UNKNOWN;
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact.data(), artifact.size(), &header, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_source_descriptor_t src = {};
    result = ik_kv_source_parse_prefill_seq_state(
        &header, artifact.data() + 48, payload.size(), &src, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(src.n_stream, 1); // effectively single stream after pruning empties

    ik_kv_dest_descriptor_t dst = {};
    dst.n_layers = src.n_layers;
    dst.n_ctx = src.n_ctx;
    dst.type_k = src.type_k;
    dst.type_v = src.type_v;
    dst.v_trans = 0;
    dst.n_stream = 1;

    ik_kv_compat_plan_t plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(plan.is_compatible);

    std::vector<uint8_t> out(payload.size(), 0);
    ik_kv_convert_ctx_t cvt = {};
    cvt.src = &src;
    cvt.dst = &dst;
    cvt.plan = &plan;
    cvt.output_buf = out.data();
    cvt.output_size = out.size();

    result = ik_kv_convert_prefill_to_ik_seq_blob(&cvt);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(read_u32_le(out.data()), 1);
    ASSERT_EQ(read_u32_le(out.data() + 8), 0); // rewritten n_seq_id

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_017_rtx_multistream_multiple_active_merged) {
    ik_kv_bridge_init();

    const std::vector<uint8_t> payload = build_rtx_payload_two_streams_two_active();
    const std::vector<uint8_t> artifact = build_rtx_artifact(48, payload, /*token_count=*/12);

    ik_kva_header_t header = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_UNKNOWN;
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact.data(), artifact.size(), &header, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_source_descriptor_t src = {};
    result = ik_kv_source_parse_prefill_seq_state(
        &header, artifact.data() + 48, payload.size(), &src, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(src.n_stream, 1); // merged to single IK stream for planning

    ik_kv_dest_descriptor_t dst = {};
    dst.n_layers = src.n_layers;
    dst.n_ctx = src.n_ctx;
    dst.type_k = src.type_k;
    dst.type_v = src.type_v;
    dst.v_trans = 0;
    dst.n_stream = 1;

    ik_kv_compat_plan_t plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(plan.is_compatible);

    std::vector<uint8_t> out(payload.size(), 0);
    ik_kv_convert_ctx_t cvt = {};
    cvt.src = &src;
    cvt.dst = &dst;
    cvt.plan = &plan;
    cvt.output_buf = out.data();
    cvt.output_size = out.size();

    result = ik_kv_convert_prefill_to_ik_seq_blob(&cvt);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(cvt.bytes_written, (size_t) 60);
    ASSERT_EQ(read_u32_le(out.data()), 2);      // merged cell_count
    ASSERT_EQ(read_u32_le(out.data() + 8), 0);  // rewritten n_seq_id for cell 0
    ASSERT_EQ(read_u32_le(out.data() + 16), 0); // rewritten n_seq_id for cell 1

    // K payload is concatenated stream order: [11,22] + [11,22]
    ASSERT_EQ(out[40], 0x11);
    ASSERT_EQ(out[41], 0x22);
    ASSERT_EQ(out[42], 0x11);
    ASSERT_EQ(out[43], 0x22);
    // V payload is concatenated stream order: [33,44] + [33,44]
    ASSERT_EQ(out[56], 0x33);
    ASSERT_EQ(out[57], 0x44);
    ASSERT_EQ(out[58], 0x33);
    ASSERT_EQ(out[59], 0x44);

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_018_rtx_output_size_tracks_converted_payload) {
    ik_kv_bridge_init();

    const std::vector<uint8_t> payload = build_rtx_payload_single_stream();
    const std::vector<uint8_t> artifact = build_rtx_artifact(48, payload, /*token_count=*/13);

    ik_kva_header_t header = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_UNKNOWN;
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact.data(), artifact.size(), &header, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_source_descriptor_t src = {};
    result = ik_kv_source_parse_prefill_seq_state(
        &header, artifact.data() + 48, payload.size(), &src, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(src.source_format, IK_KV_SOURCE_FORMAT_RTX_KVARTIF1);

    ik_kv_dest_descriptor_t dst = {};
    dst.n_layers = src.n_layers;
    dst.n_ctx = src.n_ctx;
    dst.type_k = src.type_k;
    dst.type_v = src.type_v;
    dst.v_trans = 0;
    dst.n_stream = 1;

    const size_t out_size = ik_kv_convert_get_output_size(&src, &dst);
    ASSERT_EQ((int) out_size, (int) (payload.size() - 8));

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_019_ik_conversion_uses_plan_layer_mappings) {
    ik_kv_bridge_init();

    const std::vector<uint8_t> rtx_payload = build_rtx_payload_single_stream();
    const std::vector<uint8_t> artifact = build_rtx_artifact(48, rtx_payload, /*token_count=*/17);

    ik_kva_header_t header = {};
    ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_UNKNOWN;
    ik_kv_compat_convert_result_t result = ik_kv_source_parse_kva_header(
        artifact.data(), artifact.size(), &header, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_source_descriptor_t rtx_src = {};
    result = ik_kv_source_parse_prefill_seq_state(
        &header, artifact.data() + 48, rtx_payload.size(), &rtx_src, &reject);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_dest_descriptor_t dst = {};
    dst.n_layers = rtx_src.n_layers;
    dst.n_ctx = rtx_src.n_ctx;
    dst.type_k = rtx_src.type_k;
    dst.type_v = rtx_src.type_v;
    dst.v_trans = 0;
    dst.n_stream = 1;

    ik_kv_compat_plan_t rtx_plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&rtx_src, &dst, &rtx_plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(rtx_plan.is_compatible);

    std::vector<uint8_t> ik_payload(rtx_payload.size(), 0);
    ik_kv_convert_ctx_t rtx_cvt = {};
    rtx_cvt.src = &rtx_src;
    rtx_cvt.dst = &dst;
    rtx_cvt.plan = &rtx_plan;
    rtx_cvt.output_buf = ik_payload.data();
    rtx_cvt.output_size = ik_payload.size();
    result = ik_kv_convert_prefill_to_ik_seq_blob(&rtx_cvt);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ik_payload.resize(rtx_cvt.bytes_written);

    ik_kv_source_descriptor_t ik_src = {};
    ik_src.n_layers = 1;
    ik_src.n_ctx = rtx_src.n_ctx;
    ik_src.n_head_kv = rtx_src.n_head_kv;
    ik_src.type_k = rtx_src.type_k;
    ik_src.type_v = rtx_src.type_v;
    ik_src.v_trans = 0;
    ik_src.n_stream = 1;
    ik_src.source_format = IK_KV_SOURCE_FORMAT_IK_KVA;
    ik_src.payload_size = (uint64_t) ik_payload.size();
    ik_src.payload = ik_payload.data();
    memset(ik_src.model_fingerprint, 0xCD, sizeof(ik_src.model_fingerprint));

    ik_kv_dest_descriptor_t ik_dst = {};
    ik_dst.n_layers = 1;
    ik_dst.n_ctx = ik_src.n_ctx;
    ik_dst.n_head_kv = ik_src.n_head_kv;
    ik_dst.type_k = ik_src.type_k;
    ik_dst.type_v = ik_src.type_v;
    ik_dst.v_trans = ik_src.v_trans;
    ik_dst.n_stream = 1;
    memcpy(ik_dst.model_fingerprint, ik_src.model_fingerprint, sizeof(ik_dst.model_fingerprint));

    ik_kv_compat_plan_t ik_plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&ik_src, &ik_dst, &ik_plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(ik_plan.is_compatible);
    ASSERT_EQ((int) ik_plan.n_layers, 1);
    ASSERT_TRUE(ik_plan.layer_mappings[0].k_row_size > 0);
    ASSERT_TRUE(ik_plan.layer_mappings[0].v_row_size > 0);

    const ik_kv_layer_mapping_t base_map = ik_plan.layer_mappings[0];
    ASSERT_TRUE(base_map.k_row_size + base_map.k_offset <= ik_payload.size());
    ASSERT_TRUE(base_map.v_row_size + base_map.v_offset <= ik_payload.size());

    // Swap destination K/V placements to verify conversion applies plan mappings.
    ik_plan.layer_mappings[0].dst_k_offset = base_map.v_offset;
    ik_plan.layer_mappings[0].dst_v_offset = base_map.k_offset;

    std::vector<uint8_t> expected = ik_payload;
    std::vector<uint8_t> src_k(ik_payload.begin() + base_map.k_offset,
                               ik_payload.begin() + base_map.k_offset + base_map.k_row_size);
    std::vector<uint8_t> src_v(ik_payload.begin() + base_map.v_offset,
                               ik_payload.begin() + base_map.v_offset + base_map.v_row_size);
    memcpy(expected.data() + ik_plan.layer_mappings[0].dst_k_offset, src_k.data(), src_k.size());
    memcpy(expected.data() + ik_plan.layer_mappings[0].dst_v_offset, src_v.data(), src_v.size());

    std::vector<uint8_t> remapped(ik_payload.size(), 0);
    ik_kv_convert_ctx_t ik_cvt = {};
    ik_cvt.src = &ik_src;
    ik_cvt.dst = &ik_dst;
    ik_cvt.plan = &ik_plan;
    ik_cvt.output_buf = remapped.data();
    ik_cvt.output_size = remapped.size();
    result = ik_kv_convert_prefill_to_ik_seq_blob(&ik_cvt);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ((int) ik_cvt.bytes_written, (int) remapped.size());
    ASSERT_EQ(memcmp(remapped.data(), expected.data(), remapped.size()), 0);

    ik_kv_bridge_shutdown();
    tests_passed++;
}

//
// Additional basic tests for plan building
//

TEST(kvb_ut_030_plan_key_deterministic) {
    ik_kv_bridge_init();
    
    // Create two identical source descriptors
    ik_kv_source_descriptor_t src1, src2;
    memset(&src1, 0, sizeof(src1));
    memset(&src2, 0, sizeof(src2));
    
    src1.n_layers = 32;
    src1.n_ctx = 4096;
    src1.n_head_kv = 32;
    src1.type_k = IK_KV_TYPE_F16;
    src1.type_v = IK_KV_TYPE_F16;
    src1.v_trans = 0;
    src1.n_stream = 1;
    memset(src1.model_fingerprint, 0xAB, 32);
    
    memcpy(&src2, &src1, sizeof(src2));
    
    // Create destination descriptor
    ik_kv_dest_descriptor_t dst;
    memset(&dst, 0, sizeof(dst));
    dst.n_layers = 32;
    dst.n_ctx = 4096;
    dst.n_head_kv = 32;
    dst.type_k = IK_KV_TYPE_F16;
    dst.type_v = IK_KV_TYPE_F16;
    dst.v_trans = 0;
    dst.n_stream = 1;
    memset(dst.model_fingerprint, 0xAB, 32);
    
    // Build keys
    ik_kv_compat_plan_key_t key1, key2;
    ik_kv_compat_convert_result_t result;
    
    result = ik_kv_compat_plan_key_build(&src1, &dst, &key1);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    
    result = ik_kv_compat_plan_key_build(&src2, &dst, &key2);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    
    // Keys should be identical
    ASSERT_EQ(memcmp(&key1, &key2, sizeof(key1)), 0);
    
    ik_kv_bridge_shutdown();
    tests_passed++;
}

//
// Strict profile tests (KVB-UT-040, KVB-UT-041, KVB-UT-042)
//

TEST(kvb_ut_040_strict_profile_accepts_compatible) {
    ik_kv_bridge_init();
    
    // Create compatible source and destination
    ik_kv_source_descriptor_t src;
    ik_kv_dest_descriptor_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    
    src.n_layers = dst.n_layers = 32;
    src.n_ctx = dst.n_ctx = 4096;
    src.n_head_kv = dst.n_head_kv = 32;
    src.type_k = dst.type_k = IK_KV_TYPE_F16;
    src.type_v = dst.type_v = IK_KV_TYPE_F16;
    src.v_trans = dst.v_trans = 0;
    src.n_stream = dst.n_stream = 1;
    memset(src.model_fingerprint, 0xAB, 32);
    memcpy(dst.model_fingerprint, src.model_fingerprint, 32);
    
    ik_kv_compat_plan_t plan;
    ik_kv_compat_convert_result_t result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(plan.is_compatible);
    ASSERT_EQ(plan.reject_reason, IK_KV_COMPAT_REJECT_NONE);
    
    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_041_strict_profile_rejects_dtype_mismatch) {
    ik_kv_bridge_init();
    
    // Create source and destination with different type_k
    ik_kv_source_descriptor_t src;
    ik_kv_dest_descriptor_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    
    src.n_layers = dst.n_layers = 32;
    src.n_ctx = dst.n_ctx = 4096;
    src.n_head_kv = dst.n_head_kv = 32;
    src.type_k = IK_KV_TYPE_F16;
    dst.type_k = IK_KV_TYPE_Q8_0;  // Mismatch!
    src.type_v = dst.type_v = IK_KV_TYPE_F16;
    src.v_trans = dst.v_trans = 0;
    src.n_stream = dst.n_stream = 1;
    memset(src.model_fingerprint, 0xAB, 32);
    memcpy(dst.model_fingerprint, src.model_fingerprint, 32);
    
    ik_kv_compat_plan_t plan;
    ik_kv_compat_convert_result_t result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_FALSE(plan.is_compatible);
    ASSERT_EQ(plan.reject_reason, IK_KV_COMPAT_REJECT_TYPE_K_MISMATCH);
    
    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_042_strict_profile_rejects_nstream_not_one) {
    ik_kv_bridge_init();
    
    // Create source with n_stream != 1
    ik_kv_source_descriptor_t src;
    ik_kv_dest_descriptor_t dst;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    
    src.n_layers = dst.n_layers = 32;
    src.n_ctx = dst.n_ctx = 4096;
    src.n_head_kv = dst.n_head_kv = 32;
    src.type_k = dst.type_k = IK_KV_TYPE_F16;
    src.type_v = dst.type_v = IK_KV_TYPE_F16;
    src.v_trans = dst.v_trans = 0;
    src.n_stream = 2;  // Not 1!
    dst.n_stream = 1;
    memset(src.model_fingerprint, 0xAB, 32);
    memcpy(dst.model_fingerprint, src.model_fingerprint, 32);
    
    ik_kv_compat_plan_t plan;
    ik_kv_compat_convert_result_t result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_FALSE(plan.is_compatible);
    ASSERT_EQ(plan.reject_reason, IK_KV_COMPAT_REJECT_N_STREAM_UNSUPPORTED);
    
    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_050_plan_cache_load_store_hit_path) {
    ik_kv_bridge_init();

    const std::string cache_dir = make_test_cache_dir("hit");
    fs::create_directories(cache_dir);

    ik_kv_bridge_config_t cfg;
    ik_kv_bridge_get_config(&cfg);
    cfg.plan_cache_dir = cache_dir.c_str();
    ik_kv_bridge_set_config(&cfg);

    ik_kv_source_descriptor_t src;
    ik_kv_dest_descriptor_t dst;
    build_compatible_src_dst(src, dst);

    ik_kv_compat_plan_key_t key = {};
    ik_kv_compat_convert_result_t result = ik_kv_compat_plan_key_build(&src, &dst, &key);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_compat_plan_t plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(plan.is_compatible);

    result = ik_kv_plan_cache_store(&key, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_compat_plan_t loaded = {};
    result = ik_kv_plan_cache_load(&key, &loaded);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_EQ(memcmp(&plan, &loaded, sizeof(plan)), 0);

    ik_kv_plan_cache_clear();
    std::error_code ec;
    fs::remove_all(cache_dir, ec);
    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_051_plan_cache_invalidate) {
    ik_kv_bridge_init();

    const std::string cache_dir = make_test_cache_dir("invalidate");
    fs::create_directories(cache_dir);

    ik_kv_bridge_config_t cfg;
    ik_kv_bridge_get_config(&cfg);
    cfg.plan_cache_dir = cache_dir.c_str();
    ik_kv_bridge_set_config(&cfg);

    ik_kv_source_descriptor_t src;
    ik_kv_dest_descriptor_t dst;
    build_compatible_src_dst(src, dst);

    ik_kv_compat_plan_key_t key = {};
    ik_kv_compat_convert_result_t result = ik_kv_compat_plan_key_build(&src, &dst, &key);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_compat_plan_t plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    result = ik_kv_plan_cache_store(&key, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    result = ik_kv_plan_cache_invalidate(&key);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_compat_plan_t loaded = {};
    result = ik_kv_plan_cache_load(&key, &loaded);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD);

    ik_kv_plan_cache_clear();
    std::error_code ec;
    fs::remove_all(cache_dir, ec);
    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_052_plan_cache_rejects_corrupted_file) {
    ik_kv_bridge_init();

    const std::string cache_dir = make_test_cache_dir("corrupt");
    fs::create_directories(cache_dir);

    ik_kv_bridge_config_t cfg;
    ik_kv_bridge_get_config(&cfg);
    cfg.plan_cache_dir = cache_dir.c_str();
    ik_kv_bridge_set_config(&cfg);

    ik_kv_source_descriptor_t src;
    ik_kv_dest_descriptor_t dst;
    build_compatible_src_dst(src, dst);

    ik_kv_compat_plan_key_t key = {};
    ik_kv_compat_convert_result_t result = ik_kv_compat_plan_key_build(&src, &dst, &key);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    ik_kv_compat_plan_t plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src, &dst, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    result = ik_kv_plan_cache_store(&key, &plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);

    fs::path cache_file;
    for (const auto & entry : fs::directory_iterator(cache_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ikvc") {
            cache_file = entry.path();
            break;
        }
    }
    ASSERT_FALSE(cache_file.empty());

    {
        std::vector<uint8_t> blob;
        std::ifstream ifs(cache_file, std::ios::binary);
        ASSERT_TRUE(ifs.is_open());
        ifs.seekg(0, std::ios::end);
        const std::streamoff n = ifs.tellg();
        ASSERT_TRUE(n > 0);
        ifs.seekg(0, std::ios::beg);
        blob.resize((size_t) n);
        ifs.read((char *) blob.data(), (std::streamsize) blob.size());
        ASSERT_TRUE(ifs.good() || ifs.eof());
        blob[blob.size() - 1] ^= 0x5Au;

        std::ofstream ofs(cache_file, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(ofs.is_open());
        ofs.write((const char *) blob.data(), (std::streamsize) blob.size());
        ASSERT_TRUE(ofs.good());
    }

    ik_kv_compat_plan_t loaded = {};
    result = ik_kv_plan_cache_load(&key, &loaded);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD);

    ik_kv_plan_cache_clear();
    std::error_code ec;
    fs::remove_all(cache_dir, ec);
    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_070_bridge_config_roundtrip) {
    ik_kv_bridge_init();

    ik_kv_bridge_config_t cfg = {};
    cfg.mode = IK_KV_BRIDGE_MODE_RELAXED;
    cfg.plan_cache_dir = "/tmp/ik_kv_bridge_cfg_roundtrip";
    cfg.allow_vtrans_convert = true;
    cfg.dry_run = true;
    cfg.no_fallback = true;
    ik_kv_bridge_set_config(&cfg);

    ik_kv_bridge_config_t out = {};
    ik_kv_bridge_get_config(&out);
    ASSERT_EQ(out.mode, IK_KV_BRIDGE_MODE_RELAXED);
    ASSERT_TRUE(out.plan_cache_dir != nullptr);
    ASSERT_STR_EQ(out.plan_cache_dir, "/tmp/ik_kv_bridge_cfg_roundtrip");
    ASSERT_TRUE(out.allow_vtrans_convert);
    ASSERT_TRUE(out.dry_run);
    ASSERT_TRUE(out.no_fallback);

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_080_bridge_metrics_reset_zeroes_state) {
    ik_kv_bridge_init();

    ik_kv_bridge_reset_metrics();
    ik_kv_bridge_metrics_t metrics = {};
    ik_kv_bridge_get_last_metrics(&metrics);

    ASSERT_EQ(metrics.plan_key, 0);
    ASSERT_EQ(metrics.plan_cache_hit, false);
    ASSERT_EQ(metrics.convert_us, 0);
    ASSERT_EQ(metrics.bytes_in, 0);
    ASSERT_EQ(metrics.bytes_out, 0);
    ASSERT_EQ(metrics.mode, 0);
    ASSERT_EQ(metrics.status, 0);
    ASSERT_EQ(metrics.reject_reason, 0);

    ik_kv_bridge_shutdown();
    tests_passed++;
}

TEST(kvb_ut_090_relaxed_vtrans_plan_rebuild_accepts_profile) {
    ik_kv_bridge_init();

    ik_kv_source_descriptor_t src;
    ik_kv_dest_descriptor_t dst;
    build_compatible_src_dst(src, dst);

    src.v_trans = 1;
    dst.v_trans = 0;

    ik_kv_compat_plan_t strict_plan = {};
    ik_kv_compat_convert_result_t result =
        ik_kv_compat_plan_build_strict_v1(&src, &dst, &strict_plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_FALSE(strict_plan.is_compatible);
    ASSERT_EQ(strict_plan.reject_reason, IK_KV_COMPAT_REJECT_VTRANS_MISMATCH);

    ik_kv_source_descriptor_t src_relaxed = src;
    src_relaxed.v_trans = dst.v_trans;
    ik_kv_compat_plan_t relaxed_plan = {};
    result = ik_kv_compat_plan_build_strict_v1(&src_relaxed, &dst, &relaxed_plan);
    ASSERT_EQ(result, IK_KV_COMPAT_CONVERT_OK);
    ASSERT_TRUE(relaxed_plan.is_compatible);
    ASSERT_EQ(relaxed_plan.reject_reason, IK_KV_COMPAT_REJECT_NONE);

    ik_kv_bridge_shutdown();
    tests_passed++;
}

//
// Main test runner
//

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;
    
    printf("=== KV Bridge Unit Tests ===\n\n");
    
    // Run all tests
    test_kvb_ut_001_module_symbols_link();
    test_kvb_ut_010_valid_kva_header_parse();
    test_kvb_ut_011_invalid_header_reject();
    test_kvb_ut_012_payload_crc_mismatch_reject();
    test_kvb_ut_013_valid_rtx_kvartif1_header_parse();
    test_kvb_ut_014_rtx_convert_to_ik_seq_blob_rewrites_meta();
    test_kvb_ut_015_rtx_vstate_unknown_does_not_force_vtrans();
    test_kvb_ut_016_rtx_multistream_single_active_supported();
    test_kvb_ut_017_rtx_multistream_multiple_active_merged();
    test_kvb_ut_018_rtx_output_size_tracks_converted_payload();
    test_kvb_ut_019_ik_conversion_uses_plan_layer_mappings();
    test_kvb_ut_030_plan_key_deterministic();
    test_kvb_ut_040_strict_profile_accepts_compatible();
    test_kvb_ut_041_strict_profile_rejects_dtype_mismatch();
    test_kvb_ut_042_strict_profile_rejects_nstream_not_one();
    test_kvb_ut_050_plan_cache_load_store_hit_path();
    test_kvb_ut_051_plan_cache_invalidate();
    test_kvb_ut_052_plan_cache_rejects_corrupted_file();
    test_kvb_ut_070_bridge_config_roundtrip();
    test_kvb_ut_080_bridge_metrics_reset_zeroes_state();
    test_kvb_ut_090_relaxed_vtrans_plan_rebuild_accepts_profile();
    
    printf("\n=== Results ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
