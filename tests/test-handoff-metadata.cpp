#include "server-handoff-metadata.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    std::printf("Running %s... ", #name); \
    test_##name(); \
    if (tests_failed == 0) { \
        std::printf("PASSED\n"); \
    } \
} while (0)

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        std::printf("FAILED at line %d: expected true\n", __LINE__); \
        tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        std::printf("FAILED at line %d: expected equality\n", __LINE__); \
        tests_failed++; \
        return; \
    } \
} while (0)

#define ASSERT_STR_EQ(a, b) do { \
    if (std::strcmp((a), (b)) != 0) { \
        std::printf("FAILED at line %d: \"%s\" != \"%s\"\n", __LINE__, (a), (b)); \
        tests_failed++; \
        return; \
    } \
} while (0)

TEST(layer_map_roundtrip) {
    std::vector<server_handoff_layer_span> spans;
    std::string error;
    ASSERT_TRUE(server_handoff_parse_layer_map_legacy(
        "local_gpu:0-16|tb_remote_0:16-32", &spans, &error));
    ASSERT_EQ((int) spans.size(), 2);
    ASSERT_STR_EQ(spans[0].node.c_str(), "local_gpu");
    ASSERT_EQ(spans[0].layer_start, 0);
    ASSERT_EQ(spans[0].layer_end, 16);
    ASSERT_STR_EQ(spans[1].node.c_str(), "tb_remote_0");
    ASSERT_EQ(spans[1].layer_start, 16);
    ASSERT_EQ(spans[1].layer_end, 32);

    const std::string serialized = server_handoff_serialize_layer_map_legacy(spans);
    ASSERT_STR_EQ(serialized.c_str(), "local_gpu:0-16|tb_remote_0:16-32");
    tests_passed++;
}

TEST(invalid_legacy_layer_map_rejected) {
    std::vector<server_handoff_layer_span> spans;
    std::string error;
    ASSERT_FALSE(server_handoff_parse_layer_map_legacy("badtoken", &spans, &error));
    ASSERT_TRUE(!error.empty());
    tests_passed++;
}

TEST(legacy_upgrade_generates_fallback_map) {
    server_handoff_legacy_fields legacy;
    legacy.remote_nodes = 2;
    legacy.expected_gpu_layers = 12;
    legacy.expected_remote_layers = 20;
    legacy.remote_failover_policy = "reroute";
    legacy.layer_map = "";
    legacy.handoff_session_id = "hs_1";
    legacy.topology_epoch = "epoch7";
    legacy.transport_mode = "rdma";

    server_handoff_metadata_v2 upgraded;
    std::string error;
    ASSERT_TRUE(server_handoff_metadata_v2_from_legacy(legacy, &upgraded, &error));
    ASSERT_EQ((int) upgraded.layer_map.size(), 2);
    ASSERT_STR_EQ(upgraded.layer_map[0].node.c_str(), "local_gpu");
    ASSERT_EQ(upgraded.layer_map[0].layer_start, 0);
    ASSERT_EQ(upgraded.layer_map[0].layer_end, 12);
    ASSERT_STR_EQ(upgraded.layer_map[1].node.c_str(), "tb_remote_0");
    ASSERT_EQ(upgraded.layer_map[1].layer_start, 12);
    ASSERT_EQ(upgraded.layer_map[1].layer_end, 32);
    tests_passed++;
}

TEST(v2_json_parse_and_validate) {
    const nlohmann::json payload = {
        {"schema", "prefill_handoff_v2"},
        {"schema_version", 1},
        {"handoff_session_id", "hs_2"},
        {"topology_epoch", "epoch9"},
        {"remote_nodes", 2},
        {"expected_gpu_layers", 10},
        {"expected_remote_layers", 22},
        {"execution_mode", "decoupled"},
        {"transport_mode", "rdma"},
        {"remote_failover_policy", "reroute"},
        {"balance", "roundrobin"},
        {"remote_ranges", "tb_remote_0:10-22,tb_remote_1:22-32"},
        {"artifact_crc32", 99},
        {"layer_map", {
            {{"node", "local_gpu"}, {"layer_start", 0}, {"layer_end", 10}},
            {{"node", "tb_remote_0"}, {"layer_start", 10}, {"layer_end", 22}},
            {{"node", "tb_remote_1"}, {"layer_start", 22}, {"layer_end", 32}},
        }},
        {"remote_node_descriptors", {
            {{"node_id", "tb_remote_0"}, {"kv_host", "10.0.0.2"}, {"kv_port", 19001}, {"healthy", true}, {"weight", 2}},
            {{"node_id", "tb_remote_1"}, {"kv_host", "10.0.0.3"}, {"kv_port", 19002}, {"healthy", true}, {"weight", 1}},
        }},
    };

    server_handoff_metadata_v2 parsed;
    std::string error;
    ASSERT_TRUE(server_handoff_metadata_v2_from_json(payload, &parsed, &error));
    ASSERT_EQ((int) parsed.layer_map.size(), 3);
    ASSERT_EQ((int) parsed.remote_node_descriptors.size(), 2);
    ASSERT_TRUE(server_handoff_metadata_v2_validate(parsed, &error));

    const nlohmann::ordered_json out = server_handoff_metadata_v2_to_json(parsed);
    ASSERT_TRUE(out.is_object());
    ASSERT_EQ((int) out.at("remote_nodes"), 2);
    tests_passed++;
}

int main() {
    RUN_TEST(layer_map_roundtrip);
    RUN_TEST(invalid_legacy_layer_map_rejected);
    RUN_TEST(legacy_upgrade_generates_fallback_map);
    RUN_TEST(v2_json_parse_and_validate);

    if (tests_failed > 0) {
        std::printf("FAILED: %d tests failed, %d passed\n", tests_failed, tests_passed);
        return 1;
    }
    std::printf("All tests passed (%d)\n", tests_passed);
    return 0;
}
