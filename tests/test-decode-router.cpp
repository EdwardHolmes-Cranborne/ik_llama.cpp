#include "server-decode-router.h"

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

static server_handoff_metadata_v2 make_handoff(const char * policy) {
    server_handoff_metadata_v2 handoff;
    handoff.schema = "prefill_handoff_v2";
    handoff.schema_version = 1;
    handoff.handoff_session_id = "sess_route";
    handoff.topology_epoch = "epoch1";
    handoff.remote_nodes = 1;
    handoff.expected_gpu_layers = 10;
    handoff.expected_remote_layers = 22;
    handoff.remote_failover_policy = policy;
    handoff.layer_map = {
        server_handoff_layer_span{"local_gpu", 0, 10},
        server_handoff_layer_span{"tb_remote_0", 10, 32},
    };
    return handoff;
}

TEST(route_healthy_remote_no_fallback) {
    const server_handoff_metadata_v2 handoff = make_handoff("reroute");
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, true, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_FALSE(plan.fallback_applied);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[0].owner.c_str(), "local_gpu");
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "tb_remote_0");
    ASSERT_EQ((int) plan.required_nodes.size(), 1);
    ASSERT_STR_EQ(plan.required_nodes[0].c_str(), "tb_remote_0");
    tests_passed++;
}

TEST(cluster_nodes_json_parse) {
    const nlohmann::json payload = {
        {"nodes", {
            {{"node_id", "tb_remote_0"}, {"kv_host", "10.0.0.2"}, {"kv_port", 19001}, {"healthy", true}, {"weight", 2}},
            {{"node_id", "tb_remote_1"}, {"kv_host", "10.0.0.3"}, {"kv_port", 19002}, {"healthy", false}, {"weight", 1}},
        }},
    };

    std::vector<server_decode_cluster_node> nodes;
    std::string error;
    ASSERT_TRUE(server_decode_cluster_nodes_from_json(payload, &nodes, &error));
    ASSERT_EQ((int) nodes.size(), 2);
    ASSERT_STR_EQ(nodes[0].node_id.c_str(), "tb_remote_0");
    ASSERT_EQ(nodes[0].weight, 2);
    ASSERT_FALSE(nodes[1].healthy);
    tests_passed++;
}

TEST(cluster_nodes_json_invalid_rejected) {
    const nlohmann::json payload = {
        {"nodes", {123}},
    };

    std::vector<server_decode_cluster_node> nodes;
    std::string error;
    ASSERT_FALSE(server_decode_cluster_nodes_from_json(payload, &nodes, &error));
    ASSERT_TRUE(!error.empty());
    tests_passed++;
}

TEST(route_unhealthy_reroutes_to_healthy_peer) {
    const server_handoff_metadata_v2 handoff = make_handoff("reroute");
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, false, false},
        server_decode_cluster_node{"tb_remote_1", "10.0.0.3", 19001, "", "worker", 1, true, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(plan.fallback_applied);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "tb_remote_1");
    ASSERT_EQ((int) plan.required_nodes.size(), 1);
    ASSERT_STR_EQ(plan.required_nodes[0].c_str(), "tb_remote_1");
    tests_passed++;
}

TEST(route_unhealthy_local_failover) {
    const server_handoff_metadata_v2 handoff = make_handoff("local");
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, false, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(plan.fallback_applied);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "local_cpu");
    ASSERT_EQ((int) plan.required_nodes.size(), 0);
    tests_passed++;
}

TEST(route_unhealthy_fail_policy_errors) {
    const server_handoff_metadata_v2 handoff = make_handoff("fail");
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, false, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_FALSE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(!error.empty());
    tests_passed++;
}

TEST(route_local_node_id_not_required_remote) {
    const server_handoff_metadata_v2 handoff = make_handoff("reroute");
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, true, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "tb_remote_0", &plan, &error));
    ASSERT_FALSE(plan.fallback_applied);
    ASSERT_EQ((int) plan.required_nodes.size(), 0);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "tb_remote_0");
    tests_passed++;
}

TEST(route_node_id_normalization_matches_cluster_entry) {
    server_handoff_metadata_v2 handoff = make_handoff("reroute");
    handoff.layer_map = {
        server_handoff_layer_span{"local_gpu", 0, 10},
        server_handoff_layer_span{"  TB_Remote_0  ", 10, 32},
    };
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, true, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_FALSE(plan.fallback_applied);
    ASSERT_EQ((int) plan.required_nodes.size(), 1);
    ASSERT_STR_EQ(plan.required_nodes[0].c_str(), "tb_remote_0");
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "tb_remote_0");
    tests_passed++;
}

TEST(route_local_node_id_normalization_treats_assignment_as_local) {
    server_handoff_metadata_v2 handoff = make_handoff("reroute");
    handoff.layer_map = {
        server_handoff_layer_span{"local_gpu", 0, 10},
        server_handoff_layer_span{"TB_REMOTE_0", 10, 32},
    };
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, true, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "  tb_remote_0  ", &plan, &error));
    ASSERT_FALSE(plan.fallback_applied);
    ASSERT_EQ((int) plan.required_nodes.size(), 0);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "tb_remote_0");
    tests_passed++;
}

TEST(route_unhealthy_reroutes_to_promotable_when_no_primary) {
    const server_handoff_metadata_v2 handoff = make_handoff("reroute");
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, false, false},
        server_decode_cluster_node{"tb_remote_2", "10.0.0.4", 19003, "", "standby", 1, true, true},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(plan.fallback_applied);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "tb_remote_2");
    ASSERT_EQ((int) plan.required_nodes.size(), 1);
    ASSERT_STR_EQ(plan.required_nodes[0].c_str(), "tb_remote_2");
    tests_passed++;
}

TEST(route_unhealthy_standby_nonpromotable_degrades_local) {
    const server_handoff_metadata_v2 handoff = make_handoff("reroute");
    const std::vector<server_decode_cluster_node> nodes = {
        server_decode_cluster_node{"tb_remote_0", "10.0.0.2", 19001, "", "worker", 1, false, false},
        server_decode_cluster_node{"tb_remote_3", "10.0.0.5", 19004, "", "standby", 1, true, false},
    };

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(plan.fallback_applied);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "local_cpu");
    ASSERT_EQ((int) plan.required_nodes.size(), 0);
    tests_passed++;
}

TEST(route_unknown_node_empty_cluster_local_policy_degrades_local) {
    server_handoff_metadata_v2 handoff = make_handoff("local");
    handoff.layer_map = {
        server_handoff_layer_span{"local_gpu", 0, 10},
        server_handoff_layer_span{"tb_remote_9", 10, 32},
    };
    const std::vector<server_decode_cluster_node> nodes = {};

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(plan.fallback_applied);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "local_cpu");
    ASSERT_EQ((int) plan.required_nodes.size(), 0);
    tests_passed++;
}

TEST(route_unknown_node_empty_cluster_reroute_policy_degrades_local) {
    server_handoff_metadata_v2 handoff = make_handoff("reroute");
    handoff.layer_map = {
        server_handoff_layer_span{"local_gpu", 0, 10},
        server_handoff_layer_span{"tb_remote_9", 10, 32},
    };
    const std::vector<server_decode_cluster_node> nodes = {};

    server_decode_route_plan plan;
    std::string error;
    ASSERT_TRUE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(plan.fallback_applied);
    ASSERT_EQ((int) plan.assignments.size(), 2);
    ASSERT_STR_EQ(plan.assignments[1].owner.c_str(), "local_cpu");
    ASSERT_EQ((int) plan.required_nodes.size(), 0);
    tests_passed++;
}

TEST(route_unknown_node_empty_cluster_fail_policy_errors) {
    server_handoff_metadata_v2 handoff = make_handoff("fail");
    handoff.layer_map = {
        server_handoff_layer_span{"local_gpu", 0, 10},
        server_handoff_layer_span{"tb_remote_9", 10, 32},
    };
    const std::vector<server_decode_cluster_node> nodes = {};

    server_decode_route_plan plan;
    std::string error;
    ASSERT_FALSE(server_decode_router_build_plan(handoff, nodes, "", &plan, &error));
    ASSERT_TRUE(!error.empty());
    tests_passed++;
}

int main() {
    RUN_TEST(cluster_nodes_json_parse);
    RUN_TEST(cluster_nodes_json_invalid_rejected);
    RUN_TEST(route_healthy_remote_no_fallback);
    RUN_TEST(route_unhealthy_reroutes_to_healthy_peer);
    RUN_TEST(route_unhealthy_local_failover);
    RUN_TEST(route_unhealthy_fail_policy_errors);
    RUN_TEST(route_local_node_id_not_required_remote);
    RUN_TEST(route_node_id_normalization_matches_cluster_entry);
    RUN_TEST(route_local_node_id_normalization_treats_assignment_as_local);
    RUN_TEST(route_unhealthy_reroutes_to_promotable_when_no_primary);
    RUN_TEST(route_unhealthy_standby_nonpromotable_degrades_local);
    RUN_TEST(route_unknown_node_empty_cluster_local_policy_degrades_local);
    RUN_TEST(route_unknown_node_empty_cluster_reroute_policy_degrades_local);
    RUN_TEST(route_unknown_node_empty_cluster_fail_policy_errors);

    if (tests_failed > 0) {
        std::printf("FAILED: %d tests failed, %d passed\n", tests_failed, tests_passed);
        return 1;
    }
    std::printf("All tests passed (%d)\n", tests_passed);
    return 0;
}
