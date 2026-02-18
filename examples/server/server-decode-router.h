#pragma once

#include "server-handoff-metadata.h"

#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

struct server_decode_cluster_node {
    std::string node_id;
    std::string kv_host;
    int32_t kv_port = 0;
    std::string rpc_endpoint;
    std::string role = "worker";
    int32_t weight = 1;
    bool healthy = true;
    bool promotable = false;
};

struct server_decode_route_assignment {
    std::string owner;
    int32_t layer_start = 0; // inclusive
    int32_t layer_end = 0;   // exclusive
};

struct server_decode_route_plan {
    std::string route_plan_id;
    std::string topology_epoch = "epoch0";
    std::string failover_policy = "reroute";
    std::vector<server_decode_route_assignment> assignments;
    std::vector<std::string> required_nodes;
    bool fallback_applied = false;
    std::string fallback_reason;
};

bool server_decode_cluster_nodes_from_json(
    const nlohmann::json & json_in,
    std::vector<server_decode_cluster_node> * out,
    std::string * error);

bool server_decode_cluster_nodes_load_file(
    const std::string & path,
    std::vector<server_decode_cluster_node> * out,
    std::string * error);

bool server_decode_router_build_plan(
    const server_handoff_metadata_v2 & handoff,
    const std::vector<server_decode_cluster_node> & nodes,
    const std::string & local_node_id,
    server_decode_route_plan * out,
    std::string * error);

nlohmann::ordered_json server_decode_route_plan_to_json(
    const server_decode_route_plan & plan);
