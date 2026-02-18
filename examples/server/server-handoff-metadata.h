#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

struct server_handoff_layer_span {
    std::string node;
    int32_t layer_start = 0; // inclusive
    int32_t layer_end = 0;   // exclusive
};

struct server_handoff_remote_node {
    std::string node_id;
    std::string kv_host;
    int32_t kv_port = 0;
    std::string rpc_endpoint;
    std::string role = "worker";
    int32_t weight = 1;
    bool healthy = true;
    bool promotable = false;
};

struct server_handoff_metadata_v2 {
    std::string schema = "prefill_handoff_v2";
    int32_t schema_version = 1;

    std::string handoff_session_id;
    std::string topology_epoch = "epoch0";

    int32_t remote_nodes = 1;
    int32_t expected_gpu_layers = 0;
    int32_t expected_remote_layers = 0;

    std::string execution_mode = "coupled";
    std::string transport_mode = "auto";
    std::string remote_failover_policy = "reroute";
    std::string balance = "roundrobin";
    std::string remote_ranges;
    uint32_t artifact_crc32 = 0;

    std::vector<server_handoff_layer_span> layer_map;
    std::vector<server_handoff_remote_node> remote_node_descriptors;
};

struct server_handoff_legacy_fields {
    int32_t remote_nodes = 1;
    int32_t expected_gpu_layers = 0;
    int32_t expected_remote_layers = 0;
    std::string execution_mode = "coupled";
    std::string balance = "roundrobin";
    std::string remote_ranges;
    std::string remote_failover_policy = "reroute";
    std::string layer_map;
    std::string handoff_session_id;
    std::string topology_epoch = "epoch0";
    uint32_t artifact_crc32 = 0;
    std::string transport_mode = "auto";
};

bool server_handoff_parse_layer_map_legacy(
    const std::string & raw,
    std::vector<server_handoff_layer_span> * out,
    std::string * error);

std::string server_handoff_serialize_layer_map_legacy(
    const std::vector<server_handoff_layer_span> & spans);

bool server_handoff_metadata_v2_validate(
    const server_handoff_metadata_v2 & metadata,
    std::string * error);

nlohmann::ordered_json server_handoff_metadata_v2_to_json(
    const server_handoff_metadata_v2 & metadata);

bool server_handoff_metadata_v2_from_json(
    const nlohmann::json & json_in,
    server_handoff_metadata_v2 * out,
    std::string * error);

bool server_handoff_metadata_v2_from_legacy(
    const server_handoff_legacy_fields & legacy,
    server_handoff_metadata_v2 * out,
    std::string * error);

bool server_handoff_metadata_v2_from_legacy_json(
    const nlohmann::json & legacy_json,
    server_handoff_metadata_v2 * out,
    std::string * error);
