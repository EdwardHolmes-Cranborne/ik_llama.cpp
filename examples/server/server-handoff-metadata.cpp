#include "server-handoff-metadata.h"

#include <algorithm>
#include <cctype>
#include <sstream>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

std::string trim_copy(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

bool parse_i32(const std::string & value, int32_t * out) {
    if (!out) {
        return false;
    }
    try {
        *out = (int32_t) std::stoi(value);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_u32(const std::string & value, uint32_t * out) {
    if (!out) {
        return false;
    }
    try {
        const unsigned long long v = std::stoull(value);
        if (v > 0xFFFFFFFFull) {
            return false;
        }
        *out = (uint32_t) v;
        return true;
    } catch (...) {
        return false;
    }
}

bool is_known_failover_policy(const std::string & raw) {
    const std::string value = lower_copy(trim_copy(raw));
    return value == "reroute" || value == "local" || value == "fail";
}

std::string normalize_failover_policy(const std::string & raw) {
    const std::string value = lower_copy(trim_copy(raw));
    if (is_known_failover_policy(value)) {
        return value;
    }
    return "reroute";
}

void append_map_coverage_spans(
    int32_t expected_gpu_layers,
    int32_t expected_remote_layers,
    std::vector<server_handoff_layer_span> * out) {
    if (!out) {
        return;
    }
    out->clear();

    const int32_t gpu_layers = std::max<int32_t>(0, expected_gpu_layers);
    const int32_t remote_layers = std::max<int32_t>(0, expected_remote_layers);

    int32_t cursor = 0;
    if (gpu_layers > 0) {
        out->push_back(server_handoff_layer_span{
            /*.node=*/"local_gpu",
            /*.layer_start=*/cursor,
            /*.layer_end=*/cursor + gpu_layers,
        });
        cursor += gpu_layers;
    }
    if (remote_layers > 0) {
        out->push_back(server_handoff_layer_span{
            /*.node=*/"tb_remote_0",
            /*.layer_start=*/cursor,
            /*.layer_end=*/cursor + remote_layers,
        });
    }
}

} // namespace

bool server_handoff_parse_layer_map_legacy(
    const std::string & raw,
    std::vector<server_handoff_layer_span> * out,
    std::string * error) {
    if (!out) {
        if (error) {
            *error = "null output span vector";
        }
        return false;
    }

    out->clear();
    const std::string compact = trim_copy(raw);
    if (compact.empty()) {
        return true;
    }

    std::stringstream ss(compact);
    std::string token;
    while (std::getline(ss, token, '|')) {
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }

        const size_t colon = token.find(':');
        const size_t dash = token.find('-', colon == std::string::npos ? 0 : colon + 1);
        if (colon == std::string::npos || dash == std::string::npos || dash <= colon + 1) {
            if (error) {
                *error = "invalid layer_map token '" + token + "' (expected node:start-end)";
            }
            return false;
        }

        const std::string node = trim_copy(token.substr(0, colon));
        const std::string start_s = trim_copy(token.substr(colon + 1, dash - (colon + 1)));
        const std::string end_s = trim_copy(token.substr(dash + 1));

        int32_t start = 0;
        int32_t end = 0;
        if (node.empty() || !parse_i32(start_s, &start) || !parse_i32(end_s, &end)) {
            if (error) {
                *error = "invalid layer_map token '" + token + "'";
            }
            return false;
        }
        if (start < 0 || end <= start) {
            if (error) {
                *error = "invalid layer bounds in token '" + token + "'";
            }
            return false;
        }

        out->push_back(server_handoff_layer_span{
            /*.node=*/node,
            /*.layer_start=*/start,
            /*.layer_end=*/end,
        });
    }

    return true;
}

std::string server_handoff_serialize_layer_map_legacy(
    const std::vector<server_handoff_layer_span> & spans) {
    std::ostringstream oss;
    for (size_t i = 0; i < spans.size(); ++i) {
        const auto & s = spans[i];
        if (i > 0) {
            oss << "|";
        }
        oss << s.node << ":" << s.layer_start << "-" << s.layer_end;
    }
    return oss.str();
}

bool server_handoff_metadata_v2_validate(
    const server_handoff_metadata_v2 & metadata,
    std::string * error) {
    if (metadata.schema != "prefill_handoff_v2") {
        if (error) {
            *error = "unsupported handoff schema '" + metadata.schema + "'";
        }
        return false;
    }
    if (metadata.schema_version != 1) {
        if (error) {
            *error = "unsupported handoff schema version " + std::to_string(metadata.schema_version);
        }
        return false;
    }
    if (metadata.remote_nodes < 1) {
        if (error) {
            *error = "remote_nodes must be >= 1";
        }
        return false;
    }
    if (metadata.expected_gpu_layers < 0 || metadata.expected_remote_layers < 0) {
        if (error) {
            *error = "expected layer counters must be non-negative";
        }
        return false;
    }
    if (!is_known_failover_policy(metadata.remote_failover_policy)) {
        if (error) {
            *error = "invalid failover policy '" + metadata.remote_failover_policy + "'";
        }
        return false;
    }

    int32_t prev_start = -1;
    for (const auto & span : metadata.layer_map) {
        if (span.node.empty()) {
            if (error) {
                *error = "layer_map contains empty node id";
            }
            return false;
        }
        if (span.layer_start < 0 || span.layer_end <= span.layer_start) {
            if (error) {
                *error = "layer_map contains invalid bounds for node '" + span.node + "'";
            }
            return false;
        }
        if (prev_start > span.layer_start) {
            if (error) {
                *error = "layer_map must be ordered by ascending layer_start";
            }
            return false;
        }
        prev_start = span.layer_start;
    }

    return true;
}

json server_handoff_metadata_v2_to_json(const server_handoff_metadata_v2 & metadata) {
    json layer_map = json::array();
    for (const auto & span : metadata.layer_map) {
        layer_map.push_back({
            {"node", span.node},
            {"layer_start", span.layer_start},
            {"layer_end", span.layer_end},
        });
    }

    json remote_nodes = json::array();
    for (const auto & node : metadata.remote_node_descriptors) {
        remote_nodes.push_back({
            {"node_id", node.node_id},
            {"kv_host", node.kv_host},
            {"kv_port", node.kv_port},
            {"rpc_endpoint", node.rpc_endpoint},
            {"role", node.role},
            {"weight", node.weight},
            {"healthy", node.healthy},
            {"promotable", node.promotable},
        });
    }

    return {
        {"schema", metadata.schema},
        {"schema_version", metadata.schema_version},
        {"handoff_session_id", metadata.handoff_session_id},
        {"topology_epoch", metadata.topology_epoch},
        {"remote_nodes", metadata.remote_nodes},
        {"expected_gpu_layers", metadata.expected_gpu_layers},
        {"expected_remote_layers", metadata.expected_remote_layers},
        {"execution_mode", metadata.execution_mode},
        {"transport_mode", metadata.transport_mode},
        {"remote_failover_policy", metadata.remote_failover_policy},
        {"balance", metadata.balance},
        {"remote_ranges", metadata.remote_ranges},
        {"artifact_crc32", metadata.artifact_crc32},
        {"layer_map", layer_map},
        {"remote_node_descriptors", remote_nodes},
    };
}

bool server_handoff_metadata_v2_from_json(
    const nlohmann::json & json_in,
    server_handoff_metadata_v2 * out,
    std::string * error) {
    if (!out) {
        if (error) {
            *error = "null metadata output";
        }
        return false;
    }
    if (!json_in.is_object()) {
        if (error) {
            *error = "handoff metadata must be a JSON object";
        }
        return false;
    }

    server_handoff_metadata_v2 tmp;
    tmp.schema = json_in.value("schema", tmp.schema);
    tmp.schema_version = json_in.value("schema_version", tmp.schema_version);
    tmp.handoff_session_id = json_in.value("handoff_session_id", std::string{});
    tmp.topology_epoch = json_in.value("topology_epoch", tmp.topology_epoch);
    tmp.remote_nodes = std::max(1, json_in.value("remote_nodes", tmp.remote_nodes));
    tmp.expected_gpu_layers = std::max(0, json_in.value("expected_gpu_layers", tmp.expected_gpu_layers));
    tmp.expected_remote_layers = std::max(0, json_in.value("expected_remote_layers", tmp.expected_remote_layers));
    tmp.execution_mode = json_in.value("execution_mode", tmp.execution_mode);
    tmp.transport_mode = json_in.value("transport_mode", tmp.transport_mode);
    tmp.remote_failover_policy = normalize_failover_policy(
        json_in.value("remote_failover_policy", tmp.remote_failover_policy));
    tmp.balance = json_in.value("balance", tmp.balance);
    tmp.remote_ranges = json_in.value("remote_ranges", std::string{});
    tmp.artifact_crc32 = json_in.value("artifact_crc32", 0u);

    if (json_in.contains("layer_map")) {
        const auto & layer_map = json_in.at("layer_map");
        if (!layer_map.is_array()) {
            if (error) {
                *error = "layer_map must be an array";
            }
            return false;
        }
        for (const auto & e : layer_map) {
            if (!e.is_object()) {
                if (error) {
                    *error = "layer_map entries must be objects";
                }
                return false;
            }
            server_handoff_layer_span span;
            span.node = e.value("node", std::string{});
            span.layer_start = e.value("layer_start", 0);
            span.layer_end = e.value("layer_end", 0);
            tmp.layer_map.push_back(std::move(span));
        }
    }

    if (json_in.contains("remote_node_descriptors")) {
        const auto & remote_nodes = json_in.at("remote_node_descriptors");
        if (!remote_nodes.is_array()) {
            if (error) {
                *error = "remote_node_descriptors must be an array";
            }
            return false;
        }
        for (const auto & e : remote_nodes) {
            if (!e.is_object()) {
                if (error) {
                    *error = "remote_node_descriptors entries must be objects";
                }
                return false;
            }
            server_handoff_remote_node node;
            node.node_id = e.value("node_id", std::string{});
            node.kv_host = e.value("kv_host", std::string{});
            node.kv_port = e.value("kv_port", 0);
            node.rpc_endpoint = e.value("rpc_endpoint", std::string{});
            node.role = e.value("role", std::string{"worker"});
            node.weight = std::max(1, e.value("weight", 1));
            node.healthy = e.value("healthy", true);
            node.promotable = e.value("promotable", false);
            tmp.remote_node_descriptors.push_back(std::move(node));
        }
    }

    if (!server_handoff_metadata_v2_validate(tmp, error)) {
        return false;
    }
    *out = std::move(tmp);
    return true;
}

bool server_handoff_metadata_v2_from_legacy(
    const server_handoff_legacy_fields & legacy,
    server_handoff_metadata_v2 * out,
    std::string * error) {
    if (!out) {
        if (error) {
            *error = "null metadata output";
        }
        return false;
    }

    server_handoff_metadata_v2 tmp;
    tmp.remote_nodes = std::max(1, legacy.remote_nodes);
    tmp.expected_gpu_layers = std::max(0, legacy.expected_gpu_layers);
    tmp.expected_remote_layers = std::max(0, legacy.expected_remote_layers);
    tmp.execution_mode = legacy.execution_mode.empty() ? "coupled" : legacy.execution_mode;
    tmp.balance = legacy.balance.empty() ? "roundrobin" : legacy.balance;
    tmp.remote_ranges = legacy.remote_ranges;
    tmp.remote_failover_policy = normalize_failover_policy(legacy.remote_failover_policy);
    tmp.handoff_session_id = legacy.handoff_session_id;
    tmp.topology_epoch = legacy.topology_epoch.empty() ? "epoch0" : legacy.topology_epoch;
    tmp.artifact_crc32 = legacy.artifact_crc32;
    tmp.transport_mode = legacy.transport_mode.empty() ? "auto" : legacy.transport_mode;

    if (!legacy.layer_map.empty()) {
        if (!server_handoff_parse_layer_map_legacy(legacy.layer_map, &tmp.layer_map, error)) {
            return false;
        }
    } else {
        append_map_coverage_spans(tmp.expected_gpu_layers, tmp.expected_remote_layers, &tmp.layer_map);
    }

    if (!server_handoff_metadata_v2_validate(tmp, error)) {
        return false;
    }
    *out = std::move(tmp);
    return true;
}

bool server_handoff_metadata_v2_from_legacy_json(
    const nlohmann::json & legacy_json,
    server_handoff_metadata_v2 * out,
    std::string * error) {
    if (!legacy_json.is_object()) {
        if (error) {
            *error = "legacy handoff metadata must be an object";
        }
        return false;
    }

    server_handoff_legacy_fields legacy;
    legacy.remote_nodes = std::max(1, legacy_json.value("remote_nodes", 1));
    legacy.expected_gpu_layers = std::max(0, legacy_json.value("expected_gpu_layers", 0));
    legacy.expected_remote_layers = std::max(0, legacy_json.value("expected_remote_layers", 0));
    legacy.execution_mode = legacy_json.value("execution_mode", std::string{"coupled"});
    legacy.balance = legacy_json.value("balance", std::string{"roundrobin"});
    legacy.remote_ranges = legacy_json.value("remote_ranges", std::string{});
    legacy.remote_failover_policy = legacy_json.value("remote_failover_policy", std::string{"reroute"});
    legacy.layer_map = legacy_json.value("layer_map", std::string{});
    legacy.handoff_session_id = legacy_json.value("handoff_session_id", std::string{});
    legacy.topology_epoch = legacy_json.value("topology_epoch", std::string{"epoch0"});
    legacy.artifact_crc32 = legacy_json.value("artifact_crc32", 0u);
    legacy.transport_mode = legacy_json.value("transport_mode", std::string{"auto"});

    return server_handoff_metadata_v2_from_legacy(legacy, out, error);
}
