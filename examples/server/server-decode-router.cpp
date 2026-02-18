#include "server-decode-router.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

namespace {

std::string trim_copy(std::string value) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
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

std::string normalize_node_id(std::string value) {
    return lower_copy(trim_copy(std::move(value)));
}

bool is_local_owner(const std::string & owner_raw, const std::string & local_node_id_raw) {
    const std::string owner = normalize_node_id(owner_raw);
    const std::string local_node_id = normalize_node_id(local_node_id_raw);
    if (owner.empty()) {
        return true;
    }
    if (owner == "local_cpu" || owner == "local_gpu") {
        return true;
    }
    if (!local_node_id.empty() && owner == local_node_id) {
        return true;
    }
    return owner.rfind("local_", 0) == 0;
}

bool is_known_failover_policy(const std::string & value_raw) {
    const std::string value = lower_copy(trim_copy(value_raw));
    return value == "reroute" || value == "local" || value == "fail";
}

std::string normalize_failover_policy(const std::string & value_raw) {
    const std::string value = lower_copy(trim_copy(value_raw));
    if (is_known_failover_policy(value)) {
        return value;
    }
    return "reroute";
}

void append_reason(std::string & dst, const std::string & reason) {
    if (reason.empty()) {
        return;
    }
    if (!dst.empty()) {
        dst += "; ";
    }
    dst += reason;
}

server_decode_cluster_node to_cluster_node(const server_handoff_remote_node & node) {
    server_decode_cluster_node out;
    out.node_id = normalize_node_id(node.node_id);
    out.kv_host = node.kv_host;
    out.kv_port = node.kv_port;
    out.rpc_endpoint = node.rpc_endpoint;
    out.role = node.role;
    out.weight = std::max(1, node.weight);
    out.healthy = node.healthy;
    out.promotable = node.promotable;
    return out;
}

} // namespace

bool server_decode_cluster_nodes_from_json(
    const nlohmann::json & json_in,
    std::vector<server_decode_cluster_node> * out,
    std::string * error) {
    if (!out) {
        if (error) {
            *error = "null node output";
        }
        return false;
    }

    out->clear();
    if (json_in.is_null()) {
        return true;
    }

    const nlohmann::json * nodes_json = nullptr;
    if (json_in.is_array()) {
        nodes_json = &json_in;
    } else if (json_in.is_object() && json_in.contains("nodes")) {
        nodes_json = &json_in.at("nodes");
    } else {
        if (error) {
            *error = "decode node json must be an array or object with 'nodes'";
        }
        return false;
    }

    if (!nodes_json->is_array()) {
        if (error) {
            *error = "decode node list must be an array";
        }
        return false;
    }

    for (const auto & e : *nodes_json) {
        if (!e.is_object()) {
            if (error) {
                *error = "decode node entry must be an object";
            }
            return false;
        }

        server_decode_cluster_node node;
        node.node_id = e.value("node_id", e.value("id", std::string{}));
        node.kv_host = e.value("kv_host", std::string{});
        node.kv_port = e.value("kv_port", 0);
        node.rpc_endpoint = e.value("rpc_endpoint", std::string{});
        node.role = e.value("role", std::string{"worker"});
        node.weight = std::max(1, e.value("weight", 1));
        node.healthy = e.value("healthy", true);
        node.promotable = e.value("promotable", false);

        node.node_id = normalize_node_id(node.node_id);
        if (node.node_id.empty()) {
            if (error) {
                *error = "decode node entry missing node_id";
            }
            return false;
        }

        out->push_back(std::move(node));
    }

    return true;
}

bool server_decode_cluster_nodes_load_file(
    const std::string & path,
    std::vector<server_decode_cluster_node> * out,
    std::string * error) {
    if (!out) {
        if (error) {
            *error = "null node output";
        }
        return false;
    }

    out->clear();
    if (path.empty()) {
        return true;
    }

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        if (error) {
            *error = "failed to open decode node file: " + path;
        }
        return false;
    }

    nlohmann::json parsed;
    try {
        ifs >> parsed;
    } catch (const std::exception & ex) {
        if (error) {
            *error = "invalid decode node json: " + std::string(ex.what());
        }
        return false;
    }

    return server_decode_cluster_nodes_from_json(parsed, out, error);
}

bool server_decode_router_build_plan(
    const server_handoff_metadata_v2 & handoff,
    const std::vector<server_decode_cluster_node> & nodes,
    const std::string & local_node_id,
    server_decode_route_plan * out,
    std::string * error) {
    if (!out) {
        if (error) {
            *error = "null route plan output";
        }
        return false;
    }

    std::string handoff_error;
    if (!server_handoff_metadata_v2_validate(handoff, &handoff_error)) {
        if (error) {
            *error = "invalid handoff metadata: " + handoff_error;
        }
        return false;
    }
    if (handoff.layer_map.empty()) {
        if (error) {
            *error = "handoff layer_map is empty";
        }
        return false;
    }

    server_decode_route_plan plan;
    plan.route_plan_id = handoff.handoff_session_id.empty()
        ? ("route_" + handoff.topology_epoch)
        : ("route_" + handoff.handoff_session_id);
    plan.topology_epoch = handoff.topology_epoch.empty() ? "epoch0" : handoff.topology_epoch;
    plan.failover_policy = normalize_failover_policy(handoff.remote_failover_policy);

    std::unordered_map<std::string, server_decode_cluster_node> node_by_id;
    auto upsert_node = [&node_by_id](const server_decode_cluster_node & node) {
        const std::string key = normalize_node_id(node.node_id);
        if (key.empty()) {
            return;
        }
        server_decode_cluster_node normalized = node;
        normalized.node_id = key;
        node_by_id[key] = std::move(normalized);
    };

    for (const auto & desc : handoff.remote_node_descriptors) {
        upsert_node(to_cluster_node(desc));
    }
    for (const auto & node : nodes) {
        upsert_node(node);
    }

    std::vector<std::string> reroute_primary_cycle;
    std::vector<std::string> reroute_promotable_cycle;
    reroute_primary_cycle.reserve(node_by_id.size());
    reroute_promotable_cycle.reserve(node_by_id.size());
    for (const auto & kv : node_by_id) {
        const auto & node = kv.second;
        if (!node.healthy || is_local_owner(node.node_id, local_node_id)) {
            continue;
        }
        const std::string role = lower_copy(trim_copy(node.role));
        const bool standby_like = role == "standby" || role == "spare" || role == "backup";
        if (standby_like && !node.promotable) {
            continue;
        }
        const int32_t weight = std::max(1, node.weight);
        for (int32_t i = 0; i < weight; ++i) {
            if (node.promotable) {
                reroute_promotable_cycle.push_back(node.node_id);
            } else {
                reroute_primary_cycle.push_back(node.node_id);
            }
        }
    }
    std::sort(reroute_primary_cycle.begin(), reroute_primary_cycle.end());
    std::sort(reroute_promotable_cycle.begin(), reroute_promotable_cycle.end());
    size_t reroute_primary_rr = 0;
    size_t reroute_promotable_rr = 0;

    auto choose_reroute_target = [&]() -> std::string {
        if (!reroute_primary_cycle.empty()) {
            const std::string target =
                reroute_primary_cycle[reroute_primary_rr % reroute_primary_cycle.size()];
            reroute_primary_rr++;
            return target;
        }
        if (!reroute_promotable_cycle.empty()) {
            const std::string target =
                reroute_promotable_cycle[reroute_promotable_rr % reroute_promotable_cycle.size()];
            reroute_promotable_rr++;
            return target;
        }
        return {};
    };

    for (const auto & span : handoff.layer_map) {
        server_decode_route_assignment assignment;
        assignment.owner = normalize_node_id(span.node);
        if (assignment.owner.empty()) {
            assignment.owner = "local_cpu";
        }
        assignment.layer_start = span.layer_start;
        assignment.layer_end = span.layer_end;

        if (!is_local_owner(assignment.owner, local_node_id)) {
            const auto it = node_by_id.find(assignment.owner);
            const bool known = it != node_by_id.end();
            if (!known) {
                if (plan.failover_policy == "fail") {
                    if (error) {
                        *error = "node '" + assignment.owner + "' is not present in decode cluster";
                    }
                    return false;
                }
                if (plan.failover_policy == "local") {
                    assignment.owner = "local_cpu";
                    plan.fallback_applied = true;
                    append_reason(plan.fallback_reason,
                                  "degraded unknown node '" + span.node + "' to local_cpu");
                } else {
                    const std::string reroute = choose_reroute_target();
                    if (!reroute.empty()) {
                        assignment.owner = reroute;
                        plan.fallback_applied = true;
                        append_reason(plan.fallback_reason,
                                      "rerouted unknown node '" + span.node + "' to '" + reroute + "'");
                    } else {
                        assignment.owner = "local_cpu";
                        plan.fallback_applied = true;
                        append_reason(plan.fallback_reason,
                                      "no healthy remote node configured for unknown '" + span.node +
                                          "'; degraded to local_cpu");
                    }
                }
            } else if (known && !it->second.healthy) {
                if (plan.failover_policy == "fail") {
                    if (error) {
                        *error = "node '" + assignment.owner + "' is unhealthy and failover_policy=fail";
                    }
                    return false;
                }

                if (plan.failover_policy == "local") {
                    assignment.owner = "local_cpu";
                    plan.fallback_applied = true;
                    append_reason(plan.fallback_reason,
                                  "degraded unhealthy node '" + span.node + "' to local_cpu");
                } else {
                    const std::string reroute = choose_reroute_target();
                    if (reroute.empty()) {
                        assignment.owner = "local_cpu";
                        plan.fallback_applied = true;
                        append_reason(plan.fallback_reason,
                                      "no healthy remote node for '" + span.node + "'; degraded to local_cpu");
                    } else {
                        assignment.owner = reroute;
                        plan.fallback_applied = true;
                        append_reason(plan.fallback_reason,
                                      "rerouted unhealthy node '" + span.node + "' to '" + reroute + "'");
                    }
                }
            }
        }

        plan.assignments.push_back(std::move(assignment));
    }

    std::vector<std::string> required;
    required.reserve(plan.assignments.size());
    for (const auto & assignment : plan.assignments) {
        if (!is_local_owner(assignment.owner, local_node_id)) {
            required.push_back(assignment.owner);
        }
    }
    std::sort(required.begin(), required.end());
    required.erase(std::unique(required.begin(), required.end()), required.end());
    plan.required_nodes = std::move(required);

    *out = std::move(plan);
    return true;
}

nlohmann::ordered_json server_decode_route_plan_to_json(
    const server_decode_route_plan & plan) {
    json assignments = json::array();
    for (const auto & a : plan.assignments) {
        assignments.push_back({
            {"owner", a.owner},
            {"layer_start", a.layer_start},
            {"layer_end", a.layer_end},
        });
    }

    json required_nodes = json::array();
    for (const auto & node : plan.required_nodes) {
        required_nodes.push_back(node);
    }

    return {
        {"route_plan_id", plan.route_plan_id},
        {"topology_epoch", plan.topology_epoch},
        {"failover_policy", plan.failover_policy},
        {"fallback_applied", plan.fallback_applied},
        {"fallback_reason", plan.fallback_reason},
        {"required_nodes", required_nodes},
        {"assignments", assignments},
    };
}
