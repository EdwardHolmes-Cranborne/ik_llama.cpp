//
// KV Bridge CLI parsing tests (KVB-UT-070)
// Copyright (C) 2026 Iwan Kawrakow / ik_llama contributors
// MIT license
// SPDX-License-Identifier: MIT
//

#include "common.h"

#include <assert.h>
#include <stdio.h>
#include <string>
#include <vector>

static bool parse_with_args(const std::vector<std::string> & args, gpt_params & out) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (const std::string & arg : args) {
        argv.push_back(const_cast<char *>(arg.c_str()));
    }
    return gpt_params_parse((int) argv.size(), argv.data(), out);
}

int main() {
    {
        gpt_params params;
        const bool ok = parse_with_args({
            "llama-server",
            "--kv-bridge-mode", "relaxed",
            "--kv-bridge-plan-cache-dir", "/tmp/ik_kv_bridge_cli_test",
            "--kv-bridge-allow-vtrans-convert",
            "--kv-bridge-dry-run",
            "--kv-bridge-no-fallback",
            "--no-kv-bridge-telemetry",
        }, params);

        assert(ok);
        assert(params.kv_bridge_mode == "relaxed");
        assert(params.kv_bridge_plan_cache_dir == "/tmp/ik_kv_bridge_cli_test");
        assert(params.kv_bridge_allow_vtrans_convert);
        assert(params.kv_bridge_dry_run);
        assert(params.kv_bridge_no_fallback);
        assert(!params.kv_bridge_telemetry);
    }

    {
        gpt_params params;
        const bool ok = parse_with_args({
            "llama-cli",
            "--prefill-streaming",
            "--prefill-overlap",
            "--prefill-buffers", "3",
            "--prefill-prefetch", "2",
            "--prefill-slab-bytes", "8388608",
            "--prefill-min-stream-batch-tokens", "1024",
            "--prefill-decode-mode", "split_thunderbolt",
            "--prefill-decode-transport-required",
            "--prefill-transport-mode", "progressive",
            "--prefill-execution-mode", "decoupled",
            "--decode-gpu-layers", "12",
            "--decode-remote-layers", "20",
            "--decode-remote-nodes", "2",
            "--decode-remote-ranges", "0:0-10,1:10-20",
            "--decode-remote-failover", "local",
            "--prefill-transport-chunk-bytes", "2097152",
            "--prefill-transport-session-dir", "/tmp/ik_prefill_transport",
            "--kv-transport", "rdma",
            "--kv-transport-fallback",
            "--kv-host", "10.40.0.20",
            "--kv-port", "19001",
            "--kv-streams", "1",
            "--kv-stream-chunk-bytes", "4194304",
            "--kv-max-inflight-bytes", "268435456",
            "--kv-socket-send-buf", "1048576",
            "--kv-socket-recv-buf", "1048576",
            "--kv-bind-addrs", "10.40.0.10",
            "--kv-peer-addrs", "10.40.0.20",
            "--kv-balance", "roundrobin",
            "--tb-direct-endpoint", "rdma://10.40.0.20:19001",
        }, params);

        assert(ok);
        assert(params.prefill_streaming);
        assert(params.prefill_overlap);
        assert(params.prefill_buffers == 3);
        assert(params.prefill_prefetch == 2);
        assert(params.prefill_slab_bytes == 8388608);
        assert(params.prefill_min_stream_batch_tokens == 1024);
        assert(params.prefill_decode_mode == "split_thunderbolt");
        assert(params.prefill_decode_transport_required);
        assert(params.prefill_transport_mode == "progressive");
        assert(params.prefill_execution_mode == "decoupled");
        assert(params.decode_gpu_layers_hint == 12);
        assert(params.decode_remote_layers_hint == 20);
        assert(params.decode_remote_nodes_hint == 2);
        assert(params.decode_remote_ranges == "0:0-10,1:10-20");
        assert(params.decode_remote_failover_policy == "local");
        assert(params.prefill_transport_chunk_bytes == 2097152);
        assert(params.prefill_transport_session_dir == "/tmp/ik_prefill_transport");
        assert(params.kv_transport == "rdma");
        assert(params.kv_transport_fallback);
        assert(params.kv_host == "10.40.0.20");
        assert(params.kv_port == 19001);
        assert(params.kv_streams == 1);
        assert(params.kv_stream_chunk_bytes == 4194304);
        assert(params.kv_max_inflight_bytes == 268435456);
        assert(params.kv_socket_send_buf == 1048576);
        assert(params.kv_socket_recv_buf == 1048576);
        assert(params.kv_bind_addrs == "10.40.0.10");
        assert(params.kv_peer_addrs == "10.40.0.20");
        assert(params.kv_balance == "roundrobin");
        assert(params.tb_direct_endpoint == "rdma://10.40.0.20:19001");
    }

    {
        gpt_params params;
        const bool ok = parse_with_args({
            "llama-server",
            "--decode-node-id", "mac_decode_a",
            "--decode-cluster-file", "/tmp/ik_decode_cluster.json",
            "--decode-cluster-nodes-json", "{\"nodes\":[{\"node_id\":\"tb_remote_2\",\"kv_host\":\"10.40.0.30\",\"kv_port\":19003}]}",
            "--decode-route-dispatch-enable",
            "--decode-route-dispatch-max-hops", "2",
            "--decode-route-dispatch-session-dir", "/tmp/ik_decode_dispatch",
            "--decode-route-dispatch-streams", "2",
            "--decode-route-dispatch-chunk-bytes", "2097152",
            "--decode-route-dispatch-max-inflight-bytes", "536870912",
        }, params);

        assert(ok);
        assert(params.decode_node_id == "mac_decode_a");
        assert(params.decode_cluster_file == "/tmp/ik_decode_cluster.json");
        assert(params.decode_cluster_nodes_json == "{\"nodes\":[{\"node_id\":\"tb_remote_2\",\"kv_host\":\"10.40.0.30\",\"kv_port\":19003}]}");
        assert(params.decode_route_dispatch_enable);
        assert(params.decode_route_dispatch_max_hops == 2);
        assert(params.decode_route_dispatch_session_dir == "/tmp/ik_decode_dispatch");
        assert(params.decode_route_dispatch_streams == 2);
        assert(params.decode_route_dispatch_chunk_bytes == 2097152);
        assert(params.decode_route_dispatch_max_inflight_bytes == 536870912);
    }

    {
        gpt_params params;
        const bool ok = parse_with_args({
            "llama-server",
            "--kv-bridge-mode", "invalid_mode",
        }, params);
        assert(!ok);
    }

    {
        gpt_params params;
        const bool ok = parse_with_args({
            "llama-cli",
            "--prefill-decode-mode", "invalid_mode",
        }, params);
        assert(!ok);
    }

    printf("kv bridge CLI parsing tests passed\n");
    return 0;
}
