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
            "llama-server",
            "--kv-bridge-mode", "invalid_mode",
        }, params);
        assert(!ok);
    }

    printf("kv bridge CLI parsing tests passed\n");
    return 0;
}

