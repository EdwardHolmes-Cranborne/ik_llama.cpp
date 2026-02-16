//
// KV Bridge CLI utility
// Copyright (C) 2026 Iwan Kawrakow / ik_llama contributors
// MIT license
// SPDX-License-Identifier: MIT
//

#include "ik-kv-compat.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out, std::string & err) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        err = "failed to open file: " + path;
        return false;
    }

    const std::streamsize size = ifs.tellg();
    if (size < 0) {
        err = "failed to get file size: " + path;
        return false;
    }
    ifs.seekg(0, std::ios::beg);

    out.resize((size_t) size);
    if (size > 0 && !ifs.read((char *) out.data(), size)) {
        err = "failed to read file bytes: " + path;
        return false;
    }
    return true;
}

void usage(const char * argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --kv-bridge-inspect-artifact FILE\n"
        << "  --kv-bridge-print-plan FILE\n"
        << "  --kv-bridge-validate-only FILE\n"
        << "  --model MODEL_PATH                 (required for print-plan/validate-only)\n"
        << "  --kv-bridge-allow-vtrans-convert   (for relaxed validation)\n"
        << "  --help\n";
}

} // namespace

int main(int argc, char ** argv) {
    std::string inspect_artifact;
    std::string print_plan_artifact;
    std::string validate_artifact;
    std::string model_path;
    bool allow_vtrans_convert = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string & flag) -> const char * {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << "\n";
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--kv-bridge-inspect-artifact") {
            inspect_artifact = need_value(arg);
            continue;
        }
        if (arg == "--kv-bridge-print-plan") {
            print_plan_artifact = need_value(arg);
            continue;
        }
        if (arg == "--kv-bridge-validate-only") {
            validate_artifact = need_value(arg);
            continue;
        }
        if (arg == "--model") {
            model_path = need_value(arg);
            continue;
        }
        if (arg == "--kv-bridge-allow-vtrans-convert") {
            allow_vtrans_convert = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        }
        std::cerr << "unknown argument: " << arg << "\n";
        usage(argv[0]);
        return 2;
    }

    if (inspect_artifact.empty() && print_plan_artifact.empty() && validate_artifact.empty()) {
        usage(argv[0]);
        return 2;
    }

    ik_kv_bridge_init();

    auto run_parse = [&](const std::string & artifact_path,
                         ik_kva_header_t & header,
                         ik_kv_source_descriptor_t & src_desc,
                         std::vector<uint8_t> & artifact_bytes) -> bool {
        std::string err;
        if (!read_file_bytes(artifact_path, artifact_bytes, err)) {
            std::cerr << err << "\n";
            return false;
        }
        ik_kv_compat_reject_reason_t reject = IK_KV_COMPAT_REJECT_NONE;
        const ik_kv_compat_convert_result_t rc = ik_kv_source_parse_kva_header(
            artifact_bytes.data(), artifact_bytes.size(), &header, &reject);
        if (rc != IK_KV_COMPAT_CONVERT_OK) {
            std::cerr << "header parse failed: " << ik_kv_compat_result_str(rc)
                      << ", reject=" << ik_kv_compat_reject_str(reject) << "\n";
            return false;
        }
        if (artifact_bytes.size() < sizeof(header) + (size_t) header.payload_size) {
            std::cerr << "artifact payload is truncated\n";
            return false;
        }
        const uint8_t * payload = artifact_bytes.data() + sizeof(header);
        const size_t payload_size = (size_t) header.payload_size;
        if (!ik_kv_source_validate_payload(&header, payload, payload_size)) {
            std::cerr << "payload validation failed\n";
            return false;
        }

        reject = IK_KV_COMPAT_REJECT_NONE;
        const ik_kv_compat_convert_result_t src_rc = ik_kv_source_parse_prefill_seq_state(
            &header, payload, payload_size, &src_desc, &reject);
        if (src_rc != IK_KV_COMPAT_CONVERT_OK) {
            std::cerr << "payload parse failed: " << ik_kv_compat_result_str(src_rc)
                      << ", reject=" << ik_kv_compat_reject_str(reject) << "\n";
            return false;
        }
        return true;
    };

    if (!inspect_artifact.empty()) {
        std::vector<uint8_t> artifact;
        std::string err;
        if (!read_file_bytes(inspect_artifact, artifact, err)) {
            std::cerr << err << "\n";
            return 1;
        }
        ik_kv_bridge_inspect_artifact(artifact.data(), artifact.size());
    }

    auto run_plan_action = [&](const std::string & artifact_path, bool validate_only_mode) -> int {
        if (model_path.empty()) {
            std::cerr << "--model MODEL_PATH is required for this action\n";
            return 2;
        }

        std::vector<uint8_t> artifact;
        ik_kva_header_t header = {};
        ik_kv_source_descriptor_t src_desc = {};
        if (!run_parse(artifact_path, header, src_desc, artifact)) {
            return 1;
        }

        ik_kv_dest_descriptor_t dst_desc = {};
        const ik_kv_compat_convert_result_t dst_rc =
            ik_kv_dest_introspect_from_model(model_path.c_str(), &dst_desc);
        if (dst_rc != IK_KV_COMPAT_CONVERT_OK) {
            std::cerr << "destination introspection failed: "
                      << ik_kv_compat_result_str(dst_rc) << "\n";
            return 1;
        }

        ik_kv_compat_plan_t plan = {};
        const ik_kv_compat_convert_result_t plan_rc =
            ik_kv_compat_plan_build_strict_v1(&src_desc, &dst_desc, &plan);
        if (plan_rc != IK_KV_COMPAT_CONVERT_OK) {
            std::cerr << "plan build failed: " << ik_kv_compat_result_str(plan_rc) << "\n";
            return 1;
        }

        if (!plan.is_compatible && allow_vtrans_convert &&
            plan.reject_reason == IK_KV_COMPAT_REJECT_VTRANS_MISMATCH) {
            ik_kv_source_descriptor_t src_relaxed = src_desc;
            src_relaxed.v_trans = dst_desc.v_trans;
            ik_kv_compat_plan_t relaxed_plan = {};
            const ik_kv_compat_convert_result_t relaxed_rc =
                ik_kv_compat_plan_build_strict_v1(&src_relaxed, &dst_desc, &relaxed_plan);
            if (relaxed_rc == IK_KV_COMPAT_CONVERT_OK && relaxed_plan.is_compatible) {
                plan = relaxed_plan;
            }
        }

        if (!validate_only_mode) {
            ik_kv_bridge_print_source_descriptor(&src_desc);
            ik_kv_bridge_print_dest_descriptor(&dst_desc);
            ik_kv_bridge_print_plan(&plan);
            return plan.is_compatible ? 0 : 1;
        }

        if (plan.is_compatible) {
            std::cout << "validate-only: compatible\n";
            return 0;
        }
        std::cout << "validate-only: incompatible, reject="
                  << ik_kv_compat_reject_str(plan.reject_reason) << "\n";
        return 1;
    };

    if (!print_plan_artifact.empty()) {
        const int rc = run_plan_action(print_plan_artifact, false);
        if (rc != 0) {
            return rc;
        }
    }

    if (!validate_artifact.empty()) {
        const int rc = run_plan_action(validate_artifact, true);
        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}

