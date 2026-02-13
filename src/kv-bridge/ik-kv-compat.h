//
// KV Bridge - Main Compatibility Interface
// Copyright (C) 2025 Iwan Kawrakow / ik_llama contributors
// MIT license
// SPDX-License-Identifier: MIT
//
// This module implements the KV compatibility bridge for converting
// prefill-side .kva artifacts into ik-compatible sequence-state bytes.
//
// Usage:
//   1. Parse source artifact with ik_kv_source_parse_kva_header()
//   2. Get destination descriptor with ik_kv_dest_introspect_from_ctx()
//   3. Build compatibility plan with ik_kv_compat_plan_build()
//   4. Convert with ik_kv_convert_prefill_to_ik_seq_blob()
//   5. Import into context with llama_state_seq_set_data()
//

#ifndef IK_KV_COMPAT_H
#define IK_KV_COMPAT_H

#include "ik-kv-compat-types.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Initialization and configuration
//

// Initialize the KV bridge module (call once at startup)
// Returns true on success, false on failure
bool ik_kv_bridge_init(void);

// Shutdown the KV bridge module (call at shutdown)
void ik_kv_bridge_shutdown(void);

// Set global bridge configuration
void ik_kv_bridge_set_config(const ik_kv_bridge_config_t * config);

// Get current global bridge configuration
void ik_kv_bridge_get_config(ik_kv_bridge_config_t * config);

//
// Source parsing (KVB-002)
//

// Parse KVA artifact header
// Returns IK_KV_COMPAT_CONVERT_OK on success, error code on failure
// Sets reject_reason on parse failure
ik_kv_compat_convert_result_t ik_kv_source_parse_kva_header(
    const uint8_t * artifact_data,
    size_t artifact_size,
    ik_kva_header_t * header_out,
    ik_kv_compat_reject_reason_t * reject_reason
);

// Parse source sequence state payload into a descriptor
// The descriptor holds non-owning references to the payload data
// Returns IK_KV_COMPAT_CONVERT_OK on success, error code on failure
ik_kv_compat_convert_result_t ik_kv_source_parse_prefill_seq_state(
    const ik_kva_header_t * header,
    const uint8_t * payload_data,
    size_t payload_size,
    ik_kv_source_descriptor_t * desc_out,
    ik_kv_compat_reject_reason_t * reject_reason
);

// Validate payload bounds and CRC
// Returns true if valid, false otherwise
bool ik_kv_source_validate_payload(
    const ik_kva_header_t * header,
    const uint8_t * payload_data,
    size_t payload_size
);

//
// Destination introspection (KVB-003)
//

// Forward declaration (from llama.h)
struct llama_context;

// Introspect destination KV geometry from llama context
// Returns IK_KV_COMPAT_CONVERT_OK on success, error code on failure
ik_kv_compat_convert_result_t ik_kv_dest_introspect_from_ctx(
    struct llama_context * ctx,
    ik_kv_dest_descriptor_t * desc_out
);

// Build destination descriptor from model parameters (without context)
// This is useful for pre-computing plans before context creation
ik_kv_compat_convert_result_t ik_kv_dest_introspect_from_model(
    const char * model_path,
    ik_kv_dest_descriptor_t * desc_out
);

//
// Compatibility plan key and fingerprinting (KVB-004)
//

// Build compatibility plan key from source and destination descriptors
// The key is used for plan cache lookup
ik_kv_compat_convert_result_t ik_kv_compat_plan_key_build(
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst,
    ik_kv_compat_plan_key_t * key_out
);

// Build model fingerprint from GGUF file
// Returns IK_KV_COMPAT_CONVERT_OK on success
ik_kv_compat_convert_result_t ik_kv_model_fingerprint_build(
    const char * model_path,
    uint8_t fingerprint[32]
);

// Compare two fingerprints
// Returns 0 if equal, non-zero otherwise
int ik_kv_fingerprint_compare(
    const uint8_t fp1[32],
    const uint8_t fp2[32]
);

//
// Plan building (KVB-005)
//

// Build strict v1 compatibility plan
// In strict mode, any mismatch results in rejection
// Returns IK_KV_COMPAT_CONVERT_OK if plan was built successfully
// Returns error code and sets plan->reject_reason if incompatible
ik_kv_compat_convert_result_t ik_kv_compat_plan_build_strict_v1(
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst,
    ik_kv_compat_plan_t * plan_out
);

// Validate an existing plan
// Returns true if plan is valid for the given source/destination pair
bool ik_kv_compat_plan_validate(
    const ik_kv_compat_plan_t * plan,
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst
);

//
// Plan cache (KVB-006)
//

// Load plan from cache
// Returns IK_KV_COMPAT_CONVERT_OK if found in cache
// Returns error code if not found or cache error
ik_kv_compat_convert_result_t ik_kv_plan_cache_load(
    const ik_kv_compat_plan_key_t * key,
    ik_kv_compat_plan_t * plan_out
);

// Store plan to cache
// Returns IK_KV_COMPAT_CONVERT_OK on success
ik_kv_compat_convert_result_t ik_kv_plan_cache_store(
    const ik_kv_compat_plan_key_t * key,
    const ik_kv_compat_plan_t * plan
);

// Invalidate a cached plan
// Returns IK_KV_COMPAT_CONVERT_OK on success
ik_kv_compat_convert_result_t ik_kv_plan_cache_invalidate(
    const ik_kv_compat_plan_key_t * key
);

// Clear entire plan cache
void ik_kv_plan_cache_clear(void);

//
// Conversion runtime (KVB-007)
//

// Convert prefill KV state to ik sequence blob
// This performs copy/pack/scatter only - no model math
// Returns IK_KV_COMPAT_CONVERT_OK on success
// On failure, sets ctx->result and ctx->reject appropriately
ik_kv_compat_convert_result_t ik_kv_convert_prefill_to_ik_seq_blob(
    ik_kv_convert_ctx_t * ctx
);

// Get required output buffer size for conversion
// Returns 0 on error
size_t ik_kv_convert_get_output_size(
    const ik_kv_source_descriptor_t * src,
    const ik_kv_dest_descriptor_t * dst
);

//
// Decode import hook (KVB-008)
//

// Import converted KV state into llama context
// This is the main entry point for the bridge
// Returns IK_KV_COMPAT_CONVERT_OK on success
// On failure, returns error code and sets reject_reason
ik_kv_compat_convert_result_t ik_kv_import_into_context(
    struct llama_context * ctx,
    const uint8_t * artifact_data,
    size_t artifact_size,
    int32_t dest_seq_id,
    ik_kv_compat_reject_reason_t * reject_reason
);

// Import with pre-built plan (for repeated imports with same source/dest profile)
ik_kv_compat_convert_result_t ik_kv_import_into_context_with_plan(
    struct llama_context * ctx,
    const ik_kv_compat_plan_t * plan,
    const uint8_t * payload_data,
    size_t payload_size,
    int32_t dest_seq_id,
    ik_kv_compat_reject_reason_t * reject_reason
);

//
// Telemetry (KVB-010)
//

// Get metrics for the last conversion operation
void ik_kv_bridge_get_last_metrics(ik_kv_bridge_metrics_t * metrics);

// Reset metrics
void ik_kv_bridge_reset_metrics(void);

// Enable/disable telemetry logging
void ik_kv_bridge_set_telemetry_enabled(bool enabled);

//
// Debug utilities (KVB-012)
//

// Print artifact header to stdout (for debugging)
void ik_kv_bridge_inspect_artifact(const uint8_t * artifact_data, size_t artifact_size);

// Print plan to stdout (for debugging)
void ik_kv_bridge_print_plan(const ik_kv_compat_plan_t * plan);

// Print descriptor to stdout (for debugging)
void ik_kv_bridge_print_source_descriptor(const ik_kv_source_descriptor_t * desc);
void ik_kv_bridge_print_dest_descriptor(const ik_kv_dest_descriptor_t * desc);

// Validate-only mode - parse and check compatibility without converting
ik_kv_compat_convert_result_t ik_kv_bridge_validate_only(
    struct llama_context * ctx,
    const uint8_t * artifact_data,
    size_t artifact_size,
    ik_kv_compat_reject_reason_t * reject_reason
);

#ifdef __cplusplus
}
#endif

#endif // IK_KV_COMPAT_H
