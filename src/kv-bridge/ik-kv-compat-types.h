//
// KV Bridge - Type Definitions
// Copyright (C) 2025 Iwan Kawrakow / ik_llama contributors
// MIT license
// SPDX-License-Identifier: MIT
//
// This module implements the KV compatibility bridge for converting
// prefill-side .kva artifacts into ik-compatible sequence-state bytes.
//

#ifndef IK_KV_COMPAT_TYPES_H
#define IK_KV_COMPAT_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//
// Magic numbers and versioning
//

#define IK_KVA_MAGIC           0x6B766121U  // 'kva!' - KV Artifact
#define IK_KVA_FORMAT_MAJOR    1
#define IK_KVA_FORMAT_MINOR    0

#define IK_KV_BRIDGE_VERSION_MAJOR  1
#define IK_KV_BRIDGE_VERSION_MINOR  0

//
// Conversion result codes
//

typedef enum ik_kv_compat_convert_result {
    IK_KV_COMPAT_CONVERT_OK                = 0,   // Conversion succeeded
    IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG   = 1,   // Invalid argument
    IK_KV_COMPAT_CONVERT_ERR_NO_MEM        = 2,   // Memory allocation failed
    IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE     = 3,   // Source artifact parse error
    IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH  = 4,   // Destination schema mismatch
    IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD    = 5,   // Plan build failed
    IK_KV_COMPAT_CONVERT_ERR_CONVERT       = 6,   // Conversion runtime error
    IK_KV_COMPAT_CONVERT_ERR_CHECKSUM      = 7,   // Checksum validation failed
    IK_KV_COMPAT_CONVERT_ERR_SIZE          = 8,   // Size validation failed
    IK_KV_COMPAT_CONVERT_ERR_INTERNAL      = 9,   // Internal error
} ik_kv_compat_convert_result_t;

//
// Reject reasons for strict mode failures
//

typedef enum ik_kv_compat_reject_reason {
    IK_KV_COMPAT_REJECT_NONE                  = 0,   // No rejection (success)
    IK_KV_COMPAT_REJECT_MAGIC_MISMATCH        = 1,   // Artifact magic number mismatch
    IK_KV_COMPAT_REJECT_VERSION_UNSUPPORTED   = 2,   // Artifact version not supported
    IK_KV_COMPAT_REJECT_HEADER_MALFORMED      = 3,   // Header parsing failed
    IK_KV_COMPAT_REJECT_PAYLOAD_CRC_MISMATCH  = 4,   // Payload CRC check failed
    IK_KV_COMPAT_REJECT_MODEL_FINGERPRINT     = 5,   // Model fingerprint mismatch
    IK_KV_COMPAT_REJECT_TYPE_K_MISMATCH       = 6,   // type_k mismatch
    IK_KV_COMPAT_REJECT_TYPE_V_MISMATCH       = 7,   // type_v mismatch
    IK_KV_COMPAT_REJECT_VTRANS_MISMATCH       = 8,   // v_trans mode mismatch
    IK_KV_COMPAT_REJECT_N_STREAM_UNSUPPORTED  = 9,   // n_stream != 1 not supported in strict mode
    IK_KV_COMPAT_REJECT_N_LAYER_MISMATCH      = 10,  // Layer count mismatch
    IK_KV_COMPAT_REJECT_N_CTX_MISMATCH        = 11,  // Context length mismatch
    IK_KV_COMPAT_REJECT_N_HEAD_MISMATCH       = 12,  // Head count mismatch
    IK_KV_COMPAT_REJECT_INCOMPATIBLE_PROFILE  = 13,  // Profile not in compatibility matrix
    IK_KV_COMPAT_REJECT_PARTIAL_UNSUPPORTED   = 14,  // Partial state not supported
    IK_KV_COMPAT_REJECT_UNKNOWN               = 99,  // Unknown rejection reason
} ik_kv_compat_reject_reason_t;

//
// KV tensor types (mirrors ggml types used in KV cache)
//

typedef enum ik_kv_tensor_type {
    IK_KV_TYPE_F32    = 0,
    IK_KV_TYPE_F16    = 1,
    IK_KV_TYPE_BF16   = 2,
    IK_KV_TYPE_Q4_0   = 3,
    IK_KV_TYPE_Q4_1   = 4,
    IK_KV_TYPE_Q5_0   = 5,
    IK_KV_TYPE_Q5_1   = 6,
    IK_KV_TYPE_Q8_0   = 7,
    IK_KV_TYPE_Q8_1   = 8,
    IK_KV_TYPE_IQ4_NL = 9,
    IK_KV_TYPE_COUNT,
} ik_kv_tensor_type_t;

//
// Source artifact format identifiers
//

typedef enum ik_kv_source_format {
    IK_KV_SOURCE_FORMAT_UNKNOWN      = 0,
    IK_KV_SOURCE_FORMAT_IK_KVA       = 1,
    IK_KV_SOURCE_FORMAT_RTX_KVARTIF1 = 2,
} ik_kv_source_format_t;

//
// KVA artifact header
//

typedef struct ik_kva_header {
    uint32_t magic;           // Must be IK_KVA_MAGIC
    uint16_t format_major;    // Format major version
    uint16_t format_minor;    // Format minor version
    uint32_t n_layers;        // Number of layers
    uint32_t n_ctx;           // Context length
    uint32_t n_head_kv;       // KV heads count
    uint8_t  type_k;          // K tensor type (ik_kv_tensor_type_t)
    uint8_t  type_v;          // V tensor type (ik_kv_tensor_type_t)
    uint8_t  v_trans;         // V transpose mode (0 or 1)
    uint8_t  n_stream;        // Number of streams (must be 1 for strict v1)
    uint64_t payload_size;    // Size of payload following header
    uint32_t payload_crc;     // CRC32 of payload
    uint8_t  model_fingerprint[32];  // SHA-256 of model file or GGUF checksum
    uint8_t  reserved[32];    // Reserved for future use
} ik_kva_header_t;

//
// Source descriptor - describes the source KV state from prefill artifact
//

typedef struct ik_kv_source_descriptor {
    uint32_t n_layers;        // Number of layers
    uint32_t n_ctx;           // Context length
    uint32_t n_head_kv;       // KV heads
    uint32_t n_embd_head;     // Embedding dimension per head
    uint32_t token_count;     // Prompt token count from artifact metadata (0 if unavailable)
    uint8_t  type_k;          // K tensor type
    uint8_t  type_v;          // V tensor type
    uint8_t  v_trans;         // V transpose mode
    uint8_t  n_stream;        // Number of streams
    uint8_t  source_format;   // ik_kv_source_format_t
    uint8_t  reserved0[3];
    uint64_t payload_size;    // Raw payload size
    const uint8_t * payload;  // Pointer to payload data (non-owning)
    uint8_t  model_fingerprint[32];  // Source model fingerprint
} ik_kv_source_descriptor_t;

//
// Destination descriptor - describes the ik_llama context KV geometry
//

typedef struct ik_kv_dest_descriptor {
    uint32_t n_layers;        // Number of layers
    uint32_t n_ctx;           // Context length  
    uint32_t n_head_kv;       // KV heads
    uint32_t n_embd_head;     // Embedding dimension per head
    uint8_t  type_k;          // K tensor type
    uint8_t  type_v;          // V tensor type
    uint8_t  v_trans;         // V transpose mode
    uint8_t  n_stream;        // Number of streams
    uint8_t  model_fingerprint[32];  // Destination model fingerprint
} ik_kv_dest_descriptor_t;

//
// Compatibility plan key - used for caching
//

#define IK_KV_PLAN_KEY_SIZE 64

typedef struct ik_kv_compat_plan_key {
    uint8_t data[IK_KV_PLAN_KEY_SIZE];
} ik_kv_compat_plan_key_t;

//
// Layer mapping descriptor - describes how to map one layer's KV
//

typedef struct ik_kv_layer_mapping {
    uint32_t layer_idx;       // Layer index
    uint32_t k_row_size;      // K row size in bytes
    uint32_t v_row_size;      // V row size in bytes
    uint32_t k_offset;        // Source K offset
    uint32_t v_offset;        // Source V offset
    uint32_t dst_k_offset;    // Destination K offset
    uint32_t dst_v_offset;    // Destination V offset
    uint8_t  needs_scatter;   // Requires scatter (non-contiguous)
    uint8_t  needs_v_trans;   // Requires V transpose
    uint8_t  reserved[6];     // Reserved
} ik_kv_layer_mapping_t;

//
// Compatibility plan - precomputed mapping metadata
//

#define IK_KV_MAX_LAYERS 256

typedef struct ik_kv_compat_plan {
    ik_kv_compat_plan_key_t key;                // Plan cache key
    uint32_t n_layers;                          // Number of layers
    uint32_t total_src_size;                    // Total source payload size
    uint32_t total_dst_size;                    // Total destination payload size
    ik_kv_compat_reject_reason_t reject_reason; // Pre-computed reject reason if incompatible
    bool is_compatible;                         // True if compatible for conversion
    ik_kv_layer_mapping_t layer_mappings[IK_KV_MAX_LAYERS]; // Per-layer mappings
} ik_kv_compat_plan_t;

//
// Conversion context - state for a single conversion operation
//

typedef struct ik_kv_convert_ctx {
    const ik_kv_source_descriptor_t * src;   // Source descriptor
    const ik_kv_dest_descriptor_t * dst;     // Destination descriptor
    const ik_kv_compat_plan_t * plan;        // Precomputed plan
    uint8_t * output_buf;                    // Output buffer
    size_t output_size;                      // Output buffer size
    size_t bytes_written;                    // Bytes actually written
    ik_kv_compat_convert_result_t result;    // Conversion result
    ik_kv_compat_reject_reason_t reject;     // Reject reason if failed
} ik_kv_convert_ctx_t;

//
// Telemetry metrics
//

typedef struct ik_kv_bridge_metrics {
    uint64_t plan_key;              // Plan key hash
    bool plan_cache_hit;            // Cache hit flag
    uint64_t convert_us;            // Conversion time in microseconds
    uint64_t bytes_in;              // Input bytes
    uint64_t bytes_out;             // Output bytes
    uint8_t mode;                   // Bridge mode (0=off, 1=strict, 2=relaxed)
    uint8_t status;                 // Status (0=ok, 1=reject, 2=error)
    uint8_t reject_reason;          // Reject reason enum value
} ik_kv_bridge_metrics_t;

//
// Bridge mode configuration
//

typedef enum ik_kv_bridge_mode {
    IK_KV_BRIDGE_MODE_OFF     = 0,   // Bridge disabled
    IK_KV_BRIDGE_MODE_STRICT  = 1,   // Strict mode - fail on any mismatch
    IK_KV_BRIDGE_MODE_RELAXED = 2,   // Relaxed mode - allow safe conversions
} ik_kv_bridge_mode_t;

//
// Bridge configuration
//

typedef struct ik_kv_bridge_config {
    ik_kv_bridge_mode_t mode;           // Bridge mode
    const char * plan_cache_dir;        // Plan cache directory (NULL = no caching)
    bool allow_vtrans_convert;          // Allow V transpose in relaxed mode
    bool dry_run;                       // Validate only, don't convert
    bool no_fallback;                   // Fail-closed, no silent fallback
} ik_kv_bridge_config_t;

//
// Helper functions for type conversion
//

// Convert result to string
const char * ik_kv_compat_result_str(ik_kv_compat_convert_result_t result);

// Convert reject reason to string
const char * ik_kv_compat_reject_str(ik_kv_compat_reject_reason_t reason);

// Get KVA header size
size_t ik_kva_header_size(void);

// Calculate expected payload size from source descriptor
size_t ik_kv_source_payload_size(const ik_kv_source_descriptor_t * desc);

// Calculate expected destination blob size
size_t ik_kv_dest_blob_size(const ik_kv_dest_descriptor_t * desc);

#ifdef __cplusplus
}
#endif

#endif // IK_KV_COMPAT_TYPES_H
