//
// KV Bridge - Type Definitions Implementation
// Copyright (C) 2025 Iwan Kawrakow / ik_llama contributors
// MIT license
// SPDX-License-Identifier: MIT
//

#include "ik-kv-compat-types.h"
#include <string.h>

//
// String conversion helpers
//

static const char * result_strings[] = {
    "OK",                   // IK_KV_COMPAT_CONVERT_OK
    "ERR_INVALID_ARG",      // IK_KV_COMPAT_CONVERT_ERR_INVALID_ARG
    "ERR_NO_MEM",           // IK_KV_COMPAT_CONVERT_ERR_NO_MEM
    "ERR_SRC_PARSE",        // IK_KV_COMPAT_CONVERT_ERR_SRC_PARSE
    "ERR_DST_MISMATCH",     // IK_KV_COMPAT_CONVERT_ERR_DST_MISMATCH
    "ERR_PLAN_BUILD",       // IK_KV_COMPAT_CONVERT_ERR_PLAN_BUILD
    "ERR_CONVERT",          // IK_KV_COMPAT_CONVERT_ERR_CONVERT
    "ERR_CHECKSUM",         // IK_KV_COMPAT_CONVERT_ERR_CHECKSUM
    "ERR_SIZE",             // IK_KV_COMPAT_CONVERT_ERR_SIZE
    "ERR_INTERNAL",         // IK_KV_COMPAT_CONVERT_ERR_INTERNAL
};

static const char * reject_strings[] = {
    "NONE",                     // IK_KV_COMPAT_REJECT_NONE
    "MAGIC_MISMATCH",           // IK_KV_COMPAT_REJECT_MAGIC_MISMATCH
    "VERSION_UNSUPPORTED",      // IK_KV_COMPAT_REJECT_VERSION_UNSUPPORTED
    "HEADER_MALFORMED",         // IK_KV_COMPAT_REJECT_HEADER_MALFORMED
    "PAYLOAD_CRC_MISMATCH",     // IK_KV_COMPAT_REJECT_PAYLOAD_CRC_MISMATCH
    "MODEL_FINGERPRINT",        // IK_KV_COMPAT_REJECT_MODEL_FINGERPRINT
    "TYPE_K_MISMATCH",          // IK_KV_COMPAT_REJECT_TYPE_K_MISMATCH
    "TYPE_V_MISMATCH",          // IK_KV_COMPAT_REJECT_TYPE_V_MISMATCH
    "VTRANS_MISMATCH",          // IK_KV_COMPAT_REJECT_VTRANS_MISMATCH
    "N_STREAM_UNSUPPORTED",     // IK_KV_COMPAT_REJECT_N_STREAM_UNSUPPORTED
    "N_LAYER_MISMATCH",         // IK_KV_COMPAT_REJECT_N_LAYER_MISMATCH
    "N_CTX_MISMATCH",           // IK_KV_COMPAT_REJECT_N_CTX_MISMATCH
    "N_HEAD_MISMATCH",          // IK_KV_COMPAT_REJECT_N_HEAD_MISMATCH
    "INCOMPATIBLE_PROFILE",     // IK_KV_COMPAT_REJECT_INCOMPATIBLE_PROFILE
    "PARTIAL_UNSUPPORTED",      // IK_KV_COMPAT_REJECT_PARTIAL_UNSUPPORTED
    "UNKNOWN",                  // IK_KV_COMPAT_REJECT_UNKNOWN
};

const char * ik_kv_compat_result_str(ik_kv_compat_convert_result_t result) {
    if (result >= 0 && result < (ik_kv_compat_convert_result_t)(sizeof(result_strings) / sizeof(result_strings[0]))) {
        return result_strings[result];
    }
    return "UNKNOWN_RESULT";
}

const char * ik_kv_compat_reject_str(ik_kv_compat_reject_reason_t reason) {
    if (reason >= 0 && reason < (ik_kv_compat_reject_reason_t)(sizeof(reject_strings) / sizeof(reject_strings[0]))) {
        return reject_strings[reason];
    }
    return "UNKNOWN_REJECT";
}

//
// Size helpers
//

size_t ik_kva_header_size(void) {
    return sizeof(ik_kva_header_t);
}

// Get block size for a given tensor type
static size_t get_type_block_size(ik_kv_tensor_type_t type) {
    switch (type) {
        case IK_KV_TYPE_F32:   return 1;
        case IK_KV_TYPE_F16:   return 1;
        case IK_KV_TYPE_BF16:  return 1;
        case IK_KV_TYPE_Q4_0:  return 32;
        case IK_KV_TYPE_Q4_1:  return 32;
        case IK_KV_TYPE_Q5_0:  return 32;
        case IK_KV_TYPE_Q5_1:  return 32;
        case IK_KV_TYPE_Q8_0:  return 32;
        case IK_KV_TYPE_Q8_1:  return 32;
        case IK_KV_TYPE_IQ4_NL: return 32;
        default:               return 1;
    }
}

// Get bytes per element for a given tensor type
static size_t get_type_bytes_per_element(ik_kv_tensor_type_t type) {
    switch (type) {
        case IK_KV_TYPE_F32:   return 4;
        case IK_KV_TYPE_F16:   return 2;
        case IK_KV_TYPE_BF16:  return 2;
        case IK_KV_TYPE_Q4_0:  return 2;   // 2 bytes per 32 elements = 0.5 bp-e
        case IK_KV_TYPE_Q4_1:  return 4;   // 4 bytes per 32 elements = 1 bp-e
        case IK_KV_TYPE_Q5_0:  return 3;   // 3 bytes per 32 elements (roughly)
        case IK_KV_TYPE_Q5_1:  return 4;   // 4 bytes per 32 elements (roughly)
        case IK_KV_TYPE_Q8_0:  return 2;   // 2 bytes per element
        case IK_KV_TYPE_Q8_1:  return 4;   // 4 bytes per element
        case IK_KV_TYPE_IQ4_NL: return 2;  // 2 bytes per 32 elements = 0.5 bp-e
        default:               return 4;
    }
}

size_t ik_kv_source_payload_size(const ik_kv_source_descriptor_t * desc) {
    if (!desc) {
        return 0;
    }
    
    // Calculate KV size per layer
    // K cache: n_head_kv * n_embd_head * n_ctx elements
    // V cache: n_head_kv * n_embd_head * n_ctx elements (possibly transposed)
    
    size_t k_bytes_per_element = get_type_bytes_per_element((ik_kv_tensor_type_t)desc->type_k);
    size_t v_bytes_per_element = get_type_bytes_per_element((ik_kv_tensor_type_t)desc->type_v);
    
    size_t k_per_layer = desc->n_head_kv * desc->n_embd_head * desc->n_ctx * k_bytes_per_element;
    size_t v_per_layer = desc->n_head_kv * desc->n_embd_head * desc->n_ctx * v_bytes_per_element;
    
    return desc->n_layers * (k_per_layer + v_per_layer);
}

size_t ik_kv_dest_blob_size(const ik_kv_dest_descriptor_t * desc) {
    if (!desc) {
        return 0;
    }
    
    size_t k_bytes_per_element = get_type_bytes_per_element((ik_kv_tensor_type_t)desc->type_k);
    size_t v_bytes_per_element = get_type_bytes_per_element((ik_kv_tensor_type_t)desc->type_v);
    
    size_t k_per_layer = desc->n_head_kv * desc->n_embd_head * desc->n_ctx * k_bytes_per_element;
    size_t v_per_layer = desc->n_head_kv * desc->n_embd_head * desc->n_ctx * v_bytes_per_element;
    
    return desc->n_layers * (k_per_layer + v_per_layer);
}
