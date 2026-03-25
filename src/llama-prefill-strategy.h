#pragma once

#include <cmath>
#include <cstdint>

enum llama_prefill_strategy {
    LLAMA_PREFILL_STRATEGY_AUTO = 0,
    LLAMA_PREFILL_STRATEGY_MAC  = 1,
    LLAMA_PREFILL_STRATEGY_RTX  = 2,
};

struct llama_prefill_strategy_params {
    llama_prefill_strategy strategy = LLAMA_PREFILL_STRATEGY_AUTO;
    int32_t crossover_tokens       = 0;
    float   rtx_streaming_floor_ms = 0.0f;
    float   mac_prefill_tok_s      = 0.0f;
    float   pcie_bandwidth_gbs     = 64.0f;
    float   rtx_compute_tok_s      = 2000.0f;
};

struct llama_prefill_strategy_decision {
    llama_prefill_strategy chosen;
    int32_t     crossover_tokens;
    float       estimated_mac_ms;
    float       estimated_rtx_ms;
    const char *reason;
};

llama_prefill_strategy_decision llama_prefill_strategy_select(
    const llama_prefill_strategy_params & params,
    int32_t n_prompt_tokens,
    float   model_size_gb);
