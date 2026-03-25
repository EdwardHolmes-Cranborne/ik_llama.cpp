#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// KV sync: accumulates tokens received from remote decode and batch-prefills
// them into the local KV cache so the RTX host stays in sync with the Mac.
struct llama_kv_sync {
    int32_t batch_size = 512;

    void push(llama_token tok);
    bool flush(llama_context * ctx);
    bool is_empty() const;
    int  pending_count() const;

private:
    std::vector<llama_token> pending_;
    int32_t next_pos_ = 0;
};
