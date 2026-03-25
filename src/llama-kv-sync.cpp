#include "llama-kv-sync.h"

#include <cstdio>

void llama_kv_sync::push(llama_token tok) {
    pending_.push_back(tok);
}

bool llama_kv_sync::flush(llama_context * ctx) {
    if (pending_.empty()) return true;

    const int n = (int)pending_.size();
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;

    for (int i = 0; i < n; i++) {
        batch.token[i]     = pending_[i];
        batch.pos[i]       = next_pos_ + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;  // no logits needed for sync
    }

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "llama_kv_sync::flush: decode failed for %d tokens at pos %d\n",
                n, next_pos_);
        llama_batch_free(batch);
        return false;
    }

    next_pos_ += n;
    pending_.clear();
    llama_batch_free(batch);
    return true;
}

bool llama_kv_sync::is_empty() const {
    return pending_.empty();
}

int llama_kv_sync::pending_count() const {
    return (int)pending_.size();
}
