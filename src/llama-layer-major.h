#pragma once

#include "llama-context.h"

// Layer-major streaming prefill: execute a compute graph one layer at a time
// with per-layer eval callbacks for weight streaming, KV-RAM staging, profiling.
//
// This is the single-ubatch fast path -- the entire graph is computed in one
// scheduler pass. Eval callbacks inject synchronization points at layer
// boundaries (detected by "-{il}" tensor name suffixes).

// Full layer-major decode: builds graph, allocates, sets inputs, computes
// per-layer, extracts outputs. Drop-in replacement for the standard decode
// path inside llama_decode_internal when layer_major mode is active.
// Returns 0 on success, negative on error.
int llama_decode_layer_major(
    llama_context & lctx,
    llama_batch     batch_all);
