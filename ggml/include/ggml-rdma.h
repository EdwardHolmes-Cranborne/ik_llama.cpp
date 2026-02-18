#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// RDMA transport backends — software-only, no extra hardware required.
//   JACCL   — macOS: IB verbs over Thunderbolt 5 (requires macOS SDK 26.2+)
//   IBVERBS — Linux: Soft-RoCE (rdma_rxe) or Soft-iWARP (siw) kernel modules
//   TCP     — fallback on all platforms (current ggml-rpc.cpp path)
enum ggml_rdma_backend {
  GGML_RDMA_TCP = 0,     // TCP fallback (always available)
  GGML_RDMA_JACCL = 1,   // macOS IB verbs over TB5
  GGML_RDMA_IBVERBS = 2, // Linux libibverbs (Soft-RoCE / Soft-iWARP / HW)
};

struct ggml_rdma_config {
  enum ggml_rdma_backend backend; // GGML_RDMA_TCP for auto-detect
  const char
      *device_name; // IB device: "en2"(mac), "rxe0"/"siw0"(linux), NULL=auto
  int rank;         // this node's rank (0-based)
  int world_size;   // total number of nodes
};

// Detect best available RDMA backend (compile-time + runtime checks).
GGML_API enum ggml_rdma_backend ggml_rdma_best_available(void);

// Initialize the RDMA transport. Returns true on success.
// Pass NULL config for auto-detect with defaults.
GGML_API bool ggml_rdma_init(const struct ggml_rdma_config *config);

// Shutdown RDMA transport, release resources.
GGML_API void ggml_rdma_shutdown(void);

// Zero-copy tensor send/recv over RDMA.
// Falls back to TCP send_data/recv_data if RDMA is not initialized.
GGML_API bool ggml_rdma_send_tensor(const struct ggml_tensor *tensor,
                                    int dest_rank);
GGML_API bool ggml_rdma_recv_tensor(struct ggml_tensor *tensor, int src_rank);

// Query whether RDMA transport is active.
GGML_API bool ggml_rdma_is_active(void);

// Get human-readable name of the active backend.
GGML_API const char *ggml_rdma_backend_name(enum ggml_rdma_backend backend);

#ifdef __cplusplus
}
#endif
