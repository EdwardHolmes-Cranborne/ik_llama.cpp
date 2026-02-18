// ggml-rdma.cpp — RDMA transport dispatcher
//
// Routes ggml_rdma_* API calls to the platform-specific backend:
//   macOS  → ggml-rdma-jaccl.cpp   (IB verbs over TB5)
//   Linux  → ggml-rdma-ibverbs.cpp (Soft-RoCE / Soft-iWARP)
//   Others → TCP fallback           (no-op, callers use existing TCP path)

#include "ggml-rdma.h"
#include "ggml-rdma-internal.h"
#include <cstdio>

// ── active backend tracking ─────────────────────────────────────────────────

static enum ggml_rdma_backend g_active_backend = GGML_RDMA_TCP;

// ── public API ──────────────────────────────────────────────────────────────

enum ggml_rdma_backend ggml_rdma_best_available(void) {
#ifdef GGML_RDMA_USE_JACCL
  return GGML_RDMA_JACCL;
#elif defined(GGML_RDMA_USE_IBVERBS)
  return GGML_RDMA_IBVERBS;
#else
  return GGML_RDMA_TCP;
#endif
}

bool ggml_rdma_init(const struct ggml_rdma_config *config) {
  enum ggml_rdma_backend backend = config ? config->backend : GGML_RDMA_TCP;

  // Auto-detect if backend is TCP (default / unset)
  if (backend == GGML_RDMA_TCP) {
    backend = ggml_rdma_best_available();
  }

  // Still TCP after auto-detect → no RDMA compiled in
  if (backend == GGML_RDMA_TCP) {
    fprintf(stderr,
            "[ggml-rdma] no RDMA backend available, using TCP fallback\n");
    g_active_backend = GGML_RDMA_TCP;
    return true; // not a failure, just no RDMA
  }

  bool ok = false;
  switch (backend) {
  case GGML_RDMA_JACCL:
    ok = ggml_rdma_jaccl_init(config);
    break;
  case GGML_RDMA_IBVERBS:
    ok = ggml_rdma_ibverbs_init(config);
    break;
  default:
    break;
  }

  if (ok) {
    g_active_backend = backend;
    fprintf(stderr, "[ggml-rdma] active backend: %s\n",
            ggml_rdma_backend_name(backend));
  } else {
    fprintf(stderr, "[ggml-rdma] %s init failed, falling back to TCP\n",
            ggml_rdma_backend_name(backend));
    g_active_backend = GGML_RDMA_TCP;
  }

  return ok;
}

void ggml_rdma_shutdown(void) {
  switch (g_active_backend) {
  case GGML_RDMA_JACCL:
    ggml_rdma_jaccl_shutdown();
    break;
  case GGML_RDMA_IBVERBS:
    ggml_rdma_ibverbs_shutdown();
    break;
  default:
    break;
  }
  g_active_backend = GGML_RDMA_TCP;
}

bool ggml_rdma_send_tensor(const struct ggml_tensor *tensor, int dest_rank) {
  switch (g_active_backend) {
  case GGML_RDMA_JACCL:
    return ggml_rdma_jaccl_send_tensor(tensor, dest_rank);
  case GGML_RDMA_IBVERBS:
    return ggml_rdma_ibverbs_send_tensor(tensor, dest_rank);
  default:
    return false; // caller should use TCP path
  }
}

bool ggml_rdma_recv_tensor(struct ggml_tensor *tensor, int src_rank) {
  switch (g_active_backend) {
  case GGML_RDMA_JACCL:
    return ggml_rdma_jaccl_recv_tensor(tensor, src_rank);
  case GGML_RDMA_IBVERBS:
    return ggml_rdma_ibverbs_recv_tensor(tensor, src_rank);
  default:
    return false;
  }
}

bool ggml_rdma_is_active(void) { return g_active_backend != GGML_RDMA_TCP; }

const char *ggml_rdma_backend_name(enum ggml_rdma_backend backend) {
  switch (backend) {
  case GGML_RDMA_JACCL:
    return "jaccl";
  case GGML_RDMA_IBVERBS:
    return "ibverbs";
  case GGML_RDMA_TCP:
    return "tcp";
  default:
    return "unknown";
  }
}
