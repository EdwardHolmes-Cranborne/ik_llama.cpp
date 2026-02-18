// ggml-rdma-jaccl.cpp — macOS JACCL backend (IB verbs over Thunderbolt 5)
//
// Requires macOS SDK 26.2+ with infiniband/verbs.h exposed by the
// Thunderbolt 5 RDMA subsystem.  This backend talks to the same IB verbs
// layer that MLX's "jaccl" distributed backend uses, but at the C level
// so we don't depend on the MLX Python runtime.
//
// Build gate:  #ifdef GGML_RDMA_USE_JACCL  (set by CMake on macOS when
//              the SDK version is >= 26.2 and infiniband/verbs.h exists)

#include "ggml-rdma-internal.h"
#include "ggml-rdma.h"

#ifdef GGML_RDMA_USE_JACCL

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <infiniband/verbs.h>
#include <string>

// ── internal state ──────────────────────────────────────────────────────────

static struct {
  bool initialised;
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  struct ibv_qp *qp;
  struct ibv_mr *mr; // single MR covering send/recv buffer
  int rank;
  int world_size;
  std::string device_name;
} g_jaccl = {};

// ── helpers ─────────────────────────────────────────────────────────────────

static struct ibv_context *jaccl_open_device(const char *name) {
  int n = 0;
  struct ibv_device **devs = ibv_get_device_list(&n);
  if (!devs || n == 0) {
    fprintf(stderr,
            "[ggml-rdma-jaccl] no IB devices found (is rdma_ctl enabled?)\n");
    return nullptr;
  }

  struct ibv_context *ctx = nullptr;
  for (int i = 0; i < n; i++) {
    if (name && *name && strcmp(ibv_get_device_name(devs[i]), name) != 0) {
      continue;
    }
    ctx = ibv_open_device(devs[i]);
    if (ctx) {
      fprintf(stderr, "[ggml-rdma-jaccl] opened device: %s\n",
              ibv_get_device_name(devs[i]));
      break;
    }
  }
  ibv_free_device_list(devs);
  return ctx;
}

// ── public API ──────────────────────────────────────────────────────────────

bool ggml_rdma_jaccl_init(const struct ggml_rdma_config *config) {
  if (g_jaccl.initialised)
    return true;

  const char *dev_name = config ? config->device_name : nullptr;
  g_jaccl.ctx = jaccl_open_device(dev_name);
  if (!g_jaccl.ctx) {
    fprintf(stderr, "[ggml-rdma-jaccl] failed to open IB device\n");
    return false;
  }

  // Allocate protection domain
  g_jaccl.pd = ibv_alloc_pd(g_jaccl.ctx);
  if (!g_jaccl.pd) {
    fprintf(stderr, "[ggml-rdma-jaccl] ibv_alloc_pd failed\n");
    ibv_close_device(g_jaccl.ctx);
    g_jaccl.ctx = nullptr;
    return false;
  }

  // Create completion queue (256 entries should be plenty for tensor ops)
  g_jaccl.cq = ibv_create_cq(g_jaccl.ctx, 256, nullptr, nullptr, 0);
  if (!g_jaccl.cq) {
    fprintf(stderr, "[ggml-rdma-jaccl] ibv_create_cq failed\n");
    ibv_dealloc_pd(g_jaccl.pd);
    ibv_close_device(g_jaccl.ctx);
    g_jaccl.pd = nullptr;
    g_jaccl.ctx = nullptr;
    return false;
  }

  // Create RC (Reliable Connection) queue pair
  struct ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = g_jaccl.cq;
  qp_attr.recv_cq = g_jaccl.cq;
  qp_attr.cap.max_send_wr = 128;
  qp_attr.cap.max_recv_wr = 128;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_attr.qp_type = IBV_QPT_RC;

  g_jaccl.qp = ibv_create_qp(g_jaccl.pd, &qp_attr);
  if (!g_jaccl.qp) {
    fprintf(stderr, "[ggml-rdma-jaccl] ibv_create_qp failed\n");
    ibv_destroy_cq(g_jaccl.cq);
    ibv_dealloc_pd(g_jaccl.pd);
    ibv_close_device(g_jaccl.ctx);
    g_jaccl.cq = nullptr;
    g_jaccl.pd = nullptr;
    g_jaccl.ctx = nullptr;
    return false;
  }

  g_jaccl.rank = config ? config->rank : 0;
  g_jaccl.world_size = config ? config->world_size : 1;
  g_jaccl.device_name = dev_name ? dev_name : "auto";
  g_jaccl.initialised = true;

  fprintf(stderr,
          "[ggml-rdma-jaccl] initialised: rank=%d world_size=%d device=%s\n",
          g_jaccl.rank, g_jaccl.world_size, g_jaccl.device_name.c_str());
  return true;
}

void ggml_rdma_jaccl_shutdown(void) {
  if (!g_jaccl.initialised)
    return;

  if (g_jaccl.mr) {
    ibv_dereg_mr(g_jaccl.mr);
    g_jaccl.mr = nullptr;
  }
  if (g_jaccl.qp) {
    ibv_destroy_qp(g_jaccl.qp);
    g_jaccl.qp = nullptr;
  }
  if (g_jaccl.cq) {
    ibv_destroy_cq(g_jaccl.cq);
    g_jaccl.cq = nullptr;
  }
  if (g_jaccl.pd) {
    ibv_dealloc_pd(g_jaccl.pd);
    g_jaccl.pd = nullptr;
  }
  if (g_jaccl.ctx) {
    ibv_close_device(g_jaccl.ctx);
    g_jaccl.ctx = nullptr;
  }

  g_jaccl.initialised = false;
  fprintf(stderr, "[ggml-rdma-jaccl] shutdown complete\n");
}

bool ggml_rdma_jaccl_send_tensor(const struct ggml_tensor *tensor,
                                 int dest_rank) {
  if (!g_jaccl.initialised || !tensor || !tensor->data)
    return false;

  size_t nbytes = ggml_nbytes(tensor);
  void *data = tensor->data;

  // Register memory region for this tensor (re-register if different from
  // cached)
  if (g_jaccl.mr) {
    ibv_dereg_mr(g_jaccl.mr);
    g_jaccl.mr = nullptr;
  }
  g_jaccl.mr = ibv_reg_mr(g_jaccl.pd, data, nbytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!g_jaccl.mr) {
    fprintf(stderr,
            "[ggml-rdma-jaccl] ibv_reg_mr failed for send (%zu bytes)\n",
            nbytes);
    return false;
  }

  // Post send work request
  struct ibv_sge sge = {};
  sge.addr = (uintptr_t)data;
  sge.length = (uint32_t)nbytes;
  sge.lkey = g_jaccl.mr->lkey;

  struct ibv_send_wr wr = {};
  wr.wr_id = (uint64_t)dest_rank;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr *bad_wr = nullptr;
  int rc = ibv_post_send(g_jaccl.qp, &wr, &bad_wr);
  if (rc != 0) {
    fprintf(stderr, "[ggml-rdma-jaccl] ibv_post_send failed: %d\n", rc);
    return false;
  }

  // Poll for completion
  struct ibv_wc wc = {};
  int poll_rc;
  do {
    poll_rc = ibv_poll_cq(g_jaccl.cq, 1, &wc);
  } while (poll_rc == 0);

  if (poll_rc < 0 || wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "[ggml-rdma-jaccl] send completion failed: status=%d\n",
            poll_rc < 0 ? -1 : (int)wc.status);
    return false;
  }

  return true;
}

bool ggml_rdma_jaccl_recv_tensor(struct ggml_tensor *tensor, int src_rank) {
  if (!g_jaccl.initialised || !tensor || !tensor->data)
    return false;

  size_t nbytes = ggml_nbytes(tensor);
  void *data = tensor->data;

  if (g_jaccl.mr) {
    ibv_dereg_mr(g_jaccl.mr);
    g_jaccl.mr = nullptr;
  }
  g_jaccl.mr = ibv_reg_mr(g_jaccl.pd, data, nbytes,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!g_jaccl.mr) {
    fprintf(stderr,
            "[ggml-rdma-jaccl] ibv_reg_mr failed for recv (%zu bytes)\n",
            nbytes);
    return false;
  }

  // Post receive work request
  struct ibv_sge sge = {};
  sge.addr = (uintptr_t)data;
  sge.length = (uint32_t)nbytes;
  sge.lkey = g_jaccl.mr->lkey;

  struct ibv_recv_wr wr = {};
  wr.wr_id = (uint64_t)src_rank;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr *bad_wr = nullptr;
  int rc = ibv_post_recv(g_jaccl.qp, &wr, &bad_wr);
  if (rc != 0) {
    fprintf(stderr, "[ggml-rdma-jaccl] ibv_post_recv failed: %d\n", rc);
    return false;
  }

  // Poll for completion
  struct ibv_wc wc = {};
  int poll_rc;
  do {
    poll_rc = ibv_poll_cq(g_jaccl.cq, 1, &wc);
  } while (poll_rc == 0);

  if (poll_rc < 0 || wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "[ggml-rdma-jaccl] recv completion failed: status=%d\n",
            poll_rc < 0 ? -1 : (int)wc.status);
    return false;
  }

  return true;
}

bool ggml_rdma_jaccl_is_active(void) { return g_jaccl.initialised; }

#else // !GGML_RDMA_USE_JACCL — stub for non-macOS or old SDK

bool ggml_rdma_jaccl_init(const struct ggml_rdma_config *) { return false; }
void ggml_rdma_jaccl_shutdown(void) {}
bool ggml_rdma_jaccl_send_tensor(const struct ggml_tensor *, int) {
  return false;
}
bool ggml_rdma_jaccl_recv_tensor(struct ggml_tensor *, int) { return false; }
bool ggml_rdma_jaccl_is_active(void) { return false; }

#endif // GGML_RDMA_USE_JACCL
