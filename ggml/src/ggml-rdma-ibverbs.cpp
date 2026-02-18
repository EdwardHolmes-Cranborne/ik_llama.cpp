// ggml-rdma-ibverbs.cpp — Linux libibverbs backend (Soft-RoCE / Soft-iWARP)
//
// Works with any IB verbs provider:
//   - rdma_rxe kernel module (Soft-RoCE over UDP — no HW needed)
//   - siw     kernel module (Soft-iWARP over TCP — no HW needed)
//   - Hardware IB / RoCE NICs
//
// Build gate:  #ifdef GGML_RDMA_USE_IBVERBS  (set by CMake on Linux when
//              libibverbs is found via pkg-config / find_library)

#include "ggml-rdma-internal.h"
#include "ggml-rdma.h"

#ifdef GGML_RDMA_USE_IBVERBS

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
  struct ibv_mr *mr;
  int rank;
  int world_size;
  std::string device_name;
} g_ibv = {};

// ── helpers ─────────────────────────────────────────────────────────────────

static struct ibv_context *ibv_open_by_name(const char *name) {
  int n = 0;
  struct ibv_device **devs = ibv_get_device_list(&n);
  if (!devs || n == 0) {
    fprintf(stderr, "[ggml-rdma-ibverbs] no IB devices found\n");
    fprintf(stderr, "[ggml-rdma-ibverbs] hint: modprobe rdma_rxe && "
                    "rdma link add rxe0 type rxe netdev <your_iface>\n");
    return nullptr;
  }

  struct ibv_context *ctx = nullptr;
  for (int i = 0; i < n; i++) {
    const char *dname = ibv_get_device_name(devs[i]);
    if (name && *name && strcmp(dname, name) != 0) {
      continue;
    }
    ctx = ibv_open_device(devs[i]);
    if (ctx) {
      fprintf(stderr, "[ggml-rdma-ibverbs] opened device: %s\n", dname);
      break;
    }
  }

  if (!ctx) {
    fprintf(stderr, "[ggml-rdma-ibverbs] available devices:\n");
    for (int i = 0; i < n; i++) {
      fprintf(stderr, "  [%d] %s\n", i, ibv_get_device_name(devs[i]));
    }
  }

  ibv_free_device_list(devs);
  return ctx;
}

// ── public API ──────────────────────────────────────────────────────────────

bool ggml_rdma_ibverbs_init(const struct ggml_rdma_config *config) {
  if (g_ibv.initialised)
    return true;

  const char *dev_name = config ? config->device_name : nullptr;
  g_ibv.ctx = ibv_open_by_name(dev_name);
  if (!g_ibv.ctx) {
    return false;
  }

  g_ibv.pd = ibv_alloc_pd(g_ibv.ctx);
  if (!g_ibv.pd) {
    fprintf(stderr, "[ggml-rdma-ibverbs] ibv_alloc_pd failed\n");
    ibv_close_device(g_ibv.ctx);
    g_ibv.ctx = nullptr;
    return false;
  }

  g_ibv.cq = ibv_create_cq(g_ibv.ctx, 256, nullptr, nullptr, 0);
  if (!g_ibv.cq) {
    fprintf(stderr, "[ggml-rdma-ibverbs] ibv_create_cq failed\n");
    ibv_dealloc_pd(g_ibv.pd);
    ibv_close_device(g_ibv.ctx);
    g_ibv.pd = nullptr;
    g_ibv.ctx = nullptr;
    return false;
  }

  struct ibv_qp_init_attr qp_attr = {};
  qp_attr.send_cq = g_ibv.cq;
  qp_attr.recv_cq = g_ibv.cq;
  qp_attr.cap.max_send_wr = 128;
  qp_attr.cap.max_recv_wr = 128;
  qp_attr.cap.max_send_sge = 1;
  qp_attr.cap.max_recv_sge = 1;
  qp_attr.qp_type = IBV_QPT_RC;

  g_ibv.qp = ibv_create_qp(g_ibv.pd, &qp_attr);
  if (!g_ibv.qp) {
    fprintf(stderr, "[ggml-rdma-ibverbs] ibv_create_qp failed\n");
    ibv_destroy_cq(g_ibv.cq);
    ibv_dealloc_pd(g_ibv.pd);
    ibv_close_device(g_ibv.ctx);
    g_ibv.cq = nullptr;
    g_ibv.pd = nullptr;
    g_ibv.ctx = nullptr;
    return false;
  }

  g_ibv.rank = config ? config->rank : 0;
  g_ibv.world_size = config ? config->world_size : 1;
  g_ibv.device_name = dev_name ? dev_name : "auto";
  g_ibv.initialised = true;

  fprintf(stderr,
          "[ggml-rdma-ibverbs] initialised: rank=%d world_size=%d device=%s\n",
          g_ibv.rank, g_ibv.world_size, g_ibv.device_name.c_str());
  return true;
}

void ggml_rdma_ibverbs_shutdown(void) {
  if (!g_ibv.initialised)
    return;

  if (g_ibv.mr) {
    ibv_dereg_mr(g_ibv.mr);
    g_ibv.mr = nullptr;
  }
  if (g_ibv.qp) {
    ibv_destroy_qp(g_ibv.qp);
    g_ibv.qp = nullptr;
  }
  if (g_ibv.cq) {
    ibv_destroy_cq(g_ibv.cq);
    g_ibv.cq = nullptr;
  }
  if (g_ibv.pd) {
    ibv_dealloc_pd(g_ibv.pd);
    g_ibv.pd = nullptr;
  }
  if (g_ibv.ctx) {
    ibv_close_device(g_ibv.ctx);
    g_ibv.ctx = nullptr;
  }

  g_ibv.initialised = false;
  fprintf(stderr, "[ggml-rdma-ibverbs] shutdown complete\n");
}

bool ggml_rdma_ibverbs_send_tensor(const struct ggml_tensor *tensor,
                                   int dest_rank) {
  if (!g_ibv.initialised || !tensor || !tensor->data)
    return false;

  size_t nbytes = ggml_nbytes(tensor);
  void *data = tensor->data;

  if (g_ibv.mr) {
    ibv_dereg_mr(g_ibv.mr);
    g_ibv.mr = nullptr;
  }
  g_ibv.mr = ibv_reg_mr(g_ibv.pd, data, nbytes,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!g_ibv.mr) {
    fprintf(stderr,
            "[ggml-rdma-ibverbs] ibv_reg_mr failed for send (%zu bytes)\n",
            nbytes);
    return false;
  }

  struct ibv_sge sge = {};
  sge.addr = (uintptr_t)data;
  sge.length = (uint32_t)nbytes;
  sge.lkey = g_ibv.mr->lkey;

  struct ibv_send_wr wr = {};
  wr.wr_id = (uint64_t)dest_rank;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr *bad_wr = nullptr;
  int rc = ibv_post_send(g_ibv.qp, &wr, &bad_wr);
  if (rc != 0) {
    fprintf(stderr, "[ggml-rdma-ibverbs] ibv_post_send failed: %d\n", rc);
    return false;
  }

  struct ibv_wc wc = {};
  int poll_rc;
  do {
    poll_rc = ibv_poll_cq(g_ibv.cq, 1, &wc);
  } while (poll_rc == 0);

  if (poll_rc < 0 || wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "[ggml-rdma-ibverbs] send completion failed: status=%d\n",
            poll_rc < 0 ? -1 : (int)wc.status);
    return false;
  }

  return true;
}

bool ggml_rdma_ibverbs_recv_tensor(struct ggml_tensor *tensor, int src_rank) {
  if (!g_ibv.initialised || !tensor || !tensor->data)
    return false;

  size_t nbytes = ggml_nbytes(tensor);
  void *data = tensor->data;

  if (g_ibv.mr) {
    ibv_dereg_mr(g_ibv.mr);
    g_ibv.mr = nullptr;
  }
  g_ibv.mr = ibv_reg_mr(g_ibv.pd, data, nbytes,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!g_ibv.mr) {
    fprintf(stderr,
            "[ggml-rdma-ibverbs] ibv_reg_mr failed for recv (%zu bytes)\n",
            nbytes);
    return false;
  }

  struct ibv_sge sge = {};
  sge.addr = (uintptr_t)data;
  sge.length = (uint32_t)nbytes;
  sge.lkey = g_ibv.mr->lkey;

  struct ibv_recv_wr wr = {};
  wr.wr_id = (uint64_t)src_rank;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr *bad_wr = nullptr;
  int rc = ibv_post_recv(g_ibv.qp, &wr, &bad_wr);
  if (rc != 0) {
    fprintf(stderr, "[ggml-rdma-ibverbs] ibv_post_recv failed: %d\n", rc);
    return false;
  }

  struct ibv_wc wc = {};
  int poll_rc;
  do {
    poll_rc = ibv_poll_cq(g_ibv.cq, 1, &wc);
  } while (poll_rc == 0);

  if (poll_rc < 0 || wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "[ggml-rdma-ibverbs] recv completion failed: status=%d\n",
            poll_rc < 0 ? -1 : (int)wc.status);
    return false;
  }

  return true;
}

bool ggml_rdma_ibverbs_is_active(void) { return g_ibv.initialised; }

#else // !GGML_RDMA_USE_IBVERBS — stub for non-Linux

bool ggml_rdma_ibverbs_init(const struct ggml_rdma_config *) { return false; }
void ggml_rdma_ibverbs_shutdown(void) {}
bool ggml_rdma_ibverbs_send_tensor(const struct ggml_tensor *, int) {
  return false;
}
bool ggml_rdma_ibverbs_recv_tensor(struct ggml_tensor *, int) { return false; }
bool ggml_rdma_ibverbs_is_active(void) { return false; }

#endif // GGML_RDMA_USE_IBVERBS
