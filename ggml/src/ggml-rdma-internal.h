// ggml-rdma-internal.h — shared declarations between RDMA backends and
// dispatcher
#pragma once

#include "ggml-rdma.h"

#ifdef __cplusplus
extern "C" {
#endif

// JACCL backend (macOS)
bool ggml_rdma_jaccl_init(const struct ggml_rdma_config *config);
void ggml_rdma_jaccl_shutdown(void);
bool ggml_rdma_jaccl_send_tensor(const struct ggml_tensor *tensor,
                                 int dest_rank);
bool ggml_rdma_jaccl_recv_tensor(struct ggml_tensor *tensor, int src_rank);
bool ggml_rdma_jaccl_is_active(void);

// ibverbs backend (Linux: Soft-RoCE / Soft-iWARP)
bool ggml_rdma_ibverbs_init(const struct ggml_rdma_config *config);
void ggml_rdma_ibverbs_shutdown(void);
bool ggml_rdma_ibverbs_send_tensor(const struct ggml_tensor *tensor,
                                   int dest_rank);
bool ggml_rdma_ibverbs_recv_tensor(struct ggml_tensor *tensor, int src_rank);
bool ggml_rdma_ibverbs_is_active(void);

#ifdef __cplusplus
}
#endif
