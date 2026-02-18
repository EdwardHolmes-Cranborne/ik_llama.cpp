// Metal+RPC split buffer type for graph split (tensor parallelism)
//
// This implements a buffer type that distributes tensor rows across multiple
// devices (Metal + RPC) for graph split mode. It mirrors CUDA's
// ggml_backend_cuda_split_buffer_type but uses the generic backend buffer API
// so it works with any combination of Metal and RPC backends.

#include "ggml-metal-rpc-split.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>

// ============================================================================
// Context structures
// ============================================================================

struct ggml_backend_split_buffer_type_context {
  std::vector<ggml_backend_buffer_type_t> device_bufts;
  std::vector<float> splits;
  int n_device;
};

struct ggml_backend_split_buffer_context {
  ggml_backend_split_buffer_type_context *type_ctx;
  std::vector<ggml_backend_buffer_t> sub_buffers; // owned per-device buffers

  ~ggml_backend_split_buffer_context() {
    for (auto *buf : sub_buffers) {
      if (buf) {
        ggml_backend_buffer_free(buf);
      }
    }
  }
};

// ============================================================================
// Buffer interface
// ============================================================================

static const char *
ggml_backend_split_buffer_get_name(ggml_backend_buffer_t buffer) {
  GGML_UNUSED(buffer);
  return "Metal_RPC_Split";
}

GGML_CALL static void
ggml_backend_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
  auto *ctx = (ggml_backend_split_buffer_context *)buffer->context;
  delete ctx;
}

GGML_CALL static void *
ggml_backend_split_buffer_get_base(ggml_backend_buffer_t buffer) {
  GGML_UNUSED(buffer);
  return (void *)0x1000; // dummy — real data is in per-device sub-tensors
}

GGML_CALL static void
ggml_backend_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                      ggml_tensor *tensor) {
  if (!tensor->extra)
    return;

  auto *extra = (ggml_split_tensor_t *)tensor->extra;
  auto *buf_ctx = (ggml_backend_split_buffer_context *)buffer->context;
  auto *type_ctx = buf_ctx->type_ctx;

  GGML_ASSERT(extra->n_device <= type_ctx->n_device);

  for (int i = 0; i < extra->n_device; ++i) {
    if (!extra->splits[i])
      continue;

    auto *split = extra->splits[i];
    auto size = ggml_nbytes(split);

    // Allocate on the device's native buffer type
    auto *dev_buffer =
        ggml_backend_buft_alloc_buffer(type_ctx->device_bufts[i], size);

    if (!dev_buffer) {
      fprintf(stderr, "%s: failed to allocate %zu bytes on device %d for %s\n",
              __func__, size, i, tensor->name);
      GGML_ABORT("split buffer allocation failed");
    }

    ggml_backend_buffer_set_usage(dev_buffer,
                                  GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Assign the buffer to the sub-tensor
    split->buffer = dev_buffer;
    split->data = ggml_backend_buffer_get_base(dev_buffer);

    // Initialize the tensor in the device buffer
    if (dev_buffer->iface.init_tensor) {
      dev_buffer->iface.init_tensor(dev_buffer, split);
    }

    buf_ctx->sub_buffers.push_back(dev_buffer);
  }
}

GGML_CALL static void
ggml_backend_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor, const void *data,
                                     size_t offset, size_t size) {
  GGML_UNUSED(buffer);
  if (!tensor->extra)
    return;

  // Split tensors must always be set in their entirety
  GGML_ASSERT(offset == 0);
  GGML_ASSERT(size == ggml_nbytes(tensor));

  auto *extra = (ggml_split_tensor_t *)tensor->extra;

  if (extra->split_dim < 0) {
    // Replicate: full tensor to each device
    GGML_ASSERT(ggml_is_contiguous(tensor));
    auto nbytes = ggml_nbytes(tensor);
    for (int i = 0; i < extra->n_device; ++i) {
      auto *split = extra->splits[i];
      if (!split || !split->buffer)
        continue;
      GGML_ASSERT(split->type == tensor->type);
      GGML_ASSERT(ggml_are_same_shape(tensor, split));
      split->buffer->iface.set_tensor(split->buffer, split, data, 0, nbytes);
    }
  } else if (extra->split_dim == 1) {
    // Split along dim 1 (rows of weight matrices)
    auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (tensor->ne[2] > 1) {
      // 3D tensor: gather rows from each ne[2] slice
      std::vector<char> host_buffer;
      int64_t ne1_offset = 0;

      for (int i = 0; i < extra->n_device; ++i) {
        auto *split = extra->splits[i];
        if (!split || !split->buffer)
          continue;

        auto dev_size = ggml_nbytes(split);
        if (host_buffer.size() < dev_size)
          host_buffer.resize(dev_size);

        for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
          auto *dst = host_buffer.data() + i02 * split->ne[1] * row_size;
          auto *src = (const char *)data + i02 * tensor->nb[2] +
                      ne1_offset * tensor->nb[1];
          memcpy(dst, src, split->ne[1] * row_size);
        }

        split->buffer->iface.set_tensor(split->buffer, split,
                                        host_buffer.data(), 0, dev_size);
        ne1_offset += split->ne[1];
      }
    } else {
      // 2D tensor: contiguous row range per device
      size_t cur_offset = 0;
      for (int i = 0; i < extra->n_device; ++i) {
        auto *split = extra->splits[i];
        if (!split || !split->buffer)
          continue;

        auto dev_size = ggml_nbytes(split);
        const char *buf_host = (const char *)data + cur_offset;
        split->buffer->iface.set_tensor(split->buffer, split, buf_host, 0,
                                        dev_size);
        cur_offset += dev_size;
      }
    }
  } else if (extra->split_dim == 0) {
    // Split along dim 0 (columns)
    auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    int nrows = ggml_nrows(tensor);
    std::vector<char> host_buffer;

    int64_t ne0_offset = 0;
    for (int i = 0; i < extra->n_device; ++i) {
      auto *split = extra->splits[i];
      if (!split || !split->buffer)
        continue;

      auto split_row_size = ggml_row_size(split->type, split->ne[0]);
      auto dev_size = (size_t)nrows * split_row_size;
      if (host_buffer.size() < dev_size)
        host_buffer.resize(dev_size);

      auto src_col_offset = ggml_row_size(tensor->type, ne0_offset);
      for (int r = 0; r < nrows; ++r) {
        auto *dst = host_buffer.data() + r * split_row_size;
        auto *src = (const char *)data + r * row_size + src_col_offset;
        memcpy(dst, src, split_row_size);
      }

      split->buffer->iface.set_tensor(split->buffer, split, host_buffer.data(),
                                      0, dev_size);
      ne0_offset += split->ne[0];
    }
  } else {
    fprintf(stderr, "%s: not implemented for split dim %d\n", __func__,
            extra->split_dim);
    GGML_ABORT("fatal error");
  }
}

GGML_CALL static void
ggml_backend_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                     const ggml_tensor *tensor, void *data,
                                     size_t offset, size_t size) {
  GGML_UNUSED(buffer);
  if (!tensor->extra)
    return;

  auto *extra = (ggml_split_tensor_t *)tensor->extra;
  for (int i = 0; i < extra->n_device; ++i) {
    if (extra->splits[i] && extra->splits[i]->buffer) {
      auto *split = extra->splits[i];
      split->buffer->iface.get_tensor(split->buffer, split, data, offset, size);
      return;
    }
  }
}

GGML_CALL static void
ggml_backend_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
  auto *ctx = (ggml_backend_split_buffer_context *)buffer->context;
  for (auto *buf : ctx->sub_buffers) {
    if (buf) {
      ggml_backend_buffer_clear(buf, value);
    }
  }
}

static struct ggml_backend_buffer_i ggml_backend_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_split_buffer_clear,
    /* .reset           = */ NULL,
};

// ============================================================================
// Buffer type interface
// ============================================================================

GGML_CALL static const char *
ggml_backend_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
  GGML_UNUSED(buft);
  return "Metal_RPC_Split";
}

GGML_CALL static ggml_backend_buffer_t
ggml_backend_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                            size_t size) {
  auto *type_ctx = (ggml_backend_split_buffer_type_context *)buft->context;
  auto *buf_ctx = new ggml_backend_split_buffer_context();
  buf_ctx->type_ctx = type_ctx;

  return ggml_backend_buffer_init(buft, ggml_backend_split_buffer_interface,
                                  buf_ctx, size);
}

GGML_CALL static size_t
ggml_backend_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
  auto *type_ctx = (ggml_backend_split_buffer_type_context *)buft->context;
  size_t max_align = 0;
  for (int i = 0; i < type_ctx->n_device; ++i) {
    size_t align = ggml_backend_buft_get_alignment(type_ctx->device_bufts[i]);
    if (align > max_align)
      max_align = align;
  }
  return max_align;
}

GGML_CALL static size_t
ggml_backend_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                              const ggml_tensor *tensor) {
  GGML_UNUSED(buft);
  auto *extra = (ggml_split_tensor_t *)tensor->extra;
  if (!extra) {
    return ggml_nbytes(tensor);
  }

  size_t total = 0;
  for (int i = 0; i < extra->n_device; ++i) {
    if (extra->splits[i]) {
      total += ggml_nbytes(extra->splits[i]);
    }
  }
  return total;
}

GGML_CALL static bool
ggml_backend_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
  GGML_UNUSED(buft);
  return false;
}

static struct ggml_backend_buffer_type_i
    ggml_backend_split_buffer_type_interface = {
        /* .get_name         = */ ggml_backend_split_buffer_type_name,
        /* .alloc_buffer     = */ ggml_backend_split_buffer_type_alloc_buffer,
        /* .get_alignment    = */ ggml_backend_split_buffer_type_get_alignment,
        /* .get_max_size     = */ NULL,
        /* .get_alloc_size   = */ ggml_backend_split_buffer_type_get_alloc_size,
        /* .is_host          = */ ggml_backend_split_buffer_type_is_host,
};

// ============================================================================
// Public API
// ============================================================================

bool ggml_backend_buffer_is_metal_rpc_split(ggml_backend_buffer_t buffer) {
  return buffer && buffer->iface.get_name == ggml_backend_split_buffer_get_name;
}

ggml_backend_buffer_type_t ggml_backend_metal_rpc_split_buffer_type(
    ggml_backend_buffer_type_t *device_bufts, int n_devices,
    const float *tensor_split) {
  static ggml_backend_split_buffer_type_context ctx;
  static ggml_backend_buffer_type buft = {
      /* .iface   = */ ggml_backend_split_buffer_type_interface,
      /* .context = */ &ctx,
  };

  ctx.n_device = n_devices;
  ctx.device_bufts.assign(device_bufts, device_bufts + n_devices);
  ctx.splits.assign(tensor_split, tensor_split + n_devices);

  return &buft;
}
