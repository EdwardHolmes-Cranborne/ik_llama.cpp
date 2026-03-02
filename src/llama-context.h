#pragma once

#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-sampling.h"

struct llama_model;

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

struct llama_kv_cell {
  llama_pos pos = -1;
  llama_pos delta = 0;
  int32_t src = 0; // used by recurrent state models to copy states

  std::set<llama_seq_id> seq_id;

  bool has_seq_id(const llama_seq_id &id) const {
    return seq_id.find(id) != seq_id.end();
  }

  bool is_empty() const { return seq_id.empty(); }

  bool is_same_seq(const llama_kv_cell &other) const {
    return seq_id == other.seq_id;
  }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
  bool has_shift = false;
  bool do_defrag = false;
  bool do_copy   = false;
  bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
  bool hybrid    = false;
  bool v_trans   = true;  // the value tensor is transposed

  // Note: The value of head isn't only used to optimize searching
  // for a free KV slot. llama_decode_internal also uses it, so it
  // cannot be freely changed after a slot has been allocated.
  uint32_t head = 0;
  uint32_t size = 0;
  uint32_t used = 0; // used cells (i.e. at least one seq_id)

  // computed before each graph build
  uint32_t n = 0;

  ggml_type type_k = GGML_TYPE_F16;
  ggml_type type_v = GGML_TYPE_F16;

  std::vector<llama_kv_cell> cells;

  std::vector<struct ggml_tensor *> k_l; // per layer
  std::vector<struct ggml_tensor *> v_l;
  std::vector<struct ggml_tensor *> s_l; // per layer recurrent state storage (Qwen3Next)

  std::vector<llama_split_tensor> split_k_l;
  std::vector<llama_split_tensor> split_v_l;

  std::vector<struct ggml_context *> ctxs;
  std::vector<ggml_backend_buffer_t> bufs;

  size_t total_size() const {
    size_t size = 0;
    for (ggml_backend_buffer_t buf : bufs) {
      size += ggml_backend_buffer_get_size(buf);
    }
    return size;
  }

  ~llama_kv_cache() {
    for (struct ggml_context *ctx : ctxs) {
      ggml_free(ctx);
    }
    for (ggml_backend_buffer_t buf : bufs) {
      ggml_backend_buffer_free(buf);
    }
  }
};

struct llama_control_vector {
  std::vector<struct ggml_tensor *> tensors; // per layer
  std::vector<struct ggml_context *> ctxs;
  std::vector<ggml_backend_buffer_t> bufs;

  int32_t layer_start = -1;
  int32_t layer_end = -1;

  struct ggml_tensor *tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end ||
        (size_t)il >= tensors.size()) {
      return nullptr;
    }
    return tensors[il];
  }

  struct ggml_tensor *apply_to(struct ggml_context *ctx,
                               struct ggml_tensor *cur, int il) const {
    ggml_tensor *layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
      cur = ggml_add(ctx, cur, layer_dir);
    }
    return cur;
  }

  ~llama_control_vector() {
    for (struct ggml_context *ctx : ctxs) {
      ggml_free(ctx);
    }
    for (ggml_backend_buffer_t buf : bufs) {
      ggml_backend_buffer_free(buf);
    }
  }
};

struct llama_context {

  llama_context(const llama_model &model);

  ~llama_context();

  const struct llama_model &model;

  struct llama_cparams cparams;
  struct llama_sampling sampling;
  struct llama_kv_cache kv_self;
  struct llama_control_vector cvec;

  std::vector<float> scale_data;

  std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;

  std::vector<ggml_backend_t> backends;
#ifdef GGML_USE_METAL
  ggml_backend_t backend_metal = nullptr;
#endif
#ifdef GGML_USE_BLAS
  ggml_backend_t backend_blas = nullptr;
#endif
  ggml_backend_t backend_cpu = nullptr;

  bool has_evaluated_once = false;

  int64_t t_start_us;
  int64_t t_load_us;
  int64_t t_p_eval_us = 0;
  int64_t t_eval_us = 0;

  int64_t t_compute_start_us = 0;
  int64_t n_queued_tokens = 0;

  int32_t n_p_eval =
      0; // number of tokens in eval calls for the prompt (with batch size > 1)
  int32_t n_eval = 0; // number of eval calls

  // host buffer for the model output (logits and embeddings)
  ggml_backend_buffer_t buf_output = nullptr;

  // decode output (2-dimensional array: [n_outputs][n_vocab])
  size_t logits_size = 0; // capacity (of floats) for logits
  float *logits = nullptr;

  std::vector<int32_t> output_ids; // map batch token positions to ids of the
                                   // logits and embd buffers
  size_t output_size =
      0; // capacity (of tokens positions) for the output buffers
  int32_t n_outputs = 0; // number of actually-used outputs in the current
                         // ubatch or last logical batch

  bool logits_all = false;

  // embeddings output (2-dimensional array: [n_outputs][n_embd])
  // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
  size_t embd_size = 0; // capacity (of floats) for embeddings
  float *embd = nullptr;

  // sequence embeddings output (map of [n_embd] vectors)
  // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
  std::map<llama_seq_id, std::vector<float>> embd_seq;

  // whether we are computing encoder output or decoder output
  bool is_encoding = false;

  // output of the encoder part of the encoder-decoder models
  std::vector<float> embd_enc;
  std::vector<std::set<llama_seq_id>> seq_ids_enc;

  // memory buffers used to evaluate the model
  std::vector<uint8_t> buf_compute_meta;
  ggml_backend_sched_t sched = nullptr;

  ggml_abort_callback abort_callback = nullptr;
  void *abort_callback_data = nullptr;

  const float * draft_input_hidden_state = nullptr;

  // input tensors
  struct ggml_tensor *inp_tokens;          // I32 [n_batch]
  struct ggml_tensor *inp_embd;            // F32 [n_embd, n_batch]
  struct ggml_tensor *inp_pos;             // I32 [n_batch]
  struct ggml_tensor *inp_out_ids;         // I32 [n_outputs]
  struct ggml_tensor *inp_KQ_mask;         // F32 [kv_size, n_batch]
  struct ggml_tensor *inp_KQ_mask_swa;     // F32 [kv_size, n_batch]
  struct ggml_tensor *inp_K_shift;         // I32 [kv_size]
  struct ggml_tensor *inp_mean;            // F32 [n_batch, n_batch]
  struct ggml_tensor *inp_cls;             // I32 [n_batch]
  struct ggml_tensor *inp_s_copy;          // I32 [kv_size]
  struct ggml_tensor *inp_s_mask;          // F32 [1, n_kv]
  struct ggml_tensor *inp_s_seq;           // I32 [n_kv, n_batch]
  struct ggml_tensor *inp_s_seq_qnext;     // I32 [1, n_batch]
  struct ggml_tensor *inp_pos_bucket;      // I32 [n_batch|n_kv, n_batch]
  struct ggml_tensor *inp_embd_enc;        // F32 [n_embd, n_outputs_enc]
  struct ggml_tensor *inp_KQ_mask_cross;   // F32 [n_outputs_enc, n_batch]
  struct ggml_tensor *inp_scale = nullptr; // F32 [n_tokens]
  struct ggml_tensor *inp_mtp_states = nullptr;

  ggml_backend_t ggml_backend_by_name(const char *name);

  struct Prev;
  std::unique_ptr<Prev> prev;

  void reset_scheduler();
  bool can_reuse_graph(const llama_batch &u_batch);

  struct CacheCopy {
    ggml_tensor *cpy = nullptr;
    size_t step = 0;
  };
  std::vector<CacheCopy> cache_copies;

  bool update_cache_copies();
  bool prepare_mtp_graph_inputs(
      struct llama_context & lctx);
  void set_mtp_op_type(llama_mtp_op_type value);

  // === RTX Accelerated Prefill Streaming State ===
  bool prefill_streaming = false;
  bool prefill_telemetry = true;
  bool prefill_overlap = false;
  int prefill_buffers = 2;
  int prefill_prefetch = 1;
  size_t prefill_slab_bytes = 16 * 1024 * 1024;
  int prefill_min_stream_batch_tokens = -1;

  // Prefill->decode handoff planner/runtime controls
  int32_t prefill_decode_mode = LLAMA_PREFILL_DECODE_MODE_AUTO;
  int32_t prefill_transport_mode = LLAMA_PREFILL_TRANSPORT_MODE_DISABLED;
  int32_t prefill_execution_mode = LLAMA_PREFILL_EXECUTION_MODE_COUPLED;
  int32_t decode_gpu_layers_hint = -1;
  int32_t decode_remote_layers_hint = 0;
  int32_t decode_remote_nodes_hint = 1;
  int32_t prefill_transport_chunk_bytes = 4 * 1024 * 1024;
  bool prefill_decode_transport_required = false;

  std::string decode_remote_ranges;
  std::string decode_remote_failover_policy = "reroute";
  std::string prefill_transport_session_dir;
  std::string kv_transport;
  std::string tb_direct_endpoint;
  std::string kv_host;
  int32_t kv_port = 0;
  int32_t kv_streams = 0;
  int32_t kv_stream_chunk_bytes = 0;
  int32_t kv_max_inflight_bytes = 0;
  int32_t kv_socket_send_buf = 0;
  int32_t kv_socket_recv_buf = 0;
  std::string kv_bind_addrs;
  std::string kv_peer_addrs;
  std::string kv_balance;
  bool kv_transport_fallback = false;

  // Per-layer callbacks for streaming prefill weight upload
  std::function<void(int, int)> pre_layer_cb;
  std::function<void(int, int)> post_layer_cb;

  // Per-layer callback for ANE FFN dispatch (fires when ffn_inp-N is computed)
  // Signature: (layer_idx, n_layers, ffn_inp_tensor)
  std::function<void(int, int, struct ggml_tensor *)> pre_ffn_cb;

  // Tensor pointer for last completed l_out (set by eval callback for post_layer_cb)
  struct ggml_tensor * last_l_out_tensor = nullptr;

  // ANE GPU+ANE split prefill state and config
  struct ane_dispatch_ctx * ane_ctx = nullptr;
  bool ane_prefill_active = false;
  bool prefill_ane = false;
  float prefill_ane_ratio = 0.5f;
  bool prefill_ane_validate = false;
  std::string prefill_ane_cache;

  // Per-layer compute times (populated by build context)
  std::vector<float> last_layer_compute_times_ms;

  struct LayerEvalCallbacksState {
    ggml_backend_sched_eval_callback user_cb = nullptr;
    void * user_cb_user_data = nullptr;
    bool active = false;
    int32_t n_layers = 0;
    int32_t current_layer = -1;
    int64_t current_layer_start_us = 0;
    bool first_layer_started = false;
    std::unordered_set<const ggml_tensor *> user_observed_nodes;
  } layer_eval_callbacks;
};
