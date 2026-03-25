# Plan: Streaming Prefill + Hybrid Decode Port to ik_llama

**Source:** RTX_ACCELERATED_MAC_PREFILL_LLAMA (`prefill_llama.cpp/`)
**Target:** ik_llama.cpp
**Date:** 2026-03-24

## Goal

Port the full RTX streaming prefill + Mac decode hybrid pipeline from the RTX project into ik_llama.cpp. The end result: ik_llama supports layer-major streaming prefill on an RTX host, KV artifact export over TCP to a Mac (or other remote), autonomous decode on the remote, token streaming back, and KV sync — all compatible with ik_llama's graph split multi-GPU infrastructure.

### Target Architecture

```
┌─────────────────────────────────────────────────┐
│  RTX Host (Windows, 1+ GPUs, WDDM)             │
│                                                  │
│  llama-server (Web UI)                          │
│    ↓ prompt                                     │
│  Prefill Strategy Selector                      │
│    ↓ decision: RTX streaming or Mac standard    │
│  Layer-Major Streaming Prefill                  │
│    • single-ubatch, per-layer eval callbacks    │
│    • KV-RAM mode (optional VRAM savings)        │
│    • graph-split compatible (multi-GPU RTX)     │
│    ↓ KV artifact                                │
│  KV Transport (TCP, TBP1 protocol)             │
│    ↓ TCP to Mac                                 │
│  Token Stream Client ← reads decoded tokens     │
│    ↓ tokens                                     │
│  KV Sync (batch-prefill tokens into local KV)   │
└─────────────────────────────────────────────────┘
          │ TCP (KV)          ▲ TCP (tokens)
          ▼                   │
┌─────────────────────────────────────────────────┐
│  Mac Host (macOS, Metal, large unified RAM)     │
│                                                  │
│  Decode Receiver (standalone daemon)            │
│    • listens on TCP for KV artifact             │
│    • imports KV via scatter map                 │
│    • runs autonomous decode (temp sampling)     │
│    ↓ tokens                                     │
│  Token Stream Server → sends tokens to RTX      │
│    • JSON-line protocol over TCP                │
│    • session loop (multi-turn without restart)  │
└─────────────────────────────────────────────────┘
```

## Phases Overview

| Phase | Name | Scope | Depends On |
|-------|------|-------|------------|
| **A** | Layer-Major Foundation | Single-GPU layer-major decode, per-layer callbacks, KV-RAM | — |
| **B** | Graph Split Compatibility | Thread-safe callbacks, per-device weight streaming, split KV-RAM | A |
| **C** | Async Multi-GPU Overlap | Pipeline weight upload, overlap KV transfers across devices | B |
| **D** | KV Artifact & Serialization | Wire format, export/import, scatter map, MLA support | A |
| **E** | TCP Transport (TBP1) | KV sender on RTX, KV receiver on Mac, chunked transfer | D |
| **F** | Decode Receiver | Standalone Mac daemon: load model, import KV, autonomous decode | D, E |
| **G** | Token Stream Relay | Mac→RTX token streaming, JSON-line protocol, ring buffer | F |
| **H** | KV Sync | Batch-prefill received tokens into RTX local KV cache | G |
| **I** | Prefill Strategy Selector | Auto-choose RTX streaming vs Mac-standard based on prompt length | A |
| **J** | Server & Web UI Integration | Wire into llama-server, session management, error recovery | A-I |
| **K** | Multi-GPU RTX Host | Multiple RTX GPUs for streaming prefill with graph splits | B, C, J |

```
A ──→ B ──→ C ──→ K (multi-GPU RTX)
│           │
├──→ D ──→ E ──→ F ──→ G ──→ H
│                              │
├──→ I                         │
│                              ▼
└──────────────────────────→  J (server integration)
```

---

## Phase A: Single-GPU Layer-Major Foundation

**Goal:** `llama_decode_internal` can dispatch via layer-major path on single GPU with per-layer callbacks.

### A.1: Per-layer callback infrastructure

**Files:** `src/llama-context.h`, `src/llama.cpp`

Add to `llama_context`:
```cpp
using layer_callback_fn = std::function<void(int il, int n_layer)>;

layer_callback_fn per_layer_pre_cb;
layer_callback_fn per_layer_post_cb;
std::vector<float> last_layer_compute_times_ms;

void set_per_layer_callbacks(layer_callback_fn pre, layer_callback_fn post);
bool has_per_layer_callbacks() const;
```

### A.2: graph_compute_per_layer()

**New function in** `src/llama.cpp` (or `src/llama-layer-major.cpp`):

1. Scan graph nodes for `-{il}` suffix → build `first_idx[il]`, `last_idx[il]`
2. Create `per_layer_eval_state` struct with boundary info
3. Register eval callback via `ggml_backend_sched_set_eval_callback()`
4. Callback returns `true` at layer boundaries to force scheduler sync
5. Pre-compute: fire `pre_cb(il, n_layer)` on first node of layer il
6. Post-compute: fire `post_cb(il, n_layer)` on last node, record timing
7. Single `ggml_backend_sched_graph_compute_async(sched, gf)` call

### A.3: decode_layer_major() single-ubatch fast path

**New function:** `llama_decode_layer_major(llama_context & lctx, llama_batch & batch)`

1. Validate batch, update KV cache metadata
2. Build graph via `llm_build_context::llama_build_graph()`
3. Force all ops to GPU backend
4. `ggml_backend_sched_alloc_graph()`
5. Set inputs (positions, masks, embeddings)
6. `graph_compute_per_layer(gf, true, pre_cb, post_cb)`
7. Extract logits/embeddings
8. Update KV head pointer

### A.4: KV-RAM mode

```cpp
struct kv_ram_layer {
    std::vector<uint8_t> k_ram, v_ram;
    void * orig_k_data, * orig_v_data;
    size_t k_bytes, v_bytes;
};
```

- Allocate per-layer RAM buffers + single GPU staging buffer
- Redirect all KV tensors to staging buffer, free full GPU KV buffer
- `pre_cb(il)`: H2D upload layer il K/V from RAM → staging
- `post_cb(il)`: D2H download staging → RAM

### A.5: Wire into llama_decode_internal

```cpp
if (lctx.has_per_layer_callbacks() || cparams.layer_major) {
    return llama_decode_layer_major(lctx, batch_all);
}
```

### A.6: Public API

**File:** `include/llama.h`

```cpp
LLAMA_API void llama_set_layer_callbacks(
    struct llama_context * ctx,
    llama_layer_callback pre_layer,
    llama_layer_callback post_layer,
    void * user_data);

LLAMA_API bool llama_enable_kv_ram(struct llama_context * ctx);
```

### A.7: Graph builder GPU-force

In `graph_get_cb()`: when `has_per_layer_callbacks()`, force all ops to first non-CPU backend. Prevents unnecessary CPU-GPU copies for embedding/norm ops.

---

## Phase B: Graph Split Compatibility

**Goal:** Layer-major works with `--split-mode graph` on multi-GPU setups.

### B.1: Thread-safe eval callback state

```cpp
std::vector<std::atomic<int>> layer_state;  // 0=pending, 1=started, 2=completed
```

`compare_exchange_strong` ensures pre_cb fires exactly once per layer under parallel split execution.

### B.2: Device-aware callbacks

```cpp
using layer_callback_fn = std::function<void(int il, int n_layer, int device_id)>;
```

Eval callback determines `device_id` from current split's `backend_id`. Weight streaming uploads per-device slices via `tensor->extra->splits[device_id]`.

### B.3: Per-split KV-RAM staging

- One staging buffer per device (or shared with device-indexed offsets)
- Pre-layer uploads `split_k_l[il].tensor_splits[device_id]`
- Post-layer downloads back to per-device RAM storage

### B.4: Split-aware layer boundary detection

Atomic counters: `devices_started[il]`, `devices_completed[il]` — fire pre_cb on first device to start layer, post_cb when last device completes.

### B.5: Validation

Test with 2+ GPUs, `--split-mode graph`:
- Layer boundaries correct across splits
- Per-split weight upload works
- Output matches standard decode path

---

## Phase C: Async Multi-GPU Overlap

**Goal:** Pipeline weight upload and KV transfers across devices for maximum throughput.

### C.1: Pipeline weight upload

Double-buffer weight staging per device. Upload layer il+1 on idle stream while il computes.

### C.2: Overlap KV transfers

Async memcpy in post-layer callback. Pre-layer waits on previous transfer completion.

### C.3: Benchmark

| Config | Metric |
|--------|--------|
| Standard decode | tok/s baseline |
| Layer-major, no KV-RAM | tok/s |
| Layer-major + KV-RAM | tok/s + VRAM saved |
| Layer-major + graph splits (multi-GPU) | tok/s |
| Layer-major + graph splits + async overlap | tok/s |

---

## Phase D: KV Artifact Serialization

**Goal:** KV cache can be serialized to a portable binary artifact and deserialized on a different host.

### D.1: KV artifact wire format

**New files:** `src/llama-kv-artifact.h`, `src/llama-kv-artifact.cpp`

Port from RTX source:
- Magic: `"KVARTIF1"` (8 bytes)
- Base header (44 bytes): format version, n_layers, n_ctx, token_count, type_k, type_v, flags
- Extension header (16 bytes, v1.1): last_token_id, v_trans, is_mla, model_fingerprint
- CRC32 trailer
- Pragma pack(push,1) for wire alignment

API:
```cpp
int  llama_kv_artifact_write(llama_context * ctx, const char * path, int token_count);
int  llama_kv_artifact_read(llama_context * ctx, const char * path);
int  llama_kv_artifact_read_mem(llama_context * ctx, const uint8_t * buf, size_t len);
uint32_t llama_kv_artifact_crc32(const uint8_t * data, size_t len);
```

### D.2: KV import scatter map

**New files:** `src/llama-kv-import.h`, `src/llama-kv-import.cpp`

Port from RTX source:
- `kv_scatter_entry`: layer_idx, tensor_kind (K=0, V=1), artifact_offset, tensor_offset, bytes
- `kv_scatter_map`: vector of entries, total_payload_bytes
- `kv_scatter_map_build()`: builds scatter map from model info and token_count
- Handles K and V layouts (non-transposed, transposed V)
- MLA support: V stored in K tensor, no separate V scatter

### D.3: MLA-aware serialization

Kimi K2.5 and similar MLA models store compressed K+V in K tensor only (V is nullptr). Serialization must:
- Set `is_mla` flag in header
- Only write/read K tensors
- Scatter map skips V entries when `is_mla`

---

## Phase E: TCP Transport (TBP1 Protocol)

**Goal:** KV artifacts transfer reliably over TCP between RTX and Mac hosts.

### E.1: TBP1 wire protocol

**New files:** `src/llama-tb-transport.h`, `src/llama-tb-transport.cpp`

Port from RTX source:
- Message types: HELLO, SESSION_START, KV_SEG_BEGIN, KV_CHUNK, KV_SEG_END, KV_DONE
- Magic: `0x54425031` ("TBP1")
- Chunked transfer with configurable chunk size and max inflight bytes
- Winsock (Windows) / POSIX (macOS/Linux) socket abstraction

```cpp
struct llama_tb_transfer_options {
    std::string host;
    int port = 9100;
    size_t chunk_size;
    size_t max_inflight_bytes;
    int socket_sndbuf, socket_rcvbuf;
};

struct llama_tb_transfer_result {
    size_t bytes_sent, chunks_sent;
    double transfer_ms, throughput_gbps;
};

bool llama_tb_transport_send_artifact(
    const uint8_t * artifact, size_t len,
    const llama_tb_transfer_options & opts,
    llama_tb_transfer_result * result);
```

### E.2: KV receiver (listener)

**New files:** `src/llama-kv-receiver.h`, `src/llama-kv-receiver.cpp`

Port from RTX source:
- TCP listener on configurable bind_host:port
- `accept_timeout_ms` (2 min), `recv_timeout_ms` (5 min)
- CRC32 verification
- Populates buffer suitable for `llama_kv_artifact_read_mem()`

```cpp
struct llama_kv_receiver_config {
    std::string bind_host = "0.0.0.0";
    int port = 9100;
    int accept_timeout_ms = 120000;
    int recv_timeout_ms = 300000;
    bool verify_crc = true;
};

bool llama_kv_receiver_accept_artifact(
    const llama_kv_receiver_config & cfg,
    std::vector<uint8_t> & artifact_out,
    llama_kv_receiver_result * result);
```

---

## Phase F: Decode Receiver (Mac Daemon)

**Goal:** Standalone executable on Mac that loads a model, listens for KV artifacts, imports them, and runs autonomous decode.

### F.1: Decode receiver binary

**New file:** `tools/decode-receiver/main.cpp`

Port from RTX source:
- Modes:
  - **File mode** (`--kv-artifact PATH`): read KV from file (testing)
  - **Network mode** (`--kv-port PORT`): listen for KV over TCP
- Flow:
  1. Load model + create context
  2. Listen/read KV artifact
  3. Validate artifact against model (fingerprint, n_layers, types)
  4. Import KV via scatter map into local KV cache
  5. Run autoregressive decode (temp sampling)
  6. Stream tokens via Token Stream Server (Phase G)
  7. Loop back to step 2 for next turn (session loop)

### F.2: CLI flags

```
-m, --model PATH          Model file (GGUF)
--kv-artifact PATH        File mode KV artifact
--kv-port PORT            Network listener port (default: 9100)
--kv-host HOST            Bind address (default: 0.0.0.0)
--token-stream-host HOST  Token relay host
--token-stream-port PORT  Token relay port (default: 9101)
-c, --ctx-size N          Context size (default: 8192)
-ngl, --n-gpu-layers N    GPU layers (default: 99)
-n, --predict N           Max decode tokens (default: 2048)
--temp F                  Temperature (default: 0.7)
--one-shot                Exit after first decode
--mlock                   Lock model in memory
```

### F.3: Session management

- KV cache cleared between turns
- Receiver loops back to accept without restart
- `--one-shot` flag for testing (exit after first decode)

---

## Phase G: Token Stream Relay

**Goal:** Mac streams decoded tokens back to RTX in real-time over TCP.

### G.1: Token message protocol

**New files:** `src/llama-token-stream.h`, `src/llama-token-stream.cpp`

Port from RTX source:
- JSON-line protocol (one JSON object per line, `'\n'` terminated)
- Message types:
  - `llama_token_msg`: token_id, text, pos, timestamp_us, done flag, decode_tok_s
  - `llama_token_handshake`: version, status, model_fingerprint, token_count

### G.2: Token stream server (Mac-side)

Runs inside decode receiver:
```cpp
struct llama_token_stream_server {
    bool start(const std::string & host, int port);
    bool wait_for_client(int timeout_ms);
    bool send_handshake(const llama_token_handshake & hs);
    void enqueue(const llama_token_msg & msg);  // push to ring, sender thread drains
    void send_done();
    void stop();
};
```

- 16k lock-free SPSC ring buffer (`llama_token_ring`)
- Async sender thread drains ring → TCP
- Push never blocks (overwrites oldest on overflow, tracks `dropped_count`)

### G.3: Token stream client (RTX-side)

```cpp
struct llama_token_stream_client {
    bool connect(const std::string & host, int port, int retries, int retry_ms);
    bool read_handshake(llama_token_handshake & hs, int timeout_ms);
    bool read_msg(llama_token_msg & msg, int timeout_ms);
    void disconnect();
};
```

- Winsock/POSIX socket abstraction
- JSON parsing per line

---

## Phase H: KV Sync

**Goal:** RTX batch-prefills received tokens into its local KV cache so it stays in sync with the Mac's decode state.

### H.1: KV sync engine

As tokens stream back from Mac, batch them and prefill into local KV:

```cpp
struct llama_kv_sync {
    int batch_size = 512;          // tokens per sync batch
    std::vector<llama_token> pending;

    void push(llama_token tok);    // accumulate
    bool flush(llama_context * ctx); // batch-decode pending tokens into KV
};
```

- Accumulates tokens from token stream client
- When `pending.size() >= batch_size` or `done` received, batch-decode into local KV
- After sync, RTX KV cache matches Mac's state — enables next turn without re-prefilling full history

---

## Phase I: Prefill Strategy Selector

**Goal:** Auto-choose RTX streaming prefill vs Mac-standard prefill based on prompt length and hardware estimates.

### I.1: Strategy logic

**New files:** `src/llama-prefill-strategy.h`, `src/llama-prefill-strategy.cpp`

Port from RTX source — pure function, no I/O:

```cpp
struct llama_prefill_strategy_params {
    int crossover_tokens = 0;        // 0 = auto-compute
    float mac_prefill_tok_s = 300;
    float pcie_bandwidth_gbs = 64;   // PCIe 5.0
    float rtx_compute_tok_s = 2000;
    float rtx_streaming_floor_ms;    // model_size / PCIe_bw
};

struct llama_prefill_strategy_decision {
    enum { AUTO, MAC, RTX } chosen;
    int crossover_tokens;
    float estimated_mac_ms, estimated_rtx_ms;
    std::string reason;
};

llama_prefill_strategy_decision llama_prefill_strategy_select(
    int n_tokens,
    const llama_prefill_strategy_params & params);
```

Crossover = `model_size_gb / pcie_bandwidth_gbs * mac_tok_s`. Below crossover → Mac is faster. Above → RTX streaming is faster. Minimum threshold 100 tokens.

---

## Phase J: Server & Web UI Integration

**Goal:** llama-server orchestrates the full hybrid flow, users chat via browser.

### J.1: Decode handoff infrastructure

**New files:** `src/llama-decode-handoff.h`, `src/llama-decode-handoff.cpp`

Port from RTX source:
- `llama_decode_handoff_runtime`: transport config, layer split hints, token relay config, KV sync batch size
- `llama_decode_handoff_plan`: resolved execution mode, transport enable/disable, layer map
- `llama_decode_executor_i`: virtual interface for executor implementations
  - `begin_session()`
  - `publish_kv_artifact()` — serialize + send KV to Mac
  - `relay_tokens()` — receive tokens from Mac, sync KV locally
- `llama_decode_executor_create()`: factory from plan

### J.2: Server handoff trigger

**File:** `tools/server/server-context.cpp`

After streaming prefill completes, trigger handoff:
```cpp
if (ctx->prefill_streaming && handoff_plan.transport_enabled) {
    llama_decode_trigger_handoff(ctx);
}
```

- Environment variable `LLAMA_PREFILL_WF=1` enables the workflow
- Executor publishes KV artifact, then enters token relay loop
- Tokens relayed back appear as SSE chunks to the web UI client

### J.3: CLI flags for server

```
--prefill-streaming           Enable RTX streaming prefill
--prefill-strategy {auto|mac|rtx}  Force prefill strategy
--prefill-decode-mode {auto|cpu_kv|gpu_kv|hybrid|split}
--kv-transport-host HOST      Mac KV receiver host
--kv-transport-port PORT      Mac KV receiver port (default: 9100)
--token-stream-host HOST      Mac token stream host
--token-stream-port PORT      Token stream port (default: 9101)
--kv-sync-batch-size N        Tokens per KV sync batch (default: 512)
--decode-ngl N                GPU layers for hybrid decode
```

### J.4: Error recovery

- Accept/recv timeouts with configurable values
- Fallback to local decode if Mac unavailable
- Reconnection with backoff for token stream drops
- Session resume without full re-prefill (KV sync keeps RTX in sync)

### J.5: Web UI

The existing ik_llama server web UI (Svelte) needs minimal changes — SSE streaming already works. The hybrid pipeline is transparent to the frontend; tokens appear the same whether decoded locally or relayed from Mac.

Minor additions:
- Status indicator showing prefill mode (streaming/standard)
- Prefill timing in response metadata
- Error display for handoff failures

---

## Phase K: Multi-GPU RTX Host

**Goal:** Multiple RTX GPUs on the host cooperate for streaming prefill, fully integrated with ik_llama's graph split infrastructure.

### K.1: Multi-GPU weight streaming

With `--split-mode graph` and multiple GPUs:
- Each GPU gets a portion of each layer's weight tensors (via `ggml_split_tensor_t`)
- Layer-major callbacks stream per-device weight slices: `tensor->extra->splits[device_id]`
- Double-buffer per device for pipelined upload

### K.2: Multi-GPU KV export

When exporting KV artifact from multi-GPU:
- Gather split KV from all devices: `split_k_l[il].tensor_splits[0..n_device-1]`
- Concatenate into unified artifact for transport
- Scatter map on receiver side doesn't need to know about source splits

### K.3: Multi-GPU prefill + handoff

Flow:
1. Build graph with split markers (automatic from `build_std_attention`)
2. Layer-major with per-device callbacks streams weights to each GPU
3. Scheduler runs splits in parallel across GPUs (async mode)
4. After prefill, gather KV from all GPUs
5. Serialize unified KV artifact
6. Transport to Mac as usual

### K.4: Benchmark multi-GPU

Compare 1-GPU vs 2-GPU vs 4-GPU streaming prefill for:
- Kimi K2.5 at 8K, 32K prompt lengths
- Per-layer timing breakdown showing multi-GPU speedup
- Transport overhead (gather KV from multiple GPUs)

---

## Complete File Inventory

### New files to create in ik_llama

| File | Phase | Description |
|------|-------|-------------|
| `src/llama-layer-major.cpp` | A | Layer-major decoder + graph_compute_per_layer |
| `src/llama-kv-artifact.h` | D | KV artifact wire format |
| `src/llama-kv-artifact.cpp` | D | KV serialization/deserialization |
| `src/llama-kv-import.h` | D | Scatter map for KV placement |
| `src/llama-kv-import.cpp` | D | Scatter map builder |
| `src/llama-tb-transport.h` | E | TCP transport options/results |
| `src/llama-tb-transport.cpp` | E | TBP1 sender implementation |
| `src/llama-kv-receiver.h` | E | TCP listener config |
| `src/llama-kv-receiver.cpp` | E | TBP1 receiver implementation |
| `src/llama-token-stream.h` | G | Token message, ring buffer, server/client |
| `src/llama-token-stream.cpp` | G | JSON-line protocol, socket I/O |
| `src/llama-prefill-strategy.h` | I | Strategy selection |
| `src/llama-prefill-strategy.cpp` | I | Crossover computation |
| `src/llama-decode-handoff.h` | J | Handoff plan/runtime/executor interface |
| `src/llama-decode-handoff.cpp` | J | Plan builder, executor implementations |
| `tools/decode-receiver/main.cpp` | F | Mac decode daemon |
| `tools/decode-receiver/CMakeLists.txt` | F | Build config |

### Existing files to modify

| File | Phase | Changes |
|------|-------|---------|
| `src/llama-context.h` | A | Callback fields, kv_ram_layer, handoff state |
| `src/llama.cpp` | A, J | decode_layer_major, wire into decode_internal, API functions |
| `src/llama-build-context.cpp` | A | GPU-force in graph_get_cb |
| `src/llama-cparams.h` | A | `layer_major` flag, handoff params |
| `include/llama.h` | A, D, J | Public API surface, enums, typedefs |
| `tools/server/server-context.cpp` | J | Handoff trigger |
| `tools/server/server.cpp` | J | CLI flags |
| `CMakeLists.txt` | F | decode-receiver target |

---

## Risks & Mitigations

| Risk | Phase | Mitigation |
|------|-------|------------|
| Eval callback incompatible with split_mode_graph async | B | Phase A works without splits; Phase B adds thread-safe callbacks |
| Layer boundary detection breaks for new architectures | A | Uses same `-{il}` naming all ik_llama builders use |
| KV-RAM incompatible with split KV cache | B | Phase B.3 handles per-split staging explicitly |
| MLA serialization breaks for non-Kimi models | D | is_mla flag + V=nullptr guard; test with dense models too |
| Winsock/POSIX divergence in transport code | E | Abstract socket_fd_t type, WSAStartup guard, existing pattern from RTX |
| Token stream drops on slow network | G | Ring buffer overflow tracking, reconnection in Phase J |
| Multi-GPU KV gather overhead | K | Async gather overlapped with last layer compute |
| ik_llama upstream changes break integration | All | Keep changes modular in new files; minimize modifications to existing code |

## Execution Order

```
Phase A (single-GPU layer-major) ──────────────────────────────────────┐
  ↓                                                                    │
Phase B (graph split compat) ──→ Phase C (async overlap)               │
  ↓                                                                    │
Phase D (KV artifact) ──→ Phase E (TCP transport) ──→ Phase F (receiver)
                                                        ↓              │
                                                   Phase G (token stream)
                                                        ↓              │
Phase I (strategy selector)                        Phase H (KV sync)   │
  ↓                                                     ↓              │
  └────────────────────────────→ Phase J (server integration) ◄────────┘
                                        ↓
                                   Phase K (multi-GPU RTX host)
```

Phases A and D can start in parallel. Phases B/C and E/F/G/H are independent tracks that converge at J.

---
*Plan created: 2026-03-24*
*Updated: 2026-03-24 — added full handoff pipeline (D-J) and multi-GPU RTX host (K)*
