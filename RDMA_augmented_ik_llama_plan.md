# RDMA Augmented ik_llama Plan

## 1. Executive Summary

This plan adds **RDMA as an optional transport backend** to `ik_llama` multi-device execution, while preserving current TCP-based RPC compatibility.

Target outcome:
- Keep existing split scheduling (`layer`, `attn`, `graph`) unchanged.
- Replace or bypass slow payload movement paths with RDMA when available.
- Support per-OS providers with runtime capability negotiation and safe fallback.

Primary performance objective:
- Reduce inter-device tensor transfer latency and CPU overhead in `graph`/`attn` split modes where same-layer partial results are computed on multiple devices and merged.

---

## 2. Platform Constraints and Assumptions

1. **Thunderbolt 5 / USB4 is the link substrate, not RDMA by itself.**
   - RDMA requires a supported provider/driver/device path per endpoint OS.

2. **Mac<->Mac TB5 RDMA is the first delivery gate.**
   - exo is a strong confidence signal; still validate ik_llama graph-split behavior, collective paths, and soak stability.

3. **Cross-OS reliability is mandatory even when RDMA is unavailable.**
   - All unsupported provider pairs must degrade to deterministic TCP/QUIC fallback.

4. **Split semantics are preserved.**
   - `layer`, `attn`, and `graph` scheduling logic stays intact; this plan accelerates transport and reduction movement, not model math.

---

## 3. Current ik_llama Integration Points

Current multi-device behavior already exists and should be reused:

- Split modes:
  - `LLAMA_SPLIT_MODE_LAYER`
  - `LLAMA_SPLIT_MODE_ATTN`
  - `LLAMA_SPLIT_MODE_GRAPH`
- Tensor split/reduce mechanics:
  - `ggml_split_tensor_t`
  - `ggml_reduce(..., GGML_OP_ADD)`
- RPC transport currently on TCP sockets.

Key files:
- `/Users/edwardholmes/Documents/GitHub/AI_Things/ik_llama.cpp/src/llama.cpp`
- `/Users/edwardholmes/Documents/GitHub/AI_Things/ik_llama.cpp/src/llama-load-tensors.cpp`
- `/Users/edwardholmes/Documents/GitHub/AI_Things/ik_llama.cpp/src/llama-build-context.cpp`
- `/Users/edwardholmes/Documents/GitHub/AI_Things/ik_llama.cpp/ggml/src/ggml-rpc.cpp`
- `/Users/edwardholmes/Documents/GitHub/AI_Things/ik_llama.cpp/ggml/src/ggml-backend.cpp`

---

## 4. Scope

### In scope

- Add transport abstraction to decouple RPC logic from socket implementation.
- Add RDMA providers as optional backends.
- Runtime negotiation (`auto`, `rdma`, `tcp`) and fallback.
- Large-payload acceleration (`SET_TENSOR`, `GET_TENSOR`, `COPY_TENSOR`, graph input copies).
- Telemetry and benchmark gates.

### Out of scope (initial)

- Changing split math/scheduler semantics.
- Rewriting ggml graph partitioning.
- Guaranteeing direct Mac<->Windows RDMA in phase 1.

---

## 5. Requirements

### Functional requirements

1. Preserve existing TCP behavior with zero regressions when RDMA disabled.
2. Runtime transport selection per endpoint:
   - `auto` (prefer RDMA, fallback TCP)
   - `tcp`
   - `rdma` (hard fail if unavailable)
3. Capability negotiation and protocol versioning.
4. Fallback on transport failure without process crash.
5. Cross-OS interoperability at least via TCP fallback.

### Non-functional requirements

1. No data corruption; checksum/length validation on all bulk paths.
2. Stable long-run generation (>= 1 hour soak).
3. Observability: per-transfer metrics and mode logs.
4. No open-network insecure defaults beyond current RPC posture.

### Performance requirements (target)

1. P50/P95 transfer latency improvement for large tensors.
2. Lower CPU usage on sender/receiver during heavy split runs.
3. Net tok/s uplift for `graph`/`attn` split on RDMA-capable links.

---

## 6. Proposed Architecture

### 6.1 Transport abstraction layer

Add a pluggable transport API under ggml RPC:

- `ggml/include/ggml-transport.h`
- `ggml/src/ggml-transport.cpp`
- `ggml/src/ggml-transport-registry.cpp`

Conceptual interface:

- `connect(endpoint, opts)`
- `listen(bind, opts)`
- `send(ctrl_or_data, bytes)`
- `recv(ctrl_or_data, bytes)`
- `register_memory(ptr, len)` (optional capability)
- `deregister_memory(handle)`
- `write/read remote` (optional capability)
- `poll/completion`

### 6.2 Providers

- `tcp` provider (existing behavior, refactored).
- `rdma_verbs` provider (Linux first).
- `rdma_ndk` provider (Windows; provider dependent).
- `rdma_macos_tb5` provider (Mac; feasibility-gated).

### 6.3 Protocol model

Keep current control RPC semantics, add negotiated bulk path:

1. HELLO/feature exchange over control channel.
2. If both ends support same RDMA provider, open RDMA data channel.
3. For payloads above threshold, transfer over RDMA path.
4. Control ops and small payloads remain on control channel.
5. If RDMA path drops, fall back to TCP payload channel and continue.

### 6.4 Data plane shape

- Keep a small control channel for:
  - session setup,
  - capability negotiation,
  - memory-token exchange,
  - health/fallback signaling.
- Use RDMA data path for large payload transfers:
  - pre-registered buffer pool,
  - chunk descriptors and completion signaling,
  - provider-specific completion queues.
- Where provider supports it, use write-with-notify semantics (e.g., WRITE+IMM style) to reduce control round-trips.
- Keep CPU-side spin/poll policy configurable; do not hardcode busy-spin for all environments.

### 6.5 Why this architecture

- Minimal risk to existing scheduler behavior.
- Isolates OS/provider complexity.
- Enables phased delivery and partial availability.

---

## 7. OS and Interop Strategy

### Linux

- Preferred stack: `rdma-core`/verbs provider.
- Supports IB/RoCE capable NICs; can include USB4-attached RDMA devices if drivers expose them properly.

### Windows

- Use Windows RDMA provider (NDK/RDMA-capable NIC path).
- Requires validated hardware/driver matrix.

### macOS

- Treat as experimental provider until API/driver path is validated.
- Must gate by OS/hardware/runtime checks.

### Cross-OS

- Baseline guaranteed: **TCP/QUIC fallback**.
- RDMA cross-OS only when both ends expose compatible provider semantics.
- Do not assume Mac<->Windows RDMA compatibility without explicit proof.

### Validation-first bring-up checklist (all OS)

1. Enumerate provider/device list at runtime and log it.
2. Confirm memory registration succeeds for representative tensor sizes.
3. Validate send/recv and write-with-notify loopback before distributed runs.
4. Validate error paths (timeout, peer restart, permission failure).
5. Only then enable split-mode acceleration in production benchmarks.

---

## 8. Detailed Staged Plan (Gate-Driven)

### Prerequisites (non-gate, required before Gate 1)

Deliverables:
- Transport abstraction in `ggml-rpc` with TCP parity.
- Capability negotiation + fallback state machine (`auto|tcp|rdma`).
- Baseline instrumentation and benchmark harness for split modes.

Exit criteria:
- No functional regressions versus current TCP behavior.
- Baseline latency/tok/s numbers captured for comparison.

## Gate 1: Mac<->Mac Thunderbolt RDMA decode (`graph`/`attn`) first

Deliverables:
- Integrate a macOS TB5 RDMA provider path into transport abstraction.
- Wire split-mode reduction hot paths to RDMA-capable collective/offload path (fallback to `ggml_reduce` preserved).
- Validate multi-Mac graph-split decode end-to-end with strict no-fallback mode.

Exit criteria:
- Deterministic output parity versus TCP baseline.
- Measurable reduction in transfer/reduction latency and decode tok/s uplift on Mac cluster.
- Soak stability (no transport crashes, leaks, or silent fallbacks).

## Gate 2: RTX accelerated streaming prefill -> Mac graph-split decode handoff

Deliverables:
- Integrate RTX prefill pipeline from `RTX_ACCELERATED_MAC_PREFILL_LLAMA` as upstream prefill stage.
- Implement handoff path from RTX prefill output/KV artifact into ik_llama Mac graph-split decode workers.
- Add orchestration controls for prefill host + Mac decode cluster topology.

Exit criteria:
- End-to-end prefill->handoff->decode flow is reproducible.
- KV integrity checks and partition telemetry pass.
- End-to-end latency improves versus non-streaming or non-distributed baselines.

## Gate 2.5 (required): KV compatibility bridge and import path hardening

This gate exists because prefill-side exported state and ik-side ingest are not currently wire-identical by default, even when both load the same GGUF model.

Deliverables:
- Define and implement a dedicated KV compatibility bridge:
  - source: prefill artifact payload (sequence-state blob wrapped in `.kva`)
  - destination: ik_llama `llama_state_seq_set_data` input blob
- Add one-time model-aware mapping plan generation at startup (or first request) and cache it.
- Ensure conversion path is copy/pack/scatter only at runtime (no model-graph recompute).
- Add strict compatibility checks and typed failure reasons.
- Add telemetry for conversion path timing, bytes moved, and reject causes.

Exit criteria:
- Same-GGUF prefill->ik handoff works deterministically for supported profiles.
- First decode token logits parity and token parity are within defined tolerance versus local prefill+decode baseline.
- Conversion succeeds in strict mode with no silent fallback.
- Unsupported profiles fail closed with explicit diagnostics.

## Gate 3: RDMA for RTX host -> Mac graph-split decode path

Deliverables:
- Enable RDMA transport between RTX host and Mac decode side for handoff/bulk activation paths where applicable.
- Ensure fallback behavior remains deterministic when RTX<->Mac RDMA cannot be established.
- Add transport telemetry breakdown for RTX-host links.

Exit criteria:
- Stable RTX-host to Mac-cluster runs with RDMA active.
- Measurable handoff and/or activation transfer improvement versus TCP.
- No regression in Gate 2 behavior when RDMA is disabled.

## Gate 4: Full heterogeneous RTX+Mac graph-split decode over RDMA

Deliverables:
- Include RTX GPUs as active decode participants (not just prefill) in graph-split execution with Macs.
- Add heterogeneous placement/scheduling policy for CUDA + remote Mac backends.
- Optimize collective/reduction path for mixed RTX+Mac decode topology.

Exit criteria:
- Proven decode tok/s uplift versus Mac-only decode (Gate 3) and RTX-only baseline.
- Numerical parity within defined tolerance versus reference path.
- Strict no-fallback runs succeed on supported hardware topology.

## Gate 5 (optional expansion): Cross-OS RDMA matrix beyond core Mac-first path

Deliverables:
- Linux RDMA provider (`verbs`) and Windows provider (`ndk`) in same transport abstraction.
- Interop matrix validation:
  - Linux<->Linux
  - Windows<->Windows
  - Mac<->Mac
  - Mac<->Linux
  - Mac<->Windows
  - Linux<->Windows
- Explicit support table for RDMA-capable pairs vs fallback-only pairs.

Exit criteria:
- No data-integrity failures in matrix tests.
- Unsupported pairs degrade cleanly to TCP/QUIC fallback.

## Gate 6: Hardening and release

Deliverables:
- Long-duration soak tests, fault injection, and reconnection validation.
- Resource lifecycle hardening for pinned/registered buffers.
- Final operator documentation and tuning guide.

Exit criteria:
- Release-candidate quality and reproducible performance report per gate.

---

## 9. Implementation Specification

## 9.1 File-level changes (proposed)

New files:
- `ggml/include/ggml-transport.h`
- `ggml/src/ggml-transport.cpp`
- `ggml/src/ggml-transport-registry.cpp`
- `ggml/src/transports/tcp/ggml-transport-tcp.cpp`
- `ggml/src/transports/rdma_verbs/ggml-transport-rdma-verbs.cpp`
- `ggml/src/transports/rdma_ndk/ggml-transport-rdma-ndk.cpp`
- `ggml/src/transports/rdma_macos_tb5/ggml-transport-rdma-macos.cpp` (conditional/experimental)
- `src/kv-bridge/ik-kv-compat.h`
- `src/kv-bridge/ik-kv-compat.cpp`
- `src/kv-bridge/ik-kv-compat-plan-cache.h`
- `src/kv-bridge/ik-kv-compat-plan-cache.cpp`
- `src/kv-bridge/ik-kv-compat-cli.cpp` (debug/inspection utility)

Modified files:
- `ggml/src/ggml-rpc.cpp` (replace direct socket ops with transport API)
- `ggml/include/ggml-backend.h` (if needed for transport config structs)
- `CMakeLists.txt` + ggml build files for provider flags
- `common/common.cpp` / CLI parsing for transport options
- `src/llama.cpp` (or equivalent decode entrypoint) for KV import hook wiring
- `common/common.h` for KV-bridge strictness and diagnostics toggles

## 9.2 RPC changes

- Keep existing RPC commands intact.
- Add optional transport metadata in HELLO.
- Add bulk transfer descriptors:
  - `transfer_id`, `payload_len`, `checksum`, `memory_token`.

## 9.3 Buffer strategy

- Use pooled registered buffers for RDMA data path.
- Configurable chunk size and in-flight window.
- Avoid per-transfer register/deregister churn.

## 9.4 Fallback state machine

States:
- `TCP_ONLY`
- `RDMA_NEGOTIATING`
- `RDMA_ACTIVE`
- `RDMA_DEGRADED`
- `TCP_RECOVERY`

Policy:
- `auto`: attempt RDMA, fallback transparently.
- `rdma`: hard fail if unable to establish/maintain.

## 9.5 Telemetry

Per transfer:
- bytes
- duration
- effective GB/s
- retries/retransmits
- completion timeout count
- active transport mode

Per session:
- negotiated provider
- fallback count
- error categories

## 9.6 Core design choices

1. **Out-of-band bootstrap within RPC handshake**
   - Exchange capabilities and memory tokens during session setup.

2. **Pinned/registered memory pool**
   - Use persistent registration for bulk buffers to avoid per-transfer registration overhead.

3. **Low-overhead completion signaling**
   - Use provider-native completion queues and, where possible, immediate/notify signaling primitives.

4. **Graph-split reduction acceleration**
   - Add dedicated collective integration for split-mode reduction hot paths.

## 9.7 Design guardrails

1. Register/pin the actual backend buffers used by transfer paths; do not replace ownership pointers.
2. Keep collective logic topology-aware; do not hardcode two-party assumptions.
3. Use provider-specific backends behind one transport interface; avoid forcing a fake universal API.
4. Discover provider/device/runtime capabilities dynamically; do not hardcode names or setup commands.
5. Expose adaptive completion policy (poll/spin/block) instead of one fixed mode.

## 9.8 Concrete implementation notes

- Prefer opaque memory tokens (`region_id`, `remote_key`, bounds, lifetime) over raw virtual addresses in public protocol fields.
- Add explicit transport capability bits:
  - `supports_registered_io`
  - `supports_remote_write`
  - `supports_write_notify`
  - `supports_remote_read`
- Extend RPC message schema with:
  - `bulk_begin`
  - `bulk_chunk`
  - `bulk_commit`
  - `bulk_abort`
- Guard every RDMA path with deterministic fallback into existing TCP copy path at the operation boundary.

## 9.9 KV compatibility bridge (detailed)

### Objective

Use a one-time model-aware mapping plan to convert prefill-exported KV sequence state into ik-ingestible KV sequence state without runtime graph recompute.

### Why a bridge is required

- Same GGUF does not guarantee identical state blob layout across forks/versions.
- Session/sequence state versioning and memory-module serialization can diverge.
- The prefill artifact wrapper (`.kva`) adds metadata but does not guarantee consumer compatibility.

### Scope of bridge

- Input:
  - `.kva` metadata
  - payload bytes containing source sequence state
- Output:
  - destination sequence-state blob accepted by ik `llama_state_seq_set_data(...)`
- Non-goal:
  - changing transformer math or recomputing attention/FFN

## 9.10 One-time mapping plan design

### Plan key (cache identity)

Build a deterministic `kv_compat_plan_key` using:
- source model fingerprint
- destination model fingerprint
- source KV schema id + version
- destination KV schema id + version
- `type_k`, `type_v`, `v_trans` mode
- `n_ctx`, `n_layer`, `n_head_kv`/GQA-relevant hparams

### Fingerprint contents

The fingerprint should include:
- GGUF checksum (or model file digest)
- architecture string
- layer count and per-layer KV geometry
- RoPE/MLA settings impacting KV layout

### Plan payload

The plan stores precomputed mapping metadata:
- per-layer descriptors
  - source row widths (K/V)
  - destination row widths (K/V)
  - stream/layer routing
  - transposed/non-transposed V handling policy
- byte-order and stride assumptions
- fast-path eligibility flags (contiguous copy vs scatter)
- strict reject predicates

### Cache lifecycle

- Build once on first compatible request or on startup warmup.
- Persist plan cache on disk by key.
- Reload cache on process start; invalidate if fingerprint mismatch.

## 9.11 Runtime conversion flow

1. Receive artifact and parse `.kva` header.
2. Verify metadata guardrails (`n_layers`, `n_ctx`, `type_k`, `type_v`, flags).
3. Resolve/load `kv_compat_plan` by key.
4. Parse source sequence-state payload into a lightweight read view.
5. Emit destination sequence-state blob using plan:
   - copy/pack/scatter only
   - optional deterministic transpose path if explicitly enabled
6. Validate output blob shape/size/checksum.
7. Call destination `llama_state_seq_set_data(...)`.
8. Record timing/bytes/status telemetry.

Runtime performance rule:
- no model tensor remapping recomputation
- no graph rebuild caused by conversion
- conversion must be linear data movement over precomputed descriptors

## 9.12 Compatibility policy and staged support

### v1 strict profile (required)

- same GGUF fingerprint at both ends
- same `type_k`/`type_v`
- same `v_trans` mode
- single-stream attention KV (`n_stream == 1`)
- no partial-only flags unless explicitly matched

If any condition fails in strict mode:
- hard fail with specific reason code
- no silent fallback to unsafe import

### v1 optional relaxations (feature flags)

- allow safe V transpose conversion when both sides pass geometry checks
- allow explicit partial-state import modes where both sides support them

### later profiles

- multi-stream state import
- heterogeneous profile families with declared compatibility matrices

## 9.13 Validation and observability for KV bridge

### Required tests

- golden roundtrip tests (source export -> bridge -> destination ingest)
- corrupted payload/header rejection tests
- dtype/geometry mismatch rejection tests
- strict-vs-relaxed mode behavior tests
- first-token parity tests after handoff

### Required metrics

Per request:
- `kv_bridge_plan_key`
- `kv_bridge_plan_cache_hit` (bool)
- `kv_bridge_convert_us`
- `kv_bridge_bytes_in`
- `kv_bridge_bytes_out`
- `kv_bridge_mode` (`strict|relaxed`)
- `kv_bridge_status` (`ok|reject|error`)
- `kv_bridge_reject_reason` (enum)

### Operational diagnostics

- log source and destination schema versions in handoff summary
- log compatibility profile chosen
- log exact reject predicate when failing strict import

---

## 10. Configuration and UX

Proposed CLI/env additions:

- `--rpc-transport auto|tcp|rdma`
- `--rpc-rdma-provider auto|verbs|ndk|macos_tb5`
- `--rpc-rdma-threshold-bytes N`
- `--rpc-rdma-chunk-bytes N`
- `--rpc-rdma-inflight-bytes N`
- `--rpc-rdma-timeout-ms N`
- `--rpc-rdma-no-fallback` (equivalent to hard `rdma` mode)

Recommended defaults:
- `auto` transport
- conservative threshold for RDMA bulk path
- fallback enabled

---

## 11. Validation Matrix

## Correctness

- Exact output/token parity tests versus TCP baseline for deterministic settings.
- CRC/hash checks for all transferred tensor payloads.

## Stability

- 1h and 8h soak tests with mixed prompt lengths.
- Reconnect/retry tests under injected failures.

## Performance

- Split mode benchmarks:
  - `layer`
  - `attn`
  - `graph`
- Compare TCP vs RDMA on identical hardware/config.

## Interop

- Same-OS and cross-OS matrix with explicit expected transport mode per pair.

## KV bridge compatibility

- Source/destination schema matrix tests (supported and intentionally unsupported pairs).
- Plan-cache hit-rate and correctness checks under repeated prompts.
- Strict mode fail-closed behavior validation.
- Relaxed mode parity/perf regression checks (when enabled).

---

## 12. Risks and Mitigations

1. **Mac RDMA userspace API uncertainty**
   - Mitigation: keep Gate 1 validation-first and only expand beyond Mac<->Mac after measured stability.

2. **Cross-OS RDMA incompatibility**
   - Mitigation: capability negotiation + deterministic TCP fallback.

3. **Protocol complexity/regressions**
   - Mitigation: keep control path stable; add bulk path incrementally.

4. **Resource leaks / pinned memory pressure**
   - Mitigation: pooled buffers with strict lifecycle and diagnostics.

---

## 13. Recommended Delivery Order (pragmatic)

1. Land transport abstraction + TCP parity + negotiation/fallback first.
2. Deliver Gate 1 (Mac<->Mac TB5 RDMA graph-split decode).
3. Deliver Gate 2 (RTX accelerated streaming prefill handoff into Mac decode).
4. Deliver Gate 2.5 (KV compatibility bridge + strict import hardening).
5. Deliver Gate 3 (RDMA path for RTX host -> Mac decode links).
6. Deliver Gate 4 (heterogeneous RTX+Mac decode over RDMA).
7. Expand to Gate 5 cross-OS RDMA matrix only after core path is stable.
8. Complete Gate 6 hardening and release docs.

---

## 14. Final Recommendation

Execute the roadmap as a **Mac-first RDMA program**:
- First prove Mac<->Mac TB5 RDMA decode gains in ik_llama.
- Then integrate RTX prefill handoff into the Mac decode cluster.
- Then add RTX-host RDMA links, followed by full heterogeneous RTX+Mac decode.

Keep cross-OS reliability guaranteed through negotiated fallback, and treat cross-OS RDMA as an expansion gate after the core path is stable.
