# RDMA + KV Handoff Implementation Log

## 2026-02-16

### Scope completed this cycle

1. Removed RTX multi-active-stream import limitation in KV bridge:
   - `convert_rtx_seq_blob_to_ik()` now merges multiple active RTX stream blocks into one IK sequence-state stream.
   - Merge path preserves stream-order cell metadata and concatenates per-layer K/V payload rows safely.
   - Cross-stream layout mismatches now fail with explicit compatibility/header parse errors.
2. Updated RTX source-shape normalization for planning:
   - RTX payload descriptors now report effective single-stream compatibility for bridge planning because merge is performed at conversion time.
3. Updated parser coverage:
   - `KVB-UT-017` now validates multi-active-stream merge success (cell count + payload concatenation assertions).
4. Removed obsolete Phase-1 queue guardrail:
   - External command submit no longer rejects `--kv-streams != 1`.
   - Queue self-test and operator docs updated accordingly.
5. Updated deployment/test docs:
   - Dual-Mac setup and test guides now describe multi-stream import as supported (instead of constrained to single stream).
6. Finished queue/wrapper stream-count plumbing:
   - Added `--kv-streams` passthrough and validation in:
     - `scripts/run_phase1_prefill_handoff_job.sh`
     - `scripts/run_phase2_prefill_to_artifact.sh`
     - `scripts/run_single_machine_buffered_e2e.sh`
   - Added queue submit/request support for `--kv-streams` and propagated it through single-machine + Phase-2 worker stages.
   - Extended Phase-2 queue self-test to assert `--kv-streams` propagation into prefill stage wrappers.

### Verification completed

1. Build:
   - `cmake --build build_codex --target test-kv-bridge-parser -j8` passes.
2. Tests:
   - `build_codex/bin/test-kv-bridge-parser` passes.
   - `ctest --test-dir build_codex -L kv-bridge --output-on-failure` passes.

## 2026-02-15

### Scope completed this cycle

1. Completed decode-side KV bridge policy wiring through CLI/env and runtime:
   - Added flags:
     - `--kv-bridge-mode off|strict|relaxed`
     - `--kv-bridge-plan-cache-dir PATH`
     - `--kv-bridge-allow-vtrans-convert`
     - `--kv-bridge-dry-run`
     - `--kv-bridge-no-fallback`
     - `--no-kv-bridge-telemetry`
   - Added env equivalents (`LLAMA_ARG_KV_BRIDGE_*`).
2. Wired bridge configuration into server startup model lifecycle (`server-context`):
   - Sets bridge mode, cache dir, relaxed-vtrans toggle, dry-run, fallback policy, telemetry toggle.
3. Hardened bridge import runtime behavior:
   - Added default plan-cache directory fallback (`~/.ik_llama_kv_plan_cache`).
   - Added telemetry emission for every bridge attempt with status/result/reject/bytes/timing/cache-hit/fallback.
   - Added explicit fallback payload import behavior controlled by `no_fallback`.
   - Added dry-run behavior that validates/converts without final `llama_state_seq_set_data`.
   - Added relaxed-mode v-trans mismatch plan rebuild path behind `allow_vtrans_convert`.
4. Expanded KV bridge tests:
   - Extended parser suite with `KVB-UT-070`, `KVB-UT-080`, `KVB-UT-090`.
   - Added new CLI parser test binary `test-kv-bridge-cli`.
5. Added operator utilities and docs:
   - `scripts/run_kv_bridge_matrix.sh`
   - `scripts/bench_kv_bridge.sh`
   - `build_codex/bin/kv-bridge-cli` (`src/kv-bridge/ik-kv-compat-cli.cpp`)
   - `docs/development/kv_bridge_compatibility_profile.md`
   - `docs/benchmarks/kv_bridge_report_template.md`

### Verification completed

1. Build verification:
   - `cmake --build build_codex --target llama-server test-kv-bridge-parser test-kv-bridge-cli -j8` passes.
2. Unit tests:
   - `build_codex/bin/test-kv-bridge-parser` passes (`Passed: 21`, `Failed: 0`).
   - `build_codex/bin/test-kv-bridge-cli` passes.
3. Matrix and queue checks:
   - `scripts/run_kv_bridge_matrix.sh --build-dir build_codex --no-queue-tests` passes.
   - `scripts/test_prefill_decode_job_queue.sh` passes.
   - `scripts/test_prefill_decode_phase2_pipeline.sh` passes.
4. Help/CLI verification:
   - `llama-server --help` includes all new `--kv-bridge*` options.

## 2026-02-13

### Scope completed this cycle

1. Added decode-side KV handoff transport receiver in `llama-server`:
   - New module: `examples/server/server-kv-receiver.h`
   - New module: `examples/server/server-kv-receiver.cpp`
2. Added server configuration for prefill->decode handoff transport switching:
   - `--kv-transport auto|rdma|tcp|mixed|disabled`
   - `--kv-transport-fallback`
   - `--kv-recv-enable`
   - bind/slot/ACK/socket tuning flags
3. Wired receiver service lifecycle into server startup/shutdown:
   - starts after model load
   - stops during server shutdown
4. Added per-session artifact reassembly and slot-restore queue integration:
   - reassemble received TBP chunks into artifact
   - enqueue `SERVER_TASK_TYPE_SLOT_RESTORE`
5. Added receiver hardening and observability:
   - worker lifecycle cleanup/reaping
   - safer socket shutdown/connection tracking
   - artifact validation before restore enqueue
   - per-session JSON summary output
   - `GET /kv-receiver/status` endpoint
6. Added receiver transport validation mode:
   - `--kv-recv-dry-run` reassembles/validates artifacts without consuming slot restore.
7. Added receiver artifact/header compatibility and chunk integrity hardening:
   - Accepts RTX/IK header layout variants during payload validation.
   - Parses `chunks_sent` from `TBP_MSG_KV_SEGMENT_END` and enforces expected chunk count.
8. Added receiver payload-byte completeness enforcement:
   - Parses expected payload bytes from session/segment metadata.
   - Verifies reassembled artifact byte size before validation/restore queueing.
9. Corrected CRC failure signaling behavior:
   - When `--kv-recv-no-nack-on-crc-bad` is set, receiver no longer sends false positive ACKs on CRC-bad chunks.
10. Tightened chunk-set completeness enforcement:
   - Receiver now enforces exact chunk-count match (`got == expected`) when `chunks_sent` metadata is present.
11. Improved ACK semantics and persistence safety:
   - Receiver now honors sender `ack_required` metadata and can suppress ACK/NACK chatter for no-ack sessions.
   - Receiver now returns `nack=io` when chunk persistence fails, allowing sender retransmit instead of silent bad ACK.
12. Added receiver stale-session lifecycle management:
   - New config to force-finalize sessions after idle frame timeout.
   - New config to prune finalized sessions after retention window.
   - Periodic maintenance sweep with status counters for stale-finalized/pruned sessions.
13. Hardened prefill-side split transport correctness (RTX fork):
   - Split handoff now treats `kv_transport=disabled` as a real fallback to local decode (no false transport success).
   - Split handoff publish now rejects filesystem backend in transport-required paths.
   - Decode alias ordering fixed in CLI arg tables to satisfy parser invariants.
14. Added single-machine buffered prefill->decode E2E harness (no core engine edits):
   - New orchestrator script: `scripts/run_single_machine_buffered_e2e.sh`
   - New replay utility: `scripts/tbp_replay_to_kv_receiver.py`
   - New guide: `docs/development/single_machine_buffered_e2e.md`
   - Receiver validation upgraded to assert `artifacts_reassembled`, `artifacts_validated`, `restore_tasks_enqueued`, and finalized validated session state.
15. Started Phase-1 single-flight handoff queue implementation (external tooling):
   - New persistent queue worker: `scripts/prefill_decode_job_queue.py`
   - New operator guide: `docs/development/prefill_decode_phase1_queue.md`
   - Added queue status summary command and external-command execution mode for deterministic CI/local testing.
   - Added queue self-test harness: `scripts/test_prefill_decode_job_queue.sh`
   - Added submit-time guardrails for external commands:
     reject `--kv-streams != 1` and reject graph-split restore commands without flash-attn.
   - Added per-job transport metadata (`--kv-transport`) exported to worker child env (`IK_PDQ_KV_TRANSPORT`).
   - Added queue-friendly prefill handoff wrapper script:
     `scripts/run_phase1_prefill_handoff_job.sh`.
   - Added queue-mode usage and validation sections to dual-Mac setup/test guides.
   - Buffered E2E runner now supports threshold-preserving prefill mode selection:
     `--prefill-min-stream-batch-tokens` (default `-1`, runtime crossover logic).
16. Hardened Phase-1 queue worker launch-failure handling:
   - Worker now catches external runner launch exceptions (e.g. missing binary) and records error context.
   - Failed launch paths now store `result.runner_exception`, persist worker stream logs, and transition to retry/failed states correctly.
   - Queue self-test now covers unlaunchable command path to prevent regressions.
17. Implemented Phase-2 split prefill/handoff pipeline queue tooling:
   - Queue mode `phase2_split_pipeline` added to `scripts/prefill_decode_job_queue.py`.
   - Added dedicated `prefill-worker` and `handoff-worker` loops with separate locks and stage-specific retry handling.
   - Added stage wrappers:
     - `scripts/run_phase2_prefill_to_artifact.sh` (prefill -> disk artifact)
     - `scripts/run_phase2_handoff_from_artifact.sh` (artifact replay -> decode)
   - Added deterministic Phase-2 regression test:
     `scripts/test_prefill_decode_phase2_pipeline.sh`.
   - Added new operator guide:
     `docs/development/prefill_decode_phase2_pipeline.md`.
18. Added transport-aware replay controls for Phase-2 artifact handoff:
   - `scripts/tbp_replay_to_kv_receiver.py` now supports `--transport-mode auto|rdma|tcp|mixed|disabled` and `--transport-fallback`.
   - Replay now resolves mode-specific endpoints/bind addresses (`LLAMA_PREFILL_KV_RDMA_*`, `LLAMA_PREFILL_KV_TCP_*`, `LLAMA_PREFILL_KV_*`) and reports resolved transport metadata.
   - Queue submit now tracks `kv_transport_fallback`; Phase-2 prefill/handoff wrappers consume and enforce both mode and fallback.
   - Phase-2 self-test now validates handoff flag propagation for `rdma` mode and no-fallback path.

### Commits produced

- `47115392` Add KV handoff receiver transport config and server listener
- `a81fac07` Harden KV receiver worker lifecycle and socket shutdown
- `45be6445` Validate KV receiver slot and sanitize runtime limits
- `e4a059d9` Add KV receiver dry-run mode and status telemetry endpoint
- `aec132a0` Add RTX5090 dual-Mac deployment and test guides
- `6844c1fb` Harden KV receiver chunk completeness and payload parsing
- `9979071b` Track expected KV payload bytes for receiver validation
- `efcd76cf` Avoid false ACKs on CRC failures when NACKs are disabled
- `d3e9e38c` Enforce strict expected chunk count on KV reassembly
- `eaed6ee0` Honor sender ACK policy and NACK on chunk persistence errors
- `fe16986e` Add KV receiver stale-finalize and session retention maintenance
- `efb93a5` [RTX fork] Require network transport for split handoff and fix decode arg alias order
- `2d42d766` Harden phase-1 queue worker launch failure handling
- `344fb3fb` Document phase-1 queue launch-failure hardening
- `37e2c7de` Implement phase-2 split prefill/handoff queue pipeline
- `23b7703c` Document phase-2 split prefill/handoff queue workflow
- `5ffa2dcc` Add transport-aware phase-2 artifact replay controls
- `e9e5124c` Document phase-2 replay transport mode and fallback

### Verification completed

1. Build verification:
   - `cmake --build build_codex --target llama-server -j8` passes.
2. CLI verification:
   - `llama-server --help` includes all new `--kv-recv*` and `--kv-transport*` options.
3. No-regression checks:
   - Existing slot restore fallback path (`.kva` import via KV bridge) remains unchanged in `server-context.cpp`.
4. Script/tooling validation:
   - `python3 -m py_compile scripts/tbp_replay_to_kv_receiver.py` passes.
   - `bash -n scripts/run_single_machine_buffered_e2e.sh` passes.
   - `scripts/* --help` smoke checks pass.
   - `scripts/test_prefill_decode_job_queue.sh` passes including launch-failure state transition checks.
   - `scripts/test_prefill_decode_phase2_pipeline.sh` passes (artifact queueing + handoff retry path).

### Remaining work for full gate closure

1. Full three-host E2E runs on target topology:
   - WRX90 + RTX 5090 prefill host
   - Mac Studio 512GB + MacBook Pro M3 Max 128GB decode cluster
2. Long soak and failure-injection runs over transport fallback matrix.
3. Decode quality and first-token parity reporting against local baseline.
4. Throughput and latency baselining for `tcp` vs `rdma` mode selection on the real USB4/TB fabric.
5. Execute and archive the single-machine buffered suite on real model artifacts (requires local socket/model runtime; not runnable in this sandbox).
6. Add kernel-verbs-native RDMA replay client path for Phase-2 artifact handoff (current replay helper uses TBP socket transport with RDMA-mode endpoint/bind semantics).
