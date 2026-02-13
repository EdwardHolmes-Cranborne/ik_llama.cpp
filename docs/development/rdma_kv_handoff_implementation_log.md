# RDMA + KV Handoff Implementation Log

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
   - Buffered E2E runner now supports threshold-preserving prefill mode selection:
     `--prefill-min-stream-batch-tokens` (default `-1`, runtime crossover logic).

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

### Remaining work for full gate closure

1. Full three-host E2E runs on target topology:
   - WRX90 + RTX 5090 prefill host
   - Mac Studio 512GB + MacBook Pro M3 Max 128GB decode cluster
2. Long soak and failure-injection runs over transport fallback matrix.
3. Decode quality and first-token parity reporting against local baseline.
4. Throughput and latency baselining for `tcp` vs `rdma` mode selection on the real USB4/TB fabric.
5. Execute and archive the single-machine buffered suite on real model artifacts (requires local socket/model runtime; not runnable in this sandbox).
