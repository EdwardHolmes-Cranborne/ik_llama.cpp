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

### Commits produced

- `47115392` Add KV handoff receiver transport config and server listener
- `a81fac07` Harden KV receiver worker lifecycle and socket shutdown
- `45be6445` Validate KV receiver slot and sanitize runtime limits
- `e4a059d9` Add KV receiver dry-run mode and status telemetry endpoint
- `aec132a0` Add RTX5090 dual-Mac deployment and test guides
- `6844c1fb` Harden KV receiver chunk completeness and payload parsing
- `9979071b` Track expected KV payload bytes for receiver validation
- `efcd76cf` Avoid false ACKs on CRC failures when NACKs are disabled

### Verification completed

1. Build verification:
   - `cmake --build build_codex --target llama-server -j8` passes.
2. CLI verification:
   - `llama-server --help` includes all new `--kv-recv*` and `--kv-transport*` options.
3. No-regression checks:
   - Existing slot restore fallback path (`.kva` import via KV bridge) remains unchanged in `server-context.cpp`.

### Remaining work for full gate closure

1. Full three-host E2E runs on target topology:
   - WRX90 + RTX 5090 prefill host
   - Mac Studio 512GB + MacBook Pro M3 Max 128GB decode cluster
2. Long soak and failure-injection runs over transport fallback matrix.
3. Decode quality and first-token parity reporting against local baseline.
4. Throughput and latency baselining for `tcp` vs `rdma` mode selection on the real USB4/TB fabric.
