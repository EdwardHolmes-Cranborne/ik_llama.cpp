# Dual Codex Coordination Log

This file is the shared handoff + review ledger for both machines.

## Coordination Protocol

1. `git pull --ff-only` before starting a new fix.
2. Append one entry in this file before and after major edits.
3. Push each checkpoint commit immediately so the other side can review.
4. Review incoming commits from the other side with:
   - `git log --oneline --decorate -n 20`
   - `git show <commit>`
5. Record any review findings in this file under `Review Notes`.

## Active Work (2026-02-18)

- Owner: local Codex on M3 Max host
- Branch: `feature/kv-bridge-gate2.5`
- Focus:
  - stabilize RPC behavior when both machines connect concurrently
  - keep validator green for dual-mac prefill/decode startup

### Changes In Progress

- `ggml/src/ggml-rpc.cpp`
  - RPC server now accepts clients concurrently (thread-per-connection).
  - Backend command execution is serialized with a global mutex to avoid backend re-entrancy issues.
  - Accept-loop no longer exits the server on a transient accept failure.
  - Pulled from the other side: reduce socket chunk sizes + stream `SET_TENSOR` payloads (avoid multi-GiB client allocs).

- `scripts/validate_dual_mac_decode_server.sh`
  - Default `OUTPUT_DIR` includes PID suffix to avoid collisions in parallel runs.

- `src/unicode.cpp`
  - Fixed `unicode_tolower()` to use hash lookup on `unicode_map_lowercase` (unordered_map), restoring correct WPM/BERT lowercasing behavior.

- `examples/eval-callback/CMakeLists.txt`
  - Removed `test-eval-callback` tinyllamas download test target to keep active workflow fully local/no-HF.

### Validation Snapshots

- PASS: `/tmp/ik_dual_mac_decode_validate_20260218_143141`
  - profile: `m3_max128_to_m3_ultra512`
  - rpc: `127.0.0.1:50052`
  - completion smoke: default `n_predict=16`, default timeout (180s)

- PASS (repeat): `/tmp/ik_dual_mac_decode_validate_20260218_144951`
  - profile: `m3_max128_to_m3_ultra512`
  - rpc: `127.0.0.1:50052`

- PASS: `/tmp/ik_dual_mac_decode_validate_20260218_142916`
  - profile: `m3_max128_to_m3_ultra512`
  - rpc: `127.0.0.1:50052`
  - completion smoke: `n_predict=1`, timeout 180s

- PASS (concurrent): `/tmp/ik_dual_mac_decode_validate_20260218_145521_9474`
  - profile: custom (ctx=4096, ngl=1, http=8080, kv=19001)
  - rpc: `127.0.0.1:50052`

- PASS (concurrent): `/tmp/ik_dual_mac_decode_validate_20260218_145521_9475`
  - profile: custom (ctx=4096, ngl=1, http=8081, kv=19002)
  - rpc: `127.0.0.1:50052`

- Earlier run reached startup but timed out on long completion:
  - `/tmp/ik_dual_mac_decode_validate_20260218_142508`

### Review Notes

- 2026-02-18: fetched `origin/feature/kv-bridge-gate2.5` before/after patch set; no unseen incoming commits at review time.
- Pending review from other side for:
  - concurrency safety of threaded accept model in `ggml-rpc.cpp`
  - whether backend-global serialization should become per-device serialization
- 2026-02-18: local build + `ctest` now passes 22/22 after tokenizer lowercasing fix and eval-callback test removal.
- 2026-02-18: reviewed `937dfcf9` (`update layer split`) and corrected inverted hardware profile defaults in `scripts/validate_dual_mac_decode_server.sh`:
  - `m3_ultra512_to_m3_max128` -> `0.82,0.18`
  - `m3_max128_to_m3_ultra512` -> `0.18,0.82`
