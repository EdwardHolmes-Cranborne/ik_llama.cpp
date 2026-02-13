# KV Bridge Implementation Tasklist

## 1. Purpose

This tasklist operationalizes `RDMA_augmented_ik_llama_plan.md` Gate 2.5.

Goal:
- Accept prefill-side `.kva` artifacts (payload = prefill sequence-state bytes).
- Convert them into ik-compatible sequence-state bytes.
- Import via ik `llama_state_seq_set_data(...)` with strict fail-closed checks.
- Keep runtime conversion to copy/pack/scatter only using a precomputed mapping plan.

Non-goal:
- Recompute model activations or transformer math during conversion.

---

## 2. Delivery Milestones

- `M0` Interface + scaffolding + build wiring
- `M1` Source/destination schema readers + compatibility plan builder
- `M2` Plan cache + conversion runtime + decode import hook
- `M3` Strict-mode policy + telemetry + diagnostics
- `M4` Test matrix + parity + soak/perf validation

---

## 3. Ticket Conventions

- Prefix: `KVB-`
- Status values: `todo`, `in_progress`, `blocked`, `done`
- Priority: `P0` blocking, `P1` required, `P2` optional
- All tickets include explicit test IDs; no ticket is `done` without green mapped tests.

---

## 4. Core Backlog

### KVB-001 (P0) Create KV bridge module skeleton

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-compat.h`
  - `src/kv-bridge/ik-kv-compat.cpp`
  - `src/kv-bridge/ik-kv-compat-types.h`
  - `src/kv-bridge/ik-kv-compat-types.cpp`
  - `CMakeLists.txt` (root and/or `src/CMakeLists.txt` integration)
- Functions/interfaces:
  - `ik_kv_compat_convert_result`
  - `ik_kv_compat_reject_reason`
  - `ik_kv_compat_convert(...)`
- Steps:
  1. Add headers/source stubs and wire compile target.
  2. Define stable error/reject enums.
  3. Add unit-test-visible API surface.
- Acceptance:
  - Project builds with bridge module compiled in (no-op behavior acceptable).
- Tests:
  - `KVB-UT-001`

---

### KVB-002 (P0) Add KVA parser and source schema reader

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-source-prefill.h`
  - `src/kv-bridge/ik-kv-source-prefill.cpp`
- Functions/interfaces:
  - `ik_kv_source_parse_kva_header(...)`
  - `ik_kv_source_parse_prefill_seq_state(...)`
  - `ik_kv_source_descriptor`
- Steps:
  1. Parse `.kva` metadata (`format_major/minor`, `n_layers`, `n_ctx`, `type_k/v`, flags, payload size/crc).
  2. Parse source sequence-state payload into a view struct (no deep copies).
  3. Validate payload bounds and malformed frame handling.
- Acceptance:
  - Corrupt artifacts fail with deterministic reject reason.
- Tests:
  - `KVB-UT-010`, `KVB-UT-011`, `KVB-UT-012`

---

### KVB-003 (P0) Add destination schema introspection for ik context

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-dest-ik.h`
  - `src/kv-bridge/ik-kv-dest-ik.cpp`
  - `src/llama.cpp` (read-only hookups if needed)
- Functions/interfaces:
  - `ik_kv_dest_introspect_from_ctx(struct llama_context *)`
  - `ik_kv_dest_descriptor`
- Steps:
  1. Capture destination KV geometry from context/model/hparams.
  2. Capture type and transpose expectations.
  3. Expose a normalized descriptor for plan builder.
- Acceptance:
  - Descriptor is deterministic for identical model/context settings.
- Tests:
  - `KVB-UT-020`, `KVB-UT-021`

---

### KVB-004 (P0) Implement compatibility plan key + fingerprinting

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-compat-plan.h`
  - `src/kv-bridge/ik-kv-compat-plan.cpp`
- Functions/interfaces:
  - `ik_kv_compat_plan_key_build(...)`
  - `ik_kv_model_fingerprint_build(...)`
- Steps:
  1. Build key from source/destination schema versions + model fingerprint + KV layout params.
  2. Add stable serialization/hash for cache lookup.
  3. Include strict-mode guard predicates in key material where relevant.
- Acceptance:
  - Identical configs produce identical keys; changed configs produce different keys.
- Tests:
  - `KVB-UT-030`, `KVB-UT-031`

---

### KVB-005 (P0) Implement strict v1 plan builder

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-compat-plan.cpp`
- Functions/interfaces:
  - `ik_kv_compat_plan_build_strict_v1(...)`
  - `ik_kv_compat_plan_validate(...)`
- Steps:
  1. Build per-layer copy descriptors (K/V row sizes, offsets, route rules).
  2. Enforce strict v1 constraints:
     - same GGUF fingerprint
     - same `type_k`, `type_v`
     - same `v_trans` mode
     - `n_stream == 1`
  3. Emit explicit reject reason on first violated predicate.
- Acceptance:
  - Unsupported profiles fail before conversion starts.
- Tests:
  - `KVB-UT-040`, `KVB-UT-041`, `KVB-UT-042`

---

### KVB-006 (P1) Implement on-disk plan cache

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-compat-plan-cache.h`
  - `src/kv-bridge/ik-kv-compat-plan-cache.cpp`
- Functions/interfaces:
  - `ik_kv_plan_cache_load(...)`
  - `ik_kv_plan_cache_store(...)`
  - `ik_kv_plan_cache_invalidate(...)`
- Steps:
  1. Add versioned cache file format with checksum.
  2. Cache by plan key.
  3. Hard-invalidate on key/schema mismatch.
- Acceptance:
  - Cache hit path bypasses plan recomputation.
- Tests:
  - `KVB-UT-050`, `KVB-UT-051`, `KVB-UT-052`

---

### KVB-007 (P0) Implement converter runtime (copy/pack/scatter only)

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-convert.h`
  - `src/kv-bridge/ik-kv-convert.cpp`
- Functions/interfaces:
  - `ik_kv_convert_prefill_to_ik_seq_blob(...)`
- Steps:
  1. Build destination sequence-state bytes using plan descriptors.
  2. No graph/model recompute; data movement only.
  3. Add output checksum and size validation.
- Acceptance:
  - Produced blob ingests via ik sequence-state API under strict profile.
- Tests:
  - `KVB-UT-060`, `KVB-IT-100`

---

### KVB-008 (P0) Wire decode import hook

- Status: `todo`
- Files:
  - `src/llama.cpp` (or decode orchestration entrypoint used for remote handoff)
  - `src/kv-bridge/ik-kv-compat.cpp`
- Functions/interfaces:
  - `ik_kv_import_into_context(...)`
  - calls into `llama_state_seq_set_data(...)`
- Steps:
  1. Insert bridge path between received artifact payload and decode start.
  2. Fail-closed when strict mode and bridge/import fails.
  3. Keep existing non-bridge paths unchanged when feature disabled.
- Acceptance:
  - End-to-end handoff path can import converted state and continue decode.
- Tests:
  - `KVB-IT-101`, `KVB-IT-102`

---

### KVB-009 (P1) Add CLI/config controls

- Status: `todo`
- Files:
  - `common/common.h`
  - `common/common.cpp`
- Flags:
  - `--kv-bridge-mode off|strict|relaxed`
  - `--kv-bridge-plan-cache-dir PATH`
  - `--kv-bridge-allow-vtrans-convert`
  - `--kv-bridge-dry-run`
  - `--kv-bridge-no-fallback`
- Steps:
  1. Add param plumbing to runtime config.
  2. Add help text and defaults (`strict` for Gate 2.5 validation runs).
  3. Add config dump output.
- Acceptance:
  - Flags are parseable and reflected in runtime behavior.
- Tests:
  - `KVB-UT-070`, `KVB-IT-103`

---

### KVB-010 (P1) Add telemetry and diagnostics

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-telemetry.h`
  - `src/kv-bridge/ik-kv-telemetry.cpp`
  - `src/llama.cpp` (or common metrics emitter)
- Metrics:
  - `kv_bridge_plan_key`
  - `kv_bridge_plan_cache_hit`
  - `kv_bridge_convert_us`
  - `kv_bridge_bytes_in`
  - `kv_bridge_bytes_out`
  - `kv_bridge_mode`
  - `kv_bridge_status`
  - `kv_bridge_reject_reason`
- Steps:
  1. Emit structured logs and counters/histograms.
  2. Ensure reject reasons are always emitted on failure.
  3. Surface metrics in benchmark outputs used by gate reports.
- Acceptance:
  - Every bridge attempt is observable with a terminal status.
- Tests:
  - `KVB-UT-080`, `KVB-IT-104`

---

### KVB-011 (P2) Optional relaxed conversion: V-transpose conversion

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-convert.cpp`
  - `src/kv-bridge/ik-kv-compat-plan.cpp`
- Functions/interfaces:
  - `ik_kv_convert_vtrans_mode(...)`
- Steps:
  1. Implement explicit, deterministic transpose conversion path.
  2. Guard behind `--kv-bridge-allow-vtrans-convert`.
  3. Keep strict mode default fail for transpose mismatches.
- Acceptance:
  - Relaxed mode works where mathematically/layout safe; strict mode unchanged.
- Tests:
  - `KVB-UT-090`, `KVB-IT-105`

---

### KVB-012 (P1) Add debug inspection utility

- Status: `todo`
- Files:
  - `src/kv-bridge/ik-kv-compat-cli.cpp`
- Commands:
  - `--kv-bridge-inspect-artifact FILE`
  - `--kv-bridge-print-plan FILE_OR_KEY`
  - `--kv-bridge-validate-only`
- Steps:
  1. Add artifact/header inspector.
  2. Add plan dump (human-readable summary).
  3. Add validation-only pipeline for CI triage.
- Acceptance:
  - Tool can diagnose mismatch causes without running full decode.
- Tests:
  - `KVB-IT-106`

---

### KVB-013 (P0) Add unit tests and register in CMake

- Status: `todo`
- Files:
  - `tests/test-kv-bridge-parser.cpp`
  - `tests/test-kv-bridge-plan.cpp`
  - `tests/test-kv-bridge-cache.cpp`
  - `tests/test-kv-bridge-convert.cpp`
  - `tests/CMakeLists.txt`
- Steps:
  1. Add parser/plan/cache/converter unit tests.
  2. Register with `llama_target_and_test(...)`.
  3. Add deterministic fixtures.
- Acceptance:
  - All KV bridge unit tests pass in CI profile.
- Tests:
  - `KVB-UT-001` through `KVB-UT-099`

---

### KVB-014 (P0) Add integration + parity test pipeline

- Status: `todo`
- Files:
  - `tests/test-kv-bridge-integration.cpp`
  - `scripts/run_kv_bridge_matrix.sh`
- Steps:
  1. Load same GGUF in source/target harness.
  2. Run prefill-export -> bridge -> import -> decode for fixed prompts.
  3. Validate first-token logits parity and token parity.
- Acceptance:
  - Gate 2.5 parity thresholds met.
- Tests:
  - `KVB-IT-100`, `KVB-IT-101`, `KVB-IT-102`, `KVB-IT-103`

---

### KVB-015 (P1) Add soak and performance characterization

- Status: `todo`
- Files:
  - `scripts/bench_kv_bridge.sh`
  - `docs/benchmarks/kv_bridge_report_template.md`
- Steps:
  1. Measure conversion overhead (`us`, throughput MB/s).
  2. Run 1h soak with repeated handoffs.
  3. Compare strict mode with bridge disabled baseline.
- Acceptance:
  - No crash/leak; overhead within accepted envelope.
- Tests:
  - `KVB-PERF-200`, `KVB-SOAK-300`

---

## 5. Test Case Catalog (IDs)

### Unit tests (`UT`)

- `KVB-UT-001`: module compiles, API symbols link.
- `KVB-UT-010`: valid `.kva` header parse.
- `KVB-UT-011`: invalid magic/version/header size reject.
- `KVB-UT-012`: payload CRC mismatch reject.
- `KVB-UT-020`: destination descriptor extraction stable across runs.
- `KVB-UT-021`: descriptor changes when KV layout params change.
- `KVB-UT-030`: plan key deterministic.
- `KVB-UT-031`: plan key changes on fingerprint/schema mismatch.
- `KVB-UT-040`: strict profile accepts compatible inputs.
- `KVB-UT-041`: strict profile rejects dtype mismatch.
- `KVB-UT-042`: strict profile rejects `n_stream != 1`.
- `KVB-UT-050`: cache load/store hit path.
- `KVB-UT-051`: cache invalidates on schema version change.
- `KVB-UT-052`: cache rejects corrupted file.
- `KVB-UT-060`: converter emits valid destination sequence blob.
- `KVB-UT-070`: CLI flag parsing and default semantics.
- `KVB-UT-080`: telemetry emits terminal status for every attempt.
- `KVB-UT-090`: optional transpose conversion path correctness.

### Integration tests (`IT`)

- `KVB-IT-100`: source export -> bridge -> destination `llama_state_seq_set_data` success.
- `KVB-IT-101`: end-to-end handoff then decode first token parity.
- `KVB-IT-102`: strict mode fail-closed on incompatibility (no silent fallback).
- `KVB-IT-103`: CLI mode matrix (`off|strict|relaxed`) behavior.
- `KVB-IT-104`: telemetry/log schema completeness under success/failure.
- `KVB-IT-105`: relaxed transpose mode success where enabled.
- `KVB-IT-106`: debug inspector prints actionable mismatch diagnostics.

### Performance and reliability tests

- `KVB-PERF-200`: conversion latency and throughput benchmark across payload sizes.
- `KVB-SOAK-300`: 1h repeated handoff/import/decode stability run.

---

## 6. Dependencies and External Coordination

- Prefill-side artifact producer must keep `.kva` metadata stable or versioned.
- Cross-repo contract:
  - source schema id/version
  - compatibility profile flags
  - reject reason vocabulary
- Gate 3 (RDMA host link) depends on Gate 2.5 completion.

---

## 7. Definition of Done (Gate 2.5)

All of the following:
- `KVB-001`..`KVB-010`, `KVB-013`, `KVB-014` are `done`.
- Required tests green:
  - `KVB-UT-010,011,012,040,041,042,060`
  - `KVB-IT-100,101,102,103,104`
- Artifacts:
  - compatibility profile document
  - benchmark report (includes conversion overhead and parity results)
  - failure-mode table with reject reasons and operator guidance

