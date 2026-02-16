# KV Bridge Compatibility Profile

This document defines the compatibility policy used by the decode-side KV bridge.

## Strict v1 (default)

`mode=strict`

Required predicates:

1. Source and destination model fingerprint must match when both are populated.
2. `type_k` must match.
3. `type_v` must match.
4. `v_trans` must match when both source and destination publish concrete values (`0` or `1`).
5. `n_stream == 1` on both source and destination descriptors.
6. `n_layers`, `n_ctx`, and `n_head_kv` must match when both sides provide non-zero values.

Failure behavior:

- Import is rejected with a typed `reject_reason`.
- If `--kv-bridge-no-fallback` is not set, runtime may attempt payload pass-through fallback.

## Relaxed v1

`mode=relaxed`

Baseline behavior is strict-v1 plus optional features behind explicit flags:

1. `--kv-bridge-allow-vtrans-convert`:
   - Allows a v-trans mismatch path where a relaxed plan is rebuilt against destination v-trans mode.
   - The plan is marked with `needs_v_trans` for conversion bookkeeping.
2. `--kv-bridge-no-fallback`:
   - Disables pass-through fallback and forces fail-closed behavior.

## Off mode

`mode=off`

- Bridge conversion path is disabled.
- Runtime attempts direct payload import only when fallback is allowed.

## Operational Notes

- `--kv-bridge-dry-run` validates/bridges without final context import.
- `--kv-bridge-plan-cache-dir PATH` overrides cache directory.
- Default cache directory is `~/.ik_llama_kv_plan_cache`.
- Telemetry can be disabled via `--no-kv-bridge-telemetry`.

