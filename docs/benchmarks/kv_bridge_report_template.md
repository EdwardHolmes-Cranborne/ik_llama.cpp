# KV Bridge Benchmark Report Template

## 1. Build/Runtime Metadata

- Date:
- Commit:
- Host(s):
- Model:
- Artifact source:
- Build flags:

## 2. Test Matrix

| Case | Mode | Fallback | Streams | Split mode | Flash-attn | Result |
|---|---|---|---|---|---|---|
| strict baseline | strict | no | 1 | graph | on | |
| relaxed vtrans | relaxed | no | 1 | graph | on | |
| strict fallback | strict | yes | 1 | graph | on | |
| transport auto | strict | yes | 1 | graph | on | |

## 3. Bridge Metrics

Capture:

- `kv_bridge_plan_key`
- `kv_bridge_plan_cache_hit`
- `kv_bridge_convert_us`
- `kv_bridge_bytes_in`
- `kv_bridge_bytes_out`
- `kv_bridge_mode`
- `kv_bridge_status`
- `kv_bridge_reject_reason`

## 4. Latency/Throughput

| Scenario | p50 (ms) | p95 (ms) | Throughput (MB/s) | Notes |
|---|---|---|---|---|
| Conversion only | | | | |
| Import only | | | | |
| Full handoff | | | | |

## 5. Correctness

- First-token logits parity:
- Token parity:
- Any reject reasons seen:
- Any fallback activations:

## 6. Stability

- Soak duration:
- Runs completed:
- Crashes:
- Leak/regression notes:

## 7. Conclusions

- Gate pass/fail:
- Known gaps:
- Next actions:

