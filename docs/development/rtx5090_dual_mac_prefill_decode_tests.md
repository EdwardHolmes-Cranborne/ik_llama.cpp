# RTX5090 -> Dual-Mac Decode Test Guide

This guide validates correctness, fallback behavior, and stability for:

- WRX90 + RTX5090 prefill
- Mac Studio + MacBook Pro decode
- ik KV handoff receiver + bridge import

## 1. Preflight checks

Run before functional tests.

### 1.1 Link and ports

From `RTX_HOST`:

```bash
ping -c 3 10.40.0.20
ping -c 3 10.40.0.21
nc -vz 10.40.0.20 19001
nc -vz 10.40.0.20 8080
nc -vz 10.40.0.20 50052
nc -vz 10.40.0.21 50052
```

### 1.2 Decode service status

From any host:

```bash
curl -s http://10.40.0.20:8080/health | jq
curl -s http://10.40.0.20:8080/kv-receiver/status | jq
```

## 2. Transport-only validation (no slot restore)

Start decode coordinator with:

- `--kv-recv-dry-run`

Then run one prefill transfer from RTX host using a long prompt and check:

```bash
curl -s http://10.40.0.20:8080/kv-receiver/status | jq '.counters'
```

Expected:

- `artifacts_reassembled` increases
- `artifacts_validated` increases
- `restore_tasks_skipped_dry_run` increases
- `sessions_stale_finalized` stays stable for healthy runs
- `sessions_pruned` stays stable during short functional runs
- `sessions[].validation_ok == true`
- `sessions[].expected_chunks > 0`
- `sessions[].expected_payload_bytes > 0`
- `sessions[].expected_payload_bytes == sessions[].bytes_received` for no-retransmit runs

Inspect latest session quickly:

```bash
curl -s http://10.40.0.20:8080/kv-receiver/status \
  | jq '.sessions | sort_by(.session_id) | last'
```

## 3. Full handoff validation (restore enabled)

Restart decode coordinator without `--kv-recv-dry-run` and repeat run.

Check:

```bash
curl -s http://10.40.0.20:8080/kv-receiver/status | jq '.counters'
```

Expected:

- `restore_tasks_enqueued` increases
- `sessions[].restore_enqueued == true`
- no new `validation_error`
- `sessions[].expected_chunks` / `sessions[].expected_payload_bytes` stay non-zero

Compatibility checks (must stay clear):

- no `sessions[].validation_error` containing `N_STREAM_UNSUPPORTED`
- no decode logs containing:
  - `Transposed V cache is not sypported with split mode 'graph'`

Also verify decode server still responds:

```bash
curl -s http://10.40.0.20:8080/health | jq
```

## 4. Transport switch matrix

Run each row with same prompt/model and capture receiver counters plus wall times.

1. `rdma strict`:
   - prefill: `--kv-transport rdma --no-kv-transport-fallback`
   - decode: `--kv-transport rdma --no-kv-transport-fallback`
2. `rdma->tcp fallback`:
   - prefill: `--kv-transport rdma --kv-transport-fallback`
   - decode: `--kv-transport rdma --kv-transport-fallback`
3. `tcp strict`:
   - prefill: `--kv-transport tcp --no-kv-transport-fallback`
   - decode: `--kv-transport tcp --no-kv-transport-fallback`
4. `auto`:
   - prefill: `--kv-transport auto`
   - decode: `--kv-transport auto`

Pass criteria:

- No crashes or hangs
- No corrupted session (`validation_ok=false`)
- Fallback runs complete with successful reassembly/validation

## 5. Session lifecycle controls validation

Start decode with:

- `--kv-recv-stale-finalize-timeout 30`
- `--kv-recv-session-retention 60`
- `--kv-recv-cleanup-interval 5`

Validation steps:

1. Run one normal handoff and verify `sessions_stale_finalized` does not increment.
2. Start a transfer and interrupt sender before `KV_DONE`; after ~30s idle, verify `sessions_stale_finalized` increments.
3. Wait >60s and verify `sessions_pruned` increments and old session folders are removed from `/tmp/ik_kv_handoff`.

## 6. Prefill threshold/crossover validation

Goal: confirm RTX prefill is used only when prompt length merits it.

1. Run short prompt below crossover (e.g. 512 tokens).
2. Run prompt near crossover.
3. Run long prompt well above crossover (e.g. 8k+ tokens).

Use:

- `--prefill-min-stream-batch-tokens -1`
- `LLAMA_PREFILL_STREAM_FLOOR_MS`
- `LLAMA_PREFILL_STREAM_TOK_S`
- `LLAMA_PREFILL_DECODE_TOK_S`

Pass criteria:

- short prompts avoid expensive streaming path
- long prompts use prefill handoff path
- behavior tracks configured crossover assumptions

## 7. Restore compatibility guardrails

1. Multi-stream import support:
   - run with `--kv-streams 1` and `--kv-streams 2` on prefill sender
   - verify both complete without `N_STREAM_UNSUPPORTED` reject in receiver session summaries
2. Graph split + flash-attn constraint:
   - for `--split-mode graph`, run decode with `--flash-attn`
   - if testing decode without flash-attn, switch away from graph split for import/restore runs
   - verify decode logs do not contain `Transposed V cache is not sypported with split mode 'graph'`

## 8. Soak test (recommended 1-4h)

Example loop from `RTX_HOST`:

```bash
for i in $(seq 1 200); do
  echo "run ${i}"
  ./build/bin/llama-cli \
    -m /models/your-model.gguf \
    -f /prompts/soak_prompt.txt \
    -c 32768 -n 32 \
    -ps --prefill-overlap \
    --prefill-min-stream-batch-tokens -1 \
    --prefill-decode-mode split_thunderbolt \
    --prefill-transport-mode progressive \
    --kv-transport auto --kv-transport-fallback \
    --kv-host 10.40.0.20 --kv-port 19001 || break

  curl -s http://10.40.0.20:8080/kv-receiver/status | jq '.counters'
done
```

Pass criteria:

- no stuck runs
- no steady growth of `frames_bad`
- no invalid artifacts
- stable decode service health

## 9. Artifacts to archive for each test batch

1. Receiver snapshot:
   - `curl http://10.40.0.20:8080/kv-receiver/status`
2. Receiver session files:
   - `/tmp/ik_kv_handoff/session_*/session_summary.json`
3. Decode server logs and RPC server logs
4. Prefill host logs with transport mode and threshold settings

## 10. Phase-1 queue validation

1. Local queue self-test:

```bash
cd /path/to/ik_llama.cpp
./scripts/test_prefill_decode_job_queue.sh
```

2. Serialized worker check on real commands:

- Start queue worker
- Submit multiple jobs
- Verify `status` shows at most one active non-terminal job at a time
- Verify completed jobs progress through `prefill_running -> artifact_ready -> handoff_running -> decode_running -> done`

## 11. Phase-2 split pipeline queue validation

1. Local deterministic self-test:

```bash
cd /path/to/ik_llama.cpp
./scripts/test_prefill_decode_phase2_pipeline.sh
```

2. Real-environment split-worker check:

- Start `prefill-worker` and `handoff-worker` in parallel.
- Submit at least 3 `phase2_split_pipeline` jobs with different prompt files.
- Confirm queue status shows:
  - `artifact_ready_depth` can increase above 0 while handoff is busy
  - only one handoff/decode stage active at a time
- Confirm failed handoff retries return state to `artifact_ready` (not `queued`) and do not rerun prefill.

3. Transport/crossover consistency check:

- Submit with `--kv-transport tcp`, then `rdma`, then `auto`.
- For strict mode checks, include `--no-kv-transport-fallback`; then repeat with fallback enabled.
- Keep `--prefill-min-stream-batch-tokens -1`.
- Verify expected state progression and no compatibility regressions (`N_STREAM_UNSUPPORTED`, graph-split restore errors).
