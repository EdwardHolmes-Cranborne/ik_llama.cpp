# Phase-2 Prefill/Decode Queue (Split Pipeline)

Phase-2 adds a split workflow so prefill can continue while decode/handoff remains single-flight.

Goal:

- Keep one active handoff/decode import at a time
- Allow additional queued requests to run prefill and persist KV artifacts
- Drain prepared artifacts to decode in order

## Scope

- Queue mode: `phase2_split_pipeline`
- Two workers:
  - `prefill-worker`: `queued -> prefill_running -> artifact_ready`
  - `handoff-worker`: `artifact_ready -> handoff_running -> decode_running -> done`
- Stage-specific retries:
  - `max_prefill_retries`
  - `max_handoff_retries`
- Persisted stage results under `result.prefill` and `result.handoff`

## Files

- Queue engine:
  - `scripts/prefill_decode_job_queue.py`
- Stage wrappers:
  - `scripts/run_phase2_prefill_to_artifact.sh`
  - `scripts/run_phase2_handoff_from_artifact.sh`
- Self-test:
  - `scripts/test_prefill_decode_phase2_pipeline.sh`

## Start workers

Use two terminals/services:

```bash
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue init
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue prefill-worker
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue handoff-worker
```

## Submit jobs

```bash
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue submit \
  --mode phase2_split_pipeline \
  --model /models/your-model.gguf \
  --prompt-file /prompts/request_a.txt \
  --rtx-repo /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA \
  --decode-host 10.40.0.20 \
  --decode-port 19001 \
  --kv-transport auto \
  --prefill-min-stream-batch-tokens -1 \
  --max-prefill-retries 1 \
  --max-handoff-retries 2
```

Submit additional prompts while decode is busy; they can advance to `artifact_ready` and wait for handoff.

## Inspect and cancel

```bash
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue status --json
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue list
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue show <job_id>
./scripts/prefill_decode_job_queue.py --spool-dir /tmp/ik_prefill_decode_queue cancel <job_id>
```

`cancel` is allowed for `queued` and `artifact_ready` states.

## Runtime behavior

- `prefill-worker` never starts handoff.
- `handoff-worker` never reruns prefill.
- On prefill success, artifact path is persisted in job metadata (`artifact_path`).
- On handoff failure with retries remaining, job returns to `artifact_ready` without recomputing prefill.

## Transport and compatibility notes

1. Prefill crossover logic remains enabled by default with:
   - `--prefill-min-stream-batch-tokens -1`
2. Import compatibility guardrails still apply:
   - keep `--kv-streams 1`
   - for decode `--split-mode graph`, keep `--flash-attn` enabled
3. Phase-2 handoff stage uses replay utility `scripts/tbp_replay_to_kv_receiver.py`.
   - Replay now accepts `--transport-mode auto|rdma|tcp|mixed` plus `--transport-fallback`.
   - `rdma` mode uses RDMA endpoint/bind-address resolution semantics (`LLAMA_PREFILL_KV_RDMA_*`) with TBP socket transport.
   - For strict live kernel-level RDMA-verbs validation, continue using direct prefill handoff flow (Phase-1 wrapper / direct sender path).

## Test

```bash
./scripts/test_prefill_decode_phase2_pipeline.sh
```

The test validates:

- multiple artifacts can accumulate in `artifact_ready`
- handoff retry returns job to `artifact_ready`
- final progression to `done` with both stage results persisted
