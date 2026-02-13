# Phase-1 Prefill/Decode Queue (Single-Flight)

This queue provides a serialized prefill -> handoff -> decode workflow using external scripts only.

It is intended to prevent multi-sequence handoff collisions while current bridge import is single-stream constrained.

For split prefill/handoff workers (prefill overlap with serialized handoff), see:
`docs/development/prefill_decode_phase2_pipeline.md`.

## Scope

- Persistent on-disk queue/spool
- Single active worker lock (single-flight execution)
- Job states and event log
- Runs existing buffered E2E runner:
  - `scripts/run_single_machine_buffered_e2e.sh`

## Files

- Queue script:
  - `scripts/prefill_decode_job_queue.py`
- Default spool root:
  - `/tmp/ik_prefill_decode_queue`

Spool layout:

- `config.json`
- `jobs/<job_id>.json`
- `events/<job_id>.jsonl`
- `runs/<job_id>/...`

## Job states

- `queued`
- `prefill_running`
- `artifact_ready`
- `handoff_running`
- `decode_running`
- `done`
- `failed`
- `canceled`

## Initialize

```bash
./scripts/prefill_decode_job_queue.py init \
  --spool-dir /tmp/ik_prefill_decode_queue \
  --max-queued 256
```

## Submit

```bash
./scripts/prefill_decode_job_queue.py submit \
  --spool-dir /tmp/ik_prefill_decode_queue \
  --model /path/to/model.gguf \
  --prompt-file /path/to/prompt.txt \
  --rtx-repo /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA \
  --kv-transport auto \
  --prefill-min-stream-batch-tokens -1
```

Notes:

- `--prefill-min-stream-batch-tokens -1` preserves runtime crossover/threshold logic.
- For compatibility with current bridge import, keep handoff artifacts effectively single logical non-empty stream.
- Transport metadata is tracked per job via `--kv-transport` and exported to child env as `IK_PDQ_KV_TRANSPORT`.

## Run worker

Process one job:

```bash
./scripts/prefill_decode_job_queue.py worker --once
```

Run continuously:

```bash
./scripts/prefill_decode_job_queue.py worker
```

## Inspect jobs

```bash
./scripts/prefill_decode_job_queue.py list
./scripts/prefill_decode_job_queue.py status --json
./scripts/prefill_decode_job_queue.py show <job_id>
```

Cancel queued job:

```bash
./scripts/prefill_decode_job_queue.py cancel <job_id>
```

## External Command Mode (testing/integration)

For deterministic queue/worker tests without model/network dependencies:

```bash
./scripts/prefill_decode_job_queue.py submit \
  --mode external_command \
  --command "/path/to/runner_or_test_script.sh" \
  --kv-transport rdma
```

Recommended real handoff wrapper:

```bash
./scripts/prefill_decode_job_queue.py submit \
  --mode external_command \
  --kv-transport auto \
  --command "./scripts/run_phase1_prefill_handoff_job.sh \
    --model /models/your-model.gguf \
    --prompt-file /prompts/long_prompt.txt \
    --rtx-repo /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA \
    --decode-host 10.40.0.20 \
    --decode-port 19001"
```

Built-in self-test:

```bash
./scripts/test_prefill_decode_job_queue.sh
```

## Guardrails

- Worker lock enforces one active job.
- Queue size and spool-byte limits can be enforced via `config.json`.
- Runner logs are persisted under `runs/<job_id>/worker_stream.log`.
- Worker launch exceptions are captured and persisted in `result.runner_exception`
  with `return_code=127`; jobs retry/fail cleanly instead of being left active.
- External command mode guardrails (enabled by default):
  - reject `--kv-streams` values other than `1`
  - reject `--split-mode graph` unless `--flash-attn` is present
  - reject `--split-mode graph` with `--no-flash-attn`
