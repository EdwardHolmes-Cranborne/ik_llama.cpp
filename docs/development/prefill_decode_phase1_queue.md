# Phase-1 Prefill/Decode Queue (Single-Flight)

This queue provides a serialized prefill -> handoff -> decode workflow using external scripts only.

It is intended to prevent multi-sequence handoff collisions while current bridge import is single-stream constrained.

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
  --prefill-min-stream-batch-tokens -1
```

Notes:

- `--prefill-min-stream-batch-tokens -1` preserves runtime crossover/threshold logic.
- For compatibility with current bridge import, keep handoff artifacts effectively single logical non-empty stream.

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
./scripts/prefill_decode_job_queue.py show <job_id>
```

Cancel queued job:

```bash
./scripts/prefill_decode_job_queue.py cancel <job_id>
```

## Guardrails

- Worker lock enforces one active job.
- Queue size and spool-byte limits can be enforced via `config.json`.
- Runner logs are persisted under `runs/<job_id>/worker_stream.log`.
