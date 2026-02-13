# Single-Machine Buffered E2E Suite

This suite runs end-to-end prefill and decode on one machine without changing core `llama` or `ik_llama` engine code.

Flow:

1. Start `ik_llama` `llama-server` with KV receiver enabled.
2. Start RTX fork `tbp_loopback_receiver.py` as a virtual network buffer that persists incoming TBP chunks to disk and reassembles `kv_artifact.bin`.
3. Run RTX prefill sender into the loopback buffer (`split_thunderbolt` + transport enabled).
4. Replay the persisted artifact back into `ik_llama` KV receiver over TBP.
5. Validate receiver counters/session state (`reassembled`, `validated`, `restore enqueued`) and run a decode smoke request.

## Prerequisites

- Built binaries:
  - `ik_llama.cpp/build_codex/bin/llama-server`
  - `RTX_ACCELERATED_MAC_PREFILL_LLAMA/prefill_llama.cpp/build_codex/bin/llama-cli`
    - or `.../build/bin/llama-cli`
- Python 3 and `curl`.
- One GGUF model path usable by both binaries.
- Prompt file for prefill.

## Run

From `ik_llama.cpp` repo root:

```bash
./scripts/run_single_machine_buffered_e2e.sh \
  --model /path/to/model.gguf \
  --prompt-file /path/to/prompt.txt \
  --rtx-repo /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA
```

By default the runner uses `--prefill-min-stream-batch-tokens -1` to preserve runtime crossover/threshold behavior.

## Outputs

The suite writes a timestamped output directory under `/tmp/ik_rtx_single_machine_buffered_e2e_*` containing:

- `ik_server.log`
- `prefill.log`
- `loopback_receiver.log`
- `replay.log`
- `buffer/reassembled/session_*/kv_artifact.bin`
- `replay/replay_result.json`
- `ik_kv_receiver_status.json`
- `decode_response.json`

## Notes

- This suite is intentionally external tooling only.
- It validates buffered handoff mechanics and receiver-side restore path in one host environment.
- It does not replace real multi-host RDMA/TCP hardware matrix testing.
- It intentionally uses single-stream replay/send semantics; current bridge import path rejects artifacts with multiple non-empty RTX streams.
- For decode runs that use `--split-mode graph`, keep `--flash-attn` enabled to avoid transposed-V restore incompatibility.
