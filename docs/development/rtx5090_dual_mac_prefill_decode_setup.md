# RTX5090 -> Dual-Mac Decode Setup Guide

This guide documents the target deployment:

- Prefill host: WRX90 + RTX 5090 + NVMe SSD
- Decode cluster:
  - Mac Studio (512GB)
  - MacBook Pro M3 Max (128GB)
- Transport fabric: USB4 / Thunderbolt networking/tunneling
- Flow: RTX prefill -> KV handoff -> ik `llama-server` decode across both Macs

## 1. Host roles and addresses

Use fixed addresses on the Thunderbolt/USB4 fabric:

- `RTX_HOST` (WRX90): `10.40.0.10`
- `MAC_STUDIO` (decode coordinator): `10.40.0.20`
- `MAC_MBP` (decode worker): `10.40.0.21`

All commands below assume these values.

## 2. Build binaries

### 2.1 Build ik on both Macs

Run on both `MAC_STUDIO` and `MAC_MBP`:

```bash
cd /path/to/ik_llama.cpp
cmake -S . -B build_codex -DGGML_METAL=ON -DGGML_RPC=ON
cmake --build build_codex -j
```

### 2.2 Build RTX prefill fork on WRX90

```bash
cd /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA/prefill_llama.cpp
cmake -S . -B build -DGGML_CUDA=ON
cmake --build build -j
```

## 3. Start decode worker RPC servers (both Macs)

### 3.1 Mac Studio RPC server

```bash
cd /path/to/ik_llama.cpp
./build_codex/bin/rpc-server --host 0.0.0.0 --port 50052 --device METAL0
```

### 3.2 MacBook Pro RPC server

```bash
cd /path/to/ik_llama.cpp
./build_codex/bin/rpc-server --host 0.0.0.0 --port 50052 --device METAL0
```

## 4. Start decode coordinator (`llama-server`) on Mac Studio

Create decode route cluster config (used for post-restore fanout to additional decode nodes):

```bash
cat > /tmp/ik_decode_cluster.json <<'JSON'
{
  "nodes": [
    {
      "node_id": "mac_studio",
      "kv_host": "10.40.0.20",
      "kv_port": 19001,
      "rpc_endpoint": "10.40.0.20:19001",
      "role": "coordinator",
      "weight": 2,
      "healthy": true,
      "promotable": false
    },
    {
      "node_id": "mac_mbp",
      "kv_host": "10.40.0.21",
      "kv_port": 19001,
      "rpc_endpoint": "10.40.0.21:19001",
      "role": "worker",
      "weight": 2,
      "healthy": true,
      "promotable": true
    }
  ]
}
JSON
```

```bash
cd /path/to/ik_llama.cpp
mkdir -p /tmp/ik_slots /tmp/ik_kv_handoff

./build_codex/bin/llama-server \
  -m /models/your-model.gguf \
  --host 0.0.0.0 --port 8080 \
  -c 32768 \
  --flash-attn on \
  -ngl 999 \
  --split-mode graph \
  --max-gpu 2 \
  --tensor-split 0.70,0.30 \
  --rpc 10.40.0.20:50052,10.40.0.21:50052 \
  --slot-save-path /tmp/ik_slots \
  --kv-recv-enable \
  --kv-transport auto \
  --kv-transport-fallback \
  --kv-recv-host 0.0.0.0 \
  --kv-recv-port 19001 \
  --kv-recv-slot 0 \
  --kv-recv-output-dir /tmp/ik_kv_handoff \
  --kv-recv-max-connections 64 \
  --kv-recv-idle-timeout 120 \
  --kv-recv-stale-finalize-timeout 180 \
  --kv-recv-session-retention 3600 \
  --kv-recv-cleanup-interval 10 \
  --decode-node-id mac_studio \
  --decode-cluster-file /tmp/ik_decode_cluster.json \
  --decode-route-dispatch-enable \
  --decode-route-dispatch-max-hops 1 \
  --decode-route-dispatch-streams 2 \
  --decode-route-dispatch-chunk-bytes 4194304 \
  --decode-route-dispatch-max-inflight-bytes 268435456
```

Verify receiver state:

```bash
curl -s http://10.40.0.20:8080/kv-receiver/status | jq
curl -s http://10.40.0.20:8080/props | jq '.decode_route'
```

Receiver lifecycle notes:

1. `--kv-recv-stale-finalize-timeout N`: force finalize incomplete sessions after `N` idle seconds (`0` disables).
2. `--kv-recv-session-retention N`: keep finalized sessions and artifacts for `N` seconds before pruning (`0` disables).
3. `--kv-recv-cleanup-interval N`: maintenance sweep interval.

## 5. Start RTX prefill client on WRX90

Set crossover logic inputs (keep dynamic prefill threshold behavior enabled):

```bash
export LLAMA_PREFILL_TB_ENABLE=1
export LLAMA_PREFILL_STREAM_FLOOR_MS=4000
export LLAMA_PREFILL_STREAM_TOK_S=2500
export LLAMA_PREFILL_DECODE_TOK_S=220
```

Run prefill/decode handoff:

```bash
cd /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA/prefill_llama.cpp

./build/bin/llama-cli \
  -m /models/your-model.gguf \
  -f /prompts/long_prompt.txt \
  -c 32768 \
  -n 128 \
  -ps --prefill-overlap \
  --prefill-min-stream-batch-tokens -1 \
  --prefill-decode-mode split_thunderbolt \
  --prefill-decode-transport-required \
  --prefill-transport-mode progressive \
  --prefill-execution-mode coupled \
  --kv-transport auto \
  --kv-transport-fallback \
  --kv-host 10.40.0.20 \
  --kv-port 19001 \
  --kv-streams 2 \
  --kv-stream-chunk-bytes 4194304 \
  --kv-max-inflight-bytes 268435456
```

## 6. Transport mode switching (both supported)

Use the same deployment and change only these controls:

1. `rdma preferred`:
   - prefill: `--kv-transport rdma --kv-transport-fallback`
   - decode: `--kv-transport rdma --kv-transport-fallback`
2. `tcp only`:
   - prefill: `--kv-transport tcp`
   - decode: `--kv-transport tcp`
3. `auto`:
   - prefill: `--kv-transport auto`
   - decode: `--kv-transport auto`

## 7. KV import compatibility constraints (current)

1. RTX payloads with multiple active KV streams are merged into a single IK sequence-state stream during bridge conversion:
   - Bridge import no longer requires single-active-stream RTX payloads.
   - `--kv-streams` can be tuned for transport throughput independently of bridge compatibility.
2. For decode `--split-mode graph`, keep `--flash-attn` enabled:
   - Graph-split restore rejects transposed-V restore path.
   - If you must run without flash attention, avoid `--split-mode graph` for restore/import runs.

## 8. Notes for first production run

1. Start with `--kv-recv-dry-run` on decode for transport-only validation.
2. Then remove dry-run to allow slot restore/import.
3. Keep prompt lengths above the computed crossover for RTX prefill benefits; with `--prefill-min-stream-batch-tokens -1`, crossover remains auto-derived from runtime inputs.

## 9. Phase-1 serialized queue mode (recommended current default)

Use the queue worker to enforce one active handoff/decode job and avoid multi-sequence import collisions:

```bash
cd /path/to/ik_llama.cpp
./scripts/prefill_decode_job_queue.py init
./scripts/prefill_decode_job_queue.py worker
```

For deployment integration, submit `external_command` jobs that run your prefill+handoff wrapper:

```bash
./scripts/prefill_decode_job_queue.py submit \
  --mode external_command \
  --command "./scripts/run_phase1_prefill_handoff_job.sh \
    --model /models/your-model.gguf \
    --prompt-file /prompts/long_prompt.txt \
    --rtx-repo /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA \
    --decode-host 10.40.0.20 \
    --decode-port 19001 \
    --kv-streams 2" \
  --kv-transport auto
```

Queue env exported to the child process includes:

- `IK_PDQ_JOB_ID`
- `IK_PDQ_JOB_MODE`
- `IK_PDQ_KV_TRANSPORT`

## 10. Phase-2 split pipeline queue mode (prefill overlap + serialized handoff)

Phase-2 allows prefill jobs to keep running and staging artifacts while decode import/handoff remains serialized.

Start both workers:

```bash
cd /path/to/ik_llama.cpp
./scripts/prefill_decode_job_queue.py init
./scripts/prefill_decode_job_queue.py prefill-worker
./scripts/prefill_decode_job_queue.py handoff-worker
```

Submit one job:

```bash
./scripts/prefill_decode_job_queue.py submit \
  --mode phase2_split_pipeline \
  --model /models/your-model.gguf \
  --prompt-file /prompts/long_prompt.txt \
  --rtx-repo /path/to/RTX_ACCELERATED_MAC_PREFILL_LLAMA \
  --decode-host 10.40.0.20 \
  --decode-port 19001 \
  --kv-transport auto \
  --kv-streams 2 \
  --prefill-min-stream-batch-tokens -1 \
  --max-prefill-retries 1 \
  --max-handoff-retries 2
```

Operational notes:

1. `prefill-worker` advances jobs to `artifact_ready`; `handoff-worker` drains `artifact_ready` one-at-a-time.
2. Handoff retry does not recompute prefill; it reuses persisted artifact path.
3. Phase-2 handoff replay supports `--kv-transport auto|rdma|tcp|mixed` plus fallback control, and maps `rdma` mode to RDMA endpoint/bind-address environment (`LLAMA_PREFILL_KV_RDMA_*`).
4. For strict live kernel-level RDMA-verbs validation, keep using direct prefill sender handoff flow.
