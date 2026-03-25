# Hybrid RTX+Mac Decode Handoff Test Guide

End-to-end test procedure for the layer-major streaming prefill and decode handoff pipeline. The RTX machine prefills the prompt, exports KV cache over TCP, and the decode receiver imports KV and runs autoregressive decode.

## Prerequisites

- ik_llama.cpp built with CUDA: `cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release`
- Two executables: `build/bin/Release/llama-cli.exe` and `build/bin/Release/decode-receiver.exe`
- Model GGUF files accessible on both sides (or same machine for localhost test)

## Test 1: Ministral 3B Q4_K_S (small, fast, single-shard)

Model: `D:/models/Models/bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF/mistralai_Ministral-3-3B-Instruct-2512-Q4_K_S.gguf`

### Terminal 1 - Start decode receiver

```bash
./build/bin/Release/decode-receiver.exe \
  -m "D:/models/Models/bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF/mistralai_Ministral-3-3B-Instruct-2512-Q4_K_S.gguf" \
  -ngl 99 \
  -c 8000 \
  -b 8000 \
  --kv-port 9100 \
  --token-stream-port 9101 \
  -n 128
```

Wait for: `Listening for KV artifact on 0.0.0.0:9100...`

### Terminal 2 - Run prefiller with handoff

```bash
./build/bin/Release/llama-cli.exe \
  -m "D:/models/Models/bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF/mistralai_Ministral-3-3B-Instruct-2512-Q4_K_S.gguf" \
  -ngl 99 \
  -c 8000 \
  -b 8000 \
  --layer-major \
  --handoff-kv-host 127.0.0.1 \
  --handoff-kv-port 9100 \
  -p "Explain the theory of general relativity in simple terms." \
  -n 128 \
  --no-display-prompt
```

### Expected output

**Prefiller (Terminal 2):**
```
main: prefill complete (N tokens), handing off to 127.0.0.1:9100...
KV artifact published: ~1.5 MB in <10 ms (>1 Gbps)
```

**Receiver (Terminal 1):**
```
Received ~1.5 MB in <10 ms, 1 chunks
Artifact parsed: N tokens, 26 layers, ~1.5 MB payload
KV state restored: ~1.5 MB bytes loaded
Starting decode (max 128 tokens, temp=0.70)...
[coherent English text about general relativity]
Decode complete: 128 tokens in ~500 ms (~250 tok/s)
```

### What to verify

- Receiver prints coherent English (not garbage)
- KV artifact size is proportional to token count x layers x head dims
- No crashes, no assertion failures
- Prefiller exits after handoff (does not decode locally)

---

## Test 2: Kimi K2.5 Q3_K_XL (500GB MoE, multi-shard)

Model: `D:/big_models/unsloth/Kimi_25_Q3_K_XL/Kimi-K2.5-UD-Q3_K_XL-00001-of-00011.gguf`

This is a 500GB MoE model with 61 layers and 384 experts. It will NOT fit in GPU VRAM without streaming. The layer-major mode is essential here.

### Terminal 1 - Start decode receiver

```bash
./build/bin/Release/decode-receiver.exe \
  -m "D:/big_models/unsloth/Kimi_25_Q3_K_XL/Kimi-K2.5-UD-Q3_K_XL-00001-of-00011.gguf" \
  -ngl 99 \
  -c 8000 \
  -b 8000 \
  --kv-port 9100 \
  --token-stream-port 9101 \
  -n 128
```

Wait for: `Listening for KV artifact on 0.0.0.0:9100...`

Note: Model load will take 2-3 minutes (500GB across 11 shards). KV cache at 8k context with 61 layers will be ~4 GB.

### Terminal 2 - Run prefiller with handoff

```bash
./build/bin/Release/llama-cli.exe \
  -m "D:/big_models/unsloth/Kimi_25_Q3_K_XL/Kimi-K2.5-UD-Q3_K_XL-00001-of-00011.gguf" \
  -ngl 99 \
  -c 8000 \
  -b 8000 \
  --layer-major \
  --handoff-kv-host 127.0.0.1 \
  --handoff-kv-port 9100 \
  -p "Explain quantum entanglement and why Einstein called it spooky action at a distance." \
  -n 128 \
  --no-display-prompt
```

### Expected output

**Prefiller (Terminal 2):**
```
main: prefill complete (N tokens), handing off to 127.0.0.1:9100...
KV artifact published: ~50-100 MB in <100 ms
```

**Receiver (Terminal 1):**
```
Received ~50-100 MB in <100 ms
Artifact parsed: N tokens, 61 layers, ~50-100 MB payload
KV state restored
Starting decode...
[coherent English text about quantum entanglement]
Decode complete: 128 tokens in ~X ms
```

### What to verify

- Same coherence checks as Test 1
- KV artifact is much larger (61 layers, MLA compressed K cache)
- Layer-major prefill actually ran (not standard decode) -- check for per-layer timing if LLAMA_PREFILL_LAYER_TRACE=1 is set
- Both processes can load the 11-shard model without OOM

---

## Flags reference

### Prefiller (llama-cli)

| Flag | Purpose |
|------|---------|
| `--layer-major` | Enable layer-major decode (per-layer callback scheduling) |
| `--handoff-kv-host HOST` | Remote KV receiver IP (triggers handoff after prefill) |
| `--handoff-kv-port PORT` | KV receiver TCP port (default: 9100) |
| `--handoff-ts-host HOST` | Token stream host (default: same as kv-host) |
| `--handoff-ts-port PORT` | Token stream TCP port (default: 9101) |
| `-c N` | Context size |
| `-b N` | Batch size (should match context for single-batch prefill) |

### Decode receiver

| Flag | Purpose |
|------|---------|
| `-m PATH` | Model GGUF (must match prefiller's model) |
| `-ngl N` | GPU layers (99 = all) |
| `-c N` | Context size (must be >= prefiller's token count) |
| `-b N` | Batch size |
| `--kv-port PORT` | Listen port for KV artifacts (default: 9100) |
| `--token-stream-port PORT` | Token relay port (default: 9101) |
| `-n N` | Max decode tokens |
| `--temp F` | Temperature (default: 0.7) |
| `--one-shot` | Exit after first decode (otherwise loops) |
| `--kv-artifact PATH` | File mode: read KV from file instead of TCP |

### Environment variables

| Var | Purpose |
|-----|---------|
| `LLAMA_PREFILL_LAYER_TRACE=1` | Print per-layer begin/end with timing |

---

## Troubleshooting

**"failed to connect to HOST:PORT"** - Receiver isn't listening yet. Wait for the `Listening for KV artifact` message before starting the prefiller.

**"accept timeout"** - Receiver waited too long (default 2 min). Start the prefiller sooner, or increase timeout.

**Garbage output from receiver** - Model mismatch between prefiller and receiver, or KV cache format incompatibility. Ensure both load the exact same GGUF.

**OOM on receiver** - Context size too large for available VRAM. Reduce `-c` or `-ngl`.

**"Oops: tensor with strange name output_norm.weight"** - Harmless diagnostic from the layer boundary scanner. The output layer doesn't have a `-N` suffix so it gets flagged. Does not affect correctness.

---

## Cross-machine test (RTX to Mac)

For the real hybrid topology, replace `127.0.0.1` with the Mac's IP address. Both machines must:
- Load the same model
- Have TCP connectivity on ports 9100 and 9101
- Use matching context sizes

The decode receiver runs natively on Mac (Metal backend) while the prefiller runs on RTX (CUDA backend). The KV artifact format is architecture-neutral.
