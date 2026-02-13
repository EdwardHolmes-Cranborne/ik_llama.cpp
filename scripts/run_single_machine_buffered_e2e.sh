#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_RTX_REPO="$(cd -- "${IK_REPO}/.." && pwd)/RTX_ACCELERATED_MAC_PREFILL_LLAMA"

RTX_REPO="${DEFAULT_RTX_REPO}"
MODEL_PATH=""
PROMPT_FILE=""
OUTPUT_DIR="/tmp/ik_rtx_single_machine_buffered_e2e_$(date +%Y%m%d_%H%M%S)"

IK_HTTP_PORT=8080
IK_KV_PORT=19001
BUFFER_PORT=29001
CTX_SIZE=8192
PREFILL_N_PREDICT=8
DECODE_N_PREDICT=16
PREFILL_MIN_STREAM_BATCH_TOKENS=-1
KV_CHUNK_BYTES=$((4 * 1024 * 1024))
KV_MAX_INFLIGHT=$((256 * 1024 * 1024))
LOOPBACK_IDLE_TIMEOUT=20
REPLAY_USE_ACK=1

IK_SERVER_PID=""
LOOPBACK_PID=""

usage() {
    cat <<'EOF'
Single-machine buffered E2E suite (prefill -> disk buffer -> replay -> ik decode).

Required:
  --model PATH                GGUF model path (used by both prefill and ik server)
  --prompt-file PATH          prompt file for prefill run

Optional:
  --rtx-repo PATH             RTX prefill fork root (default: ../RTX_ACCELERATED_MAC_PREFILL_LLAMA)
  --output-dir PATH           suite output directory
  --ik-http-port N            ik llama-server HTTP port (default: 8080)
  --ik-kv-port N              ik kv receiver port (default: 19001)
  --buffer-port N             loopback disk-buffer port (default: 29001)
  --ctx-size N                context size for both runs (default: 8192)
  --prefill-n-predict N       prefill CLI generation tokens (default: 8)
  --prefill-min-stream-batch-tokens N
                              prefill streaming threshold (-1 = runtime auto/crossover)
  --decode-n-predict N        decode validation request tokens (default: 16)
  --kv-chunk-bytes N          replay/send chunk size (default: 4194304)
  --kv-max-inflight N         sender in-flight window bytes (default: 268435456)
  --loopback-idle-timeout N   loopback idle timeout seconds (default: 20)
  --no-replay-ack             replay without waiting on receiver ACKs
  -h, --help                  show this help
EOF
}

cleanup() {
    set +e
    if [[ -n "${LOOPBACK_PID}" ]]; then
        kill "${LOOPBACK_PID}" >/dev/null 2>&1 || true
        wait "${LOOPBACK_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${IK_SERVER_PID}" ]]; then
        kill "${IK_SERVER_PID}" >/dev/null 2>&1 || true
        wait "${IK_SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --rtx-repo) RTX_REPO="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --ik-http-port) IK_HTTP_PORT="$2"; shift 2 ;;
        --ik-kv-port) IK_KV_PORT="$2"; shift 2 ;;
        --buffer-port) BUFFER_PORT="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --prefill-n-predict) PREFILL_N_PREDICT="$2"; shift 2 ;;
        --prefill-min-stream-batch-tokens) PREFILL_MIN_STREAM_BATCH_TOKENS="$2"; shift 2 ;;
        --decode-n-predict) DECODE_N_PREDICT="$2"; shift 2 ;;
        --kv-chunk-bytes) KV_CHUNK_BYTES="$2"; shift 2 ;;
        --kv-max-inflight) KV_MAX_INFLIGHT="$2"; shift 2 ;;
        --loopback-idle-timeout) LOOPBACK_IDLE_TIMEOUT="$2"; shift 2 ;;
        --no-replay-ack) REPLAY_USE_ACK=0; shift 1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL_PATH}" || -z "${PROMPT_FILE}" ]]; then
    echo "missing required args: --model and --prompt-file" >&2
    usage
    exit 2
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "model not found: ${MODEL_PATH}" >&2
    exit 2
fi
if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "prompt file not found: ${PROMPT_FILE}" >&2
    exit 2
fi
if [[ ! -d "${RTX_REPO}" ]]; then
    echo "rtx repo not found: ${RTX_REPO}" >&2
    exit 2
fi

command -v curl >/dev/null
command -v python3 >/dev/null

IK_SERVER_BIN_CANDIDATES=(
    "${IK_REPO}/build_codex/bin/llama-server"
    "${IK_REPO}/build/bin/llama-server"
)
IK_SERVER_BIN=""
for c in "${IK_SERVER_BIN_CANDIDATES[@]}"; do
    if [[ -x "${c}" ]]; then
        IK_SERVER_BIN="${c}"
        break
    fi
done
if [[ -z "${IK_SERVER_BIN}" ]]; then
    echo "ik llama-server binary not found (checked build_codex/bin and build/bin)" >&2
    exit 2
fi

PREFILL_BIN_CANDIDATES=(
    "${RTX_REPO}/prefill_llama.cpp/build_codex/bin/llama-cli"
    "${RTX_REPO}/prefill_llama.cpp/build/bin/llama-cli"
)
PREFILL_BIN=""
for c in "${PREFILL_BIN_CANDIDATES[@]}"; do
    if [[ -x "${c}" ]]; then
        PREFILL_BIN="${c}"
        break
    fi
done
if [[ -z "${PREFILL_BIN}" ]]; then
    echo "prefill llama-cli binary not found (checked build_codex/bin and build/bin)" >&2
    exit 2
fi

LOOPBACK_RX_SCRIPT="${RTX_REPO}/scripts/tbp_loopback_receiver.py"
if [[ ! -f "${LOOPBACK_RX_SCRIPT}" ]]; then
    echo "loopback receiver script not found: ${LOOPBACK_RX_SCRIPT}" >&2
    exit 2
fi
REPLAY_SCRIPT="${IK_REPO}/scripts/tbp_replay_to_kv_receiver.py"
if [[ ! -f "${REPLAY_SCRIPT}" ]]; then
    echo "replay script not found: ${REPLAY_SCRIPT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/ik_slots" "${OUTPUT_DIR}/ik_kv_recv" "${OUTPUT_DIR}/buffer/chunks" "${OUTPUT_DIR}/buffer/reassembled" "${OUTPUT_DIR}/replay"

IK_LOG="${OUTPUT_DIR}/ik_server.log"
LOOPBACK_LOG="${OUTPUT_DIR}/loopback_receiver.log"
PREFILL_LOG="${OUTPUT_DIR}/prefill.log"
REPLAY_LOG="${OUTPUT_DIR}/replay.log"

echo "[1/7] start ik llama-server with kv-receiver"
"${IK_SERVER_BIN}" \
  -m "${MODEL_PATH}" \
  --host 127.0.0.1 --port "${IK_HTTP_PORT}" \
  -c "${CTX_SIZE}" \
  --slot-save-path "${OUTPUT_DIR}/ik_slots" \
  --kv-recv-enable \
  --kv-transport tcp \
  --kv-recv-host 127.0.0.1 \
  --kv-recv-port "${IK_KV_PORT}" \
  --kv-recv-slot 0 \
  --kv-recv-output-dir "${OUTPUT_DIR}/ik_kv_recv" \
  --kv-recv-max-connections 32 \
  --kv-recv-idle-timeout 60 \
  >"${IK_LOG}" 2>&1 &
IK_SERVER_PID=$!

for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${IK_HTTP_PORT}/health" >/dev/null; then
        break
    fi
    sleep 1
done
if ! curl -sf "http://127.0.0.1:${IK_HTTP_PORT}/health" >/dev/null; then
    echo "ik llama-server health check failed; see ${IK_LOG}" >&2
    exit 1
fi

echo "[2/7] start loopback disk buffer receiver"
python3 "${LOOPBACK_RX_SCRIPT}" \
  --host 127.0.0.1 \
  --port "${BUFFER_PORT}" \
  --idle-timeout-sec "${LOOPBACK_IDLE_TIMEOUT}" \
  --ack-mode always \
  --chunk-output-dir "${OUTPUT_DIR}/buffer/chunks" \
  --reassemble-output-dir "${OUTPUT_DIR}/buffer/reassembled" \
  --output "${OUTPUT_DIR}/buffer/loopback_receiver.json" \
  >"${LOOPBACK_LOG}" 2>&1 &
LOOPBACK_PID=$!

echo "[3/7] run prefill sender into disk buffer"
LLAMA_PREFILL_TB_ENABLE=1 \
"${PREFILL_BIN}" \
  -m "${MODEL_PATH}" \
  -f "${PROMPT_FILE}" \
  -c "${CTX_SIZE}" \
  -n "${PREFILL_N_PREDICT}" \
  -ps --prefill-overlap \
  --prefill-min-stream-batch-tokens "${PREFILL_MIN_STREAM_BATCH_TOKENS}" \
  --decode-mode split_thunderbolt \
  --decode-remote-layers 1 \
  --prefill-decode-transport-required \
  --prefill-transport-mode progressive \
  --prefill-execution-mode coupled \
  --kv-transport tcp \
  --kv-host 127.0.0.1 \
  --kv-port "${BUFFER_PORT}" \
  --kv-streams 1 \
  --kv-stream-chunk-bytes "${KV_CHUNK_BYTES}" \
  --kv-max-inflight-bytes "${KV_MAX_INFLIGHT}" \
  >"${PREFILL_LOG}" 2>&1

wait "${LOOPBACK_PID}"
LOOPBACK_PID=""

ARTIFACT_PATH="$(find "${OUTPUT_DIR}/buffer/reassembled" -type f -name 'kv_artifact.bin' | sort | tail -n 1)"
if [[ -z "${ARTIFACT_PATH}" || ! -f "${ARTIFACT_PATH}" ]]; then
    echo "no reassembled artifact found under ${OUTPUT_DIR}/buffer/reassembled; see ${LOOPBACK_LOG}" >&2
    exit 1
fi

echo "[4/7] replay buffered artifact to ik kv-receiver"
replay_args=(
  --artifact "${ARTIFACT_PATH}"
  --host 127.0.0.1
  --port "${IK_KV_PORT}"
  --chunk-bytes "${KV_CHUNK_BYTES}"
  --output "${OUTPUT_DIR}/replay/replay_result.json"
)
if [[ "${REPLAY_USE_ACK}" == "1" ]]; then
    replay_args+=(--ack-required --wait-ack)
fi
python3 "${REPLAY_SCRIPT}" "${replay_args[@]}" >"${REPLAY_LOG}" 2>&1

echo "[5/7] capture kv-receiver status"
curl -sf "http://127.0.0.1:${IK_HTTP_PORT}/kv-receiver/status" > "${OUTPUT_DIR}/ik_kv_receiver_status.json"

echo "[6/7] validate replay counters"
python3 - "${OUTPUT_DIR}/ik_kv_receiver_status.json" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
data = json.loads(status_path.read_text(encoding="utf-8"))
counters = data.get("counters", {})
validated = int(counters.get("artifacts_validated", 0))
reassembled = int(counters.get("artifacts_reassembled", 0))
restore_enqueued = int(counters.get("restore_tasks_enqueued", 0))
if reassembled < 1 or validated < 1 or restore_enqueued < 1:
    raise SystemExit(
        "expected artifacts_reassembled>=1, artifacts_validated>=1, "
        f"and restore_tasks_enqueued>=1; got reassembled={reassembled}, "
        f"validated={validated}, restore_tasks_enqueued={restore_enqueued}"
    )
sessions = data.get("sessions", [])
good_session = None
for session in sessions:
    if (
        bool(session.get("finalized")) and
        bool(session.get("validation_ok")) and
        bool(session.get("restore_enqueued")) and
        int(session.get("bytes_received", 0)) > 0
    ):
        good_session = {
            "session_id": session.get("session_id"),
            "chunks_received": int(session.get("chunks_received", 0)),
            "bytes_received": int(session.get("bytes_received", 0)),
        }
        break
if good_session is None:
    raise SystemExit("no finalized+validated+restore_enqueued receiver session found")

print(json.dumps({
    "ok": True,
    "artifacts_reassembled": reassembled,
    "artifacts_validated": validated,
    "restore_tasks_enqueued": restore_enqueued,
    "session": good_session,
}, indent=2))
PY

echo "[7/7] run decode smoke request on ik server"
cat > "${OUTPUT_DIR}/decode_request.json" <<EOF
{
  "prompt": "Buffered handoff decode smoke test.",
  "n_predict": ${DECODE_N_PREDICT}
}
EOF
curl -sf \
  -H "Content-Type: application/json" \
  -d @"${OUTPUT_DIR}/decode_request.json" \
  "http://127.0.0.1:${IK_HTTP_PORT}/completion" \
  > "${OUTPUT_DIR}/decode_response.json"

python3 - "${OUTPUT_DIR}/decode_response.json" <<'PY'
import json
import sys
from pathlib import Path

resp = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(resp, dict):
    raise SystemExit("decode response is not a JSON object")
if "content" not in resp and "choices" not in resp:
    raise SystemExit("decode response missing expected content fields")
print(json.dumps({"ok": True, "keys": sorted(resp.keys())[:10]}, indent=2))
PY

echo
echo "single-machine buffered e2e suite completed"
echo "output_dir: ${OUTPUT_DIR}"
echo "ik_server_log: ${IK_LOG}"
echo "prefill_log: ${PREFILL_LOG}"
echo "loopback_log: ${LOOPBACK_LOG}"
echo "replay_log: ${REPLAY_LOG}"
