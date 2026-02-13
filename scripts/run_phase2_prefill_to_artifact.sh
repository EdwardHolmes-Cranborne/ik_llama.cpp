#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH=""
PROMPT_FILE=""
RTX_REPO=""
CTX_SIZE=32768
N_PREDICT=128
PREFILL_MIN_STREAM_BATCH_TOKENS=-1
KV_TRANSPORT="${IK_PDQ_KV_TRANSPORT:-auto}"
KV_STREAM_CHUNK_BYTES=$((4 * 1024 * 1024))
KV_MAX_INFLIGHT_BYTES=$((256 * 1024 * 1024))
BUFFER_HOST="127.0.0.1"
BUFFER_PORT=29001
LOOPBACK_IDLE_TIMEOUT=20
OUTPUT_DIR="/tmp/ik_phase2_prefill_$(date +%Y%m%d_%H%M%S)"

LOOPBACK_PID=""

usage() {
    cat <<'EOF'
Run Phase-2 prefill stage: capture prefill KV stream into disk artifact.

Required:
  --model PATH
  --prompt-file PATH
  --rtx-repo PATH

Optional:
  --ctx-size N
  --n-predict N
  --prefill-min-stream-batch-tokens N
  --kv-transport MODE          auto|tcp|rdma|mixed|disabled
  --kv-stream-chunk-bytes N
  --kv-max-inflight-bytes N
  --buffer-port N
  --loopback-idle-timeout N
  --output-dir PATH
  -h, --help
EOF
}

cleanup() {
    set +e
    if [[ -n "${LOOPBACK_PID}" ]]; then
        kill "${LOOPBACK_PID}" >/dev/null 2>&1 || true
        wait "${LOOPBACK_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --rtx-repo) RTX_REPO="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --n-predict) N_PREDICT="$2"; shift 2 ;;
        --prefill-min-stream-batch-tokens) PREFILL_MIN_STREAM_BATCH_TOKENS="$2"; shift 2 ;;
        --kv-transport) KV_TRANSPORT="$2"; shift 2 ;;
        --kv-stream-chunk-bytes) KV_STREAM_CHUNK_BYTES="$2"; shift 2 ;;
        --kv-max-inflight-bytes) KV_MAX_INFLIGHT_BYTES="$2"; shift 2 ;;
        --buffer-port) BUFFER_PORT="$2"; shift 2 ;;
        --loopback-idle-timeout) LOOPBACK_IDLE_TIMEOUT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL_PATH}" || -z "${PROMPT_FILE}" || -z "${RTX_REPO}" ]]; then
    echo "missing required args: --model --prompt-file --rtx-repo" >&2
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

case "${KV_TRANSPORT}" in
    auto|tcp|rdma|mixed|disabled) ;;
    *)
        echo "invalid --kv-transport: ${KV_TRANSPORT}" >&2
        exit 2
        ;;
esac

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

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/chunks" "${OUTPUT_DIR}/reassembled"
LOOPBACK_LOG="${OUTPUT_DIR}/loopback_receiver.log"
PREFILL_LOG="${OUTPUT_DIR}/prefill.log"
META_PATH="${OUTPUT_DIR}/prefill_artifact_meta.json"

cat > "${OUTPUT_DIR}/prefill_command.txt" <<EOF
LLAMA_PREFILL_TB_ENABLE=1 \
"${PREFILL_BIN}" \
  -m "${MODEL_PATH}" \
  -f "${PROMPT_FILE}" \
  -c "${CTX_SIZE}" \
  -n "${N_PREDICT}" \
  -ps --prefill-overlap \
  --prefill-min-stream-batch-tokens "${PREFILL_MIN_STREAM_BATCH_TOKENS}" \
  --prefill-decode-mode split_thunderbolt \
  --prefill-decode-transport-required \
  --prefill-transport-mode progressive \
  --prefill-execution-mode coupled \
  --kv-transport "${KV_TRANSPORT}" \
  --kv-transport-fallback \
  --kv-host "${BUFFER_HOST}" \
  --kv-port "${BUFFER_PORT}" \
  --kv-streams 1 \
  --kv-stream-chunk-bytes "${KV_STREAM_CHUNK_BYTES}" \
  --kv-max-inflight-bytes "${KV_MAX_INFLIGHT_BYTES}"
EOF

echo "[phase2/1] start loopback receiver"
python3 "${LOOPBACK_RX_SCRIPT}" \
  --host "${BUFFER_HOST}" \
  --port "${BUFFER_PORT}" \
  --idle-timeout-sec "${LOOPBACK_IDLE_TIMEOUT}" \
  --ack-mode always \
  --chunk-output-dir "${OUTPUT_DIR}/chunks" \
  --reassemble-output-dir "${OUTPUT_DIR}/reassembled" \
  --output "${OUTPUT_DIR}/loopback_receiver.json" \
  >"${LOOPBACK_LOG}" 2>&1 &
LOOPBACK_PID=$!

echo "[phase2/1] run prefill sender"
LLAMA_PREFILL_TB_ENABLE=1 \
"${PREFILL_BIN}" \
  -m "${MODEL_PATH}" \
  -f "${PROMPT_FILE}" \
  -c "${CTX_SIZE}" \
  -n "${N_PREDICT}" \
  -ps --prefill-overlap \
  --prefill-min-stream-batch-tokens "${PREFILL_MIN_STREAM_BATCH_TOKENS}" \
  --prefill-decode-mode split_thunderbolt \
  --prefill-decode-transport-required \
  --prefill-transport-mode progressive \
  --prefill-execution-mode coupled \
  --kv-transport "${KV_TRANSPORT}" \
  --kv-transport-fallback \
  --kv-host "${BUFFER_HOST}" \
  --kv-port "${BUFFER_PORT}" \
  --kv-streams 1 \
  --kv-stream-chunk-bytes "${KV_STREAM_CHUNK_BYTES}" \
  --kv-max-inflight-bytes "${KV_MAX_INFLIGHT_BYTES}" \
  >"${PREFILL_LOG}" 2>&1

wait "${LOOPBACK_PID}"
LOOPBACK_PID=""

ARTIFACT_PATH="$(find "${OUTPUT_DIR}/reassembled" -type f -name 'kv_artifact.bin' | sort | tail -n 1)"
if [[ -z "${ARTIFACT_PATH}" || ! -f "${ARTIFACT_PATH}" ]]; then
    echo "no reassembled artifact found under ${OUTPUT_DIR}/reassembled; see ${LOOPBACK_LOG}" >&2
    exit 1
fi

ARTIFACT_BYTES="$(wc -c < "${ARTIFACT_PATH}" | tr -d '[:space:]')"

cat > "${META_PATH}" <<EOF
{
  "artifact_path": "${ARTIFACT_PATH}",
  "artifact_bytes": ${ARTIFACT_BYTES},
  "kv_transport": "${KV_TRANSPORT}",
  "buffer_host": "${BUFFER_HOST}",
  "buffer_port": ${BUFFER_PORT}
}
EOF

echo "[phase2/1] artifact ready: ${ARTIFACT_PATH}"
