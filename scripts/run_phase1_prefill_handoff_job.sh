#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH=""
PROMPT_FILE=""
RTX_REPO=""
DECODE_HOST=""
DECODE_PORT=19001
CTX_SIZE=32768
N_PREDICT=128
KV_STREAM_CHUNK_BYTES=$((4 * 1024 * 1024))
KV_MAX_INFLIGHT_BYTES=$((256 * 1024 * 1024))
PREFILL_MIN_STREAM_BATCH_TOKENS=-1
OUTPUT_DIR="/tmp/ik_phase1_prefill_handoff_$(date +%Y%m%d_%H%M%S)"
KV_TRANSPORT="${IK_PDQ_KV_TRANSPORT:-auto}"

usage() {
    cat <<'EOF'
Run one Phase-1 prefill->handoff job (queue-friendly wrapper).

Required:
  --model PATH                GGUF model path for prefill
  --prompt-file PATH          prompt file path
  --rtx-repo PATH             RTX prefill fork root
  --decode-host HOST          decode receiver host

Optional:
  --decode-port N             decode KV receiver port (default: 19001)
  --ctx-size N                context size (default: 32768)
  --n-predict N               prefill cli generation tokens (default: 128)
  --prefill-min-stream-batch-tokens N
                              prefill streaming threshold (-1 = runtime crossover)
  --kv-transport MODE         auto|tcp|rdma|mixed|disabled
                              default: IK_PDQ_KV_TRANSPORT env or auto
  --kv-stream-chunk-bytes N   default: 4194304
  --kv-max-inflight-bytes N   default: 268435456
  --output-dir PATH           output directory
  -h, --help                  show help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --rtx-repo) RTX_REPO="$2"; shift 2 ;;
        --decode-host) DECODE_HOST="$2"; shift 2 ;;
        --decode-port) DECODE_PORT="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --n-predict) N_PREDICT="$2"; shift 2 ;;
        --prefill-min-stream-batch-tokens) PREFILL_MIN_STREAM_BATCH_TOKENS="$2"; shift 2 ;;
        --kv-transport) KV_TRANSPORT="$2"; shift 2 ;;
        --kv-stream-chunk-bytes) KV_STREAM_CHUNK_BYTES="$2"; shift 2 ;;
        --kv-max-inflight-bytes) KV_MAX_INFLIGHT_BYTES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL_PATH}" || -z "${PROMPT_FILE}" || -z "${RTX_REPO}" || -z "${DECODE_HOST}" ]]; then
    echo "missing required args: --model --prompt-file --rtx-repo --decode-host" >&2
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

mkdir -p "${OUTPUT_DIR}"
LOG_PATH="${OUTPUT_DIR}/prefill_handoff.log"
CMD_PATH="${OUTPUT_DIR}/prefill_handoff_command.txt"
META_PATH="${OUTPUT_DIR}/prefill_handoff_meta.json"

cat > "${CMD_PATH}" <<EOF
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
  --kv-host "${DECODE_HOST}" \
  --kv-port "${DECODE_PORT}" \
  --kv-streams 1 \
  --kv-stream-chunk-bytes "${KV_STREAM_CHUNK_BYTES}" \
  --kv-max-inflight-bytes "${KV_MAX_INFLIGHT_BYTES}"
EOF

cat > "${META_PATH}" <<EOF
{
  "ik_pdq_job_id": "${IK_PDQ_JOB_ID:-}",
  "ik_pdq_job_mode": "${IK_PDQ_JOB_MODE:-}",
  "ik_pdq_kv_transport": "${IK_PDQ_KV_TRANSPORT:-}",
  "resolved_kv_transport": "${KV_TRANSPORT}",
  "decode_host": "${DECODE_HOST}",
  "decode_port": ${DECODE_PORT},
  "prefill_min_stream_batch_tokens": ${PREFILL_MIN_STREAM_BATCH_TOKENS}
}
EOF

echo "phase1 prefill handoff: starting"
echo "output_dir: ${OUTPUT_DIR}"
echo "transport: ${KV_TRANSPORT}"
echo "decode_target: ${DECODE_HOST}:${DECODE_PORT}"

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
  --kv-host "${DECODE_HOST}" \
  --kv-port "${DECODE_PORT}" \
  --kv-streams 1 \
  --kv-stream-chunk-bytes "${KV_STREAM_CHUNK_BYTES}" \
  --kv-max-inflight-bytes "${KV_MAX_INFLIGHT_BYTES}" \
  >"${LOG_PATH}" 2>&1

echo "phase1 prefill handoff: completed"
echo "log_path: ${LOG_PATH}"
