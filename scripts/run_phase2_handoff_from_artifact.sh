#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPLAY_SCRIPT="${IK_REPO}/scripts/tbp_replay_to_kv_receiver.py"

ARTIFACT_PATH=""
DECODE_HOST=""
DECODE_PORT=19001
KV_CHUNK_BYTES=$((4 * 1024 * 1024))
ACK_REQUIRED=0
WAIT_ACK=0
DECODE_SMOKE=0
DECODE_HTTP_URL=""
DECODE_SMOKE_PROMPT="Buffered handoff decode smoke test."
DECODE_SMOKE_N_PREDICT=16
OUTPUT_DIR="/tmp/ik_phase2_handoff_$(date +%Y%m%d_%H%M%S)"

usage() {
    cat <<'EOF'
Run Phase-2 handoff stage: replay artifact to decode receiver, optional decode smoke.

Required:
  --artifact PATH
  --decode-host HOST

Optional:
  --decode-port N
  --kv-chunk-bytes N
  --ack-required
  --wait-ack
  --decode-smoke
  --decode-http-url URL        default: http://<decode-host>:8080
  --decode-smoke-prompt TEXT
  --decode-smoke-n-predict N
  --output-dir PATH
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --artifact) ARTIFACT_PATH="$2"; shift 2 ;;
        --decode-host) DECODE_HOST="$2"; shift 2 ;;
        --decode-port) DECODE_PORT="$2"; shift 2 ;;
        --kv-chunk-bytes) KV_CHUNK_BYTES="$2"; shift 2 ;;
        --ack-required) ACK_REQUIRED=1; shift 1 ;;
        --wait-ack) WAIT_ACK=1; shift 1 ;;
        --decode-smoke) DECODE_SMOKE=1; shift 1 ;;
        --decode-http-url) DECODE_HTTP_URL="$2"; shift 2 ;;
        --decode-smoke-prompt) DECODE_SMOKE_PROMPT="$2"; shift 2 ;;
        --decode-smoke-n-predict) DECODE_SMOKE_N_PREDICT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${ARTIFACT_PATH}" || -z "${DECODE_HOST}" ]]; then
    echo "missing required args: --artifact --decode-host" >&2
    usage
    exit 2
fi
if [[ ! -f "${ARTIFACT_PATH}" ]]; then
    echo "artifact not found: ${ARTIFACT_PATH}" >&2
    exit 2
fi
if [[ ! -f "${REPLAY_SCRIPT}" ]]; then
    echo "replay script not found: ${REPLAY_SCRIPT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"
REPLAY_JSON="${OUTPUT_DIR}/replay_result.json"
REPLAY_LOG="${OUTPUT_DIR}/replay.log"
DECODE_RESPONSE_JSON="${OUTPUT_DIR}/decode_response.json"
META_PATH="${OUTPUT_DIR}/handoff_meta.json"

if [[ -z "${DECODE_HTTP_URL}" ]]; then
    DECODE_HTTP_URL="http://${DECODE_HOST}:8080"
fi

echo "[phase2/2] replay artifact"
replay_args=(
  --artifact "${ARTIFACT_PATH}"
  --host "${DECODE_HOST}"
  --port "${DECODE_PORT}"
  --chunk-bytes "${KV_CHUNK_BYTES}"
  --output "${REPLAY_JSON}"
)
if [[ "${ACK_REQUIRED}" == "1" ]]; then
    replay_args+=(--ack-required)
fi
if [[ "${WAIT_ACK}" == "1" ]]; then
    replay_args+=(--wait-ack)
fi
python3 "${REPLAY_SCRIPT}" "${replay_args[@]}" >"${REPLAY_LOG}" 2>&1

if [[ "${DECODE_SMOKE}" == "1" ]]; then
    echo "[phase2/2] decode smoke request"
    python3 - "${DECODE_SMOKE_PROMPT}" "${DECODE_SMOKE_N_PREDICT}" "${OUTPUT_DIR}/decode_request.json" <<'PY'
import json
import sys
from pathlib import Path

prompt = sys.argv[1]
n_predict = int(sys.argv[2])
out_path = Path(sys.argv[3])
out_path.write_text(
    json.dumps({"prompt": prompt, "n_predict": n_predict}, indent=2) + "\n",
    encoding="utf-8",
)
PY
    curl -sf \
      -H "Content-Type: application/json" \
      -d @"${OUTPUT_DIR}/decode_request.json" \
      "${DECODE_HTTP_URL}/completion" \
      > "${DECODE_RESPONSE_JSON}"
fi

cat > "${META_PATH}" <<EOF
{
  "artifact_path": "${ARTIFACT_PATH}",
  "decode_host": "${DECODE_HOST}",
  "decode_port": ${DECODE_PORT},
  "decode_http_url": "${DECODE_HTTP_URL}",
  "decode_smoke": ${DECODE_SMOKE},
  "replay_output": "${REPLAY_JSON}",
  "replay_log": "${REPLAY_LOG}"
}
EOF

echo "[phase2/2] handoff complete"
