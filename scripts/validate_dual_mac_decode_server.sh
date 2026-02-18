#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH=""
RPC_ENDPOINTS=""
BIND_HOST="0.0.0.0"
HTTP_HOST="127.0.0.1"
HTTP_PORT=8080
KV_RECV_HOST="0.0.0.0"
KV_RECV_PORT=19001
CTX_SIZE=32768
DEFAULT_N_GPU_LAYERS=128
N_GPU_LAYERS="${DEFAULT_N_GPU_LAYERS}"
SPLIT_MODE="graph"
MAX_GPU=2
DEFAULT_TENSOR_SPLIT="0.70,0.30"
TENSOR_SPLIT="${DEFAULT_TENSOR_SPLIT}"
FLASH_ATTN="on"
KV_TRANSPORT="auto"
KV_TRANSPORT_FALLBACK=1
SLOT_SAVE_PATH="/tmp/ik_slots"
KV_RECV_OUTPUT_DIR="/tmp/ik_kv_handoff"
KV_RECV_MAX_CONNECTIONS=64
KV_RECV_IDLE_TIMEOUT=120
KV_RECV_STALE_FINALIZE_TIMEOUT=180
KV_RECV_SESSION_RETENTION=3600
KV_RECV_CLEANUP_INTERVAL=10
STARTUP_TIMEOUT_SEC=1800
POLL_INTERVAL_SEC=2
COMPLETION_N_PREDICT=16
COMPLETION_TIMEOUT_SEC=180
COMPLETION_PROMPT="Dual-mac validation completion smoke test."
KEEP_SERVER=0
# Include PID to avoid collisions when running multiple validators in parallel.
OUTPUT_DIR="/tmp/ik_dual_mac_decode_validate_$(date +%Y%m%d_%H%M%S)_$$"
DEFAULT_SAFE_NGL_CAP=192
SAFE_NGL_CAP="${DEFAULT_SAFE_NGL_CAP}"
ALLOW_HIGH_NGL=0
HARDWARE_PROFILE=""

EXTRA_ARGS=()

SERVER_PID=""
CLEANUP_SERVER=1

usage() {
    cat <<'USAGE'
Validate dual-mac decode coordinator startup and decode API readiness.

Required:
  --model PATH                GGUF model path

Optional:
  --rpc LIST                  comma-separated rpc endpoints (ip:port)
  --bind-host HOST            llama-server bind host (default: 0.0.0.0)
  --http-host HOST            host used for health/completion probes (default: 127.0.0.1)
  --http-port N               llama-server HTTP port (default: 8080)
  --kv-recv-host HOST         kv-receiver bind host (default: 0.0.0.0)
  --kv-recv-port N            kv-receiver port (default: 19001)
  --ctx-size N                context size (default: 32768)
  --ngl N                     number of layers to offload (default: 128)
  --split-mode MODE           split mode (default: graph)
  --max-gpu N                 max gpus for split mode (default: 2)
  --tensor-split CSV          tensor split fractions (default: 0.70,0.30)
  --hardware-profile NAME     apply safe split defaults for known hardware pairs
                              m3_ultra512_to_m3_max128  (Studio coordinator -> MacBook RPC)
                              m3_max128_to_m3_ultra512  (MacBook coordinator -> Studio RPC)
  --flash-attn on|off         flash attention mode (default: on)
  --kv-transport MODE         auto|tcp|rdma|mixed|disabled (default: auto)
  --kv-transport-fallback
  --no-kv-transport-fallback
  --slot-save-path PATH       slot directory (default: /tmp/ik_slots)
  --kv-recv-output-dir PATH   kv receiver output directory (default: /tmp/ik_kv_handoff)
  --startup-timeout-sec N     max wait for /health (default: 1800)
  --poll-interval-sec N       poll interval for /health (default: 2)
  --completion-prompt TEXT    completion smoke prompt
  --completion-n-predict N    completion smoke output tokens (default: 16)
  --completion-timeout-sec N  completion smoke HTTP timeout (default: 180)
  --output-dir PATH           artifact directory
  --safe-ngl-cap N            memory-safe ngl cap unless overridden (default: 192)
  --allow-high-ngl            allow --ngl above safe cap
  --keep-server               do not stop server at script exit
  --extra-arg ARG             append raw extra llama-server arg (repeatable)
  -h, --help
USAGE
}

cleanup() {
    set +e
    if [[ "${CLEANUP_SERVER}" == "1" && -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --rpc) RPC_ENDPOINTS="$2"; shift 2 ;;
        --bind-host) BIND_HOST="$2"; shift 2 ;;
        --http-host) HTTP_HOST="$2"; shift 2 ;;
        --http-port) HTTP_PORT="$2"; shift 2 ;;
        --kv-recv-host) KV_RECV_HOST="$2"; shift 2 ;;
        --kv-recv-port) KV_RECV_PORT="$2"; shift 2 ;;
        --ctx-size) CTX_SIZE="$2"; shift 2 ;;
        --ngl) N_GPU_LAYERS="$2"; shift 2 ;;
        --split-mode) SPLIT_MODE="$2"; shift 2 ;;
        --max-gpu) MAX_GPU="$2"; shift 2 ;;
        --tensor-split) TENSOR_SPLIT="$2"; shift 2 ;;
        --hardware-profile) HARDWARE_PROFILE="$2"; shift 2 ;;
        --flash-attn) FLASH_ATTN="$2"; shift 2 ;;
        --kv-transport) KV_TRANSPORT="$2"; shift 2 ;;
        --kv-transport-fallback) KV_TRANSPORT_FALLBACK=1; shift 1 ;;
        --no-kv-transport-fallback) KV_TRANSPORT_FALLBACK=0; shift 1 ;;
        --slot-save-path) SLOT_SAVE_PATH="$2"; shift 2 ;;
        --kv-recv-output-dir) KV_RECV_OUTPUT_DIR="$2"; shift 2 ;;
        --startup-timeout-sec) STARTUP_TIMEOUT_SEC="$2"; shift 2 ;;
        --poll-interval-sec) POLL_INTERVAL_SEC="$2"; shift 2 ;;
        --completion-prompt) COMPLETION_PROMPT="$2"; shift 2 ;;
        --completion-n-predict) COMPLETION_N_PREDICT="$2"; shift 2 ;;
        --completion-timeout-sec) COMPLETION_TIMEOUT_SEC="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --safe-ngl-cap) SAFE_NGL_CAP="$2"; shift 2 ;;
        --allow-high-ngl) ALLOW_HIGH_NGL=1; shift 1 ;;
        --keep-server) KEEP_SERVER=1; shift 1 ;;
        --extra-arg) EXTRA_ARGS+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "missing required arg: --model" >&2
    usage
    exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "model not found: ${MODEL_PATH}" >&2
    exit 2
fi

if [[ -n "${HARDWARE_PROFILE}" ]]; then
    profile_ngl=""
    profile_tensor_split=""
    profile_safe_ngl_cap=""
    case "${HARDWARE_PROFILE}" in
        m3_ultra512_to_m3_max128)
            profile_ngl=96
            profile_tensor_split="0.18,0.82"
            profile_safe_ngl_cap=128
            ;;
        m3_max128_to_m3_ultra512)
            profile_ngl=96
            profile_tensor_split="0.82,0.18"
            profile_safe_ngl_cap=128
            ;;
        *)
            echo "invalid --hardware-profile: ${HARDWARE_PROFILE}" >&2
            exit 2
            ;;
    esac

    # Keep explicit CLI overrides: only replace still-default values.
    if [[ "${N_GPU_LAYERS}" == "${DEFAULT_N_GPU_LAYERS}" ]]; then
        N_GPU_LAYERS="${profile_ngl}"
    fi
    if [[ "${TENSOR_SPLIT}" == "${DEFAULT_TENSOR_SPLIT}" ]]; then
        TENSOR_SPLIT="${profile_tensor_split}"
    fi
    if [[ "${SAFE_NGL_CAP}" == "${DEFAULT_SAFE_NGL_CAP}" ]]; then
        SAFE_NGL_CAP="${profile_safe_ngl_cap}"
    fi

    echo "hardware profile applied: ${HARDWARE_PROFILE} (ngl=${N_GPU_LAYERS}, tensor-split=${TENSOR_SPLIT}, safe-ngl-cap=${SAFE_NGL_CAP})"
fi

case "${KV_TRANSPORT}" in
    auto|tcp|rdma|mixed|disabled) ;;
    *)
        echo "invalid --kv-transport: ${KV_TRANSPORT}" >&2
        exit 2
        ;;
esac
if [[ "${FLASH_ATTN}" != "on" && "${FLASH_ATTN}" != "off" ]]; then
    echo "invalid --flash-attn: ${FLASH_ATTN} (expected on/off)" >&2
    exit 2
fi
for tuple in \
    "--http-port ${HTTP_PORT}" \
    "--kv-recv-port ${KV_RECV_PORT}" \
    "--ctx-size ${CTX_SIZE}" \
    "--ngl ${N_GPU_LAYERS}" \
    "--max-gpu ${MAX_GPU}" \
    "--startup-timeout-sec ${STARTUP_TIMEOUT_SEC}" \
    "--poll-interval-sec ${POLL_INTERVAL_SEC}" \
    "--completion-n-predict ${COMPLETION_N_PREDICT}" \
    "--completion-timeout-sec ${COMPLETION_TIMEOUT_SEC}" \
    "--safe-ngl-cap ${SAFE_NGL_CAP}"; do
    flag="${tuple%% *}"
    value="${tuple##* }"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
        echo "invalid ${flag}: ${value} (must be >=1)" >&2
        exit 2
    fi
done
if [[ "${ALLOW_HIGH_NGL}" != "1" ]] && [[ "${N_GPU_LAYERS}" -gt "${SAFE_NGL_CAP}" ]]; then
    echo "refusing --ngl=${N_GPU_LAYERS}: exceeds safe cap ${SAFE_NGL_CAP}. Use --allow-high-ngl to override." >&2
    exit 2
fi

command -v curl >/dev/null
command -v python3 >/dev/null

SERVER_BIN=""
for candidate in \
    "${IK_REPO}/build_codex/bin/llama-server" \
    "${IK_REPO}/build/bin/llama-server"; do
    if [[ -x "${candidate}" ]]; then
        SERVER_BIN="${candidate}"
        break
    fi
done
if [[ -z "${SERVER_BIN}" ]]; then
    echo "llama-server binary not found under build_codex/bin or build/bin" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${SLOT_SAVE_PATH}" "${KV_RECV_OUTPUT_DIR}"
SERVER_LOG="${OUTPUT_DIR}/server.log"
CMD_FILE="${OUTPUT_DIR}/server_command.txt"
HEALTH_JSON="${OUTPUT_DIR}/health.json"
HEALTH_LAST_JSON="${OUTPUT_DIR}/health_last.json"
KV_STATUS_JSON="${OUTPUT_DIR}/kv_receiver_status.json"
COMPLETION_REQ_JSON="${OUTPUT_DIR}/completion_request.json"
COMPLETION_RESP_JSON="${OUTPUT_DIR}/completion_response.json"
SUMMARY_JSON="${OUTPUT_DIR}/summary.json"

cmd=(
    "${SERVER_BIN}"
    -m "${MODEL_PATH}"
    --host "${BIND_HOST}" --port "${HTTP_PORT}"
    -c "${CTX_SIZE}"
    --flash-attn "${FLASH_ATTN}"
    -ngl "${N_GPU_LAYERS}"
    --split-mode "${SPLIT_MODE}"
    --max-gpu "${MAX_GPU}"
    --tensor-split "${TENSOR_SPLIT}"
    --slot-save-path "${SLOT_SAVE_PATH}"
    --kv-recv-enable
    --kv-transport "${KV_TRANSPORT}"
    --kv-recv-host "${KV_RECV_HOST}"
    --kv-recv-port "${KV_RECV_PORT}"
    --kv-recv-slot 0
    --kv-recv-output-dir "${KV_RECV_OUTPUT_DIR}"
    --kv-recv-max-connections "${KV_RECV_MAX_CONNECTIONS}"
    --kv-recv-idle-timeout "${KV_RECV_IDLE_TIMEOUT}"
    --kv-recv-stale-finalize-timeout "${KV_RECV_STALE_FINALIZE_TIMEOUT}"
    --kv-recv-session-retention "${KV_RECV_SESSION_RETENTION}"
    --kv-recv-cleanup-interval "${KV_RECV_CLEANUP_INTERVAL}"
)
if [[ -n "${RPC_ENDPOINTS}" ]]; then
    cmd+=(--rpc "${RPC_ENDPOINTS}")
fi
if [[ "${KV_TRANSPORT_FALLBACK}" == "1" ]]; then
    cmd+=(--kv-transport-fallback)
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
fi

printf '# %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "${CMD_FILE}"
printf '%q ' "${cmd[@]}" >> "${CMD_FILE}"
printf '\n' >> "${CMD_FILE}"

echo "[1/4] starting llama-server"
"${cmd[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "pid=${SERVER_PID}"
echo "log=${SERVER_LOG}"

echo "[2/4] waiting for /health (timeout=${STARTUP_TIMEOUT_SEC}s)"
start_epoch=$(date +%s)
deadline=$((start_epoch + STARTUP_TIMEOUT_SEC))
next_progress=$((start_epoch + 30))
ready=0
while true; do
    now=$(date +%s)
    health_code="$(curl -sS --max-time 2 -o "${HEALTH_LAST_JSON}.tmp" -w '%{http_code}' "http://${HTTP_HOST}:${HTTP_PORT}/health" || true)"
    if [[ "${health_code}" == "200" ]]; then
        mv "${HEALTH_LAST_JSON}.tmp" "${HEALTH_JSON}"
        ready=1
        break
    fi

    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        rc=0
        wait "${SERVER_PID}" || rc=$?
        echo "llama-server exited before /health became ready (exit=${rc})" >&2
        echo "--- server log tail ---" >&2
        tail -n 120 "${SERVER_LOG}" >&2 || true
        exit 1
    fi

    if [[ "${health_code}" == "500" ]] && grep -qi "model failed to load" "${HEALTH_LAST_JSON}.tmp"; then
        mv "${HEALTH_LAST_JSON}.tmp" "${HEALTH_LAST_JSON}"
        echo "health endpoint reported model load failure" >&2
        echo "--- health body ---" >&2
        cat "${HEALTH_LAST_JSON}" >&2 || true
        echo >&2
        echo "--- server log tail ---" >&2
        tail -n 120 "${SERVER_LOG}" >&2 || true
        exit 1
    fi

    if (( now >= deadline )); then
        echo "timed out waiting for /health after ${STARTUP_TIMEOUT_SEC}s" >&2
        echo "--- listener check ---" >&2
        lsof -nP -iTCP:"${HTTP_PORT}" -sTCP:LISTEN >&2 || true
        if [[ -s "${HEALTH_LAST_JSON}.tmp" ]]; then
            mv "${HEALTH_LAST_JSON}.tmp" "${HEALTH_LAST_JSON}"
            echo "--- last /health body ---" >&2
            cat "${HEALTH_LAST_JSON}" >&2 || true
            echo >&2
        fi
        echo "--- server log tail ---" >&2
        tail -n 120 "${SERVER_LOG}" >&2 || true
        exit 1
    fi

    if (( now >= next_progress )); then
        elapsed=$((now - start_epoch))
        if [[ "${health_code}" != "000" && "${health_code}" != "0" ]]; then
            health_preview="$(tr '\n' ' ' < "${HEALTH_LAST_JSON}.tmp" | cut -c1-180)"
            echo "[wait] /health code=${health_code} elapsed=${elapsed}s body='${health_preview}'"
        else
            echo "[wait] /health not reachable yet (elapsed=${elapsed}s)"
        fi
        next_progress=$((now + 30))
    fi

    sleep "${POLL_INTERVAL_SEC}"
done

if [[ "${ready}" != "1" ]]; then
    echo "internal error: readiness loop exited without ready state" >&2
    exit 1
fi

python3 - "${COMPLETION_PROMPT}" "${COMPLETION_N_PREDICT}" "${COMPLETION_REQ_JSON}" <<'PY'
import json
import sys
from pathlib import Path

prompt = sys.argv[1]
n_predict = int(sys.argv[2])
out = Path(sys.argv[3])
out.write_text(json.dumps({"prompt": prompt, "n_predict": n_predict}, indent=2) + "\n", encoding="utf-8")
PY

echo "[3/4] probing kv-receiver status"
curl -sf --max-time 5 "http://${HTTP_HOST}:${HTTP_PORT}/kv-receiver/status" > "${KV_STATUS_JSON}"

echo "[4/4] completion smoke request (timeout=${COMPLETION_TIMEOUT_SEC}s)"
curl -sf --max-time "${COMPLETION_TIMEOUT_SEC}" \
    -H "Content-Type: application/json" \
    -d @"${COMPLETION_REQ_JSON}" \
    "http://${HTTP_HOST}:${HTTP_PORT}/completion" \
    > "${COMPLETION_RESP_JSON}"

python3 - "${HEALTH_JSON}" "${KV_STATUS_JSON}" "${COMPLETION_RESP_JSON}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

health_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])
resp_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

health = json.loads(health_path.read_text(encoding="utf-8"))
status = json.loads(status_path.read_text(encoding="utf-8"))
resp = json.loads(resp_path.read_text(encoding="utf-8"))

if not isinstance(status, dict):
    raise SystemExit("/kv-receiver/status did not return a JSON object")
if "counters" not in status:
    raise SystemExit("/kv-receiver/status missing counters")
if not isinstance(resp, dict):
    raise SystemExit("/completion did not return a JSON object")
if "content" not in resp and "choices" not in resp:
    raise SystemExit("/completion response missing content/choices")

summary = {
    "ok": True,
    "health_keys": sorted(health.keys()),
    "kv_counter_keys": sorted(status.get("counters", {}).keys()),
    "completion_keys": sorted(resp.keys()),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

if [[ "${KEEP_SERVER}" == "1" ]]; then
    CLEANUP_SERVER=0
    echo "server retained: pid=${SERVER_PID}"
fi

echo "validation complete"
echo "output_dir=${OUTPUT_DIR}"
