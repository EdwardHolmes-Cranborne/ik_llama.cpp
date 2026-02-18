#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HOST="0.0.0.0"
PORT=50052
DEVICE="METAL0"
RESTART_DELAY_SEC=2
LOG_PATH="/tmp/ik_rpc_server_keepalive.log"

usage() {
    cat <<'USAGE'
Run rpc-server in a restart loop to keep the endpoint available.

Optional:
  --host HOST               bind host (default: 0.0.0.0)
  --port N                  bind port (default: 50052)
  --device NAME             backend device id (default: METAL0)
  --restart-delay-sec N     delay before restart after exit (default: 2)
  --log PATH                log file path (default: /tmp/ik_rpc_server_keepalive.log)
  -h, --help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --restart-delay-sec) RESTART_DELAY_SEC="$2"; shift 2 ;;
        --log) LOG_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if ! [[ "${PORT}" =~ ^[0-9]+$ ]] || [[ "${PORT}" -lt 1 ]] || [[ "${PORT}" -gt 65535 ]]; then
    echo "invalid --port: ${PORT}" >&2
    exit 2
fi
if ! [[ "${RESTART_DELAY_SEC}" =~ ^[0-9]+$ ]]; then
    echo "invalid --restart-delay-sec: ${RESTART_DELAY_SEC}" >&2
    exit 2
fi

RPC_BIN=""
for candidate in \
    "${IK_REPO}/build_codex/bin/rpc-server" \
    "${IK_REPO}/build/bin/rpc-server"; do
    if [[ -x "${candidate}" ]]; then
        RPC_BIN="${candidate}"
        break
    fi
done
if [[ -z "${RPC_BIN}" ]]; then
    echo "rpc-server binary not found under build_codex/bin or build/bin" >&2
    exit 2
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] keepalive start host=${HOST} port=${PORT} device=${DEVICE}" | tee -a "${LOG_PATH}"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching rpc-server" | tee -a "${LOG_PATH}"
    set +e
    "${RPC_BIN}" --host "${HOST}" --port "${PORT}" --device "${DEVICE}" >>"${LOG_PATH}" 2>&1
    rc=$?
    set -e
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] rpc-server exited rc=${rc}" | tee -a "${LOG_PATH}"
    sleep "${RESTART_DELAY_SEC}"
done
