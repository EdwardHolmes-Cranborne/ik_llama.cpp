#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IK_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

HOST="0.0.0.0"
PORT=50052
DEVICE=""
RESTART_DELAY_SEC=2
LOG_PATH="/tmp/ik_rpc_server_keepalive.log"
RDMA=0
BIND_ADDR=""
SOCKET_SEND_BUF=""
SOCKET_RECV_BUF=""

usage() {
    cat <<'USAGE'
Run rpc-server in a restart loop to keep the endpoint available.

Optional:
  --host HOST               bind host (default: 0.0.0.0)
  --port N                  bind port (default: 50052)
  --device NAME             backend device id (default: rpc-server default backend)
  --restart-delay-sec N     delay before restart after exit (default: 2)
  --log PATH                log file path (default: /tmp/ik_rpc_server_keepalive.log)
  --rdma                    auto-detect Thunderbolt interfaces (macOS)
  --bind-addr ADDR          explicit bind address (overrides --host)
  --socket-send-buf N       SO_SNDBUF in bytes (0 = system default)
  --socket-recv-buf N       SO_RCVBUF in bytes (0 = system default)
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
        --rdma) RDMA=1; shift 1 ;;
        --bind-addr) BIND_ADDR="$2"; shift 2 ;;
        --socket-send-buf) SOCKET_SEND_BUF="$2"; shift 2 ;;
        --socket-recv-buf) SOCKET_RECV_BUF="$2"; shift 2 ;;
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

# Build the argument list.
RPC_ARGS=(--host "${HOST}" --port "${PORT}")
if [[ -n "${DEVICE}" ]]; then
    RPC_ARGS+=(--device "${DEVICE}")
fi
if [[ "${RDMA}" == "1" ]]; then
    RPC_ARGS+=(--rdma)
fi
if [[ -n "${BIND_ADDR}" ]]; then
    RPC_ARGS+=(--bind-addr "${BIND_ADDR}")
fi
if [[ -n "${SOCKET_SEND_BUF}" ]]; then
    RPC_ARGS+=(--socket-send-buf "${SOCKET_SEND_BUF}")
fi
if [[ -n "${SOCKET_RECV_BUF}" ]]; then
    RPC_ARGS+=(--socket-recv-buf "${SOCKET_RECV_BUF}")
fi

device_label="${DEVICE:-<default>}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] keepalive start host=${HOST} port=${PORT} device=${device_label} rdma=${RDMA}" | tee -a "${LOG_PATH}"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] launching rpc-server ${RPC_ARGS[*]}" | tee -a "${LOG_PATH}"
    set +e
    "${RPC_BIN}" "${RPC_ARGS[@]}" >>"${LOG_PATH}" 2>&1
    rc=$?
    set -e
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] rpc-server exited rc=${rc}" | tee -a "${LOG_PATH}"
    sleep "${RESTART_DELAY_SEC}"
done
