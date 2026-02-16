#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build_codex"
RUN_CTEST=1
RUN_QUEUE_TESTS=1

usage() {
  cat <<'EOF'
Usage: scripts/run_kv_bridge_matrix.sh [options]

Options:
  --build-dir DIR         Build directory (default: build_codex)
  --no-ctest              Skip ctest kv-bridge label run
  --no-queue-tests        Skip phase queue self-tests
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"; shift 2 ;;
    --no-ctest)
      RUN_CTEST=0; shift ;;
    --no-queue-tests)
      RUN_QUEUE_TESTS=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

PARSER_BIN="${BUILD_DIR}/bin/test-kv-bridge-parser"
CLI_BIN="${BUILD_DIR}/bin/test-kv-bridge-cli"

if [[ ! -x "${PARSER_BIN}" ]]; then
  echo "missing ${PARSER_BIN}; build first:" >&2
  echo "  cmake --build ${BUILD_DIR} --target test-kv-bridge-parser test-kv-bridge-cli -j8" >&2
  exit 1
fi

echo "[kv-bridge] running parser unit tests"
"${PARSER_BIN}"

if [[ -x "${CLI_BIN}" ]]; then
  echo "[kv-bridge] running CLI parsing tests"
  "${CLI_BIN}"
fi

if [[ "${RUN_CTEST}" == "1" ]]; then
  echo "[kv-bridge] running ctest label=kv-bridge"
  ctest --test-dir "${BUILD_DIR}" -L kv-bridge --output-on-failure
fi

if [[ "${RUN_QUEUE_TESTS}" == "1" ]]; then
  echo "[kv-bridge] running phase queue self-tests"
  bash scripts/test_prefill_decode_job_queue.sh
  bash scripts/test_prefill_decode_phase2_pipeline.sh
fi

echo "[kv-bridge] matrix completed"

