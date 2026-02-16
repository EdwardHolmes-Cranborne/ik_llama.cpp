#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build_codex"
ITERATIONS=10

usage() {
  cat <<'EOF'
Usage: scripts/bench_kv_bridge.sh [options]

Options:
  --build-dir DIR         Build directory (default: build_codex)
  --iterations N          Number of runs (default: 10)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"; shift 2 ;;
    --iterations)
      ITERATIONS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

TEST_BIN="${BUILD_DIR}/bin/test-kv-bridge-parser"
if [[ ! -x "${TEST_BIN}" ]]; then
  echo "missing ${TEST_BIN}; build first:" >&2
  echo "  cmake --build ${BUILD_DIR} --target test-kv-bridge-parser -j8" >&2
  exit 1
fi

tmp_results="$(mktemp /tmp/kv_bridge_bench.XXXXXX)"
trap 'rm -f "${tmp_results}"' EXIT

echo "[kv-bridge-bench] binary=${TEST_BIN} iterations=${ITERATIONS}"

for i in $(seq 1 "${ITERATIONS}"); do
  t0=$(python3 - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
)
  "${TEST_BIN}" >/dev/null
  t1=$(python3 - <<'PY'
import time
print(f"{time.perf_counter():.9f}")
PY
)
  dt=$(python3 - <<PY
t0=${t0}
t1=${t1}
print(f"{(t1-t0)*1000.0:.3f}")
PY
)
  echo "${dt}" | tee -a "${tmp_results}"
done

python3 - <<PY
from statistics import mean, median
vals=[float(x.strip()) for x in open("${tmp_results}") if x.strip()]
vals_sorted=sorted(vals)
p95=vals_sorted[max(0, int(len(vals_sorted)*0.95)-1)]
print(f"[kv-bridge-bench] runs={len(vals)} mean_ms={mean(vals):.3f} median_ms={median(vals):.3f} p95_ms={p95:.3f} min_ms={min(vals):.3f} max_ms={max(vals):.3f}")
PY

