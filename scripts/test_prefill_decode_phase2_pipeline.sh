#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QUEUE_SCRIPT="${REPO_ROOT}/scripts/prefill_decode_job_queue.py"

TMP_ROOT="$(mktemp -d /tmp/ik_phase2_queue_test_XXXXXX)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

SPOOL_DIR="${TMP_ROOT}/spool"
MODEL_PATH="${TMP_ROOT}/model.gguf"
PROMPT_A="${TMP_ROOT}/prompt_a.txt"
PROMPT_B="${TMP_ROOT}/prompt_b.txt"
RTX_REPO="${TMP_ROOT}/rtx_repo"
FAKE_PREFILL="${TMP_ROOT}/fake_phase2_prefill.sh"
FAKE_HANDOFF="${TMP_ROOT}/fake_phase2_handoff.sh"

mkdir -p "${RTX_REPO}"
printf 'fake model\n' > "${MODEL_PATH}"
printf 'hello from prompt a\n' > "${PROMPT_A}"
printf 'hello from prompt b\n' > "${PROMPT_B}"

cat > "${FAKE_PREFILL}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR=""
PROMPT_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    *) shift 1 ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" || -z "${PROMPT_FILE}" ]]; then
  echo "missing output or prompt" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
ARTIFACT_PATH="${OUTPUT_DIR}/kv_artifact.bin"
printf 'artifact for %s\n' "${PROMPT_FILE}" > "${ARTIFACT_PATH}"
ARTIFACT_BYTES="$(wc -c < "${ARTIFACT_PATH}" | tr -d '[:space:]')"
cat > "${OUTPUT_DIR}/prefill_artifact_meta.json" <<JSON
{
  "artifact_path": "${ARTIFACT_PATH}",
  "artifact_bytes": ${ARTIFACT_BYTES}
}
JSON
echo "[phase2/1] artifact ready: ${ARTIFACT_PATH}"
EOF
chmod +x "${FAKE_PREFILL}"

cat > "${FAKE_HANDOFF}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_PATH=""
OUTPUT_DIR=""
DECODE_SMOKE=0
KV_TRANSPORT=""
HAS_TRANSPORT_FALLBACK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact) ARTIFACT_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --kv-transport) KV_TRANSPORT="$2"; shift 2 ;;
    --kv-transport-fallback) HAS_TRANSPORT_FALLBACK=1; shift 1 ;;
    --decode-smoke) DECODE_SMOKE=1; shift 1 ;;
    *) shift 1 ;;
  esac
done

if [[ -z "${ARTIFACT_PATH}" || -z "${OUTPUT_DIR}" ]]; then
  echo "missing artifact or output dir" >&2
  exit 2
fi
if [[ ! -f "${ARTIFACT_PATH}" ]]; then
  echo "artifact missing: ${ARTIFACT_PATH}" >&2
  exit 2
fi
if [[ "${KV_TRANSPORT}" != "rdma" ]]; then
  echo "expected --kv-transport rdma, got '${KV_TRANSPORT}'" >&2
  exit 2
fi
if [[ "${HAS_TRANSPORT_FALLBACK}" != "0" ]]; then
  echo "expected no --kv-transport-fallback flag in test path" >&2
  exit 2
fi

if [[ -n "${IK_PDQ_FAIL_HANDOFF_ONCE_FILE:-}" && ! -f "${IK_PDQ_FAIL_HANDOFF_ONCE_FILE}" ]]; then
  touch "${IK_PDQ_FAIL_HANDOFF_ONCE_FILE}"
  echo "intentional handoff failure" >&2
  exit 13
fi

mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/handoff_meta.json" <<JSON
{
  "artifact_path": "${ARTIFACT_PATH}",
  "job_id": "${IK_PDQ_JOB_ID:-}"
}
JSON

echo "[phase2/2] replay artifact"
if [[ "${DECODE_SMOKE}" == "1" ]]; then
  echo "[phase2/2] decode smoke request"
fi
echo "[phase2/2] handoff complete"
EOF
chmod +x "${FAKE_HANDOFF}"

export IK_PDQ_PHASE2_PREFILL_SCRIPT="${FAKE_PREFILL}"
export IK_PDQ_PHASE2_HANDOFF_SCRIPT="${FAKE_HANDOFF}"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" init --max-queued 8 >/dev/null

SUBMIT_A_JSON="${TMP_ROOT}/submit_a.json"
SUBMIT_B_JSON="${TMP_ROOT}/submit_b.json"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" submit \
  --mode phase2_split_pipeline \
  --model "${MODEL_PATH}" \
  --prompt-file "${PROMPT_A}" \
  --rtx-repo "${RTX_REPO}" \
  --decode-host 127.0.0.1 \
  --kv-transport rdma \
  --no-kv-transport-fallback \
  --decode-smoke \
  --priority 10 \
  --max-prefill-retries 0 \
  --max-handoff-retries 1 > "${SUBMIT_A_JSON}"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" submit \
  --mode phase2_split_pipeline \
  --model "${MODEL_PATH}" \
  --prompt-file "${PROMPT_B}" \
  --rtx-repo "${RTX_REPO}" \
  --decode-host 127.0.0.1 \
  --kv-transport rdma \
  --no-kv-transport-fallback \
  --decode-smoke \
  --priority 5 \
  --max-prefill-retries 0 \
  --max-handoff-retries 1 > "${SUBMIT_B_JSON}"

JOB_A="$(python3 - <<'PY' "${SUBMIT_A_JSON}"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["job_id"])
PY
)"
JOB_B="$(python3 - <<'PY' "${SUBMIT_B_JSON}"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["job_id"])
PY
)"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" prefill-worker --once >/dev/null
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" prefill-worker --once >/dev/null

STATUS_PREFILL_JSON="${TMP_ROOT}/status_prefill.json"
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" status --json > "${STATUS_PREFILL_JSON}"
python3 - <<'PY' "${STATUS_PREFILL_JSON}"
import json
import sys
status = json.load(open(sys.argv[1], encoding="utf-8"))
if int(status.get("artifact_ready_depth", 0)) < 2:
    raise SystemExit("expected at least two artifact_ready jobs after prefill stage")
PY

export IK_PDQ_FAIL_HANDOFF_ONCE_FILE="${TMP_ROOT}/fail_once.marker"
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" handoff-worker --once >/dev/null

JOB_A_JSON="${SPOOL_DIR}/jobs/${JOB_A}.json"
python3 - <<'PY' "${JOB_A_JSON}"
import json
import sys
job = json.load(open(sys.argv[1], encoding="utf-8"))
if job.get("state") != "artifact_ready":
    raise SystemExit(f"expected first handoff attempt to requeue artifact_ready, got {job.get('state')}")
if int(job.get("handoff_attempt", 0)) != 1:
    raise SystemExit(f"expected handoff_attempt=1 after first failure, got {job.get('handoff_attempt')}")
PY

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" handoff-worker --once >/dev/null
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" handoff-worker --once >/dev/null

STATUS_FINAL_JSON="${TMP_ROOT}/status_final.json"
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" status --json > "${STATUS_FINAL_JSON}"

python3 - <<'PY' "${STATUS_FINAL_JSON}" "${SPOOL_DIR}" "${JOB_A}" "${JOB_B}"
import json
import pathlib
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
spool = pathlib.Path(sys.argv[2])
job_ids = [sys.argv[3], sys.argv[4]]

counts = status.get("counts", {})
if int(counts.get("done", 0)) < 2:
    raise SystemExit("expected both jobs done")
if int(status.get("artifact_ready_depth", 0)) != 0:
    raise SystemExit("expected no pending artifact_ready jobs")

for job_id in job_ids:
    job = json.load(open(spool / "jobs" / f"{job_id}.json", encoding="utf-8"))
    if job.get("state") != "done":
        raise SystemExit(f"expected job {job_id} done, got {job.get('state')}")
    result = job.get("result", {})
    if "prefill" not in result:
        raise SystemExit(f"job {job_id} missing result.prefill")
    if "handoff" not in result:
        raise SystemExit(f"job {job_id} missing result.handoff")

job_a_events = (spool / "events" / f"{job_ids[0]}.jsonl").read_text(encoding="utf-8")
if "handoff failed rc=13; retrying" not in job_a_events:
    raise SystemExit("expected retry event text for first job")
for needle in ("prefill_running", "artifact_ready", "handoff_running", "decode_running", "done"):
    if needle not in job_a_events:
        raise SystemExit(f"missing expected state in events: {needle}")
PY

echo "phase2 queue self-test passed"
