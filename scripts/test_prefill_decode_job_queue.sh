#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
QUEUE_SCRIPT="${REPO_ROOT}/scripts/prefill_decode_job_queue.py"

TMP_ROOT="$(mktemp -d /tmp/ik_queue_test_XXXXXX)"
trap 'rm -rf "${TMP_ROOT}"' EXIT

SPOOL_DIR="${TMP_ROOT}/spool"
FAKE_OK="${TMP_ROOT}/fake_ok.sh"

cat > "${FAKE_OK}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${IK_PDQ_KV_TRANSPORT:-}" != "tcp" ]]; then
  echo "unexpected IK_PDQ_KV_TRANSPORT=${IK_PDQ_KV_TRANSPORT:-}"
  exit 9
fi
echo "[3/7] run prefill sender into disk buffer"
echo "[4/7] replay buffered artifact to ik kv-receiver"
echo "[7/7] run decode smoke request on ik server"
echo "single-machine buffered e2e suite completed"
EOF
chmod +x "${FAKE_OK}"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" init --max-queued 8 >/dev/null

OK_JOB_JSON="${TMP_ROOT}/ok_submit.json"
FAIL_JOB_JSON="${TMP_ROOT}/fail_submit.json"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" submit \
  --mode external_command \
  --command "${FAKE_OK}" \
  --kv-transport tcp \
  --priority 5 > "${OK_JOB_JSON}"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" submit \
  --mode external_command \
  --command "bash -lc 'echo fail_job; exit 7'" \
  --priority 1 > "${FAIL_JOB_JSON}"

set +e
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" submit \
  --mode external_command \
  --command "/bin/echo --kv-streams 2" \
  >/dev/null 2>&1
GUARDRAIL_RC=$?
set -e
if [[ "${GUARDRAIL_RC}" -eq 0 ]]; then
  echo "expected guardrail submit rejection for --kv-streams 2"
  exit 1
fi

OK_JOB_ID="$(python3 - <<'PY' "${OK_JOB_JSON}"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["job_id"])
PY
)"

FAIL_JOB_ID="$(python3 - <<'PY' "${FAIL_JOB_JSON}"
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["job_id"])
PY
)"

python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" worker --once >/dev/null
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" worker --once >/dev/null

STATUS_JSON="${TMP_ROOT}/status.json"
python3 "${QUEUE_SCRIPT}" --spool-dir "${SPOOL_DIR}" status --json > "${STATUS_JSON}"

python3 - <<'PY' "${STATUS_JSON}" "${SPOOL_DIR}" "${OK_JOB_ID}" "${FAIL_JOB_ID}"
import json
import pathlib
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
spool = pathlib.Path(sys.argv[2])
ok_job_id = sys.argv[3]
fail_job_id = sys.argv[4]

counts = status.get("counts", {})
if int(counts.get("done", 0)) < 1:
    raise SystemExit("expected at least one done job")
if int(counts.get("failed", 0)) < 1:
    raise SystemExit("expected at least one failed job")

ok_job = json.load(open(spool / "jobs" / f"{ok_job_id}.json", encoding="utf-8"))
if ok_job.get("state") != "done":
    raise SystemExit(f"expected ok job state done, got {ok_job.get('state')}")

evt_path = spool / "events" / f"{ok_job_id}.jsonl"
events = [json.loads(line) for line in evt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
event_states = [evt.get("state") for evt in events]
for required_state in ("prefill_running", "artifact_ready", "handoff_running", "decode_running", "done"):
    if required_state not in event_states:
        raise SystemExit(f"missing expected event state: {required_state}")

fail_job = json.load(open(spool / "jobs" / f"{fail_job_id}.json", encoding="utf-8"))
if fail_job.get("state") != "failed":
    raise SystemExit(f"expected fail job state failed, got {fail_job.get('state')}")
PY

echo "queue self-test passed"
