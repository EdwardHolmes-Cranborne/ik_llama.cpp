#!/usr/bin/env python3
"""
Phase-1 prefill->handoff->decode single-flight queue.

External tooling only; does not modify llama/ik_llama engine paths.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


JOB_VERSION = 1
DEFAULT_SPOOL = "/tmp/ik_prefill_decode_queue"
DEFAULT_MAX_QUEUED = 256
DEFAULT_MAX_SPOOL_BYTES = 0
DEFAULT_POLL_SEC = 2.0
STATE_TERMINAL = {"done", "failed", "canceled"}
STATE_QUEUEABLE = {"queued"}


def now_us() -> int:
    return int(time.time() * 1_000_000)


def ensure_dirs(spool: Path) -> None:
    (spool / "jobs").mkdir(parents=True, exist_ok=True)
    (spool / "events").mkdir(parents=True, exist_ok=True)
    (spool / "runs").mkdir(parents=True, exist_ok=True)


def config_path(spool: Path) -> Path:
    return spool / "config.json"


def load_config(spool: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "version": 1,
        "max_queued_jobs": DEFAULT_MAX_QUEUED,
        "max_spool_bytes": DEFAULT_MAX_SPOOL_BYTES,
        "max_retries_default": 0,
        "poll_seconds": DEFAULT_POLL_SEC,
    }
    p = config_path(spool)
    if not p.exists():
        return defaults
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return defaults
    if not isinstance(loaded, dict):
        return defaults
    out = dict(defaults)
    out.update(loaded)
    return out


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_event(spool: Path, job_id: str, state: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
    evt = {
        "ts_unix_us": now_us(),
        "job_id": job_id,
        "state": state,
        "message": message,
    }
    if extra:
        evt["extra"] = extra
    p = spool / "events" / f"{job_id}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(evt, sort_keys=True) + "\n")


def job_path(spool: Path, job_id: str) -> Path:
    return spool / "jobs" / f"{job_id}.json"


def load_job(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(path: Path, job: Dict[str, Any]) -> None:
    job["updated_at_unix_us"] = now_us()
    atomic_write_json(path, job)


def list_jobs(spool: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted((spool / "jobs").glob("*.json")):
        try:
            job = load_job(p)
            job["_path"] = str(p)
            out.append(job)
        except Exception:
            continue
    out.sort(
        key=lambda j: (
            -int(j.get("priority", 0)),
            int(j.get("created_at_unix_us", 0)),
            str(j.get("job_id", "")),
        )
    )
    return out


def runnable_jobs(spool: Path) -> List[Dict[str, Any]]:
    return [j for j in list_jobs(spool) if j.get("state") in STATE_QUEUEABLE]


def spool_size_bytes(spool: Path) -> int:
    total = 0
    for root, _, files in os.walk(spool):
        for name in files:
            p = Path(root) / name
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def ensure_queue_capacity(spool: Path, cfg: Dict[str, Any]) -> None:
    max_queued = int(cfg.get("max_queued_jobs", DEFAULT_MAX_QUEUED))
    if max_queued > 0 and len(runnable_jobs(spool)) >= max_queued:
        raise SystemExit(f"queue full: queued jobs >= max_queued_jobs ({max_queued})")

    max_spool = int(cfg.get("max_spool_bytes", DEFAULT_MAX_SPOOL_BYTES))
    if max_spool > 0:
        used = spool_size_bytes(spool)
        if used >= max_spool:
            raise SystemExit(f"spool full: bytes={used} >= max_spool_bytes={max_spool}")


def require_file(path: str, name: str) -> None:
    if not Path(path).is_file():
        raise SystemExit(f"{name} not found: {path}")


def run_job_command(
    repo_root: Path,
    job: Dict[str, Any],
    run_dir: Path,
    log_path: Path,
    on_stage: Optional[Callable[[str, str], None]] = None,
) -> int:
    req = job["request"]
    cmd: List[str] = [
        str(repo_root / "scripts" / "run_single_machine_buffered_e2e.sh"),
        "--model", req["model"],
        "--prompt-file", req["prompt_file"],
        "--output-dir", str(run_dir),
        "--ctx-size", str(req["ctx_size"]),
        "--prefill-n-predict", str(req["prefill_n_predict"]),
        "--prefill-min-stream-batch-tokens", str(req["prefill_min_stream_batch_tokens"]),
        "--decode-n-predict", str(req["decode_n_predict"]),
        "--kv-chunk-bytes", str(req["kv_chunk_bytes"]),
        "--kv-max-inflight", str(req["kv_max_inflight"]),
        "--loopback-idle-timeout", str(req["loopback_idle_timeout"]),
    ]
    rtx_repo = req.get("rtx_repo", "")
    if rtx_repo:
        cmd.extend(["--rtx-repo", rtx_repo])
    if bool(req.get("no_replay_ack", False)):
        cmd.append("--no-replay-ack")
    for extra in req.get("extra_args", []):
        cmd.append(str(extra))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "queue_command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")

    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if "[3/7] run prefill sender" in line:
                if on_stage:
                    on_stage("prefill_running", "prefill sender started")
            elif "[4/7] replay buffered artifact" in line:
                if on_stage:
                    on_stage("artifact_ready", "artifact reassembled and ready")
                    on_stage("handoff_running", "artifact replay/handoff started")
            elif "[7/7] run decode smoke request" in line:
                if on_stage:
                    on_stage("decode_running", "decode smoke request started")
        return proc.wait()


def update_state(spool: Path, job: Dict[str, Any], state: str, message: str, patch: Optional[Dict[str, Any]] = None) -> None:
    job["state"] = state
    if patch:
        job.update(patch)
    p = job_path(spool, job["job_id"])
    save_job(p, job)
    append_event(spool, job["job_id"], state, message)


def pick_next_job(spool: Path) -> Optional[Dict[str, Any]]:
    jobs = runnable_jobs(spool)
    return jobs[0] if jobs else None


def acquire_worker_lock(spool: Path) -> int:
    lock_path = spool / "worker.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SystemExit("worker lock is already held by another process")
    return fd


def cmd_init(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    cfg = load_config(spool)
    if args.max_queued is not None:
        cfg["max_queued_jobs"] = int(args.max_queued)
    if args.max_spool_bytes is not None:
        cfg["max_spool_bytes"] = int(args.max_spool_bytes)
    if args.max_retries_default is not None:
        cfg["max_retries_default"] = int(args.max_retries_default)
    if args.poll_seconds is not None:
        cfg["poll_seconds"] = float(args.poll_seconds)
    atomic_write_json(config_path(spool), cfg)
    print(json.dumps({"ok": True, "spool_dir": str(spool), "config": cfg}, indent=2))
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    cfg = load_config(spool)
    ensure_queue_capacity(spool, cfg)

    require_file(args.model, "model")
    require_file(args.prompt_file, "prompt file")
    if args.rtx_repo and not Path(args.rtx_repo).is_dir():
        raise SystemExit(f"rtx repo not found: {args.rtx_repo}")

    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    retries = int(args.max_retries if args.max_retries is not None else cfg.get("max_retries_default", 0))
    job: Dict[str, Any] = {
        "version": JOB_VERSION,
        "job_id": job_id,
        "state": "queued",
        "priority": int(args.priority),
        "attempt": 0,
        "max_retries": max(0, retries),
        "created_at_unix_us": now_us(),
        "updated_at_unix_us": now_us(),
        "request": {
            "model": str(Path(args.model).resolve()),
            "prompt_file": str(Path(args.prompt_file).resolve()),
            "rtx_repo": str(Path(args.rtx_repo).resolve()) if args.rtx_repo else "",
            "ctx_size": int(args.ctx_size),
            "prefill_n_predict": int(args.prefill_n_predict),
            "prefill_min_stream_batch_tokens": int(args.prefill_min_stream_batch_tokens),
            "decode_n_predict": int(args.decode_n_predict),
            "kv_chunk_bytes": int(args.kv_chunk_bytes),
            "kv_max_inflight": int(args.kv_max_inflight),
            "loopback_idle_timeout": int(args.loopback_idle_timeout),
            "no_replay_ack": bool(args.no_replay_ack),
            "extra_args": list(args.extra_arg or []),
        },
        "result": {},
    }
    p = job_path(spool, job_id)
    save_job(p, job)
    append_event(spool, job_id, "queued", "job submitted")
    print(json.dumps({"ok": True, "job_id": job_id, "path": str(p)}, indent=2))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    jobs = list_jobs(spool)
    if args.state:
        wanted = set(args.state.split(","))
        jobs = [j for j in jobs if j.get("state") in wanted]
    if args.json:
        print(json.dumps(jobs, indent=2))
        return 0
    if not jobs:
        print("no jobs")
        return 0
    for j in jobs:
        print(
            f"{j.get('job_id')} state={j.get('state')} prio={j.get('priority')} "
            f"attempt={j.get('attempt')}/{j.get('max_retries')} updated_us={j.get('updated_at_unix_us')}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    p = job_path(spool, args.job_id)
    if not p.exists():
        raise SystemExit(f"job not found: {args.job_id}")
    job = load_job(p)
    print(json.dumps(job, indent=2))
    events_p = spool / "events" / f"{args.job_id}.jsonl"
    if events_p.exists():
        print("\n# events")
        print(events_p.read_text(encoding="utf-8").rstrip())
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    p = job_path(spool, args.job_id)
    if not p.exists():
        raise SystemExit(f"job not found: {args.job_id}")
    job = load_job(p)
    state = str(job.get("state", ""))
    if state in STATE_TERMINAL:
        print(json.dumps({"ok": True, "job_id": args.job_id, "state": state}, indent=2))
        return 0
    if state != "queued":
        raise SystemExit(f"job is not cancelable in state={state} (only queued)")
    update_state(spool, job, "canceled", "job canceled by user")
    print(json.dumps({"ok": True, "job_id": args.job_id, "state": "canceled"}, indent=2))
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    cfg = load_config(spool)
    poll_sec = float(args.poll_seconds if args.poll_seconds is not None else cfg.get("poll_seconds", DEFAULT_POLL_SEC))
    lock_fd = acquire_worker_lock(spool)
    repo_root = Path(__file__).resolve().parents[1]

    try:
        while True:
            job = pick_next_job(spool)
            if job is None:
                if args.once:
                    print(json.dumps({"ok": True, "idle": True}, indent=2))
                    return 0
                time.sleep(max(0.05, poll_sec))
                continue

            job["attempt"] = int(job.get("attempt", 0)) + 1
            update_state(spool, job, "prefill_running", "worker picked job")

            run_dir = spool / "runs" / job["job_id"]
            log_path = run_dir / "worker_stream.log"
            stage_state = {"last": str(job.get("state", ""))}

            def on_stage(state: str, message: str) -> None:
                if state == stage_state["last"]:
                    return
                stage_state["last"] = state
                update_state(spool, job, state, message)

            rc = run_job_command(repo_root, job, run_dir, log_path, on_stage=on_stage)

            if rc == 0:
                update_state(
                    spool,
                    job,
                    "done",
                    "job completed successfully",
                    patch={
                        "result": {
                            "return_code": rc,
                            "run_dir": str(run_dir),
                            "worker_stream_log": str(log_path),
                        }
                    },
                )
            else:
                max_retries = int(job.get("max_retries", 0))
                attempt = int(job.get("attempt", 1))
                if attempt <= max_retries:
                    update_state(
                        spool,
                        job,
                        "queued",
                        f"job failed rc={rc}; retrying",
                        patch={"result": {"return_code": rc, "run_dir": str(run_dir)}},
                    )
                else:
                    update_state(
                        spool,
                        job,
                        "failed",
                        f"job failed rc={rc}; retries exhausted",
                        patch={"result": {"return_code": rc, "run_dir": str(run_dir), "worker_stream_log": str(log_path)}},
                    )

            if args.once:
                return 0
    finally:
        os.close(lock_fd)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase-1 prefill/decode queue (single-flight)")
    ap.add_argument("--spool-dir", default=DEFAULT_SPOOL, help=f"queue spool directory (default: {DEFAULT_SPOOL})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="initialize queue spool/config")
    p_init.add_argument("--max-queued", type=int, default=None)
    p_init.add_argument("--max-spool-bytes", type=int, default=None)
    p_init.add_argument("--max-retries-default", type=int, default=None)
    p_init.add_argument("--poll-seconds", type=float, default=None)
    p_init.set_defaults(func=cmd_init)

    p_submit = sub.add_parser("submit", help="submit one queued job")
    p_submit.add_argument("--model", required=True)
    p_submit.add_argument("--prompt-file", required=True)
    p_submit.add_argument("--rtx-repo", default="")
    p_submit.add_argument("--ctx-size", type=int, default=8192)
    p_submit.add_argument("--prefill-n-predict", type=int, default=8)
    p_submit.add_argument("--prefill-min-stream-batch-tokens", type=int, default=-1)
    p_submit.add_argument("--decode-n-predict", type=int, default=16)
    p_submit.add_argument("--kv-chunk-bytes", type=int, default=4 * 1024 * 1024)
    p_submit.add_argument("--kv-max-inflight", type=int, default=256 * 1024 * 1024)
    p_submit.add_argument("--loopback-idle-timeout", type=int, default=20)
    p_submit.add_argument("--no-replay-ack", action="store_true")
    p_submit.add_argument("--priority", type=int, default=0)
    p_submit.add_argument("--max-retries", type=int, default=None)
    p_submit.add_argument("--extra-arg", action="append", default=[], help="extra arg passed through to E2E runner")
    p_submit.set_defaults(func=cmd_submit)

    p_list = sub.add_parser("list", help="list jobs")
    p_list.add_argument("--state", default="")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="show one job")
    p_show.add_argument("job_id")
    p_show.set_defaults(func=cmd_show)

    p_cancel = sub.add_parser("cancel", help="cancel queued job")
    p_cancel.add_argument("job_id")
    p_cancel.set_defaults(func=cmd_cancel)

    p_worker = sub.add_parser("worker", help="run worker loop")
    p_worker.add_argument("--once", action="store_true", help="process at most one job")
    p_worker.add_argument("--poll-seconds", type=float, default=None)
    p_worker.set_defaults(func=cmd_worker)

    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
