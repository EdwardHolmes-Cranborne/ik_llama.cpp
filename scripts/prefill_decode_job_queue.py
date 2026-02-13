#!/usr/bin/env python3
"""
Prefill->handoff->decode queue.

Supports:
- Phase-1 single-flight execution (`worker`)
- Phase-2 split pipeline (`prefill-worker` + `handoff-worker`)

External tooling only; does not modify llama/ik_llama engine paths.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


JOB_VERSION = 2
DEFAULT_SPOOL = "/tmp/ik_prefill_decode_queue"
DEFAULT_MAX_QUEUED = 256
DEFAULT_MAX_SPOOL_BYTES = 0
DEFAULT_POLL_SEC = 2.0

STATE_TERMINAL = {"done", "failed", "canceled"}
STATE_QUEUEABLE = {"queued"}
STATE_ARTIFACT_READY = "artifact_ready"

JOB_MODE_SINGLE_MACHINE = "single_machine_runner"
JOB_MODE_EXTERNAL_COMMAND = "external_command"
JOB_MODE_PHASE2_SPLIT_PIPELINE = "phase2_split_pipeline"

CANCELABLE_STATES = {"queued", STATE_ARTIFACT_READY}


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


def job_mode(job: Dict[str, Any]) -> str:
    req = job.get("request", {})
    if isinstance(req, dict):
        return str(req.get("mode", JOB_MODE_SINGLE_MACHINE))
    return JOB_MODE_SINGLE_MACHINE


def runnable_jobs(spool: Path, *, states: Optional[set[str]] = None, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    wanted_states = states if states is not None else STATE_QUEUEABLE
    out: List[Dict[str, Any]] = []
    for j in list_jobs(spool):
        if str(j.get("state", "")) not in wanted_states:
            continue
        if mode is not None and job_mode(j) != mode:
            continue
        out.append(j)
    return out


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


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def parse_flag_value(argv: List[str], flag: str) -> Optional[str]:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def has_flag(argv: List[str], flag: str) -> bool:
    for tok in argv:
        if tok == flag or tok.startswith(flag + "="):
            return True
    return False


def validate_external_command_guardrails(command_text: str) -> None:
    argv = shlex.split(command_text)
    kv_streams = parse_flag_value(argv, "--kv-streams")
    if kv_streams is not None and kv_streams != "1":
        raise SystemExit("guardrail violation: external command uses --kv-streams != 1")

    split_mode = parse_flag_value(argv, "--split-mode")
    if split_mode == "graph":
        if has_flag(argv, "--no-flash-attn"):
            raise SystemExit("guardrail violation: --split-mode graph with --no-flash-attn is not supported for restore/import")
        if not has_flag(argv, "--flash-attn"):
            raise SystemExit("guardrail violation: --split-mode graph requires --flash-attn")


def update_state(spool: Path, job: Dict[str, Any], state: str, message: str, patch: Optional[Dict[str, Any]] = None) -> None:
    job["state"] = state
    if patch:
        job.update(patch)
    p = job_path(spool, job["job_id"])
    save_job(p, job)
    append_event(spool, job["job_id"], state, message)


def merge_stage_result(job: Dict[str, Any], stage_name: str, stage_payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(job.get("result"), dict):
        out.update(job["result"])
    out[stage_name] = stage_payload
    return out


def run_logged_command(
    *,
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    on_line: Optional[Callable[[str], None]] = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
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
            if on_line:
                on_line(line)
        return proc.wait()


def run_job_command(
    repo_root: Path,
    job: Dict[str, Any],
    run_dir: Path,
    log_path: Path,
    on_stage: Optional[Callable[[str, str], None]] = None,
) -> int:
    req = job["request"]
    mode = str(req.get("mode", JOB_MODE_SINGLE_MACHINE))
    if mode == JOB_MODE_EXTERNAL_COMMAND:
        command_text = str(req.get("command", "")).strip()
        if not command_text:
            raise RuntimeError("external_command mode requires non-empty request.command")
        cmd = shlex.split(command_text)
    else:
        cmd = [
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
    (run_dir / "queue_command.txt").write_text(shell_join(cmd) + "\n", encoding="utf-8")

    child_env = os.environ.copy()
    child_env["IK_PDQ_JOB_ID"] = str(job.get("job_id", ""))
    child_env["IK_PDQ_JOB_MODE"] = mode
    child_env["IK_PDQ_KV_TRANSPORT"] = str(req.get("kv_transport", "auto"))
    child_env["IK_PDQ_MAX_RETRIES"] = str(job.get("max_retries", 0))

    def on_line(line: str) -> None:
        if "[3/7] run prefill sender" in line:
            if on_stage:
                on_stage("prefill_running", "prefill sender started")
        elif "[4/7] replay buffered artifact" in line:
            if on_stage:
                on_stage(STATE_ARTIFACT_READY, "artifact reassembled and ready")
                on_stage("handoff_running", "artifact replay/handoff started")
        elif "[7/7] run decode smoke request" in line:
            if on_stage:
                on_stage("decode_running", "decode smoke request started")

    return run_logged_command(cmd=cmd, cwd=repo_root, env=child_env, log_path=log_path, on_line=on_line)


def phase2_prefill_script(repo_root: Path) -> str:
    override = str(os.environ.get("IK_PDQ_PHASE2_PREFILL_SCRIPT", "")).strip()
    if override:
        return override
    return str(repo_root / "scripts" / "run_phase2_prefill_to_artifact.sh")


def phase2_handoff_script(repo_root: Path) -> str:
    override = str(os.environ.get("IK_PDQ_PHASE2_HANDOFF_SCRIPT", "")).strip()
    if override:
        return override
    return str(repo_root / "scripts" / "run_phase2_handoff_from_artifact.sh")


def run_phase2_prefill_stage(
    repo_root: Path,
    job: Dict[str, Any],
    run_dir: Path,
    log_path: Path,
) -> Tuple[int, Dict[str, Any]]:
    req = job["request"]
    mode = job_mode(job)
    if mode != JOB_MODE_PHASE2_SPLIT_PIPELINE:
        raise RuntimeError(f"run_phase2_prefill_stage called on non-phase2 job mode={mode}")

    stage_dir = run_dir / "prefill_stage"
    cmd = [
        phase2_prefill_script(repo_root),
        "--model", str(req["model"]),
        "--prompt-file", str(req["prompt_file"]),
        "--rtx-repo", str(req["rtx_repo"]),
        "--ctx-size", str(req["ctx_size"]),
        "--n-predict", str(req["prefill_n_predict"]),
        "--prefill-min-stream-batch-tokens", str(req["prefill_min_stream_batch_tokens"]),
        "--kv-transport", str(req.get("kv_transport", "auto")),
        "--kv-stream-chunk-bytes", str(req["kv_chunk_bytes"]),
        "--kv-max-inflight-bytes", str(req["kv_max_inflight"]),
        "--buffer-port", str(req["prefill_buffer_port"]),
        "--loopback-idle-timeout", str(req["loopback_idle_timeout"]),
        "--output-dir", str(stage_dir),
    ]
    if bool(req.get("kv_transport_fallback", True)):
        cmd.append("--kv-transport-fallback")
    else:
        cmd.append("--no-kv-transport-fallback")

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "queue_command.txt").write_text(shell_join(cmd) + "\n", encoding="utf-8")

    child_env = os.environ.copy()
    child_env["IK_PDQ_JOB_ID"] = str(job.get("job_id", ""))
    child_env["IK_PDQ_JOB_MODE"] = mode
    child_env["IK_PDQ_KV_TRANSPORT"] = str(req.get("kv_transport", "auto"))
    child_env["IK_PDQ_STAGE"] = "prefill"

    rc = run_logged_command(cmd=cmd, cwd=repo_root, env=child_env, log_path=log_path)

    meta_path = stage_dir / "prefill_artifact_meta.json"
    stage_result: Dict[str, Any] = {
        "return_code": rc,
        "stage_dir": str(stage_dir),
        "worker_stream_log": str(log_path),
        "meta_path": str(meta_path),
    }

    if rc == 0:
        if not meta_path.is_file():
            raise RuntimeError(f"phase2 prefill meta not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        artifact_path = str(meta.get("artifact_path", "")).strip()
        if not artifact_path:
            raise RuntimeError("phase2 prefill meta missing artifact_path")
        if not Path(artifact_path).is_file():
            raise RuntimeError(f"phase2 prefill artifact not found: {artifact_path}")
        stage_result["artifact_path"] = artifact_path
        stage_result["artifact_bytes"] = int(meta.get("artifact_bytes", 0))

    return rc, stage_result


def run_phase2_handoff_stage(
    repo_root: Path,
    job: Dict[str, Any],
    run_dir: Path,
    artifact_path: str,
    log_path: Path,
    on_stage: Optional[Callable[[str, str], None]] = None,
) -> Tuple[int, Dict[str, Any]]:
    req = job["request"]
    mode = job_mode(job)
    if mode != JOB_MODE_PHASE2_SPLIT_PIPELINE:
        raise RuntimeError(f"run_phase2_handoff_stage called on non-phase2 job mode={mode}")

    stage_dir = run_dir / f"handoff_stage_attempt_{int(job.get('handoff_attempt', 0))}"
    cmd = [
        phase2_handoff_script(repo_root),
        "--artifact", artifact_path,
        "--decode-host", str(req["decode_host"]),
        "--decode-port", str(req["decode_port"]),
        "--kv-transport", str(req.get("kv_transport", "auto")),
        "--kv-chunk-bytes", str(req["kv_chunk_bytes"]),
        "--output-dir", str(stage_dir),
    ]
    if bool(req.get("kv_transport_fallback", True)):
        cmd.append("--kv-transport-fallback")
    else:
        cmd.append("--no-kv-transport-fallback")
    if bool(req.get("replay_ack_required", True)):
        cmd.append("--ack-required")
    if bool(req.get("replay_wait_ack", True)):
        cmd.append("--wait-ack")
    if bool(req.get("decode_smoke", False)):
        cmd.extend([
            "--decode-smoke",
            "--decode-smoke-n-predict", str(req.get("decode_n_predict", 16)),
            "--decode-smoke-prompt", str(req.get("decode_smoke_prompt", "Buffered handoff decode smoke test.")),
        ])
        decode_http_url = str(req.get("decode_http_url", "")).strip()
        if decode_http_url:
            cmd.extend(["--decode-http-url", decode_http_url])

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "queue_command.txt").write_text(shell_join(cmd) + "\n", encoding="utf-8")

    child_env = os.environ.copy()
    child_env["IK_PDQ_JOB_ID"] = str(job.get("job_id", ""))
    child_env["IK_PDQ_JOB_MODE"] = mode
    child_env["IK_PDQ_KV_TRANSPORT"] = str(req.get("kv_transport", "auto"))
    child_env["IK_PDQ_STAGE"] = "handoff"

    def on_line(line: str) -> None:
        if "[phase2/2] decode smoke request" in line and on_stage:
            on_stage("decode_running", "decode smoke request started")

    rc = run_logged_command(cmd=cmd, cwd=repo_root, env=child_env, log_path=log_path, on_line=on_line)

    meta_path = stage_dir / "handoff_meta.json"
    stage_result: Dict[str, Any] = {
        "return_code": rc,
        "stage_dir": str(stage_dir),
        "worker_stream_log": str(log_path),
        "meta_path": str(meta_path),
        "artifact_path": artifact_path,
    }

    if rc == 0 and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                stage_result["meta"] = meta
        except Exception:
            pass

    return rc, stage_result


def pick_next_phase1_job(spool: Path) -> Optional[Dict[str, Any]]:
    jobs = runnable_jobs(spool)
    for job in jobs:
        if job_mode(job) != JOB_MODE_PHASE2_SPLIT_PIPELINE:
            return job
    return None


def pick_next_phase2_prefill_job(spool: Path) -> Optional[Dict[str, Any]]:
    jobs = runnable_jobs(spool, states={"queued"}, mode=JOB_MODE_PHASE2_SPLIT_PIPELINE)
    return jobs[0] if jobs else None


def pick_next_phase2_handoff_job(spool: Path) -> Optional[Dict[str, Any]]:
    jobs = runnable_jobs(spool, states={STATE_ARTIFACT_READY}, mode=JOB_MODE_PHASE2_SPLIT_PIPELINE)
    if not jobs:
        return None
    jobs.sort(
        key=lambda j: (
            -int(j.get("priority", 0)),
            int(j.get("artifact_ready_at_unix_us", 0)),
            int(j.get("created_at_unix_us", 0)),
            str(j.get("job_id", "")),
        )
    )
    return jobs[0]


def acquire_lock(spool: Path, filename: str, already_held_message: str) -> int:
    lock_path = spool / filename
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise SystemExit(already_held_message)
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

    mode = str(args.mode)
    if mode not in (JOB_MODE_SINGLE_MACHINE, JOB_MODE_EXTERNAL_COMMAND, JOB_MODE_PHASE2_SPLIT_PIPELINE):
        raise SystemExit(f"invalid mode: {mode}")

    if mode == JOB_MODE_SINGLE_MACHINE:
        if not args.model or not args.prompt_file:
            raise SystemExit("single_machine_runner mode requires --model and --prompt-file")
        require_file(args.model, "model")
        require_file(args.prompt_file, "prompt file")
        if args.rtx_repo and not Path(args.rtx_repo).is_dir():
            raise SystemExit(f"rtx repo not found: {args.rtx_repo}")
    elif mode == JOB_MODE_EXTERNAL_COMMAND:
        if not args.command.strip():
            raise SystemExit("external_command mode requires --command")
        if args.enforce_guardrails:
            validate_external_command_guardrails(args.command)
    else:
        if not args.model or not args.prompt_file or not args.rtx_repo:
            raise SystemExit("phase2_split_pipeline mode requires --model --prompt-file --rtx-repo")
        if not args.decode_host:
            raise SystemExit("phase2_split_pipeline mode requires --decode-host")
        if str(args.kv_transport) == "disabled":
            raise SystemExit("phase2_split_pipeline mode requires kv transport enabled (use auto|rdma|tcp|mixed)")
        require_file(args.model, "model")
        require_file(args.prompt_file, "prompt file")
        if not Path(args.rtx_repo).is_dir():
            raise SystemExit(f"rtx repo not found: {args.rtx_repo}")
        if args.decode_http_url and not str(args.decode_http_url).startswith(("http://", "https://")):
            raise SystemExit("--decode-http-url must start with http:// or https://")

    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    retries_default = int(args.max_retries if args.max_retries is not None else cfg.get("max_retries_default", 0))
    prefill_retries = int(
        args.max_prefill_retries if args.max_prefill_retries is not None else retries_default
    )
    handoff_retries = int(
        args.max_handoff_retries if args.max_handoff_retries is not None else retries_default
    )

    job: Dict[str, Any] = {
        "version": JOB_VERSION,
        "job_id": job_id,
        "state": "queued",
        "priority": int(args.priority),
        "attempt": 0,
        "prefill_attempt": 0,
        "handoff_attempt": 0,
        "max_retries": max(0, retries_default),
        "max_prefill_retries": max(0, prefill_retries),
        "max_handoff_retries": max(0, handoff_retries),
        "created_at_unix_us": now_us(),
        "updated_at_unix_us": now_us(),
        "request": {
            "mode": mode,
            "command": str(args.command),
            "kv_transport": str(args.kv_transport),
            "kv_transport_fallback": bool(args.kv_transport_fallback),
            "enforce_guardrails": bool(args.enforce_guardrails),
            "model": str(Path(args.model).resolve()) if args.model else "",
            "prompt_file": str(Path(args.prompt_file).resolve()) if args.prompt_file else "",
            "rtx_repo": str(Path(args.rtx_repo).resolve()) if args.rtx_repo else "",
            "decode_host": str(args.decode_host),
            "decode_port": int(args.decode_port),
            "decode_http_url": str(args.decode_http_url),
            "decode_smoke": bool(args.decode_smoke),
            "decode_smoke_prompt": str(args.decode_smoke_prompt),
            "ctx_size": int(args.ctx_size),
            "prefill_n_predict": int(args.prefill_n_predict),
            "prefill_min_stream_batch_tokens": int(args.prefill_min_stream_batch_tokens),
            "prefill_buffer_port": int(args.prefill_buffer_port),
            "decode_n_predict": int(args.decode_n_predict),
            "kv_chunk_bytes": int(args.kv_chunk_bytes),
            "kv_max_inflight": int(args.kv_max_inflight),
            "loopback_idle_timeout": int(args.loopback_idle_timeout),
            "no_replay_ack": bool(args.no_replay_ack),
            "replay_ack_required": not bool(args.no_replay_ack),
            "replay_wait_ack": not bool(args.no_replay_ack),
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
        mode = job_mode(j)
        print(
            f"{j.get('job_id')} state={j.get('state')} prio={j.get('priority')} "
            f"attempt={j.get('attempt')}/{j.get('max_retries')} "
            f"pf={j.get('prefill_attempt')}/{j.get('max_prefill_retries')} "
            f"ho={j.get('handoff_attempt')}/{j.get('max_handoff_retries')} "
            f"mode={mode} updated_us={j.get('updated_at_unix_us')}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    jobs = list_jobs(spool)
    counts: Dict[str, int] = {}
    for job in jobs:
        st = str(job.get("state", "unknown"))
        counts[st] = counts.get(st, 0) + 1

    active = [
        {
            "job_id": j.get("job_id"),
            "state": j.get("state"),
            "attempt": j.get("attempt"),
            "prefill_attempt": j.get("prefill_attempt"),
            "handoff_attempt": j.get("handoff_attempt"),
            "priority": j.get("priority"),
            "mode": job_mode(j),
        }
        for j in jobs
        if str(j.get("state", "")) not in STATE_TERMINAL
        and str(j.get("state", "")) not in {"queued", STATE_ARTIFACT_READY}
    ]
    queued = [j.get("job_id") for j in jobs if str(j.get("state", "")) == "queued"]
    artifact_ready = [j.get("job_id") for j in jobs if str(j.get("state", "")) == STATE_ARTIFACT_READY]
    payload = {
        "spool_dir": str(spool),
        "spool_bytes": spool_size_bytes(spool),
        "counts": counts,
        "queued_depth": len(queued),
        "queued_job_ids": queued,
        "artifact_ready_depth": len(artifact_ready),
        "artifact_ready_job_ids": artifact_ready,
        "active_jobs": active,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"spool_dir={payload['spool_dir']}")
        print(f"spool_bytes={payload['spool_bytes']}")
        print(f"queued_depth={payload['queued_depth']}")
        print(f"artifact_ready_depth={payload['artifact_ready_depth']}")
        print("counts=" + json.dumps(counts, sort_keys=True))
        if active:
            print("active_jobs=" + json.dumps(active, sort_keys=True))
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
    if state not in CANCELABLE_STATES:
        raise SystemExit(f"job is not cancelable in state={state} (allowed: {sorted(CANCELABLE_STATES)})")
    update_state(spool, job, "canceled", "job canceled by user")
    print(json.dumps({"ok": True, "job_id": args.job_id, "state": "canceled"}, indent=2))
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    cfg = load_config(spool)
    poll_sec = float(args.poll_seconds if args.poll_seconds is not None else cfg.get("poll_seconds", DEFAULT_POLL_SEC))
    lock_fd = acquire_lock(spool, "worker.lock", "worker lock is already held by another process")
    repo_root = Path(__file__).resolve().parents[1]

    try:
        while True:
            job = pick_next_phase1_job(spool)
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

            runner_exception: Optional[str] = None
            try:
                rc = run_job_command(repo_root, job, run_dir, log_path, on_stage=on_stage)
            except Exception as exc:
                rc = 127
                runner_exception = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write(f"[queue-worker] job runner exception: {runner_exception}\n")
                    logf.write(tb)
                    if not tb.endswith("\n"):
                        logf.write("\n")
                sys.stderr.write(f"[queue-worker] {job['job_id']} runner exception: {runner_exception}\n")
                sys.stderr.flush()

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
                result_patch: Dict[str, Any] = {
                    "return_code": rc,
                    "run_dir": str(run_dir),
                    "worker_stream_log": str(log_path),
                }
                if runner_exception:
                    result_patch["runner_exception"] = runner_exception
                max_retries = int(job.get("max_retries", 0))
                attempt = int(job.get("attempt", 1))
                if attempt <= max_retries:
                    update_state(
                        spool,
                        job,
                        "queued",
                        f"job failed rc={rc}; retrying",
                        patch={"result": result_patch},
                    )
                else:
                    update_state(
                        spool,
                        job,
                        "failed",
                        f"job failed rc={rc}; retries exhausted",
                        patch={"result": result_patch},
                    )

            if args.once:
                return 0
    finally:
        os.close(lock_fd)


def cmd_prefill_worker(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    cfg = load_config(spool)
    poll_sec = float(args.poll_seconds if args.poll_seconds is not None else cfg.get("poll_seconds", DEFAULT_POLL_SEC))
    lock_fd = acquire_lock(spool, "prefill_worker.lock", "prefill worker lock is already held by another process")
    repo_root = Path(__file__).resolve().parents[1]

    try:
        while True:
            job = pick_next_phase2_prefill_job(spool)
            if job is None:
                if args.once:
                    print(json.dumps({"ok": True, "idle": True}, indent=2))
                    return 0
                time.sleep(max(0.05, poll_sec))
                continue

            job["prefill_attempt"] = int(job.get("prefill_attempt", 0)) + 1
            job["attempt"] = int(job.get("attempt", 0)) + 1
            update_state(spool, job, "prefill_running", "prefill worker picked job")

            run_dir = spool / "runs" / job["job_id"]
            log_path = run_dir / f"prefill_worker_attempt_{job['prefill_attempt']}.log"

            runner_exception: Optional[str] = None
            stage_result: Dict[str, Any] = {}
            try:
                rc, stage_result = run_phase2_prefill_stage(repo_root, job, run_dir, log_path)
            except Exception as exc:
                rc = 127
                runner_exception = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write(f"[prefill-worker] job runner exception: {runner_exception}\n")
                    logf.write(tb)
                    if not tb.endswith("\n"):
                        logf.write("\n")
                sys.stderr.write(f"[prefill-worker] {job['job_id']} runner exception: {runner_exception}\n")
                sys.stderr.flush()

            if rc == 0:
                artifact_path = str(stage_result.get("artifact_path", "")).strip()
                if not artifact_path:
                    rc = 127
                    runner_exception = "RuntimeError: phase2 prefill stage did not produce artifact_path"
                else:
                    result_patch = merge_stage_result(job, "prefill", stage_result)
                    update_state(
                        spool,
                        job,
                        STATE_ARTIFACT_READY,
                        "prefill artifact ready",
                        patch={
                            "result": result_patch,
                            "artifact_path": artifact_path,
                            "artifact_ready_at_unix_us": now_us(),
                        },
                    )
                    if args.once:
                        return 0
                    continue

            fail_stage_result: Dict[str, Any] = {
                "return_code": rc,
                "run_dir": str(run_dir),
                "worker_stream_log": str(log_path),
            }
            if runner_exception:
                fail_stage_result["runner_exception"] = runner_exception
            result_patch = merge_stage_result(job, "prefill", fail_stage_result)

            max_retries = int(job.get("max_prefill_retries", 0))
            prefill_attempt = int(job.get("prefill_attempt", 1))
            if prefill_attempt <= max_retries:
                update_state(
                    spool,
                    job,
                    "queued",
                    f"prefill failed rc={rc}; retrying",
                    patch={"result": result_patch},
                )
            else:
                update_state(
                    spool,
                    job,
                    "failed",
                    f"prefill failed rc={rc}; retries exhausted",
                    patch={"result": result_patch},
                )

            if args.once:
                return 0
    finally:
        os.close(lock_fd)


def cmd_handoff_worker(args: argparse.Namespace) -> int:
    spool = Path(args.spool_dir)
    ensure_dirs(spool)
    cfg = load_config(spool)
    poll_sec = float(args.poll_seconds if args.poll_seconds is not None else cfg.get("poll_seconds", DEFAULT_POLL_SEC))
    lock_fd = acquire_lock(spool, "handoff_worker.lock", "handoff worker lock is already held by another process")
    repo_root = Path(__file__).resolve().parents[1]

    try:
        while True:
            job = pick_next_phase2_handoff_job(spool)
            if job is None:
                if args.once:
                    print(json.dumps({"ok": True, "idle": True}, indent=2))
                    return 0
                time.sleep(max(0.05, poll_sec))
                continue

            artifact_path = str(job.get("artifact_path", "")).strip()
            if not artifact_path:
                update_state(
                    spool,
                    job,
                    "failed",
                    "handoff worker missing artifact_path",
                    patch={"result": merge_stage_result(job, "handoff", {"return_code": 127, "error": "missing artifact_path"})},
                )
                if args.once:
                    return 0
                continue

            job["handoff_attempt"] = int(job.get("handoff_attempt", 0)) + 1
            job["attempt"] = int(job.get("attempt", 0)) + 1
            update_state(spool, job, "handoff_running", "handoff worker picked artifact")

            run_dir = spool / "runs" / job["job_id"]
            log_path = run_dir / f"handoff_worker_attempt_{job['handoff_attempt']}.log"
            stage_state = {"last": str(job.get("state", ""))}

            def on_stage(state: str, message: str) -> None:
                if state == stage_state["last"]:
                    return
                stage_state["last"] = state
                update_state(spool, job, state, message)

            runner_exception: Optional[str] = None
            stage_result: Dict[str, Any] = {}
            try:
                rc, stage_result = run_phase2_handoff_stage(
                    repo_root,
                    job,
                    run_dir,
                    artifact_path,
                    log_path,
                    on_stage=on_stage,
                )
            except Exception as exc:
                rc = 127
                runner_exception = f"{type(exc).__name__}: {exc}"
                tb = traceback.format_exc()
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write(f"[handoff-worker] job runner exception: {runner_exception}\n")
                    logf.write(tb)
                    if not tb.endswith("\n"):
                        logf.write("\n")
                sys.stderr.write(f"[handoff-worker] {job['job_id']} runner exception: {runner_exception}\n")
                sys.stderr.flush()

            if rc == 0:
                result_patch = merge_stage_result(job, "handoff", stage_result)
                update_state(
                    spool,
                    job,
                    "done",
                    "handoff/decode completed successfully",
                    patch={"result": result_patch},
                )
                if args.once:
                    return 0
                continue

            fail_stage_result: Dict[str, Any] = {
                "return_code": rc,
                "run_dir": str(run_dir),
                "worker_stream_log": str(log_path),
                "artifact_path": artifact_path,
            }
            if runner_exception:
                fail_stage_result["runner_exception"] = runner_exception
            result_patch = merge_stage_result(job, "handoff", fail_stage_result)

            max_retries = int(job.get("max_handoff_retries", 0))
            handoff_attempt = int(job.get("handoff_attempt", 1))
            if handoff_attempt <= max_retries:
                update_state(
                    spool,
                    job,
                    STATE_ARTIFACT_READY,
                    f"handoff failed rc={rc}; retrying",
                    patch={"result": result_patch},
                )
            else:
                update_state(
                    spool,
                    job,
                    "failed",
                    f"handoff failed rc={rc}; retries exhausted",
                    patch={"result": result_patch},
                )

            if args.once:
                return 0
    finally:
        os.close(lock_fd)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Prefill/decode queue (phase-1 + phase-2)")
    ap.add_argument("--spool-dir", default=DEFAULT_SPOOL, help=f"queue spool directory (default: {DEFAULT_SPOOL})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="initialize queue spool/config")
    p_init.add_argument("--max-queued", type=int, default=None)
    p_init.add_argument("--max-spool-bytes", type=int, default=None)
    p_init.add_argument("--max-retries-default", type=int, default=None)
    p_init.add_argument("--poll-seconds", type=float, default=None)
    p_init.set_defaults(func=cmd_init)

    p_submit = sub.add_parser("submit", help="submit one queued job")
    p_submit.add_argument(
        "--mode",
        choices=[JOB_MODE_SINGLE_MACHINE, JOB_MODE_EXTERNAL_COMMAND, JOB_MODE_PHASE2_SPLIT_PIPELINE],
        default=JOB_MODE_SINGLE_MACHINE,
    )
    p_submit.add_argument("--command", default="", help="command for external_command mode")
    p_submit.add_argument("--kv-transport", choices=["auto", "tcp", "rdma", "mixed", "disabled"], default="auto")
    p_submit.add_argument("--kv-transport-fallback", dest="kv_transport_fallback", action="store_true")
    p_submit.add_argument("--no-kv-transport-fallback", dest="kv_transport_fallback", action="store_false")
    p_submit.add_argument("--enforce-guardrails", dest="enforce_guardrails", action="store_true")
    p_submit.add_argument("--no-enforce-guardrails", dest="enforce_guardrails", action="store_false")
    p_submit.set_defaults(enforce_guardrails=True, kv_transport_fallback=True)

    p_submit.add_argument("--model", default="")
    p_submit.add_argument("--prompt-file", default="")
    p_submit.add_argument("--rtx-repo", default="")
    p_submit.add_argument("--decode-host", default="")
    p_submit.add_argument("--decode-port", type=int, default=19001)
    p_submit.add_argument("--decode-http-url", default="")
    p_submit.add_argument("--decode-smoke", action="store_true")
    p_submit.add_argument("--decode-smoke-prompt", default="Buffered handoff decode smoke test.")

    p_submit.add_argument("--ctx-size", type=int, default=8192)
    p_submit.add_argument("--prefill-n-predict", type=int, default=8)
    p_submit.add_argument("--prefill-min-stream-batch-tokens", type=int, default=-1)
    p_submit.add_argument("--prefill-buffer-port", type=int, default=29001)
    p_submit.add_argument("--decode-n-predict", type=int, default=16)

    p_submit.add_argument("--kv-chunk-bytes", type=int, default=4 * 1024 * 1024)
    p_submit.add_argument("--kv-max-inflight", type=int, default=256 * 1024 * 1024)
    p_submit.add_argument("--loopback-idle-timeout", type=int, default=20)
    p_submit.add_argument("--no-replay-ack", action="store_true")

    p_submit.add_argument("--priority", type=int, default=0)
    p_submit.add_argument("--max-retries", type=int, default=None)
    p_submit.add_argument("--max-prefill-retries", type=int, default=None)
    p_submit.add_argument("--max-handoff-retries", type=int, default=None)
    p_submit.add_argument("--extra-arg", action="append", default=[], help="extra arg passed through to E2E runner")
    p_submit.set_defaults(func=cmd_submit)

    p_list = sub.add_parser("list", help="list jobs")
    p_list.add_argument("--state", default="")
    p_list.add_argument("--json", action="store_true")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="queue status summary")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_show = sub.add_parser("show", help="show one job")
    p_show.add_argument("job_id")
    p_show.set_defaults(func=cmd_show)

    p_cancel = sub.add_parser("cancel", help="cancel queued/artifact-ready job")
    p_cancel.add_argument("job_id")
    p_cancel.set_defaults(func=cmd_cancel)

    p_worker = sub.add_parser("worker", help="run phase-1 single-flight worker loop")
    p_worker.add_argument("--once", action="store_true", help="process at most one job")
    p_worker.add_argument("--poll-seconds", type=float, default=None)
    p_worker.set_defaults(func=cmd_worker)

    p_prefill = sub.add_parser("prefill-worker", help="run phase-2 prefill worker loop")
    p_prefill.add_argument("--once", action="store_true", help="process at most one job")
    p_prefill.add_argument("--poll-seconds", type=float, default=None)
    p_prefill.set_defaults(func=cmd_prefill_worker)

    p_handoff = sub.add_parser("handoff-worker", help="run phase-2 handoff/decode worker loop")
    p_handoff.add_argument("--once", action="store_true", help="process at most one job")
    p_handoff.add_argument("--poll-seconds", type=float, default=None)
    p_handoff.set_defaults(func=cmd_handoff_worker)

    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
