#!/usr/bin/env python3
"""
Replay a captured KV artifact (or chunk directory) to the ik KV receiver over TBP.

Supports transport mode selection for phase-2 handoff workflows:
- tcp
- rdma (address/interface selection semantics)
- auto|mixed (mode fallback order)
"""

import argparse
import json
import os
import socket
import struct
import time
import zlib
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

TBP_MAGIC = 0x54425031  # "TBP1"
TBP_PROTO_MAJOR = 1
TBP_PROTO_MINOR = 0
TBP_HEADER_SIZE = 52

TBP_MSG_HELLO = 1
TBP_MSG_SESSION_START = 3
TBP_MSG_KV_SEGMENT_BEGIN = 7
TBP_MSG_KV_CHUNK = 8
TBP_MSG_KV_SEGMENT_END = 9
TBP_MSG_KV_ACK = 10
TBP_MSG_KV_DONE = 12


def env_str(key: str, default: str = "") -> str:
    val = os.environ.get(key, "")
    return val if val else default


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def split_csv(raw: str) -> List[str]:
    out: List[str] = []
    for item in raw.split(","):
        v = item.strip()
        if v and v not in out:
            out.append(v)
    return out


def normalize_transport_mode(raw: str) -> str:
    mode = raw.strip().lower()
    if mode == "tb-direct":
        return "rdma"
    if mode in ("tb-ethernet", "ethernet"):
        return "tcp"
    return mode


def mode_order(mode: str, fallback: bool) -> List[str]:
    if mode in ("auto", "mixed"):
        return ["rdma", "tcp"]
    if mode == "rdma":
        return ["rdma", "tcp"] if fallback else ["rdma"]
    if mode == "tcp":
        return ["tcp", "rdma"] if fallback else ["tcp"]
    if mode == "disabled":
        return ["disabled"]
    return []


def crc32_u32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def parse_session_id(raw: str) -> int:
    if raw.startswith("0x") or raw.startswith("0X"):
        return int(raw, 16)
    return int(raw, 10)


def build_tbp_frame(
    msg_type: int,
    flags: int,
    session_id: int,
    stream_id: int,
    seq_no: int,
    payload: bytes,
) -> bytes:
    payload_crc = crc32_u32(payload) if payload else 0
    hdr = bytearray()
    hdr += struct.pack("<I", TBP_MAGIC)
    hdr += struct.pack("<H", TBP_PROTO_MAJOR)
    hdr += struct.pack("<H", TBP_PROTO_MINOR)
    hdr += struct.pack("<H", msg_type)
    hdr += struct.pack("<H", flags)
    hdr += struct.pack("<I", TBP_HEADER_SIZE)
    hdr += struct.pack("<I", len(payload))
    hdr += struct.pack("<Q", session_id)
    hdr += struct.pack("<Q", stream_id)
    hdr += struct.pack("<Q", seq_no)
    hdr += struct.pack("<I", 0)  # header CRC placeholder
    hdr += struct.pack("<I", payload_crc)

    header_crc = crc32_u32(bytes(hdr))
    hdr[44:48] = struct.pack("<I", header_crc)
    return bytes(hdr) + payload


def recv_exact(sock: socket.socket, n: int, timeout_sec: float) -> Optional[bytes]:
    out = bytearray()
    deadline = time.monotonic() + max(timeout_sec, 0.01)
    while len(out) < n:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        sock.settimeout(remaining)
        try:
            chunk = sock.recv(n - len(out))
        except (socket.timeout, TimeoutError):
            return None
        except OSError:
            return None
        if not chunk:
            return None
        out.extend(chunk)
    return bytes(out)


def recv_tbp_frame(sock: socket.socket, timeout_sec: float) -> Optional[Tuple[int, int, int, int, int, bytes]]:
    hdr = recv_exact(sock, TBP_HEADER_SIZE, timeout_sec)
    if hdr is None:
        return None
    (
        magic,
        _maj,
        _min,
        msg_type,
        flags,
        header_bytes,
        payload_bytes,
        session_id,
        stream_id,
        seq_no,
        _hdr_crc,
        _payload_crc,
    ) = struct.unpack("<IHHHHIIQQQII", hdr)
    if magic != TBP_MAGIC or header_bytes != TBP_HEADER_SIZE:
        return None
    payload = b""
    if payload_bytes > 0:
        got = recv_exact(sock, payload_bytes, timeout_sec)
        if got is None:
            return None
        payload = got
    return msg_type, session_id, stream_id, seq_no, flags, payload


def iter_chunks_from_artifact(path: Path, chunk_bytes: int) -> Iterator[bytes]:
    with path.open("rb") as f:
        while True:
            blob = f.read(chunk_bytes)
            if not blob:
                break
            yield blob


def parse_seq_from_name(path: Path) -> Optional[int]:
    name = path.name
    if not name.startswith("seq_") or not name.endswith(".bin"):
        return None
    raw = name[4:-4]
    if not raw.isdigit():
        return None
    return int(raw, 10)


def iter_chunks_from_dir(path: Path) -> Iterator[bytes]:
    seq_files: List[Tuple[int, Path]] = []
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        seq = parse_seq_from_name(entry)
        if seq is None:
            continue
        seq_files.append((seq, entry))
    seq_files.sort(key=lambda p: p[0])
    for _, p in seq_files:
        yield p.read_bytes()


def send_frame(
    sock: socket.socket,
    msg_type: int,
    session_id: int,
    stream_id: int,
    seq_no: int,
    payload_text: str = "",
    payload_bytes: Optional[bytes] = None,
) -> None:
    payload = payload_bytes if payload_bytes is not None else payload_text.encode("utf-8")
    frame = build_tbp_frame(msg_type, 0, session_id, stream_id, seq_no, payload)
    sock.sendall(frame)


def parse_endpoint(raw: str) -> Tuple[str, int]:
    text = raw.strip()
    if not text:
        raise ValueError("empty endpoint")

    if text.startswith("["):
        # [ipv6]:port
        end = text.find("]")
        if end <= 1 or end + 2 > len(text) or text[end + 1] != ":":
            raise ValueError(f"invalid endpoint format: {raw}")
        host = text[1:end]
        port = int(text[end + 2 :])
        return host, port

    if ":" not in text:
        raise ValueError(f"missing port in endpoint: {raw}")
    host, port_text = text.rsplit(":", 1)
    if not host:
        raise ValueError(f"missing host in endpoint: {raw}")
    return host, int(port_text)


def default_endpoint_for_mode(mode: str, host_arg: str, port_arg: int) -> str:
    if mode == "rdma":
        explicit = env_str("LLAMA_PREFILL_KV_RDMA_ENDPOINT", env_str("LLAMA_PREFILL_TB_ENDPOINT", ""))
        if explicit:
            return explicit
        host = env_str("LLAMA_PREFILL_KV_RDMA_HOST", env_str("LLAMA_PREFILL_KV_HOST", host_arg))
        port = env_int("LLAMA_PREFILL_KV_RDMA_PORT", env_int("LLAMA_PREFILL_KV_PORT", port_arg))
        return f"{host}:{port}"

    # tcp path
    explicit = env_str("LLAMA_PREFILL_KV_TCP_ENDPOINT", "")
    if explicit:
        return explicit
    host = env_str("LLAMA_PREFILL_KV_HOST", host_arg)
    port = env_int("LLAMA_PREFILL_KV_PORT", port_arg)
    return f"{host}:{port}"


def resolve_endpoints(mode: str, args: argparse.Namespace) -> List[str]:
    out: List[str] = []

    def add(ep: str) -> None:
        ep = ep.strip()
        if not ep:
            return
        if ep not in out:
            out.append(ep)

    add(args.endpoint)
    for ep in split_csv(args.peer_addrs):
        add(ep)

    env_peers = split_csv(env_str("LLAMA_PREFILL_KV_PEER_ADDRS", ""))
    for ep in env_peers:
        add(ep)

    if mode == "rdma":
        for ep in split_csv(env_str("LLAMA_PREFILL_KV_RDMA_PEER_ADDRS", "")):
            add(ep)
    else:
        for ep in split_csv(env_str("LLAMA_PREFILL_KV_TCP_PEER_ADDRS", "")):
            add(ep)

    add(default_endpoint_for_mode(mode, args.host, args.port))

    return out


def resolve_bind_addrs(mode: str, args: argparse.Namespace) -> List[str]:
    if args.bind_addrs.strip():
        return split_csv(args.bind_addrs)

    if mode == "rdma":
        raw = env_str("LLAMA_PREFILL_KV_RDMA_BIND_ADDRS", env_str("LLAMA_PREFILL_KV_BIND_ADDRS", ""))
    else:
        raw = env_str("LLAMA_PREFILL_KV_TCP_BIND_ADDRS", env_str("LLAMA_PREFILL_KV_BIND_ADDRS", ""))
    return split_csv(raw)


def connect_transport(mode: str, args: argparse.Namespace) -> Tuple[socket.socket, str, str]:
    endpoints = resolve_endpoints(mode, args)
    if not endpoints:
        raise RuntimeError(f"no endpoints resolved for mode={mode}")

    bind_addrs = resolve_bind_addrs(mode, args)
    bind_candidates = [""]
    bind_candidates.extend([b for b in bind_addrs if b])

    timeout_sec = max(0.05, float(args.connect_timeout_ms) / 1000.0)
    errors: List[str] = []

    for endpoint in endpoints:
        try:
            host, port = parse_endpoint(endpoint)
        except Exception as exc:
            errors.append(f"{endpoint}: parse failed ({exc})")
            continue

        for bind_addr in bind_candidates:
            source = (bind_addr, 0) if bind_addr else None
            try:
                sock = socket.create_connection((host, port), timeout=timeout_sec, source_address=source)
                return sock, endpoint, bind_addr
            except OSError as exc:
                btxt = bind_addr if bind_addr else "<none>"
                errors.append(f"{endpoint} bind={btxt}: {exc}")

    joined = "; ".join(errors[-6:])
    raise RuntimeError(f"connect failed for mode={mode}: {joined}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay buffered KV stream to ik kv-receiver")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact", help="path to KV artifact file")
    src.add_argument("--chunks-dir", help="directory containing seq_*.bin chunk files")

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19001)
    ap.add_argument("--endpoint", default="", help="explicit endpoint host:port override")
    ap.add_argument("--peer-addrs", default="", help="comma-separated peer endpoint overrides")
    ap.add_argument("--bind-addrs", default="", help="comma-separated local source addresses for outgoing socket bind")

    ap.add_argument("--transport-mode", default="tcp", help="auto|rdma|tcp|mixed|disabled (aliases: tb-direct,tb-ethernet,ethernet)")
    ap.add_argument("--transport-fallback", action="store_true", help="allow rdma<->tcp fallback when transport-mode is strict")
    ap.add_argument("--connect-timeout-ms", type=int, default=5000)

    ap.add_argument("--session-id", default="")
    ap.add_argument("--stream-id", type=int, default=1)
    ap.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    ap.add_argument("--ack-required", action="store_true")
    ap.add_argument("--wait-ack", action="store_true")
    ap.add_argument("--ack-timeout-ms", type=int, default=1000)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--mode", choices=["progressive", "bulk"], default="progressive")
    ap.add_argument("--execution-mode", choices=["coupled", "decoupled"], default="coupled")
    ap.add_argument("--output", default="", help="optional json summary output path")
    args = ap.parse_args()

    requested_mode = normalize_transport_mode(str(args.transport_mode))
    if requested_mode not in {"auto", "rdma", "tcp", "mixed", "disabled"}:
        raise SystemExit(
            f"invalid --transport-mode '{args.transport_mode}' "
            "(allowed: auto,rdma,tcp,mixed,disabled; aliases: tb-direct,tb-ethernet,ethernet)"
        )
    if requested_mode == "disabled":
        raise SystemExit("transport mode 'disabled' cannot replay artifact")

    if args.session_id:
        wire_session_id = parse_session_id(args.session_id)
    else:
        wire_session_id = int(time.time_ns() & 0xFFFFFFFFFFFFFFFF)

    stream_id = max(1, int(args.stream_id))
    chunk_bytes = max(1, int(args.chunk_bytes))
    timeout_sec = max(0.01, args.ack_timeout_ms / 1000.0)
    max_retries = max(0, int(args.max_retries))

    if args.artifact:
        source_path = Path(args.artifact)
        if not source_path.exists():
            raise SystemExit(f"artifact not found: {source_path}")
        chunks = list(iter_chunks_from_artifact(source_path, chunk_bytes))
        artifact_name = source_path.name
    else:
        source_path = Path(args.chunks_dir)
        if not source_path.exists() or not source_path.is_dir():
            raise SystemExit(f"chunk dir not found: {source_path}")
        chunks = list(iter_chunks_from_dir(source_path))
        artifact_name = source_path.name + ".replayed"

    if not chunks:
        raise SystemExit("no chunks to replay")

    total_bytes = sum(len(c) for c in chunks)
    seq = 0
    retransmit_chunks = 0
    ack_frames_seen = 0
    nack_frames_seen = 0

    attempts = mode_order(requested_mode, bool(args.transport_fallback))
    if not attempts:
        raise SystemExit(f"could not resolve transport mode order for mode={requested_mode}")

    connect_errors: List[str] = []
    sock: Optional[socket.socket] = None
    resolved_mode = ""
    resolved_endpoint = ""
    resolved_bind_addr = ""
    for mode in attempts:
        if mode == "disabled":
            continue
        try:
            sock, resolved_endpoint, resolved_bind_addr = connect_transport(mode, args)
            resolved_mode = mode
            break
        except Exception as exc:
            connect_errors.append(f"{mode}: {exc}")

    if sock is None:
        detail = "; ".join(connect_errors[-6:])
        raise SystemExit(f"transport connect failed: {detail}")

    started = time.time()
    with sock:
        # HELLO
        send_frame(
            sock,
            TBP_MSG_HELLO,
            wire_session_id,
            0,
            seq,
            payload_text="node_role=replay-buffer;capabilities=bulk,progressive,multistream",
        )
        seq += 1

        # SESSION_START
        send_frame(
            sock,
            TBP_MSG_SESSION_START,
            wire_session_id,
            0,
            seq,
            payload_text=(
                f"mode={args.mode};artifact={artifact_name};bytes={total_bytes};"
                f"remote_nodes=1;execution_mode={args.execution_mode};streams=1;"
                f"ack_required={1 if args.ack_required else 0};balance=roundrobin;"
                f"replay=1;transport_mode={resolved_mode}"
            ),
        )
        seq += 1

        # SEGMENT_BEGIN
        send_frame(
            sock,
            TBP_MSG_KV_SEGMENT_BEGIN,
            wire_session_id,
            stream_id,
            seq,
            payload_text=(
                f"segment_id=0;payload_bytes={total_bytes};"
                f"chunk_bytes_nominal={chunk_bytes};stream_count=1;replay=1"
            ),
        )
        seq += 1

        # CHUNKS
        chunk_seq_start = seq
        for chunk in chunks:
            chunk_seq = seq
            send_frame(
                sock,
                TBP_MSG_KV_CHUNK,
                wire_session_id,
                stream_id,
                chunk_seq,
                payload_bytes=chunk,
            )
            seq += 1

            if args.wait_ack and args.ack_required:
                retries = 0
                while True:
                    frame = recv_tbp_frame(sock, timeout_sec)
                    if frame is None:
                        if retries >= max_retries:
                            raise SystemExit(f"ACK timeout for seq={chunk_seq}")
                        retries += 1
                        retransmit_chunks += 1
                        send_frame(
                            sock,
                            TBP_MSG_KV_CHUNK,
                            wire_session_id,
                            stream_id,
                            chunk_seq,
                            payload_bytes=chunk,
                        )
                        continue

                    msg_type, sid, sid_stream, sid_seq, _flags, payload = frame
                    if msg_type != TBP_MSG_KV_ACK:
                        continue
                    if sid != wire_session_id or sid_stream != stream_id:
                        continue
                    if sid_seq != chunk_seq:
                        continue

                    payload_txt = payload.decode("utf-8", errors="ignore").strip().lower()
                    if payload_txt.startswith("nack="):
                        nack_frames_seen += 1
                        if retries >= max_retries:
                            raise SystemExit(f"NACK retry budget exceeded for seq={chunk_seq}: {payload_txt}")
                        retries += 1
                        retransmit_chunks += 1
                        send_frame(
                            sock,
                            TBP_MSG_KV_CHUNK,
                            wire_session_id,
                            stream_id,
                            chunk_seq,
                            payload_bytes=chunk,
                        )
                        continue

                    ack_frames_seen += 1
                    break

        chunk_count = len(chunks)

        # SEGMENT_END
        send_frame(
            sock,
            TBP_MSG_KV_SEGMENT_END,
            wire_session_id,
            stream_id,
            seq,
            payload_text=f"segment_id=0;chunks_sent={chunk_count}",
        )
        seq += 1

        # DONE
        send_frame(
            sock,
            TBP_MSG_KV_DONE,
            wire_session_id,
            stream_id,
            seq,
            payload_text="done=1;replay=1",
        )
        seq += 1

    ended = time.time()
    out = {
        "ok": True,
        "host": args.host,
        "port": args.port,
        "transport_mode_requested": requested_mode,
        "transport_mode_resolved": resolved_mode,
        "transport_backend": resolved_mode,
        "transport_fallback": bool(args.transport_fallback),
        "endpoint": resolved_endpoint,
        "bind_addr": resolved_bind_addr,
        "mode_attempt_order": attempts,
        "session_id": wire_session_id,
        "stream_id": stream_id,
        "source": str(source_path),
        "artifact_name": artifact_name,
        "chunk_count": chunk_count,
        "chunk_seq_start": chunk_seq_start,
        "bytes_sent": total_bytes,
        "ack_required": args.ack_required,
        "wait_ack": args.wait_ack,
        "ack_frames_seen": ack_frames_seen,
        "nack_frames_seen": nack_frames_seen,
        "retransmit_chunks": retransmit_chunks,
        "duration_s": max(0.0, ended - started),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
