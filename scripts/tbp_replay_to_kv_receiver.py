#!/usr/bin/env python3
"""
Replay a captured KV artifact (or chunk directory) to the ik KV receiver over TBP.

This is intended for single-machine buffered E2E validation:
1) prefill sender -> loopback disk buffer
2) replay disk buffer -> ik llama-server /kv-receiver
"""

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay buffered KV stream to ik kv-receiver")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact", help="path to KV artifact file")
    src.add_argument("--chunks-dir", help="directory containing seq_*.bin chunk files")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19001)
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

    started = time.time()
    with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
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
                f"ack_required={1 if args.ack_required else 0};balance=roundrobin;replay=1"
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
