#!/usr/bin/env python3
"""Small reliability helpers for the SSH-over-WebSocket tunnel."""

from __future__ import annotations

import base64
import json
import os
from collections import OrderedDict
from typing import Any

PROTOCOL_VERSION = 1

# Supply the SSO cookie at runtime; never store credentials in source control.
WS_SSH_COOKIE = os.environ.get("WS_SSH_COOKIE", "").strip()


class SequenceGapError(RuntimeError):
    """Raised when a stream receives data out of order."""


class PendingBuffer:
    """Tracks outbound chunks until the peer confirms them with an ack."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.next_seq = 1
        self.sent_seq = 0
        self.acked_seq = 0
        self.pending_bytes = 0
        self._chunks: OrderedDict[int, bytes] = OrderedDict()

    def add(self, payload: bytes) -> int:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if self.pending_bytes + len(payload) > self.max_bytes:
            raise BufferError(
                f"pending buffer limit exceeded: {self.pending_bytes + len(payload)} > {self.max_bytes}"
            )
        seq = self.next_seq
        self.next_seq += 1
        self._chunks[seq] = payload
        self.pending_bytes += len(payload)
        return seq

    def ack(self, seq: int) -> None:
        if seq <= self.acked_seq:
            return
        if seq > self.sent_seq:
            raise ValueError(f"ack for unsent seq {seq}; sent_seq={self.sent_seq}")
        self.acked_seq = seq
        for chunk_seq in list(self._chunks.keys()):
            if chunk_seq > seq:
                break
            self.pending_bytes -= len(self._chunks.pop(chunk_seq))

    def mark_sent(self, seq: int) -> None:
        if seq <= self.sent_seq:
            return
        if seq >= self.next_seq:
            raise ValueError(f"sent unknown seq {seq}; next_seq={self.next_seq}")
        if seq != self.sent_seq + 1:
            raise SequenceGapError(f"expected to send seq {self.sent_seq + 1}, got {seq}")
        self.sent_seq = seq

    def chunks_after(self, seq: int) -> list[tuple[int, bytes]]:
        return [(chunk_seq, payload) for chunk_seq, payload in self._chunks.items() if chunk_seq > seq]


class SeqReceiver:
    """Accepts in-order chunks and ignores duplicates after reconnect replay."""

    def __init__(self):
        self.committed_seq = 0

    def should_commit(self, seq: int) -> bool:
        if seq <= self.committed_seq:
            return False
        if seq != self.committed_seq + 1:
            raise SequenceGapError(f"expected seq {self.committed_seq + 1}, got {seq}")
        return True

    def commit(self, seq: int) -> None:
        if seq != self.committed_seq + 1:
            raise SequenceGapError(f"expected seq {self.committed_seq + 1}, got {seq}")
        self.committed_seq = seq


def dumps_message(message: dict[str, Any]) -> str:
    payload = dict(message)
    if isinstance(payload.get("payload"), bytes):
        payload["payload_b64"] = base64.b64encode(payload.pop("payload")).decode("ascii")
    payload.setdefault("v", PROTOCOL_VERSION)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def loads_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    message = json.loads(raw)
    if message.get("v", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {message.get('v')}")
    if "payload_b64" in message:
        message["payload"] = base64.b64decode(message.pop("payload_b64").encode("ascii"))
    message.pop("v", None)
    return message
