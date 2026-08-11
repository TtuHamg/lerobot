#!/usr/bin/env python3
"""
SSH WebSocket 隧道 - Server 端

一条 SSH session 可以跨多次 WebSocket 连接恢复。WebSocket 被代理硬切时，
只解绑当前 WS，不关闭后端 sshd socket，等待 client 使用 session_id 恢复。
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path

import websockets
from websockets.server import serve

from ws_ssh_reliable import PendingBuffer, SeqReceiver, SequenceGapError, dumps_message, loads_message

LOG_FILE = Path(__file__).with_suffix(".log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

WS_HOST = "0.0.0.0"
WS_PORT = 15590
SSH_HOST = "127.0.0.1"
SSH_PORT = 220
WS_PATH = "/api/v1/live/events"

PING_INTERVAL = None
PING_TIMEOUT = None
APP_HEARTBEAT_INTERVAL = 2
HEARTBEAT_TIMEOUT = 15
BUF_SIZE = 64 * 1024
MAX_PENDING_BYTES = 32 * 1024 * 1024
SESSION_RESUME_DEADLINE = 10 * 60
CLEANUP_INTERVAL = 15

sessions: dict[str, "ServerSession"] = {}
sessions_lock = asyncio.Lock()


def _peer(ws):
    try:
        addr = getattr(ws, "remote_address", None)
        if addr and isinstance(addr, (list, tuple)) and len(addr) >= 2:
            return f"{addr[0]}:{addr[1]}"
        if addr:
            return str(addr)
        req = getattr(ws, "request", None)
        return str(getattr(req, "client", "")) if req else "?"
    except Exception:
        return "?"


class ServerSession:
    def __init__(self, session_id: str, ssh_reader: asyncio.StreamReader, ssh_writer: asyncio.StreamWriter):
        self.session_id = session_id
        self.ssh_reader = ssh_reader
        self.ssh_writer = ssh_writer
        self.to_client = PendingBuffer(MAX_PENDING_BYTES)
        self.from_client = SeqReceiver()
        self.current_ws = None
        self.current_peer = ""
        self.send_lock = asyncio.Lock()
        self.closed = False
        self.close_reason = ""
        self.created_at = time.monotonic()
        self.detached_at: float | None = None
        self.last_activity_at = self.created_at
        self.last_rx_at = self.created_at
        self.ws_to_ssh_bytes = 0
        self.ssh_to_ws_bytes = 0
        self.resume_count = 0
        self.ssh_task = asyncio.create_task(self._read_ssh())

    async def attach(self, ws, peer_info: str, client_received_seq: int) -> None:
        self.to_client.ack(client_received_seq)
        old_ws = self.current_ws
        if old_ws is not None and old_ws is not ws:
            with contextlib.suppress(Exception):
                await old_ws.close(4010, "replaced by resumed websocket")
        resent = 0
        async with self.send_lock:
            self.current_ws = ws
            self.current_peer = peer_info
            self.detached_at = None
            self.resume_count += 1
            now = time.monotonic()
            self.last_activity_at = now
            self.last_rx_at = now
            await self._send_to_ws_locked(
                ws,
                {
                    "type": "hello_ack",
                    "session_id": self.session_id,
                    "received_seq": self.from_client.committed_seq,
                    "resume_count": self.resume_count,
                },
            )
            for seq, payload in self.to_client.chunks_after(client_received_seq):
                if not await self._send_to_ws_locked(ws, {"type": "data", "seq": seq, "payload": payload}):
                    break
                self.to_client.mark_sent(seq)
                resent += 1
        log.info(
            "session attached id=%s peer=%s resume_count=%s client_received_seq=%s resent=%s pending_to_client=%sB",
            self.session_id,
            peer_info,
            self.resume_count,
            client_received_seq,
            resent,
            self.to_client.pending_bytes,
        )

    async def detach(self, ws, reason: str) -> None:
        if self.current_ws is ws:
            self.current_ws = None
            self.detached_at = time.monotonic()
            log.info(
                "session detached id=%s peer=%s reason=%s pending_to_client=%sB pending_from_client_ack=%s",
                self.session_id,
                self.current_peer,
                reason,
                self.to_client.pending_bytes,
                self.from_client.committed_seq,
            )

    async def handle_message(self, ws, message: dict) -> None:
        msg_type = message.get("type")
        now = time.monotonic()
        self.last_activity_at = now
        self.last_rx_at = now
        if msg_type == "data":
            await self._handle_client_data(ws, message)
        elif msg_type == "ack":
            self.to_client.ack(int(message.get("seq", 0)))
        elif msg_type == "ping":
            await self._send_to_ws(ws, {"type": "pong", "ts": message.get("ts")})
        elif msg_type == "pong":
            return
        elif msg_type == "close":
            await self.close(f"client close: {message.get('reason', '')}", notify=False)
        else:
            log.warning("未知消息 id=%s type=%r", self.session_id, msg_type)

    async def _handle_client_data(self, ws, message: dict) -> None:
        seq = int(message["seq"])
        payload = message["payload"]
        try:
            should_write = self.from_client.should_commit(seq)
        except SequenceGapError as e:
            log.error("client seq gap id=%s %s", self.session_id, e)
            await self.close(str(e), notify=True)
            return
        if should_write:
            self.ssh_writer.write(payload)
            await self.ssh_writer.drain()
            self.from_client.commit(seq)
            self.ws_to_ssh_bytes += len(payload)
        await self._send_to_ws(ws, {"type": "ack", "seq": self.from_client.committed_seq})

    async def _read_ssh(self) -> None:
        try:
            while not self.closed:
                data = await self.ssh_reader.read(BUF_SIZE)
                if not data:
                    log.info("SSH 对端关闭 id=%s", self.session_id)
                    await self.close("ssh eof", notify=True)
                    return
                try:
                    seq = self.to_client.add(data)
                except BufferError as e:
                    log.error("server pending buffer full id=%s %s", self.session_id, e)
                    await self.close(str(e), notify=True)
                    return
                self.ssh_to_ws_bytes += len(data)
                ws = self.current_ws
                if ws is not None:
                    if await self._send_to_ws(ws, {"type": "data", "seq": seq, "payload": data}):
                        self.to_client.mark_sent(seq)
        except (ConnectionResetError, BrokenPipeError) as e:
            log.info("SSH socket closed id=%s %s", self.session_id, type(e).__name__)
            await self.close(type(e).__name__, notify=True)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("SSH reader crashed id=%s", self.session_id)
            await self.close("ssh reader crashed", notify=True)

    async def _send_to_ws(self, ws, message: dict) -> bool:
        if self.closed and message.get("type") != "close":
            return False
        async with self.send_lock:
            return await self._send_to_ws_locked(ws, message)

    async def _send_to_ws_locked(self, ws, message: dict) -> bool:
        if self.current_ws is not ws and message.get("type") != "close":
            return False
        try:
            await ws.send(dumps_message(message))
            self.last_activity_at = time.monotonic()
            return True
        except websockets.ConnectionClosed as e:
            await self.detach(ws, f"send failed code={e.code}")
            return False

    async def heartbeat(self, ws) -> None:
        try:
            while self.current_ws is ws and not self.closed:
                await asyncio.sleep(APP_HEARTBEAT_INTERVAL)
                if self.current_ws is not ws or self.closed:
                    return
                if time.monotonic() - self.last_rx_at > HEARTBEAT_TIMEOUT:
                    log.info("heartbeat timeout id=%s peer=%s", self.session_id, self.current_peer)
                    with contextlib.suppress(Exception):
                        ws.transport.close()
                    await self.detach(ws, "heartbeat timeout")
                    return
                if not await self._send_to_ws(ws, {"type": "ping", "ts": time.time()}):
                    return
        except asyncio.CancelledError:
            pass

    async def close(self, reason: str, notify: bool) -> None:
        if self.closed:
            return
        self.closed = True
        self.close_reason = reason
        ws = self.current_ws
        if notify and ws is not None:
            with contextlib.suppress(Exception):
                await self._send_to_ws(ws, {"type": "close", "reason": reason})
        with contextlib.suppress(Exception):
            self.ssh_writer.close()
            await self.ssh_writer.wait_closed()
        if asyncio.current_task() is not self.ssh_task:
            self.ssh_task.cancel()
        duration = time.monotonic() - self.created_at
        log.info(
            "session closed id=%s reason=%s duration=%.1fs ws->ssh=%sB ssh->ws=%sB resumes=%s",
            self.session_id,
            reason,
            duration,
            self.ws_to_ssh_bytes,
            self.ssh_to_ws_bytes,
            self.resume_count,
        )


async def get_or_create_session(session_id: str, resume: bool) -> ServerSession | None:
    async with sessions_lock:
        session = sessions.get(session_id)
        if session is not None and session.closed:
            sessions.pop(session_id, None)
            session = None
    if session is not None:
        return session
    if resume:
        return None

    ssh_reader, ssh_writer = await asyncio.wait_for(asyncio.open_connection(SSH_HOST, SSH_PORT), timeout=10.0)
    session = ServerSession(session_id, ssh_reader, ssh_writer)
    async with sessions_lock:
        existing = sessions.get(session_id)
        if existing is not None:
            await session.close("duplicate session race", notify=False)
            return existing
        sessions[session_id] = session
    log.info("session created id=%s -> SSH %s:%s", session_id, SSH_HOST, SSH_PORT)
    return session


async def cleanup_sessions() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        now = time.monotonic()
        stale: list[tuple[str, ServerSession, str]] = []
        async with sessions_lock:
            for session_id, session in list(sessions.items()):
                if session.closed:
                    stale.append((session_id, session, session.close_reason or "closed"))
                elif session.current_ws is None and session.detached_at is not None:
                    detached_for = now - session.detached_at
                    if detached_for > SESSION_RESUME_DEADLINE:
                        stale.append((session_id, session, f"resume timeout {detached_for:.1f}s"))
            for session_id, _, _ in stale:
                sessions.pop(session_id, None)
        for session_id, session, reason in stale:
            await session.close(reason, notify=False)
            log.info("session removed id=%s reason=%s", session_id, reason)


async def handler(ws, path=None):
    peer_info = str(_peer(ws))
    if path is None:
        path = getattr(ws, "path", "") or getattr(getattr(ws, "request", None), "path", "") or ""
    if path.rstrip("/") != WS_PATH.rstrip("/"):
        log.warning("拒绝连接 [%s] 路径不匹配 path=%r", peer_info, path)
        await ws.close(4004, "Not Found")
        return

    session = None
    heartbeat_task = None
    try:
        raw_hello = await asyncio.wait_for(ws.recv(), timeout=10.0)
        hello = loads_message(raw_hello)
        if hello.get("type") != "hello":
            await ws.close(4400, "expected hello")
            return
        session_id = str(hello.get("session_id") or "")
        if not session_id:
            await ws.close(4400, "missing session_id")
            return
        resume = bool(hello.get("resume", False))
        client_received_seq = int(hello.get("received_seq", 0))
        session = await get_or_create_session(session_id, resume=resume)
        if session is None:
            log.warning("resume requested for missing session id=%s peer=%s", session_id, peer_info)
            await ws.send(dumps_message({"type": "close", "reason": "session_not_found"}))
            await ws.close(4404, "session_not_found")
            return

        await session.attach(ws, peer_info, client_received_seq)
        heartbeat_task = asyncio.create_task(session.heartbeat(ws))
        async for raw in ws:
            message = loads_message(raw)
            await session.handle_message(ws, message)
            if session.closed:
                break
    except websockets.ConnectionClosed as e:
        if session is not None:
            await session.detach(ws, f"ws closed code={e.code} reason={e.reason or ''}")
    except (asyncio.TimeoutError, ValueError, KeyError, TypeError) as e:
        log.warning("连接协议错误 [%s] %s", peer_info, e)
        with contextlib.suppress(Exception):
            await ws.close(4400, str(e)[:120])
    except (ConnectionRefusedError, OSError) as e:
        log.error("连接本地 SSH 失败 [%s] %s", peer_info, e)
        with contextlib.suppress(Exception):
            await ws.close(4500, str(e)[:120])
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if session is not None:
            await session.detach(ws, "handler exit")


async def main():
    cleanup_task = asyncio.create_task(cleanup_sessions())
    try:
        async with serve(
            handler,
            WS_HOST,
            WS_PORT,
            ping_interval=PING_INTERVAL,
            ping_timeout=PING_TIMEOUT,
            max_size=2**22,
            close_timeout=5,
        ):
            log.info(
                "WebSocket 监听 ws://%s:%s%s -> SSH %s:%s resume_timeout=%ss",
                WS_HOST,
                WS_PORT,
                WS_PATH,
                SSH_HOST,
                SSH_PORT,
                SESSION_RESUME_DEADLINE,
            )
            await asyncio.Future()
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())