#!/usr/bin/env python3
"""
SSH WebSocket 隧道 - Client 端

运行在 Mac 本地，将远程 HTTPS URL 映射到本机 TCP 端口，供 ssh/Cursor 连接。
WebSocket 断开时会用 session_id 恢复同一条 SSH 字节流。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import fcntl
import logging
import os
import plistlib
import secrets
import shlex
import shutil
import ssl
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import websockets

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="websockets.client.connect is deprecated",
        category=DeprecationWarning,
    )
    from websockets.client import connect

from ws_ssh_reliable import (
    WS_SSH_COOKIE,
    PendingBuffer,
    SeqReceiver,
    SequenceGapError,
    dumps_message,
    loads_message,
)

LOG_FILE = Path(__file__).with_suffix(".log")
LOCK_FILE = Path(__file__).with_suffix(".lock")
TMUX_SESSION_NAME = "ws_ssh_client"
LAUNCH_AGENT_LABEL = "local.ws-ssh-client"
LAUNCH_AGENT_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"

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

REMOTE_WS_URL = os.environ.get(
    "WS_SSH_REMOTE_URL",
    "wss://kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com/api/v1/live/events",
)
SKIP_SSL_VERIFY = os.environ.get("WS_SSH_SKIP_SSL_VERIFY", "1") != "0"
LOCAL_PORT = int(os.environ.get("WS_SSH_LOCAL_PORT", "2222"))
LOCAL_HOST = os.environ.get("WS_SSH_LOCAL_HOST", "127.0.0.1")

APP_HEARTBEAT_INTERVAL = 2
HEARTBEAT_TIMEOUT = 15
BUF_SIZE = 64 * 1024
MAX_PENDING_BYTES = 32 * 1024 * 1024
SESSION_RESUME_DEADLINE = 10 * 60
RECONNECT_DELAYS = (0.0, 0.1, 0.3, 1.0, 2.0)


def _load_cookie() -> str:
    return WS_SSH_COOKIE.strip()


COOKIE = _load_cookie()
COOKIE_PROVIDER: Callable[[], str] | None = None
COOKIE_REFRESH_INTERVAL_S = 60.0
_cookie_refreshed_at = 0.0

EXTRA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}
if COOKIE:
    EXTRA_HEADERS["Cookie"] = COOKIE


def set_cookie_provider(provider: Callable[[], str]) -> None:
    global COOKIE_PROVIDER, _cookie_refreshed_at
    COOKIE_PROVIDER = provider
    _cookie_refreshed_at = 0.0


def _connection_headers() -> dict[str, str]:
    global COOKIE, _cookie_refreshed_at
    now = time.monotonic()
    if COOKIE_PROVIDER is not None and now - _cookie_refreshed_at >= COOKIE_REFRESH_INTERVAL_S:
        try:
            refreshed = COOKIE_PROVIDER().strip()
            if not refreshed:
                raise ValueError("cookie provider returned an empty value")
            COOKIE = refreshed
            _cookie_refreshed_at = now
            log.info("已从本机浏览器刷新 KML SSO Cookie。")
        except Exception as error:
            log.warning("刷新 KML SSO Cookie 失败：%s", error)
    headers = dict(EXTRA_HEADERS)
    if COOKIE:
        headers["Cookie"] = COOKIE
    else:
        headers.pop("Cookie", None)
    return headers


def _invalidate_cookie_on_sso_redirect(error: BaseException) -> None:
    global _cookie_refreshed_at
    message = str(error)
    if "sso.corp.kuaishou.com" in message or "accessproxy_sso_callback" in message:
        _cookie_refreshed_at = 0.0


def _browser_cookie_provider(browser: str = "auto") -> str:
    tools_dir = Path(__file__).resolve().parents[2] / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from start_kml_tunnel_client import resolve_browser

    target = urlsplit(REMOTE_WS_URL)
    _, cookie = resolve_browser(
        browser,
        str(target.hostname),
        target.path or "/",
        target.scheme == "wss",
        "Default",
    )
    return cookie


def acquire_instance_lock(lock_path=LOCK_FILE):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115 - held for process lifetime
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        lock_file.close()
        if e.errno in (errno.EACCES, errno.EAGAIN):
            return None
        raise
    lock_file.write(f"{os.getpid()}\n")
    lock_file.truncate()
    lock_file.flush()
    return lock_file


def tmux_startup_shell_command(script_path=None, python_executable=None):
    script_path = Path(script_path) if script_path is not None else Path(__file__).resolve()
    python_executable = python_executable or sys.executable
    run_command = shlex.join([python_executable, str(script_path)])
    log_path = script_path.with_suffix(".log")
    return (
        f"{run_command}; "
        "status=$?; "
        "echo; "
        f"echo {shlex.quote('ws_ssh_client 启动命令已退出，继续显示日志。按 Ctrl-C 可停止看日志。')}; "
        f"echo {shlex.quote(f'日志文件: {log_path}')}; "
        f"exec tail -n 80 -f {shlex.quote(str(log_path))}"
    )


def tmux_executable():
    for candidate in (
        shutil.which("tmux"),
        "/opt/homebrew/bin/tmux",
        "/usr/local/bin/tmux",
        "/usr/bin/tmux",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def tmux_session_exists(session_name=TMUX_SESSION_NAME, tmux_bin="tmux"):
    result = subprocess.run(
        [tmux_bin, "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def ensure_tmux_session():
    tmux_bin = tmux_executable()
    if tmux_bin is None:
        raise FileNotFoundError("tmux")
    if tmux_session_exists(tmux_bin=tmux_bin):
        print(f"tmux session 已存在: {TMUX_SESSION_NAME}")
        print(f"查看: tmux attach -t {TMUX_SESSION_NAME}")
        return 0
    subprocess.run([tmux_bin, "new-session", "-d", "-s", TMUX_SESSION_NAME], check=True)
    subprocess.run(
        [tmux_bin, "send-keys", "-t", TMUX_SESSION_NAME, tmux_startup_shell_command(), "C-m"],
        check=True,
    )
    print(f"tmux session 已启动: {TMUX_SESSION_NAME}")
    print(f"查看: tmux attach -t {TMUX_SESSION_NAME}")
    return 0


def launch_agent_plist(script_path=None, python_executable=None):
    script_path = Path(script_path) if script_path is not None else Path(__file__).resolve()
    python_executable = python_executable or sys.executable
    plist = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": [python_executable, str(script_path), "--tmux"],
        "RunAtLoad": True,
        "WorkingDirectory": str(script_path.parent),
        "StandardOutPath": str(LOG_FILE.with_name("ws_ssh_client.launchd.out.log")),
        "StandardErrorPath": str(LOG_FILE.with_name("ws_ssh_client.launchd.err.log")),
    }
    return plistlib.dumps(plist, sort_keys=False).decode("utf-8")


def install_launch_agent():
    LAUNCH_AGENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_AGENT_PATH.write_text(launch_agent_plist(), encoding="utf-8")
    print(f"LaunchAgent 已写入: {LAUNCH_AGENT_PATH}")
    print("下次登录时会自动确保 tmux session 存在。")
    print(f"现在也可以手动启动: python3 {Path(__file__).resolve()} --tmux")
    return 0


def _ssl_context():
    if not SKIP_SSL_VERIFY:
        return True
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _connect_ssl_arg(url: str):
    return _ssl_context() if url.lower().startswith("wss://") else None


class TunnelSession:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, local_peer: str):
        self.reader = reader
        self.writer = writer
        self.local_peer = local_peer
        self.session_id = secrets.token_hex(16)
        self.to_server = PendingBuffer(MAX_PENDING_BYTES)
        self.from_server = SeqReceiver()
        self.current_ws = None
        self.send_lock = asyncio.Lock()
        self.close_event = asyncio.Event()
        self.closed = False
        self.started_at = time.monotonic()
        self.detached_at: float | None = None
        self.last_rx_at = self.started_at
        self.tcp_to_ws_bytes = 0
        self.ws_to_tcp_bytes = 0
        self.reconnect_count = 0

    async def run(self):
        log.info("session start [%s] id=%s", self.local_peer, self.session_id)
        tcp_task = asyncio.create_task(self._read_tcp())
        ws_task = asyncio.create_task(self._connect_loop())
        done, pending = await asyncio.wait([tcp_task, ws_task], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*done, return_exceptions=True)
        await self.close("session run finished", notify=True)

    async def _read_tcp(self):
        try:
            while not self.closed:
                data = await self.reader.read(BUF_SIZE)
                if not data:
                    await self.close("local tcp eof", notify=True)
                    return
                try:
                    seq = self.to_server.add(data)
                except BufferError as e:
                    log.error("client pending buffer full [%s] %s", self.local_peer, e)
                    await self.close(str(e), notify=True)
                    return
                self.tcp_to_ws_bytes += len(data)
                if await self._send_current({"type": "data", "seq": seq, "payload": data}):
                    self.to_server.mark_sent(seq)
        except (ConnectionResetError, BrokenPipeError):
            await self.close("local tcp reset", notify=True)
        except asyncio.CancelledError:
            pass

    async def _connect_loop(self):
        attempt = 0
        resume = False
        while not self.closed:
            if self.detached_at is not None and time.monotonic() - self.detached_at > SESSION_RESUME_DEADLINE:
                await self.close("resume deadline exceeded", notify=False)
                return
            delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
            if delay:
                await asyncio.sleep(delay)
            try:
                async with connect(
                    REMOTE_WS_URL,
                    extra_headers=_connection_headers(),
                    ping_interval=None,
                    ping_timeout=None,
                    max_size=2**22,
                    close_timeout=5,
                    ssl=_connect_ssl_arg(REMOTE_WS_URL),
                ) as ws:
                    await ws.send(
                        dumps_message(
                            {
                                "type": "hello",
                                "session_id": self.session_id,
                                "resume": resume,
                                "received_seq": self.from_server.committed_seq,
                            }
                        )
                    )
                    raw_ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    hello_ack = loads_message(raw_ack)
                    if hello_ack.get("type") == "close":
                        await self.close(str(hello_ack.get("reason", "server closed")), notify=False)
                        return
                    if hello_ack.get("type") != "hello_ack":
                        raise RuntimeError(f"expected hello_ack, got {hello_ack!r}")
                    server_received_seq = int(hello_ack.get("received_seq", 0))
                    self.to_server.ack(server_received_seq)
                    self.reconnect_count += 1 if resume else 0
                    self.detached_at = None
                    attempt = 0
                    self.last_rx_at = time.monotonic()
                    log.info(
                        "ws attached [%s] id=%s resume=%s server_received_seq=%s local_received_seq=%s pending_to_server=%sB",
                        self.local_peer,
                        self.session_id,
                        resume,
                        server_received_seq,
                        self.from_server.committed_seq,
                        self.to_server.pending_bytes,
                    )
                    if not await self._attach_and_replay(ws, server_received_seq):
                        raise ConnectionError("failed to replay pending data")
                    heartbeat_task = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for raw in ws:
                            await self._handle_ws_message(ws, loads_message(raw))
                            if self.closed:
                                return
                    finally:
                        heartbeat_task.cancel()
                        await asyncio.gather(heartbeat_task, return_exceptions=True)
            except asyncio.CancelledError:
                return
            except Exception as e:
                if self.closed:
                    return
                if self.detached_at is None:
                    self.detached_at = time.monotonic()
                _invalidate_cookie_on_sso_redirect(e)
                log.info(
                    "ws detached [%s] id=%s attempt=%s error=%s", self.local_peer, self.session_id, attempt, e
                )
                self.current_ws = None
                resume = True
                attempt += 1

    async def _handle_ws_message(self, ws, message: dict):
        self.last_rx_at = time.monotonic()
        msg_type = message.get("type")
        if msg_type == "data":
            await self._handle_server_data(ws, message)
        elif msg_type == "ack":
            self.to_server.ack(int(message.get("seq", 0)))
        elif msg_type == "ping":
            await self._send_to_ws(ws, {"type": "pong", "ts": message.get("ts")})
        elif msg_type == "pong":
            return
        elif msg_type == "close":
            await self.close(f"server close: {message.get('reason', '')}", notify=False)
        else:
            log.warning("未知消息 [%s] type=%r", self.local_peer, msg_type)

    async def _handle_server_data(self, ws, message: dict):
        seq = int(message["seq"])
        payload = message["payload"]
        try:
            should_write = self.from_server.should_commit(seq)
        except SequenceGapError as e:
            log.error("server seq gap [%s] %s", self.local_peer, e)
            await self.close(str(e), notify=True)
            return
        if should_write:
            self.writer.write(payload)
            await self.writer.drain()
            self.from_server.commit(seq)
            self.ws_to_tcp_bytes += len(payload)
        await self._send_to_ws(ws, {"type": "ack", "seq": self.from_server.committed_seq})

    async def _attach_and_replay(self, ws, server_received_seq: int) -> bool:
        resent = 0
        async with self.send_lock:
            self.current_ws = ws
            for seq, payload in self.to_server.chunks_after(server_received_seq):
                if not await self._send_to_ws_locked(ws, {"type": "data", "seq": seq, "payload": payload}):
                    return False
                self.to_server.mark_sent(seq)
                resent += 1
        if resent:
            log.info("replayed [%s] id=%s chunks=%s", self.local_peer, self.session_id, resent)
        return True

    async def _send_current(self, message: dict) -> bool:
        ws = self.current_ws
        if ws is None:
            return False
        return await self._send_to_ws(ws, message)

    async def _send_to_ws(self, ws, message: dict) -> bool:
        async with self.send_lock:
            return await self._send_to_ws_locked(ws, message)

    async def _send_to_ws_locked(self, ws, message: dict) -> bool:
        if self.current_ws is not ws and message.get("type") != "close":
            return False
        try:
            await ws.send(dumps_message(message))
            return True
        except websockets.ConnectionClosed:
            if self.detached_at is None:
                self.detached_at = time.monotonic()
            self.current_ws = None
            return False

    async def _heartbeat(self, ws):
        try:
            while self.current_ws is ws and not self.closed:
                await asyncio.sleep(APP_HEARTBEAT_INTERVAL)
                if self.current_ws is not ws or self.closed:
                    return
                if time.monotonic() - self.last_rx_at > HEARTBEAT_TIMEOUT:
                    log.info("heartbeat timeout [%s] id=%s", self.local_peer, self.session_id)
                    with contextlib.suppress(Exception):
                        ws.transport.close()
                    if self.detached_at is None:
                        self.detached_at = time.monotonic()
                    self.current_ws = None
                    return
                if not await self._send_to_ws(ws, {"type": "ping", "ts": time.time()}):
                    return
        except asyncio.CancelledError:
            pass

    async def close(self, reason: str, notify: bool):
        if self.closed:
            return
        self.closed = True
        self.close_event.set()
        ws = self.current_ws
        if notify and ws is not None:
            with contextlib.suppress(Exception):
                await self._send_to_ws(ws, {"type": "close", "reason": reason})
        with contextlib.suppress(Exception):
            self.writer.close()
            await self.writer.wait_closed()
        duration = time.monotonic() - self.started_at
        log.info(
            "session end [%s] id=%s reason=%s duration=%.1fs tcp->ws=%sB ws->tcp=%sB reconnects=%s",
            self.local_peer,
            self.session_id,
            reason,
            duration,
            self.tcp_to_ws_bytes,
            self.ws_to_tcp_bytes,
            self.reconnect_count,
        )


async def handle_local(reader, writer):
    peer = writer.get_extra_info("peername")
    local_peer = f"{peer[0]}:{peer[1]}" if peer and len(peer) >= 2 else (str(peer) if peer else "?")
    session = TunnelSession(reader, writer, local_peer)
    await session.run()


async def main():
    if not COOKIE and COOKIE_PROVIDER is None:
        log.warning("未设置 WS_SSH_COOKIE；如果远端需要登录态，WebSocket 握手会失败。")
    server = await asyncio.start_server(handle_local, LOCAL_HOST, LOCAL_PORT)
    log.info("本地监听 %s:%s -> %s", LOCAL_HOST, LOCAL_PORT, REMOTE_WS_URL)
    log.info("使用: ssh -p %s <用户>@127.0.0.1", LOCAL_PORT)
    async with server:
        await server.serve_forever()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SSH WebSocket 隧道客户端")
    parser.add_argument("--tmux", action="store_true", help="确保脚本在固定 tmux session 中运行")
    parser.add_argument("--install-launch-agent", action="store_true", help="写入 macOS LaunchAgent")
    parser.add_argument(
        "--browser-cookie",
        action="store_true",
        help="从本机已登录浏览器读取 KML Cookie，并在 SSO 过期后自动刷新",
    )
    parser.add_argument(
        "--browser",
        choices=("auto", "chrome", "edge", "chromium", "brave", "firefox"),
        default="auto",
        help="--browser-cookie 使用的浏览器（默认 auto）",
    )
    return parser.parse_args(argv)


def cli(argv=None):
    args = parse_args(argv)
    if args.browser_cookie:
        set_cookie_provider(lambda: _browser_cookie_provider(args.browser))
    if args.install_launch_agent:
        return install_launch_agent()
    if args.tmux:
        try:
            return ensure_tmux_session()
        except FileNotFoundError:
            print("未找到 tmux，请先安装 tmux。", file=sys.stderr)
            return 1

    lock_file = acquire_instance_lock()
    if lock_file is None:
        log.info("已有 ws_ssh_client 实例在运行，本次退出。")
        return 0
    try:
        asyncio.run(main())
    finally:
        lock_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(cli())
