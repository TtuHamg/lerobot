#!/usr/bin/env python
"""Small TCP-over-WebSocket tunnel for testing LeRobot gRPC through web gateways.

Server side, on the policy machine:
    python tools/ws_tcp_tunnel.py server --listen-port 15174 --target-port 15173

Client side, on the robot machine:
    python tools/ws_tcp_tunnel.py client --listen-port 8080 --ws-url ws://HOST/ws

Then point the gRPC client at 127.0.0.1:8080.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable
from email.utils import formatdate
from http.cookies import CookieError, SimpleCookie
from typing import Any, NamedTuple
from urllib.parse import urljoin, urlsplit, urlunsplit

from aiohttp import ClientSession, CookieJar, DummyCookieJar, WSMsgType, WSServerHandshakeError, web
from yarl import URL


LOGGER = logging.getLogger("ws_tcp_tunnel")
_AUTH_REFRESH_COOLDOWN_S = 30.0
_WEBDRIVER_START_TIMEOUT_S = 15.0


class _BrowserCookie(NamedTuple):
    name: str
    value: str
    domain: str
    path: str
    secure: bool
    http_only: bool
    expiry: float | None
    host_only: bool


class _BrowserAuthResult(NamedTuple):
    cookies: tuple[_BrowserCookie, ...]
    user_agent: str | None


def _redact_url_for_log(url: object) -> str:
    """Return a URL without user info, query parameters, or fragments."""
    try:
        parsed = urlsplit(str(url))
        if not parsed.scheme or not parsed.hostname:
            return "<invalid-url>"
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        path = parsed.path or "/"
        if path not in {"/", "/ws"}:
            path = "/<redacted-path>"
        return urlunsplit((parsed.scheme, host, path, "", ""))
    except (TypeError, ValueError):
        return "<invalid-url>"


def _http_url_from_ws_url(ws_url: str) -> str:
    parsed = urlsplit(ws_url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("--ws-url must be an absolute ws:// or wss:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--ws-url must not contain user information")
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunsplit((scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def _looks_like_sso_url(url: object) -> bool:
    try:
        parsed = urlsplit(str(url))
    except (TypeError, ValueError):
        return False
    hostname = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    return (
        hostname.startswith("sso.")
        or "/cas/login" in path
        or "accessproxy_sso_callback" in path
    )


def _classify_sso_handshake_error(exc: WSServerHandshakeError) -> bool:
    """Detect an AccessProxy/CAS redirect without exposing its URL or headers."""
    request_info = getattr(exc, "request_info", None)
    for attribute in ("real_url", "url"):
        if request_info is not None and _looks_like_sso_url(getattr(request_info, attribute, None)):
            return True

    history = getattr(exc, "history", ()) or ()
    for response in history:
        response_url = getattr(response, "url", None)
        if _looks_like_sso_url(response_url):
            return True
        location = getattr(response, "headers", {}).get("Location")
        if location and _looks_like_sso_url(urljoin(str(response_url or ""), location)):
            return True

    location = getattr(exc, "headers", {}).get("Location")
    return bool(location and _looks_like_sso_url(location))


def _webdriver_request(
    http: Any,
    method: str,
    url: str,
    *,
    operation: str,
    timeout: float,
    json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        response = http.request(method, url, json=json, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"WebDriver {operation} did not respond") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"WebDriver {operation} returned HTTP {response.status_code} without JSON") from exc

    value = payload.get("value")
    error_code = value.get("error") if isinstance(value, dict) else None
    if response.status_code >= 400 or error_code:
        suffix = f", error={error_code}" if error_code else ""
        raise RuntimeError(f"WebDriver {operation} failed (HTTP {response.status_code}{suffix})")
    return payload


def _target_cookies(raw_cookies: object, target_url: str) -> tuple[_BrowserCookie, ...]:
    if not isinstance(raw_cookies, list):
        return ()

    parsed = urlsplit(target_url)
    target_host = (parsed.hostname or "").lower()
    target_path = parsed.path or "/"
    now = time.time()
    result: list[_BrowserCookie] = []

    for cookie in raw_cookies:
        if not isinstance(cookie, dict):
            continue
        name = cookie.get("name")
        value = cookie.get("value")
        raw_domain = str(cookie.get("domain") or target_host)
        domain = raw_domain.lstrip(".").lower()
        cookie_path = str(cookie.get("path") or "/")
        expiry = cookie.get("expiry")
        domain_matches = target_host == domain or target_host.endswith(f".{domain}")
        path_matches = target_path == cookie_path or target_path.startswith(cookie_path.rstrip("/") + "/")
        expiry_value = float(expiry) if isinstance(expiry, (int, float)) and not isinstance(expiry, bool) else None
        is_expired = expiry_value is not None and expiry_value <= now
        secure_mismatch = bool(cookie.get("secure")) and parsed.scheme != "https"
        if (
            isinstance(name, str)
            and isinstance(value, str)
            and domain_matches
            and path_matches
            and not is_expired
            and not secure_mismatch
        ):
            result.append(
                _BrowserCookie(
                    name=name,
                    value=value,
                    domain=domain,
                    path=cookie_path,
                    secure=bool(cookie.get("secure")),
                    http_only=bool(cookie.get("httpOnly")),
                    expiry=expiry_value,
                    # WebDriver does not expose hostOnly explicitly. A leading dot is the
                    # only portable signal; treating an exact target domain as host-only
                    # safely narrows ambiguous cookie scope.
                    host_only=not raw_domain.startswith(".") and domain == target_host,
                )
            )
    return tuple(result)


def _load_browser_cookies(cookie_jar: CookieJar, cookies: tuple[_BrowserCookie, ...], target_url: str) -> None:
    """Load browser cookies without flattening their network-relevant attributes."""
    cookie_jar.clear()
    response_url = URL(target_url)
    for browser_cookie in cookies:
        cookie = SimpleCookie()
        try:
            cookie[browser_cookie.name] = browser_cookie.value
        except CookieError as exc:
            raise RuntimeError("Firefox returned a cookie with an invalid name") from exc
        morsel = cookie[browser_cookie.name]
        morsel["path"] = browser_cookie.path
        if not browser_cookie.host_only:
            morsel["domain"] = browser_cookie.domain
        if browser_cookie.secure:
            morsel["secure"] = True
        if browser_cookie.http_only:
            morsel["httponly"] = True
        if browser_cookie.expiry is not None:
            morsel["expires"] = formatdate(browser_cookie.expiry, usegmt=True)
        cookie_jar.update_cookies(cookie, response_url=response_url)


def _load_cookie_header_from_env(env_name: str, target_url: str) -> tuple[str, int]:
    raw_cookie = os.environ.pop(env_name, None)
    if not raw_cookie:
        raise RuntimeError(f"Cookie environment variable {env_name!r} is not set or is empty")
    if urlsplit(target_url).scheme != "wss":
        raise ValueError("--cookie-env requires a wss:// target URL")

    cookie_parts: list[str] = []
    for part in raw_cookie.split(";"):
        if "=" not in part:
            raise RuntimeError(f"Cookie environment variable {env_name!r} is invalid")
        name, value = (item.strip() for item in part.split("=", 1))
        if not name or any(
            character in ";=" or ord(character) < 0x21 or ord(character) == 0x7F
            for character in name
        ):
            raise RuntimeError(f"Cookie environment variable {env_name!r} is invalid")
        if any(character == ";" or ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise RuntimeError(f"Cookie environment variable {env_name!r} is invalid")
        cookie_parts.append(f"{name}={value}")
    if not cookie_parts:
        raise RuntimeError(f"Cookie environment variable {env_name!r} contains no usable cookies")

    # This header is attached only to ws_connect's first target request. aiohttp removes
    # an explicit Cookie header when a redirect changes origin, so it cannot reach the SSO host.
    return "; ".join(cookie_parts), len(cookie_parts)


def _is_auth_target(current_url: object, target_url: str) -> bool:
    try:
        current = urlsplit(str(current_url))
        target = urlsplit(target_url)
        current_port = current.port or (443 if current.scheme == "https" else 80)
        target_port = target.port or (443 if target.scheme == "https" else 80)
        return (
            current.scheme == target.scheme
            and current.hostname == target.hostname
            and current_port == target_port
            and (current.path or "/") == (target.path or "/")
        )
    except (TypeError, ValueError):
        return False


def _find_executable(value: str | None, default_name: str) -> str:
    if value:
        resolved = shutil.which(value) if os.sep not in value else value
    else:
        resolved = shutil.which(default_name)
    if not resolved or not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Could not find executable {value or default_name!r}")
    return resolved


def _create_firefox_profile_root(driver_path: str) -> str:
    profile_parent: str | None = None
    if driver_path.startswith("/snap/"):
        profile_parent = os.path.expanduser("~/snap/firefox/common")
        if not os.path.isdir(profile_parent) or not os.access(profile_parent, os.W_OK):
            raise RuntimeError(
                "Snap Firefox needs a writable ~/snap/firefox/common directory for its temporary profile"
            )
    try:
        profile_root = tempfile.mkdtemp(prefix="ws_tcp_tunnel-", dir=profile_parent)
        os.chmod(profile_root, 0o700)
        return profile_root
    except OSError as exc:
        raise RuntimeError("Could not create the temporary Firefox profile directory") from exc


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _collect_kml_firefox_auth(
    target_url: str,
    *,
    auth_timeout: float,
    geckodriver: str | None,
    firefox_binary: str | None,
    cancel_event: threading.Event,
) -> _BrowserAuthResult:
    """Run a temporary, visible Firefox session and return target-scoped auth data."""
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError(
            "KML Firefox authentication needs a graphical desktop. "
            "Run this command from a terminal inside your desktop session."
        )

    # Imported lazily so the tunnel's normal unauthenticated mode only needs aiohttp.
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("--auth=kml-firefox requires the Python package 'requests'") from exc

    if cancel_event.is_set():
        raise RuntimeError("KML authentication was cancelled")
    driver_path = _find_executable(geckodriver, "geckodriver")
    browser_binary = _find_executable(firefox_binary, "firefox") if firefox_binary else None
    profile_root = _create_firefox_profile_root(driver_path)
    port = _reserve_loopback_port()
    driver_base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + auth_timeout
    http = requests.Session()
    http.trust_env = False
    process: subprocess.Popen[bytes] | None = None
    session_id: str | None = None

    try:
        try:
            process = subprocess.Popen(  # noqa: S603
                [
                    driver_path,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--profile-root",
                    profile_root,
                    "--log",
                    "fatal",
                ],  # noqa: S607
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError("Could not start geckodriver") from exc
        start_deadline = min(deadline, time.monotonic() + _WEBDRIVER_START_TIMEOUT_S)
        while time.monotonic() < start_deadline:
            if cancel_event.is_set():
                raise RuntimeError("KML authentication was cancelled")
            if process.poll() is not None:
                raise RuntimeError("geckodriver exited before Firefox could start")
            with contextlib.suppress(RuntimeError):
                status = _webdriver_request(
                    http,
                    "GET",
                    f"{driver_base_url}/status",
                    operation="status check",
                    timeout=1.0,
                )
                if isinstance(status.get("value"), dict) and status["value"].get("ready", True):
                    break
            cancel_event.wait(0.1)
        else:
            raise RuntimeError("geckodriver did not become ready")

        if cancel_event.is_set():
            raise RuntimeError("KML authentication was cancelled")
        capabilities: dict[str, Any] = {
            "browserName": "firefox",
            "acceptInsecureCerts": False,
        }
        if browser_binary:
            capabilities["moz:firefoxOptions"] = {"binary": browser_binary}
        created = _webdriver_request(
            http,
            "POST",
            f"{driver_base_url}/session",
            operation="session creation",
            timeout=min(20.0, max(1.0, deadline - time.monotonic())),
            json={"capabilities": {"alwaysMatch": capabilities}},
        )
        created_value = created.get("value")
        if isinstance(created_value, dict):
            session_id = created_value.get("sessionId")
        session_id = session_id or created.get("sessionId")
        if not isinstance(session_id, str):
            raise RuntimeError("WebDriver did not return a Firefox session id")

        if cancel_event.is_set():
            raise RuntimeError("KML authentication was cancelled")
        session_url = f"{driver_base_url}/session/{session_id}"
        _webdriver_request(
            http,
            "POST",
            f"{session_url}/url",
            operation="navigation",
            timeout=min(20.0, max(1.0, deadline - time.monotonic())),
            json={"url": target_url},
        )
        LOGGER.warning(
            "Firefox opened for KML authentication. Complete SSO and MFA in that window; "
            "the tunnel will continue automatically."
        )

        saw_target_url = False
        while time.monotonic() < deadline:
            if cancel_event.is_set():
                raise RuntimeError("KML authentication was cancelled")
            current = _webdriver_request(
                http,
                "GET",
                f"{session_url}/url",
                operation="URL check",
                timeout=min(5.0, max(1.0, deadline - time.monotonic())),
            ).get("value")
            if _is_auth_target(current, target_url):
                saw_target_url = True
                cookies_payload = _webdriver_request(
                    http,
                    "GET",
                    f"{session_url}/cookie",
                    operation="cookie read",
                    timeout=5.0,
                ).get("value")
                cookies = _target_cookies(cookies_payload, target_url)
                if cookies:
                    user_agent: str | None = None
                    try:
                        value = _webdriver_request(
                            http,
                            "POST",
                            f"{session_url}/execute/sync",
                            operation="user-agent read",
                            timeout=5.0,
                            json={"script": "return navigator.userAgent;", "args": []},
                        ).get("value")
                        if isinstance(value, str):
                            user_agent = value
                    except RuntimeError:
                        LOGGER.warning("Firefox user-agent could not be read; continuing with aiohttp default")
                    return _BrowserAuthResult(cookies=cookies, user_agent=user_agent)
            cancel_event.wait(0.25)

        if saw_target_url:
            raise RuntimeError(
                "Firefox returned to the target URL, but no applicable KML cookie became available"
            )
        raise RuntimeError(f"KML authentication did not finish within {auth_timeout:g} seconds")
    finally:
        if session_id is not None:
            with contextlib.suppress(Exception):
                http.delete(f"{driver_base_url}/session/{session_id}", timeout=5.0)
        http.close()
        if process is not None and process.poll() is None:
            with contextlib.suppress(OSError):
                process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5.0)
        try:
            shutil.rmtree(profile_root)
        except OSError:
            LOGGER.warning(
                "Temporary Firefox profile cleanup failed; remove the newest "
                "ws_tcp_tunnel-* directory before the next authentication"
            )


class _KmlFirefoxAuthenticator:
    def __init__(self, session: ClientSession, args: argparse.Namespace) -> None:
        self._session = session
        self._target_url = _http_url_from_ws_url(args.ws_url)
        self._auth_timeout = args.auth_timeout
        self._geckodriver = args.geckodriver
        self._firefox_binary = args.firefox_binary
        self._lock = asyncio.Lock()
        self._generation = 0
        self._last_refresh = 0.0

    @property
    def generation(self) -> int:
        return self._generation

    async def authenticate(self, *, expected_generation: int | None = None, force: bool = False) -> bool:
        async with self._lock:
            if expected_generation is not None and expected_generation != self._generation:
                return True
            if not force and time.monotonic() - self._last_refresh < _AUTH_REFRESH_COOLDOWN_S:
                return False

            cancel_event = threading.Event()
            worker = asyncio.create_task(
                asyncio.to_thread(
                    _collect_kml_firefox_auth,
                    self._target_url,
                    auth_timeout=self._auth_timeout,
                    geckodriver=self._geckodriver,
                    firefox_binary=self._firefox_binary,
                    cancel_event=cancel_event,
                )
            )
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancel_event.set()
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await worker
                raise
            # This jar is private to the tunnel. Clearing it also removes any cookies collected
            # while aiohttp followed a rejected handshake to the SSO host.
            _load_browser_cookies(
                self._session.cookie_jar,
                result.cookies,
                self._target_url,
            )
            if result.user_agent:
                self._session.headers["User-Agent"] = result.user_agent
            self._generation += 1
            self._last_refresh = time.monotonic()
            LOGGER.info(
                "KML authentication loaded into memory for %s (%d target cookie(s))",
                _redact_url_for_log(self._target_url),
                len(result.cookies),
            )
            return True


async def _pipe_reader_to_ws(
    reader: asyncio.StreamReader,
    ws_send: Callable[[bytes], Awaitable[None]],
    *,
    label: str,
) -> None:
    try:
        while True:
            data = await reader.read(64 * 1024)
            if not data:
                break
            await ws_send(data)
    except (ConnectionError, asyncio.CancelledError):
        raise
    except Exception:
        LOGGER.exception("%s pipe failed", label)


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()


async def _run_until_first_completed(*coroutines: Awaitable[None]) -> None:
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            try:
                await task
            except (ConnectionError, asyncio.CancelledError):
                pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def run_server(args: argparse.Namespace) -> None:
    async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=args.heartbeat)
        await ws.prepare(request)
        peer = request.remote or "unknown"
        LOGGER.info("websocket connected from %s path=%s", peer, request.path)

        try:
            target_reader, target_writer = await asyncio.open_connection(args.target_host, args.target_port)
        except Exception as exc:
            LOGGER.error("failed to connect target %s:%s: %s", args.target_host, args.target_port, exc)
            await ws.close(message=str(exc).encode())
            return ws

        async def tcp_to_ws() -> None:
            await _pipe_reader_to_ws(target_reader, ws.send_bytes, label="tcp->ws")

        async def ws_to_tcp() -> None:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    target_writer.write(msg.data)
                    await target_writer.drain()
                elif msg.type == WSMsgType.TEXT:
                    target_writer.write(msg.data.encode())
                    await target_writer.drain()
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED, WSMsgType.ERROR):
                    break

        try:
            await _run_until_first_completed(tcp_to_ws(), ws_to_tcp())
        finally:
            await _close_writer(target_writer)
            await ws.close()
        LOGGER.info("websocket disconnected from %s", peer)
        return ws

    app = web.Application()
    app.router.add_get("/{tail:.*}", websocket_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.listen_host, args.listen_port)
    await site.start()
    LOGGER.info(
        "server listening on %s:%s, forwarding websocket streams to %s:%s",
        args.listen_host,
        args.listen_port,
        args.target_host,
        args.target_port,
    )
    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


async def _handle_local(
    local_reader: asyncio.StreamReader,
    local_writer: asyncio.StreamWriter,
    *,
    session: ClientSession,
    args: argparse.Namespace,
    authenticator: _KmlFirefoxAuthenticator | None,
    cookie_header: str | None = None,
) -> None:
    peer = local_writer.get_extra_info("peername")
    safe_ws_url = _redact_url_for_log(args.ws_url)
    LOGGER.info("local tcp connected from %s", peer)
    try:
        for attempt in range(2 if authenticator else 1):
            auth_generation = authenticator.generation if authenticator else 0
            try:
                async with session.ws_connect(
                    args.ws_url,
                    heartbeat=args.heartbeat,
                    max_msg_size=args.max_msg_size,
                    headers={"Cookie": cookie_header} if cookie_header else None,
                ) as ws:
                    async def local_to_ws() -> None:
                        await _pipe_reader_to_ws(local_reader, ws.send_bytes, label="local->ws")

                    async def ws_to_local() -> None:
                        async for msg in ws:
                            if msg.type == WSMsgType.BINARY:
                                local_writer.write(msg.data)
                                await local_writer.drain()
                            elif msg.type == WSMsgType.TEXT:
                                local_writer.write(msg.data.encode())
                                await local_writer.drain()
                            elif msg.type in (
                                WSMsgType.CLOSE,
                                WSMsgType.CLOSING,
                                WSMsgType.CLOSED,
                                WSMsgType.ERROR,
                            ):
                                break

                    await _run_until_first_completed(local_to_ws(), ws_to_local())
                return
            except WSServerHandshakeError as exc:
                is_sso = _classify_sso_handshake_error(exc)
                if is_sso and authenticator is not None and attempt == 0:
                    LOGGER.warning(
                        "KML SSO rejected the WebSocket handshake for %s (HTTP %s); "
                        "checking whether interactive refresh is needed",
                        safe_ws_url,
                        exc.status,
                    )
                    try:
                        refreshed = await authenticator.authenticate(expected_generation=auth_generation)
                    except (RuntimeError, ValueError) as auth_exc:
                        LOGGER.error("KML authentication refresh failed: %s", auth_exc)
                        return
                    if refreshed:
                        continue
                detail = "; KML SSO authentication required" if is_sso else ""
                LOGGER.error(
                    "websocket handshake failed for %s (HTTP %s%s)",
                    safe_ws_url,
                    exc.status,
                    detail,
                )
                return
            except Exception as exc:
                LOGGER.error(
                    "websocket connection failed for %s (%s)",
                    safe_ws_url,
                    type(exc).__name__,
                )
                return
    finally:
        await _close_writer(local_writer)
        LOGGER.info("local tcp disconnected from %s", peer)


async def run_client(args: argparse.Namespace) -> None:
    cookie_env = getattr(args, "cookie_env", None)
    cookie_header: str | None = None
    if cookie_env:
        if args.auth != "none":
            raise ValueError("--cookie-env and --auth=kml-firefox cannot be used together")
        cookie_header, cookie_count = _load_cookie_header_from_env(cookie_env, args.ws_url)
        LOGGER.info("loaded %d target-scoped cookie(s) from the environment", cookie_count)
    # A normal jar would merge any response cookies into the explicit header on a later
    # connection, collapsing duplicate names and changing Firefox's RFC ordering.
    cookie_jar = DummyCookieJar() if cookie_header else CookieJar()
    async with ClientSession(cookie_jar=cookie_jar) as session:
        authenticator: _KmlFirefoxAuthenticator | None = None
        if args.auth == "kml-firefox":
            authenticator = _KmlFirefoxAuthenticator(session, args)
            await authenticator.authenticate(force=True)

        def handle_local(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> Awaitable[None]:
            return _handle_local(
                reader,
                writer,
                session=session,
                args=args,
                authenticator=authenticator,
                cookie_header=cookie_header,
            )

        server = await asyncio.start_server(handle_local, args.listen_host, args.listen_port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
        LOGGER.info(
            "client listening on %s, forwarding to %s",
            sockets,
            _redact_url_for_log(args.ws_url),
        )
        async with server:
            await server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server = subparsers.add_parser("server", help="Accept WebSocket and forward to local TCP target")
    server.add_argument("--listen-host", default="0.0.0.0")
    server.add_argument("--listen-port", type=int, default=15174)
    server.add_argument("--target-host", default="127.0.0.1")
    server.add_argument("--target-port", type=int, default=15173)
    server.add_argument("--heartbeat", type=float, default=20.0)

    client = subparsers.add_parser("client", help="Accept local TCP and forward to WebSocket server")
    client.add_argument("--listen-host", default="127.0.0.1")
    client.add_argument("--listen-port", type=int, default=8080)
    client.add_argument("--ws-url", required=True)
    client.add_argument("--heartbeat", type=float, default=20.0)
    client.add_argument("--max-msg-size", type=int, default=0, help="0 disables aiohttp message size limit")
    client.add_argument(
        "--auth",
        choices=["none", "kml-firefox"],
        default="none",
        help=(
            "kml-firefox opens a temporary browser for interactive SSO/MFA, then imports its "
            "target cookies into tunnel memory"
        ),
    )
    client.add_argument(
        "--auth-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for interactive KML login (default: 300)",
    )
    client.add_argument("--geckodriver", help="geckodriver executable (default: search PATH)")
    client.add_argument(
        "--cookie-env",
        help="read a Cookie header from this environment variable, then delete it from the process environment",
    )
    client.add_argument(
        "--firefox-binary",
        help="Firefox executable (default: auto-detect; leave unset for Ubuntu Snap Firefox)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.mode == "server":
            asyncio.run(run_server(args))
        elif args.mode == "client":
            if args.auth_timeout <= 0:
                raise ValueError("--auth-timeout must be greater than zero")
            asyncio.run(run_client(args))
    except KeyboardInterrupt:
        LOGGER.info("stopped")
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
