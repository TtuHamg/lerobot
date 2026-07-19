from __future__ import annotations

import argparse
import asyncio
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


aiohttp = pytest.importorskip("aiohttp")
client_reqrep = pytest.importorskip("aiohttp.client_reqrep")
test_utils = pytest.importorskip("aiohttp.test_utils")
multidict = pytest.importorskip("multidict")
yarl = pytest.importorskip("yarl")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools/ws_tcp_tunnel.py"
SPEC = importlib.util.spec_from_file_location("ws_tcp_tunnel_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tunnel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tunnel
SPEC.loader.exec_module(tunnel)


def _handshake_error(
    real_url: str,
    *,
    status: int,
    history: tuple[object, ...] = (),
    headers: dict[str, str] | None = None,
) -> aiohttp.WSServerHandshakeError:
    request_headers = multidict.CIMultiDictProxy(multidict.CIMultiDict())
    url = yarl.URL(real_url)
    request_info = client_reqrep.RequestInfo(url, "GET", request_headers, url)
    return aiohttp.WSServerHandshakeError(
        request_info,
        history,
        status=status,
        message="Invalid response status",
        headers=multidict.CIMultiDict(headers or {}),
    )


class _FakeWriter:
    def __init__(self, peer_port: int) -> None:
        self.peer = ("127.0.0.1", peer_port)
        self.buffer = bytearray()
        self.closed = False
        self.wait_closed_calls = 0

    def get_extra_info(self, name: str) -> object:
        return self.peer if name == "peername" else None

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        self.wait_closed_calls += 1


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> object:
        raise StopAsyncIteration


class _WebSocketContext:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket
        self.exited = False

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *_args: object) -> None:
        self.exited = True


class _RaisingWebSocketContext:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def __aenter__(self) -> None:
        raise self.exc

    async def __aexit__(self, *_args: object) -> None:
        return None


def _eof_reader() -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_eof()
    return reader


def _client_args(ws_url: str) -> argparse.Namespace:
    return argparse.Namespace(
        auth="none",
        heartbeat=20.0,
        listen_host="127.0.0.1",
        listen_port=8080,
        max_msg_size=0,
        ws_url=ws_url,
    )


def test_redact_url_removes_user_info_query_and_fragment() -> None:
    raw_url = (
        "wss://alice:password-secret@machine.example:8443/ws/control"
        "?ticket=query-secret&service=private#fragment-secret"
    )

    assert tunnel._redact_url_for_log(raw_url) == "wss://machine.example:8443/<redacted-path>"
    assert tunnel._redact_url_for_log("not-an-absolute-url") == "<invalid-url>"


def test_classifies_final_cas_login_page_after_sso_redirect() -> None:
    target_url = "https://machine.example/ws"
    cas_url = (
        "https://sso.corp.kuaishou.com/cas/login"
        "?service=https%3A%2F%2Fmachine.example%2Faccessproxy_sso_callback&ticket=secret"
    )
    redirect = SimpleNamespace(
        status=302,
        url=yarl.URL(target_url),
        headers={"Location": cas_url},
    )
    exc = _handshake_error(cas_url, status=200, history=(redirect,))

    assert tunnel._classify_sso_handshake_error(exc) is True


@pytest.mark.parametrize(
    ("real_url", "history", "headers"),
    [
        ("https://machine.example/ws", (), {}),
        (
            "https://machine.example/maintenance",
            (
                SimpleNamespace(
                    status=302,
                    url=yarl.URL("https://machine.example/ws"),
                    headers={"Location": "/maintenance"},
                ),
            ),
            {},
        ),
        ("https://not-sso.example/ws", (), {"Location": "https://login.example/sign-in"}),
    ],
)
def test_does_not_misclassify_non_sso_handshake_errors(
    real_url: str,
    history: tuple[object, ...],
    headers: dict[str, str],
) -> None:
    exc = _handshake_error(real_url, status=200, history=history, headers=headers)

    assert tunnel._classify_sso_handshake_error(exc) is False


def test_target_cookies_filter_domain_path_expiry_and_secure_scheme() -> None:
    now = time.time()
    target_url = "https://robot.machine.corp.kuaishou.com/ws/control"
    cookies = [
        {
            "name": "exact_host",
            "value": "exact-value",
            "domain": "robot.machine.corp.kuaishou.com",
            "path": "/ws",
            "expiry": now + 60,
            "secure": True,
        },
        {
            "name": "parent_domain",
            "value": "parent-value",
            "domain": ".corp.kuaishou.com",
            "path": "/",
        },
        {
            "name": "default_scope",
            "value": "default-value",
        },
        {
            "name": "wrong_domain",
            "value": "must-not-load",
            "domain": "attacker.example",
            "path": "/ws",
        },
        {
            "name": "wrong_path",
            "value": "must-not-load",
            "domain": "robot.machine.corp.kuaishou.com",
            "path": "/admin",
        },
        {
            "name": "expired",
            "value": "must-not-load",
            "domain": "robot.machine.corp.kuaishou.com",
            "path": "/ws",
            "expiry": now - 1,
        },
    ]

    selected = {cookie.name: cookie for cookie in tunnel._target_cookies(cookies, target_url)}
    assert set(selected) == {"exact_host", "parent_domain", "default_scope"}
    assert selected["exact_host"] == tunnel._BrowserCookie(
        name="exact_host",
        value="exact-value",
        domain="robot.machine.corp.kuaishou.com",
        path="/ws",
        secure=True,
        http_only=False,
        expiry=pytest.approx(now + 60),
        host_only=True,
    )
    assert selected["parent_domain"] == tunnel._BrowserCookie(
        name="parent_domain",
        value="parent-value",
        domain="corp.kuaishou.com",
        path="/",
        secure=False,
        http_only=False,
        expiry=None,
        host_only=False,
    )
    assert selected["default_scope"].host_only is True
    assert (
        tunnel._target_cookies(
            [
                {
                    "name": "secure_only",
                    "value": "must-not-load",
                    "domain": "robot.machine.corp.kuaishou.com",
                    "path": "/ws",
                    "secure": True,
                }
            ],
            "http://robot.machine.corp.kuaishou.com/ws/control",
        )
        == ()
    )


def test_load_browser_cookies_preserves_same_name_scope_and_attributes() -> None:
    target_url = "https://robot.machine.corp.kuaishou.com/ws/control"
    expiry = time.time() + 3600
    raw_cookies = [
        {
            "name": "access_session",
            "value": "host-secure-value",
            "domain": "robot.machine.corp.kuaishou.com",
            "path": "/ws",
            "secure": True,
            "httpOnly": True,
            "expiry": expiry,
        },
        {
            "name": "access_session",
            "value": "parent-plain-value",
            "domain": ".corp.kuaishou.com",
            "path": "/",
            "secure": False,
            "httpOnly": False,
        },
    ]
    browser_cookies = tunnel._target_cookies(raw_cookies, target_url)

    async def run_case() -> None:
        cookie_jar = aiohttp.CookieJar()
        tunnel._load_browser_cookies(cookie_jar, browser_cookies, target_url)

        host_key = ("robot.machine.corp.kuaishou.com", "/ws")
        parent_key = ("corp.kuaishou.com", "")
        assert set(cookie_jar._cookies) == {host_key, parent_key}

        host_cookie = cookie_jar._cookies[host_key]["access_session"]
        parent_cookie = cookie_jar._cookies[parent_key]["access_session"]
        assert host_cookie.value == "host-secure-value"
        assert host_cookie["domain"] == "robot.machine.corp.kuaishou.com"
        assert host_cookie["path"] == "/ws"
        assert host_cookie["secure"] is True
        assert host_cookie["httponly"] is True
        assert host_cookie["expires"]
        assert cookie_jar._expirations[
            ("robot.machine.corp.kuaishou.com", "/ws", "access_session")
        ] == pytest.approx(expiry, abs=1.0)

        assert parent_cookie.value == "parent-plain-value"
        assert parent_cookie["domain"] == "corp.kuaishou.com"
        assert parent_cookie["path"] == "/"
        assert parent_cookie["secure"] == ""
        assert parent_cookie["httponly"] == ""

        http_cookies = cookie_jar.filter_cookies(
            yarl.URL("http://robot.machine.corp.kuaishou.com/ws/control")
        )
        assert http_cookies["access_session"].value == "parent-plain-value"
        assert "host-secure-value" not in http_cookies.output()

    asyncio.run(run_case())


def test_load_cookie_header_from_env_deletes_env_and_preserves_raw_item_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "LEROBOT_TEST_KML_COOKIE"
    target_ws_url = "wss://machine.corp.kuaishou.com/ws"
    raw_header = "session=fake-target-session; Domain=.corp.kuaishou.com; Path=/"
    monkeypatch.setenv(env_name, raw_header)

    cookie_header, count = tunnel._load_cookie_header_from_env(env_name, target_ws_url)

    assert cookie_header == raw_header
    assert count == 3
    assert env_name not in tunnel.os.environ


def test_explicit_cookie_header_is_removed_on_cross_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "LEROBOT_TEST_REDIRECT_COOKIE"
    raw_header = "session=first; Domain=.corp.kuaishou.com; Path=/; session=second"
    monkeypatch.setenv(env_name, raw_header)
    cookie_header, count = tunnel._load_cookie_header_from_env(
        env_name,
        "wss://machine.corp.kuaishou.com/ws",
    )
    target_headers: list[str | None] = []
    sso_headers: list[str | None] = []

    async def run_case() -> None:
        async def sso_handler(request: aiohttp.web.Request) -> aiohttp.web.Response:
            sso_headers.append(request.headers.get("Cookie"))
            return aiohttp.web.Response(text="SSO login page")

        sso_app = aiohttp.web.Application()
        sso_app.router.add_get("/cas/login", sso_handler)
        sso_server = test_utils.TestServer(sso_app)
        try:
            await sso_server.start_server()
        except PermissionError:
            pytest.skip("loopback sockets are disabled by this execution sandbox")

        async def target_handler(request: aiohttp.web.Request) -> aiohttp.web.Response:
            target_headers.append(request.headers.get("Cookie"))
            raise aiohttp.web.HTTPFound(location=str(sso_server.make_url("/cas/login")))

        target_app = aiohttp.web.Application()
        target_app.router.add_get("/ws", target_handler)
        target_server = test_utils.TestServer(target_app)
        try:
            await target_server.start_server()
        except PermissionError:
            await sso_server.close()
            pytest.skip("loopback sockets are disabled by this execution sandbox")
        try:
            target_url = target_server.make_url("/ws").with_scheme("ws")
            async with aiohttp.ClientSession() as session:
                with pytest.raises(aiohttp.WSServerHandshakeError) as exc_info:
                    await session.ws_connect(
                        target_url,
                        headers={"Cookie": cookie_header},
                    )
            assert exc_info.value.status == 200
        finally:
            await target_server.close()
            await sso_server.close()

    asyncio.run(run_case())

    assert count == 4
    assert target_headers == [raw_header]
    assert sso_headers == [None]


def test_run_client_rejects_cookie_env_with_interactive_auth() -> None:
    args = _client_args("wss://machine.corp.kuaishou.com/ws")
    args.auth = "kml-firefox"
    args.cookie_env = "KML_COOKIE"

    with pytest.raises(
        ValueError,
        match=r"--cookie-env and --auth=kml-firefox cannot be used together",
    ):
        asyncio.run(tunnel.run_client(args))


def test_run_client_uses_dummy_cookie_jar_and_per_call_raw_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_name = "LEROBOT_TEST_DUMMY_JAR_COOKIE"
    raw_header = "session=first; preference=middle; session=second"
    monkeypatch.setenv(env_name, raw_header)
    args = _client_args("wss://machine.corp.kuaishou.com/ws")
    args.cookie_env = env_name
    sessions: list[object] = []
    writers: list[_FakeWriter] = []

    class FakeSession:
        def __init__(self, *, cookie_jar: aiohttp.AbstractCookieJar) -> None:
            self.cookie_jar = cookie_jar
            self.headers_seen: list[dict[str, str] | None] = []

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def ws_connect(self, _url: str, **kwargs: object) -> _WebSocketContext:
            self.headers_seen.append(kwargs.get("headers"))
            return _WebSocketContext(_FakeWebSocket())

    def session_factory(*, cookie_jar: aiohttp.AbstractCookieJar) -> FakeSession:
        session = FakeSession(cookie_jar=cookie_jar)
        sessions.append(session)
        return session

    class FakeServer:
        sockets: tuple[()] = ()

        def __init__(self, callback: object) -> None:
            self.callback = callback

        async def __aenter__(self) -> FakeServer:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def serve_forever(self) -> None:
            writer = _FakeWriter(43000)
            writers.append(writer)
            await self.callback(_eof_reader(), writer)

    async def fake_start_server(callback: object, _host: str, _port: int) -> FakeServer:
        return FakeServer(callback)

    monkeypatch.setattr(tunnel, "ClientSession", session_factory)
    monkeypatch.setattr(tunnel.asyncio, "start_server", fake_start_server)

    asyncio.run(tunnel.run_client(args))

    assert env_name not in tunnel.os.environ
    assert len(sessions) == 1
    session = sessions[0]
    assert isinstance(session.cookie_jar, aiohttp.DummyCookieJar)
    assert session.headers_seen == [{"Cookie": raw_header}]
    assert len(writers) == 1
    assert writers[0].closed and writers[0].wait_closed_calls == 1


def test_is_auth_target_normalizes_https_port_and_rejects_http_downgrade() -> None:
    implicit_port = "https://machine.example/ws"
    explicit_port = "https://machine.example:443/ws"

    assert tunnel._is_auth_target(implicit_port, explicit_port) is True
    assert tunnel._is_auth_target(explicit_port, implicit_port) is True
    assert tunnel._is_auth_target("http://machine.example/ws", implicit_port) is False
    assert tunnel._is_auth_target("http://machine.example:443/ws", explicit_port) is False
    assert tunnel._is_auth_target("https://machine.example:444/ws", implicit_port) is False


def test_authenticate_cancellation_signals_and_reaps_collector_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector_started = threading.Event()
    collector_stopped = threading.Event()
    received_cancel_events: list[threading.Event] = []

    def fake_collector(
        _target_url: str,
        *,
        auth_timeout: float,
        geckodriver: str | None,
        firefox_binary: str | None,
        cancel_event: threading.Event,
    ) -> tunnel._BrowserAuthResult:
        assert auth_timeout == 10.0
        assert geckodriver is None
        assert firefox_binary is None
        received_cancel_events.append(cancel_event)
        assert cancel_event.is_set()
        collector_stopped.set()
        raise RuntimeError("collector stopped after cancellation")

    async def fake_to_thread(function: object, *args: object, **kwargs: object) -> object:
        assert function is fake_collector
        collector_started.set()
        cancel_event = kwargs["cancel_event"]
        assert isinstance(cancel_event, threading.Event)
        while not cancel_event.is_set():
            await asyncio.sleep(0)
        return function(*args, **kwargs)

    monkeypatch.setattr(tunnel, "_collect_kml_firefox_auth", fake_collector)
    monkeypatch.setattr(tunnel.asyncio, "to_thread", fake_to_thread)

    async def run_case() -> None:
        session = SimpleNamespace(cookie_jar=aiohttp.CookieJar(), headers={})
        args = argparse.Namespace(
            ws_url="wss://machine.example/ws",
            auth_timeout=10.0,
            geckodriver=None,
            firefox_binary=None,
        )
        authenticator = tunnel._KmlFirefoxAuthenticator(session, args)
        baseline_tasks = asyncio.all_tasks()
        authenticate_task = asyncio.get_running_loop().create_task(
            authenticator.authenticate(force=True)
        )

        async def wait_until_collector_starts() -> None:
            while not collector_started.is_set():
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_until_collector_starts(), timeout=2.0)
        live_worker_tasks = asyncio.all_tasks() - baseline_tasks - {authenticate_task}
        assert len(live_worker_tasks) == 1
        worker_task = live_worker_tasks.pop()
        authenticate_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(authenticate_task, timeout=2.0)

        assert len(received_cancel_events) == 1
        assert received_cancel_events[0].is_set()
        assert collector_stopped.is_set()
        assert worker_task.done() and not worker_task.cancelled()
        assert isinstance(worker_task.exception(), RuntimeError)
        assert authenticator.generation == 0

    asyncio.run(run_case())


def test_run_client_reuses_one_session_and_cookie_jar_across_local_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_ws_url = "wss://machine.example/ws"
    target_http_url = yarl.URL("https://machine.example/ws")
    cookie_name = "accessproxy_session"
    cookie_value = "opaque-cookie-value"
    sessions: list[object] = []
    writers: list[_FakeWriter] = []

    class FakeSession:
        def __init__(self, *, cookie_jar: aiohttp.CookieJar) -> None:
            self.cookie_jar = cookie_jar
            self.closed = False
            self.exit_calls = 0
            self.cookies_seen: list[dict[str, str]] = []
            self.contexts: list[_WebSocketContext] = []

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.closed = True
            self.exit_calls += 1

        def ws_connect(self, url: str, **_kwargs: object) -> _WebSocketContext:
            assert url == target_ws_url
            filtered = self.cookie_jar.filter_cookies(target_http_url)
            self.cookies_seen.append({name: morsel.value for name, morsel in filtered.items()})
            if len(self.cookies_seen) == 1:
                self.cookie_jar.update_cookies(
                    {cookie_name: cookie_value},
                    response_url=target_http_url,
                )
            context = _WebSocketContext(_FakeWebSocket())
            self.contexts.append(context)
            return context

    def session_factory(*, cookie_jar: aiohttp.CookieJar) -> FakeSession:
        session = FakeSession(cookie_jar=cookie_jar)
        sessions.append(session)
        return session

    class FakeServer:
        sockets: tuple[()] = ()

        def __init__(self, callback: object) -> None:
            self.callback = callback
            self.exited = False

        async def __aenter__(self) -> FakeServer:
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.exited = True

        async def serve_forever(self) -> None:
            for index in range(2):
                writer = _FakeWriter(41000 + index)
                writers.append(writer)
                await self.callback(_eof_reader(), writer)

    servers: list[FakeServer] = []

    async def fake_start_server(callback: object, host: str, port: int) -> FakeServer:
        assert (host, port) == ("127.0.0.1", 8080)
        server = FakeServer(callback)
        servers.append(server)
        return server

    monkeypatch.setattr(tunnel, "ClientSession", session_factory)
    monkeypatch.setattr(tunnel.asyncio, "start_server", fake_start_server)

    asyncio.run(tunnel.run_client(_client_args(target_ws_url)))

    assert len(sessions) == 1
    session = sessions[0]
    assert session.cookies_seen == [{}, {cookie_name: cookie_value}]
    assert session.closed is True
    assert session.exit_calls == 1
    assert len(session.contexts) == 2
    assert all(context.exited for context in session.contexts)
    assert len(servers) == 1 and servers[0].exited is True
    assert len(writers) == 2
    assert all(writer.closed and writer.wait_closed_calls == 1 for writer in writers)


def test_handshake_error_log_does_not_leak_url_or_cookie_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target_ticket = "target-query-ticket-secret"
    cas_ticket = "cas-ticket-secret"
    cookie_secret = "access-cookie-secret"
    password_secret = "userinfo-password-secret"
    args = _client_args(
        f"wss://alice:{password_secret}@machine.example/ws?ticket={target_ticket}#private-fragment"
    )
    cas_url = f"https://sso.corp.kuaishou.com/cas/login?ticket={cas_ticket}"
    exc = _handshake_error(
        cas_url,
        status=200,
        headers={
            "Set-Cookie": f"accessproxy_session={cookie_secret}; Path=/; HttpOnly",
        },
    )

    class FailingSession:
        def ws_connect(self, _url: str, **_kwargs: object) -> _RaisingWebSocketContext:
            return _RaisingWebSocketContext(exc)

    writer = _FakeWriter(42000)
    caplog.set_level(logging.DEBUG, logger=tunnel.LOGGER.name)

    async def run_case() -> None:
        await tunnel._handle_local(
            _eof_reader(),
            writer,
            session=FailingSession(),
            args=args,
            authenticator=None,
        )

    asyncio.run(run_case())

    assert "websocket handshake failed for wss://machine.example/ws" in caplog.text
    assert "HTTP 200; KML SSO authentication required" in caplog.text
    for secret in (target_ticket, cas_ticket, cookie_secret, password_secret):
        assert secret not in caplog.text
    assert "Set-Cookie" not in caplog.text
    assert writer.closed is True
    assert writer.wait_closed_calls == 1
