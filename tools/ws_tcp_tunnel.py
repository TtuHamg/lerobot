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
from collections.abc import Awaitable, Callable

from aiohttp import ClientSession, WSMsgType, web


LOGGER = logging.getLogger("ws_tcp_tunnel")


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


async def run_server(args: argparse.Namespace) -> None:
    async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=args.heartbeat)
        await ws.prepare(request)
        peer = request.remote or "unknown"
        LOGGER.info("websocket connected from %s path=%s", peer, request.path_qs)

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

        tasks = [asyncio.create_task(tcp_to_ws()), asyncio.create_task(ws_to_tcp())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(asyncio.CancelledError):
                await task
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
    await asyncio.Event().wait()


async def run_client(args: argparse.Namespace) -> None:
    async def handle_local(local_reader: asyncio.StreamReader, local_writer: asyncio.StreamWriter) -> None:
        peer = local_writer.get_extra_info("peername")
        LOGGER.info("local tcp connected from %s", peer)
        async with ClientSession() as session:
            try:
                async with session.ws_connect(
                    args.ws_url,
                    heartbeat=args.heartbeat,
                    max_msg_size=args.max_msg_size,
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

                    tasks = [asyncio.create_task(local_to_ws()), asyncio.create_task(ws_to_local())]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in pending:
                        task.cancel()
                    for task in done:
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
            except Exception:
                LOGGER.exception("websocket connection failed: %s", args.ws_url)
            finally:
                await _close_writer(local_writer)
                LOGGER.info("local tcp disconnected from %s", peer)

    server = await asyncio.start_server(handle_local, args.listen_host, args.listen_port)
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    LOGGER.info("client listening on %s, forwarding to %s", sockets, args.ws_url)
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.mode == "server":
        asyncio.run(run_server(args))
    elif args.mode == "client":
        asyncio.run(run_client(args))


if __name__ == "__main__":
    main()
