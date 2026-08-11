"""Low-load local web dashboard for Franka Cartesian action chunks.

The dashboard is a separate process from the LeRobot client.  Camera ROS
subscriptions do not exist until the browser explicitly enables them, and are
destroyed immediately when disabled.  All caches are bounded by wall-clock
time so long-running validation never grows an unbounded timeline.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from .ros2_runtime import _compose_pose, _invert_pose

LOGGER = logging.getLogger("franka_web_dashboard")
WEB_ROOT = Path(__file__).with_name("web")
JOINT_NAMES = tuple(f"fr3_joint{i + 1}" for i in range(7))


def _stamp_seconds(stamp: Any) -> float:
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return value if value > 0.0 else time.time()


def _duration_seconds(duration: Any) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def _plan_key(session_id: str, plan_id: int) -> str:
    return f"{session_id}:{int(plan_id)}"


class RollingStore:
    """Thread-safe, time-bounded dashboard state."""

    def __init__(self, history_seconds: float, default_pose_frame: str = "eef") -> None:
        if history_seconds <= 0:
            raise ValueError("history_seconds must be positive")
        if default_pose_frame not in {"eef", "link8"}:
            raise ValueError("default_pose_frame must be 'eef' or 'link8'")
        self.history_seconds = float(history_seconds)
        self.default_pose_frame = default_pose_frame
        self._lock = threading.RLock()
        self._chunks: deque[dict[str, Any]] = deque()
        self._chunk_by_key: dict[str, dict[str, Any]] = {}
        self._actual: deque[dict[str, Any]] = deque()
        self._cameras: dict[str, deque[dict[str, Any]]] = {
            "camera1": deque(),
            "camera2": deque(),
        }
        self._camera_enabled = {"camera1": False, "camera2": False}
        self._pending_ik: dict[str, tuple[float, list[list[float]]]] = {}
        self._pending_ack: dict[str, dict[str, Any]] = {}
        self._status: dict[str, Any] | None = None
        self._next_chunk_idx = 0
        self._ee_t_link8: tuple[np.ndarray, np.ndarray] | None = None

    def _to_link8(self, eef: list[float]) -> list[float] | None:
        if self._ee_t_link8 is None:
            return None
        position, quaternion = _compose_pose(
            eef[:3],
            eef[3:7],
            self._ee_t_link8[0],
            self._ee_t_link8[1],
        )
        return [*(float(value) for value in position), *(float(value) for value in quaternion)]

    def _prune_locked(self, now_s: float) -> None:
        cutoff = now_s - self.history_seconds
        while self._chunks and self._chunks[0]["received_t"] < cutoff:
            removed = self._chunks.popleft()
            self._chunk_by_key.pop(removed["key"], None)
        while self._actual and self._actual[0]["t"] < cutoff:
            self._actual.popleft()
        for frames in self._cameras.values():
            while frames and frames[0]["t"] < cutoff:
                frames.popleft()
        stale_ack_keys = [
            key
            for key, value in self._pending_ack.items()
            if float(value.get("received_t", 0.0)) < cutoff
        ]
        for key in stale_ack_keys:
            self._pending_ack.pop(key, None)
        stale_ik_keys = [
            key for key, (received_t, _) in self._pending_ik.items() if received_t < cutoff
        ]
        for key in stale_ik_keys:
            self._pending_ik.pop(key, None)

    def add_chunk(self, message: Any) -> dict[str, Any]:
        now_s = time.time()
        t0 = _stamp_seconds(message.header.stamp)
        period_s = max(_duration_seconds(message.period), 1e-6)
        key = _plan_key(str(message.session_id), int(message.plan_id))
        eef = [
            [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ]
            for pose in message.poses
        ]
        with self._lock:
            pending_ik = self._pending_ik.pop(key, None)
            chunk = {
                "key": key,
                "session_id": str(message.session_id),
                "plan_id": int(message.plan_id),
                "chunk_idx": self._next_chunk_idx,
                "source_timestep": int(message.source_timestep),
                "received_t": now_s,
                "t0": t0,
                "period_s": period_s,
                "times": [t0 + (index + 1) * period_s for index in range(len(eef))],
                "eef": eef,
                "link8": [self._to_link8(row) for row in eef]
                if self._ee_t_link8 is not None
                else None,
                "gripper": [float(value) for value in message.gripper],
                "ik_joints": pending_ik[1] if pending_ik is not None else None,
                "ack": self._pending_ack.pop(key, None),
            }
            self._next_chunk_idx += 1
            self._chunks.append(chunk)
            self._chunk_by_key[key] = chunk
            self._prune_locked(now_s)
            return {"type": "chunk", "chunk": copy.deepcopy(chunk)}

    def add_ik(self, message: Any) -> dict[str, Any]:
        key = _plan_key(str(message.session_id), int(message.plan_id))
        count = int(message.waypoint_count)
        flat = [float(value) for value in message.positions]
        expected = count * len(JOINT_NAMES)
        if count <= 0 or len(flat) != expected:
            raise ValueError(
                f"invalid IK chunk shape: waypoint_count={count}, positions={len(flat)}"
            )
        joints = [flat[index * 7 : (index + 1) * 7] for index in range(count)]
        with self._lock:
            chunk = self._chunk_by_key.get(key)
            if chunk is None:
                self._pending_ik[key] = (time.time(), joints)
            else:
                chunk["ik_joints"] = joints
            self._prune_locked(time.time())
        return {"type": "ik", "key": key, "joints": joints}

    def add_ack(self, message: Any) -> dict[str, Any]:
        key = _plan_key(str(message.session_id), int(message.plan_id))
        ack = {
            "accepted": bool(message.accepted),
            "result": int(message.result),
            "waypoint_count": int(message.waypoint_count),
            "detail": str(message.detail),
            "received_t": time.time(),
        }
        with self._lock:
            chunk = self._chunk_by_key.get(key)
            if chunk is None:
                self._pending_ack[key] = ack
            else:
                chunk["ack"] = ack
            self._prune_locked(ack["received_t"])
        return {"type": "ack", "key": key, "ack": copy.deepcopy(ack)}

    def add_status(self, message: Any) -> dict[str, Any]:
        status = {
            "t": time.time(),
            "armed": bool(message.armed),
            "state": int(message.state),
            "state_fresh": bool(message.state_fresh),
            "robot_ready": bool(message.robot_ready),
            "controller_ready": bool(message.controller_ready),
            "has_active_plan": bool(message.has_active_plan),
            "session_id": str(message.session_id),
            "plan_id": int(message.plan_id),
            "next_waypoint_index": int(message.next_waypoint_index),
            "applied_waypoint_index": int(message.applied_waypoint_index),
            "detail": str(message.detail),
        }
        with self._lock:
            self._status = status
        return {"type": "status", "status": copy.deepcopy(status)}

    def add_actual_pose(self, message: Any) -> dict[str, Any]:
        pose = message.pose
        sample = {
            "t": _stamp_seconds(message.header.stamp),
            "eef": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }
        with self._lock:
            sample["link8"] = self._to_link8(sample["eef"])
            self._actual.append(sample)
            self._prune_locked(time.time())
        return {"type": "actual", "sample": copy.deepcopy(sample)}

    def set_camera_enabled(self, name: str, enabled: bool) -> dict[str, Any]:
        if name not in self._cameras:
            raise ValueError(f"unknown camera {name!r}")
        with self._lock:
            self._camera_enabled[name] = bool(enabled)
            if not enabled:
                self._cameras[name].clear()
        return {"type": "camera_state", "camera": name, "enabled": bool(enabled)}

    def add_camera_frame(self, name: str, stamp_s: float, jpeg: bytes) -> dict[str, Any]:
        frame = {
            "t": float(stamp_s),
            "jpeg": base64.b64encode(jpeg).decode("ascii"),
        }
        with self._lock:
            if not self._camera_enabled[name]:
                return {"type": "noop"}
            self._cameras[name].append(frame)
            self._prune_locked(time.time())
        return {"type": "camera", "camera": name, "frame": copy.deepcopy(frame)}

    def set_tool_transform(
        self,
        position: list[float],
        quaternion_xyzw: list[float],
    ) -> dict[str, Any]:
        inverse = _invert_pose(position, quaternion_xyzw)
        with self._lock:
            self._ee_t_link8 = inverse
            chunks = []
            for chunk in self._chunks:
                chunk["link8"] = [self._to_link8(row) for row in chunk["eef"]]
                chunks.append({"key": chunk["key"], "link8": copy.deepcopy(chunk["link8"])})
            actual = []
            for sample in self._actual:
                sample["link8"] = self._to_link8(sample["eef"])
                actual.append({"t": sample["t"], "link8": copy.deepcopy(sample["link8"])})
        return {
            "type": "frame_transform",
            "available": True,
            "chunks": chunks,
            "actual": actual,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._prune_locked(time.time())
            return {
                "type": "snapshot",
                "server_t": time.time(),
                "history_seconds": self.history_seconds,
                "default_pose_frame": self.default_pose_frame,
                "link8_available": self._ee_t_link8 is not None,
                "chunks": copy.deepcopy(list(self._chunks)),
                "actual": copy.deepcopy(list(self._actual)),
                "cameras": {
                    name: copy.deepcopy(list(frames))
                    for name, frames in self._cameras.items()
                },
                "camera_enabled": dict(self._camera_enabled),
                "status": copy.deepcopy(self._status),
            }


class EventHub:
    """Bridge callback threads into one aiohttp event loop."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=512)
        self._task = asyncio.create_task(self._broadcast_loop())

    def emit(self, event: dict[str, Any]) -> None:
        if event.get("type") == "noop":
            return
        loop, event_queue = self._loop, self._queue
        if loop is None or event_queue is None or loop.is_closed():
            return

        def enqueue() -> None:
            if event_queue.full():
                with suppress(asyncio.QueueEmpty):
                    event_queue.get_nowait()
            event_queue.put_nowait(event)

        loop.call_soon_threadsafe(enqueue)

    async def _broadcast_loop(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            payload = json.dumps(event, separators=(",", ":"), allow_nan=False)
            dead: list[web.WebSocketResponse] = []
            for client in tuple(self._clients):
                try:
                    await client.send_str(payload)
                except (ConnectionError, RuntimeError):
                    dead.append(client)
            for client in dead:
                self._clients.discard(client)

    def add_client(self, client: web.WebSocketResponse) -> None:
        self._clients.add(client)

    def remove_client(self, client: web.WebSocketResponse) -> None:
        self._clients.discard(client)

    async def stop(self) -> None:
        for client in tuple(self._clients):
            await client.close()
        self._clients.clear()
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task


class DashboardRosNode:
    """ROS subscriptions feeding the dashboard store."""

    def __init__(
        self,
        *,
        store: RollingStore,
        hub: EventHub,
        camera_topics: dict[str, str],
        camera_fps: float,
        camera_max_width: int,
        jpeg_quality: int,
        topics: dict[str, str],
    ) -> None:
        import rclpy
        from franka_msgs.msg import FrankaRobotState
        from geometry_msgs.msg import PoseStamped
        from lerobot_franka_interfaces.msg import (
            CartesianActionChunk,
            CartesianActionChunkAck,
            IkJointActionChunk,
            SafetyGatewayStatus,
        )
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        class NodeImpl(Node):
            pass

        self.rclpy = rclpy
        self.node = NodeImpl("franka_web_dashboard")
        self.store = store
        self.hub = hub
        self.Image = Image
        self.camera_topics = camera_topics
        self.camera_period_s = 1.0 / max(float(camera_fps), 0.1)
        self.camera_max_width = max(64, int(camera_max_width))
        self.jpeg_quality = min(max(int(jpeg_quality), 20), 95)
        self._last_camera_monotonic = {"camera1": 0.0, "camera2": 0.0}
        self._camera_subscriptions: dict[str, Any] = {}
        self._camera_requests: queue.SimpleQueue[tuple[str, bool]] = queue.SimpleQueue()
        self._last_actual_emit = 0.0
        self._robot_state_subscription: Any | None = None

        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        best_effort = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        transient = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._subscriptions = [
            self.node.create_subscription(
                CartesianActionChunk,
                topics["action_chunk"],
                lambda message: self.hub.emit(self.store.add_chunk(message)),
                reliable,
            ),
            self.node.create_subscription(
                CartesianActionChunkAck,
                topics["ack"],
                lambda message: self.hub.emit(self.store.add_ack(message)),
                reliable,
            ),
            self.node.create_subscription(
                IkJointActionChunk,
                topics["ik"],
                lambda message: self.hub.emit(self.store.add_ik(message)),
                transient,
            ),
            self.node.create_subscription(
                SafetyGatewayStatus,
                topics["status"],
                lambda message: self.hub.emit(self.store.add_status(message)),
                best_effort,
            ),
            self.node.create_subscription(
                PoseStamped,
                topics["current_pose"],
                self._on_actual_pose,
                best_effort,
            ),
        ]
        self._robot_state_subscription = self.node.create_subscription(
            FrankaRobotState,
            topics["robot_state"],
            self._on_robot_state,
            best_effort,
        )
        self._subscriptions.append(self._robot_state_subscription)
        self._camera_qos = best_effort
        self.node.create_timer(0.1, self._apply_camera_requests)

    def _on_actual_pose(self, message: Any) -> None:
        now = time.monotonic()
        if now - self._last_actual_emit < 0.1:
            return
        self._last_actual_emit = now
        self.hub.emit(self.store.add_actual_pose(message))

    def _on_robot_state(self, message: Any) -> None:
        pose = message.f_t_ee.pose
        event = self.store.set_tool_transform(
            [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
            [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        )
        self.hub.emit(event)
        subscription = self._robot_state_subscription
        if subscription is not None and self.node.destroy_subscription(subscription):
            self._robot_state_subscription = None
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
            self.node.get_logger().info(
                "F_T_EE captured for EEF/link8 display toggle; removed 1 kHz RobotState subscription"
            )

    def request_camera(self, name: str, enabled: bool) -> None:
        self._camera_requests.put((name, bool(enabled)))

    def _apply_camera_requests(self) -> None:
        latest: dict[str, bool] = {}
        while True:
            try:
                name, enabled = self._camera_requests.get_nowait()
            except queue.Empty:
                break
            latest[name] = enabled
        for name, enabled in latest.items():
            if name not in self.camera_topics:
                continue
            current = name in self._camera_subscriptions
            if enabled and not current:
                def callback(message: Any, camera: str = name) -> None:
                    self._on_image(camera, message)

                self._camera_subscriptions[name] = self.node.create_subscription(
                    self.Image,
                    self.camera_topics[name],
                    callback,
                    self._camera_qos,
                )
            elif not enabled and current:
                self.node.destroy_subscription(self._camera_subscriptions.pop(name))
            self.hub.emit(self.store.set_camera_enabled(name, enabled))

    def _on_image(self, name: str, message: Any) -> None:
        now = time.monotonic()
        if now - self._last_camera_monotonic[name] < self.camera_period_s:
            return
        self._last_camera_monotonic[name] = now
        try:
            raw = np.frombuffer(message.data, dtype=np.uint8)
            rows = raw.reshape(int(message.height), int(message.step))
            channels = 4 if str(message.encoding).lower() in {"rgba8", "bgra8"} else 3
            image = rows[:, : int(message.width) * channels].reshape(
                int(message.height), int(message.width), channels
            )
            encoding = str(message.encoding).lower()
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif encoding == "rgba8":
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif encoding == "bgra8":
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif encoding != "bgr8":
                raise ValueError(f"unsupported camera encoding {message.encoding!r}")
            if image.shape[1] > self.camera_max_width:
                scale = self.camera_max_width / image.shape[1]
                image = cv2.resize(
                    image,
                    (self.camera_max_width, max(1, round(image.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                raise RuntimeError("cv2.imencode returned false")
            event = self.store.add_camera_frame(
                name,
                _stamp_seconds(message.header.stamp),
                encoded.tobytes(),
            )
            self.hub.emit(event)
        except Exception as error:
            self.node.get_logger().warning(f"{name} thumbnail encoding failed: {error}")

    def spin(self) -> None:
        self.rclpy.spin(self.node)

    def close(self) -> None:
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def _open_browser(url: str) -> None:
    environment = os.environ.copy()
    if not environment.get("DISPLAY"):
        authority = f"/run/user/{os.getuid()}/gdm/Xauthority"
        if os.path.isfile(authority):
            environment["DISPLAY"] = ":0"
            environment["XAUTHORITY"] = authority
            environment.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    try:
        subprocess.Popen(
            ["xdg-open", url],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as error:
        LOGGER.warning("Could not open browser automatically: %s", error)


def _build_app(
    *,
    store: RollingStore,
    hub: EventHub,
    ros_node: DashboardRosNode,
) -> web.Application:
    app = web.Application()

    async def index(_: web.Request) -> web.FileResponse:
        return web.FileResponse(WEB_ROOT / "dashboard.html")

    async def health(_: web.Request) -> web.Response:
        return web.json_response({"ok": True, "time": time.time()})

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        hub.add_client(ws)
        await ws.send_str(json.dumps(store.snapshot(), separators=(",", ":"), allow_nan=False))
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                    if payload.get("type") == "camera":
                        ros_node.request_camera(
                            str(payload.get("camera")),
                            bool(payload.get("enabled")),
                        )
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    await ws.send_json({"type": "error", "detail": str(error)})
        finally:
            hub.remove_client(ws)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/healthz", health)
    app.router.add_get("/ws", websocket)
    app.router.add_static("/static", WEB_ROOT)
    return app


async def _run(args: argparse.Namespace) -> None:
    import rclpy

    rclpy.init()
    store = RollingStore(args.history_seconds, default_pose_frame=args.default_pose_frame)
    hub = EventHub()
    hub.start()
    ros_node = DashboardRosNode(
        store=store,
        hub=hub,
        camera_topics={"camera1": args.camera1_topic, "camera2": args.camera2_topic},
        camera_fps=args.camera_fps,
        camera_max_width=args.camera_max_width,
        jpeg_quality=args.jpeg_quality,
        topics={
            "action_chunk": args.action_chunk_topic,
            "ack": args.ack_topic,
            "ik": args.ik_topic,
            "status": args.status_topic,
            "current_pose": args.current_pose_topic,
            "robot_state": args.robot_state_topic,
        },
    )
    ros_thread = threading.Thread(target=ros_node.spin, name="dashboard-ros", daemon=True)
    ros_thread.start()

    runner = web.AppRunner(_build_app(store=store, hub=hub, ros_node=ros_node))
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    url = f"http://{args.host}:{args.port}"
    LOGGER.info("Franka web dashboard: %s", url)
    if args.open_browser:
        _open_browser(url)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)
    try:
        await stop_event.wait()
    finally:
        await hub.stop()
        await runner.cleanup()
        ros_node.close()
        ros_thread.join(timeout=2.0)


def _topic(value: str) -> str:
    if not value.startswith("/") or any(character.isspace() for character in value):
        raise argparse.ArgumentTypeError("must be an absolute ROS topic without whitespace")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--history-seconds", type=float, default=60.0)
    parser.add_argument("--camera-fps", type=float, default=3.0)
    parser.add_argument("--camera-max-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-pose-frame", choices=("eef", "link8"), default="eef")
    parser.add_argument("--action-chunk-topic", type=_topic, default="/lerobot/franka/action_chunk")
    parser.add_argument("--ack-topic", type=_topic, default="/lerobot/franka/action_chunk_ack")
    parser.add_argument(
        "--ik-topic", type=_topic, default="/lerobot/franka/ik_joint_action_chunk"
    )
    parser.add_argument(
        "--status-topic", type=_topic, default="/lerobot/franka/safety_gateway_status"
    )
    parser.add_argument(
        "--current-pose-topic",
        type=_topic,
        default="/franka_robot_state_broadcaster/current_pose",
    )
    parser.add_argument(
        "--robot-state-topic",
        type=_topic,
        default="/franka_robot_state_broadcaster/robot_state",
    )
    parser.add_argument(
        "--camera1-topic", type=_topic, default="/camera1/camera1/color/image_raw"
    )
    parser.add_argument(
        "--camera2-topic", type=_topic, default="/camera2/camera2/color/image_raw"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with suppress(KeyboardInterrupt):
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
