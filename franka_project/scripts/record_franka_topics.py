#!/usr/bin/env python3
"""Record selected Franka ROS 2 topics as MCAP or camera-only MP4 files."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "recordings"

TOPIC_ALIASES: dict[str, str] = {
    "camera1": "/camera1/camera1/color/image_raw",
    "camera2": "/camera2/camera2/color/image_raw",
    "current_pos": "/franka/joint_states",
    "qpos": "/franka/joint_states",
    "eef": "/franka_robot_state_broadcaster/current_pose",
    "current_pose": "/franka_robot_state_broadcaster/current_pose",
    "gripper": "/gripper/joint_states",
    "action": "/lerobot/franka/action_chunk",
    "action_ack": "/lerobot/franka/action_chunk_ack",
    "gateway_status": "/lerobot/franka/safety_gateway_status",
}
VIDEO_TOPICS = frozenset(
    {TOPIC_ALIASES["camera1"], TOPIC_ALIASES["camera2"]}
)
SUPPORTED_IMAGE_ENCODINGS = ("rgb8", "bgr8", "rgba8", "bgra8", "mono8")
DEFAULT_VIDEO_FPS_FALLBACK = 15.0
FPS_SAMPLE_FRAMES = 12
_STOP_REQUESTED = False


def _request_stop(signum: int, frame: Any) -> None:
    """Translate service-manager/terminal shutdown signals into a clean loop exit."""

    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _install_stop_signal_handlers() -> None:
    # Some shells start background jobs with SIGINT ignored. Restore Python's
    # normal handler so both this process and the subsequently exec'd rosbag2
    # child can still use SIGINT for prompt, metadata-safe finalization.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    for signal_name in ("SIGTERM", "SIGHUP"):
        stop_signal = getattr(signal, signal_name, None)
        if stop_signal is not None:
            signal.signal(stop_signal, _request_stop)


def resolve_topics(values: Sequence[str]) -> list[str]:
    """Resolve aliases/comma-separated values and preserve first-seen order."""

    topics: list[str] = []
    for value in values:
        for token in value.split(","):
            name = token.strip()
            if not name:
                continue
            topic = TOPIC_ALIASES.get(name, name)
            if not topic.startswith("/") or any(character.isspace() for character in topic):
                aliases = ", ".join(sorted(TOPIC_ALIASES))
                raise ValueError(
                    f"Unknown topic alias {name!r}; use one of [{aliases}] or an absolute ROS topic"
                )
            if topic not in topics:
                topics.append(topic)
    if not topics:
        raise ValueError("At least one non-empty topic must be selected")
    return topics


def select_output_format(requested: str, topics: Sequence[str]) -> str:
    """Use MP4 only when the selection contains camera streams and nothing else."""

    if requested not in {"auto", "mcap", "mp4"}:
        raise ValueError(f"Unsupported output format: {requested!r}")
    if requested != "auto":
        return requested
    return "mp4" if topics and set(topics).issubset(VIDEO_TOPICS) else "mcap"


def default_output_path(output_format: str, *, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return DEFAULT_OUTPUT_ROOT / f"{timestamp}_{output_format}"


def build_mcap_command(
    topics: Sequence[str],
    output: Path,
    *,
    storage_profile: str = "zstd_fast",
    qos_overrides_path: Path | None = None,
) -> list[str]:
    """Build the rosbag2 command without using a shell."""

    command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(output),
        "--disable-keyboard-controls",
    ]
    if storage_profile != "none":
        command.extend(["--storage-preset-profile", storage_profile])
    if qos_overrides_path is not None:
        command.extend(["--qos-profile-overrides-path", str(qos_overrides_path)])
    command.extend(["--topics", *topics])
    return command


def _signal_process_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError):
        process.send_signal(sig)


def _stop_process(process: subprocess.Popen[Any], *, finalize_timeout_s: float = 15.0) -> None:
    """Ask rosbag2 to finalize metadata, escalating only if it does not exit."""

    if process.poll() is not None:
        return
    _signal_process_group(process, signal.SIGINT)
    try:
        process.wait(timeout=finalize_timeout_s)
        return
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait(timeout=5.0)


def record_mcap(
    topics: Sequence[str],
    output: Path,
    *,
    duration_s: float | None,
    storage_profile: str,
    qos_overrides_path: Path | None,
) -> None:
    if _STOP_REQUESTED:
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing MCAP output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if qos_overrides_path is not None and not qos_overrides_path.is_file():
        raise FileNotFoundError(f"QoS overrides file does not exist: {qos_overrides_path}")
    command = build_mcap_command(
        topics,
        output,
        storage_profile=storage_profile,
        qos_overrides_path=qos_overrides_path,
    )
    print("Starting MCAP recorder:", " ".join(command), flush=True)
    try:
        process = subprocess.Popen(command, start_new_session=True)  # noqa: S603
    except FileNotFoundError as error:
        raise RuntimeError("ros2 was not found; source the ROS 2 Jazzy environment first") from error

    deadline = None if duration_s is None else time.monotonic() + duration_s
    interrupted = False
    try:
        while process.poll() is None and not _STOP_REQUESTED:
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        _stop_process(process)

    if process.returncode not in (0, -signal.SIGINT):
        reason = " after Ctrl+C" if interrupted else ""
        raise RuntimeError(f"ros2 bag record exited with status {process.returncode}{reason}")
    print(f"MCAP recording finalized in: {output}", flush=True)


def _safe_topic_filename(topic: str) -> str:
    preferred_names = {
        TOPIC_ALIASES["camera1"]: "camera1",
        TOPIC_ALIASES["camera2"]: "camera2",
    }
    if topic in preferred_names:
        return preferred_names[topic]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", topic.strip("/"))
    return stem or "image"


def video_output_paths(topics: Sequence[str], output: Path) -> dict[str, Path]:
    """Resolve a single explicit .mp4 or one file per topic in a directory."""

    if output.suffix.lower() == ".mp4":
        if len(topics) != 1:
            raise ValueError("A .mp4 output file can only be used with one image topic")
        return {topics[0]: output}
    paths: dict[str, Path] = {}
    used_paths: set[Path] = set()
    for topic in topics:
        stem = _safe_topic_filename(topic)
        path = output / f"{stem}.mp4"
        if path in used_paths:
            digest = hashlib.sha256(topic.encode("utf-8")).hexdigest()[:8]
            path = output / f"{stem}_{digest}.mp4"
        if path in used_paths:  # pragma: no cover - requires a SHA-256 prefix collision.
            raise ValueError(f"Video output filename collision for topic {topic!r}")
        paths[topic] = path
        used_paths.add(path)
    return paths


def estimate_video_fps(
    timestamps_ns: Sequence[int], *, fallback: float = DEFAULT_VIDEO_FPS_FALLBACK
) -> float:
    """Estimate constant MP4 FPS from positive consecutive ROS timestamp deltas."""

    deltas = [
        current - previous
        for previous, current in zip(timestamps_ns, timestamps_ns[1:], strict=False)
        if current > previous
    ]
    if not deltas:
        return fallback
    fps = 1_000_000_000 / statistics.median(deltas)
    if not math.isfinite(fps) or fps < 0.1 or fps > 240.0:
        return fallback
    return fps


def image_message_to_bgr(message: Any, *, numpy_module: Any, cv2_module: Any) -> Any:
    """Decode a sensor_msgs/Image into a contiguous BGR8 OpenCV frame."""

    encoding = str(message.encoding).lower()
    if encoding not in SUPPORTED_IMAGE_ENCODINGS:
        supported = ", ".join(SUPPORTED_IMAGE_ENCODINGS)
        raise ValueError(f"Unsupported image encoding {message.encoding!r}; expected one of {supported}")

    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    channels = 1 if encoding == "mono8" else (4 if encoding in {"rgba8", "bgra8"} else 3)
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(f"Image step {step} is smaller than packed row size {row_bytes}")
    required_bytes = step * height
    if len(message.data) < required_bytes:
        raise ValueError(
            f"Image payload is truncated: need {required_bytes} bytes, got {len(message.data)}"
        )

    rows = numpy_module.frombuffer(message.data, dtype=numpy_module.uint8, count=required_bytes)
    rows = rows.reshape(height, step)[:, :row_bytes]
    if channels == 1:
        frame = rows.reshape(height, width)
        frame = cv2_module.cvtColor(frame, cv2_module.COLOR_GRAY2BGR)
    else:
        frame = rows.reshape(height, width, channels)
        if encoding == "rgb8":
            frame = cv2_module.cvtColor(frame, cv2_module.COLOR_RGB2BGR)
        elif encoding == "rgba8":
            frame = cv2_module.cvtColor(frame, cv2_module.COLOR_RGBA2BGR)
        elif encoding == "bgra8":
            frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGRA2BGR)
        # bgr8 already matches OpenCV's byte order.
    return numpy_module.ascontiguousarray(frame)


def record_mp4(
    topics: Sequence[str],
    output: Path,
    *,
    duration_s: float | None,
    fps: float | None,
    codec: str,
    topic_wait_timeout_s: float,
) -> None:
    if _STOP_REQUESTED:
        return
    try:
        import cv2
        import numpy as np
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import Image
    except ImportError as error:
        raise RuntimeError(
            "MP4 recording requires the ROS Python environment plus python3-opencv and python3-numpy"
        ) from error

    paths = video_output_paths(topics, output)
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing video: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    class ImageRecorder(Node):
        def __init__(self) -> None:
            super().__init__("lerobot_franka_mp4_recorder")
            self.writers: dict[str, Any] = {}
            self.writer_fps: dict[str, float] = {}
            self.frame_sizes: dict[str, tuple[int, int]] = {}
            self.frame_counts = dict.fromkeys(topics, 0)
            self.decode_errors = dict.fromkeys(topics, 0)
            self.pending_frames: dict[str, list[Any]] = {topic: [] for topic in topics}
            self.pending_timestamps: dict[str, list[int]] = {topic: [] for topic in topics}
            self._image_subscriptions: list[Any] = []

        def start_subscriptions(self) -> None:
            if self._image_subscriptions:
                raise RuntimeError("Image subscriptions are already started")
            self._image_subscriptions = [
                self.create_subscription(
                    Image,
                    topic,
                    lambda message, selected_topic=topic: self.on_image(selected_topic, message),
                    qos_profile_sensor_data,
                )
                for topic in topics
            ]

        def on_image(self, topic: str, message: Any) -> None:
            try:
                frame = image_message_to_bgr(message, numpy_module=np, cv2_module=cv2)
                size = (int(message.width), int(message.height))
                if topic in self.frame_sizes and self.frame_sizes[topic] != size:
                    raise ValueError(
                        f"Image size changed from {self.frame_sizes[topic]} to {size} on {topic}"
                    )
            except ValueError as error:
                self.decode_errors[topic] += 1
                if self.decode_errors[topic] <= 5:
                    self.get_logger().error(f"Dropping frame from {topic}: {error}")
                return

            self.frame_sizes[topic] = size
            self.frame_counts[topic] += 1
            writer = self.writers.get(topic)
            if writer is not None:
                writer.write(frame)
                return

            stamp = message.header.stamp
            timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            if timestamp_ns <= 0:
                timestamp_ns = self.get_clock().now().nanoseconds
            self.pending_frames[topic].append(frame)
            self.pending_timestamps[topic].append(timestamp_ns)
            if fps is not None or len(self.pending_frames[topic]) >= FPS_SAMPLE_FRAMES:
                # Writer initialization failure is fatal and propagates out of
                # spin_once; continuing would grow pending_frames without bound.
                self.open_writer_and_flush(topic)

        def open_writer_and_flush(self, topic: str) -> None:
            frames = self.pending_frames[topic]
            if not frames or topic in self.writers:
                return
            selected_fps = fps
            if selected_fps is None:
                selected_fps = estimate_video_fps(self.pending_timestamps[topic])
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(paths[topic]), fourcc, selected_fps, self.frame_sizes[topic]
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(
                    f"OpenCV could not open {paths[topic]} with codec {codec!r}"
                )
            self.writers[topic] = writer
            self.writer_fps[topic] = selected_fps
            for frame in frames:
                writer.write(frame)
            frames.clear()
            self.pending_timestamps[topic].clear()
            fps_source = "configured" if fps is not None else "estimated"
            self.get_logger().info(
                f"Writing {topic} to {paths[topic]} at {selected_fps:.3f} FPS ({fps_source})"
            )

        def close(self) -> None:
            try:
                for topic in topics:
                    self.open_writer_and_flush(topic)
            finally:
                for writer in self.writers.values():
                    writer.release()

    if len(codec) != 4:
        raise ValueError("--codec must contain exactly four characters, for example mp4v")
    if fps is not None and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("--fps must be finite and greater than zero")

    # Keep this process's SIGTERM/SIGHUP handlers. rclpy's defaults would
    # replace SIGTERM and raise ExternalShutdownException before our clean
    # writer-finalization path can mark the stop as intentional.
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = ImageRecorder()
    try:
        topic_deadline = time.monotonic() + topic_wait_timeout_s
        missing = set(topics)
        wrong_types: dict[str, list[str]] = {}
        while missing and time.monotonic() < topic_deadline and not _STOP_REQUESTED:
            publisher_types = {
                topic: sorted(
                    {endpoint.topic_type for endpoint in node.get_publishers_info_by_topic(topic)}
                )
                for topic in topics
            }
            missing = {topic for topic, types in publisher_types.items() if not types}
            wrong_types = {
                topic: types
                for topic, types in publisher_types.items()
                if types and "sensor_msgs/msg/Image" not in types
            }
            if wrong_types:
                details = ", ".join(f"{topic}={types}" for topic, types in wrong_types.items())
                raise TypeError(f"MP4 mode only supports sensor_msgs/msg/Image topics: {details}")
            if missing:
                rclpy.spin_once(node, timeout_sec=0.1)
        if missing and not _STOP_REQUESTED:
            raise TimeoutError(
                "Timed out waiting for image topic publishers: " + ", ".join(sorted(missing))
            )

        if not _STOP_REQUESTED:
            # Subscriptions are intentionally created only after every
            # publisher passes discovery. Otherwise the node would discover
            # its own subscription as a topic, and early streams could exceed
            # --duration-s while waiting for a later camera.
            node.start_subscriptions()
            print("Recording MP4; press Ctrl+C to stop.", flush=True)
            deadline = None if duration_s is None else time.monotonic() + duration_s
            while (
                rclpy.ok()
                and not _STOP_REQUESTED
                and (deadline is None or time.monotonic() < deadline)
            ):
                rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.close()
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    if _STOP_REQUESTED and not any(node.frame_counts.values()):
        print("MP4 recording stopped before the first frame was received.", flush=True)
        return
    empty_topics = [topic for topic, count in node.frame_counts.items() if count == 0]
    if empty_topics:
        raise RuntimeError("No frames were received from: " + ", ".join(empty_topics))
    for topic, path in paths.items():
        print(
            f"{topic}: {node.frame_counts[topic]} frames at "
            f"{node.writer_fps[topic]:.3f} FPS -> {path}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    aliases = ", ".join(sorted(TOPIC_ALIASES))
    parser = argparse.ArgumentParser(
        description=(
            "Record selected ROS 2 topics. Auto mode writes camera-only selections as MP4; "
            "all mixed/non-video selections use MCAP."
        )
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        metavar="TOPIC",
        help=f"Aliases or absolute ROS topics (aliases: {aliases})",
    )
    parser.add_argument("--format", choices=("auto", "mcap", "mp4"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        help="MCAP output directory, MP4 directory, or one .mp4 path for one camera",
    )
    parser.add_argument("--duration-s", type=float, help="Stop cleanly after this many seconds")
    parser.add_argument(
        "--fps",
        type=float,
        help="MP4 playback frame rate (default: estimate each stream from ROS timestamps)",
    )
    parser.add_argument("--codec", default="mp4v", help="FourCC used by OpenCV for MP4")
    parser.add_argument(
        "--topic-wait-timeout-s",
        type=float,
        default=10.0,
        help="How long MP4 mode waits for selected image publishers",
    )
    parser.add_argument(
        "--mcap-storage-profile",
        choices=("none", "fastwrite", "zstd_fast", "zstd_small"),
        default="zstd_fast",
    )
    parser.add_argument(
        "--qos-profile-overrides-path",
        type=Path,
        help="Optional rosbag2 QoS overrides YAML for selected topics",
    )
    parser.add_argument("--list-aliases", action="store_true", help="Print aliases and exit")
    return parser


def _validate_positive_optional(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be finite and greater than zero")


def main(argv: Sequence[str] | None = None) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    _install_stop_signal_handlers()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_aliases:
        for alias, topic in TOPIC_ALIASES.items():
            print(f"{alias:16} {topic}")
        return 0
    if not args.topics:
        parser.error("--topics is required unless --list-aliases is used")

    try:
        _validate_positive_optional(args.duration_s, "--duration-s")
        _validate_positive_optional(args.topic_wait_timeout_s, "--topic-wait-timeout-s")
        _validate_positive_optional(args.fps, "--fps")
        topics = resolve_topics(args.topics)
        output_format = select_output_format(args.format, topics)
        output = (args.output or default_output_path(output_format)).expanduser().resolve()
        print(f"Selected topics: {', '.join(topics)}", flush=True)
        print(f"Output format: {output_format}", flush=True)
        if output_format == "mcap":
            record_mcap(
                topics,
                output,
                duration_s=args.duration_s,
                storage_profile=args.mcap_storage_profile,
                qos_overrides_path=(
                    args.qos_profile_overrides_path.expanduser().resolve()
                    if args.qos_profile_overrides_path is not None
                    else None
                ),
            )
        else:
            record_mp4(
                topics,
                output,
                duration_s=args.duration_s,
                fps=args.fps,
                codec=args.codec,
                topic_wait_timeout_s=args.topic_wait_timeout_s,
            )
    except (FileExistsError, FileNotFoundError, RuntimeError, TimeoutError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
