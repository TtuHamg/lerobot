#!/usr/bin/env python3
"""Record the two Franka ROS camera streams directly to MP4 files."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


CAMERA_TOPICS = {
    "camera1": "/camera1/camera1/color/image_raw",
    "camera2": "/camera2/camera2/color/image_raw",
}


def image_to_bgr(message: Image) -> np.ndarray:
    """Convert a stride-aware ROS Image into an OpenCV BGR frame."""

    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    if encoding not in channels_by_encoding:
        raise ValueError(f"unsupported image encoding: {message.encoding!r}")

    channels = channels_by_encoding[encoding]
    packed_step = width * channels
    if step < packed_step:
        raise ValueError(f"image step {step} is smaller than packed row size {packed_step}")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_bytes = height * step
    if raw.size < required_bytes:
        raise ValueError(f"image payload has {raw.size} bytes, expected at least {required_bytes}")
    packed = raw[:required_bytes].reshape(height, step)[:, :packed_step]

    if encoding == "mono8":
        frame = packed.reshape(height, width)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    frame = packed.reshape(height, width, channels)
    conversions = {
        "rgb8": cv2.COLOR_RGB2BGR,
        "rgba8": cv2.COLOR_RGBA2BGR,
        "bgra8": cv2.COLOR_BGRA2BGR,
    }
    conversion = conversions.get(encoding)
    return frame if conversion is None else cv2.cvtColor(frame, conversion)


def message_stamp_ns(message: Image) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


@dataclass
class VideoStream:
    name: str
    topic: str
    output_dir: Path
    fps: float
    codec: str
    writer: Any | None = None
    frame_size: tuple[int, int] | None = None
    frame_count: int = 0
    first_stamp_ns: int | None = None
    last_stamp_ns: int | None = None
    last_frame_monotonic: float | None = None

    @property
    def partial_path(self) -> Path:
        return self.output_dir / f"{self.name}_color.partial.mp4"

    @property
    def final_path(self) -> Path:
        return self.output_dir / f"{self.name}_color.mp4"

    def write(self, message: Image) -> None:
        frame = image_to_bgr(message)
        height, width = frame.shape[:2]
        frame_size = (width, height)
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.partial_path),
                fourcc,
                self.fps,
                frame_size,
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"failed to open video writer: {self.partial_path}")
            self.frame_size = frame_size
        elif frame_size != self.frame_size:
            raise ValueError(f"image size changed from {self.frame_size} to {frame_size}")

        self.writer.write(frame)
        stamp_ns = message_stamp_ns(message)
        self.frame_count += 1
        self.first_stamp_ns = stamp_ns if self.first_stamp_ns is None else self.first_stamp_ns
        self.last_stamp_ns = stamp_ns
        self.last_frame_monotonic = time.monotonic()

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        if self.frame_count > 0 and self.partial_path.exists():
            self.partial_path.replace(self.final_path)

    def metadata(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "video": str(self.final_path),
            "frames": self.frame_count,
            "frame_size": self.frame_size,
            "fps": self.fps,
            "codec": self.codec,
            "first_ros_stamp_ns": self.first_stamp_ns,
            "last_ros_stamp_ns": self.last_stamp_ns,
        }


class DualCameraRecorder:
    def __init__(
        self,
        *,
        output_dir: Path,
        fps: float,
        codec: str,
        startup_timeout_s: float,
        idle_timeout_s: float,
        publisher_grace_s: float,
    ) -> None:
        self.output_dir = output_dir
        self.startup_timeout_s = startup_timeout_s
        self.idle_timeout_s = idle_timeout_s
        self.publisher_grace_s = publisher_grace_s
        self.started_at = datetime.now().astimezone()
        self.stop_requested = False
        self.stop_reason = "unknown"
        self.fatal_error: BaseException | None = None
        self.last_error_log: dict[str, float] = {}
        self.streams = {
            name: VideoStream(name, topic, output_dir, fps, codec)
            for name, topic in CAMERA_TOPICS.items()
        }

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node = Node(f"franka_camera_video_recorder_{os.getpid()}")
        self.subscriptions = [
            self.node.create_subscription(
                Image,
                stream.topic,
                lambda message, stream=stream: self._on_image(stream, message),
                qos,
            )
            for stream in self.streams.values()
        ]

    def _on_image(self, stream: VideoStream, message: Image) -> None:
        if self.fatal_error is not None:
            return
        try:
            stream.write(message)
            if all(item.frame_count > 0 for item in self.streams.values()):
                ready_path = self.output_dir / "READY"
                if not ready_path.exists():
                    ready_path.write_text("both camera videos are recording\n", encoding="utf-8")
                    print(
                        "Recording started: "
                        + ", ".join(str(item.final_path) for item in self.streams.values()),
                        flush=True,
                    )
        except Exception as error:
            now = time.monotonic()
            if now - self.last_error_log.get(stream.name, float("-inf")) >= 5.0:
                self.node.get_logger().error(f"{stream.name}: {error}")
                self.last_error_log[stream.name] = now
            self.fatal_error = error

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_reason = signal.Signals(signum).name
        self.stop_requested = True

    def run(self) -> int:
        startup_deadline = time.monotonic() + self.startup_timeout_s
        publishers_absent_since: float | None = None
        while rclpy.ok() and not self.stop_requested:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            now = time.monotonic()

            if self.fatal_error is not None:
                self.stop_reason = f"fatal_error: {self.fatal_error}"
                return 1

            ready = all(stream.frame_count > 0 for stream in self.streams.values())
            if not ready:
                if now >= startup_deadline:
                    missing = [stream.name for stream in self.streams.values() if stream.frame_count == 0]
                    self.stop_reason = f"startup_timeout: missing {', '.join(missing)}"
                    print(f"ERROR: {self.stop_reason}", file=sys.stderr, flush=True)
                    return 1
                continue

            if all(self.node.count_publishers(stream.topic) == 0 for stream in self.streams.values()):
                publishers_absent_since = publishers_absent_since or now
                if now - publishers_absent_since >= self.publisher_grace_s:
                    self.stop_reason = "camera_publishers_stopped"
                    break
            else:
                publishers_absent_since = None

            stalled = [
                stream.name
                for stream in self.streams.values()
                if stream.last_frame_monotonic is not None
                and now - stream.last_frame_monotonic >= self.idle_timeout_s
            ]
            if stalled:
                self.stop_reason = f"camera_stream_stalled: {', '.join(stalled)}"
                print(f"ERROR: {self.stop_reason}", file=sys.stderr, flush=True)
                return 1
        return 0

    def close(self) -> None:
        for stream in self.streams.values():
            stream.close()
        self.node.destroy_node()
        ended_at = datetime.now().astimezone()
        metadata = {
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_s": (ended_at - self.started_at).total_seconds(),
            "stop_reason": self.stop_reason,
            "streams": {
                name: stream.metadata() for name, stream in self.streams.items()
            },
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Recording stopped ({self.stop_reason}); "
            + ", ".join(
                f"{stream.name}={stream.frame_count} frames" for stream in self.streams.values()
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", default="avc1")
    parser.add_argument("--startup-timeout-s", type=float, default=15.0)
    parser.add_argument("--idle-timeout-s", type=float, default=10.0)
    parser.add_argument("--publisher-grace-s", type=float, default=1.0)
    args = parser.parse_args()
    for name in ("fps", "startup_timeout_s", "idle_timeout_s", "publisher_grace_s"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if len(args.codec) != 4:
        parser.error("--codec must be a four-character code")
    return args


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    recorder = DualCameraRecorder(
        output_dir=args.output_dir,
        fps=args.fps,
        codec=args.codec,
        startup_timeout_s=args.startup_timeout_s,
        idle_timeout_s=args.idle_timeout_s,
        publisher_grace_s=args.publisher_grace_s,
    )
    signal.signal(signal.SIGINT, recorder.request_stop)
    signal.signal(signal.SIGTERM, recorder.request_stop)
    signal.signal(signal.SIGHUP, recorder.request_stop)
    try:
        return recorder.run()
    except KeyboardInterrupt:
        recorder.stop_reason = "KeyboardInterrupt"
        return 0
    finally:
        recorder.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
