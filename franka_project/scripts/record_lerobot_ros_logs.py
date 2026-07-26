#!/usr/bin/env python3
"""Record the LeRobot -> ROS -> Franka execution chain into per-topic JSONL files."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import rclpy
from franka_safety_interfaces.msg import SafetyCommandFeedback
from lerobot_franka_interfaces.msg import (
    CartesianActionChunk,
    CartesianActionChunkAck,
    SafetyGatewayStatus,
)
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


SCHEMA_VERSION = 1
CORE_TOPICS = {
    "action_chunk": "/lerobot/franka/action_chunk",
    "action_chunk_ack": "/lerobot/franka/action_chunk_ack",
    "safety_gateway_status": "/lerobot/franka/safety_gateway_status",
    "safety_command_feedback": "/franka/safety_command_feedback",
}
SELECTED_OPTION_NAMES = (
    "host",
    "port",
    "server_address",
    "task",
    "policy_type",
    "pretrained_name_or_path",
    "policy_device",
    "client_device",
    "actions_per_chunk",
    "fps",
    "inference_latency",
    "obs_queue_timeout",
    "observation_similarity_mode",
    "action_offset",
    "chunk_size_threshold",
    "aggregate_fn_name",
    "enable_pending_observation",
    "pending_observation_timeout_s",
    "observation_trigger_mode",
    "post_action_observation_delay_s",
    "debug_visualize_queue_size",
    "robot.type",
    "robot.id",
    "robot.dry_run",
    "robot.ros2_interface_only",
    "robot.ros2_node_name",
    "robot.max_action_chunk_waypoints",
    "robot.action_chunk_topic",
    "robot.camera1_topic",
    "robot.camera2_topic",
    "robot.eef_pose_topic",
    "robot.qpos_topic",
    "robot.gripper_topic",
    "robot.base_frame",
    "robot.observation_buffer_size",
    "robot.max_observation_age_s",
    "robot.camera2_max_skew_s",
    "robot.eef_max_skew_s",
    "robot.qpos_max_skew_s",
    "robot.gripper_max_skew_s",
    "robot.action_chunk_validity_s",
    "rename_map",
)
RELATED_PROCESS_MARKERS = (
    "lerobot_robot_franka_ros.ros2_client",
    "serve_franka_pi0_async",
    "franka_cartesian_safety_gateway",
    "joint_impedance_ik_controller",
    "ws_tcp_tunnel",
)

ACK_RESULT_NAMES = {
    0: "ACCEPTED_SHADOW",
    1: "ACCEPTED_FOR_EXECUTION",
    2: "ACCEPTED_PREFLIGHT_ONLY",
    10: "REJECTED_SCHEMA",
    11: "REJECTED_FRAME",
    12: "REJECTED_SHAPE",
    13: "REJECTED_NONFINITE",
    14: "REJECTED_QUATERNION",
    15: "REJECTED_GRIPPER",
    16: "REJECTED_ORDERING",
    17: "REJECTED_EXPIRED",
    18: "REJECTED_SCHEDULE",
    19: "REJECTED_WORKSPACE",
    20: "REJECTED_CARTESIAN_STEP",
    21: "REJECTED_NOT_ARMED",
    22: "REJECTED_STATE_STALE",
    23: "REJECTED_PREFLIGHT_UNAVAILABLE",
}
GATEWAY_STATE_NAMES = {0: "DISABLED", 1: "SHADOW", 2: "ARMED", 3: "HOLD", 4: "FAULT"}
FEEDBACK_STATUS_NAMES = {
    0: "REJECTED",
    1: "ACCEPTED",
    2: "APPLIED",
    3: "STALE",
    4: "HOLDING",
}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _time_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def _duration_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def _receive_fields() -> dict[str, Any]:
    return {
        "received_at": _now_iso(),
        "received_monotonic_ns": time.monotonic_ns(),
    }


def _header_fields(message: Any) -> dict[str, Any]:
    return {
        "header_stamp_ns": _time_ns(message.header.stamp),
        "header_frame_id": message.header.frame_id,
    }


def _run_command(command: list[str], *, cwd: Path | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def _git_snapshot(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"path": str(path), "available": False}
    head = _run_command(["git", "rev-parse", "HEAD"], cwd=path)
    branch = _run_command(["git", "branch", "--show-current"], cwd=path)
    status = _run_command(["git", "status", "--short"], cwd=path)
    return {
        "path": str(path),
        "available": True,
        "head": head.get("stdout", "").strip(),
        "branch": branch.get("stdout", "").strip(),
        "dirty": bool(status.get("stdout", "").strip()),
        "status": status.get("stdout", "").splitlines(),
    }


def _extract_cli_options(argv: list[str]) -> dict[str, list[str]]:
    """Preserve repeated allow-listed CLI values in command-line order."""

    options: dict[str, list[str]] = {}
    index = 0
    allowed = set(SELECTED_OPTION_NAMES)
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            index += 1
            continue
        option = token[2:]
        if "=" in option:
            name, value = option.split("=", 1)
        else:
            name = option
            value = "true"
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                index += 1
                value = argv[index]
        if name in allowed:
            options.setdefault(name, []).append(value)
        index += 1
    return options


def _related_processes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        command_text = " ".join(argv)
        if not argv or not any(marker in command_text for marker in RELATED_PROCESS_MARKERS):
            continue
        record: dict[str, Any] = {"pid": int(entry.name), "argv": argv}
        selected_options = _extract_cli_options(argv)
        if selected_options:
            record["selected_options"] = selected_options
            record["selected_option_last_values"] = {
                name: values[-1] for name, values in selected_options.items() if values
            }
        if "lerobot_robot_franka_ros.ros2_client" in command_text:
            record["client_options"] = selected_options
            record["client_option_last_values"] = {
                name: values[-1] for name, values in selected_options.items() if values
            }
        records.append(record)
    return sorted(records, key=lambda item: item["pid"])


def _merge_related_processes(
    observed: dict[tuple[int, tuple[str, ...]], dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Keep every related PID/argv combination observed during the run."""

    for record in records:
        key = (int(record["pid"]), tuple(record["argv"]))
        observed[key] = record


def _create_run_directory(output_root: Path, label: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")
        if safe_label:
            run_id = f"{run_id}_{safe_label}"
    candidate = output_root / run_id
    suffix = 0
    while True:
        run_dir = candidate if suffix == 0 else output_root / f"{run_id}_{suffix}"
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1
    latest = output_root / "latest"
    if latest.is_symlink() or not latest.exists():
        latest.unlink(missing_ok=True)
        latest.symlink_to(run_dir.name)
    return run_dir


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_runtime_snapshot(path: Path) -> None:
    commands: list[tuple[str, list[str]]] = [
        ("ros_nodes", ["ros2", "node", "list", "--no-daemon"]),
        ("ros_topics", ["ros2", "topic", "list", "-t", "--no-daemon"]),
        ("controllers", ["ros2", "control", "list_controllers"]),
        (
            "gateway_parameters",
            ["ros2", "param", "dump", "/franka_cartesian_safety_gateway"],
        ),
        (
            "controller_parameters",
            ["ros2", "param", "dump", "/joint_impedance_ik_controller"],
        ),
        ("client_ros_parameters", ["ros2", "param", "dump", "/lerobot_franka_interface"]),
    ]
    for name, topic in CORE_TOPICS.items():
        commands.append(
            (
                f"topic_info_{name}",
                ["ros2", "topic", "info", "--verbose", topic, "--no-daemon"],
            )
        )
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write(f"snapshot_at: {_now_iso()}\n")
        for title, command in commands:
            stream.write(f"\n===== {title} =====\n")
            result = _run_command(command)
            stream.write(f"command: {' '.join(command)}\n")
            if "error" in result:
                stream.write(f"error: {result['error']}\n")
                continue
            stream.write(f"returncode: {result['returncode']}\n")
            stream.write(result.get("stdout", ""))
            if result.get("stderr"):
                stream.write("\n--- stderr ---\n")
                stream.write(result["stderr"])


class EventLog:
    def __init__(self, path: Path) -> None:
        self._stream = path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        values = " ".join(f"{name}={value!r}" for name, value in fields.items())
        with self._lock:
            self._stream.write(f"{_now_iso()} {event} {values}".rstrip() + "\n")

    def close(self) -> None:
        self._stream.close()


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._stream = path.open("w", encoding="utf-8", buffering=1)
        self.count = 0

    def write(self, value: dict[str, Any]) -> None:
        self._stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        self._stream.close()


class LeRobotRosLogRecorder(Node):
    def __init__(self, run_dir: Path, events: EventLog) -> None:
        super().__init__(f"lerobot_ros_log_recorder_{os.getpid()}")
        self._events = events
        self._writers = {
            "action_chunk": JsonlWriter(run_dir / "10_action_chunk.jsonl"),
            "action_chunk_summary": JsonlWriter(run_dir / "11_action_chunk_summary.jsonl"),
            "action_chunk_ack": JsonlWriter(run_dir / "12_action_chunk_ack.jsonl"),
            "safety_gateway_status": JsonlWriter(run_dir / "13_safety_gateway_status.jsonl"),
            "safety_feedback_applied": JsonlWriter(
                run_dir / "14_safety_command_feedback_applied.jsonl"
            ),
            "safety_feedback_other": JsonlWriter(
                run_dir / "15_safety_command_feedback_other.jsonl"
            ),
        }
        action_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        feedback_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1000,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._subscriptions = [
            self.create_subscription(
                CartesianActionChunk,
                CORE_TOPICS["action_chunk"],
                self._on_action_chunk,
                action_qos,
            ),
            self.create_subscription(
                CartesianActionChunkAck,
                CORE_TOPICS["action_chunk_ack"],
                self._on_action_chunk_ack,
                action_qos,
            ),
            self.create_subscription(
                SafetyGatewayStatus,
                CORE_TOPICS["safety_gateway_status"],
                self._on_gateway_status,
                action_qos,
            ),
            self.create_subscription(
                SafetyCommandFeedback,
                CORE_TOPICS["safety_command_feedback"],
                self._on_safety_feedback,
                feedback_qos,
            ),
        ]
        self._events.write("RECORDER_READY", node=self.get_name())

    @property
    def counts(self) -> dict[str, int]:
        return {name: writer.count for name, writer in self._writers.items()}

    def close_files(self) -> None:
        for writer in self._writers.values():
            writer.close()

    def _on_action_chunk(self, message: CartesianActionChunk) -> None:
        period_ns = _duration_ns(message.period)
        timestep_count = len(message.timesteps)
        pose_count = len(message.poses)
        gripper_count = len(message.gripper)
        record = {
            **_receive_fields(),
            **_header_fields(message),
            "schema_version": int(message.schema_version),
            "session_id": message.session_id,
            "plan_id": int(message.plan_id),
            "source_timestep": int(message.source_timestep),
            "client_observation_stamp_ns": _time_ns(message.client_observation_stamp),
            "server_send_stamp_ns": _time_ns(message.server_send_stamp),
            "valid_until_ns": _time_ns(message.valid_until),
            "period_ns": period_ns,
            "waypoint_count": timestep_count,
            "timestep_count": timestep_count,
            "pose_count": pose_count,
            "gripper_count": gripper_count,
            "sequence_lengths_match": timestep_count == pose_count == gripper_count,
            "timesteps": [int(value) for value in message.timesteps],
            "poses": [
                {
                    "position": [pose.position.x, pose.position.y, pose.position.z],
                    "orientation_xyzw": [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ],
                }
                for pose in message.poses
            ],
            "gripper": [float(value) for value in message.gripper],
        }
        self._writers["action_chunk"].write(record)
        timesteps = record["timesteps"]
        self._writers["action_chunk_summary"].write(
            {
                **_receive_fields(),
                "session_id": message.session_id,
                "plan_id": int(message.plan_id),
                "source_timestep": int(message.source_timestep),
                "actual_waypoint_count": timestep_count,
                "timestep_count": timestep_count,
                "pose_count": pose_count,
                "gripper_count": gripper_count,
                "sequence_lengths_match": timestep_count == pose_count == gripper_count,
                "first_timestep": timesteps[0] if timesteps else None,
                "last_timestep": timesteps[-1] if timesteps else None,
                "period_ns": period_ns,
                "nominal_horizon_ns": period_ns * len(timesteps),
            }
        )

    def _on_action_chunk_ack(self, message: CartesianActionChunkAck) -> None:
        result = int(message.result)
        self._writers["action_chunk_ack"].write(
            {
                **_receive_fields(),
                **_header_fields(message),
                "schema_version": int(message.schema_version),
                "session_id": message.session_id,
                "plan_id": int(message.plan_id),
                "result": result,
                "result_name": ACK_RESULT_NAMES.get(result, "UNKNOWN"),
                "accepted": bool(message.accepted),
                "replaced_active_plan": bool(message.replaced_active_plan),
                "waypoint_count": int(message.waypoint_count),
                "detail": message.detail,
            }
        )

    def _on_gateway_status(self, message: SafetyGatewayStatus) -> None:
        state = int(message.state)
        self._writers["safety_gateway_status"].write(
            {
                **_receive_fields(),
                **_header_fields(message),
                "schema_version": int(message.schema_version),
                "state": state,
                "state_name": GATEWAY_STATE_NAMES.get(state, "UNKNOWN"),
                "shadow": bool(message.shadow),
                "armed": bool(message.armed),
                "state_fresh": bool(message.state_fresh),
                "preflight_available": bool(message.preflight_available),
                "robot_ready": bool(message.robot_ready),
                "controller_ready": bool(message.controller_ready),
                "applied_feedback_fresh": bool(message.applied_feedback_fresh),
                "has_active_plan": bool(message.has_active_plan),
                "session_id": message.session_id,
                "plan_id": int(message.plan_id),
                "next_waypoint_index": int(message.next_waypoint_index),
                "accepted_waypoint_index": int(message.accepted_waypoint_index),
                "applied_waypoint_index": int(message.applied_waypoint_index),
                "measured_waypoint_index": int(message.measured_waypoint_index),
                "last_command_sequence": int(message.last_command_sequence),
                "last_applied_sequence": int(message.last_applied_sequence),
                "accepted_source_timestep": int(message.accepted_source_timestep),
                "applied_source_timestep": int(message.applied_source_timestep),
                "measured_source_timestep": int(message.measured_source_timestep),
                "joint_tracking_error_rad": float(message.joint_tracking_error_rad),
                "tcp_position_error_m": float(message.tcp_position_error_m),
                "tcp_orientation_error_rad": float(message.tcp_orientation_error_rad),
                "applied_feedback_age_ns": _duration_ns(message.applied_feedback_age),
                "robot_readiness_age_ns": _duration_ns(message.robot_readiness_age),
                "valid_until_ns": _time_ns(message.valid_until),
                "next_waypoint_due_ns": _time_ns(message.next_waypoint_due),
                "detail": message.detail,
            }
        )

    def _on_safety_feedback(self, message: SafetyCommandFeedback) -> None:
        status = int(message.status)
        record = {
            **_receive_fields(),
            **_header_fields(message),
            "session_id": message.session_id,
            "plan_id": int(message.plan_id),
            "waypoint_index": int(message.waypoint_index),
            "source_timestep": int(message.source_timestep),
            "sequence": int(message.sequence),
            "accepted": bool(message.accepted),
            "applied": bool(message.applied),
            "status": status,
            "status_name": FEEDBACK_STATUS_NAMES.get(status, "UNKNOWN"),
            "reason": message.reason,
            "commanded_positions": [float(value) for value in message.commanded_positions],
            "measured_positions": [float(value) for value in message.measured_positions],
            "command_received_at_ns": _time_ns(message.command_received_at),
            "command_applied_at_ns": _time_ns(message.command_applied_at),
        }
        # Exact equivalent of: --filter 'm.applied and m.status == 2'.
        writer_name = (
            "safety_feedback_applied"
            if message.applied and status == SafetyCommandFeedback.STATUS_APPLIED
            else "safety_feedback_other"
        )
        self._writers[writer_name].write(record)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record LeRobot/Franka ROS execution topics into one timestamped run directory."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "franka" / "logs" / "lerobot_runs",
        help="Parent directory. A YYYYmmdd_HHMMSS_mmm child is created for this run.",
    )
    parser.add_argument("--label", help="Optional safe suffix appended to the timestamp directory.")
    parser.add_argument(
        "--duration-s",
        type=float,
        help="Stop automatically after this many seconds; otherwise run until Ctrl+C.",
    )
    parser.add_argument(
        "--skip-runtime-snapshot",
        action="store_true",
        help="Skip ros2 graph/parameter snapshots (topic recording is unchanged).",
    )
    parser.add_argument(
        "--snapshot-at-end",
        action="store_true",
        help="Also take a second graph/parameter snapshot while stopping.",
    )
    args = parser.parse_args()
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error("--duration-s must be greater than zero")
    return args


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    run_dir = _create_run_directory(output_root, args.label)
    events = EventLog(run_dir / "00_events.log")
    events.write("RUN_DIRECTORY_CREATED", path=str(run_dir))

    start_time = _now_iso()
    related_processes_start = _related_processes()
    observed_related_processes: dict[tuple[int, tuple[str, ...]], dict[str, Any]] = {}
    _merge_related_processes(observed_related_processes, related_processes_start)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "start_time": start_time,
        "end_time": None,
        "stop_reason": None,
        "run_directory": str(run_dir),
        "host": socket.gethostname(),
        "user": getpass.getuser(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "ros_environment": {
            name: os.environ[name]
            for name in (
                "ROS_DISTRO",
                "ROS_DOMAIN_ID",
                "RMW_IMPLEMENTATION",
                "CUDA_VISIBLE_DEVICES",
                "PYTHONPATH",
            )
            if name in os.environ
        },
        "topics": CORE_TOPICS,
        "feedback_applied_filter": "m.applied and m.status == 2",
        "notes": [
            "actions_per_chunk is a Client/Server CLI setting, not a ROS topic.",
            "action_chunk_summary.actual_waypoint_count is the number actually published to ROS.",
            "No /franka/safe_joint_command subscription is created, so "
            "controller-subscriber safety gates are unchanged.",
        ],
        "related_processes_start": related_processes_start,
        "related_processes_end": [],
        "related_processes": related_processes_start,
        "git": {
            "lerobot": _git_snapshot(Path.home() / "Projects" / "lerobot"),
            "franka": _git_snapshot(Path.home() / "franka"),
        },
        "message_counts": {},
    }
    _write_json(run_dir / "01_manifest.json", manifest)
    disk = shutil.disk_usage(output_root)
    events.write("DISK_SPACE", free_bytes=disk.free, total_bytes=disk.total)

    rclpy.init()
    node: LeRobotRosLogRecorder | None = None
    executor: SingleThreadedExecutor | None = None
    snapshot_thread: threading.Thread | None = None
    stop_reason = "completed"
    exit_code = 0
    try:
        node = LeRobotRosLogRecorder(run_dir, events)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        if not args.skip_runtime_snapshot:
            snapshot_thread = threading.Thread(
                target=_write_runtime_snapshot,
                args=(run_dir / "02_runtime_snapshot_start.log",),
                name="lerobot-runtime-snapshot",
                daemon=True,
            )
            snapshot_thread.start()
        deadline = time.monotonic() + args.duration_s if args.duration_s is not None else None
        next_process_scan = time.monotonic() + 5.0
        print(f"LeRobot ROS logs: {run_dir}", flush=True)
        print("Stop with Ctrl+C.", flush=True)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
            now_monotonic = time.monotonic()
            if now_monotonic >= next_process_scan:
                _merge_related_processes(observed_related_processes, _related_processes())
                next_process_scan = now_monotonic + 5.0
            if deadline is not None and now_monotonic >= deadline:
                stop_reason = "duration_elapsed"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except BaseException as exc:
        stop_reason = f"error:{type(exc).__name__}"
        exit_code = 1
        events.write("RECORDER_ERROR", error=repr(exc))
        raise
    finally:
        related_processes_end = _related_processes()
        _merge_related_processes(observed_related_processes, related_processes_end)
        manifest["related_processes_end"] = related_processes_end
        manifest["related_processes"] = sorted(
            observed_related_processes.values(), key=lambda item: item["pid"]
        )
        if snapshot_thread is not None:
            snapshot_thread.join(timeout=2.0)
            if snapshot_thread.is_alive():
                events.write("START_SNAPSHOT_STILL_RUNNING")
        if not args.skip_runtime_snapshot and args.snapshot_at_end:
            _write_runtime_snapshot(run_dir / "03_runtime_snapshot_end.log")
        if node is not None:
            manifest["message_counts"] = node.counts
            events.write("RECORDER_STOPPING", reason=stop_reason, counts=node.counts)
            if executor is not None:
                executor.remove_node(node)
            node.destroy_node()
            node.close_files()
        if executor is not None:
            executor.shutdown(timeout_sec=1.0)
        if rclpy.ok():
            rclpy.shutdown()
        manifest["end_time"] = _now_iso()
        manifest["stop_reason"] = stop_reason
        manifest["exit_code"] = exit_code
        _write_json(run_dir / "01_manifest.json", manifest)
        events.write("RECORDER_STOPPED", reason=stop_reason, exit_code=exit_code)
        events.close()
        print(f"Logs finalized: {run_dir}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
