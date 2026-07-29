"""Lifecycle wrapper for the optional Franka action RViz launch.

This module intentionally depends only on the Python standard library. The
visualizer remains a separate ROS process, so importing the LeRobot plugin does
not import ``rclpy`` or initialize another ROS context in the client process.
"""

from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class ActionVisualizationLaunchError(RuntimeError):
    """Raised when an explicitly requested action visualizer cannot be run."""


class _LaunchProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def send_signal(self, sig: signal.Signals) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


ProcessFactory = Callable[..., _LaunchProcess]


def _require_nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _require_topic(value: str, *, name: str) -> str:
    topic = _require_nonempty(value, name=name)
    if not topic.startswith("/") or any(character.isspace() for character in topic):
        raise ValueError(f"{name} must be an absolute ROS topic without whitespace")
    return topic


@dataclass(frozen=True, slots=True)
class ActionVisualizationLaunchSpec:
    """ROS launch inputs copied from the resolved Franka client configuration."""

    launch_rviz: bool
    fixed_frame: str
    action_chunk_topic: str
    current_pose_topic: str
    camera1_topic: str
    camera2_topic: str

    def __post_init__(self) -> None:
        if not isinstance(self.launch_rviz, bool):
            raise ValueError("launch_rviz must be a bool")
        object.__setattr__(
            self,
            "fixed_frame",
            _require_nonempty(self.fixed_frame, name="fixed_frame"),
        )
        for field_name in (
            "action_chunk_topic",
            "current_pose_topic",
            "camera1_topic",
            "camera2_topic",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_topic(getattr(self, field_name), name=field_name),
            )

    def command(self) -> list[str]:
        """Return the argv-only launch command; no shell expansion is involved."""

        return [
            "ros2",
            "launch",
            "franka_lerobot_rviz",
            "action_viz.launch.py",
            f"launch_rviz:={'true' if self.launch_rviz else 'false'}",
            f"fixed_frame:={self.fixed_frame}",
            f"action_chunk_topic:={self.action_chunk_topic}",
            f"current_pose_topic:={self.current_pose_topic}",
            f"camera1_topic:={self.camera1_topic}",
            f"camera2_topic:={self.camera2_topic}",
        ]


class ActionVisualizationLauncher:
    """Start and fully reap one isolated ``ros2 launch`` process group."""

    def __init__(
        self,
        spec: ActionVisualizationLaunchSpec,
        *,
        startup_timeout_s: float = 1.0,
        shutdown_timeout_s: float = 5.0,
        process_factory: ProcessFactory = subprocess.Popen,
        logger: logging.Logger | None = None,
    ) -> None:
        if startup_timeout_s <= 0.0:
            raise ValueError("startup_timeout_s must be greater than zero")
        if shutdown_timeout_s <= 0.0:
            raise ValueError("shutdown_timeout_s must be greater than zero")
        self.spec = spec
        self.startup_timeout_s = float(startup_timeout_s)
        self.shutdown_timeout_s = float(shutdown_timeout_s)
        self._process_factory = process_factory
        self._logger = logger or logging.getLogger(__name__)
        self._process: _LaunchProcess | None = None

    @property
    def is_running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    def start(self) -> None:
        if self._process is not None:
            raise ActionVisualizationLaunchError("Action visualization is already started")

        command = self.spec.command()
        rendered_command = shlex.join(command)
        self._logger.info("Starting read-only action visualization: %s", rendered_command)
        try:
            process = self._process_factory(
                command,
                shell=False,
                start_new_session=True,
            )
        except OSError as error:
            raise ActionVisualizationLaunchError(
                "Could not start the requested action visualization. Ensure ROS 2 is sourced and "
                "franka_lerobot_rviz is built in franka_project/ros2_ws. "
                f"Command: {rendered_command}. Error: {error}"
            ) from error

        self._process = process
        try:
            returncode = process.wait(timeout=self.startup_timeout_s)
        except subprocess.TimeoutExpired:
            self._logger.info("Action visualization launch is running (pid=%d)", process.pid)
            return
        except BaseException:
            self.stop()
            raise

        try:
            self._stop_orphaned_process_group()
        finally:
            self._process = None
        raise ActionVisualizationLaunchError(
            "The requested action visualization exited during startup "
            f"with status {returncode}. Command: {rendered_command}"
        )

    def _signal_process_group(self, sig: signal.Signals) -> bool:
        process = self._process
        if process is None:
            return False
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return False
        return True

    def _process_group_exists(self) -> bool:
        process = self._process
        if process is None:
            return False
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _wait_for_process_group_exit(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while self._process_group_exists():
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                return False
            time.sleep(min(0.05, remaining_s))
        return True

    def _stop_orphaned_process_group(self) -> None:
        """Stop launch children when their parent has already exited."""

        timeout_s = min(2.0, self.shutdown_timeout_s)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not self._signal_process_group(sig):
                return
            if self._wait_for_process_group_exit(timeout_s):
                return
            if sig != signal.SIGKILL:
                self._logger.warning(
                    "Orphaned action visualization group survived SIGTERM; escalating"
                )
        raise ActionVisualizationLaunchError(
            "Orphaned action visualization process group did not exit after SIGKILL"
        )

    def stop(self) -> None:
        process = self._process
        if process is None:
            return

        returncode = process.poll()
        if returncode is not None:
            try:
                self._stop_orphaned_process_group()
            finally:
                self._process = None
            self._logger.error(
                "Action visualization exited unexpectedly with status %d",
                returncode,
            )
            return

        try:
            # Signal only ros2 launch first. It owns the normal shutdown fan-out
            # to its child nodes; broadcasting SIGINT to the whole new session
            # would make children receive one signal directly and another from
            # launch, which can interrupt their cleanup a second time.
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                self._stop_orphaned_process_group()
                return
            try:
                process.wait(timeout=self.shutdown_timeout_s)
                self._stop_orphaned_process_group()
                self._logger.info("Action visualization stopped")
                return
            except subprocess.TimeoutExpired:
                self._logger.warning(
                    "Action visualization did not stop after parent SIGINT; escalating"
                )

            group_escalation = (
                (signal.SIGTERM, min(2.0, self.shutdown_timeout_s)),
                (signal.SIGKILL, min(2.0, self.shutdown_timeout_s)),
            )
            for index, (sig, timeout_s) in enumerate(group_escalation):
                self._signal_process_group(sig)
                try:
                    process.wait(timeout=timeout_s)
                    self._stop_orphaned_process_group()
                    self._logger.info("Action visualization stopped")
                    return
                except subprocess.TimeoutExpired:
                    if index + 1 < len(group_escalation):
                        self._logger.warning(
                            "Action visualization did not stop after %s; escalating",
                            sig.name,
                        )
            raise ActionVisualizationLaunchError(
                "Action visualization process group did not exit after SIGKILL"
            )
        finally:
            self._process = None

    def __enter__(self) -> ActionVisualizationLauncher:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()


__all__ = [
    "ActionVisualizationLaunchError",
    "ActionVisualizationLauncher",
    "ActionVisualizationLaunchSpec",
]
