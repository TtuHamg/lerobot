"""Client-lifecycle launcher for the local Franka web dashboard."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from .visualization_launcher import ActionVisualizationLauncher


def _topic(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be an absolute ROS topic without whitespace")
    return value


@dataclass(frozen=True, slots=True)
class WebDashboardLaunchSpec:
    host: str
    port: int
    history_seconds: float
    camera_fps: float
    open_browser: bool
    default_pose_frame: str
    action_chunk_topic: str
    ack_topic: str
    ik_topic: str
    status_topic: str
    current_pose_topic: str
    robot_state_topic: str
    camera1_topic: str
    camera2_topic: str

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("web dashboard host is restricted to localhost")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("web dashboard port must be in [1, 65535]")
        if not 1.0 <= float(self.history_seconds) <= 600.0:
            raise ValueError("web dashboard history_seconds must be in [1, 600]")
        if not 0.1 <= float(self.camera_fps) <= 10.0:
            raise ValueError("web dashboard camera_fps must be in [0.1, 10]")
        if not isinstance(self.open_browser, bool):
            raise ValueError("web dashboard open_browser must be bool")
        if self.default_pose_frame not in {"eef", "link8"}:
            raise ValueError("web dashboard default_pose_frame must be 'eef' or 'link8'")
        for name in (
            "action_chunk_topic",
            "ack_topic",
            "ik_topic",
            "status_topic",
            "current_pose_topic",
            "robot_state_topic",
            "camera1_topic",
            "camera2_topic",
        ):
            object.__setattr__(self, name, _topic(getattr(self, name), name))

    def command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "lerobot_robot_franka_ros.web_dashboard",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--history-seconds",
            str(self.history_seconds),
            "--camera-fps",
            str(self.camera_fps),
            "--default-pose-frame",
            self.default_pose_frame,
            "--action-chunk-topic",
            self.action_chunk_topic,
            "--ack-topic",
            self.ack_topic,
            "--ik-topic",
            self.ik_topic,
            "--status-topic",
            self.status_topic,
            "--current-pose-topic",
            self.current_pose_topic,
            "--robot-state-topic",
            self.robot_state_topic,
            "--camera1-topic",
            self.camera1_topic,
            "--camera2-topic",
            self.camera2_topic,
        ]
        command.append("--open-browser" if self.open_browser else "--no-open-browser")
        return command


class WebDashboardLauncher(ActionVisualizationLauncher):
    """Reuse the proven process-group lifecycle for the dashboard subprocess."""

    def __init__(
        self,
        spec: WebDashboardLaunchSpec,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            spec,  # type: ignore[arg-type]  # structural command() contract
            startup_timeout_s=1.5,
            shutdown_timeout_s=5.0,
            logger=logger,
        )


__all__ = ["WebDashboardLaunchSpec", "WebDashboardLauncher"]
