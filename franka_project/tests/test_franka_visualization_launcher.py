from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_MODULE = (
    PROJECT_ROOT / "ros_lerobot/src/lerobot_robot_franka_ros/visualization_launcher.py"
)
SPEC = importlib.util.spec_from_file_location("franka_visualization_launcher", LAUNCHER_MODULE)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _spec(*, launch_rviz: bool = True) -> Any:
    return launcher.ActionVisualizationLaunchSpec(
        launch_rviz=launch_rviz,
        fixed_frame="base",
        action_chunk_topic="/custom/action_chunk",
        current_pose_topic="/custom/current_pose",
        camera1_topic="/custom/camera1",
        camera2_topic="/custom/camera2",
    )


class FakeProcess:
    def __init__(self, waits: list[int | BaseException], *, poll_result: int | None = None) -> None:
        self.pid = 4321
        self.waits = waits
        self.poll_result = poll_result
        self.wait_timeouts: list[float | None] = []
        self.sent_signals: list[signal.Signals] = []

    def poll(self) -> int | None:
        return self.poll_result

    def send_signal(self, sig: signal.Signals) -> None:
        self.sent_signals.append(sig)

    def wait(self, timeout: float | None = None) -> int:
        self.wait_timeouts.append(timeout)
        if not self.waits:
            raise AssertionError("Unexpected wait() call")
        result = self.waits.pop(0)
        if isinstance(result, BaseException):
            raise result
        self.poll_result = result
        return result


def test_launch_command_uses_resolved_client_topics_without_a_shell() -> None:
    command = _spec(launch_rviz=False).command()

    assert command == [
        "ros2",
        "launch",
        "franka_lerobot_rviz",
        "action_viz.launch.py",
        "launch_rviz:=false",
        "fixed_frame:=base",
        "action_chunk_topic:=/custom/action_chunk",
        "current_pose_topic:=/custom/current_pose",
        "camera1_topic:=/custom/camera1",
        "camera2_topic:=/custom/camera2",
    ]


def test_top_level_client_config_survives_resolution_and_drives_launch_topics() -> None:
    pytest.importorskip("draccus")
    plugin_src = PROJECT_ROOT / "ros_lerobot/src"
    sys.path.insert(0, str(plugin_src))

    from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig
    from lerobot_robot_franka_ros.ros2_client import (
        FrankaRos2ClientConfig,
        _make_action_visualization_launcher,
        resolve_franka_client_policy_config,
    )

    config = FrankaRos2ClientConfig(
        robot=FrankaRosConfig(
            id="visualization-test",
            action_chunk_topic="/lerobot/configured/action_chunk",
            eef_pose_topic="/configured/current_pose",
            camera1_topic="/configured/camera1",
            camera2_topic="/configured/camera2",
        ),
        policy_type="pi0",
        action_offset=1,
        aggregate_fn_name="latest_only",
        visualize_action=True,
        visualization_launch_rviz=False,
    )

    resolved = resolve_franka_client_policy_config(config)
    assert isinstance(resolved, FrankaRos2ClientConfig)
    assert resolved.visualize_action is True
    assert resolved.visualization_launch_rviz is False
    manager = _make_action_visualization_launcher(resolved, logger=launcher.logging.getLogger())
    assert manager.spec.command()[4:] == [
        "launch_rviz:=false",
        "fixed_frame:=base",
        "action_chunk_topic:=/lerobot/configured/action_chunk",
        "current_pose_topic:=/configured/current_pose",
        "camera1_topic:=/configured/camera1",
        "camera2_topic:=/configured/camera2",
    ]


def test_launcher_starts_a_new_process_group_and_reaps_it_with_sigint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(
        [
            subprocess.TimeoutExpired(cmd="ros2", timeout=0.01),
            0,
        ]
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []
    signals: list[tuple[int, signal.Signals]] = []

    def process_factory(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((command, kwargs))
        return process

    def no_remaining_process_group(pid: int, sig: int | signal.Signals) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(launcher.os, "killpg", no_remaining_process_group)
    manager = launcher.ActionVisualizationLauncher(
        _spec(),
        startup_timeout_s=0.01,
        shutdown_timeout_s=0.01,
        process_factory=process_factory,
    )

    manager.start()
    assert manager.is_running
    manager.stop()

    assert calls == [
        (
            _spec().command(),
            {"shell": False, "start_new_session": True},
        )
    ]
    assert process.sent_signals == [signal.SIGINT]
    assert signals == []
    assert not manager.is_running


def test_launcher_cleans_child_group_after_launch_parent_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess(
        [
            subprocess.TimeoutExpired(cmd="ros2", timeout=0.01),
            0,
        ]
    )
    group_alive = True
    group_signals: list[signal.Signals] = []

    def process_group_signal(pid: int, sig: int | signal.Signals) -> None:
        nonlocal group_alive
        if not group_alive:
            raise ProcessLookupError
        if sig == 0:
            return
        group_signals.append(signal.Signals(sig))
        group_alive = False

    monkeypatch.setattr(launcher.os, "killpg", process_group_signal)
    manager = launcher.ActionVisualizationLauncher(
        _spec(),
        startup_timeout_s=0.01,
        shutdown_timeout_s=0.01,
        process_factory=lambda *args, **kwargs: process,
    )

    manager.start()
    manager.stop()

    assert process.sent_signals == [signal.SIGINT]
    assert group_signals == [signal.SIGTERM]
    assert not manager.is_running


def test_launcher_reports_missing_ros2_command_clearly() -> None:
    def missing_process(*args: Any, **kwargs: Any) -> FakeProcess:
        raise FileNotFoundError("ros2")

    manager = launcher.ActionVisualizationLauncher(_spec(), process_factory=missing_process)
    with pytest.raises(launcher.ActionVisualizationLaunchError, match="Ensure ROS 2 is sourced"):
        manager.start()


def test_launcher_rejects_a_launch_that_exits_during_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess([2])
    signals: list[signal.Signals] = []

    def process_group_signal(pid: int, sig: int | signal.Signals) -> None:
        if sig == 0:
            raise ProcessLookupError
        signals.append(signal.Signals(sig))

    monkeypatch.setattr(launcher.os, "killpg", process_group_signal)
    manager = launcher.ActionVisualizationLauncher(
        _spec(),
        process_factory=lambda *args, **kwargs: process,
    )

    with pytest.raises(
        launcher.ActionVisualizationLaunchError,
        match="exited during startup with status 2",
    ):
        manager.start()
    assert signals == [signal.SIGTERM]
    assert not manager.is_running


def test_launcher_escalates_and_reaps_the_whole_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = subprocess.TimeoutExpired(cmd="ros2", timeout=0.01)
    process = FakeProcess([timeout, timeout, timeout, 0])
    signals: list[signal.Signals] = []
    group_alive = True

    def process_group_signal(pid: int, sig: int | signal.Signals) -> None:
        nonlocal group_alive
        if not group_alive:
            raise ProcessLookupError
        if sig == 0:
            return
        normalized = signal.Signals(sig)
        signals.append(normalized)
        if normalized == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(launcher.os, "killpg", process_group_signal)
    manager = launcher.ActionVisualizationLauncher(
        _spec(),
        startup_timeout_s=0.01,
        shutdown_timeout_s=0.01,
        process_factory=lambda *args, **kwargs: process,
    )

    manager.start()
    manager.stop()

    assert process.sent_signals == [signal.SIGINT]
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert not manager.is_running


def test_client_finally_reaps_visualizer_and_joins_thread_when_client_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("draccus")
    plugin_src = PROJECT_ROOT / "ros_lerobot/src"
    sys.path.insert(0, str(plugin_src))

    from lerobot_robot_franka_ros import ros2_client
    from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig

    config = ros2_client.FrankaRos2ClientConfig(
        robot=FrankaRosConfig(id="cleanup-test"),
        policy_type="pi0",
        action_offset=1,
        aggregate_fn_name="latest_only",
        visualize_action=True,
    )
    events: list[str] = []

    class FakeLogger:
        def info(self, *args: Any, **kwargs: Any) -> None:
            events.append("log.info")

        def exception(self, *args: Any, **kwargs: Any) -> None:
            events.append("log.exception")

    class FakeClient:
        logger = FakeLogger()
        action_queue_size: list[int] = []

        def start(self) -> bool:
            events.append("client.start")
            return True

        def stop(self) -> None:
            events.append("client.stop")
            raise RuntimeError("client stop failed")

        def receive_actions(self) -> None:
            events.append("client.receive_actions")

        def control_loop(self, *, task: str) -> None:
            events.append("client.control_loop")

    class FakeLauncher:
        def start(self) -> None:
            events.append("launcher.start")

        def stop(self) -> None:
            events.append("launcher.stop")

    class FakeThread:
        def __init__(self, **kwargs: Any) -> None:
            events.append("thread.init")

        def start(self) -> None:
            events.append("thread.start")

        def join(self) -> None:
            events.append("thread.join")

    monkeypatch.setattr(ros2_client, "resolve_franka_client_policy_config", lambda value: value)
    monkeypatch.setattr(ros2_client, "FrankaRos2RobotClient", lambda value: FakeClient())
    monkeypatch.setattr(
        ros2_client,
        "_make_action_visualization_launcher",
        lambda *args, **kwargs: FakeLauncher(),
    )
    monkeypatch.setattr(ros2_client.threading, "Thread", FakeThread)

    with pytest.raises(RuntimeError, match="client stop failed"):
        ros2_client.ros2_async_client.__wrapped__(config)

    assert "launcher.stop" in events
    assert "thread.join" in events
    assert events.index("client.stop") < events.index("launcher.stop") < events.index("thread.join")
