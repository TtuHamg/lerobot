from __future__ import annotations

import importlib.util
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
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

PLUGIN_SRC = PROJECT_ROOT / "ros_lerobot/src"
sys.path.insert(0, str(PLUGIN_SRC))
from lerobot_robot_franka_ros.web_dashboard import RollingStore  # noqa: E402
from lerobot_robot_franka_ros.web_dashboard_launcher import (  # noqa: E402
    WebDashboardLaunchSpec,
)


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


def _web_stamp(value: float) -> SimpleNamespace:
    seconds = int(value)
    return SimpleNamespace(sec=seconds, nanosec=round((value - seconds) * 1e9))


def _web_chunk(*, session: str = "session", plan_id: int = 0, stamp: float = 10.0):
    poses = [
        SimpleNamespace(
            position=SimpleNamespace(x=index * 0.1, y=0.0, z=0.5 + index * 0.01),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
        for index in range(3)
    ]
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_web_stamp(stamp)),
        session_id=session,
        plan_id=plan_id,
        source_timestep=4,
        period=SimpleNamespace(sec=0, nanosec=100_000_000),
        poses=poses,
        gripper=[0.0, 0.5, 1.0],
    )


def test_web_store_bounds_history_and_keeps_monotonic_chunk_idx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [100.0]
    monkeypatch.setattr(
        "lerobot_robot_franka_ros.web_dashboard.time.time",
        lambda: now[0],
    )
    store = RollingStore(history_seconds=1.0)
    assert store.add_chunk(_web_chunk(plan_id=5))["chunk"]["chunk_idx"] == 0
    now[0] = 102.0
    assert store.add_chunk(_web_chunk(plan_id=6, stamp=12.0))["chunk"]["chunk_idx"] == 1
    assert [chunk["plan_id"] for chunk in store.snapshot()["chunks"]] == [6]


def test_web_store_correlates_ik_and_ack_by_session_and_plan() -> None:
    store = RollingStore(history_seconds=60.0)
    key = store.add_chunk(_web_chunk(session="abc", plan_id=7))["chunk"]["key"]
    store.add_ik(
        SimpleNamespace(
            session_id="abc",
            plan_id=7,
            waypoint_count=3,
            positions=[float(value) for value in range(21)],
        )
    )
    store.add_ack(
        SimpleNamespace(
            session_id="abc",
            plan_id=7,
            accepted=True,
            result=1,
            waypoint_count=3,
            detail="accepted",
        )
    )
    chunk = store.snapshot()["chunks"][0]
    assert chunk["key"] == key
    assert chunk["ik_joints"][2] == [float(value) for value in range(14, 21)]
    assert chunk["ack"]["accepted"] is True


def test_web_store_destroys_disabled_camera_history() -> None:
    store = RollingStore(history_seconds=60.0)
    store.set_camera_enabled("camera1", True)
    store.add_camera_frame("camera1", time.time(), b"jpeg")
    assert len(store.snapshot()["cameras"]["camera1"]) == 1
    store.set_camera_enabled("camera1", False)
    snapshot = store.snapshot()
    assert snapshot["camera_enabled"]["camera1"] is False
    assert snapshot["cameras"]["camera1"] == []


def test_web_store_exposes_eef_and_link8_views() -> None:
    from lerobot_robot_franka_ros.ros2_runtime import _compose_pose

    link8_position = np.asarray([0.45, -0.10, 0.40])
    link8_quaternion = np.asarray([0.0, 0.0, 0.0, 1.0])
    tool_position = np.asarray([0.0, 0.0, 0.16])
    tool_quaternion = np.asarray([0.0, 0.0, -(2**-0.5), 2**-0.5])
    eef_position, eef_quaternion = _compose_pose(
        link8_position,
        link8_quaternion,
        tool_position,
        tool_quaternion,
    )
    chunk = _web_chunk()
    for pose in chunk.poses:
        pose.position.x, pose.position.y, pose.position.z = eef_position
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
            eef_quaternion
        )
    store = RollingStore(history_seconds=60.0, default_pose_frame="link8")
    store.add_chunk(chunk)
    assert store.snapshot()["chunks"][0]["link8"] is None
    store.set_tool_transform(tool_position.tolist(), tool_quaternion.tolist())
    snapshot = store.snapshot()
    assert snapshot["default_pose_frame"] == "link8"
    assert snapshot["link8_available"] is True
    assert np.allclose(snapshot["chunks"][0]["link8"][0][:3], link8_position)


def test_web_dashboard_launch_command_is_local_and_browser_optional() -> None:
    spec = WebDashboardLaunchSpec(
        host="127.0.0.1",
        port=8768,
        history_seconds=60.0,
        camera_fps=3.0,
        open_browser=False,
        default_pose_frame="link8",
        action_chunk_topic="/action",
        ack_topic="/ack",
        ik_topic="/ik",
        status_topic="/status",
        current_pose_topic="/pose",
        robot_state_topic="/robot_state",
        camera1_topic="/camera1",
        camera2_topic="/camera2",
    )
    command = spec.command()
    assert command[1:3] == ["-m", "lerobot_robot_franka_ros.web_dashboard"]
    assert command[-1] == "--no-open-browser"
    assert command[command.index("--default-pose-frame") + 1] == "link8"
    with pytest.raises(ValueError, match="restricted to localhost"):
        replace(spec, host="0.0.0.0")


def test_client_web_flag_builds_launcher_and_conflicts_with_rviz() -> None:
    from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig
    from lerobot_robot_franka_ros.ros2_client import (
        FrankaRos2ClientConfig,
        _make_web_dashboard_launcher,
    )

    config = FrankaRos2ClientConfig(
        robot=FrankaRosConfig(id="web-test"),
        policy_type="pi0",
        action_offset=1,
        aggregate_fn_name="latest_only",
        visualize_action_web=True,
        web_dashboard_open_browser=False,
    )
    web_launcher = _make_web_dashboard_launcher(config, logger=launcher.logging.getLogger())
    assert "--no-open-browser" in web_launcher.spec.command()
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(config, visualize_action=True)


def test_link8_policy_pose_round_trip_and_action_canonicalization() -> None:
    from lerobot_robot_franka_ros.ros2_runtime import (
        Ros2Runtime,
        _compose_pose,
        _invert_pose,
    )

    link8_position = np.asarray([0.42, -0.10, 0.40])
    link8_quaternion = np.asarray([0.20, -0.30, 0.10, 0.92])
    link8_quaternion /= np.linalg.norm(link8_quaternion)
    f_t_ee_position = np.asarray([0.0, 0.0, 0.16])
    f_t_ee_quaternion = np.asarray([0.0, 0.0, -2**-0.5, 2**-0.5])

    eef_position, eef_quaternion = _compose_pose(
        link8_position,
        link8_quaternion,
        f_t_ee_position,
        f_t_ee_quaternion,
    )
    inverse_position, inverse_quaternion = _invert_pose(
        f_t_ee_position,
        f_t_ee_quaternion,
    )
    recovered_position, recovered_quaternion = _compose_pose(
        eef_position,
        eef_quaternion,
        inverse_position,
        inverse_quaternion,
    )
    assert np.allclose(recovered_position, link8_position, atol=1e-9)
    assert abs(float(np.dot(recovered_quaternion, link8_quaternion))) > 1 - 1e-9

    runtime = Ros2Runtime.__new__(Ros2Runtime)
    runtime.config = SimpleNamespace(policy_eef_frame="link8")
    runtime._lifecycle_lock = threading.RLock()
    runtime._f_t_ee = (f_t_ee_position, f_t_ee_quaternion)
    policy_actions = np.asarray(
        [[*link8_position, *link8_quaternion, 0.4]],
        dtype=np.float32,
    )
    canonical = runtime.canonicalize_policy_actions(policy_actions)
    assert np.allclose(canonical[0, :3], eef_position, atol=1e-6)
    assert abs(float(np.dot(canonical[0, 3:7], eef_quaternion))) > 1 - 1e-6
    assert canonical[0, 7] == pytest.approx(0.4)


def test_policy_eef_frame_config_rejects_unknown_mode() -> None:
    from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig

    with pytest.raises(ValueError, match="policy_eef_frame"):
        FrankaRosConfig(id="bad-frame", policy_eef_frame="guess")
