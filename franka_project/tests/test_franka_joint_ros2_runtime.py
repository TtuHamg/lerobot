"""No-ROS tests for sequential Joint Gateway progress tracking."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace


PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from lerobot_robot_franka_ros.joint_ros2_runtime import (  # noqa: E402
    JointRos2Runtime,
    _PlanProgress,
)


def _runtime_with_progress(progress: _PlanProgress) -> JointRos2Runtime:
    runtime = object.__new__(JointRos2Runtime)
    runtime._plan_lock = threading.Lock()
    runtime._plan_progress = progress
    runtime._latest_gateway_status = None
    runtime._node = None
    runtime.config = SimpleNamespace(action_execution_timeout_s=10.0)
    return runtime


def _status(**overrides):
    values = {
        "session_id": "",
        "plan_id": 0,
        "has_active_plan": False,
        "armed": True,
        "shadow": False,
        "detail": "ARMED: no active plan",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_active_plan_status_marks_progress_seen() -> None:
    progress = _PlanProgress("session", 7, time.monotonic(), ack_received=True)
    runtime = _runtime_with_progress(progress)

    runtime._safety_gateway_status_callback(
        _status(session_id="session", plan_id=7, has_active_plan=True)
    )

    assert progress.seen_active
    assert not progress.completed


def test_armed_no_active_fallback_completes_after_activation_grace() -> None:
    progress = _PlanProgress(
        "session",
        8,
        time.monotonic() - 1.0,
        ack_received=True,
    )
    runtime = _runtime_with_progress(progress)

    runtime._safety_gateway_status_callback(_status())

    assert progress.completed
    assert progress.error is None


def test_no_active_fallback_waits_for_ack_and_grace() -> None:
    missing_ack = _PlanProgress("session", 9, time.monotonic() - 1.0)
    runtime = _runtime_with_progress(missing_ack)
    runtime._safety_gateway_status_callback(_status())
    assert not missing_ack.completed

    inside_grace = _PlanProgress(
        "session",
        10,
        time.monotonic(),
        ack_received=True,
    )
    runtime = _runtime_with_progress(inside_grace)
    runtime._safety_gateway_status_callback(_status())
    assert not inside_grace.completed


def test_rejected_plan_is_skipped_without_stopping_client() -> None:
    progress = _PlanProgress("session", 11, time.monotonic())
    runtime = _runtime_with_progress(progress)
    runtime._latest_gateway_status = _status()
    runtime._action_chunk_ack_callback(
        SimpleNamespace(
            session_id="session",
            plan_id=11,
            result=20,
            accepted=False,
            detail="position step exceeds limit",
        )
    )

    assert runtime.ready_for_next_observation()
    assert runtime._plan_progress is None


def test_preflight_only_ack_completes_without_active_status() -> None:
    progress = _PlanProgress("session", 12, time.monotonic())
    runtime = _runtime_with_progress(progress)
    runtime._latest_gateway_status = _status(shadow=True)
    runtime._action_chunk_ack_callback(
        SimpleNamespace(
            session_id="session",
            plan_id=12,
            result=2,
            accepted=True,
            detail="preflight-only accepted",
        )
    )

    assert progress.completed
    assert runtime.ready_for_next_observation()
