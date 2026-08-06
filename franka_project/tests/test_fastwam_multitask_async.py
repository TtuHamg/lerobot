"""Focused tests for FastWAM manifest-backed multi-task serving."""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from franka_eef_pipeline.async_server import (  # noqa: E402
    FrankaAsyncPolicyContractError,
    _resolve_fastwam_task_instructions,
)
from lerobot_robot_franka_ros.ros2_backend import Ros2Backend  # noqa: E402
from lerobot_robot_franka_ros.ros2_client import _InteractiveTaskController  # noqa: E402

from lerobot.async_inference.configs import PolicyServerConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedObservation  # noqa: E402
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402


def _observation(*, generation: int, timestep: int, request_id: str) -> TimedObservation:
    return TimedObservation(
        observation={"task": "pick up the cup"},
        timestamp=time.time(),
        timestep=timestep,
        must_go=True,
        request_id=request_id,
        task_generation=generation,
    )


def test_resolve_fastwam_tasks_filters_selected_episodes_and_preserves_order(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "episodes": [
                    {"episode_id": "a", "task_instruction": "pick up the cup"},
                    {"episode_id": "b", "task_instruction": "pick up the chips"},
                    {"episode_id": "c", "task_instruction": "pick up the cup"},
                    {"episode_id": "d", "task_instruction": "pick up the tape"},
                ]
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "source": {"manifest": str(manifest), "episode_ids": ["a", "c", "d"]},
        "task": {"default_instruction": "__MISSING_TASK_INSTRUCTION__"},
    }

    assert _resolve_fastwam_task_instructions(
        contract,
        contract_path=tmp_path / "dataset_contract.json",
    ) == ("pick up the cup", "pick up the tape")


def test_resolve_fastwam_tasks_rejects_placeholder_without_manifest(tmp_path: Path) -> None:
    contract = {
        "source": {"manifest": str(tmp_path / "missing.yaml")},
        "task": {"default_instruction": "__MISSING_TASK_INSTRUCTION__"},
    }
    with pytest.raises(FrankaAsyncPolicyContractError, match="no deployable task"):
        _resolve_fastwam_task_instructions(
            contract,
            contract_path=tmp_path / "dataset_contract.json",
        )


def test_new_generation_flushes_transport_state_without_reloading_policy() -> None:
    server = PolicyServer(PolicyServerConfig())
    policy = object()
    server.policy = policy
    first = _observation(generation=1, timestep=4, request_id="old")
    second = _observation(generation=2, timestep=4, request_id="new")

    assert server._enqueue_observation(first)
    server._generated_timesteps.add(4)
    assert server._enqueue_observation(second)

    assert server.policy is policy
    assert server._active_task_generation == 2
    assert server._generated_timesteps == set()
    assert server.observation_queue.get_nowait() is second
    assert not server._enqueue_observation(
        _observation(generation=1, timestep=5, request_id="stale")
    )


def test_interactive_task_controller_requires_select_then_arm() -> None:
    controller = _InteractiveTaskController(("pick up the cup", "pick up the tape"))

    assert controller.snapshot() == (None, 0, False)
    assert controller.select("pick up the tape") == 1
    assert controller.snapshot() == ("pick up the tape", 1, False)
    assert controller.arm() == ("pick up the tape", 1)
    assert controller.snapshot() == ("pick up the tape", 1, True)
    assert controller.stop() == 2
    assert controller.snapshot() == (None, 2, False)

    with pytest.raises(ValueError, match="not server-advertised"):
        controller.select("pick up the chips")


def test_disarm_calls_external_gateway_and_clears_plan() -> None:
    calls = []
    runtime = SimpleNamespace(
        is_running=True,
        set_gateway_armed=lambda armed, timeout_s: calls.append((armed, timeout_s))
        or (True, "hold"),
    )
    backend = object.__new__(Ros2Backend)
    backend._runtime = runtime
    backend._lock = threading.RLock()
    backend._last_published_plan = (3, 32, 1, 1)

    assert backend.set_gateway_armed(False, timeout_s=2.5) == (True, "hold")
    assert calls == [(False, 2.5)]
    assert backend._last_published_plan is None
