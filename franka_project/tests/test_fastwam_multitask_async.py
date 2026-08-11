"""Focused tests for FastWAM manifest-backed multi-task serving."""

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from franka_eef_pipeline.async_server import (  # noqa: E402
    ACTION_LABEL_MODE_DELTA_EEF,
    FrankaAsyncPolicyContractError,
    _build_fastwam_state,
    _decode_fastwam_absolute_action_chunk,
    _resolve_fastwam_task_instructions,
)
from lerobot_robot_franka_ros.config_franka_ros import FrankaRosConfig  # noqa: E402
from lerobot_robot_franka_ros.ros2_backend import Ros2Backend  # noqa: E402
from lerobot_robot_franka_ros.ros2_client import (  # noqa: E402
    FrankaRos2ClientConfig,
    FrankaRos2RobotClient,
    _InteractiveTaskController,
    resolve_franka_client_policy_config,
)
from lerobot_robot_franka_ros.ros2_runtime import (  # noqa: E402
    Ros2RuntimeStateError,
    _read_gripper_endpoint_parameters,
)

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
    assert not server._enqueue_observation(_observation(generation=1, timestep=5, request_id="stale"))


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
        set_gateway_armed=lambda armed, timeout_s: calls.append((armed, timeout_s)) or (True, "hold"),
    )
    backend = object.__new__(Ros2Backend)
    backend._runtime = runtime
    backend._lock = threading.RLock()
    backend._last_published_plan = (3, 32, 1, 1)

    assert backend.set_gateway_armed(False, timeout_s=2.5) == (True, "hold")
    assert calls == [(False, 2.5)]
    assert backend._last_published_plan is None


def test_fastwam_gripper_encoding_override_preserves_learned_closed_target() -> None:
    anchor = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    model_action = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
        ],
        dtype=torch.float32,
    )

    decoded_open = _decode_fastwam_absolute_action_chunk(
        anchor,
        model_action,
        action_label_mode=ACTION_LABEL_MODE_DELTA_EEF,
        gripper_encoding="open_0_1",
    )
    decoded_closed = _decode_fastwam_absolute_action_chunk(
        anchor,
        model_action,
        action_label_mode=ACTION_LABEL_MODE_DELTA_EEF,
        gripper_encoding="closed_0_1",
    )

    torch.testing.assert_close(decoded_open[:, 7], torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(decoded_closed[:, 7], torch.tensor([0.25, 0.75]))
    with pytest.raises(FrankaAsyncPolicyContractError, match="gripper encoding"):
        _decode_fastwam_absolute_action_chunk(
            anchor,
            model_action,
            gripper_encoding="guess",
        )


def test_policy_server_rejects_unknown_fastwam_gripper_encoding() -> None:
    with pytest.raises(ValueError, match="fastwam_action_gripper_encoding"):
        PolicyServerConfig(fastwam_action_gripper_encoding="guess")


@pytest.mark.parametrize(
    ("closed", "expected_fingers"),
    [(0.0, [0.0, 0.0]), (1.0, [0.04, -0.04])],
)
def test_fastwam_state_closed_encoding_matches_frozen_mix3_checkpoint(
    closed: float,
    expected_fingers: list[float],
) -> None:
    state = np.asarray([0.5, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, closed])

    actual = _build_fastwam_state(state, gripper_encoding="closed_0_1")

    np.testing.assert_allclose(actual[-2:], expected_fingers, rtol=0.0, atol=1e-12)


def test_policy_server_rejects_unknown_fastwam_state_gripper_encoding() -> None:
    with pytest.raises(ValueError, match="fastwam_state_gripper_encoding"):
        PolicyServerConfig(fastwam_state_gripper_encoding="guess")


def test_fastwam_client_rejects_truncated_action_horizon() -> None:
    client_config = SimpleNamespace(
        robot=FrankaRosConfig(
            id="fastwam-horizon",
            gripper_open_position=0.0,
            gripper_closed_position=0.4,
            gripper_max_skew_s=0.01,
            camera2_max_skew_s=0.1,
            eef_max_skew_s=0.05,
            max_action_chunk_waypoints=25,
        ),
        policy_type="fastwam",
        action_offset=1,
        rename_map={},
        gateway_arm_timeout_s=5.0,
        actions_per_chunk=32,
    )
    with pytest.raises(ValueError, match="complete action horizon"):
        resolve_franka_client_policy_config(client_config)


def test_fastwam_client_defers_live_gripper_endpoints_to_ros_parameter_owner() -> None:
    client_config = FrankaRos2ClientConfig(
        robot=FrankaRosConfig(
            id="fastwam-live-gripper",
            gripper_open_position=0.0,
            gripper_closed_position=0.8,
            gripper_max_skew_s=0.01,
            camera2_max_skew_s=0.1,
            eef_max_skew_s=0.05,
            max_action_chunk_waypoints=32,
        ),
        policy_type="fastwam",
        action_offset=1,
        rename_map={},
        gateway_arm_timeout_s=5.0,
        actions_per_chunk=32,
    )

    resolved = resolve_franka_client_policy_config(client_config)

    assert resolved.robot.gripper_closed_position == pytest.approx(0.8)


def test_gripper_endpoints_are_read_from_follower_parameters_once() -> None:
    class _Future:
        def done(self):
            return True

        def exception(self):
            return None

        def result(self):
            return [SimpleNamespace(value=0.0), SimpleNamespace(value=0.8)]

    class _ParameterClient:
        requested_names = None

        def __init__(self, node, remote_node_name):
            assert node == "node"
            assert remote_node_name == "/franka_gripper_follower"

        def wait_for_services(self, *, timeout_sec):
            assert timeout_sec == pytest.approx(2.0)
            return True

        def get_parameters(self, names):
            _ParameterClient.requested_names = names
            return _Future()

    executor = SimpleNamespace(
        spin_until_future_complete=lambda future, timeout_sec: None,
    )

    endpoints = _read_gripper_endpoint_parameters(
        node="node",
        executor=executor,
        parameter_client_type=_ParameterClient,
        parameter_value_to_python=lambda value: value,
        remote_node_name="/franka_gripper_follower",
        timeout_s=2.0,
    )

    assert endpoints == pytest.approx((0.0, 0.8))
    assert _ParameterClient.requested_names == ["open_position", "closed_position"]


def test_gripper_endpoint_parameter_service_is_required() -> None:
    class _UnavailableParameterClient:
        def __init__(self, node, remote_node_name):
            pass

        def wait_for_services(self, *, timeout_sec):
            return False

    with pytest.raises(Ros2RuntimeStateError, match="was not available"):
        _read_gripper_endpoint_parameters(
            node="node",
            executor=object(),
            parameter_client_type=_UnavailableParameterClient,
            parameter_value_to_python=lambda value: value,
            remote_node_name="/franka_gripper_follower",
            timeout_s=0.1,
        )


def test_interactive_client_does_not_accept_or_fetch_actions_in_hold() -> None:
    client = object.__new__(FrankaRos2RobotClient)
    client._interactive_tasks = _InteractiveTaskController(("pick up the cup",))
    generation = client._interactive_tasks.select("pick up the cup")

    with pytest.raises(RuntimeError, match="HOLD"):
        client._task_snapshot("unused")
    assert not client._accept_action_delivery("pick up the cup", generation)

    client._interactive_tasks.arm()
    assert client._task_snapshot("unused") == ("pick up the cup", generation)
    assert client._accept_action_delivery("pick up the cup", generation)


def test_arm_clears_prearm_work_and_gateway_plan_before_enabling() -> None:
    events: list[object] = []
    client = object.__new__(FrankaRos2RobotClient)
    client._interactive_tasks = _InteractiveTaskController(("pick up the cup",))
    client._interactive_tasks.select("pick up the cup")
    client.config = SimpleNamespace(gateway_arm_timeout_s=2.0)
    client.robot = SimpleNamespace(
        set_gateway_armed=lambda armed, timeout_s: events.append(("gateway", armed, timeout_s))
    )
    client.discard_pending_work = lambda: events.append("discard")
    client.must_go = threading.Event()
    client.logger = SimpleNamespace(info=lambda *args: None)

    assert client.arm_selected_task() == ("pick up the cup", 1)
    assert events == ["discard", ("gateway", False, 2.0), ("gateway", True, 2.0)]
    assert client.must_go.is_set()
