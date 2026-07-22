"""Lightweight stock-client -> gRPC -> Franka-server -> sink integration test."""

from __future__ import annotations

import json
import pickle  # nosec: the stock local gRPC contract intentionally uses pickle
import sys
import time
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
import torch

PLUGIN_SRC = Path(__file__).parents[1] / "ros_lerobot" / "src"
if str(PLUGIN_SRC) not in sys.path:
    sys.path.insert(0, str(PLUGIN_SRC))

from franka_eef_pipeline.async_server import FrankaPI0PolicyServer  # noqa: E402
from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig  # noqa: E402
from lerobot.async_inference.helpers import TimedObservation  # noqa: E402
from lerobot.async_inference.robot_client import RobotClient  # noqa: E402
from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot_robot_franka_ros import CAMERA_SHAPE, FrankaRosConfig  # noqa: E402


class _FakePI0Policy(torch.nn.Module):
    """Small deterministic policy used only to exercise the transport boundary."""

    def __init__(self) -> None:
        super().__init__()
        image_features = {
            name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
            for name in (
                "observation.images.base_0_rgb",
                "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist_0_rgb",
            )
        }
        self.config = SimpleNamespace(image_features=image_features)

    def predict_action_chunk(self, observation: dict) -> torch.Tensor:
        assert tuple(observation["observation.state"].shape) == (1, 10)
        assert tuple(observation["observation.images.base_0_rgb"].shape) == (1, 3, 224, 224)
        assert tuple(observation["observation.images.left_wrist_0_rgb"].shape) == (1, 3, 224, 224)
        assert observation["task"] == "stack the cups"
        return torch.tensor(
            [
                [
                    [0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25],
                    [0.00, -0.20, 0.05, 0.00, 0.00, 0.00, 0.75],
                ]
            ],
            dtype=torch.float32,
        )


class _LightweightFrankaServer(FrankaPI0PolicyServer):
    def _load_policy(self, policy_type: str, pretrained_name_or_path: str) -> _FakePI0Policy:
        assert policy_type == "pi0"
        assert pretrained_name_or_path == "test-only-checkpoint"
        return _FakePI0Policy()


def _write_fixture(path: Path) -> None:
    np.savez_compressed(
        path,
        state=np.asarray(
            [0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5],
            dtype=np.float32,
        ),
        camera1=np.zeros(CAMERA_SHAPE, dtype=np.uint8),
        camera2=np.full(CAMERA_SHAPE, 127, dtype=np.uint8),
    )


def test_stock_robot_client_round_trip_to_absolute_action_sink(
    tmp_path: Path, monkeypatch
) -> None:
    # Keep the stock PolicyServer's preprocessing/postprocessing orchestration,
    # but use identity callables because this test policy already consumes raw
    # batched tensors and emits denormalized relative7 actions.
    monkeypatch.setattr(
        "lerobot.async_inference.policy_server.make_pre_post_processors",
        lambda *_args, **_kwargs: (lambda value: value, lambda value: value),
    )

    fixture_path = tmp_path / "observation.npz"
    action_log_path = tmp_path / "actions.jsonl"
    _write_fixture(fixture_path)

    policy_server = _LightweightFrankaServer(
        PolicyServerConfig(
            host="127.0.0.1",
            port=1,
            fps=15,
            obs_queue_timeout=1.0,
            policy_type="pi0",
            pretrained_name_or_path="test-only-checkpoint",
            actions_per_chunk=2,
            policy_device="cpu",
        )
    )
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    port = grpc_server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    grpc_server.start()

    client = RobotClient(
        RobotClientConfig(
            robot=FrankaRosConfig(
                id="lightweight-e2e",
                calibration_dir=tmp_path / "calibration",
                fixture_path=fixture_path,
                action_log_path=action_log_path,
            ),
            server_address=f"127.0.0.1:{port}",
            task="stack the cups",
            client_device="cpu",
            fps=15,
            action_offset=1,
            aggregate_fn_name="latest_only",
            rename_map={
                "observation.images.camera1": "observation.images.base_0_rgb",
                "observation.images.camera2": "observation.images.left_wrist_0_rgb",
            },
        )
    )

    try:
        assert client.start()
        client.latest_action = 4
        observation = client.robot.get_observation()
        observation["task"] = "stack the cups"
        observation_timestep = client._next_observation_timestep(client.latest_action)
        assert observation_timestep == 5
        assert client.send_observation(
            TimedObservation(
                timestamp=time.time(),
                timestep=observation_timestep,
                observation=observation,
                must_go=True,
            )
        )

        response = client.stub.GetActions(services_pb2.Empty(), timeout=5.0)
        # Simulate losing the first response before the client can ACK it. The
        # server must replay the exact cached delivery instead of filtering the
        # observation as "already predicted" and blocking forever.
        replay = client.stub.GetActions(services_pb2.Empty(), timeout=5.0)
        assert replay.SerializeToString() == response.SerializeToString()
        timed_actions = pickle.loads(response.data)  # nosec: trusted in-process test server
        assert len(timed_actions) == 2
        assert [item.get_timestep() for item in timed_actions] == [5, 6]
        assert response.source_timestep == observation_timestep
        absolute = torch.stack([item.get_action() for item in timed_actions])
        assert tuple(absolute.shape) == (2, 8)
        assert bool(torch.isfinite(absolute).all())
        torch.testing.assert_close(absolute[0, :3], torch.tensor([0.6, 0.0, 0.4]))
        torch.testing.assert_close(
            absolute[:, 3:7],
            torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=torch.float32),
        )

        client._aggregate_action_queues(timed_actions, client.config.aggregate_fn)
        client._remember_committed_action_chunk(response.chunk_id)
        client._resolve_pending_observation(response.request_id)
        client._ack_action_delivery(response)
        assert client._pending_observation is False
        assert client._pending_observation_request_id is None
        assert policy_server._pending_delivery is None
        assert response.source_timestep in policy_server._predicted_timesteps
        client.control_loop_action()
        client.control_loop_action()
    finally:
        client.stop()
        policy_server.stop()
        grpc_server.stop(grace=0).wait(timeout=5.0)

    records = [json.loads(line) for line in action_log_path.read_text().splitlines()]
    assert [record["sequence"] for record in records] == [0, 1]
    assert all(record["dry_run"] is True for record in records)
    assert records[0]["action"]["target.x"] == pytest.approx(0.6)
    assert records[1]["action"]["target.y"] == pytest.approx(-0.2)
