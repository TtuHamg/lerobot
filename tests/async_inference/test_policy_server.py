# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import time
import pickle  # nosec: tests exercise the trusted internal transport payload
from concurrent import futures
from queue import Queue
from threading import Event

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from lerobot.utils.constants import OBS_STATE
from tests.utils import skip_if_package_missing

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation[OBS_STATE])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@skip_if_package_missing("grpcio", "grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(
    state: torch.Tensor,
    timestep: int = 0,
    must_go: bool = False,
    request_id: str | None = None,
):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.async_inference.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
        request_id=request_id,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_server_policy_config_overrides_client_policy_config():
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.async_inference.policy_server import PolicyServer

    server = PolicyServer(
        PolicyServerConfig(
            policy_type="pi0",
            pretrained_name_or_path="/server/model",
            actions_per_chunk=50,
            policy_device="cuda",
        )
    )
    client_specs = RemotePolicyConfig(
        policy_type=None,
        pretrained_name_or_path=None,
        lerobot_features={},
        actions_per_chunk=None,
        device=None,
    )

    resolved = server._resolve_policy_specs(client_specs)

    assert resolved.policy_type == "pi0"
    assert resolved.pretrained_name_or_path == "/server/model"
    assert resolved.actions_per_chunk == 50
    assert resolved.device == "cuda"


def test_server_policy_config_falls_back_to_client():
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.helpers import RemotePolicyConfig
    from lerobot.async_inference.policy_server import PolicyServer

    server = PolicyServer(PolicyServerConfig())
    client_specs = RemotePolicyConfig(
        policy_type="act",
        pretrained_name_or_path="/client/model",
        lerobot_features={},
        actions_per_chunk=20,
        device="cpu",
    )

    assert server._resolve_policy_specs(client_specs) == client_specs


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.async_inference.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    # NOTE(Steven): Smelly tests as the Server is a state machine being partially mocked. Adding these processors as a quick fix.
    policy_server.preprocessor = lambda obs: obs
    policy_server.postprocessor = lambda tensor: tensor
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_action_delivery_is_redelivered_until_client_ack(monkeypatch, policy_server):
    """A lost GetActions response must not poison or discard the generated chunk."""
    from types import SimpleNamespace

    from lerobot.async_inference.helpers import TimedAction
    from lerobot.transport import services_pb2

    inference_calls = 0

    def _fake_predict(obs):
        nonlocal inference_calls
        inference_calls += 1
        return [
            TimedAction(
                timestamp=obs.get_timestamp(),
                timestep=obs.get_timestep(),
                action=torch.zeros(6),
            )
        ]

    monkeypatch.setattr(policy_server, "_predict_action_chunk", _fake_predict)
    obs = _make_obs(torch.zeros(6), timestep=588, must_go=True, request_id="session:588")
    assert policy_server._enqueue_observation(obs) is True

    context = SimpleNamespace(peer=lambda: "test-client")
    first = policy_server.GetActions(services_pb2.Empty(), context)
    assert first.request_id == "session:588"
    assert first.chunk_id
    assert first.source_timestep == 588
    assert inference_calls == 1
    assert 588 not in policy_server._predicted_timesteps

    # Simulate losing the first response: poll again without an ACK.
    replay = policy_server.GetActions(services_pb2.Empty(), context)
    assert replay.SerializeToString() == first.SerializeToString()
    assert inference_calls == 1
    assert 588 not in policy_server._predicted_timesteps

    policy_server.AckActions(
        services_pb2.ActionDeliveryAck(
            request_id=first.request_id,
            chunk_id=first.chunk_id,
            source_timestep=first.source_timestep,
        ),
        context,
    )
    assert policy_server._pending_delivery is None
    assert 588 in policy_server._predicted_timesteps

    # ACK retries are idempotent.
    policy_server.AckActions(
        services_pb2.ActionDeliveryAck(
            request_id=first.request_id,
            chunk_id=first.chunk_id,
            source_timestep=first.source_timestep,
        ),
        context,
    )
    assert 588 in policy_server._predicted_timesteps


def test_duplicate_must_go_observation_does_not_repeat_unacked_inference(
    monkeypatch, policy_server
):
    from types import SimpleNamespace

    from lerobot.async_inference.helpers import TimedAction
    from lerobot.transport import services_pb2

    monkeypatch.setattr(
        policy_server,
        "_predict_action_chunk",
        lambda obs: [
            TimedAction(
                timestamp=obs.get_timestamp(),
                timestep=obs.get_timestep(),
                action=torch.zeros(6),
            )
        ],
    )
    first_obs = _make_obs(torch.zeros(6), timestep=7, must_go=True, request_id="request-a")
    assert policy_server._enqueue_observation(first_obs) is True
    response = policy_server.GetActions(
        services_pb2.Empty(), SimpleNamespace(peer=lambda: "test-client")
    )
    assert response.chunk_id

    duplicate = _make_obs(torch.ones(6), timestep=7, must_go=True, request_id="request-b")
    assert policy_server._enqueue_observation(duplicate) is False


def test_inference_failure_does_not_permanently_reserve_timestep(monkeypatch, policy_server):
    from types import SimpleNamespace

    from lerobot.transport import services_pb2

    obs = _make_obs(torch.zeros(6), timestep=9, must_go=True, request_id="retryable-request")
    assert policy_server._enqueue_observation(obs) is True

    def _fail(_obs):
        raise RuntimeError("injected inference failure")

    monkeypatch.setattr(policy_server, "_predict_action_chunk", _fail)
    response = policy_server.GetActions(
        services_pb2.Empty(), SimpleNamespace(peer=lambda: "test-client")
    )
    assert isinstance(response, services_pb2.Empty)
    assert 9 not in policy_server._predicted_timesteps
    assert 9 not in policy_server._inflight_timesteps
    assert policy_server._enqueue_observation(obs) is True


def test_concurrent_get_actions_replays_single_pending_delivery(monkeypatch, policy_server):
    from types import SimpleNamespace

    from lerobot.async_inference.helpers import TimedAction
    from lerobot.transport import services_pb2

    inference_started = Event()
    release_inference = Event()
    inference_calls = 0

    def _blocking_predict(obs):
        nonlocal inference_calls
        inference_calls += 1
        inference_started.set()
        assert release_inference.wait(timeout=2.0)
        return [
            TimedAction(
                timestamp=obs.get_timestamp(),
                timestep=obs.get_timestep(),
                action=torch.zeros(6),
            )
        ]

    monkeypatch.setattr(policy_server, "_predict_action_chunk", _blocking_predict)
    first_obs = _make_obs(torch.zeros(6), timestep=20, must_go=True, request_id="request-20")
    second_obs = _make_obs(torch.ones(6), timestep=21, must_go=True, request_id="request-21")
    assert policy_server._enqueue_observation(first_obs) is True
    context = SimpleNamespace(peer=lambda: "test-client")

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(policy_server.GetActions, services_pb2.Empty(), context)
        assert inference_started.wait(timeout=2.0)
        assert policy_server._enqueue_observation(second_obs) is True
        second_future = executor.submit(policy_server.GetActions, services_pb2.Empty(), context)
        release_inference.set()
        first = first_future.result(timeout=2.0)
        second = second_future.result(timeout=2.0)

    assert first.SerializeToString() == second.SerializeToString()
    assert inference_calls == 1
    assert policy_server.observation_queue.qsize() == 1


def test_observation_reservation_is_registered_before_queue_visibility(policy_server):
    class _InspectingQueue(Queue):
        def put_nowait(self, item):
            assert item.get_timestep() in policy_server._queued_timesteps
            assert item.request_id in policy_server._queued_request_ids
            return super().put_nowait(item)

    policy_server.observation_queue = _InspectingQueue(maxsize=1)
    obs = _make_obs(torch.zeros(6), timestep=30, must_go=True, request_id="request-30")

    assert policy_server._enqueue_observation(obs) is True


def test_ready_drops_observation_from_an_older_stream(policy_server):
    from types import SimpleNamespace

    from lerobot.transport import services_pb2
    from lerobot.transport.utils import send_bytes_in_chunks

    stream_entered = Event()
    release_old_stream = Event()
    old_observation = _make_obs(
        torch.zeros(6),
        timestep=40,
        must_go=True,
        request_id="old-session:40",
    )

    def _blocked_old_stream():
        stream_entered.set()
        assert release_old_stream.wait(timeout=2.0)
        yield from send_bytes_in_chunks(
            pickle.dumps(old_observation),
            services_pb2.Observation,
        )

    context = SimpleNamespace(peer=lambda: "test-client")
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        old_rpc = executor.submit(policy_server.SendObservations, _blocked_old_stream(), context)
        assert stream_entered.wait(timeout=2.0)

        policy_server.Ready(services_pb2.Empty(), context)
        release_old_stream.set()
        old_rpc.result(timeout=2.0)

    assert policy_server.observation_queue.empty()
    assert not policy_server._queued_timesteps
    assert not policy_server._queued_request_ids
