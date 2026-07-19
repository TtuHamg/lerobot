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

from __future__ import annotations

import logging
import threading
import time
from queue import Queue
from types import SimpleNamespace

import grpc

from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.robot_client import RobotClient
from lerobot.transport import services_pb2


class _SuccessStub:
    def SendObservations(self, request_iterator):  # noqa: N802
        list(request_iterator)
        return services_pb2.Empty()


class _InjectedRpcError(grpc.RpcError):
    pass


class _FailingStub:
    def SendObservations(self, request_iterator):  # noqa: N802
        raise _InjectedRpcError("injected transport failure")


def _bare_client(stub) -> RobotClient:
    client = object.__new__(RobotClient)
    client.config = SimpleNamespace(
        enable_pending_observation=True,
        pending_observation_timeout_s=0.01,
    )
    client.stub = stub
    client.logger = logging.getLogger("test_action_delivery_retry")
    client.shutdown_event = threading.Event()
    client._pending_observation_lock = threading.Lock()
    client._pending_observation = False
    client._pending_observation_sent_at = None
    client._pending_observation_request_id = None
    client._pending_observation_value = None
    client._pending_observation_retry_due = False
    client._client_session_id = "test-session"
    client._next_observation_sequence = 0
    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.action_chunk_size = 1
    client._chunk_size_threshold = 0.5
    return client


def _observation(timestep: int) -> TimedObservation:
    return TimedObservation(
        timestamp=time.time(),
        timestep=timestep,
        observation={"state": timestep},
    )


def test_successful_send_drops_retry_payload_and_timeout_allows_fresh_request():
    client = _bare_client(_SuccessStub())
    first = _observation(10)

    assert client.send_observation(first) is True
    first_request_id = first.request_id
    assert first_request_id
    assert client._pending_observation is True
    assert client._pending_observation_value is None

    # A successful SendObservations call that yields no action may have been
    # similarity-filtered. Its timeout must open the gate for a fresh frame.
    client._pending_observation_sent_at = time.perf_counter() - 1.0
    assert client._ready_to_send_observation() is True
    assert client._pending_observation is False

    second = _observation(10)
    assert client.send_observation(second) is True
    assert second.request_id != first_request_id


def test_failed_send_retries_same_request_without_resurrecting_completed_pending():
    client = _bare_client(_FailingStub())
    observation = _observation(588)

    assert client.send_observation(observation) is False
    request_id = observation.request_id
    assert request_id
    assert client._pending_observation_value is observation
    assert client._ready_to_send_observation() is True

    retry = client._claim_pending_observation_retry()
    assert retry is observation
    assert retry.request_id == request_id

    # Simulate the action receiver committing this request after the retry was
    # claimed but before its RPC is sent. A later transport error must not
    # re-arm the already completed request.
    client._resolve_pending_observation(request_id)
    assert client._send_observation_rpc(retry) is False
    assert client._pending_observation is False
    assert client._pending_observation_request_id is None
    assert client._pending_observation_value is None
    assert client._pending_observation_retry_due is False
