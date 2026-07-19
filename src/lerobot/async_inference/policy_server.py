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

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --policy_type=pi0 \
     --pretrained_name_or_path=/path/to/pretrained_model \
     --policy_device=cuda \
     --actions_per_chunk=50 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
import uuid
from collections import OrderedDict
from concurrent import futures
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


_MAX_DELIVERY_TOMBSTONES = 1024


@dataclass(frozen=True)
class _PendingActionDelivery:
    request_id: str
    chunk_id: str
    source_timestep: int
    response: Any


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        # A generated chunk is not considered delivered until AckActions is
        # received. The same lock protects all delivery state transitions.
        self._predicted_timesteps_lock = threading.RLock()
        self._predicted_timesteps = set()
        self._queued_timesteps: set[int] = set()
        self._queued_request_ids: set[str] = set()
        self._inflight_timesteps: set[int] = set()
        self._inflight_request_ids: set[str] = set()
        self._generated_timesteps: set[int] = set()
        self._pending_delivery: _PendingActionDelivery | None = None
        self._delivered_chunks: OrderedDict[str, tuple[str, int]] = OrderedDict()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.rename_map: dict[str, str] = {}
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self._loaded_policy_setup_key = None
        self._policy_setup_lock = threading.Lock()
        # gRPC may dispatch overlapping GetActions calls to different worker
        # threads during reconnects. Only one call may consume an observation
        # and create the single READY_UNACKED delivery at a time.
        self._get_actions_lock = threading.Lock()
        self._enqueue_observation_lock = threading.Lock()
        self._session_generation = 0

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self._policy_config.image_features

    @property
    def _policy_config(self):
        if self.policy is None:
            raise RuntimeError("Policy has not been initialized")

        get_base_model = getattr(self.policy, "get_base_model", None)
        if callable(get_base_model):
            return get_base_model().config

        return self.policy.config

    def _load_policy(self, policy_type: str, pretrained_name_or_path: str):
        policy_class = get_policy_class(policy_type)
        pretrained_path = Path(pretrained_name_or_path)

        if pretrained_path.is_dir() and (pretrained_path / "adapter_config.json").is_file():
            from peft import PeftConfig, PeftModel

            self.logger.info("Loading policy's PEFT adapter.")
            peft_config = PeftConfig.from_pretrained(pretrained_name_or_path)
            if not peft_config.base_model_name_or_path:
                raise ValueError(
                    "No pretrained model name found in adapter config. Can't instantiate the base policy."
                )

            policy_config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
            policy = policy_class.from_pretrained(peft_config.base_model_name_or_path, config=policy_config)
            return PeftModel.from_pretrained(policy, pretrained_name_or_path, config=peft_config)

        return policy_class.from_pretrained(pretrained_name_or_path)

    def _make_policy_setup_key(self, policy_specs: RemotePolicyConfig) -> tuple[Any, ...]:
        return (
            policy_specs.policy_type,
            policy_specs.pretrained_name_or_path,
            policy_specs.device,
            policy_specs.actions_per_chunk,
            tuple(sorted(policy_specs.rename_map.items())),
            pformat(asdict(policy_specs)["lerobot_features"], sort_dicts=True),
        )

    def _resolve_policy_specs(self, client_specs: RemotePolicyConfig) -> RemotePolicyConfig:
        """Merge client-provided robot features with server-owned policy settings."""
        policy_specs = RemotePolicyConfig(
            policy_type=self.config.policy_type or client_specs.policy_type,
            pretrained_name_or_path=self.config.pretrained_name_or_path
            or client_specs.pretrained_name_or_path,
            lerobot_features=client_specs.lerobot_features,
            actions_per_chunk=self.config.actions_per_chunk or client_specs.actions_per_chunk,
            device=self.config.policy_device or client_specs.device,
            rename_map=client_specs.rename_map,
        )

        missing = [
            name
            for name, value in (
                ("policy_type", policy_specs.policy_type),
                ("pretrained_name_or_path", policy_specs.pretrained_name_or_path),
                ("actions_per_chunk", policy_specs.actions_per_chunk),
                ("policy_device", policy_specs.device),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Missing policy configuration: "
                f"{', '.join(missing)}. Set these options on the policy server or robot client."
            )

        return policy_specs

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()
            self._queued_timesteps = set()
            self._queued_request_ids = set()
            self._inflight_timesteps = set()
            self._inflight_request_ids = set()
            self._generated_timesteps = set()
            self._pending_delivery = None
            self._delivered_chunks = OrderedDict()
            self.last_processed_obs = None

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        with self._get_actions_lock:
            with self._enqueue_observation_lock:
                self._session_generation += 1
                self._reset_server()
                self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        client_specs = pickle.loads(request.data)  # nosec

        if not isinstance(client_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(client_specs)}")

        policy_specs = self._resolve_policy_specs(client_specs)

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.rename_map = policy_specs.rename_map

        with self._policy_setup_lock:
            policy_setup_key = self._make_policy_setup_key(policy_specs)
            if (
                self._loaded_policy_setup_key == policy_setup_key
                and self.policy is not None
                and self.preprocessor is not None
                and self.postprocessor is not None
            ):
                self.logger.info("Policy setup unchanged; reusing loaded policy and processors.")
                return services_pb2.Empty()

            start = time.perf_counter()
            self.policy = self._load_policy(self.policy_type, policy_specs.pretrained_name_or_path)
            self.policy.to(self.device)

            # Load preprocessor and postprocessor, overriding device to match requested device
            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self._policy_config,
                pretrained_path=policy_specs.pretrained_name_or_path,
                preprocessor_overrides={
                    "device_processor": device_override,
                    "rename_observations_processor": {"rename_map": policy_specs.rename_map},
                },
                postprocessor_overrides={"device_processor": device_override},
            )

            end = time.perf_counter()
            self._loaded_policy_setup_key = policy_setup_key

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")
        with self._enqueue_observation_lock:
            session_generation = self._session_generation

        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        receive_time = time.time()  # payload has been fully received
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()
        client_send_timestamp = getattr(timed_observation, "client_send_timestamp", None)
        if client_send_timestamp is not None:
            client_to_server_ms = (
                receive_time - client_send_timestamp
            ) * 1000
            self.logger.info(
                f"[LATENCY] client_to_server={client_to_server_ms:.2f}ms | "
                f"observation_timestep={obs_timestep}"
            )

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation,  # wrapping a RawObservation
            session_generation=session_generation,
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    @staticmethod
    def _observation_request_id(obs: TimedObservation) -> str | None:
        request_id = getattr(obs, "request_id", None)
        if request_id is None:
            return None
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("TimedObservation.request_id must be a non-empty string when provided")
        return request_id

    def _pending_action_response(self):
        with self._predicted_timesteps_lock:
            pending = self._pending_delivery
            return None if pending is None else pending.response

    def AckActions(self, request, context):  # noqa: N802
        """Commit one generated chunk after the client has processed it locally."""

        request_id = str(request.request_id)
        chunk_id = str(request.chunk_id)
        source_timestep = int(request.source_timestep)
        if not request_id or not chunk_id:
            self.logger.warning("Ignoring action delivery ACK with an empty request_id/chunk_id")
            return services_pb2.Empty()

        with self._predicted_timesteps_lock:
            pending = self._pending_delivery
            if pending is not None and (
                pending.request_id == request_id
                and pending.chunk_id == chunk_id
                and pending.source_timestep == source_timestep
            ):
                self._pending_delivery = None
                self._generated_timesteps.discard(source_timestep)
                self._predicted_timesteps.add(source_timestep)
                self._delivered_chunks[chunk_id] = (request_id, source_timestep)
                self._delivered_chunks.move_to_end(chunk_id)
                while len(self._delivered_chunks) > _MAX_DELIVERY_TOMBSTONES:
                    self._delivered_chunks.popitem(last=False)
                self.logger.info(
                    "Action chunk #%s acknowledged by client (chunk_id=%s)",
                    source_timestep,
                    chunk_id,
                )
                return services_pb2.Empty()

            delivered = self._delivered_chunks.get(chunk_id)
            if delivered == (request_id, source_timestep):
                self.logger.debug("Duplicate ACK for committed action chunk %s", chunk_id)
                return services_pb2.Empty()

        # Do not clear any state for an unknown or mismatched ACK. The pending
        # chunk will be redelivered, allowing the client to retry safely.
        self.logger.warning(
            "Ignoring unmatched action delivery ACK: request_id=%s chunk_id=%s timestep=%s",
            request_id,
            chunk_id,
            source_timestep,
        )
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        with self._get_actions_lock:
            return self._get_actions_serialized(request, context)

    def _get_actions_serialized(self, request, context):
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions.

        For ACK-capable clients, a generated response is cached before this
        RPC returns and is replayed byte-for-byte until AckActions commits it.
        """
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        pending_response = self._pending_action_response()
        if pending_response is not None:
            self.logger.info(
                "Redelivering unacknowledged action chunk #%s (chunk_id=%s)",
                pending_response.source_timestep,
                pending_response.chunk_id,
            )
            return pending_response

        # Generate action based on the most recent observation and its timestep
        obs = None
        request_id = None
        source_timestep = None
        legacy_prediction_marked = False
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            request_id = self._observation_request_id(obs)
            source_timestep = int(obs.get_timestep())

            with self._predicted_timesteps_lock:
                self._queued_timesteps.discard(source_timestep)
                if request_id is not None:
                    self._queued_request_ids.discard(request_id)
                    self._inflight_timesteps.add(source_timestep)
                    self._inflight_request_ids.add(request_id)
                else:
                    # Preserve compatibility with clients that do not attach a
                    # request ID and therefore cannot acknowledge delivery.
                    self._predicted_timesteps.add(source_timestep)
                    legacy_prediction_marked = True

            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            server_send_timestamp = time.time()
            for timed_action in action_chunk:
                timed_action.server_send_timestamp = server_send_timestamp
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Cache the exact response before returning it. A disconnect in the
            # response window then leaves a replayable READY_UNACKED result.
            if request_id is None:
                actions = services_pb2.Actions(data=actions_bytes)
            else:
                chunk_id = uuid.uuid4().hex
                actions = services_pb2.Actions(
                    data=actions_bytes,
                    request_id=request_id,
                    chunk_id=chunk_id,
                    source_timestep=source_timestep,
                )
                with self._predicted_timesteps_lock:
                    self._inflight_timesteps.discard(source_timestep)
                    self._inflight_request_ids.discard(request_id)
                    self._generated_timesteps.add(source_timestep)
                    self._pending_delivery = _PendingActionDelivery(
                        request_id=request_id,
                        chunk_id=chunk_id,
                        source_timestep=source_timestep,
                        response=actions,
                    )

            # Similarity filtering should only use observations for which a
            # complete action chunk was successfully generated.
            self.last_processed_obs = obs

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            if source_timestep is not None:
                with self._predicted_timesteps_lock:
                    self._queued_timesteps.discard(source_timestep)
                    self._inflight_timesteps.discard(source_timestep)
                    if request_id is not None:
                        self._queued_request_ids.discard(request_id)
                        self._inflight_request_ids.discard(request_id)
                    elif legacy_prediction_marked:
                        # An inference failure must not poison this timestep.
                        self._predicted_timesteps.discard(source_timestep)
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            reserved_timesteps = (
                self._predicted_timesteps
                | self._queued_timesteps
                | self._inflight_timesteps
                | self._generated_timesteps
            )

        if obs.get_timestep() in reserved_timesteps:
            self.logger.debug(
                "Skipping observation #%s - timestep is already queued, predicting, generated, or delivered",
                obs.get_timestep(),
            )
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(
        self,
        obs: TimedObservation,
        *,
        session_generation: int | None = None,
    ) -> bool:
        with self._enqueue_observation_lock:
            if (
                session_generation is not None
                and session_generation != self._session_generation
            ):
                self.logger.info(
                    "Dropping observation #%s from stale session generation %s (current=%s)",
                    obs.get_timestep(),
                    session_generation,
                    self._session_generation,
                )
                return False
            return self._enqueue_observation_serialized(obs)

    def _enqueue_observation_serialized(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        request_id = self._observation_request_id(obs)
        timestep = int(obs.get_timestep())
        with self._predicted_timesteps_lock:
            request_already_seen = request_id is not None and (
                request_id in self._queued_request_ids
                or request_id in self._inflight_request_ids
                or (
                    self._pending_delivery is not None
                    and self._pending_delivery.request_id == request_id
                )
                or any(delivered[0] == request_id for delivered in self._delivered_chunks.values())
            )
            timestep_reserved = timestep in (
                self._predicted_timesteps
                | self._queued_timesteps
                | self._inflight_timesteps
                | self._generated_timesteps
            )

        # must_go controls similarity filtering, not delivery idempotency.
        if request_already_seen or timestep_reserved:
            self.logger.debug(
                "Skipping duplicate observation #%s (request_id=%s)",
                timestep,
                request_id or "legacy",
            )
            return False

        if obs.must_go or self.last_processed_obs is None or self._obs_sanity_checks(
            obs, self.last_processed_obs
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                try:
                    removed = self.observation_queue.get_nowait()
                except Empty:
                    # GetActions won the race and consumed the old value.
                    pass
                else:
                    removed_request_id = self._observation_request_id(removed)
                    with self._predicted_timesteps_lock:
                        self._queued_timesteps.discard(int(removed.get_timestep()))
                        if removed_request_id is not None:
                            self._queued_request_ids.discard(removed_request_id)
                    self.logger.debug("Observation queue was full, removed oldest observation")

            # Register the reservation before publishing the value to the
            # Queue. GetActions can wake immediately after put_nowait(); doing
            # this in the opposite order can leave stale queued bookkeeping.
            with self._predicted_timesteps_lock:
                self._queued_timesteps.add(timestep)
                if request_id is not None:
                    self._queued_request_ids.add(request_id)

            try:
                self.observation_queue.put_nowait(obs)
            except Full:
                with self._predicted_timesteps_lock:
                    self._queued_timesteps.discard(timestep)
                    if request_id is not None:
                        self._queued_request_ids.discard(request_id)
                raise
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
            self.rename_map,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
