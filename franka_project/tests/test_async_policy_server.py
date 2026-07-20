from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import franka_eef_pipeline.async_server as async_server_module
from franka_eef_pipeline.async_server import (
    FRANKA_RENAME_MAP,
    FRANKA_CHECKPOINT_TYPE,
    FrankaAsyncPolicyContractError,
    FrankaPI0PolicyServer,
    validate_franka_checkpoint,
)
from franka_eef_pipeline.pi0_training import PI0_CORE_WEIGHTS_NAMESPACE
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch
from lerobot.utils.constants import OBS_STATE


STATE_NAMES = (
    "eef.x",
    "eef.y",
    "eef.z",
    "eef.rot6d.col0.x",
    "eef.rot6d.col0.y",
    "eef.rot6d.col0.z",
    "eef.rot6d.col1.x",
    "eef.rot6d.col1.y",
    "eef.rot6d.col1.z",
    "gripper.closed_0_1",
)


def _write_checkpoint_envelope(root: Path) -> Path:
    geometry_manifest = {
        "schema_version": 1,
        "state10": "current measured EEF xyz + rotation6d(first two columns) + gripper_0_1",
        "task_instruction": "stack the cups",
        "action7": {
            "chunk_size": 50,
            "frequency_hz": 15,
            "gripper": "future measured target gripper_0_1",
            "rotation": "body rotvec Log(R_current.T @ R_target)",
            "translation": "base-frame target_xyz - current_xyz",
        },
        "dataset_profile": {
            "schema_version": 1,
            "profile": "action15",
            "observation_fps": 15,
            "action_fps": 15,
            "chunk_size": 50,
            "requires_project_cartesian_adapter": True,
            "requires_project_dual_rate_adapter": False,
            "real_robot_rollout_authorized": False,
        },
    }
    stats_manifest = {
        "schema_version": 1,
        "profile": "action15",
        "observation_fps": 15,
        "action_fps": 15,
        "chunk_size": 50,
        "real_robot_rollout_authorized": False,
        OBS_STATE: {
            name: [0.0] * 10 for name in ("min", "max", "mean", "std", "q01", "q99")
        },
        "action": {
            name: [0.0] * 7 for name in ("min", "max", "mean", "std", "q01", "q99")
        },
    }
    tracked = {
        "config.json": b"{}\n",
        "model.safetensors": b"fake-model",
        "policy_preprocessor.json": json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {},
                        "state_file": "preprocessor_state.safetensors",
                    }
                ]
            }
        ).encode(),
        "policy_postprocessor.json": json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "unnormalizer_processor",
                        "config": {},
                        "state_file": "postprocessor_state.safetensors",
                    }
                ]
            }
        ).encode(),
        "franka_eef_geometry_manifest.json": json.dumps(geometry_manifest).encode(),
        "pi0_eef_stats.json": json.dumps(stats_manifest).encode(),
    }
    for name, payload in tracked.items():
        (root / name).write_bytes(payload)
    (root / "preprocessor_state.safetensors").write_bytes(b"pre")
    (root / "postprocessor_state.safetensors").write_bytes(b"post")

    manifest = {
        "schema_version": 1,
        "checkpoint_type": FRANKA_CHECKPOINT_TYPE,
        "weights_namespace": PI0_CORE_WEIGHTS_NAMESPACE,
        "hub_upload": False,
        "wandb_artifact_upload": False,
        "files": {
            name: {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for name, payload in tracked.items()
        },
    }
    (root / "franka_pi0_checkpoint_manifest.json").write_text(json.dumps(manifest))
    return root


def _make_server(*, actions_per_chunk: int = 2) -> FrankaPI0PolicyServer:
    server = FrankaPI0PolicyServer(
        PolicyServerConfig(host="localhost", port=9999, fps=15, actions_per_chunk=actions_per_chunk)
    )
    server.actions_per_chunk = actions_per_chunk
    server.lerobot_features = {
        OBS_STATE: {
            "dtype": "float32",
            "shape": (10,),
            "names": list(STATE_NAMES),
        }
    }
    return server


def _make_observation(state: list[float]) -> TimedObservation:
    return TimedObservation(
        timestamp=10.0,
        timestep=4,
        observation=dict(zip(STATE_NAMES, state, strict=True)),
    )


def _client_features() -> dict[str, dict]:
    return {
        OBS_STATE: {"dtype": "float32", "shape": (10,), "names": list(STATE_NAMES)},
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
    }


def _client_specs(features: dict[str, dict], *, rename_map: dict[str, str] | None = None):
    return RemotePolicyConfig(
        policy_type="pi0",
        pretrained_name_or_path="checkpoint",
        lerobot_features=features,
        actions_per_chunk=2,
        device="cpu",
        rename_map=FRANKA_RENAME_MAP if rename_map is None else rename_map,
    )


def test_server_inherits_stock_policy_server():
    assert issubclass(FrankaPI0PolicyServer, PolicyServer)


def test_policy_setup_accepts_exact_franka_client_contract():
    server = _make_server()
    resolved = server._resolve_policy_specs(_client_specs(_client_features()))
    assert resolved.lerobot_features == _client_features()
    assert resolved.rename_map == FRANKA_RENAME_MAP


def test_prepare_observation_matches_pi0_training_letterbox():
    fixture_path = Path(__file__).parents[1] / "fixtures" / "async" / "franka_observation_v1.npz"
    with np.load(fixture_path) as fixture:
        state = fixture["state"].copy()
        cameras = {camera_key: fixture[camera_key].copy() for camera_key in ("camera1", "camera2")}

    raw_observation = dict(zip(STATE_NAMES, state.tolist(), strict=True))
    raw_observation.update(cameras)
    raw_observation["task"] = "stack the cups"

    server = _make_server()
    server.lerobot_features = _client_features()
    image_features = {
        policy_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        for policy_key in FRANKA_RENAME_MAP.values()
    }
    server.policy = SimpleNamespace(config=SimpleNamespace(image_features=image_features))

    prepared = server._prepare_observation(
        TimedObservation(timestamp=0.0, timestep=0, observation=raw_observation)
    )

    assert prepared["task"] == "stack the cups"
    assert prepared[OBS_STATE].dtype is torch.float32
    assert tuple(prepared[OBS_STATE].shape) == (1, 10)
    torch.testing.assert_close(prepared[OBS_STATE].squeeze(0), torch.from_numpy(state))

    for camera_key, policy_key in FRANKA_RENAME_MAP.items():
        training_input = (
            torch.from_numpy(cameras[camera_key.removeprefix("observation.images.")])
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div(255.0)
            .unsqueeze(0)
        )
        # PI0's model path temporarily converts BCHW to BHWC before calling
        # resize_with_pad_torch, then restores BCHW for SigLIP.
        expected = resize_with_pad_torch(
            training_input.permute(0, 2, 3, 1),
            224,
            224,
        ).permute(0, 3, 1, 2)
        actual = prepared[policy_key]

        assert tuple(actual.shape) == (1, 3, 224, 224)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert torch.count_nonzero(actual[:, :, :28, :]) == 0
        assert torch.count_nonzero(actual[:, :, -28:, :]) == 0
        assert torch.count_nonzero(actual[:, :, 28:-28, :]) > 0


@pytest.mark.parametrize("field", ["state_order", "camera_shape", "extra_feature", "rename_map"])
def test_policy_setup_rejects_client_contract_drift(field: str):
    features = _client_features()
    rename_map = FRANKA_RENAME_MAP
    if field == "state_order":
        features[OBS_STATE]["names"][0:2] = reversed(features[OBS_STATE]["names"][0:2])
    elif field == "camera_shape":
        features["observation.images.camera1"]["shape"] = (224, 224, 3)
    elif field == "extra_feature":
        features["observation.images.camera3"] = features["observation.images.camera2"].copy()
    elif field == "rename_map":
        rename_map = {}

    server = _make_server()
    with pytest.raises(FrankaAsyncPolicyContractError, match="Franka client"):
        server._resolve_policy_specs(_client_specs(features, rename_map=rename_map))


def test_checkpoint_preflight_accepts_complete_franka_envelope(tmp_path: Path):
    checkpoint = _write_checkpoint_envelope(tmp_path)
    assert validate_franka_checkpoint(checkpoint) == checkpoint.resolve()


def test_checkpoint_preflight_rejects_non_franka_manifest(tmp_path: Path):
    checkpoint = _write_checkpoint_envelope(tmp_path)
    manifest_path = checkpoint / "franka_pi0_checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_type"] = "generic_pi0"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(FrankaAsyncPolicyContractError, match="contract mismatch"):
        validate_franka_checkpoint(checkpoint)


def test_checkpoint_preflight_rejects_same_size_content_corruption(tmp_path: Path):
    checkpoint = _write_checkpoint_envelope(tmp_path)
    (checkpoint / "config.json").write_bytes(b"[]\n")

    with pytest.raises(FrankaAsyncPolicyContractError, match="SHA-256 mismatch"):
        validate_franka_checkpoint(checkpoint)


def test_strict_load_uses_project_loader_and_sets_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = _write_checkpoint_envelope(tmp_path)
    calls: list[tuple[str, object]] = []

    class DummyPolicy:
        def __init__(self, config):
            calls.append(("construct", config))
            self.training = True

        def eval(self):
            calls.append(("eval", None))
            self.training = False
            return self

    config = object()
    monkeypatch.setattr(
        async_server_module,
        "build_pi0_full_finetune_config",
        lambda path, *, device: calls.append(("config", (path, device))) or config,
    )
    monkeypatch.setattr(async_server_module, "PI0Policy", DummyPolicy)
    monkeypatch.setattr(
        async_server_module,
        "canonicalize_pi0_full_training_graph",
        lambda policy: calls.append(("canonicalize", policy)),
    )
    monkeypatch.setattr(
        async_server_module,
        "load_pi0_full_checkpoint_weights",
        lambda policy, path: calls.append(("weights", (policy, path)))
        or {"project_manifest_present": True},
    )

    server = _make_server()
    server.device = "cpu"
    policy = server._load_policy("pi0", str(checkpoint))

    assert policy.training is False
    assert [name for name, _ in calls] == ["config", "construct", "canonicalize", "weights", "eval"]


def test_strict_load_rejects_other_policy_type(tmp_path: Path):
    server = _make_server()
    server.device = "cpu"
    with pytest.raises(FrankaAsyncPolicyContractError, match="only accepts"):
        server._load_policy("act", str(tmp_path))


def test_predict_decodes_relative_chunk_against_raw_anchor(monkeypatch: pytest.MonkeyPatch):
    relative_actions = [
        torch.tensor([0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2.0, 1.2]),
        torch.tensor([-0.5, 0.0, 0.25, 0.0, 0.0, 0.0, -0.2]),
    ]

    def fake_predict(_self, _observation):
        return [
            TimedAction(timestamp=10.0 + index / 15.0, timestep=4 + index, action=action)
            for index, action in enumerate(relative_actions)
        ]

    monkeypatch.setattr(PolicyServer, "_predict_action_chunk", fake_predict)
    server = _make_server()
    observation = _make_observation([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4])

    timed_actions = server._predict_action_chunk(observation)
    absolute = torch.stack([action.get_action() for action in timed_actions])

    assert tuple(absolute.shape) == (2, 8)
    assert absolute.dtype is torch.float32
    torch.testing.assert_close(absolute[0, :3], torch.tensor([1.1, 1.8, 3.3]))
    torch.testing.assert_close(
        absolute[0, 3:7],
        torch.tensor([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=torch.float32),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(absolute[0, 7], torch.tensor(1.0))
    torch.testing.assert_close(absolute[1], torch.tensor([0.5, 2.0, 3.25, 0, 0, 0, 1, 0.0]))
    assert [action.get_timestep() for action in timed_actions] == [4, 5]
    assert [action.get_timestamp() for action in timed_actions] == [10.0, 10.0 + 1 / 15.0]


@pytest.mark.parametrize(
    ("relative_actions", "state", "message"),
    [
        ([torch.zeros(6), torch.zeros(7)], None, r"must be a \(7,\) tensor"),
        ([torch.zeros(7), torch.full((7,), torch.nan)], None, "finite floating point"),
        (
            [torch.zeros(7), torch.zeros(7)],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4],
            "Failed to decode",
        ),
    ],
)
def test_predict_fails_closed_on_invalid_shape_finite_or_geometry(
    monkeypatch: pytest.MonkeyPatch,
    relative_actions: list[torch.Tensor],
    state: list[float] | None,
    message: str,
):
    def fake_predict(_self, _observation):
        return [
            TimedAction(timestamp=float(index), timestep=index, action=action)
            for index, action in enumerate(relative_actions)
        ]

    monkeypatch.setattr(PolicyServer, "_predict_action_chunk", fake_predict)
    server = _make_server()
    valid_state = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4]
    observation = _make_observation(state or valid_state)

    with pytest.raises(FrankaAsyncPolicyContractError, match=message):
        server._predict_action_chunk(observation)


def test_predict_rejects_short_chunk_before_mutating(monkeypatch: pytest.MonkeyPatch):
    original = TimedAction(timestamp=0.0, timestep=0, action=torch.zeros(7))
    monkeypatch.setattr(PolicyServer, "_predict_action_chunk", lambda *_args: [original])
    server = _make_server(actions_per_chunk=2)
    observation = _make_observation([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.4])

    with pytest.raises(FrankaAsyncPolicyContractError, match="returned 1 actions"):
        server._predict_action_chunk(observation)
    assert tuple(original.action.shape) == (7,)
