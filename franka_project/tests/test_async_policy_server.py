from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import franka_eef_pipeline.async_server as async_server_module
import numpy as np
import pytest
import torch
from franka_eef_pipeline.async_server import (
    FRANKA_CHECKPOINT_TYPE,
    FRANKA_RENAME_MAP,
    FrankaAsyncPolicyContractError,
    FrankaCheckpointContract,
    FrankaPI0PolicyServer,
    inspect_franka_checkpoint_contract,
    validate_franka_checkpoint,
    validate_franka_server_config,
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
ROLLOUT_MISSING = object()
SOURCE_DATASET_HASH = "a" * 64


def _write_checkpoint_envelope(
    root: Path,
    *,
    profile: str = "action15",
    task: str = "stack the cups",
    fps: int = 15,
    chunk_size: int = 50,
    geometry_rollout: object = False,
    stats_rollout: object = False,
    stats_profile: str | None = None,
    stats_fps: int | None = None,
) -> Path:
    dataset_profile = {
        "schema_version": 1,
        "profile": profile,
        "task_instruction": task,
        "observation_fps": fps,
        "action_fps": fps,
        "chunk_size": chunk_size,
        "requires_project_cartesian_adapter": True,
        "requires_project_dual_rate_adapter": False,
        "partial_conversion": False,
        "source_dataset_hash": SOURCE_DATASET_HASH,
    }
    if geometry_rollout is not ROLLOUT_MISSING:
        dataset_profile["real_robot_rollout_authorized"] = geometry_rollout

    geometry_manifest = {
        "schema_version": 1,
        "state10": "current measured EEF xyz + rotation6d(first two columns) + gripper_0_1",
        "task_instruction": task,
        "action7": {
            "chunk_size": chunk_size,
            "frequency_hz": fps,
            "gripper": "future measured target gripper_0_1",
            "rotation": "body rotvec Log(R_current.T @ R_target)",
            "translation": "base-frame target_xyz - current_xyz",
        },
        "dataset_profile": dataset_profile,
    }
    stats_manifest = {
        "schema_version": 1,
        "profile": profile if stats_profile is None else stats_profile,
        "observation_fps": fps if stats_fps is None else stats_fps,
        "action_fps": fps if stats_fps is None else stats_fps,
        "chunk_size": chunk_size,
        "source_dataset_hash": SOURCE_DATASET_HASH,
        OBS_STATE: {name: [0.0] * 10 for name in ("min", "max", "mean", "std", "q01", "q99")},
        "action": {name: [0.0] * 7 for name in ("min", "max", "mean", "std", "q01", "q99")},
    }
    if stats_rollout is not ROLLOUT_MISSING:
        stats_manifest["real_robot_rollout_authorized"] = stats_rollout

    model_config = {
        "type": "pi0",
        "input_features": {
            OBS_STATE: {"type": "STATE", "shape": [10]},
        },
        "output_features": {
            "action": {"type": "ACTION", "shape": [7]},
        },
        "chunk_size": chunk_size,
        "n_action_steps": chunk_size,
        "use_relative_actions": False,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        },
    }
    tracked = {
        "config.json": f"{json.dumps(model_config)}\n".encode(),
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
        fps=15,
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


@pytest.mark.parametrize("task", ["stack the cups", "", None])
def test_prepare_observation_rejects_task_that_does_not_match_checkpoint(task: object):
    server = _make_server()
    server._checkpoint_contract = FrankaCheckpointContract(
        profile="native30",
        task_instruction="pick up the potato chip",
        observation_fps=30,
        action_fps=30,
        chunk_size=50,
        real_robot_rollout_authorized=False,
        rollout_authorization_declared=False,
    )
    observation = {"task": task} if task is not None else {}

    with pytest.raises(FrankaAsyncPolicyContractError, match="task does not match"):
        server._prepare_observation(TimedObservation(timestamp=0.0, timestep=0, observation=observation))


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


@pytest.mark.parametrize(
    ("profile", "task", "fps", "chunk_size", "rollout_value", "rollout_declared"),
    [
        ("action15", "stack the cups", 15, 50, False, True),
        ("native30", "pick up the potato chip", 30, 50, ROLLOUT_MISSING, False),
        ("future24", "future checkpoint task", 24, 17, False, True),
    ],
)
def test_checkpoint_preflight_uses_checkpoint_contract(
    tmp_path: Path,
    profile: str,
    task: str,
    fps: int,
    chunk_size: int,
    rollout_value: object,
    rollout_declared: bool,
):
    checkpoint = _write_checkpoint_envelope(
        tmp_path,
        profile=profile,
        task=task,
        fps=fps,
        chunk_size=chunk_size,
        geometry_rollout=rollout_value,
        stats_rollout=rollout_value,
    )

    contract = inspect_franka_checkpoint_contract(
        checkpoint,
        expected_fps=fps,
        expected_actions_per_chunk=chunk_size,
    )
    assert contract == FrankaCheckpointContract(
        profile=profile,
        task_instruction=task,
        observation_fps=fps,
        action_fps=fps,
        chunk_size=chunk_size,
        real_robot_rollout_authorized=False,
        rollout_authorization_declared=rollout_declared,
    )
    assert (
        validate_franka_checkpoint(
            checkpoint,
            expected_fps=fps,
            expected_actions_per_chunk=chunk_size,
        )
        == checkpoint.resolve()
    )


@pytest.mark.parametrize(
    ("expected_fps", "expected_chunk", "message"),
    [(15, 50, "fps mismatch"), (30, 49, "actions_per_chunk mismatch")],
)
def test_checkpoint_contract_rejects_cli_timing_mismatch(
    tmp_path: Path,
    expected_fps: int,
    expected_chunk: int,
    message: str,
):
    checkpoint = _write_checkpoint_envelope(
        tmp_path,
        profile="native30",
        task="pick up the potato chip",
        fps=30,
        geometry_rollout=ROLLOUT_MISSING,
        stats_rollout=ROLLOUT_MISSING,
    )

    with pytest.raises(FrankaAsyncPolicyContractError, match=message):
        inspect_franka_checkpoint_contract(
            checkpoint,
            expected_fps=expected_fps,
            expected_actions_per_chunk=expected_chunk,
        )


@pytest.mark.parametrize("rollout_value", [True, None, 1, "false"])
def test_checkpoint_contract_rejects_rollout_self_authorization_or_malformed_value(
    tmp_path: Path,
    rollout_value: object,
):
    checkpoint = _write_checkpoint_envelope(
        tmp_path,
        geometry_rollout=rollout_value,
        stats_rollout=rollout_value,
    )

    with pytest.raises(FrankaAsyncPolicyContractError, match="must be false"):
        inspect_franka_checkpoint_contract(checkpoint)


@pytest.mark.parametrize(
    ("geometry_rollout", "stats_rollout"),
    [(False, ROLLOUT_MISSING), (ROLLOUT_MISSING, False)],
)
def test_checkpoint_contract_rejects_inconsistent_rollout_declaration(
    tmp_path: Path,
    geometry_rollout: object,
    stats_rollout: object,
):
    checkpoint = _write_checkpoint_envelope(
        tmp_path,
        geometry_rollout=geometry_rollout,
        stats_rollout=stats_rollout,
    )

    with pytest.raises(FrankaAsyncPolicyContractError, match="declaration must match"):
        inspect_franka_checkpoint_contract(checkpoint)


@pytest.mark.parametrize(
    ("stats_profile", "stats_fps"),
    [("native30", None), (None, 30)],
)
def test_checkpoint_contract_rejects_geometry_stats_drift(
    tmp_path: Path,
    stats_profile: str | None,
    stats_fps: int | None,
):
    checkpoint = _write_checkpoint_envelope(
        tmp_path,
        stats_profile=stats_profile,
        stats_fps=stats_fps,
    )

    with pytest.raises(FrankaAsyncPolicyContractError, match="stats manifest contract mismatch"):
        inspect_franka_checkpoint_contract(checkpoint)


@pytest.mark.parametrize(("fps", "chunk_size"), [(15, 50), (30, 50), (24, 17)])
def test_server_config_uses_selected_checkpoint_timing(
    tmp_path: Path,
    fps: int,
    chunk_size: int,
):
    checkpoint = _write_checkpoint_envelope(
        tmp_path,
        profile=f"profile{fps}",
        task=f"task at {fps} hz",
        fps=fps,
        chunk_size=chunk_size,
    )
    config = PolicyServerConfig(
        host="127.0.0.1",
        port=45678,
        fps=fps,
        policy_type="pi0",
        pretrained_name_or_path=str(checkpoint),
        actions_per_chunk=chunk_size,
        policy_device="cpu",
    )

    contract = validate_franka_server_config(config)

    assert contract.observation_fps == fps
    assert contract.chunk_size == chunk_size


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
    config_path = checkpoint / "config.json"
    corrupted = bytearray(config_path.read_bytes())
    pi0_offset = corrupted.index(b'"pi0"') + 1
    corrupted[pi0_offset] = ord("x")
    config_path.write_bytes(corrupted)

    with pytest.raises(FrankaAsyncPolicyContractError, match="SHA-256 mismatch"):
        validate_franka_checkpoint(checkpoint)


def test_strict_load_uses_project_loader_and_sets_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import franka_eef_pipeline.pi0_training as pi0_training_module
    import lerobot.policies.pi0.modeling_pi0 as pi0_modeling_module

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
        pi0_training_module,
        "build_pi0_full_finetune_config",
        lambda path, *, device: calls.append(("config", (path, device))) or config,
    )
    monkeypatch.setattr(pi0_modeling_module, "PI0Policy", DummyPolicy)
    monkeypatch.setattr(
        pi0_training_module,
        "canonicalize_pi0_full_training_graph",
        lambda policy: calls.append(("canonicalize", policy)),
    )
    monkeypatch.setattr(
        pi0_training_module,
        "load_pi0_full_checkpoint_weights",
        lambda policy, path: calls.append(("weights", (policy, path))) or {"project_manifest_present": True},
    )

    server = _make_server(actions_per_chunk=50)
    server.device = "cpu"
    policy = server._load_policy("pi0", str(checkpoint))

    assert policy.training is False
    assert server._checkpoint_contract.task_instruction == "stack the cups"
    assert [name for name, _ in calls] == ["config", "construct", "canonicalize", "weights", "eval"]


def test_strict_load_rejects_other_policy_type(tmp_path: Path):
    server = _make_server()
    server.device = "cpu"
    with pytest.raises(FrankaAsyncPolicyContractError, match="only accepts"):
        server._load_policy("act", str(tmp_path))


def test_failed_reload_keeps_previous_checkpoint_contract(tmp_path: Path):
    previous_contract = FrankaCheckpointContract(
        profile="action15",
        task_instruction="stack the cups",
        observation_fps=15,
        action_fps=15,
        chunk_size=50,
        real_robot_rollout_authorized=False,
        rollout_authorization_declared=True,
    )
    server = _make_server(actions_per_chunk=50)
    server.device = "cpu"
    server._checkpoint_contract = previous_contract

    with pytest.raises(FrankaAsyncPolicyContractError, match="local directory"):
        server._load_policy("pi0", str(tmp_path / "missing"))

    assert server._checkpoint_contract is previous_contract


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
