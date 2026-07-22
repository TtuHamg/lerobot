from __future__ import annotations

import pickle  # nosec: exercises the trusted internal async setup protocol
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import franka_eef_pipeline.async_server as async_server_module
from franka_eef_pipeline.async_server import (
    FRANKA_CAMERA_KEYS,
    FRANKA_CAMERA_SHAPE,
    FRANKA_FASTWAM_RENAME_MAP,
    FRANKA_PI0_RENAME_MAP,
    FRANKA_STATE_NAMES,
    FrankaAsyncPolicyContractError,
    FrankaFastWAMCheckpointContract,
    FrankaFastWAMPolicyServer,
    FrankaPI0PolicyServer,
    _build_fastwam_state,
    _decode_fastwam_absolute_action_chunk,
    _FastWAMMinMaxNormalizer,
    _FastWAMRuntime,
    _prepare_fastwam_image,
    create_franka_policy_server,
    validate_franka_server_config,
)

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2
from lerobot.utils.constants import OBS_STATE


def _client_features() -> dict[str, dict]:
    return {
        OBS_STATE: {
            "dtype": "float32",
            "shape": (10,),
            "names": list(FRANKA_STATE_NAMES),
        },
        FRANKA_CAMERA_KEYS[0]: {
            "dtype": "image",
            "shape": FRANKA_CAMERA_SHAPE,
            "names": ["height", "width", "channels"],
        },
        FRANKA_CAMERA_KEYS[1]: {
            "dtype": "image",
            "shape": FRANKA_CAMERA_SHAPE,
            "names": ["height", "width", "channels"],
        },
    }


def _fastwam_specs(*, policy_type: str = "fastwam", actions_per_chunk: int = 32, rename_map=None):
    return RemotePolicyConfig(
        policy_type=policy_type,
        pretrained_name_or_path="server-owned",
        lerobot_features=_client_features(),
        actions_per_chunk=actions_per_chunk,
        device="cpu",
        rename_map=FRANKA_FASTWAM_RENAME_MAP if rename_map is None else rename_map,
        fps=30,
    )


@pytest.mark.parametrize(
    ("policy_type", "expected_type"),
    [("pi0", FrankaPI0PolicyServer), ("fastwam", FrankaFastWAMPolicyServer)],
)
def test_franka_server_factory_selects_backend(policy_type: str, expected_type: type) -> None:
    server = create_franka_policy_server(
        PolicyServerConfig(host="localhost", port=9999, policy_type=policy_type)
    )
    assert type(server) is expected_type


@pytest.mark.parametrize("policy_type", [None, "act"])
def test_franka_server_factory_rejects_unknown_backend(policy_type: str | None) -> None:
    with pytest.raises(ValueError, match="pi0.*fastwam"):
        create_franka_policy_server(
            PolicyServerConfig(host="localhost", port=9999, policy_type=policy_type)
        )


def test_server_config_dispatches_to_fastwam_inspector(monkeypatch) -> None:
    sentinel = object()
    calls = []

    def fake_inspector(path, *, expected_fps, expected_actions_per_chunk):
        calls.append((path, expected_fps, expected_actions_per_chunk))
        return sentinel

    monkeypatch.setattr(async_server_module, "inspect_fastwam_checkpoint_contract", fake_inspector)
    config = PolicyServerConfig(
        host="127.0.0.1",
        port=9999,
        fps=30,
        policy_type="fastwam",
        pretrained_name_or_path="checkpoint.pt",
        actions_per_chunk=32,
        policy_device="cuda",
    )

    assert validate_franka_server_config(config) is sentinel
    assert calls == [("checkpoint.pt", 30, 32)]


def test_fastwam_policy_specs_require_policy_specific_empty_rename_map() -> None:
    server = FrankaFastWAMPolicyServer(
        PolicyServerConfig(
            host="localhost",
            port=9999,
            fps=30,
            policy_type="fastwam",
            pretrained_name_or_path="server-checkpoint.pt",
            actions_per_chunk=32,
            policy_device="cuda",
        )
    )
    resolved = server._resolve_policy_specs(_fastwam_specs())
    assert resolved.rename_map == {}

    with pytest.raises(FrankaAsyncPolicyContractError, match="rename_map"):
        server._resolve_policy_specs(_fastwam_specs(rename_map=FRANKA_PI0_RENAME_MAP))
    with pytest.raises(FrankaAsyncPolicyContractError, match="policy_type mismatch"):
        server._resolve_policy_specs(_fastwam_specs(policy_type="pi0"))
    with pytest.raises(FrankaAsyncPolicyContractError, match="actions_per_chunk mismatch"):
        server._resolve_policy_specs(_fastwam_specs(actions_per_chunk=8))

    mismatched_fps = _fastwam_specs()
    mismatched_fps.fps = 15
    with pytest.raises(FrankaAsyncPolicyContractError, match="fps mismatch"):
        server._resolve_policy_specs(mismatched_fps)

    mismatched_protocol = _fastwam_specs()
    mismatched_protocol.protocol_version = 1
    with pytest.raises(FrankaAsyncPolicyContractError, match="protocol mismatch"):
        server._resolve_policy_specs(mismatched_protocol)

    legacy_specs = _fastwam_specs()
    del vars(legacy_specs)["protocol_version"]
    with pytest.raises(FrankaAsyncPolicyContractError, match="protocol mismatch"):
        server._resolve_policy_specs(legacy_specs)


def test_fastwam_policy_setup_bypasses_stock_registry_and_reuses_runtime(monkeypatch) -> None:
    server = FrankaFastWAMPolicyServer(
        PolicyServerConfig(
            host="localhost",
            port=9999,
            fps=30,
            policy_type="fastwam",
            pretrained_name_or_path="server-checkpoint.pt",
            actions_per_chunk=32,
            policy_device="cuda",
        )
    )
    model = object()
    runtime = object()
    loads = []

    def fake_load(policy_type, checkpoint):
        loads.append((policy_type, checkpoint))
        server._fastwam_runtime = runtime
        return model

    monkeypatch.setattr(server, "_load_policy", fake_load)
    request = services_pb2.PolicySetup(data=pickle.dumps(_fastwam_specs()))
    context = SimpleNamespace(peer=lambda: "test-client")

    first_ack = server.SendPolicyInstructions(request, context)
    reused_ack = server.SendPolicyInstructions(request, context)

    assert loads == [("fastwam", "server-checkpoint.pt")]
    assert server.policy is model
    assert server._fastwam_runtime is runtime
    assert server.preprocessor is None
    assert server.postprocessor is None
    assert first_ack == reused_ack
    assert first_ack.protocol_version == 2
    assert first_ack.fps == 30
    assert first_ack.policy_type == "fastwam"
    assert first_ack.actions_per_chunk == 32


def test_prepare_fastwam_image_crops_orders_and_normalizes_cameras() -> None:
    camera1 = np.full(FRANKA_CAMERA_SHAPE, 19, dtype=np.uint8)
    camera1[:, 80:560] = [0, 127, 255]
    camera2 = np.full(FRANKA_CAMERA_SHAPE, 23, dtype=np.uint8)
    camera2[:, 80:560] = [255, 128, 0]

    image = _prepare_fastwam_image(camera1, camera2)

    assert image.shape == (1, 3, 224, 448)
    assert image.dtype is torch.float32
    expected_left = torch.tensor([-1.0, -1.0 / 255.0, 1.0])
    expected_right = torch.tensor([1.0, 1.0 / 255.0, -1.0])
    torch.testing.assert_close(image[0, :, 100, 100], expected_left, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(image[0, :, 100, 300], expected_right, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        image[0, :, :, :224],
        expected_left[:, None, None].expand(3, 224, 224),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        image[0, :, :, 224:],
        expected_right[:, None, None].expand(3, 224, 224),
        rtol=0.0,
        atol=1e-7,
    )


def test_build_fastwam_state_uses_principal_rotvec_and_pseudo_fingers() -> None:
    state = np.asarray(
        [0.5, -0.2, 0.3, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.75],
        dtype=np.float64,
    )
    actual = _build_fastwam_state(state)
    expected = np.asarray([0.5, -0.2, 0.3, 0.0, 0.0, np.pi / 2.0, 0.01, -0.01])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "state",
    [
        np.full((10,), np.nan),
        np.asarray([0, 0, 0, 0, 0, 0, 0, 1, 0, 0.5], dtype=np.float64),
        np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0, 1.1], dtype=np.float64),
    ],
)
def test_build_fastwam_state_rejects_invalid_state(state: np.ndarray) -> None:
    with pytest.raises(FrankaAsyncPolicyContractError):
        _build_fastwam_state(state)


def test_fastwam_minmax_normalizer_matches_training_backward_formula() -> None:
    minimum = torch.tensor([-1, -2, -3, -4, -5, -6, 0.2], dtype=torch.float32)
    maximum = torch.tensor([1, 2, 3, 4, 5, 6, 0.8], dtype=torch.float32)
    normalizer = _FastWAMMinMaxNormalizer(
        state_min=torch.full((8,), -1.0),
        state_max=torch.full((8,), 1.0),
        action_min=minimum,
        action_max=maximum,
    )
    normalized = torch.tensor([[-1.0] * 7, [0.0] * 7, [1.0] * 7])
    actual = normalizer.denormalize_action(normalized)
    expected = torch.stack((minimum, (minimum + maximum) / 2.0, maximum))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_decode_fastwam_actions_cumulates_pose_and_converts_open_to_closed() -> None:
    anchor = np.asarray([1, 2, 3, 1, 0, 0, 0, 1, 0, 0.5], dtype=np.float64)
    actions = torch.tensor(
        [
            [0.1, 0, 0, 0, 0, np.pi / 2, 0.75],
            [0, 0.2, 0, 0, 0, np.pi / 2, 0.25],
        ],
        dtype=torch.float64,
    )
    actual = _decode_fastwam_absolute_action_chunk(anchor, actions)
    expected = torch.tensor(
        [
            [1.1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5), 0.25],
            [1.1, 2.2, 3, 0, 0, 1, 0, 0.75],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)


class _FakeFastWAM:
    def __init__(self, action: torch.Tensor):
        self.action = action
        self.kwargs = None

    def infer_action(self, **kwargs):
        self.kwargs = kwargs
        return {"action": self.action.clone()}


def test_fastwam_server_predicts_canonical_absolute8_without_model_dependencies() -> None:
    physical_actions = torch.tensor(
        [
            [0.1, 0, 0, 0, 0, np.pi / 2, 0.75],
            [0, 0.2, 0, 0, 0, np.pi / 2, 0.25],
        ],
        dtype=torch.float32,
    )
    normalized_actions = physical_actions.clone()
    normalized_actions[:, 6] = 2.0 * physical_actions[:, 6] - 1.0
    model = _FakeFastWAM(normalized_actions)
    normalizer = _FastWAMMinMaxNormalizer(
        state_min=torch.full((8,), -10.0),
        state_max=torch.full((8,), 10.0),
        action_min=torch.tensor([-1.0] * 6 + [0.0]),
        action_max=torch.tensor([1.0] * 7),
    )
    runtime = _FastWAMRuntime(
        model=model,
        normalizer=normalizer,
        context=torch.zeros((2, 4096), dtype=torch.bfloat16),
        context_mask=torch.ones((2,), dtype=torch.bool),
    )
    contract = FrankaFastWAMCheckpointContract(
        checkpoint_path=Path("checkpoint.pt"),
        run_dir=Path("run"),
        runtime_config_path=Path("runtime.yaml"),
        stats_path=Path("stats.json"),
        text_context_path=Path("context.pt"),
        deployment_manifest_path=Path("deployment.json"),
        vae_path=Path("vae.safetensors"),
        task_instruction="move the cup",
        observation_fps=30,
        action_fps=30,
        chunk_size=2,
        image_height=224,
        image_width=224,
        context_len=2,
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        checkpoint_step=1,
        model_dtype="bf16",
        num_inference_steps=10,
        seed=42,
        finger_scale=0.04,
        finger_signs=(1.0, -1.0),
    )
    server = FrankaFastWAMPolicyServer(
        PolicyServerConfig(host="localhost", port=9999, fps=30, actions_per_chunk=2)
    )
    server.actions_per_chunk = 2
    server.policy = model
    server._fastwam_runtime = runtime
    server._checkpoint_contract = contract
    server.lerobot_features = _client_features()

    state = [1, 2, 3, 1, 0, 0, 0, 1, 0, 0.5]
    raw = dict(zip(FRANKA_STATE_NAMES, state, strict=True))
    raw.update(
        {
            "camera1": np.zeros(FRANKA_CAMERA_SHAPE, dtype=np.uint8),
            "camera2": np.zeros(FRANKA_CAMERA_SHAPE, dtype=np.uint8),
            "task": "move the cup",
        }
    )
    result = server._predict_action_chunk(
        TimedObservation(timestamp=10.0, timestep=4, observation=raw)
    )
    actual = torch.stack([item.action for item in result])

    assert model.kwargs["input_image"].shape == (1, 3, 224, 448)
    assert model.kwargs["proprio"].shape == (1, 8)
    assert model.kwargs["action_horizon"] == 2
    assert model.kwargs["seed"] == 42
    expected = torch.tensor(
        [
            [1.1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5), 0.25],
            [1.1, 2.2, 3, 0, 0, 1, 0, 0.75],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
    assert [item.timestep for item in result] == [4, 5]
    assert [item.timestamp for item in result] == [10.0, 10.0 + 1 / 30.0]
