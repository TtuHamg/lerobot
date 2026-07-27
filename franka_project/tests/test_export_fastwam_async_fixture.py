from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/export_fastwam_async_fixture.py"
SPEC = importlib.util.spec_from_file_location("export_fastwam_async_fixture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)

SEND_SCRIPT = PROJECT_ROOT / "scripts/send_async_fixture_once.py"
SEND_SPEC = importlib.util.spec_from_file_location("send_async_fixture_once", SEND_SCRIPT)
assert SEND_SPEC is not None and SEND_SPEC.loader is not None
sender = importlib.util.module_from_spec(SEND_SPEC)
sys.modules[SEND_SPEC.name] = sender
SEND_SPEC.loader.exec_module(sender)


def test_nested_contract_check_rejects_preprocessing_drift() -> None:
    actual = {
        "images": {"preprocess": "center_crop", "height": 224, "width": 224},
        "unrelated_metadata": "allowed",
    }
    exporter._require_subset(
        actual,
        {"images": {"preprocess": "center_crop", "height": 224, "width": 224}},
        where="fixture source",
    )

    actual["images"]["preprocess"] = "stretch"
    with pytest.raises(ValueError, match="fixture source.images.preprocess drifted"):
        exporter._require_subset(
            actual,
            {"images": {"preprocess": "center_crop", "height": 224, "width": 224}},
            where="fixture source",
        )


def test_committed_fastwam_fixture_matches_ledger_and_wire_contract() -> None:
    fixture_path = PROJECT_ROOT / "fixtures/async/fastwam_grab_cups_observation_v1.npz"
    ledger = json.loads(fixture_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == ledger["fixture_sha256"]
    assert ledger["fixture_kind"] == "fastwam_training_aligned_raw_observation"
    assert ledger["policy_type"] == "fastwam"
    assert ledger["source"]["output_episode_index"] == 4
    assert ledger["source"]["output_frame_index"] == 106
    assert ledger["source"]["output_global_index"] == 935
    assert ledger["source"]["source_episode_id"] == "run_20260725_124233_00108"
    assert ledger["source"]["raw_indices"] == {
        "camera1": 107,
        "camera2": 107,
        "eef": 106,
        "gripper": 1793,
    }

    wire = ledger["wire_contract"]
    assert wire["fps"] == 30
    assert wire["actions_per_chunk"] == 32
    assert wire["rename_map"] == {}
    assert wire["state_names"] == list(exporter.WIRE_STATE_NAMES)
    assert wire["camera_order"] == list(exporter.CAMERA_NAMES)

    parity = ledger["preprocessing_parity"]
    assert parity["deployment_equals_conversion_preencode"] is True
    assert parity["raw_state_to_stored_training_state8_max_abs_error"] <= exporter.STATE_TOLERANCE
    assert parity["wire_state10_to_stored_training_state8_max_abs_error"] <= exporter.STATE_TOLERANCE

    evidence = ledger["training_window_evidence"]
    assert evidence["actions_exported_to_fixture"] is False
    assert evidence["num_observation_frames"] == 33
    assert evidence["num_action_steps"] == 32
    assert evidence["first_action_step_open_target_at_or_below_0_2_1_based"] == 12
    assert evidence["minimum_future_gripper_open_target_0_1"] < 0.02
    assert evidence["eef_z_delta_m"] > 0.07

    with np.load(fixture_path, allow_pickle=False) as fixture:
        assert set(fixture.files) == {"state", "camera1", "camera2"}
        state = fixture["state"]
        assert state.shape == (10,)
        assert state.dtype == np.float32
        assert np.isfinite(state).all()
        np.testing.assert_allclose(
            state,
            np.asarray(
                [
                    0.38589417934417725,
                    -0.11898356676101685,
                    0.2034136801958084,
                    0.6242241859436035,
                    0.6957055330276489,
                    -0.35544052720069885,
                    0.6745343804359436,
                    -0.7094818949699402,
                    -0.2040560394525528,
                    0.11788711696863174,
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=0.0,
        )
        for name in exporter.CAMERA_NAMES:
            assert fixture[name].shape == exporter.WIRE_CAMERA_SHAPE
            assert fixture[name].dtype == np.uint8
            assert exporter._array_sha256(fixture[name]) == parity["images"][name]["raw_wire_sha256"]

    reconstructed = exporter._wire_to_fastwam_state(
        state,
        finger_scale=0.04,
        finger_signs=(1.0, -1.0),
    )
    np.testing.assert_allclose(
        reconstructed,
        np.asarray(evidence["anchor_state8"]),
        rtol=0.0,
        atol=exporter.STATE_TOLERANCE,
    )


def test_one_shot_sender_verifies_fixture_sha_and_task(tmp_path: Path) -> None:
    committed = PROJECT_ROOT / "fixtures/async/fastwam_grab_cups_observation_v1.npz"
    fixture = tmp_path / committed.name
    fixture.write_bytes(committed.read_bytes())
    ledger = json.loads(committed.with_suffix(".json").read_text(encoding="utf-8"))
    fixture.with_suffix(".json").write_text(json.dumps(ledger), encoding="utf-8")

    sender._verify_fixture_ledger(fixture, task="grab the paper cup.")
    with pytest.raises(ValueError, match="source.task"):
        sender._verify_fixture_ledger(fixture, task="move a different object")
