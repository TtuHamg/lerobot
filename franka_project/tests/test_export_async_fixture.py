from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/export_async_fixture.py"
SPEC = importlib.util.spec_from_file_location("export_async_fixture", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exporter
SPEC.loader.exec_module(exporter)


def _profile() -> dict:
    return {
        "schema_version": 1,
        "profile": "action15",
        "observation_fps": 15,
        "action_fps": 15,
        "chunk_size": 50,
        "requires_project_cartesian_adapter": True,
        "requires_project_dual_rate_adapter": False,
        "real_robot_rollout_authorized": False,
    }


def _info() -> dict:
    camera = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    }
    return {
        "fps": 15,
        "robot_type": "franka_fr3_cartesian_eef",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [10],
                "names": list(exporter.STATE_NAMES),
            },
            "observation.images.camera1": camera.copy(),
            "observation.images.camera2": camera.copy(),
        },
    }


def test_source_contract_rejects_state_name_drift() -> None:
    profile = _profile()
    info = _info()
    exporter._validate_source_contract(profile, info)

    info["features"]["observation.state"]["names"][0:2] = ["eef.y", "eef.x"]
    with pytest.raises(ValueError, match="observation.state feature"):
        exporter._validate_source_contract(profile, info)


def test_committed_fixture_matches_ledger_and_wire_contract() -> None:
    fixture_path = PROJECT_ROOT / "fixtures/async/franka_observation_v1.npz"
    ledger = json.loads(fixture_path.with_suffix(".json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    assert digest == ledger["fixture_sha256"]

    with np.load(fixture_path, allow_pickle=False) as fixture:
        assert set(fixture.files) == {"state", "camera1", "camera2"}
        assert fixture["state"].shape == (10,)
        assert fixture["state"].dtype == np.float32
        assert fixture["camera1"].shape == (480, 640, 3)
        assert fixture["camera1"].dtype == np.uint8
        assert fixture["camera2"].shape == (480, 640, 3)
        assert fixture["camera2"].dtype == np.uint8
        assert np.isfinite(fixture["state"]).all()
