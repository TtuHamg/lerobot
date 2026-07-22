from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/convert_native30_to_lerobot.py"
SPEC = importlib.util.spec_from_file_location("convert_native30_to_lerobot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _config() -> dict:
    config = converter._load_yaml(converter.DEFAULT_CONFIG)
    # Unit tests must not depend on the external raw-data mount.
    return config


def _carriers(*, gap_after: int | None = None) -> SimpleNamespace:
    count = 80
    interval_ns = 33_333_333
    intervals = np.full(count - 1, interval_ns, dtype=np.int64)
    if gap_after is not None:
        intervals[gap_after] = 200_000_000
    camera_time = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(intervals, dtype=np.int64))
    )
    return SimpleNamespace(
        camera_log_time_ns=camera_time,
        observation_valid=np.ones(count, dtype=np.bool_),
        action_15hz_valid=np.ones(count - 1, dtype=np.bool_),
    )


def _manifest_config(tmp_path: Path, episodes: list[dict], scope: dict) -> dict:
    manifest_episodes = []
    for episode in episodes:
        episode_id = episode["episode_id"]
        episode_path = tmp_path / episode_id
        episode_path.mkdir()
        mcap_name = f"{episode_id}.mcap"
        (episode_path / mcap_name).touch()
        manifest_episodes.append(
            {
                **episode,
                "path": str(episode_path),
                "relative_path": episode_id,
                "mcap_files": [mcap_name],
            }
        )
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        converter.yaml.safe_dump(
            {
                "dataset": "test",
                "split": "train",
                "data_root": str(tmp_path),
                "n_episodes": len(manifest_episodes),
                "episodes": manifest_episodes,
            },
            sort_keys=False,
        )
    )
    manifest_hash = converter._sha256_file(manifest_path)
    config = _config()
    config["scope"] = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "scope_content_sha256": manifest_hash,
        "expected_episode_count": len(manifest_episodes),
        **scope,
    }
    return config


def test_frozen_native30_contract_and_task_instruction() -> None:
    config = _config()
    converter._validate_contract(config)

    assert config["contract"]["profile"] == "native30"
    assert config["contract"]["observation_fps"] == 30
    assert config["contract"]["action_fps"] == 30
    assert config["contract"]["chunk_size"] == 50
    assert config["contract"]["task_instruction"] == "pick up the potato chip"


def test_native30_complete_horizons_use_one_endpoint_per_camera_interval() -> None:
    anchors = converter._valid_native30_anchor_indices(_carriers(), _config())

    # 80 camera anchors give 79 endpoint actions and 30 complete K=50 anchors:
    # starts 0..29, with final action indices 49..78 respectively.
    np.testing.assert_array_equal(anchors, np.arange(30, dtype=np.int64))


def test_native30_horizon_never_crosses_a_long_cam1_gap() -> None:
    anchors = converter._valid_native30_anchor_indices(
        _carriers(gap_after=20), _config()
    )

    # Windows 0..20 include the long interval at index 20 and are rejected.
    np.testing.assert_array_equal(anchors, np.arange(21, 30, dtype=np.int64))


def test_native30_horizon_respects_shared_stream_validity() -> None:
    carriers = _carriers()
    carriers.action_15hz_valid[60] = False

    anchors = converter._valid_native30_anchor_indices(carriers, _config())

    # Starts 11..29 include endpoint index 60; starts 0..10 remain valid.
    np.testing.assert_array_equal(anchors, np.arange(11, dtype=np.int64))


def test_legacy_scope_does_not_implicitly_accept_relaxed_warning(tmp_path: Path) -> None:
    config = _manifest_config(
        tmp_path,
        [
            {
                "episode_id": "warning",
                "status": "WARN",
                "ok_for_training": False,
                "ok_for_training_relaxed": True,
            }
        ],
        {"required_status": "WARN", "require_ok_for_training": True},
    )

    with pytest.raises(RuntimeError, match="ok_for_training is not true"):
        converter._validate_manifest(config, limit_episodes=None)


def test_mixed_scope_uses_status_specific_eligibility_fields(tmp_path: Path) -> None:
    config = _manifest_config(
        tmp_path,
        [
            {
                "episode_id": "passed",
                "status": "PASS",
                "ok_for_training": True,
                "ok_for_training_relaxed": True,
            },
            {
                "episode_id": "warning",
                "status": "WARN",
                "ok_for_training": False,
                "ok_for_training_relaxed": True,
                "warn_reasons": ["camera gap"],
            },
        ],
        {
            "allowed_statuses": ["PASS", "WARN"],
            "require_ok_for_training": True,
            "training_eligibility_fields": {
                "PASS": "ok_for_training",
                "WARN": "ok_for_training_relaxed",
            },
        },
    )

    _, episodes, paths = converter._validate_manifest(config, limit_episodes=None)

    assert [episode["episode_id"] for episode in episodes] == ["passed", "warning"]
    assert all(path.is_file() for path in paths)
    assert converter._source_quality_summary(episodes) == {
        "status_counts": {"PASS": 1, "WARN": 1},
        "strict_eligible_episodes": 1,
        "relaxed_eligible_episodes": 2,
        "warning_reason_count": 1,
    }


@pytest.mark.parametrize(
    ("episode", "error_field"),
    [
        (
            {
                "episode_id": "passed",
                "status": "PASS",
                "ok_for_training": False,
                "ok_for_training_relaxed": True,
            },
            "ok_for_training",
        ),
        (
            {
                "episode_id": "warning",
                "status": "WARN",
                "ok_for_training": False,
                "ok_for_training_relaxed": False,
            },
            "ok_for_training_relaxed",
        ),
    ],
)
def test_mixed_scope_fails_closed_on_wrong_eligibility(
    tmp_path: Path, episode: dict, error_field: str
) -> None:
    config = _manifest_config(
        tmp_path,
        [episode],
        {
            "allowed_statuses": ["PASS", "WARN"],
            "require_ok_for_training": True,
            "training_eligibility_fields": {
                "PASS": "ok_for_training",
                "WARN": "ok_for_training_relaxed",
            },
        },
    )

    with pytest.raises(RuntimeError, match=rf"{error_field} is not true"):
        converter._validate_manifest(config, limit_episodes=None)


def test_scope_rejects_ambiguous_status_configuration() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        converter._scope_policy(
            {
                "required_status": "PASS",
                "allowed_statuses": ["PASS", "WARN"],
            }
        )
