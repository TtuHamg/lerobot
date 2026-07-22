from __future__ import annotations

from pathlib import Path

import numpy as np

from franka_eef_pipeline.action_chunk import (
    AlignmentThresholds,
    GripperMapping,
    absolute_action_chunks,
    align_episode_to_camera,
    assert_dual_rate_crosscheck,
    build_dual_rate_carriers,
    model_relative_action_chunks,
    valid_anchor_indices,
)
from franka_eef_pipeline.geometry import matrix_to_quaternion_xyzw, so3_exp
from franka_eef_pipeline.mcap_reader import EpisodeSignals, ImageStream, NumericStream


def _image_stream(time_ns: np.ndarray) -> ImageStream:
    count = len(time_ns)
    return ImageStream(
        log_time_ns=time_ns,
        publish_time_ns=time_ns.copy(),
        header_time_ns=time_ns.copy(),
        frame_ids=("camera",) * count,
        height=np.full(count, 480, dtype=np.int64),
        width=np.full(count, 640, dtype=np.int64),
        encoding=("rgb8",) * count,
    )


def _numeric_stream(time_ns: np.ndarray, values: np.ndarray, names: tuple[str, ...]) -> NumericStream:
    return NumericStream(
        log_time_ns=time_ns,
        publish_time_ns=time_ns.copy(),
        header_time_ns=time_ns.copy(),
        frame_ids=("base",) * len(time_ns),
        values=values.astype(np.float64),
        names=names,
    )


def _signals(*, cam2_start_index: int = 0) -> EpisodeSignals:
    camera_period = 66_666_667
    eef_period = 33_333_333
    camera_time = np.arange(65, dtype=np.int64) * camera_period + 1_000_000_000
    cam2_time = camera_time[cam2_start_index:] - 5_000_000
    eef_time = np.arange(132, dtype=np.int64) * eef_period + 990_000_000
    gripper_time = np.arange(2200, dtype=np.int64) * 2_000_000 + 990_000_000
    qpos_time = np.arange(4400, dtype=np.int64) * 1_000_000 + 990_000_000

    eef_seconds = (eef_time - eef_time[0]).astype(np.float64) / 1e9
    position = np.stack((0.4 + 0.01 * eef_seconds, -0.2 + 0.005 * eef_seconds, 0.3 * np.ones_like(eef_seconds)), axis=-1)
    rotation = so3_exp(np.stack((np.zeros_like(eef_seconds), np.zeros_like(eef_seconds), 0.03 * eef_seconds), axis=-1))
    quaternion = matrix_to_quaternion_xyzw(rotation)
    pose = np.concatenate((position, quaternion), axis=-1)
    gripper = np.linspace(0.0, 0.8, len(gripper_time), dtype=np.float64)[:, None]
    qpos = np.zeros((len(qpos_time), 7), dtype=np.float64)

    return EpisodeSignals(
        episode_id="synthetic",
        mcap_path=Path("synthetic.mcap"),
        cam1=_image_stream(camera_time),
        cam2=_image_stream(cam2_time),
        eef_pose_xyzw=_numeric_stream(eef_time, pose, ("x", "y", "z", "qx", "qy", "qz", "qw")),
        gripper=_numeric_stream(gripper_time, gripper, ("gripper",)),
        qpos=_numeric_stream(qpos_time, qpos, tuple(f"joint_{index}" for index in range(7))),
    )


def _thresholds() -> AlignmentThresholds:
    return AlignmentThresholds(
        cam2_age_ns=100_000_000,
        eef_age_ns=50_000_000,
        gripper_age_ns=10_000_000,
        qpos_age_ns=10_000_000,
    )


def test_common_interval_crops_missing_cam2_prefix_without_zero_fill() -> None:
    signals = _signals(cam2_start_index=3)
    aligned = align_episode_to_camera(signals, _thresholds())
    assert aligned.cam1_raw_index[0] >= 3
    assert np.all(aligned.cam2_raw_index >= 0)
    assert np.all(aligned.observation_valid)


def test_dual_rate_carrier_crosscheck_and_shapes() -> None:
    signals = _signals()
    aligned = align_episode_to_camera(signals, _thresholds())
    carriers = build_dual_rate_carriers(
        signals,
        aligned,
        _thresholds(),
        GripperMapping(raw_open=0.0, raw_closed=0.8),
    )
    assert_dual_rate_crosscheck(carriers)
    assert carriers.observation_state.shape == (aligned.num_camera_anchors, 10)
    assert carriers.action_15hz.shape == (aligned.num_camera_anchors - 1, 8)
    assert carriers.action_30hz.shape == (2 * (aligned.num_camera_anchors - 1), 8)
    assert np.all(carriers.action_30hz_target_time_ns[0::2] < carriers.action_30hz_target_time_ns[1::2])


def test_full_horizon_sampler_and_relative_chunks() -> None:
    signals = _signals()
    aligned = align_episode_to_camera(signals, _thresholds())
    carriers = build_dual_rate_carriers(
        signals,
        aligned,
        _thresholds(),
        GripperMapping(raw_open=0.0, raw_closed=0.8),
    )
    anchors_15 = valid_anchor_indices(carriers, profile="action15")
    anchors_30 = valid_anchor_indices(carriers, profile="action30")
    np.testing.assert_array_equal(anchors_15, anchors_30)
    assert len(anchors_15) == aligned.num_camera_anchors - 50

    absolute_15 = absolute_action_chunks(carriers, anchors_15[:2], profile="action15")
    absolute_30 = absolute_action_chunks(carriers, anchors_30[:2], profile="action30")
    relative_15 = model_relative_action_chunks(carriers, anchors_15[:2], profile="action15")
    relative_30 = model_relative_action_chunks(carriers, anchors_30[:2], profile="action30")
    assert absolute_15.shape == (2, 50, 8)
    assert absolute_30.shape == (2, 100, 8)
    assert relative_15.shape == (2, 50, 7)
    assert relative_30.shape == (2, 100, 7)
    np.testing.assert_allclose(absolute_30[:, 1::2], absolute_15, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(relative_30[:, 1::2], relative_15, atol=1e-12, rtol=0.0)
