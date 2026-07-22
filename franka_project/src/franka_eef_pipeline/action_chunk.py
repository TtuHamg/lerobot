"""Timestamp alignment and dual-rate Cartesian action construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    encode_relative_action,
    enforce_quaternion_continuity,
    matrix_to_rotation_6d,
    normalize_quaternion_xyzw,
    quaternion_xyzw_to_matrix,
)
from .mcap_reader import EpisodeSignals, latest_not_after_indices


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class AlignmentThresholds:
    cam2_age_ns: int
    eef_age_ns: int
    gripper_age_ns: int
    qpos_age_ns: int


@dataclass(frozen=True)
class GripperMapping:
    raw_open: float
    raw_closed: float
    clip: bool = True

    def encode(self, raw: NDArray[np.floating]) -> FloatArray:
        if not np.isfinite(self.raw_open) or not np.isfinite(self.raw_closed):
            raise ValueError("gripper endpoints must be finite")
        span = self.raw_closed - self.raw_open
        if abs(span) <= 1e-12:
            raise ValueError("gripper endpoints must differ")
        value = (np.asarray(raw, dtype=np.float64) - self.raw_open) / span
        if self.clip:
            value = np.clip(value, 0.0, 1.0)
        return value


@dataclass(frozen=True)
class AlignedEpisode:
    episode_id: str
    camera_log_time_ns: IntArray
    cam1_raw_index: IntArray
    cam2_raw_index: IntArray
    eef_raw_index: IntArray
    gripper_raw_index: IntArray
    qpos_raw_index: IntArray
    cam2_age_ns: IntArray
    eef_age_ns: IntArray
    gripper_age_ns: IntArray
    qpos_age_ns: IntArray
    observation_valid: BoolArray
    eef_pose_xyzw: FloatArray
    gripper_raw: FloatArray
    qpos: FloatArray
    common_start_ns: int
    common_end_ns: int

    @property
    def num_camera_anchors(self) -> int:
        return len(self.camera_log_time_ns)


@dataclass(frozen=True)
class DualRateCarriers:
    episode_id: str
    observation_state: FloatArray
    observation_pose_xyzw: FloatArray
    camera_log_time_ns: IntArray
    observation_valid: BoolArray
    action_15hz: FloatArray
    action_15hz_target_time_ns: IntArray
    action_15hz_eef_source_time_ns: IntArray
    action_15hz_gripper_source_time_ns: IntArray
    action_15hz_valid: BoolArray
    action_30hz: FloatArray
    action_30hz_target_time_ns: IntArray
    action_30hz_eef_source_time_ns: IntArray
    action_30hz_gripper_source_time_ns: IntArray
    action_30hz_valid: BoolArray


def _age_ns(target_time_ns: IntArray, source_time_ns: IntArray, source_index: IntArray) -> IntArray:
    age = np.full(target_time_ns.shape, np.iinfo(np.int64).max, dtype=np.int64)
    valid = source_index >= 0
    age[valid] = target_time_ns[valid] - source_time_ns[source_index[valid]]
    return age


def align_episode_to_camera(
    signals: EpisodeSignals,
    thresholds: AlignmentThresholds,
) -> AlignedEpisode:
    """Align all numeric streams and cam2 causally to the cam1 log-time axis."""

    common_start = max(
        int(signals.cam1.log_time_ns[0]),
        int(signals.cam2.log_time_ns[0]),
        int(signals.eef_pose_xyzw.log_time_ns[0]),
        int(signals.gripper.log_time_ns[0]),
    )
    common_end = min(
        int(signals.cam1.log_time_ns[-1]),
        int(signals.cam2.log_time_ns[-1]),
        int(signals.eef_pose_xyzw.log_time_ns[-1]),
        int(signals.gripper.log_time_ns[-1]),
    )
    if common_end <= common_start:
        raise ValueError(f"{signals.episode_id}: required streams have no common interval")

    cam1_mask = (signals.cam1.log_time_ns >= common_start) & (signals.cam1.log_time_ns <= common_end)
    cam1_raw_index = np.flatnonzero(cam1_mask).astype(np.int64)
    camera_time = signals.cam1.log_time_ns[cam1_raw_index]
    if len(camera_time) < 2:
        raise ValueError(f"{signals.episode_id}: fewer than two cam1 frames in common interval")

    cam2_index = latest_not_after_indices(signals.cam2.log_time_ns, camera_time)
    eef_index = latest_not_after_indices(signals.eef_pose_xyzw.log_time_ns, camera_time)
    gripper_index = latest_not_after_indices(signals.gripper.log_time_ns, camera_time)
    qpos_index = latest_not_after_indices(signals.qpos.log_time_ns, camera_time)

    cam2_age = _age_ns(camera_time, signals.cam2.log_time_ns, cam2_index)
    eef_age = _age_ns(camera_time, signals.eef_pose_xyzw.log_time_ns, eef_index)
    gripper_age = _age_ns(camera_time, signals.gripper.log_time_ns, gripper_index)
    qpos_age = _age_ns(camera_time, signals.qpos.log_time_ns, qpos_index)
    valid = (
        (cam2_index >= 0)
        & (eef_index >= 0)
        & (gripper_index >= 0)
        & (cam2_age <= thresholds.cam2_age_ns)
        & (eef_age <= thresholds.eef_age_ns)
        & (gripper_age <= thresholds.gripper_age_ns)
    )
    aligned_qpos = np.full((len(camera_time), 7), np.nan, dtype=np.float64)
    qpos_available = qpos_index >= 0
    aligned_qpos[qpos_available] = signals.qpos.values[qpos_index[qpos_available]]

    return AlignedEpisode(
        episode_id=signals.episode_id,
        camera_log_time_ns=camera_time,
        cam1_raw_index=cam1_raw_index,
        cam2_raw_index=cam2_index,
        eef_raw_index=eef_index,
        gripper_raw_index=gripper_index,
        qpos_raw_index=qpos_index,
        cam2_age_ns=cam2_age,
        eef_age_ns=eef_age,
        gripper_age_ns=gripper_age,
        qpos_age_ns=qpos_age,
        observation_valid=valid,
        eef_pose_xyzw=signals.eef_pose_xyzw.values[eef_index],
        gripper_raw=signals.gripper.values[gripper_index],
        qpos=aligned_qpos,
        common_start_ns=common_start,
        common_end_ns=common_end,
    )


def _continuous_eef_quaternions(signals: EpisodeSignals) -> FloatArray:
    quaternion = normalize_quaternion_xyzw(signals.eef_pose_xyzw.values[:, 3:7])
    return enforce_quaternion_continuity(quaternion)


def _absolute_carrier(
    signals: EpisodeSignals,
    eef_index: IntArray,
    gripper_index: IntArray,
    gripper_mapping: GripperMapping,
    continuous_quaternion: FloatArray,
) -> FloatArray:
    position = signals.eef_pose_xyzw.values[eef_index, :3]
    quaternion = continuous_quaternion[eef_index]
    gripper = gripper_mapping.encode(signals.gripper.values[gripper_index])
    return np.concatenate((position, quaternion, gripper), axis=-1)


def build_dual_rate_carriers(
    signals: EpisodeSignals,
    aligned: AlignedEpisode,
    thresholds: AlignmentThresholds,
    gripper_mapping: GripperMapping,
) -> DualRateCarriers:
    """Build paired 15 Hz and 30 Hz absolute carrier streams.

    The 30 Hz stream has two slots per camera interval: midpoint then endpoint.
    Every endpoint is copied from the corresponding 15 Hz carrier, which gives
    an exact cross-profile audit identity.
    """

    camera_time = aligned.camera_log_time_ns
    num_intervals = len(camera_time) - 1
    if num_intervals < 1:
        raise ValueError("at least one camera interval is required")
    continuous_quaternion = _continuous_eef_quaternions(signals)

    aligned_quaternion = continuous_quaternion[aligned.eef_raw_index]
    aligned_rotation = quaternion_xyzw_to_matrix(aligned_quaternion)
    observation_state = np.concatenate(
        (
            aligned.eef_pose_xyzw[:, :3],
            matrix_to_rotation_6d(aligned_rotation),
            gripper_mapping.encode(aligned.gripper_raw),
        ),
        axis=-1,
    )
    observation_pose = np.concatenate((aligned.eef_pose_xyzw[:, :3], aligned_quaternion), axis=-1)

    endpoint_time = camera_time[1:]
    endpoint_eef_index = aligned.eef_raw_index[1:]
    endpoint_gripper_index = aligned.gripper_raw_index[1:]
    action_15 = _absolute_carrier(
        signals,
        endpoint_eef_index,
        endpoint_gripper_index,
        gripper_mapping,
        continuous_quaternion,
    )
    endpoint_valid = aligned.observation_valid[1:].copy()

    midpoint_time = camera_time[:-1] + (camera_time[1:] - camera_time[:-1]) // 2
    midpoint_eef_index = latest_not_after_indices(signals.eef_pose_xyzw.log_time_ns, midpoint_time)
    midpoint_gripper_index = latest_not_after_indices(signals.gripper.log_time_ns, midpoint_time)
    midpoint_eef_age = _age_ns(midpoint_time, signals.eef_pose_xyzw.log_time_ns, midpoint_eef_index)
    midpoint_gripper_age = _age_ns(midpoint_time, signals.gripper.log_time_ns, midpoint_gripper_index)
    midpoint_valid = (
        (midpoint_eef_index >= 0)
        & (midpoint_gripper_index >= 0)
        & (midpoint_eef_age <= thresholds.eef_age_ns)
        & (midpoint_gripper_age <= thresholds.gripper_age_ns)
    )
    midpoint_action = _absolute_carrier(
        signals,
        midpoint_eef_index,
        midpoint_gripper_index,
        gripper_mapping,
        continuous_quaternion,
    )

    action_30 = np.empty((2 * num_intervals, 8), dtype=np.float64)
    action_30[0::2] = midpoint_action
    action_30[1::2] = action_15
    target_time_30 = np.empty(2 * num_intervals, dtype=np.int64)
    target_time_30[0::2] = midpoint_time
    target_time_30[1::2] = endpoint_time
    eef_source_time_30 = np.empty(2 * num_intervals, dtype=np.int64)
    eef_source_time_30[0::2] = signals.eef_pose_xyzw.log_time_ns[midpoint_eef_index]
    eef_source_time_30[1::2] = signals.eef_pose_xyzw.log_time_ns[endpoint_eef_index]
    gripper_source_time_30 = np.empty(2 * num_intervals, dtype=np.int64)
    gripper_source_time_30[0::2] = signals.gripper.log_time_ns[midpoint_gripper_index]
    gripper_source_time_30[1::2] = signals.gripper.log_time_ns[endpoint_gripper_index]
    valid_30 = np.empty(2 * num_intervals, dtype=np.bool_)
    valid_30[0::2] = midpoint_valid
    valid_30[1::2] = endpoint_valid

    return DualRateCarriers(
        episode_id=signals.episode_id,
        observation_state=observation_state,
        observation_pose_xyzw=observation_pose,
        camera_log_time_ns=camera_time,
        observation_valid=aligned.observation_valid,
        action_15hz=action_15,
        action_15hz_target_time_ns=endpoint_time,
        action_15hz_eef_source_time_ns=signals.eef_pose_xyzw.log_time_ns[endpoint_eef_index],
        action_15hz_gripper_source_time_ns=signals.gripper.log_time_ns[endpoint_gripper_index],
        action_15hz_valid=endpoint_valid,
        action_30hz=action_30,
        action_30hz_target_time_ns=target_time_30,
        action_30hz_eef_source_time_ns=eef_source_time_30,
        action_30hz_gripper_source_time_ns=gripper_source_time_30,
        action_30hz_valid=valid_30,
    )


def valid_anchor_indices(
    carriers: DualRateCarriers,
    *,
    profile: str,
    horizon_camera_intervals: int = 50,
) -> IntArray:
    """Return camera anchors with a complete, unpadded action horizon."""

    count = len(carriers.camera_log_time_ns)
    max_anchor = count - horizon_camera_intervals
    if max_anchor <= 0:
        return np.empty(0, dtype=np.int64)
    anchors = np.arange(max_anchor, dtype=np.int64)
    if profile == "action15":
        valid = np.asarray(
            [
                carriers.observation_valid[index]
                and np.all(carriers.action_15hz_valid[index : index + horizon_camera_intervals])
                for index in anchors
            ],
            dtype=np.bool_,
        )
    elif profile == "action30":
        horizon = 2 * horizon_camera_intervals
        valid = np.asarray(
            [
                carriers.observation_valid[index]
                and np.all(carriers.action_30hz_valid[2 * index : 2 * index + horizon])
                for index in anchors
            ],
            dtype=np.bool_,
        )
    else:
        raise ValueError(f"unknown action profile: {profile}")
    return anchors[valid]


def absolute_action_chunks(
    carriers: DualRateCarriers,
    anchors: IntArray,
    *,
    profile: str,
    horizon_camera_intervals: int = 50,
) -> FloatArray:
    """Gather absolute ``[anchor, K, 8]`` carrier chunks."""

    anchor_array = np.asarray(anchors, dtype=np.int64)
    if anchor_array.ndim != 1:
        raise ValueError("anchors must be one-dimensional")
    if profile == "action15":
        offsets = np.arange(horizon_camera_intervals, dtype=np.int64)
        return carriers.action_15hz[anchor_array[:, None] + offsets[None, :]]
    if profile == "action30":
        offsets = np.arange(2 * horizon_camera_intervals, dtype=np.int64)
        return carriers.action_30hz[2 * anchor_array[:, None] + offsets[None, :]]
    raise ValueError(f"unknown action profile: {profile}")


def model_relative_action_chunks(
    carriers: DualRateCarriers,
    anchors: IntArray,
    *,
    profile: str,
    horizon_camera_intervals: int = 50,
) -> FloatArray:
    """Build the model-visible relative ``[anchor, K, 7]`` chunks."""

    anchor_array = np.asarray(anchors, dtype=np.int64)
    absolute = absolute_action_chunks(
        carriers,
        anchor_array,
        profile=profile,
        horizon_camera_intervals=horizon_camera_intervals,
    )
    anchor_pose = carriers.observation_pose_xyzw[anchor_array]
    anchor_rotation = quaternion_xyzw_to_matrix(anchor_pose[:, 3:7])
    target_rotation = quaternion_xyzw_to_matrix(absolute[..., 3:7])
    return encode_relative_action(
        anchor_pose[:, :3],
        anchor_rotation,
        absolute[..., :3],
        target_rotation,
        absolute[..., 7:8],
    )


def assert_dual_rate_crosscheck(carriers: DualRateCarriers, *, atol: float = 0.0) -> None:
    """Assert that every 30 Hz endpoint exactly matches the 15 Hz carrier."""

    if not np.allclose(carriers.action_30hz[1::2], carriers.action_15hz, atol=atol, rtol=0.0):
        raise AssertionError("30 Hz endpoints do not match 15 Hz carriers")
    if not np.array_equal(carriers.action_30hz_target_time_ns[1::2], carriers.action_15hz_target_time_ns):
        raise AssertionError("30 Hz endpoint target timestamps do not match 15 Hz")
    if not np.array_equal(
        carriers.action_30hz_eef_source_time_ns[1::2], carriers.action_15hz_eef_source_time_ns
    ):
        raise AssertionError("30 Hz endpoint EEF source timestamps do not match 15 Hz")
    if not np.array_equal(
        carriers.action_30hz_gripper_source_time_ns[1::2],
        carriers.action_15hz_gripper_source_time_ns,
    ):
        raise AssertionError("30 Hz endpoint gripper source timestamps do not match 15 Hz")
