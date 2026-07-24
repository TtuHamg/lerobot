"""Read the Frank3 MCAP topics needed by the project without decoding images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from mcap_ros2.reader import read_ros2_messages
from numpy.typing import NDArray


TOPIC_JOINT = "/franka/joint_states"
TOPIC_GRIPPER = "/gripper/joint_states"
TOPIC_EEF = "/franka_robot_state_broadcaster/current_pose"
TOPIC_CAM1 = "/camera1/camera1/color/image_raw"
TOPIC_CAM2 = "/camera2/camera2/color/image_raw"

REQUIRED_TOPICS = (TOPIC_CAM1, TOPIC_CAM2, TOPIC_EEF, TOPIC_GRIPPER)
AUDIT_TOPICS = (*REQUIRED_TOPICS, TOPIC_JOINT)


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def _header_stamp_ns(message: Any) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is None:
        return -1
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _header_frame_id(message: Any) -> str:
    header = getattr(message, "header", None)
    return str(getattr(header, "frame_id", "")) if header is not None else ""


@dataclass(frozen=True)
class TimestampStream:
    log_time_ns: IntArray
    publish_time_ns: IntArray
    header_time_ns: IntArray
    frame_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.log_time_ns)


@dataclass(frozen=True)
class ImageStream(TimestampStream):
    height: NDArray[np.int64]
    width: NDArray[np.int64]
    encoding: tuple[str, ...]


@dataclass(frozen=True)
class NumericStream(TimestampStream):
    values: FloatArray
    names: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeSignals:
    episode_id: str
    mcap_path: Path
    cam1: ImageStream
    cam2: ImageStream
    eef_pose_xyzw: NumericStream
    gripper: NumericStream
    qpos: NumericStream


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
        raise ValueError(f"invalid manifest: {path}")
    if int(manifest.get("n_episodes", len(manifest["episodes"]))) != len(manifest["episodes"]):
        raise ValueError("manifest n_episodes does not match episode list")
    return manifest


def resolve_mcap_path(episode: dict[str, Any], *, data_root: str | Path) -> Path:
    episode_path = Path(episode.get("path") or Path(data_root) / episode["relative_path"])
    files = episode.get("mcap_files") or [item.name for item in episode_path.glob("*.mcap")]
    if len(files) != 1:
        raise ValueError(f"expected exactly one MCAP for {episode.get('episode_id')}, got {files}")
    path = episode_path / files[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _timestamp_stream(records: list[tuple[int, int, int, str]]) -> TimestampStream:
    return TimestampStream(
        log_time_ns=np.asarray([item[0] for item in records], dtype=np.int64),
        publish_time_ns=np.asarray([item[1] for item in records], dtype=np.int64),
        header_time_ns=np.asarray([item[2] for item in records], dtype=np.int64),
        frame_ids=tuple(item[3] for item in records),
    )


def _image_stream(records: list[tuple[int, int, int, str, int, int, str]]) -> ImageStream:
    base = _timestamp_stream([item[:4] for item in records])
    return ImageStream(
        **base.__dict__,
        height=np.asarray([item[4] for item in records], dtype=np.int64),
        width=np.asarray([item[5] for item in records], dtype=np.int64),
        encoding=tuple(item[6] for item in records),
    )


def _numeric_stream(
    records: list[tuple[int, int, int, str, NDArray[np.float64], tuple[str, ...]]],
    *,
    width: int,
) -> NumericStream:
    base = _timestamp_stream([item[:4] for item in records])
    values = (
        np.stack([item[4] for item in records]).astype(np.float64, copy=False)
        if records
        else np.empty((0, width), dtype=np.float64)
    )
    names = records[0][5] if records else ()
    if any(item[5] != names for item in records):
        raise ValueError("joint names/order changes within an episode")
    return NumericStream(**base.__dict__, values=values, names=names)


def _validate_stream_order(stream: TimestampStream, *, topic: str) -> None:
    if len(stream) == 0:
        raise ValueError(f"required topic is empty: {topic}")
    if np.any(np.diff(stream.log_time_ns) <= 0):
        raise ValueError(f"log timestamps are not strictly increasing: {topic}")


def load_episode_signals(mcap_path: str | Path, *, episode_id: str | None = None) -> EpisodeSignals:
    """Load timestamps and numeric signals while discarding image pixel buffers."""

    path = Path(mcap_path)
    images: dict[str, list[tuple[int, int, int, str, int, int, str]]] = {
        TOPIC_CAM1: [],
        TOPIC_CAM2: [],
    }
    eef_records: list[tuple[int, int, int, str, NDArray[np.float64], tuple[str, ...]]] = []
    gripper_records: list[tuple[int, int, int, str, NDArray[np.float64], tuple[str, ...]]] = []
    joint_records: list[tuple[int, int, int, str, NDArray[np.float64], tuple[str, ...]]] = []

    for record in read_ros2_messages(path, topics=list(AUDIT_TOPICS)):
        topic = record.channel.topic
        message = record.ros_msg
        common = (
            int(record.log_time_ns),
            int(record.publish_time_ns),
            _header_stamp_ns(message),
            _header_frame_id(message),
        )
        if topic in images:
            images[topic].append(
                (*common, int(message.height), int(message.width), str(message.encoding).lower())
            )
        elif topic == TOPIC_EEF:
            pose = message.pose
            value = np.asarray(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=np.float64,
            )
            eef_records.append((*common, value, ("x", "y", "z", "qx", "qy", "qz", "qw")))
        elif topic == TOPIC_GRIPPER:
            if len(message.position) < 1:
                raise ValueError("gripper message has no position")
            value = np.asarray(message.position[:1], dtype=np.float64)
            gripper_records.append((*common, value, tuple(str(name) for name in message.name[:1])))
        elif topic == TOPIC_JOINT:
            if len(message.position) < 7:
                raise ValueError("Franka joint message has fewer than seven positions")
            value = np.asarray(message.position[:7], dtype=np.float64)
            joint_records.append((*common, value, tuple(str(name) for name in message.name[:7])))

    result = EpisodeSignals(
        episode_id=episode_id or path.parent.name,
        mcap_path=path,
        cam1=_image_stream(images[TOPIC_CAM1]),
        cam2=_image_stream(images[TOPIC_CAM2]),
        eef_pose_xyzw=_numeric_stream(eef_records, width=7),
        gripper=_numeric_stream(gripper_records, width=1),
        qpos=_numeric_stream(joint_records, width=7),
    )
    for topic, stream in (
        (TOPIC_CAM1, result.cam1),
        (TOPIC_CAM2, result.cam2),
        (TOPIC_EEF, result.eef_pose_xyzw),
        (TOPIC_GRIPPER, result.gripper),
        (TOPIC_JOINT, result.qpos),
    ):
        _validate_stream_order(stream, topic=topic)
    return result


def latest_not_after_indices(source_time_ns: IntArray, target_time_ns: IntArray) -> IntArray:
    """Return the source index of the latest timestamp not after each target."""

    source = np.asarray(source_time_ns, dtype=np.int64)
    target = np.asarray(target_time_ns, dtype=np.int64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    return np.searchsorted(source, target, side="right").astype(np.int64) - 1


def stream_rate_summary(stream: TimestampStream) -> dict[str, float | int | None]:
    timestamps = stream.log_time_ns
    if len(timestamps) < 2:
        return {
            "count": len(timestamps),
            "span_s": 0.0,
            "mean_hz": None,
            "dt_median_ms": None,
            "dt_max_ms": None,
        }
    delta = np.diff(timestamps).astype(np.float64) / 1e9
    span = float((timestamps[-1] - timestamps[0]) / 1e9)
    return {
        "count": len(timestamps),
        "span_s": span,
        "mean_hz": float((len(timestamps) - 1) / span),
        "dt_median_ms": float(np.median(delta) * 1e3),
        "dt_max_ms": float(np.max(delta) * 1e3),
    }
