"""Low-memory MCAP image decoding for aligned Franka policy rows.

The numeric/timestamp pass in :mod:`franka_eef_pipeline.mcap_reader` decides
which raw camera messages belong to every cam1 anchor.  This module performs a
second, image-only pass over the MCAP and decodes only those selected messages.
It never substitutes a zero image.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mcap_ros2.reader import read_ros2_messages
from numpy.typing import NDArray

from .action_chunk import AlignedEpisode
from .mcap_reader import TOPIC_CAM1, TOPIC_CAM2


UInt8Image = NDArray[np.uint8]


class ImageDecodeError(ValueError):
    """A ROS image message is malformed or uses an unsupported encoding."""


class ImagePairIntegrityError(RuntimeError):
    """The MCAP image stream does not satisfy the frozen alignment mapping."""


@dataclass(frozen=True)
class AlignedImagePair:
    """One pair of RGB images for a 15 Hz LeRobot policy row."""

    row_index: int
    camera_log_time_ns: int
    cam1_raw_index: int
    cam2_raw_index: int
    cam1_log_time_ns: int
    cam2_log_time_ns: int
    cam1: UInt8Image
    cam2: UInt8Image


def _uint8_buffer(data: Any) -> NDArray[np.uint8]:
    try:
        return np.frombuffer(data, dtype=np.uint8)
    except (TypeError, ValueError):
        array = np.asarray(data)
        if array.dtype != np.uint8 or array.ndim != 1:
            raise ImageDecodeError(
                "Image.data must be a one-dimensional uint8 byte buffer"
            ) from None
        return array


def decode_sensor_image_rgb(message: Any) -> UInt8Image:
    """Decode a ``sensor_msgs/msg/Image`` into contiguous HWC uint8 RGB.

    Row padding is handled using ``message.step``. Supported encodings are
    ``rgb8``, ``bgr8``, ``rgba8``, ``bgra8`` and ``mono8``. The 8-bit formats
    make ``is_bigendian`` irrelevant.
    """

    try:
        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        encoding = str(message.encoding).strip().lower()
        data = message.data
    except (AttributeError, TypeError, ValueError) as error:
        raise ImageDecodeError("Image message is missing valid height/width/step/encoding/data") from error

    if height <= 0 or width <= 0:
        raise ImageDecodeError(f"Image dimensions must be positive, got {height}x{width}")
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    if encoding not in channels_by_encoding:
        raise ImageDecodeError(
            f"Unsupported image encoding {encoding!r}; expected one of {sorted(channels_by_encoding)}"
        )
    channels = channels_by_encoding[encoding]
    active_row_bytes = width * channels
    if step < active_row_bytes:
        raise ImageDecodeError(
            f"Image.step={step} is smaller than active row size {active_row_bytes} for {encoding}"
        )

    buffer = _uint8_buffer(data)
    expected_bytes = height * step
    if buffer.size != expected_bytes:
        raise ImageDecodeError(
            f"Image.data has {buffer.size} bytes, expected exactly height*step={expected_bytes}"
        )
    active = buffer.reshape(height, step)[:, :active_row_bytes]
    pixels = active.reshape(height, width, channels)

    if encoding == "rgb8":
        rgb = pixels
    elif encoding == "bgr8":
        rgb = pixels[..., ::-1]
    elif encoding == "rgba8":
        rgb = pixels[..., :3]
    elif encoding == "bgra8":
        rgb = pixels[..., [2, 1, 0]]
    else:  # mono8
        rgb = np.repeat(pixels, 3, axis=-1)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _validate_alignment(aligned: AlignedEpisode) -> tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    count = aligned.num_camera_anchors
    if count < 2:
        raise ImagePairIntegrityError(
            f"{aligned.episode_id}: at least two aligned camera anchors are required"
        )
    for name, value in (
        ("camera_log_time_ns", aligned.camera_log_time_ns),
        ("cam1_raw_index", aligned.cam1_raw_index),
        ("cam2_raw_index", aligned.cam2_raw_index),
        ("cam2_age_ns", aligned.cam2_age_ns),
    ):
        if len(value) != count:
            raise ImagePairIntegrityError(
                f"{aligned.episode_id}: {name} length {len(value)} does not match {count} anchors"
            )

    camera_time = np.asarray(aligned.camera_log_time_ns[:-1], dtype=np.int64)
    cam1_index = np.asarray(aligned.cam1_raw_index[:-1], dtype=np.int64)
    cam2_index = np.asarray(aligned.cam2_raw_index[:-1], dtype=np.int64)
    cam2_age = np.asarray(aligned.cam2_age_ns[:-1], dtype=np.int64)
    if np.any(np.diff(camera_time) <= 0):
        raise ImagePairIntegrityError(f"{aligned.episode_id}: camera anchor times are not increasing")
    if np.any(cam1_index < 0) or np.any(np.diff(cam1_index) <= 0):
        raise ImagePairIntegrityError(
            f"{aligned.episode_id}: selected cam1 raw indices must be non-negative and strictly increasing"
        )
    if np.any(cam2_index < 0) or np.any(np.diff(cam2_index) < 0):
        raise ImagePairIntegrityError(
            f"{aligned.episode_id}: selected cam2 raw indices must be non-negative and non-decreasing"
        )
    if np.any(cam2_age < 0):
        raise ImagePairIntegrityError(f"{aligned.episode_id}: cam2 selection is non-causal")
    cam2_source_time = camera_time - cam2_age
    if np.any(np.diff(cam2_source_time) < 0):
        raise ImagePairIntegrityError(
            f"{aligned.episode_id}: selected cam2 source times are not non-decreasing"
        )
    return camera_time, cam1_index, cam2_index, cam2_source_time


def iter_aligned_policy_image_pairs(
    mcap_path: str | Path,
    aligned: AlignedEpisode,
    *,
    require_matching_shape: bool = True,
) -> Iterator[AlignedImagePair]:
    """Yield exactly ``M-1`` selected cam1/cam2 pairs in policy-row order.

    Only selected raw messages are decoded. A selected cam2 message may serve
    multiple cam1 rows; it is retained until its final consumer and then
    released. Combined MCAP image messages must be in non-decreasing log-time
    order, which bounds the live decoded-image cache under the causal alignment
    contract.

    The caller must consume the iterator to exhaustion so end-of-stream
    completeness checks run.
    """

    path = Path(mcap_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    camera_time, cam1_indices, cam2_indices, cam2_source_times = _validate_alignment(aligned)
    row_count = len(camera_time)

    cam1_row_by_raw = {int(raw_index): row for row, raw_index in enumerate(cam1_indices)}
    if len(cam1_row_by_raw) != row_count:
        raise ImagePairIntegrityError(f"{aligned.episode_id}: duplicate selected cam1 raw index")

    cam2_expected_time_by_raw: dict[int, int] = {}
    for raw_index, source_time in zip(cam2_indices, cam2_source_times, strict=True):
        raw_key = int(raw_index)
        source_value = int(source_time)
        previous = cam2_expected_time_by_raw.setdefault(raw_key, source_value)
        if previous != source_value:
            raise ImagePairIntegrityError(
                f"{aligned.episode_id}: reused cam2 raw index {raw_key} has conflicting source times"
            )

    selected_cam1 = set(cam1_row_by_raw)
    selected_cam2 = set(cam2_expected_time_by_raw)
    remaining_cam2_uses = Counter(int(value) for value in cam2_indices)
    cam1_cache: dict[int, tuple[int, UInt8Image]] = {}
    cam2_cache: dict[int, tuple[int, UInt8Image]] = {}
    topic_raw_count = {TOPIC_CAM1: 0, TOPIC_CAM2: 0}
    next_row = 0
    previous_log_time: int | None = None
    records = read_ros2_messages(path, topics=[TOPIC_CAM1, TOPIC_CAM2])

    try:
        for record in records:
            topic = record.channel.topic
            if topic not in topic_raw_count:
                raise ImagePairIntegrityError(
                    f"{aligned.episode_id}: image-only reader returned unexpected topic {topic}"
                )
            log_time = int(record.log_time_ns)
            if previous_log_time is not None and log_time < previous_log_time:
                raise ImagePairIntegrityError(
                    f"{aligned.episode_id}: combined camera MCAP records are not log-time ordered"
                )
            previous_log_time = log_time
            raw_index = topic_raw_count[topic]
            topic_raw_count[topic] += 1

            needed = raw_index in (selected_cam1 if topic == TOPIC_CAM1 else selected_cam2)
            if needed:
                try:
                    image = decode_sensor_image_rgb(record.ros_msg)
                except ImageDecodeError as error:
                    raise ImageDecodeError(
                        f"{aligned.episode_id}: {topic} raw index {raw_index}: {error}"
                    ) from error
                if topic == TOPIC_CAM1:
                    expected_time = int(camera_time[cam1_row_by_raw[raw_index]])
                    if log_time != expected_time:
                        raise ImagePairIntegrityError(
                            f"{aligned.episode_id}: cam1 raw index {raw_index} log time "
                            f"{log_time} != aligned {expected_time}"
                        )
                    cam1_cache[raw_index] = (log_time, image)
                else:
                    expected_time = cam2_expected_time_by_raw[raw_index]
                    if log_time != expected_time:
                        raise ImagePairIntegrityError(
                            f"{aligned.episode_id}: cam2 raw index {raw_index} log time "
                            f"{log_time} != aligned {expected_time}"
                        )
                    cam2_cache[raw_index] = (log_time, image)

            while next_row < row_count:
                cam1_raw = int(cam1_indices[next_row])
                cam2_raw = int(cam2_indices[next_row])
                if cam1_raw not in cam1_cache or cam2_raw not in cam2_cache:
                    break
                cam1_log_time, cam1_image = cam1_cache.pop(cam1_raw)
                cam2_log_time, cam2_image = cam2_cache[cam2_raw]
                if require_matching_shape and cam1_image.shape != cam2_image.shape:
                    raise ImagePairIntegrityError(
                        f"{aligned.episode_id}: row {next_row} camera shapes differ: "
                        f"{cam1_image.shape} vs {cam2_image.shape}"
                    )
                remaining_cam2_uses[cam2_raw] -= 1
                if remaining_cam2_uses[cam2_raw] == 0:
                    del remaining_cam2_uses[cam2_raw]
                    del cam2_cache[cam2_raw]
                pair = AlignedImagePair(
                    row_index=next_row,
                    camera_log_time_ns=int(camera_time[next_row]),
                    cam1_raw_index=cam1_raw,
                    cam2_raw_index=cam2_raw,
                    cam1_log_time_ns=cam1_log_time,
                    cam2_log_time_ns=cam2_log_time,
                    cam1=cam1_image,
                    cam2=cam2_image,
                )
                next_row += 1
                yield pair

            if next_row == row_count:
                break
            # With log-time ordered records, a message cannot arrive after the
            # stream has advanced beyond its frozen aligned timestamp. Failing
            # here also prevents malformed MCAPs from accumulating images.
            next_cam1_raw = int(cam1_indices[next_row])
            next_cam2_raw = int(cam2_indices[next_row])
            if next_cam1_raw not in cam1_cache and log_time > int(camera_time[next_row]):
                raise ImagePairIntegrityError(
                    f"{aligned.episode_id}: stream passed cam1 raw index {next_cam1_raw} "
                    f"expected log time {int(camera_time[next_row])}"
                )
            if next_cam2_raw not in cam2_cache and log_time > int(cam2_source_times[next_row]):
                raise ImagePairIntegrityError(
                    f"{aligned.episode_id}: stream passed cam2 raw index {next_cam2_raw} "
                    f"expected log time {int(cam2_source_times[next_row])}"
                )
    finally:
        close = getattr(records, "close", None)
        if callable(close):
            close()

    if next_row != row_count:
        missing_row = next_row
        raise ImagePairIntegrityError(
            f"{aligned.episode_id}: MCAP ended before policy row {missing_row}/{row_count}; "
            f"needed cam1 raw {int(cam1_indices[missing_row])}, cam2 raw {int(cam2_indices[missing_row])}; "
            f"observed topic counts {topic_raw_count}"
        )
    if cam1_cache or cam2_cache or remaining_cam2_uses:
        raise ImagePairIntegrityError(
            f"{aligned.episode_id}: decoded image caches were not fully consumed"
        )
