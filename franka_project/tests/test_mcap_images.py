from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from franka_eef_pipeline.action_chunk import AlignedEpisode
from franka_eef_pipeline import mcap_images
from franka_eef_pipeline.mcap_images import (
    ImageDecodeError,
    ImagePairIntegrityError,
    decode_sensor_image_rgb,
    iter_aligned_policy_image_pairs,
)
from franka_eef_pipeline.mcap_reader import TOPIC_CAM1, TOPIC_CAM2


def _image_message(
    data: bytes,
    *,
    height: int = 1,
    width: int = 1,
    encoding: str = "rgb8",
    step: int | None = None,
) -> SimpleNamespace:
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(
        encoding, 1
    )
    return SimpleNamespace(
        height=height,
        width=width,
        encoding=encoding,
        is_bigendian=0,
        step=width * channels if step is None else step,
        data=data,
    )


@pytest.mark.parametrize(
    ("encoding", "data", "expected"),
    [
        ("rgb8", bytes([1, 2, 3]), [1, 2, 3]),
        ("bgr8", bytes([3, 2, 1]), [1, 2, 3]),
        ("rgba8", bytes([1, 2, 3, 99]), [1, 2, 3]),
        ("bgra8", bytes([3, 2, 1, 99]), [1, 2, 3]),
        ("mono8", bytes([7]), [7, 7, 7]),
    ],
)
def test_decode_supported_encodings_to_contiguous_rgb(
    encoding: str, data: bytes, expected: list[int]
) -> None:
    image = decode_sensor_image_rgb(_image_message(data, encoding=encoding))
    assert image.shape == (1, 1, 3)
    assert image.dtype == np.uint8
    assert image.flags.c_contiguous
    np.testing.assert_array_equal(image[0, 0], expected)


def test_decode_honors_step_and_ignores_row_padding() -> None:
    message = _image_message(
        bytes(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                250,
                251,
                7,
                8,
                9,
                10,
                11,
                12,
                252,
                253,
            ]
        ),
        height=2,
        width=2,
        encoding="rgb8",
        step=8,
    )
    image = decode_sensor_image_rgb(message)
    np.testing.assert_array_equal(
        image,
        np.asarray(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        ),
    )


@pytest.mark.parametrize(
    "message",
    [
        _image_message(bytes([1, 2, 3]), encoding="yuv422"),
        _image_message(bytes([1, 2]), encoding="rgb8", step=2),
        _image_message(bytes([1, 2]), encoding="rgb8", step=3),
        _image_message(b"", height=0, encoding="rgb8", step=0),
    ],
)
def test_decode_rejects_unsupported_or_malformed_messages(message: SimpleNamespace) -> None:
    with pytest.raises(ImageDecodeError):
        decode_sensor_image_rgb(message)


def _aligned_episode() -> AlignedEpisode:
    camera_time = np.asarray([100, 200, 300, 400], dtype=np.int64)
    return AlignedEpisode(
        episode_id="synthetic_images",
        camera_log_time_ns=camera_time,
        cam1_raw_index=np.asarray([1, 2, 3, 4], dtype=np.int64),
        cam2_raw_index=np.asarray([0, 1, 1, 2], dtype=np.int64),
        eef_raw_index=np.zeros(4, dtype=np.int64),
        gripper_raw_index=np.zeros(4, dtype=np.int64),
        qpos_raw_index=np.zeros(4, dtype=np.int64),
        cam2_age_ns=np.asarray([20, 20, 120, 20], dtype=np.int64),
        eef_age_ns=np.zeros(4, dtype=np.int64),
        gripper_age_ns=np.zeros(4, dtype=np.int64),
        qpos_age_ns=np.zeros(4, dtype=np.int64),
        observation_valid=np.ones(4, dtype=np.bool_),
        eef_pose_xyzw=np.zeros((4, 7), dtype=np.float64),
        gripper_raw=np.zeros((4, 1), dtype=np.float64),
        qpos=np.zeros((4, 7), dtype=np.float64),
        common_start_ns=100,
        common_end_ns=400,
    )


def _record(topic: str, log_time_ns: int, message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        channel=SimpleNamespace(topic=topic),
        log_time_ns=log_time_ns,
        ros_msg=message,
    )


def _rgb(value: int) -> SimpleNamespace:
    return _image_message(bytes([value, value + 1, value + 2]), encoding="rgb8")


def _ordered_records() -> list[SimpleNamespace]:
    return [
        # Unselected cam1 raw index 0 deliberately uses an unsupported encoding;
        # the iterator must not decode it.
        _record(TOPIC_CAM1, 50, _image_message(bytes([0]), encoding="unsupported")),
        _record(TOPIC_CAM2, 80, _rgb(10)),
        _record(TOPIC_CAM1, 100, _rgb(20)),
        _record(TOPIC_CAM2, 180, _rgb(30)),
        _record(TOPIC_CAM1, 200, _rgb(40)),
        _record(TOPIC_CAM1, 300, _rgb(50)),
    ]


def test_streams_m_minus_one_pairs_in_order_and_reuses_cam2(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    mcap_path = tmp_path / "episode.mcap"
    mcap_path.touch()
    monkeypatch.setattr(mcap_images, "read_ros2_messages", lambda *_args, **_kwargs: iter(_ordered_records()))

    pairs = list(iter_aligned_policy_image_pairs(mcap_path, _aligned_episode()))

    assert [pair.row_index for pair in pairs] == [0, 1, 2]
    assert [pair.camera_log_time_ns for pair in pairs] == [100, 200, 300]
    assert [pair.cam1_raw_index for pair in pairs] == [1, 2, 3]
    assert [pair.cam2_raw_index for pair in pairs] == [0, 1, 1]
    assert [pair.cam2_log_time_ns for pair in pairs] == [80, 180, 180]
    np.testing.assert_array_equal(pairs[0].cam1[0, 0], [20, 21, 22])
    np.testing.assert_array_equal(pairs[0].cam2[0, 0], [10, 11, 12])
    assert pairs[1].cam2 is pairs[2].cam2


def test_stream_rejects_missing_selected_message(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    mcap_path = tmp_path / "episode.mcap"
    mcap_path.touch()
    records = _ordered_records()[:-1]
    monkeypatch.setattr(mcap_images, "read_ros2_messages", lambda *_args, **_kwargs: iter(records))

    with pytest.raises(ImagePairIntegrityError, match="ended before policy row 2/3"):
        list(iter_aligned_policy_image_pairs(mcap_path, _aligned_episode()))


def test_stream_fails_early_when_it_passes_missing_causal_cam2(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    mcap_path = tmp_path / "episode.mcap"
    mcap_path.touch()
    records = _ordered_records()
    # cam2 raw index 0 is an unselected prefix at t=70. The aligned mapping
    # claims selected raw index 1 should exist at t=80, but the next record is
    # cam1 at t=100, so the ordered stream has irreversibly passed it.
    records[1] = _record(TOPIC_CAM2, 70, _rgb(5))
    aligned = replace(
        _aligned_episode(),
        cam2_raw_index=np.asarray([1, 2, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(mcap_images, "read_ros2_messages", lambda *_args, **_kwargs: iter(records))

    with pytest.raises(ImagePairIntegrityError, match="stream passed cam2 raw index 1"):
        list(iter_aligned_policy_image_pairs(mcap_path, aligned))


def test_stream_rejects_timestamp_disagreement(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    mcap_path = tmp_path / "episode.mcap"
    mcap_path.touch()
    records = _ordered_records()
    records[3] = _record(TOPIC_CAM2, 181, _rgb(30))
    monkeypatch.setattr(mcap_images, "read_ros2_messages", lambda *_args, **_kwargs: iter(records))

    with pytest.raises(ImagePairIntegrityError, match="181 != aligned 180"):
        list(iter_aligned_policy_image_pairs(mcap_path, _aligned_episode()))


def test_stream_rejects_non_monotonic_combined_log_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    mcap_path = tmp_path / "episode.mcap"
    mcap_path.touch()
    unsupported = _image_message(bytes([0]), encoding="unsupported")
    records = [
        _record(TOPIC_CAM1, 10, unsupported),
        _record(TOPIC_CAM2, 20, unsupported),
        _record(TOPIC_CAM1, 40, unsupported),
        _record(TOPIC_CAM2, 30, unsupported),
        *_ordered_records(),
    ]
    aligned = replace(
        _aligned_episode(),
        cam1_raw_index=np.asarray([3, 4, 5, 6], dtype=np.int64),
        cam2_raw_index=np.asarray([2, 3, 3, 4], dtype=np.int64),
    )
    monkeypatch.setattr(mcap_images, "read_ros2_messages", lambda *_args, **_kwargs: iter(records))

    with pytest.raises(ImagePairIntegrityError, match="not log-time ordered"):
        list(iter_aligned_policy_image_pairs(mcap_path, aligned))


def test_stream_rejects_conflicting_shapes(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    mcap_path = tmp_path / "episode.mcap"
    mcap_path.touch()
    records = _ordered_records()
    records[1] = _record(
        TOPIC_CAM2,
        80,
        _image_message(bytes([1, 2, 3, 4, 5, 6]), width=2, encoding="rgb8"),
    )
    monkeypatch.setattr(mcap_images, "read_ros2_messages", lambda *_args, **_kwargs: iter(records))

    with pytest.raises(ImagePairIntegrityError, match="camera shapes differ"):
        list(iter_aligned_policy_image_pairs(mcap_path, _aligned_episode()))
