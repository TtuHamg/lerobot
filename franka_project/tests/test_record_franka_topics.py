from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "record_franka_topics.py"
SPEC = importlib.util.spec_from_file_location("record_franka_topics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


def test_resolve_topics_accepts_aliases_absolute_names_commas_and_deduplicates():
    assert recorder.resolve_topics(
        ["camera1,camera2", "eef", "/custom/value", "camera1"]
    ) == [
        "/camera1/camera1/color/image_raw",
        "/camera2/camera2/color/image_raw",
        "/franka_robot_state_broadcaster/current_pose",
        "/custom/value",
    ]


def test_resolve_topics_rejects_unknown_non_absolute_name():
    with pytest.raises(ValueError, match="Unknown topic alias"):
        recorder.resolve_topics(["not_an_alias"])


@pytest.mark.parametrize(
    ("topics", "expected"),
    [
        ([recorder.TOPIC_ALIASES["camera1"]], "mp4"),
        ([recorder.TOPIC_ALIASES["camera1"], recorder.TOPIC_ALIASES["camera2"]], "mp4"),
        ([recorder.TOPIC_ALIASES["camera1"], recorder.TOPIC_ALIASES["eef"]], "mcap"),
        ([recorder.TOPIC_ALIASES["qpos"]], "mcap"),
    ],
)
def test_auto_format_only_uses_mp4_for_known_camera_only_selection(topics, expected):
    assert recorder.select_output_format("auto", topics) == expected


def test_explicit_format_overrides_auto_selection():
    assert recorder.select_output_format("mcap", [recorder.TOPIC_ALIASES["camera1"]]) == "mcap"
    assert recorder.select_output_format("mp4", ["/custom/image"]) == "mp4"


def test_build_mcap_command_uses_mcap_topics_and_safe_argument_list(tmp_path):
    output = tmp_path / "session"
    qos_path = tmp_path / "qos.yaml"
    command = recorder.build_mcap_command(
        ["/camera/image_raw", "/franka/joint_states"],
        output,
        storage_profile="zstd_fast",
        qos_overrides_path=qos_path,
    )
    assert command[:4] == ["ros2", "bag", "record", "--storage"]
    assert command[4] == "mcap"
    assert command[-3:] == ["--topics", "/camera/image_raw", "/franka/joint_states"]
    assert command[command.index("--output") + 1] == str(output)
    assert command[command.index("--storage-preset-profile") + 1] == "zstd_fast"
    assert command[command.index("--qos-profile-overrides-path") + 1] == str(qos_path)


def test_default_output_and_video_paths_are_deterministic(tmp_path):
    timestamp = datetime(2026, 7, 29, 12, 34, 56, 123000)
    assert recorder.default_output_path("mcap", now=timestamp).name == "20260729_123456_123_mcap"
    paths = recorder.video_output_paths(
        [recorder.TOPIC_ALIASES["camera1"], recorder.TOPIC_ALIASES["camera2"]],
        tmp_path,
    )
    assert paths[recorder.TOPIC_ALIASES["camera1"]] == tmp_path / "camera1.mp4"
    assert paths[recorder.TOPIC_ALIASES["camera2"]] == tmp_path / "camera2.mp4"


def test_single_mp4_path_is_rejected_for_multiple_topics(tmp_path):
    with pytest.raises(ValueError, match="only be used with one"):
        recorder.video_output_paths(
            [recorder.TOPIC_ALIASES["camera1"], recorder.TOPIC_ALIASES["camera2"]],
            tmp_path / "both.mp4",
        )


def test_custom_video_topic_filename_collisions_get_distinct_stable_paths(tmp_path):
    paths = recorder.video_output_paths(["/a/b", "/a_b"], tmp_path)
    assert paths["/a/b"] == tmp_path / "a_b.mp4"
    assert paths["/a_b"].name.startswith("a_b_")
    assert paths["/a_b"].suffix == ".mp4"
    assert len(set(paths.values())) == 2


def test_estimate_video_fps_uses_median_ros_timestamp_delta():
    period_ns = round(1_000_000_000 / 15)
    timestamps = [1_000_000_000 + index * period_ns for index in range(12)]
    assert recorder.estimate_video_fps(timestamps) == pytest.approx(15.0, rel=1e-6)


def test_estimate_video_fps_falls_back_for_missing_or_implausible_timestamps():
    assert recorder.estimate_video_fps([0]) == recorder.DEFAULT_VIDEO_FPS_FALLBACK
    assert recorder.estimate_video_fps([0, 1]) == recorder.DEFAULT_VIDEO_FPS_FALLBACK


def test_image_decoder_handles_rgb8_row_padding_and_converts_to_bgr():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    message = SimpleNamespace(
        encoding="rgb8",
        width=2,
        height=2,
        step=8,
        data=bytes(
            [
                255,
                0,
                0,
                0,
                255,
                0,
                99,
                99,
                0,
                0,
                255,
                255,
                255,
                255,
                88,
                88,
            ]
        ),
    )
    frame = recorder.image_message_to_bgr(message, numpy_module=np, cv2_module=cv2)
    assert frame.shape == (2, 2, 3)
    assert frame.tolist() == [
        [[0, 0, 255], [0, 255, 0]],
        [[255, 0, 0], [255, 255, 255]],
    ]


def test_image_decoder_rejects_unsupported_encoding():
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    message = SimpleNamespace(encoding="16UC1", width=1, height=1, step=2, data=b"\x00\x00")
    with pytest.raises(ValueError, match="Unsupported image encoding"):
        recorder.image_message_to_bgr(message, numpy_module=np, cv2_module=cv2)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_positive_cli_values_must_be_finite(value):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        recorder._validate_positive_optional(value, "test value")
