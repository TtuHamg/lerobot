# Copyright 2026 pnp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest

from franka_lerobot_rviz.visualization_model import (
    EXECUTION_COLOR,
    PENDING_COLOR,
    PREFLIGHT_COLOR,
    REJECTED_COLOR,
    SHADOW_COLOR,
    abbreviated_session,
    classify_ack,
    gripper_color,
    plan_key,
    scheduled_pose_nanoseconds,
    split_nanoseconds,
    validate_chunk_shape,
)


@pytest.mark.parametrize(
    ("accepted", "result", "expected_label", "expected_color"),
    [
        (None, None, "PENDING ACK", PENDING_COLOR),
        (True, 0, "ACCEPTED SHADOW (NOT EXECUTED)", SHADOW_COLOR),
        (True, 1, "ACCEPTED FOR EXECUTION", EXECUTION_COLOR),
        (True, 2, "ACCEPTED PREFLIGHT ONLY", PREFLIGHT_COLOR),
        (False, 17, "REJECTED (result=17)", REJECTED_COLOR),
        (True, 17, "REJECTED (result=17)", REJECTED_COLOR),
        (False, 1, "REJECTED (result=1)", REJECTED_COLOR),
    ],
)
def test_classify_ack(accepted, result, expected_label, expected_color):
    state = classify_ack(accepted, result)
    assert state.label == expected_label
    assert state.color == expected_color


def test_plan_key_includes_session_id():
    assert plan_key("first-session", 0) != plan_key("second-session", 0)
    assert plan_key("first-session", 7) == ("first-session", 7)


@pytest.mark.parametrize(
    ("timesteps", "poses", "gripper", "valid"),
    [
        ([1], [object()], [0.0], True),
        ([], [], [], False),
        ([1, 2], [object()], [0.0, 1.0], False),
        ([1], [object(), object()], [0.0], False),
    ],
)
def test_validate_chunk_shape(timesteps, poses, gripper, valid):
    actual, detail = validate_chunk_shape(timesteps, poses, gripper)
    assert actual is valid
    assert bool(detail) is (not valid)


def test_scheduled_pose_timestamp_uses_i_plus_one_and_normalizes_nanoseconds():
    # header 3.9 s, period 0.2 s: waypoint 0 is 4.1 s and waypoint 4 is 4.9 s.
    first = scheduled_pose_nanoseconds(3, 900_000_000, 0, 200_000_000, 0)
    fifth = scheduled_pose_nanoseconds(3, 900_000_000, 0, 200_000_000, 4)
    assert split_nanoseconds(first) == (4, 100_000_000)
    assert split_nanoseconds(fifth) == (4, 900_000_000)


def test_negative_pose_index_is_rejected():
    with pytest.raises(ValueError):
        scheduled_pose_nanoseconds(0, 0, 0, 10_000_000, -1)


def test_gripper_color_clamps_and_runs_open_to_closed():
    assert gripper_color(-10.0) == gripper_color(0.0)
    assert gripper_color(10.0) == gripper_color(1.0)
    open_color = gripper_color(0.0)
    closed_color = gripper_color(1.0)
    assert closed_color[0] > open_color[0]
    assert closed_color[1] < open_color[1]
    assert all(math.isfinite(value) for value in closed_color)


def test_abbreviated_session_preserves_both_ends():
    result = abbreviated_session("abcdefghijklmnop", max_length=9)
    assert result.startswith("abcd")
    assert result.endswith("mnop")
    assert "…" in result
