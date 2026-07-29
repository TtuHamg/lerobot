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

"""Pure helpers for classifying and timestamping action-plan visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


PlanKey = tuple[str, int]
Rgba = tuple[float, float, float, float]


PENDING_COLOR: Rgba = (0.20, 0.80, 1.00, 0.65)
EXECUTION_COLOR: Rgba = (0.10, 0.95, 0.20, 0.95)
SHADOW_COLOR: Rgba = (0.15, 0.45, 1.00, 0.75)
PREFLIGHT_COLOR: Rgba = (1.00, 0.65, 0.00, 0.85)
REJECTED_COLOR: Rgba = (1.00, 0.10, 0.10, 0.90)
HISTORY_COLOR: Rgba = (0.45, 0.45, 0.45, 0.35)
HOLD_COLOR: Rgba = (1.00, 0.55, 0.05, 0.95)

# CartesianActionChunkAck constants are duplicated here deliberately so these
# helpers remain importable without a running ROS graph or generated messages.
ACCEPTED_SHADOW = 0
ACCEPTED_FOR_EXECUTION = 1
ACCEPTED_PREFLIGHT_ONLY = 2


@dataclass(frozen=True)
class AckVisualState:
    """Human-readable state and RGBA used for one candidate plan."""

    label: str
    color: Rgba


def plan_key(session_id: str, plan_id: int) -> PlanKey:
    """
    Return the only safe identity for a plan.

    ``plan_id`` is session-local, so using it by itself can associate an ACK
    with a plan from an older client process.
    """
    return (session_id, int(plan_id))


def classify_ack(accepted: bool | None, result: int | None) -> AckVisualState:
    """Map an ACK to an explicit visual state, failing closed on inconsistency."""
    if accepted is None or result is None:
        return AckVisualState("PENDING ACK", PENDING_COLOR)

    if accepted and result == ACCEPTED_FOR_EXECUTION:
        return AckVisualState("ACCEPTED FOR EXECUTION", EXECUTION_COLOR)
    if accepted and result == ACCEPTED_SHADOW:
        return AckVisualState("ACCEPTED SHADOW (NOT EXECUTED)", SHADOW_COLOR)
    if accepted and result == ACCEPTED_PREFLIGHT_ONLY:
        return AckVisualState("ACCEPTED PREFLIGHT ONLY", PREFLIGHT_COLOR)

    # A false accepted flag, a rejection result, or an internally inconsistent
    # ACK is presented as rejected. This avoids accidentally implying motion.
    return AckVisualState(f"REJECTED (result={result})", REJECTED_COLOR)


def validate_chunk_shape(
    timesteps: Sequence[int], poses: Sequence[object], gripper: Sequence[float]
) -> tuple[bool, str]:
    """Validate the cross-field shape contract from CartesianActionChunk.msg."""
    lengths = (len(timesteps), len(poses), len(gripper))
    if 0 in lengths:
        return False, f"chunk arrays must be non-empty; lengths={lengths}"
    if len(set(lengths)) != 1:
        return False, f"chunk arrays must have equal length; lengths={lengths}"
    return True, ""


def duration_to_nanoseconds(sec: int, nanosec: int) -> int:
    """Convert a ROS duration/time pair to nanoseconds."""
    return int(sec) * 1_000_000_000 + int(nanosec)


def split_nanoseconds(total_nanoseconds: int) -> tuple[int, int]:
    """Split non-negative nanoseconds into normalized ROS sec/nanosec fields."""
    if total_nanoseconds < 0:
        raise ValueError("ROS visualization timestamps must be non-negative")
    return divmod(int(total_nanoseconds), 1_000_000_000)


def scheduled_pose_nanoseconds(
    header_sec: int,
    header_nanosec: int,
    period_sec: int,
    period_nanosec: int,
    pose_index: int,
) -> int:
    """
    Return the absolute target time of pose ``i``.

    CartesianActionChunk defines pose i at ``header.stamp + (i + 1) * period``.
    """
    if pose_index < 0:
        raise ValueError("pose_index must be non-negative")
    start = duration_to_nanoseconds(header_sec, header_nanosec)
    period = duration_to_nanoseconds(period_sec, period_nanosec)
    return start + (pose_index + 1) * period


def gripper_color(value: float) -> Rgba:
    """Map normalized open→closed gripper state from cyan to magenta."""
    normalized = max(0.0, min(1.0, float(value)))
    return (0.15 + 0.80 * normalized, 0.85 - 0.65 * normalized, 1.0, 0.90)


def abbreviated_session(session_id: str, max_length: int = 12) -> str:
    """Keep marker labels legible while retaining both ends of a session id."""
    if len(session_id) <= max_length:
        return session_id or "<empty>"
    left = max(1, (max_length - 1) // 2)
    right = max(1, max_length - left - 1)
    return f"{session_id[:left]}…{session_id[-right:]}"
