#!/usr/bin/env python3
"""Hardware-free closed-loop contract and fault-injection probe.

This models only transport validation, session ordering, and watchdog outcomes.
It does not import ROS, publish a topic, call an arm service, or command hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    schema_version: int
    frame_id: str
    session_id: str
    plan_id: int
    source_timestep: int
    stamp_ns: int
    valid_until_ns: int
    period_ns: int
    timesteps: tuple[int, ...]
    poses: tuple[tuple[float, float, float, float, float, float, float], ...]
    gripper: tuple[float, ...]


@dataclass(frozen=True)
class Result:
    stage: str
    scenario: str
    expected: str
    actual: str
    passed: bool
    evidence: dict[str, object]


def make_chunk(count: int, *, now_ns: int, session: str = "offline-session", plan_id: int = 0) -> Chunk:
    if count < 1:
        raise ValueError("count must be positive")
    source = 100
    return Chunk(
        schema_version=1,
        frame_id="base",
        session_id=session,
        plan_id=plan_id,
        source_timestep=source,
        stamp_ns=now_ns + 50_000_000,
        valid_until_ns=now_ns + 500_000_000,
        period_ns=66_666_667,
        timesteps=tuple(range(source, source + count)),
        poses=tuple((0.45 + i * 0.0001, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0) for i in range(count)),
        gripper=tuple(0.5 for _ in range(count)),
    )


def validate_chunk(chunk: Chunk, *, now_ns: int, max_waypoints: int = 50) -> str:
    count = len(chunk.timesteps)
    if chunk.schema_version != 1:
        return "REJECTED_SCHEMA"
    if chunk.frame_id != "base":
        return "REJECTED_FRAME"
    if not chunk.session_id or count < 1 or count > max_waypoints:
        return "REJECTED_SHAPE"
    if len(chunk.poses) != count or len(chunk.gripper) != count:
        return "REJECTED_SHAPE"
    if now_ns > chunk.valid_until_ns:
        return "REJECTED_EXPIRED"
    if chunk.period_ns <= 0 or chunk.stamp_ns + (count - 1) * chunk.period_ns < now_ns:
        return "REJECTED_SCHEDULE"
    if chunk.timesteps[0] < chunk.source_timestep or any(
        right != left + 1 for left, right in zip(chunk.timesteps, chunk.timesteps[1:])
    ):
        return "REJECTED_ORDERING"
    for pose in chunk.poses:
        if len(pose) != 7 or not all(math.isfinite(value) for value in pose):
            return "REJECTED_NONFINITE"
        qnorm = math.sqrt(sum(value * value for value in pose[3:]))
        if abs(qnorm - 1.0) > 1e-3:
            return "REJECTED_QUATERNION"
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in chunk.gripper):
        return "REJECTED_GRIPPER"
    return "ACCEPTED_SHADOW"


class SequenceGuard:
    """Small offline model of active-horizon session/plan replay rejection."""

    def __init__(self) -> None:
        self.session: str | None = None
        self.plan_id: int | None = None
        self.active_until_ns = 0

    def accept(self, chunk: Chunk, *, now_ns: int) -> str:
        if now_ns <= self.active_until_ns and self.session is not None:
            if chunk.session_id != self.session:
                return "REJECTED_ORDERING"
            if self.plan_id is not None and chunk.plan_id <= self.plan_id:
                return "REJECTED_ORDERING"
        self.session = chunk.session_id
        self.plan_id = chunk.plan_id
        self.active_until_ns = chunk.stamp_ns + (len(chunk.timesteps) - 1) * chunk.period_ns
        return "ACCEPTED_SHADOW"


def watchdog_state(*, last_feedback_ns: int, now_ns: int, timeout_ns: int) -> str:
    return "HOLD" if now_ns - last_feedback_ns > timeout_ns else "ARMED"


def _result(stage: str, scenario: str, expected: str, actual: str, **evidence: object) -> Result:
    return Result(stage, scenario, expected, actual, expected == actual, evidence)


def run_suite(now_ns: int | None = None) -> list[Result]:
    now = time.time_ns() if now_ns is None else now_ns
    results: list[Result] = []

    full = make_chunk(50, now_ns=now)
    results.append(_result(
        "S1", "fake_full_50_point_contract", "ACCEPTED_SHADOW",
        validate_chunk(full, now_ns=now), waypoint_count=50,
    ))

    results.append(_result(
        "S2", "feedback_timeout_transitions_hold", "HOLD",
        watchdog_state(last_feedback_ns=now - 600_000_000, now_ns=now, timeout_ns=500_000_000),
        feedback_age_ms=600, timeout_ms=500,
    ))

    expired = replace(full, valid_until_ns=now - 1)
    results.append(_result(
        "S3", "expired_chunk_rejected", "REJECTED_EXPIRED",
        validate_chunk(expired, now_ns=now), expired_by_ns=1,
    ))

    guard = SequenceGuard()
    first = make_chunk(2, now_ns=now, session="session-a", plan_id=7)
    first_actual = guard.accept(first, now_ns=now)
    replay_actual = guard.accept(first, now_ns=now)
    restarted = make_chunk(2, now_ns=now, session="session-b", plan_id=0)
    restart_during_active = guard.accept(restarted, now_ns=now)
    restart_after_horizon = guard.accept(restarted, now_ns=guard.active_until_ns + 1)
    actual = (
        "SEQUENCE_RESTART_SAFE"
        if first_actual == "ACCEPTED_SHADOW"
        and replay_actual == "REJECTED_ORDERING"
        and restart_during_active == "REJECTED_ORDERING"
        and restart_after_horizon == "ACCEPTED_SHADOW"
        else "SEQUENCE_RESTART_UNSAFE"
    )
    results.append(_result(
        "S4", "sequence_restart", "SEQUENCE_RESTART_SAFE", actual,
        initial=first_actual, replay=replay_actual,
        restart_during_active=restart_during_active,
        restart_after_horizon=restart_after_horizon,
    ))

    for count in (2, 5):
        chunk = make_chunk(count, now_ns=now, session=f"short-{count}")
        results.append(_result(
            "S5", f"short_chunk_{count}_waypoints", "ACCEPTED_SHADOW",
            validate_chunk(chunk, now_ns=now), waypoint_count=count,
        ))
    return results


def write_artifact(path: Path, results: Iterable[Result]) -> dict[str, object]:
    result_list = list(results)
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "franka_closed_loop_staged_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hardware_actuation": False,
        "overall": "PASS" if all(result.passed for result in result_list) else "FAIL",
        "results": [asdict(result) for result in result_list],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact", type=Path,
        default=Path("/tmp/franka_closed_loop/validation-offline.json"),
    )
    args = parser.parse_args()
    results = run_suite()
    payload = write_artifact(args.artifact, results)
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.stage} {result.scenario}: {result.actual}")
    print(f"artifact: {args.artifact}")
    return 0 if payload["overall"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
