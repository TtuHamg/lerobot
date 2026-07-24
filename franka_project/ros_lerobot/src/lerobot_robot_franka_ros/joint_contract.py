"""Frozen wire and fixture contracts for the FastWAM joint-space Franka adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contract import CAMERA_NAMES, CAMERA_SHAPE

JOINT_STATE_NAMES = (
    "fr3_joint1.pos",
    "fr3_joint2.pos",
    "fr3_joint3.pos",
    "fr3_joint4.pos",
    "fr3_joint5.pos",
    "fr3_joint6.pos",
    "fr3_joint7.pos",
    "gripper.pos",
)

JOINT_ACTION_NAMES = (
    "target.fr3_joint1",
    "target.fr3_joint2",
    "target.fr3_joint3",
    "target.fr3_joint4",
    "target.fr3_joint5",
    "target.fr3_joint6",
    "target.fr3_joint7",
    "target.gripper.pos",
)

FIXTURE_KEYS = frozenset(("state", *CAMERA_NAMES))


@dataclass(frozen=True, slots=True)
class JointDryRunObservation:
    """One immutable, validated joint-space observation used by the dry-run source."""

    state: NDArray[np.float32]
    camera1: NDArray[np.uint8]
    camera2: NDArray[np.uint8]

    def as_robot_observation(self) -> dict[str, float | NDArray[np.uint8]]:
        observation: dict[str, float | NDArray[np.uint8]] = {
            name: float(value) for name, value in zip(JOINT_STATE_NAMES, self.state, strict=True)
        }
        # Return copies because downstream image preparation is allowed to mutate its input.
        observation["camera1"] = self.camera1.copy()
        observation["camera2"] = self.camera2.copy()
        return observation


def load_joint_dry_run_observation(path: Path) -> JointDryRunObservation:
    """Load a joint-space fixture without pickle and reject any contract drift."""

    if not path.is_file():
        raise FileNotFoundError(f"Dry-run observation fixture does not exist: {path}")

    try:
        with np.load(path, allow_pickle=False) as fixture:
            keys = frozenset(fixture.files)
            if keys != FIXTURE_KEYS:
                missing = sorted(FIXTURE_KEYS - keys)
                extra = sorted(keys - FIXTURE_KEYS)
                raise ValueError(f"Fixture keys do not match contract; missing={missing}, extra={extra}")

            state = np.asarray(fixture["state"])
            camera1 = np.asarray(fixture["camera1"])
            camera2 = np.asarray(fixture["camera2"])
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Fixture keys"):
            raise
        raise ValueError(f"Could not read dry-run observation fixture {path}: {error}") from error

    if state.shape != (len(JOINT_STATE_NAMES),):
        raise ValueError(
            f"Fixture state must have shape ({len(JOINT_STATE_NAMES)},), got {state.shape}"
        )
    if state.dtype != np.float32:
        raise TypeError(f"Fixture state must have dtype float32, got {state.dtype}")
    if not np.isfinite(state).all():
        raise ValueError("Fixture state contains a non-finite value")

    cameras: dict[str, NDArray[np.uint8]] = {"camera1": camera1, "camera2": camera2}
    for name, camera in cameras.items():
        if camera.shape != CAMERA_SHAPE:
            raise ValueError(f"Fixture {name} must have shape {CAMERA_SHAPE}, got {camera.shape}")
        if camera.dtype != np.uint8:
            raise TypeError(f"Fixture {name} must have dtype uint8, got {camera.dtype}")

    return JointDryRunObservation(
        state=np.ascontiguousarray(state),
        camera1=np.ascontiguousarray(camera1),
        camera2=np.ascontiguousarray(camera2),
    )


def validate_joint_action(action: dict[str, Any]) -> dict[str, float]:
    """Validate and canonically order one absolute joint-space target."""

    if not isinstance(action, dict):
        raise TypeError(f"Action must be a dict, got {type(action).__name__}")

    keys = set(action)
    expected = set(JOINT_ACTION_NAMES)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"Action keys do not match contract; missing={missing}, extra={extra}")

    ordered: dict[str, float] = {}
    for name in JOINT_ACTION_NAMES:
        value = action[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Action {name!r} must be a real scalar, got {type(value).__name__}")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"Action {name!r} must be finite, got {scalar}")
        ordered[name] = scalar

    return ordered


__all__ = [
    "JOINT_ACTION_NAMES",
    "JOINT_STATE_NAMES",
    "JointDryRunObservation",
    "load_joint_dry_run_observation",
    "validate_joint_action",
]
