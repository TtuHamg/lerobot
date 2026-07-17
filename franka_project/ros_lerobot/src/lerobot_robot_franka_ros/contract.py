"""Frozen Phase 1 wire and fixture contracts for the Franka adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

STATE_NAMES = (
    "eef.x",
    "eef.y",
    "eef.z",
    "eef.rot6d.col0.x",
    "eef.rot6d.col0.y",
    "eef.rot6d.col0.z",
    "eef.rot6d.col1.x",
    "eef.rot6d.col1.y",
    "eef.rot6d.col1.z",
    "gripper.closed_0_1",
)

ABSOLUTE_ACTION_NAMES = (
    "target.x",
    "target.y",
    "target.z",
    "target.qx",
    "target.qy",
    "target.qz",
    "target.qw",
    "target.gripper.closed_0_1",
)

CAMERA_NAMES = ("camera1", "camera2")
CAMERA_SHAPE = (480, 640, 3)
FIXTURE_KEYS = frozenset(("state", *CAMERA_NAMES))


@dataclass(frozen=True, slots=True)
class DryRunObservation:
    """One immutable, validated observation used by the Phase 1 source."""

    state: NDArray[np.float32]
    camera1: NDArray[np.uint8]
    camera2: NDArray[np.uint8]

    def as_robot_observation(self) -> dict[str, float | NDArray[np.uint8]]:
        observation: dict[str, float | NDArray[np.uint8]] = {
            name: float(value) for name, value in zip(STATE_NAMES, self.state, strict=True)
        }
        # Return copies because downstream image preparation is allowed to mutate its input.
        observation["camera1"] = self.camera1.copy()
        observation["camera2"] = self.camera2.copy()
        return observation


def load_dry_run_observation(path: Path) -> DryRunObservation:
    """Load a fixture without pickle and reject any contract drift."""

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

    if state.shape != (len(STATE_NAMES),):
        raise ValueError(f"Fixture state must have shape ({len(STATE_NAMES)},), got {state.shape}")
    if state.dtype != np.float32:
        raise TypeError(f"Fixture state must have dtype float32, got {state.dtype}")
    if not np.isfinite(state).all():
        raise ValueError("Fixture state contains a non-finite value")

    # The server decodes this representation with Gram-Schmidt. Reject the
    # two singular cases here so a bad fixture cannot reach remote inference.
    first_column = state[3:6].astype(np.float64)
    second_column = state[6:9].astype(np.float64)
    first_norm = float(np.linalg.norm(first_column))
    if first_norm <= 1e-12:
        raise ValueError("Fixture rotation-6D first column is degenerate")
    first_basis = first_column / first_norm
    orthogonal = second_column - float(np.dot(first_basis, second_column)) * first_basis
    if float(np.linalg.norm(orthogonal)) <= 1e-12:
        raise ValueError("Fixture rotation-6D columns are collinear")
    gripper = float(state[-1])
    if not 0.0 <= gripper <= 1.0:
        raise ValueError(f"Fixture gripper state must be in [0, 1], got {gripper}")

    cameras: dict[str, NDArray[np.uint8]] = {"camera1": camera1, "camera2": camera2}
    for name, camera in cameras.items():
        if camera.shape != CAMERA_SHAPE:
            raise ValueError(f"Fixture {name} must have shape {CAMERA_SHAPE}, got {camera.shape}")
        if camera.dtype != np.uint8:
            raise TypeError(f"Fixture {name} must have dtype uint8, got {camera.dtype}")

    return DryRunObservation(
        state=np.ascontiguousarray(state),
        camera1=np.ascontiguousarray(camera1),
        camera2=np.ascontiguousarray(camera2),
    )


def validate_absolute_action(
    action: dict[str, Any], *, quaternion_norm_tolerance: float
) -> dict[str, float]:
    """Validate and canonically order one absolute Cartesian target."""

    if not isinstance(action, dict):
        raise TypeError(f"Action must be a dict, got {type(action).__name__}")

    keys = set(action)
    expected = set(ABSOLUTE_ACTION_NAMES)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"Action keys do not match contract; missing={missing}, extra={extra}")

    ordered: dict[str, float] = {}
    for name in ABSOLUTE_ACTION_NAMES:
        value = action[name]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"Action {name!r} must be a real scalar, got {type(value).__name__}")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"Action {name!r} must be finite, got {scalar}")
        ordered[name] = scalar

    quaternion = tuple(ordered[name] for name in ABSOLUTE_ACTION_NAMES[3:7])
    quaternion_norm = math.sqrt(sum(component * component for component in quaternion))
    if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=quaternion_norm_tolerance):
        raise ValueError(
            "Action quaternion must be unit length; "
            f"norm={quaternion_norm:.9g}, tolerance={quaternion_norm_tolerance:.9g}"
        )

    gripper = ordered[ABSOLUTE_ACTION_NAMES[-1]]
    if not 0.0 <= gripper <= 1.0:
        raise ValueError(f"Action gripper target must be in [0, 1], got {gripper}")

    return ordered
