"""Absolute EEF action decoding and optional Pinocchio IK for the joint client."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from .joint_config_franka_ros import FrankaJointRosConfig


FloatArray = NDArray[np.float64]

_JOINT_COUNT = 7
_EEF_ACTION_DIM = 10
_FR3_Q_MIN = np.asarray([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
_FR3_Q_MAX = np.asarray([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


class EefIkError(RuntimeError):
    """Raised when an EEF chunk cannot be converted into a safe joint plan."""


def decode_eef_actions(
    actions: ArrayLike,
    *,
    eps: float = 1e-12,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Decode ``[xyz, rot6d.col0, rot6d.col1, gripper]`` rows.

    Rotation-6D values are unconstrained network outputs. Gram-Schmidt
    normalizes the first column, removes its component from the second, then
    obtains the right-handed third column with a cross product.
    """

    try:
        raw = np.asarray(actions)
    except (TypeError, ValueError) as error:
        raise EefIkError("EEF actions are not a rectangular numeric array") from error
    if raw.dtype.kind not in "fiu":
        raise EefIkError(f"EEF actions must contain real numbers, got dtype {raw.dtype}")
    value = np.asarray(raw, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != _EEF_ACTION_DIM:
        raise EefIkError(f"EEF actions must have non-empty shape [K,{_EEF_ACTION_DIM}], got {value.shape}")
    if not np.isfinite(value).all():
        raise EefIkError("EEF actions contain NaN or Inf")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and greater than zero")

    first = value[:, 3:6]
    second = value[:, 6:9]
    first_norm = np.linalg.norm(first, axis=1, keepdims=True)
    invalid_first = np.flatnonzero(first_norm[:, 0] <= eps)
    if invalid_first.size:
        raise EefIkError(f"EEF rotation-6D first column is degenerate at waypoint {int(invalid_first[0])}")
    basis_1 = first / first_norm
    orthogonal = second - np.sum(basis_1 * second, axis=1, keepdims=True) * basis_1
    second_norm = np.linalg.norm(orthogonal, axis=1, keepdims=True)
    invalid_second = np.flatnonzero(second_norm[:, 0] <= eps)
    if invalid_second.size:
        raise EefIkError(f"EEF rotation-6D columns are collinear at waypoint {int(invalid_second[0])}")
    basis_2 = orthogonal / second_norm
    basis_3 = np.cross(basis_1, basis_2)
    rotations = np.stack((basis_1, basis_2, basis_3), axis=-1)
    return value[:, :3].copy(), rotations, value[:, 9].copy()


def quaternion_xyzw_to_matrix(quaternion: ArrayLike, *, eps: float = 1e-12) -> FloatArray:
    """Convert one finite ``xyzw`` quaternion into a rotation matrix."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise EefIkError(f"measured EEF quaternion must be finite shape (4,), got {value.shape}")
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        raise EefIkError("measured EEF quaternion is degenerate")
    x, y, z, w = value / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def closed_fraction_to_physical(
    values: ArrayLike,
    *,
    minimum: float,
    maximum: float,
) -> tuple[FloatArray, float, float, int]:
    """Clamp normalized close fractions and map them to Robotiq positions."""

    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
        raise EefIkError("EEF gripper actions must be a non-empty finite vector")
    bounded = np.clip(raw, 0.0, 1.0)
    physical = float(minimum) + bounded * (float(maximum) - float(minimum))
    return physical, float(np.min(raw)), float(np.max(raw)), int(np.count_nonzero(raw != bounded))


class PinocchioEefIk:
    """Warm-started FR3 IK preserving the measured redundant posture."""

    def __init__(self, config: FrankaJointRosConfig):
        try:
            import pinocchio as pin
        except (ImportError, ModuleNotFoundError) as error:
            raise EefIkError("EEF action_space requires Pinocchio in the client environment") from error

        urdf_path = config.ik_urdf_path.expanduser()
        if not urdf_path.is_file():
            raise EefIkError(f"IK URDF does not exist: {urdf_path}")

        self._pin: Any = pin
        try:
            self._model = pin.buildModelFromUrdf(str(urdf_path))
        except Exception as error:
            raise EefIkError(f"could not load IK URDF {urdf_path}: {error}") from error
        if self._model.nq != _JOINT_COUNT or self._model.nv != _JOINT_COUNT:
            raise EefIkError(
                f"IK URDF must have nq=nv={_JOINT_COUNT}, got nq={self._model.nq}, nv={self._model.nv}"
            )
        model_joint_names = tuple(str(name) for name in self._model.names[1:])
        if model_joint_names != tuple(config.arm_joint_names):
            raise EefIkError(
                "IK URDF joint order does not match arm_joint_names: "
                f"{model_joint_names} != {tuple(config.arm_joint_names)}"
            )
        if not self._model.existFrame(config.ik_tip_link):
            raise EefIkError(f"IK tip frame {config.ik_tip_link!r} is absent from {urdf_path}")

        self._data = self._model.createData()
        self._frame_id = self._model.getFrameId(config.ik_tip_link)
        margin = float(config.ik_joint_limit_margin)
        self._lower = _FR3_Q_MIN + margin
        self._upper = _FR3_Q_MAX - margin
        self._posture_gain = float(config.ik_posture_gain)
        self._tolerance = float(config.ik_tolerance)
        self._max_iterations = int(config.ik_max_iterations)
        self._damping = float(config.ik_damping)

    def _fk(self, q: FloatArray) -> Any:
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateFramePlacements(self._model, self._data)
        return self._data.oMf[self._frame_id]

    def _iterate(
        self, desired_pose: Any, q_reference: FloatArray, posture_gain: float
    ) -> tuple[FloatArray, bool]:
        q = q_reference.copy()
        identity = np.eye(6)
        for _ in range(self._max_iterations):
            current_pose = self._fk(q)
            error = self._pin.log6(current_pose.actInv(desired_pose)).vector
            if float(np.linalg.norm(error)) < self._tolerance:
                return np.clip(q, self._lower, self._upper), True

            jacobian = self._pin.computeFrameJacobian(
                self._model,
                self._data,
                q,
                self._frame_id,
            )
            normal = jacobian @ jacobian.T + self._damping * identity
            damped_pseudoinverse = jacobian.T @ np.linalg.solve(normal, identity)
            task_velocity = damped_pseudoinverse @ error

            null_direction = np.linalg.svd(jacobian, full_matrices=True)[2][-1]
            posture_error = q_reference - q
            posture_velocity = posture_gain * null_direction * float(null_direction @ posture_error)
            q = np.clip(
                self._pin.integrate(
                    self._model,
                    q,
                    task_velocity + posture_velocity,
                ),
                self._lower,
                self._upper,
            )

        error = self._pin.log6(self._fk(q).actInv(desired_pose)).vector
        return q, float(np.linalg.norm(error)) < self._tolerance

    def _solve(self, desired_pose: Any, q_seed: FloatArray) -> tuple[FloatArray, bool]:
        q_reference = np.clip(q_seed, self._lower, self._upper)
        q, converged = self._iterate(desired_pose, q_reference, self._posture_gain)
        if converged or self._posture_gain == 0.0:
            return q, converged
        return self._iterate(desired_pose, q_reference, 0.0)

    def solve_actions(
        self,
        actions: ArrayLike,
        *,
        measured_q: ArrayLike,
        measured_eef_position: ArrayLike,
        measured_eef_quaternion_xyzw: ArrayLike,
    ) -> FloatArray:
        """Convert one absolute EEF chunk into warm-started joint targets."""

        positions, rotations, _ = decode_eef_actions(actions)
        q_seed = np.asarray(measured_q, dtype=np.float64)
        eef_position = np.asarray(measured_eef_position, dtype=np.float64)
        if q_seed.shape != (_JOINT_COUNT,) or not np.isfinite(q_seed).all():
            raise EefIkError(f"measured qpos must be finite shape ({_JOINT_COUNT},), got {q_seed.shape}")
        if eef_position.shape != (3,) or not np.isfinite(eef_position).all():
            raise EefIkError(f"measured EEF position must be finite shape (3,), got {eef_position.shape}")

        measured_eef = self._pin.SE3(
            quaternion_xyzw_to_matrix(measured_eef_quaternion_xyzw),
            eef_position,
        )
        measured_link = self._fk(q_seed)
        link_to_eef = measured_link.inverse() * measured_eef
        eef_to_link = link_to_eef.inverse()

        waypoints = np.empty((len(positions), _JOINT_COUNT), dtype=np.float64)
        seed = q_seed.copy()
        for index, (position, rotation) in enumerate(zip(positions, rotations, strict=True)):
            desired_eef = self._pin.SE3(rotation, position)
            desired_link = desired_eef * eef_to_link
            solution, converged = self._solve(desired_link, seed)
            if not converged:
                raise EefIkError(f"IK did not converge at EEF waypoint {index}")
            waypoints[index] = solution
            seed = solution
        return waypoints


__all__ = [
    "EefIkError",
    "PinocchioEefIk",
    "closed_fraction_to_physical",
    "decode_eef_actions",
    "quaternion_xyzw_to_matrix",
]
