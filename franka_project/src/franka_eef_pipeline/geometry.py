"""Geometry primitives for the Franka Cartesian-action pipeline.

Conventions frozen by the approved plan:

* quaternions are ``xyzw``;
* absolute orientation uses the first two rotation-matrix columns as rotation-6D;
* translation deltas are expressed in the robot base frame;
* rotation deltas are body-frame rotation vectors:
  ``Log(R_anchor.T @ R_target)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.floating]


def _finite_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _require_last_shape(array: FloatArray, shape: tuple[int, ...], *, name: str) -> None:
    if array.shape[-len(shape) :] != shape:
        raise ValueError(f"{name} must end with shape {shape}, got {array.shape}")


def normalize_quaternion_xyzw(quaternion: ArrayLike, *, eps: float = 1e-12) -> FloatArray:
    """Normalize one or more ``xyzw`` quaternions."""

    q = _finite_array(quaternion, name="quaternion")
    _require_last_shape(q, (4,), name="quaternion")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= eps):
        raise ValueError("quaternion norm is zero or too small")
    return q / norm


def enforce_quaternion_continuity(quaternions: ArrayLike) -> FloatArray:
    """Flip quaternion signs along a trajectory so adjacent dots are non-negative."""

    q = normalize_quaternion_xyzw(quaternions)
    if q.ndim != 2:
        raise ValueError(f"quaternions must have shape [T,4], got {q.shape}")
    out = q.copy()
    for index in range(1, len(out)):
        if float(np.dot(out[index - 1], out[index])) < 0.0:
            out[index] *= -1.0
    return out


def quaternion_xyzw_to_matrix(quaternion: ArrayLike) -> FloatArray:
    """Convert normalized ``xyzw`` quaternions to rotation matrices."""

    q = normalize_quaternion_xyzw(quaternion)
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def _validate_rotation_matrix(matrix: ArrayLike, *, atol: float = 1e-6) -> FloatArray:
    rotation = _finite_array(matrix, name="rotation matrix")
    _require_last_shape(rotation, (3, 3), name="rotation matrix")
    identity = np.eye(3, dtype=np.float64)
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    if not np.allclose(gram, identity, atol=atol, rtol=0.0):
        raise ValueError("rotation matrix is not orthonormal")
    determinant = np.linalg.det(rotation)
    if not np.allclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError("rotation matrix determinant must be +1")
    return rotation


def matrix_to_quaternion_xyzw(matrix: ArrayLike) -> FloatArray:
    """Convert rotation matrices to canonical ``xyzw`` quaternions.

    The returned representative has non-negative ``w`` (with a deterministic
    tie-break at 180 degrees). Quaternion sign is physically irrelevant; use
    :func:`enforce_quaternion_continuity` for a trajectory.
    """

    rotation = _validate_rotation_matrix(matrix)
    flat = rotation.reshape((-1, 3, 3))
    output = np.empty((len(flat), 4), dtype=np.float64)
    for index, item in enumerate(flat):
        trace = float(np.trace(item))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            q = np.array(
                [
                    (item[2, 1] - item[1, 2]) / scale,
                    (item[0, 2] - item[2, 0]) / scale,
                    (item[1, 0] - item[0, 1]) / scale,
                    0.25 * scale,
                ]
            )
        else:
            diagonal = np.diag(item)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = 2.0 * np.sqrt(max(1.0 + item[0, 0] - item[1, 1] - item[2, 2], 0.0))
                q = np.array(
                    [
                        0.25 * scale,
                        (item[0, 1] + item[1, 0]) / scale,
                        (item[0, 2] + item[2, 0]) / scale,
                        (item[2, 1] - item[1, 2]) / scale,
                    ]
                )
            elif axis == 1:
                scale = 2.0 * np.sqrt(max(1.0 + item[1, 1] - item[0, 0] - item[2, 2], 0.0))
                q = np.array(
                    [
                        (item[0, 1] + item[1, 0]) / scale,
                        0.25 * scale,
                        (item[1, 2] + item[2, 1]) / scale,
                        (item[0, 2] - item[2, 0]) / scale,
                    ]
                )
            else:
                scale = 2.0 * np.sqrt(max(1.0 + item[2, 2] - item[0, 0] - item[1, 1], 0.0))
                q = np.array(
                    [
                        (item[0, 2] + item[2, 0]) / scale,
                        (item[1, 2] + item[2, 1]) / scale,
                        0.25 * scale,
                        (item[1, 0] - item[0, 1]) / scale,
                    ]
                )
        q = normalize_quaternion_xyzw(q)
        if q[3] < 0.0 or (abs(q[3]) <= 1e-15 and q[int(np.argmax(np.abs(q[:3])))] < 0.0):
            q = -q
        output[index] = q
    return output.reshape(rotation.shape[:-2] + (4,))


def matrix_to_rotation_6d(matrix: ArrayLike) -> FloatArray:
    """Encode a rotation using its first two columns, column-major."""

    rotation = _validate_rotation_matrix(matrix)
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def rotation_6d_to_matrix(rotation_6d: ArrayLike, *, eps: float = 1e-12) -> FloatArray:
    """Decode the first-two-columns rotation-6D representation."""

    value = _finite_array(rotation_6d, name="rotation_6d")
    _require_last_shape(value, (6,), name="rotation_6d")
    first = value[..., :3]
    second = value[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm <= eps):
        raise ValueError("rotation_6d first column is degenerate")
    basis_1 = first / first_norm
    orthogonal = second - np.sum(basis_1 * second, axis=-1, keepdims=True) * basis_1
    second_norm = np.linalg.norm(orthogonal, axis=-1, keepdims=True)
    if np.any(second_norm <= eps):
        raise ValueError("rotation_6d columns are collinear")
    basis_2 = orthogonal / second_norm
    basis_3 = np.cross(basis_1, basis_2)
    return np.stack((basis_1, basis_2, basis_3), axis=-1)


def _hat(vector: FloatArray) -> FloatArray:
    _require_last_shape(vector, (3,), name="rotation vector")
    x, y, z = np.moveaxis(vector, -1, 0)
    result = np.zeros(vector.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 1] = -z
    result[..., 0, 2] = y
    result[..., 1, 0] = z
    result[..., 1, 2] = -x
    result[..., 2, 0] = -y
    result[..., 2, 1] = x
    return result


def _vee_skew(matrix: FloatArray) -> FloatArray:
    return np.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        axis=-1,
    )


def so3_exp(rotation_vector: ArrayLike) -> FloatArray:
    """SO(3) exponential map for axis-angle rotation vectors."""

    vector = _finite_array(rotation_vector, name="rotation_vector")
    _require_last_shape(vector, (3,), name="rotation_vector")
    theta_sq = np.sum(vector * vector, axis=-1)
    theta = np.sqrt(theta_sq)
    small = theta_sq < 1e-12
    safe_theta = np.where(small, 1.0, theta)
    safe_theta_sq = np.where(small, 1.0, theta_sq)
    coefficient_a = np.where(
        small,
        1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0,
        np.sin(theta) / safe_theta,
    )
    coefficient_b = np.where(
        small,
        0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0,
        (1.0 - np.cos(theta)) / safe_theta_sq,
    )
    skew = _hat(vector)
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape)
    return identity + coefficient_a[..., None, None] * skew + coefficient_b[..., None, None] * (skew @ skew)


def so3_log(matrix: ArrayLike) -> FloatArray:
    """SO(3) logarithm map returning the principal rotation vector."""

    rotation = _validate_rotation_matrix(matrix, atol=2e-6)
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cosine)
    skew_vector = _vee_skew(rotation)

    result = np.empty(rotation.shape[:-2] + (3,), dtype=np.float64)
    small = theta < 1e-7
    near_pi = (np.pi - theta) < 1e-5
    regular = ~(small | near_pi)

    if np.any(small):
        # 0.5*vee(R-R.T), with the first non-zero series correction.
        half_vee = 0.5 * skew_vector[small]
        theta_sq = theta[small] ** 2
        result[small] = half_vee * (1.0 + theta_sq[..., None] / 6.0)

    if np.any(regular):
        scale = theta[regular] / (2.0 * np.sin(theta[regular]))
        result[regular] = scale[..., None] * skew_vector[regular]

    if np.any(near_pi):
        flat_rotation = rotation[near_pi].reshape((-1, 3, 3))
        flat_theta = theta[near_pi].reshape(-1)
        near_result = np.empty((len(flat_rotation), 3), dtype=np.float64)
        for index, (item, angle) in enumerate(zip(flat_rotation, flat_theta, strict=True)):
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (item + item.T))
            axis = eigenvectors[:, int(np.argmax(eigenvalues))]
            axis /= np.linalg.norm(axis)
            skew_hint = _vee_skew(item)
            if np.linalg.norm(skew_hint) > 1e-10:
                if float(np.dot(axis, skew_hint)) < 0.0:
                    axis = -axis
            else:
                dominant = int(np.argmax(np.abs(axis)))
                if axis[dominant] < 0.0:
                    axis = -axis
            near_result[index] = angle * axis
        result[near_pi] = near_result.reshape(result[near_pi].shape)
    return result


def rotation_geodesic_angle(first: ArrayLike, second: ArrayLike) -> FloatArray:
    """Return the SO(3) geodesic angle between rotation matrices, in radians."""

    rotation_a = _validate_rotation_matrix(first, atol=2e-6)
    rotation_b = _validate_rotation_matrix(second, atol=2e-6)
    relative = np.swapaxes(rotation_a, -1, -2) @ rotation_b
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _broadcast_anchor_vector(anchor: FloatArray, target: FloatArray, *, name: str) -> FloatArray:
    _require_last_shape(anchor, (3,), name=name)
    if anchor.ndim > target.ndim:
        raise ValueError(f"{name} has more dimensions than target")
    extra = target.ndim - anchor.ndim
    return anchor.reshape(anchor.shape[:-1] + (1,) * extra + (3,))


def _broadcast_anchor_matrix(anchor: FloatArray, target: FloatArray) -> FloatArray:
    _require_last_shape(anchor, (3, 3), name="anchor_rotation")
    if anchor.ndim > target.ndim:
        raise ValueError("anchor_rotation has more dimensions than target_rotation")
    extra = target.ndim - anchor.ndim
    return anchor.reshape(anchor.shape[:-2] + (1,) * extra + (3, 3))


def encode_relative_action(
    anchor_position: ArrayLike,
    anchor_rotation: ArrayLike,
    target_position: ArrayLike,
    target_rotation: ArrayLike,
    target_gripper: ArrayLike,
) -> FloatArray:
    """Encode base-frame translation/body-frame rotation/gripper as 7D actions."""

    position = _finite_array(target_position, name="target_position")
    rotation = _validate_rotation_matrix(target_rotation, atol=2e-6)
    _require_last_shape(position, (3,), name="target_position")
    if rotation.shape[:-2] != position.shape[:-1]:
        raise ValueError("target_position and target_rotation leading shapes differ")

    anchor_p = _finite_array(anchor_position, name="anchor_position")
    anchor_r = _validate_rotation_matrix(anchor_rotation, atol=2e-6)
    anchor_p = _broadcast_anchor_vector(anchor_p, position, name="anchor_position")
    anchor_r = _broadcast_anchor_matrix(anchor_r, rotation)

    delta_position = position - anchor_p
    relative_rotation = np.swapaxes(anchor_r, -1, -2) @ rotation
    delta_rotation = so3_log(relative_rotation)

    gripper = _finite_array(target_gripper, name="target_gripper")
    if gripper.shape == position.shape[:-1]:
        gripper = gripper[..., None]
    if gripper.shape != position.shape[:-1] + (1,):
        raise ValueError(
            "target_gripper must have target leading shape with optional final singleton, "
            f"got {gripper.shape} for target {position.shape}"
        )
    return np.concatenate((delta_position, delta_rotation, gripper), axis=-1)


def encode_absolute_action(
    target_position: ArrayLike,
    target_rotation: ArrayLike,
    target_gripper: ArrayLike,
) -> FloatArray:
    """Encode an absolute EEF target as ``[p_base, Log(R_base_to_eef), gripper]``.

    The rotation vector is the principal SO(3) logarithm of the absolute target
    rotation.  Unlike :func:`encode_relative_action`, no observation anchor is
    used by this representation.
    """

    position = _finite_array(target_position, name="target_position")
    rotation = _validate_rotation_matrix(target_rotation, atol=2e-6)
    _require_last_shape(position, (3,), name="target_position")
    if rotation.shape[:-2] != position.shape[:-1]:
        raise ValueError("target_position and target_rotation leading shapes differ")

    gripper = _finite_array(target_gripper, name="target_gripper")
    if gripper.shape == position.shape[:-1]:
        gripper = gripper[..., None]
    if gripper.shape != position.shape[:-1] + (1,):
        raise ValueError(
            "target_gripper must have target leading shape with optional final singleton, "
            f"got {gripper.shape} for target {position.shape}"
        )
    return np.concatenate((position, so3_log(rotation), gripper), axis=-1)


def decode_absolute_action(
    action: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Decode a 7D absolute EEF action into position, rotation and gripper."""

    value = _finite_array(action, name="action")
    _require_last_shape(value, (7,), name="action")
    return value[..., :3].copy(), so3_exp(value[..., 3:6]), value[..., 6:7].copy()


def decode_relative_action(
    anchor_position: ArrayLike,
    anchor_rotation: ArrayLike,
    action: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Decode a 7D relative action into absolute position, rotation and gripper."""

    value = _finite_array(action, name="action")
    _require_last_shape(value, (7,), name="action")
    anchor_p = _finite_array(anchor_position, name="anchor_position")
    anchor_r = _validate_rotation_matrix(anchor_rotation, atol=2e-6)
    anchor_p = _broadcast_anchor_vector(anchor_p, value[..., :3], name="anchor_position")
    target_rotation_shape = value.shape[:-1] + (3, 3)
    anchor_r = _broadcast_anchor_matrix(
        anchor_r,
        np.empty(target_rotation_shape, dtype=np.float64),
    )
    position = anchor_p + value[..., :3]
    rotation = anchor_r @ so3_exp(value[..., 3:6])
    gripper = value[..., 6:7].copy()
    return position, rotation, gripper
