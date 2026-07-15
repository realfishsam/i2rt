"""Camera geometry helpers for simulator calibration observations."""

import math
from collections.abc import Sequence

import numpy as np


def camera_intrinsics_from_fovy(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    """Build a pinhole camera matrix from MuJoCo's vertical field of view."""
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    if not math.isfinite(fovy_degrees) or not 0.0 < fovy_degrees < 180.0:
        raise ValueError("vertical field of view must be finite and between 0 and 180 degrees")

    focal_length = height / (2.0 * math.tan(math.radians(fovy_degrees) / 2.0))
    return np.array(
        [[focal_length, 0.0, width / 2.0], [0.0, focal_length, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def mujoco_camera_to_opencv_transform(
    position: Sequence[float] | np.ndarray,
    world_from_camera_rotation: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Convert a MuJoCo camera pose into an OpenCV world-to-camera transform."""
    camera_position = np.asarray(position, dtype=np.float64)
    world_from_camera = np.asarray(world_from_camera_rotation, dtype=np.float64)
    if camera_position.shape != (3,) or world_from_camera.shape != (3, 3):
        raise ValueError("camera position and rotation must have shapes (3,) and (3, 3)")
    if not np.all(np.isfinite(camera_position)) or not np.all(np.isfinite(world_from_camera)):
        raise ValueError("camera pose must contain only finite values")

    camera_from_world = world_from_camera.T
    mujoco_to_opencv = np.diag([1.0, -1.0, -1.0])
    rotation = mujoco_to_opencv @ camera_from_world
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ -camera_position
    return transform


def project_world_point(
    world_point: Sequence[float] | np.ndarray,
    world_to_camera: Sequence[Sequence[float]] | np.ndarray,
    camera_intrinsics: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Project one world-space point to pixels with a pinhole camera model."""
    point = np.asarray(world_point, dtype=np.float64)
    transform = np.asarray(world_to_camera, dtype=np.float64)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    if point.shape != (3,) or transform.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("point, transform, and intrinsics must have shapes (3,), (4, 4), and (3, 3)")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(transform)) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("projection inputs must contain only finite values")

    camera_point = transform @ np.append(point, 1.0)
    if camera_point[2] <= 0.0:
        raise ValueError("world point is behind the camera")
    homogeneous_pixel = intrinsics @ camera_point[:3]
    return homogeneous_pixel[:2] / homogeneous_pixel[2]
