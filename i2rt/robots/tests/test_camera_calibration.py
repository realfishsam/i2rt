import numpy as np
import pytest

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.camera_calibration import (
    camera_intrinsics_from_fovy,
    mujoco_camera_to_opencv_transform,
    project_world_point,
)
from i2rt.utils.viser_control_interface import ViserControlInterface


def test_camera_intrinsics_from_vertical_fov() -> None:
    intrinsics = camera_intrinsics_from_fovy(width=640, height=480, fovy_degrees=90.0)

    np.testing.assert_allclose(
        intrinsics,
        np.array([[240.0, 0.0, 320.0], [0.0, 240.0, 240.0], [0.0, 0.0, 1.0]]),
        atol=1e-9,
    )


@pytest.mark.parametrize("width,height,fovy", [(0, 480, 45.0), (640, 0, 45.0), (640, 480, 0.0), (640, 480, 180.0)])
def test_camera_intrinsics_reject_invalid_parameters(width: int, height: int, fovy: float) -> None:
    with pytest.raises(ValueError):
        camera_intrinsics_from_fovy(width=width, height=height, fovy_degrees=fovy)


def test_mujoco_camera_convention_projects_forward_to_image_center() -> None:
    world_to_camera = mujoco_camera_to_opencv_transform(
        position=np.zeros(3),
        world_from_camera_rotation=np.eye(3),
    )
    intrinsics = camera_intrinsics_from_fovy(width=640, height=480, fovy_degrees=90.0)

    np.testing.assert_allclose(world_to_camera[:3, :3], np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(project_world_point([0.0, 0.0, -1.0], world_to_camera, intrinsics), [320.0, 240.0])
    np.testing.assert_allclose(project_world_point([0.5, 0.5, -1.0], world_to_camera, intrinsics), [440.0, 120.0])


def test_mujoco_camera_transform_accounts_for_translation() -> None:
    world_to_camera = mujoco_camera_to_opencv_transform(
        position=np.array([1.0, 2.0, 3.0]),
        world_from_camera_rotation=np.eye(3),
    )

    np.testing.assert_allclose(world_to_camera @ np.array([1.0, 2.0, 2.0, 1.0]), [0.0, 0.0, 1.0, 1.0])


def test_projection_rejects_points_behind_camera() -> None:
    with pytest.raises(ValueError, match="behind"):
        project_world_point([0.0, 0.0, -1.0], np.eye(4), np.eye(3))


def test_simulator_calibration_snapshot_is_self_consistent() -> None:
    robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, sim=True)
    interface = ViserControlInterface(robot, robot.xml_path)
    interface._mirror_robot()

    snapshot = interface._camera_calibration_snapshot("overhead", width=640, height=480)
    projected = project_world_point(
        snapshot["reference"]["world"],
        snapshot["ground_truth_world_to_camera"],
        snapshot["camera_intrinsics"],
    )

    assert snapshot["camera"] == "overhead"
    assert snapshot["width"] == 640
    assert snapshot["height"] == 480
    assert snapshot["reference"]["name"] == "tcp"
    np.testing.assert_allclose(projected, snapshot["reference"]["pixel"], atol=1e-9)
