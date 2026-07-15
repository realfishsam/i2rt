import textwrap
from pathlib import Path

import mujoco
import numpy as np
import pytest

from examples.replay_viewer.replay_viewer import ReplayViewer
from i2rt.utils.viser_control_interface import ViserControlInterface

SCENE_XML = """
<mujoco>
  <worldbody>
    <body name="arm_link">
      <joint name="arm_joint" type="hinge"/>
      <geom type="box" size="0.01 0.01 0.01"/>
    </body>
    <body name="cube" pos="0.35 0 0.025">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025"/>
    </body>
  </worldbody>
</mujoco>
"""


def _scene_path(tmp_path: Path) -> str:
    path = tmp_path / "scene.xml"
    path.write_text(textwrap.dedent(SCENE_XML))
    return str(path)


def test_scene_objects_snapshot_serializes_free_joint_pose(tmp_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(_scene_path(tmp_path))
    data = mujoco.MjData(model)
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    address = model.jnt_qposadr[cube_joint]
    data.qpos[address : address + 7] = [0.46, -0.02, 0.025, 1.0, 0.0, 0.0, 0.0]

    iface = ViserControlInterface.__new__(ViserControlInterface)
    iface._model = model
    iface._data = data

    assert iface._scene_object_snapshot() == [
        {"name": "cube", "pos": [0.46, -0.02, 0.025], "wxyz": [1.0, 0.0, 0.0, 0.0]}
    ]


def test_replay_viewer_applies_recorded_scene_object_pose(tmp_path: Path) -> None:
    viewer = ReplayViewer(_scene_path(tmp_path), viser_port=8082, pose_port=8083)

    viewer._apply_pose(
        [0.25],
        0.0,
        [{"name": "cube", "pos": [0.46, -0.02, 0.025], "wxyz": [1.0, 0.0, 0.0, 0.0]}],
    )

    cube_body = mujoco.mj_name2id(viewer._model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    np.testing.assert_allclose(viewer._data.xpos[cube_body], [0.46, -0.02, 0.025])


def test_replay_viewer_ignores_unknown_scene_objects(tmp_path: Path) -> None:
    viewer = ReplayViewer(_scene_path(tmp_path), viser_port=8082, pose_port=8083)

    viewer._apply_pose(
        [0.25],
        0.0,
        [{"name": "not-in-this-scene", "pos": [1, 2, 3], "wxyz": [1, 0, 0, 0]}],
    )

    assert np.all(np.isfinite(viewer._data.qpos))


@pytest.mark.parametrize(
    "payload",
    [
        {"joints": [float("nan")], "gripper": 0.0},
        {"joints": [], "gripper": float("inf")},
        {"joints": [], "gripper": 0.0, "objects": [{"name": "cube", "pos": [0, 0, 0], "wxyz": [0, 0, 0, 0]}]},
    ],
)
def test_replay_viewer_rejects_invalid_pose_payload(payload: object) -> None:
    with pytest.raises(ValueError):
        ReplayViewer._parse_pose_payload(payload)
