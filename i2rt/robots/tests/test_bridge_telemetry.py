import numpy as np

from i2rt.utils.bridge_telemetry import sensorimotor_snapshot


def test_sensorimotor_snapshot_separates_measured_state_from_absolute_command() -> None:
    snapshot = sensorimotor_snapshot(
        timestamp=12.5,
        measured=np.array([0.1, 0.2, 0.3]),
        commanded=np.array([0.4, 0.5, 0.6]),
        arm_dofs=2,
        gripper_index=2,
        objects=[{"name": "cube", "pos": [0.3, 0.0, 0.02], "wxyz": [1, 0, 0, 0]}],
    )

    assert snapshot == {
        "t": 12.5,
        "joints": [0.1, 0.2],
        "gripper": 0.3,
        "action": [0.4, 0.5, 0.6],
        "objects": [{"name": "cube", "pos": [0.3, 0.0, 0.02], "wxyz": [1, 0, 0, 0]}],
    }
