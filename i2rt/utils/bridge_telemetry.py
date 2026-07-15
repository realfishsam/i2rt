"""Build synchronized robot state and command snapshots for bridge clients."""

from typing import Any, Dict, List, Optional

import numpy as np


def sensorimotor_snapshot(
    *,
    timestamp: float,
    measured: np.ndarray,
    commanded: np.ndarray,
    arm_dofs: int,
    gripper_index: Optional[int],
    objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep observations distinct from the absolute target sent to the robot."""
    gripper = float(measured[gripper_index]) if gripper_index is not None else 0.0
    action = [float(value) for value in commanded[:arm_dofs]]
    action.append(float(commanded[gripper_index]) if gripper_index is not None else 0.0)
    return {
        "t": float(timestamp),
        "joints": [float(value) for value in measured[:arm_dofs]],
        "gripper": gripper,
        "action": action,
        "objects": [dict(item) for item in objects],
    }
