"""View-only 3D replay viewer for recorded episodes.

Loads the same generated scene MJCF the sim uses (arm + floor + cube), serves an
interactive viser scene on --port (default 8082) with NO control widgets, and
accepts poses on a tiny HTTP endpoint on --pose-port (default 8083):

    POST /pose {"joints": [j1..j6], "gripper": 0..1, "objects": [{"name": ..., "pos": [...], "wxyz": [...]}]}

The OneShotRobot replay page embeds the viser viewport (iframe name="mirror",
which the locally patched viser client build renders chrome-free) and streams
telemetry samples here as the user scrubs the timeline.

Run:  python examples/replay_viewer/replay_viewer.py
"""

import argparse
import glob
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np


def _newest_scene() -> str:
    candidates = sorted(glob.glob("/tmp/i2rt_*_scene.xml"), key=lambda p: -__import__("os").path.getmtime(p))
    if not candidates:
        raise SystemExit(
            "No generated scene found (/tmp/i2rt_*_scene.xml). Start the sim once first:\n"
            "  python examples/control_with_viser/control_with_viser.py --sim\n"
            "or pass --scene <path>."
        )
    return candidates[0]


def _mat3_to_wxyz(mat3: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, mat3.flatten())
    return q


class ReplayViewer:
    """MuJoCo scene mirrored into viser, posed exclusively via POST /pose."""

    def __init__(self, scene_xml: str, viser_port: int, pose_port: int) -> None:
        self._model = mujoco.MjModel.from_xml_path(scene_xml)
        self._data = mujoco.MjData(self._model)
        self._viser_port = viser_port
        self._pose_port = pose_port

        free = [j for j in range(self._model.njnt) if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self._free_qpos_start = min((int(self._model.jnt_qposadr[j]) for j in free), default=self._model.nq)
        self._free_qpos_by_body_name = {
            str(mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, int(self._model.jnt_bodyid[j]))): int(
                self._model.jnt_qposadr[j]
            )
            for j in free
        }

        self._mesh_geom_ids: List[int] = []
        self._box_geom_ids: List[int] = []
        self._pending: Dict[str, Any] = {"pose": None}

        mujoco.mj_forward(self._model, self._data)

    # ---- qpos plumbing (mirrors ViserControlInterface conventions) -----------

    def _apply_pose(self, joints: List[float], gripper: float, objects: List[Dict[str, Any]]) -> None:
        vec = list(joints) + [gripper]
        n = min(len(vec), self._free_qpos_start)
        self._data.qpos[:n] = vec[:n]
        # normalised [0,1] slide joints (gripper fingers) -> physical range
        for j in range(self._model.njnt):
            adr = self._model.jnt_qposadr[j]
            if adr < n and self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self._model.jnt_range[j]
                self._data.qpos[adr] = lo + self._data.qpos[adr] * (hi - lo)
        # joint equality constraints (coupled fingers)
        for i in range(self._model.neq):
            if self._model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = self._model.jnt_qposadr[self._model.eq_obj1id[i]]
            adr2 = self._model.jnt_qposadr[self._model.eq_obj2id[i]]
            coef = self._model.eq_data[i, :5]
            self._data.qpos[adr2] = np.polyval(coef[::-1], self._data.qpos[adr1])
        for obj in objects:
            address = self._free_qpos_by_body_name.get(obj["name"])
            if address is None:
                continue
            self._data.qpos[address : address + 7] = [*obj["pos"], *obj["wxyz"]]
        mujoco.mj_forward(self._model, self._data)

    @staticmethod
    def _parse_objects(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list) or len(value) > 100:
            raise ValueError("objects must be a list of at most 100 poses")
        objects: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("object pose must be an object")
            name, pos, wxyz = item.get("name"), item.get("pos"), item.get("wxyz")
            if not isinstance(name, str) or not name or len(name) > 128:
                raise ValueError("object pose name is invalid")
            if not isinstance(pos, list) or len(pos) != 3 or not isinstance(wxyz, list) or len(wxyz) != 4:
                raise ValueError("object pose must have pos[3] and wxyz[4]")
            values = [float(component) for component in [*pos, *wxyz]]
            if not all(math.isfinite(component) for component in values):
                raise ValueError("object pose values must be finite")
            quaternion_norm = math.sqrt(sum(component * component for component in values[3:]))
            if quaternion_norm < 1e-8:
                raise ValueError("object pose quaternion must be nonzero")
            objects.append(
                {
                    "name": name,
                    "pos": values[:3],
                    "wxyz": [component / quaternion_norm for component in values[3:]],
                }
            )
        return objects

    @staticmethod
    def _parse_pose_payload(value: Any) -> tuple[List[float], float, List[Dict[str, Any]]]:
        if not isinstance(value, dict):
            raise ValueError("pose payload must be an object")
        raw_joints = value.get("joints")
        if not isinstance(raw_joints, list) or len(raw_joints) > 100:
            raise ValueError("joints must be a list of at most 100 values")
        joints = [float(component) for component in raw_joints]
        gripper = float(value.get("gripper", 0.0))
        if not all(math.isfinite(component) for component in [*joints, gripper]):
            raise ValueError("joint and gripper values must be finite")
        return joints, gripper, ReplayViewer._parse_objects(value.get("objects"))

    # ---- viser scene ----------------------------------------------------------

    def _setup_scene(self, server: Any) -> Dict[int, Any]:
        handles: Dict[int, Any] = {}
        for geom_id in range(self._model.ngeom):
            gtype = self._model.geom_type[geom_id]
            rgba = self._model.geom_rgba[geom_id]
            color = tuple(int(c * 255) for c in rgba[:3])
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_id = self._model.geom_dataid[geom_id]
                v_adr, v_num = self._model.mesh_vertadr[mesh_id], self._model.mesh_vertnum[mesh_id]
                f_adr, f_num = self._model.mesh_faceadr[mesh_id], self._model.mesh_facenum[mesh_id]
                self._mesh_geom_ids.append(geom_id)
                handles[geom_id] = server.scene.add_mesh_simple(
                    f"replay/geom_{geom_id}",
                    self._model.mesh_vert[v_adr : v_adr + v_num].copy(),
                    self._model.mesh_face[f_adr : f_adr + f_num].copy(),
                    color=color,
                    wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    position=np.zeros(3),
                )
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                self._box_geom_ids.append(geom_id)
                handles[geom_id] = server.scene.add_box(
                    f"replay/geom_{geom_id}",
                    color=color,
                    dimensions=tuple(float(v) * 2.0 for v in self._model.geom_size[geom_id]),
                    wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                    position=np.zeros(3),
                )
        return handles

    def _update_scene(self, handles: Dict[int, Any]) -> None:
        for geom_id in self._mesh_geom_ids + self._box_geom_ids:
            h = handles[geom_id]
            h.position = self._data.geom_xpos[geom_id].copy()
            h.wxyz = _mat3_to_wxyz(self._data.geom_xmat[geom_id])

    # ---- pose HTTP endpoint ----------------------------------------------------

    def _start_pose_server(self) -> None:
        pending = self._pending

        class _PoseHandler(BaseHTTPRequestHandler):
            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")

            def _json(self, code: int, obj: Dict[str, Any]) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/status":
                    self._json(200, {"ok": True, "viewer": "replay"})
                else:
                    self._json(404, {"ok": False})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/pose":
                    self._json(404, {"ok": False})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 65_536:
                        raise ValueError("pose body size is invalid")
                    payload = json.loads(self.rfile.read(length))
                    joints, gripper, objects = ReplayViewer._parse_pose_payload(payload)
                except (TypeError, KeyError, ValueError, json.JSONDecodeError):
                    self._json(
                        400,
                        {
                            "ok": False,
                            "error": 'body must contain finite "joints", "gripper", and optional object poses',
                        },
                    )
                    return
                pending["pose"] = (joints, gripper, objects)
                self._json(200, {"ok": True})

            def log_message(self, *args: object) -> None:
                pass

        httpd = ThreadingHTTPServer(("127.0.0.1", self._pose_port), _PoseHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[replay-viewer] Pose endpoint on http://127.0.0.1:{self._pose_port}/pose")

    # ---- main ------------------------------------------------------------------

    def run(self) -> None:
        import viser

        server = viser.ViserServer(port=self._viser_port)
        server.gui.configure_theme(control_layout="collapsible", show_logo=False, show_share_button=False)
        print(f"[replay-viewer] Viewer on http://localhost:{self._viser_port}")

        handles = self._setup_scene(server)
        self._update_scene(handles)

        @server.on_client_connect
        def _set_initial_camera(client: "viser.ClientHandle") -> None:
            client.camera.position = (0.75, -0.55, 0.45)
            client.camera.look_at = (0.3, 0.0, 0.1)

        self._start_pose_server()

        last: Optional[tuple] = None
        while True:
            pose = self._pending["pose"]
            if pose is not None and pose != last:
                last = pose
                self._apply_pose(*pose)
                self._update_scene(handles)
            time.sleep(1.0 / 30.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="View-only 3D replay viewer (viser + MuJoCo FK)")
    ap.add_argument("--scene", default=None, help="scene MJCF (default: newest /tmp/i2rt_*_scene.xml)")
    ap.add_argument("--port", type=int, default=8082, help="viser port")
    ap.add_argument("--pose-port", type=int, default=8083, help="pose HTTP port")
    args = ap.parse_args()
    ReplayViewer(args.scene or _newest_scene(), args.port, args.pose_port).run()


if __name__ == "__main__":
    main()
