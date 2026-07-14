"""Viser control interface for i2rt robots.

Starts in DISABLED (read-only) mode, mirroring the robot's joint state in a
browser-based 3-D viewer.  Once the user confirms visual alignment with the
real robot and clicks "Enable", three control modes become available:

  VIS         — continues mirroring without sending any commands.
  IK control  — drag the 6-DOF target frame to control via IK.
  Joint sliders — per-joint angle sliders (degrees).

A PD-gains panel is shown for robots that expose kp/kd (MotorChainRobot).

See examples/control_with_viser/ for a runnable entry-point and README.
"""

import time
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from i2rt.motor_drivers.dm_driver import PassiveEncoderInfo
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot

# Teaching-handle button indicator visuals (mirrors mujoco_control_interface.py)
_BTN_OFF_RGB = (89, 89, 89)
_BTN_ON_RGB = (26, 230, 26)
_BTN_RADIUS = 0.022
# World-vertical offsets (meters along +Z) above the TCP. Index 0 = SYNC (top), 1 = RECORD (bottom).
_BTN_Z_OFFSETS = [0.10, 0.04]
_BTN_LABELS = ["SYNC", "RECORD"]
_RESET_RAMP_S = 2.0  # seconds to glide from current pose to home on /reset


class ViserControlInterface:
    """Browser-based robot visualiser and controller with a safety gate.

    The robot stays in read-only mode until the user confirms that the 3-D
    model matches the physical robot and presses "Enable".  This prevents
    unexpected motion when the GUI is first opened.
    """

    def __init__(
        self,
        robot: Robot,
        xml_path: str,
        ee_site: str = "grasp_site",
        dt: float = 0.02,
        port: int = 8080,
    ) -> None:
        self._robot = robot
        self._ee_site = ee_site
        self._dt = dt
        self._port = port

        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._kin = Kinematics(xml_path, ee_site)

        self._nq = self._model.nq
        self._n_arm = sum(1 for j in range(self._model.njnt) if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)

        # Free-joint bodies (the sim cube): stepped with real physics in
        # _mirror_robot, ignored by the self-collision check.
        free = [j for j in range(self._model.njnt) if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self._free_qpos_start = min((int(self._model.jnt_qposadr[j]) for j in free), default=self._model.nq)
        self._robot_nv = min((int(self._model.jnt_dofadr[j]) for j in free), default=self._model.nv)
        self._has_free_bodies = bool(free)
        self._free_body_ids = {int(self._model.jnt_bodyid[j]) for j in free}
        self._n_substeps = max(1, round(dt / self._model.opt.timestep))

        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self._ee_site_id == -1:
            available = [mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self._model.nsite)]
            raise ValueError(f"Site {ee_site!r} not found in model. Available: {available}")

        info: Dict[str, Any] = robot.get_robot_info()
        n = robot.num_dofs()
        self._kp: np.ndarray = info.get("kp", np.full(n, 10.0)).copy()
        self._kd: np.ndarray = info.get("kd", np.full(n, 1.0)).copy()
        self._gripper_index: Optional[int] = info.get("gripper_index")
        self._gripper_limits: Optional[np.ndarray] = info.get("gripper_limits")
        self._is_sim: bool = info.get("sim", False)
        # tcp_site is exclusive to the teaching handle; covers sim where motor_chain is absent.
        self._with_teaching_handle: bool = ee_site == "tcp_site" or self._has_teaching_handle(robot)

        # Mesh data — filled by _collect_mesh_geoms()
        self._mesh_geom_ids: List[int] = []
        self._box_geom_ids: List[int] = []
        self._mesh_local_verts: Dict[int, np.ndarray] = {}
        self._mesh_local_faces: Dict[int, np.ndarray] = {}

        self._check_data = mujoco.MjData(self._model)
        self._in_collision = False

    @classmethod
    def from_robot(
        cls,
        robot: MotorChainRobot,
        ee_site: str = "grasp_site",
        dt: float = 0.02,
        port: int = 8080,
    ) -> "ViserControlInterface":
        return cls(robot, robot.xml_path, ee_site, dt, port)

    # ---- MuJoCo helpers -------------------------------------------------------

    def _mirror_robot(self) -> None:
        """Copy robot joint positions into MuJoCo and run forward kinematics."""
        qpos = self._robot.get_joint_pos()
        n = min(len(qpos), self._nq)
        self._data.qpos[:n] = qpos[:n]
        self._denormalize_slide_joints(n)
        self._enforce_eq_constraints()
        if self._has_free_bodies:
            # ponytail: arm is pinned to the robot's reported pose each substep
            # (kinematic pusher); only free bodies integrate. Grasping via
            # friction is weak under teleported contacts — good enough to
            # touch/push; revisit with torque-driven sim if it matters.
            pinned = self._data.qpos[: self._free_qpos_start].copy()
            for _ in range(self._n_substeps):
                self._data.qpos[: self._free_qpos_start] = pinned
                self._data.qvel[: self._robot_nv] = 0.0
                mujoco.mj_step(self._model, self._data)
            self._data.qpos[: self._free_qpos_start] = pinned
            self._data.qvel[: self._robot_nv] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def _denormalize_slide_joints(self, n_set: int) -> None:
        self._denormalize_slide_joints_on(self._data, n_set)

    def _denormalize_slide_joints_on(self, data: mujoco.MjData, n_set: int) -> None:
        """Scale normalised [0,1] slide-joint values to physical range (metres)."""
        for j in range(self._model.njnt):
            adr = self._model.jnt_qposadr[j]
            if adr >= n_set:
                continue
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self._model.jnt_range[j]
                data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)

    def _enforce_eq_constraints(self) -> None:
        self._enforce_eq_constraints_on(self._data)

    def _enforce_eq_constraints_on(self, data: mujoco.MjData) -> None:
        """Project qpos to satisfy joint equality constraints (e.g. coupled fingers)."""
        for i in range(self._model.neq):
            if self._model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = self._model.jnt_qposadr[self._model.eq_obj1id[i]]
            adr2 = self._model.jnt_qposadr[self._model.eq_obj2id[i]]
            coef = self._model.eq_data[i, :5]
            data.qpos[adr2] = np.polyval(coef[::-1], data.qpos[adr1])

    def _has_self_collision(self, target_q: np.ndarray, n: int) -> bool:
        """Return True if *target_q* would cause self-collision.

        Uses a scratch ``MjData`` so the render state is not corrupted.
        Contacts involving the ground plane or adjacent (parent-child) bodies
        are ignored — only unexpected link-link penetrations count.
        """
        self._check_data.qpos[:n] = target_q[:n]
        self._denormalize_slide_joints_on(self._check_data, n)
        self._enforce_eq_constraints_on(self._check_data)
        mujoco.mj_forward(self._model, self._check_data)
        for i in range(self._check_data.ncon):
            c = self._check_data.contact[i]
            if c.dist >= -1e-3:
                continue
            if (
                self._model.geom_type[c.geom1] == mujoco.mjtGeom.mjGEOM_PLANE
                or self._model.geom_type[c.geom2] == mujoco.mjtGeom.mjGEOM_PLANE
            ):
                continue
            b1 = self._model.geom_bodyid[c.geom1]
            b2 = self._model.geom_bodyid[c.geom2]
            if b1 in self._free_body_ids or b2 in self._free_body_ids:
                continue
            if self._model.body_parentid[b1] == b2 or self._model.body_parentid[b2] == b1:
                continue
            return True
        return False

    def _enter_vis_grav_comp(self) -> None:
        """Restore grav-comp on returning to VIS — sim resumes physics, real
        clears any lingering PD command."""
        if hasattr(self._robot, "enable_gravity_comp"):
            self._robot.enable_gravity_comp()
        elif hasattr(self._robot, "enter_gravity_comp_idle"):
            self._robot.enter_gravity_comp_idle()

    def _enter_control_grav_comp(self) -> None:
        """Pause sim grav-comp so CONTROL mode can teleport. On real, the next
        ``command_joint_pos`` implicitly switches to PD — no call needed here."""
        if hasattr(self._robot, "disable_gravity_comp"):
            self._robot.disable_gravity_comp()
        self._in_collision = False

    @staticmethod
    def _mat3_to_wxyz(mat3: np.ndarray) -> np.ndarray:
        """Convert a (3,3) or flat-9 rotation matrix to a wxyz quaternion."""
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, mat3.flatten())
        return q

    @staticmethod
    def _wxyz_to_mat3(wxyz: np.ndarray) -> np.ndarray:
        """Convert a wxyz quaternion to a (3,3) rotation matrix."""
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, wxyz)
        return mat.reshape(3, 3)

    def _ee_pose_4x4(self) -> np.ndarray:
        """Return the end-effector pose as a 4x4 homogeneous matrix."""
        site = self._data.site(self._ee_site_id)
        T = np.eye(4)
        T[:3, 3] = site.xpos.copy()
        T[:3, :3] = site.xmat.reshape(3, 3)
        return T

    # ---- Mesh extraction ------------------------------------------------------

    def _collect_mesh_geoms(self) -> None:
        """Cache per-geom mesh vertex/face arrays in local (geom) coordinates."""
        for geom_id in range(self._model.ngeom):
            if self._model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = self._model.geom_dataid[geom_id]
            v_adr = self._model.mesh_vertadr[mesh_id]
            v_num = self._model.mesh_vertnum[mesh_id]
            f_adr = self._model.mesh_faceadr[mesh_id]
            f_num = self._model.mesh_facenum[mesh_id]
            self._mesh_geom_ids.append(geom_id)
            self._mesh_local_verts[geom_id] = self._model.mesh_vert[v_adr : v_adr + v_num].copy()
            self._mesh_local_faces[geom_id] = self._model.mesh_face[f_adr : f_adr + f_num].copy()

    # ---- Viser scene ----------------------------------------------------------

    def _setup_scene(self, server: Any) -> Dict[int, Any]:
        """Add robot meshes to the viser scene; return {geom_id: mesh_handle}."""
        self._collect_mesh_geoms()
        handles: Dict[int, Any] = {}
        for geom_id in self._mesh_geom_ids:
            rgba = self._model.geom_rgba[geom_id]
            color = tuple(int(c * 255) for c in rgba[:3])
            handles[geom_id] = server.scene.add_mesh_simple(
                f"robot/geom_{geom_id}",
                self._mesh_local_verts[geom_id],
                self._mesh_local_faces[geom_id],
                color=color,
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.zeros(3),
            )
        for geom_id in range(self._model.ngeom):
            if self._model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            rgba = self._model.geom_rgba[geom_id]
            self._box_geom_ids.append(geom_id)
            handles[geom_id] = server.scene.add_box(
                f"robot/geom_{geom_id}",
                color=tuple(int(c * 255) for c in rgba[:3]),
                dimensions=tuple(float(v) * 2.0 for v in self._model.geom_size[geom_id]),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                position=np.zeros(3),
            )
        return handles

    def _update_scene(self, handles: Dict[int, Any]) -> None:
        """Refresh mesh transforms from current MuJoCo forward-kinematics state."""
        for geom_id in self._mesh_geom_ids + self._box_geom_ids:
            h = handles[geom_id]
            h.position = self._data.geom_xpos[geom_id].copy()
            h.wxyz = self._mat3_to_wxyz(self._data.geom_xmat[geom_id])

    # ---- Joint-limit helpers --------------------------------------------------

    def _hinge_joint_ranges_deg(self) -> List[tuple]:
        """Return (lo_deg, hi_deg) for each hinge joint in order."""
        ranges = []
        for j in range(self._model.njnt):
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                lo, hi = self._model.jnt_range[j]
                ranges.append((float(np.degrees(lo)), float(np.degrees(hi))))
        return ranges

    # ---- Teaching-handle helpers ---------------------------------------------

    @staticmethod
    def _has_teaching_handle(robot: Robot) -> bool:
        """Return True if robot has a teaching handle (passive encoder on the same CAN bus)."""
        chain = getattr(robot, "motor_chain", None)
        if chain is None:
            return False
        return (
            hasattr(chain, "get_same_bus_device_states")
            and hasattr(chain, "same_bus_device_driver")
            and chain.same_bus_device_driver is not None
        )

    def _get_teaching_handle_state(self) -> Optional[PassiveEncoderInfo]:
        """Read the latest teaching-handle encoder snapshot, or None if unavailable."""
        if self._is_sim:
            return None
        chain = getattr(self._robot, "motor_chain", None)
        if chain is None or not hasattr(chain, "get_same_bus_device_states"):
            return None
        if getattr(chain, "same_bus_device_driver", None) is None:
            return None
        states = chain.get_same_bus_device_states()
        return states[0] if states else None

    def _get_button_states(self) -> Optional[List[bool]]:
        """Read teaching-handle button states from real hardware, or None if unavailable."""
        state = self._get_teaching_handle_state()
        return list(state.io_inputs) if state is not None else None

    # ---- Main -----------------------------------------------------------------

    def run(self) -> None:
        """Open the viser server and run the visualisation / control loop."""
        import viser  # optional dependency — install with: pip install viser

        server = viser.ViserServer(port=self._port)
        # local patch: minimal chrome so OneShotRobot can embed this as a clean viewport
        server.gui.configure_theme(control_layout="collapsible", show_logo=False, show_share_button=False)
        print(f"[viser] Server started — open http://localhost:{self._port} in your browser")
        print("[viser] Starting in DISABLED (read-only) mode")
        print("[viser] Confirm robot alignment, then click 'Enable Robot'")

        # ---- Scene objects ----------------------------------------------------
        mesh_handles = self._setup_scene(server)
        ee_frame = server.scene.add_frame("ee_frame", axes_length=0.06, axes_radius=0.004)
        ik_ctrl = server.scene.add_transform_controls(
            "/ik_target",
            position=np.zeros(3),
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            scale=0.15,
            visible=False,
        )

        # Teaching-handle button spheres: viser icospheres are immutable in colour, so
        # add an "off" (gray) and "on" (green) sphere per position and toggle visibility.
        # Positions are refreshed each frame to track the TCP along world +Z.
        btn_spheres_off: List[Any] = []
        btn_spheres_on: List[Any] = []
        if self._with_teaching_handle:
            for i in range(len(_BTN_LABELS)):
                btn_spheres_off.append(
                    server.scene.add_icosphere(
                        f"/teaching_handle/btn_{i}_off",
                        radius=_BTN_RADIUS,
                        color=_BTN_OFF_RGB,
                        visible=True,
                    )
                )
                btn_spheres_on.append(
                    server.scene.add_icosphere(
                        f"/teaching_handle/btn_{i}_on",
                        radius=_BTN_RADIUS,
                        color=_BTN_ON_RGB,
                        visible=False,
                    )
                )

        # ---- Initial camera — start zoomed in on the arm ---------------------
        # Viser's default framing sits far back; pull each new client's camera
        # close to the arm so the robot fills the view on first load. Users can
        # still orbit/zoom freely afterwards.
        @server.on_client_connect
        def _set_initial_camera(client: viser.ClientHandle) -> None:
            client.camera.position = (0.55, 0.55, 0.45)
            client.camera.look_at = (0.0, 0.0, 0.2)

        # ---- Shared mutable state (read by loop, written by callbacks) --------
        state: Dict[str, Any] = {"enabled": False, "mode": "vis"}

        n_dofs = self._robot.num_dofs()
        info: Dict[str, Any] = self._robot.get_robot_info()
        has_kpkd = "kp" in info

        # ---- GUI — safety gate -----------------------------------------------
        with server.gui.add_folder("Safety"):
            align_cb = server.gui.add_checkbox("Alignment Confirmed", initial_value=False)
            enable_btn = server.gui.add_button("Enable Robot")
            enable_btn.disabled = True
            status_md = server.gui.add_markdown("**Status:** DISABLED (read-only)")

        # ---- GUI — mode ------------------------------------------------------
        with server.gui.add_folder("Mode"):
            mode_dd = server.gui.add_dropdown(
                "Control mode",
                options=["VIS (mirror)", "IK control", "Joint sliders"],
                initial_value="VIS (mirror)",
            )
            mode_dd.disabled = True

        # ---- GUI — arm joint sliders -----------------------------------------
        joint_ranges = self._hinge_joint_ranges_deg()
        joint_sliders: List[Any] = []
        with server.gui.add_folder("Arm joints (deg)"):
            for i in range(self._n_arm):
                lo, hi = joint_ranges[i] if i < len(joint_ranges) else (-180.0, 180.0)
                s = server.gui.add_slider(f"j{i + 1}", min=lo, max=hi, step=0.1, initial_value=0.0)
                s.disabled = True
                joint_sliders.append(s)

        # ---- GUI — gripper slider --------------------------------------------
        gripper_slider: Optional[Any] = None
        if self._gripper_index is not None and self._gripper_limits is not None:
            with server.gui.add_folder("Gripper"):
                gripper_slider = server.gui.add_slider("Position", min=0.0, max=1.0, step=0.01, initial_value=0.0)
                gripper_slider.disabled = True

        # ---- GUI — teaching-handle indicators -------------------------------
        handle_btn_md: List[Any] = []
        handle_grip_slider: Optional[Any] = None
        if self._with_teaching_handle:
            with server.gui.add_folder("Teaching Handle"):
                for label in _BTN_LABELS:
                    handle_btn_md.append(server.gui.add_markdown(f"**{label}** [○]"))
                handle_grip_slider = server.gui.add_slider(
                    "Gripper Position", min=0.0, max=1.0, step=0.01, initial_value=0.0
                )
                handle_grip_slider.disabled = True

        # ---- GUI — PD gains --------------------------------------------------
        kp_sliders: List[Any] = []
        kd_sliders: List[Any] = []
        apply_btn: Optional[Any] = None
        if has_kpkd:
            with server.gui.add_folder("PD Gains"):
                for i in range(n_dofs):
                    kp_s = server.gui.add_slider(
                        f"kp[{i}]", min=0.0, max=300.0, step=0.5, initial_value=float(self._kp[i])
                    )
                    kd_s = server.gui.add_slider(
                        f"kd[{i}]", min=0.0, max=30.0, step=0.05, initial_value=float(self._kd[i])
                    )
                    kp_sliders.append(kp_s)
                    kd_sliders.append(kd_s)
                apply_btn = server.gui.add_button("Apply Gains")

        # ---- Callbacks -------------------------------------------------------

        @align_cb.on_update
        def _(_: object) -> None:
            enable_btn.disabled = not align_cb.value

        @enable_btn.on_click
        def _(_: object) -> None:
            state["enabled"] = True
            align_cb.disabled = True
            enable_btn.disabled = True
            status_md.content = "**Status:** ENABLED"
            mode_dd.disabled = False
            print("[viser] Robot ENABLED — control active")
            # Sync sliders to current robot positions on enable
            q = self._robot.get_joint_pos()
            for i, s in enumerate(joint_sliders):
                if i < len(q):
                    s.value = float(np.degrees(q[i]))
            if gripper_slider is not None and self._gripper_index is not None:
                gripper_slider.value = float(q[self._gripper_index])

        @mode_dd.on_update
        def _(_: object) -> None:
            sel = mode_dd.value
            if sel == "VIS (mirror)":
                state["mode"] = "vis"
                ik_ctrl.visible = False
                for s in joint_sliders:
                    s.disabled = True
                if gripper_slider is not None:
                    gripper_slider.disabled = True
            elif sel == "IK control":
                state["mode"] = "ik"
                ik_ctrl.visible = True
                for s in joint_sliders:
                    s.disabled = True
                if gripper_slider is not None:
                    gripper_slider.disabled = False
                # Snap IK target to current EE pose
                T = self._ee_pose_4x4()
                ik_ctrl.position = T[:3, 3]
                ik_ctrl.wxyz = self._mat3_to_wxyz(T[:3, :3])
                # Sync gripper slider to current position
                if gripper_slider is not None and self._gripper_index is not None:
                    q = self._robot.get_joint_pos()
                    gripper_slider.value = float(q[self._gripper_index])
            elif sel == "Joint sliders":
                state["mode"] = "joint"
                ik_ctrl.visible = False
                for s in joint_sliders:
                    s.disabled = False
                if gripper_slider is not None:
                    gripper_slider.disabled = False
                # Sync sliders to current robot positions
                q = self._robot.get_joint_pos()
                for i, s in enumerate(joint_sliders):
                    if i < len(q):
                        s.value = float(np.degrees(q[i]))
                if gripper_slider is not None and self._gripper_index is not None:
                    gripper_slider.value = float(q[self._gripper_index])

        if apply_btn is not None:

            @apply_btn.on_click
            def _(_: object) -> None:
                new_kp = np.array([s.value for s in kp_sliders])
                new_kd = np.array([s.value for s in kd_sliders])
                if hasattr(self._robot, "update_kp_kd"):
                    self._robot.update_kp_kd(new_kp, new_kd)
                self._kp = new_kp
                self._kd = new_kd
                print(f"[viser] Gains applied: kp={new_kp.tolist()}, kd={new_kd.tolist()}")

        # ---- Local bridge (OneShotRobot) --------------------------------------
        # Minimal HTTP endpoint so the web UI can request a reset-to-home.
        import io
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from PIL import Image as PILImage

        bridge: Dict[str, Any] = {
            "reset": False,
            "snapshot": None,
            "enable": False,
            "move_to": None,
            "gripper": None,
            "cam_req": {},
            "cam_jpg": {},
            "ee": None,
        }
        is_sim = self._is_sim
        _CAMERAS = ("overhead", "wrist")

        class _BridgeHandler(BaseHTTPRequestHandler):
            def _cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def _json(self, code: int, obj: Dict[str, Any]) -> None:
                self.send_response(code)
                self._cors()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(obj).encode())

            def _body(self) -> Optional[Dict[str, Any]]:
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    parsed = json.loads(self.rfile.read(length))
                    return parsed if isinstance(parsed, dict) else None
                except (ValueError, KeyError):
                    return None

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self._cors()
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/status":
                    body = ('{"enabled": ' + ("true" if state["enabled"] else "false") + ', "mode": "' + state["mode"] + '"}').encode()
                    self.send_response(200)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/joints":
                    snap = bridge["snapshot"]
                    self.send_response(200 if snap is not None else 503)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(snap).encode() if snap is not None else b'{"error": "no data yet"}')
                elif self.path == "/ee_pose":
                    ee = bridge["ee"]
                    self._json(200 if ee is not None else 503, ee if ee is not None else {"error": "no data yet"})
                elif self.path.startswith("/camera/"):
                    name = self.path.rsplit("/", 1)[1]
                    if name not in _CAMERAS:
                        self._json(404, {"error": f"unknown camera {name!r}"})
                        return
                    ev = threading.Event()
                    bridge["cam_req"][name] = ev
                    jpg = bridge["cam_jpg"].get(name) if ev.wait(2.0) else None
                    if jpg:
                        self.send_response(200)
                        self._cors()
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpg)))
                        self.end_headers()
                        self.wfile.write(jpg)
                    else:
                        self._json(503, {"error": "render unavailable"})
                else:
                    self.send_response(404)
                    self._cors()
                    self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                if self.path == "/enable":
                    if not is_sim:
                        self._json(403, {"ok": False, "error": "enable via GUI on real hardware"})
                    else:
                        bridge["enable"] = True
                        self._json(200, {"ok": True})
                elif self.path == "/move_to":
                    body = self._body()
                    try:
                        target = [float(body["x"]), float(body["y"]), float(body["z"])]  # type: ignore[index]
                    except (TypeError, KeyError, ValueError):
                        self._json(400, {"ok": False, "error": "body must be {x, y, z}"})
                        return
                    if not state["enabled"]:
                        self._json(409, {"ok": False, "error": "robot disabled"})
                        return
                    bridge["move_to"] = target
                    self._json(202, {"ok": True})
                elif self.path == "/gripper":
                    body = self._body()
                    try:
                        pos = float(body["position"])  # type: ignore[index]
                    except (TypeError, KeyError, ValueError):
                        self._json(400, {"ok": False, "error": "body must be {position: 0..1}"})
                        return
                    if not state["enabled"]:
                        self._json(409, {"ok": False, "error": "robot disabled"})
                        return
                    bridge["gripper"] = float(np.clip(pos, 0.0, 1.0))
                    self._json(200, {"ok": True})
                elif self.path == "/reset":
                    bridge["reset"] = True
                    ok = state["enabled"]
                    self.send_response(200 if ok else 409)
                    self._cors()
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"ok": true}' if ok else b'{"ok": false, "error": "robot disabled"}')
                else:
                    self.send_response(404)
                    self._cors()
                    self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        try:
            _bridge_srv = ThreadingHTTPServer(("127.0.0.1", self._port + 1), _BridgeHandler)
            threading.Thread(target=_bridge_srv.serve_forever, daemon=True).start()
            print(f"[bridge] Reset endpoint on http://127.0.0.1:{self._port + 1}")
        except OSError:
            print("[bridge] Port busy — bridge disabled")

        # ---- Main loop -------------------------------------------------------
        prev_controlled = False
        reset_anim: dict | None = None
        renderer: Optional[mujoco.Renderer] = None
        try:
            while True:
                if bridge["enable"]:
                    bridge["enable"] = False
                    if not state["enabled"]:
                        state["enabled"] = True
                        align_cb.disabled = True
                        enable_btn.disabled = True
                        status_md.content = "**Status:** ENABLED"
                        mode_dd.disabled = False
                        print("[bridge] Robot ENABLED via bridge")
                        q = self._robot.get_joint_pos()
                        for i, s in enumerate(joint_sliders):
                            if i < len(q):
                                s.value = float(np.degrees(q[i]))
                        if gripper_slider is not None and self._gripper_index is not None:
                            gripper_slider.value = float(q[self._gripper_index])

                move_target = bridge["move_to"]
                bridge["move_to"] = None
                if move_target is not None and state["enabled"]:
                    if mode_dd.value != "IK control":
                        mode_dd.value = "IK control"  # fires on_update: gizmo shown, snapped to current EE
                    ik_ctrl.position = np.array(move_target)
                    print(f"[bridge] move_to {move_target}")

                grip_target = bridge["gripper"]
                bridge["gripper"] = None
                if grip_target is not None and state["enabled"]:
                    if mode_dd.value != "IK control":
                        mode_dd.value = "IK control"
                    if gripper_slider is not None:
                        gripper_slider.value = grip_target

                if bridge["reset"]:
                    bridge["reset"] = False
                    if state["enabled"]:
                        mode_dd.value = "Joint sliders"
                        q = self._robot.get_joint_pos()
                        reset_anim = {
                            "step": 0,
                            "steps": max(1, int(_RESET_RAMP_S / self._dt)),
                            "start_deg": [float(np.degrees(q[i])) if i < len(q) else 0.0 for i in range(len(joint_sliders))],
                            "grip_start": float(q[self._gripper_index]) if gripper_slider is not None and self._gripper_index is not None else 0.0,
                        }
                        print(f"[bridge] Ramping to home pose over {_RESET_RAMP_S}s")
                    else:
                        print("[bridge] Reset ignored — robot disabled")

                if reset_anim is not None:
                    if not state["enabled"] or state["mode"] != "joint":
                        reset_anim = None  # user took over — stop ramping
                    else:
                        reset_anim["step"] += 1
                        a = min(1.0, reset_anim["step"] / reset_anim["steps"])
                        # ponytail: linear ramp of slider targets; PD smooths the rest
                        for i, s in enumerate(joint_sliders):
                            s.value = reset_anim["start_deg"][i] * (1.0 - a)
                        if gripper_slider is not None:
                            gripper_slider.value = reset_anim["grip_start"] * (1.0 - a)
                        if a >= 1.0:
                            reset_anim = None
                            print("[bridge] Home pose reached")
                self._mirror_robot()
                self._update_scene(mesh_handles)

                # Publish latest joint snapshot for the bridge's GET /joints
                q_now = self._robot.get_joint_pos()
                bridge["snapshot"] = {
                    "t": time.time(),
                    "joints": [float(v) for v in q_now[: self._n_arm]],
                    "gripper": float(q_now[self._gripper_index]) if self._gripper_index is not None else 0.0,
                }

                # Update EE frame indicator
                T = self._ee_pose_4x4()
                ee_frame.position = T[:3, 3]
                ee_frame.wxyz = self._mat3_to_wxyz(T[:3, :3])
                bridge["ee"] = {
                    "t": time.time(),
                    "pos": [float(v) for v in T[:3, 3]],
                    "wxyz": [float(v) for v in self._mat3_to_wxyz(T[:3, :3])],
                }

                # Service camera snapshot requests (render on this thread — GL context affinity)
                if bridge["cam_req"]:
                    if renderer is None:
                        renderer = mujoco.Renderer(self._model, height=480, width=640)
                    for cam_name, ev in list(bridge["cam_req"].items()):
                        bridge["cam_req"].pop(cam_name, None)
                        try:
                            renderer.update_scene(self._data, camera=cam_name)
                            buf = io.BytesIO()
                            PILImage.fromarray(renderer.render()).save(buf, "JPEG", quality=85)
                            bridge["cam_jpg"][cam_name] = buf.getvalue()
                        except Exception as exc:
                            print(f"[bridge] camera {cam_name} render failed: {exc}")
                            bridge["cam_jpg"][cam_name] = None
                        ev.set()

                if self._with_teaching_handle:
                    handle_state = self._get_teaching_handle_state()
                    buttons = list(handle_state.io_inputs) if handle_state is not None else [False, False]
                    for i, md in enumerate(handle_btn_md):
                        pressed = bool(buttons[i]) if i < len(buttons) else False
                        marker = "[●]" if pressed else "[○]"
                        md.content = f"**{_BTN_LABELS[i]}** {marker}"
                    ee_pos = T[:3, 3]
                    for i in range(len(btn_spheres_off)):
                        pressed = bool(buttons[i]) if i < len(buttons) else False
                        sphere_pos = ee_pos + np.array([0.0, 0.0, _BTN_Z_OFFSETS[i]])
                        btn_spheres_off[i].position = sphere_pos
                        btn_spheres_on[i].position = sphere_pos
                        btn_spheres_off[i].visible = not pressed
                        btn_spheres_on[i].visible = pressed
                    if handle_grip_slider is not None and handle_state is not None:
                        handle_grip_slider.value = float(np.clip(1.0 - float(handle_state.position), 0.0, 1.0))

                mode = state["mode"]
                controlled = state["enabled"] and mode in ("ik", "joint")
                if controlled != prev_controlled:
                    if controlled:
                        self._enter_control_grav_comp()
                    else:
                        self._enter_vis_grav_comp()
                    prev_controlled = controlled

                if not state["enabled"]:
                    # Read-only: update sliders to reflect live robot state
                    q = self._robot.get_joint_pos()
                    for i, s in enumerate(joint_sliders):
                        if i < len(q):
                            s.value = float(np.degrees(q[i]))
                    if gripper_slider is not None and self._gripper_index is not None:
                        gripper_slider.value = float(q[self._gripper_index])

                elif mode == "vis":
                    # Mirror only — no commands
                    q = self._robot.get_joint_pos()
                    for i, s in enumerate(joint_sliders):
                        if i < len(q):
                            s.value = float(np.degrees(q[i]))

                elif mode == "ik":
                    # Build target from user-dragged transform control
                    target = np.eye(4)
                    target[:3, 3] = np.asarray(ik_ctrl.position)
                    target[:3, :3] = self._wxyz_to_mat3(np.asarray(ik_ctrl.wxyz))
                    init_q = self._data.qpos[: self._nq].copy()
                    _, ik_q = self._kin.ik(target, self._ee_site, init_q=init_q)
                    cmd = self._robot.get_joint_pos().copy()
                    cmd[: self._n_arm] = ik_q[: self._n_arm]
                    if gripper_slider is not None and self._gripper_index is not None:
                        cmd[self._gripper_index] = float(gripper_slider.value)
                    n = min(len(cmd), self._nq)
                    if self._has_self_collision(cmd, n):
                        if not self._in_collision:
                            print("[viser] Collision detected — command blocked")
                            self._in_collision = True
                    else:
                        self._robot.command_joint_pos(cmd)
                        if self._in_collision:
                            print("[viser] Collision cleared — commands resumed")
                            self._in_collision = False
                    # Reflect solved angles in sliders
                    for i, s in enumerate(joint_sliders):
                        if i < self._n_arm:
                            s.value = float(np.degrees(ik_q[i]))

                elif mode == "joint":
                    # Build command from slider values
                    cmd = self._robot.get_joint_pos().copy()
                    for i, s in enumerate(joint_sliders):
                        if i < self._n_arm:
                            cmd[i] = float(np.radians(s.value))
                    if gripper_slider is not None and self._gripper_index is not None:
                        cmd[self._gripper_index] = float(gripper_slider.value)
                    n = min(len(cmd), self._nq)
                    if self._has_self_collision(cmd, n):
                        if not self._in_collision:
                            print("[viser] Collision detected — command blocked")
                            self._in_collision = True
                    else:
                        self._robot.command_joint_pos(cmd)
                        if self._in_collision:
                            print("[viser] Collision cleared — commands resumed")
                            self._in_collision = False

                time.sleep(self._dt)

        except KeyboardInterrupt:
            pass

        print("[viser] Stopped")
