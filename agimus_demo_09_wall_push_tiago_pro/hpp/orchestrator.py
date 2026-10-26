"""
HPP orchestrator for the TIAGo Pro wall-push demo -- fixed base (right arm only).

Same flow as demo 07's deburring orchestrator (approach p1, insertion p2, retraction
p3, retreat p4, played through the MPC), simplified: no mocap, no calibration. The
"wall" is the demo-07 pylone at the same pose (config/pylone_pose.yaml), same handle
and hole (config/hpp_orchestrator_params.yaml `handle:`); p2 is followed by the
push into the part (`contact:` block).

    o = Orchestrator()
    o.sync_from_robot()  # q_init from the robot's joint state (do this before plan())
    o.plan()             # HPP: qpg, qg, p1 (approach), p2 (insertion), p3, p4
    o.init_viewer()      # viser, http://localhost:8080 : robot + pylone
    o.play(o.p1)         # animate a path in viser
    o.activate_lfc()     # switch to torque control (MPC + LFC)
    o.execute([o.p1])    # sample + publish to the MPC; o.execute() = p1..p4
    o.plan_and_execute() # both in sequence

Run via orchestrator_node.py, from the tiago_pro_nix devshell.
"""

import os
import shutil
import signal
import subprocess
import sys
import glob
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
import yaml

# ── Make agimus_msgs / rclpy importable from HPP environment ──────────────────
for _p in sorted(
    glob.glob("/home/gepetto/ros2_ws/install/*/lib/python3*/site-packages")
    + glob.glob("/home/gepetto/agimus_deps_ws/install/*/lib/python3*/site-packages")
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pyhpp.manipulation import Device, urdf  # noqa: E402
from pyhpp.manipulation import Graph, Problem, TransitionPlanner  # noqa: E402
from pyhpp.manipulation.constraint_graph_factory import ConstraintGraphFactory  # noqa: E402
from pyhpp.manipulation.security_margins import SecurityMargins  # noqa: E402
from pyhpp.constraints import ComparisonType, ComparisonTypes, LockedJoint  # noqa: E402
from pyhpp.core import RandomShortcut, SplineGradientBased_bezier3  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy  # noqa: E402
from agimus_msgs.msg import MpcInput, MpcEEInput  # noqa: E402
from control_msgs.msg import DynamicJointState  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────────

_HPP_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.join(_HPP_DIR, "..")
ROBOT_SRDF = os.path.join(_HPP_DIR, "tiago_pro.srdf")
PYLONE_SRDF = os.path.join(_HPP_DIR, "pylone.srdf")
PYLONE_URDF = os.path.join(_PKG_DIR, "urdf", "pylone.urdf")
GROUND_SRDF = os.path.join(_HPP_DIR, "ground.srdf")
GROUND_URDF = os.path.join(_PKG_DIR, "urdf", "ground.urdf")

_CFG_FILE = os.path.join(_PKG_DIR, "config", "hpp_orchestrator_params.yaml")
with open(_CFG_FILE) as _f:
    _cfg = yaml.safe_load(_f)

DT = _cfg["trajectory"]["dt"]
TIME_SCALE = _cfg["trajectory"]["time_scale"]

_PYLONE_POSE_FILE = os.path.join(_PKG_DIR, "config", "pylone_pose.yaml")

HANDLE_NAME = _cfg["handle"]["name"]

LEFT_ARM_TUCK = _cfg["tuck"]["left_arm"]
RIGHT_ARM_TUCK = _cfg["tuck"]["right_arm"]

def _reload_weights():
    """Re-read the MPC weights from hpp_orchestrator_params.yaml into the module
    globals. Called at the top of execute() so tuning the yaml doesn't need an
    orchestrator restart."""
    global W_Q, W_QDOT, W_QDDOT, W_EFFORT, W_COLLISION, W_FRAME_TRANS, W_FRAME_ROT, W_NULLSPACE_POS
    with open(_CFG_FILE) as f:
        w = yaml.safe_load(f)["weights"]
    W_Q = np.array(w["w_q"])
    W_QDOT = np.array(w["w_qdot"])
    W_QDDOT = np.array(w["w_qddot"])
    W_EFFORT = np.array(w["w_effort"])
    W_COLLISION = w["w_collision"]
    W_FRAME_TRANS = np.array(w["w_frame_trans"])
    W_FRAME_ROT = np.array(w["w_frame_rot"])
    # Default 0.0 (not 30.0, the OCP yaml's own fallback): an orchestrator
    # config predating this key shouldn't silently turn the cost on.
    W_NULLSPACE_POS = float(w.get("w_nullspace_pos", 0.0))
    print(
        f"  weights: w_q={W_Q[0]:g} w_qdot={W_QDOT[0]:g} w_qddot={W_QDDOT[0]:g} "
        f"w_effort={W_EFFORT[0]:g} w_frame_trans={W_FRAME_TRANS[0]:g} "
        f"w_frame_rot={W_FRAME_ROT[0]:g} w_nullspace_pos={W_NULLSPACE_POS:g}"
    )


W_Q = W_QDOT = W_QDDOT = W_EFFORT = W_COLLISION = W_FRAME_TRANS = W_FRAME_ROT = None
W_NULLSPACE_POS = None
_reload_weights()

_c = _cfg["contact"]
FORCE_FRAME_ID = _c["frame_id"]
# Deep insert (see hpp_orchestrator_params.yaml `contact:` and
# _append_deep_insert()): after p2, continue the tool PRESS_DEPTH into the
# part via a joint-space ramp to an IK-solved qg_deep. Force handling is
# deferred — w_force_marker / press_w_frame_trans in the yaml are unused
# this iteration.
PRESS_DEPTH = _c["press_depth_m"]
PRESS_RAMP_S = _c["press_ramp_s"]
PRESS_HOLD_S = _c["press_hold_s"]
POST_P2_HOLD_S = _c.get("post_p2_hold_s", 0.0)  # dwell at qg before pushing deeper

# ── Optional rosbag recording around execute() ───────────────────────────────
# Opt in per-call with execute(record=True) (or record="sometag"), or set
# o.record = True once for the session. Bags land in <source>/plot/runs/ and
# feed plot/plot_force_profile.py directly — no second terminal needed.
#
# realpath (not _PKG_DIR): `plot/` is NOT in CMakeLists' INSTALL_TO_SHARE, so
# under --symlink-install the installed orchestrator.py is a symlink back to
# the source tree — resolving it lands the runs next to plot_force_profile.py
# instead of in an install-space `plot/` that nothing else looks at.
_SRC_PKG_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_RUNS_DIR = os.path.join(_SRC_PKG_DIR, "plot", "runs")
_PLOT_SCRIPT = os.path.join(_SRC_PKG_DIR, "plot", "plot_force_profile.py")
RECORD_TOPICS = [
    "/mpc_input",           # reference trajectory + target force ramp (ee_inputs[FT].force)
    "/sensor_with_force",   # measured state + force (contacts[FT].wrench) + contact.active
    "/sensor",              # 1 kHz raw robot state published by the LFC
    "/ocp_x0",              # augmented x0 actually fed to the solver
    "/control",             # commanded feedforward torque + Riccati gain, and
                            # (since the next_states refactor) xs[k+1] for the
                            # LFC's u2/u3 reference interpolation -- no longer
                            # a separate /mpc_x_next topic.
    "/lfc_debug",           # LFC 1 kHz internals: alpha, q/v_ref, u_ff,
                            # u_fb raw/filt, u_cmd, force_blend, u_fb_force
    "/ocp_solve_time",      # OCP solve compute time
    "/mpc_debug",           # KKT norm, solver iters
]


class _TrajectoryPublisherNode(Node):
    """One-shot ROS2 node that publishes pre-computed MpcInput messages.

    Publishing is driven by an explicit deadline loop in Orchestrator.execute()
    (see there), NOT a rclpy timer: the old ``create_timer(DT)`` + ``spin_once
    + time.sleep(DT)`` combo actually ran at ~DT + overhead ≈ 20 ms → 49 Hz
    instead of 100 Hz, so the reference trajectory played at half speed and the
    OCP over-predicted joint velocity ~2x (see the phantom-velocity notes).
    """

    def __init__(self, messages: list):
        super().__init__("hpp_trajectory_publisher")
        self._messages = messages
        self._idx = 0
        qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._pub = self.create_publisher(MpcInput, "mpc_input", qos)
        self.get_logger().info(
            f"Publishing {len(self._messages)} trajectory points at {1 / DT:.0f} Hz …"
        )
        self._done = False

    def _publish_next(self):
        if self._idx >= len(self._messages):
            if not self._done:
                self.get_logger().info("Trajectory fully published.")
                self._done = True
            return
        self._pub.publish(self._messages[self._idx])
        self._idx += 1


class Orchestrator:
    """
    Interactive orchestrator for HPP deburring planning and MPC execution.
    Fixed-base version: only the right arm (7 DOF) is actuated.

    Usage (in IPython):
        o = Orchestrator()
        o.plan()
        o.execute()
    """

    def __init__(self, ros_node: Node = None, urdf_file: str = None):
        """urdf_file: OFFLINE mode -- load the robot model from this saved URDF instead of
        /robot_description, and refuse everything that talks to the robot (sync, execute,
        LFC switch). See scripts/save_urdf.py."""
        self._ros_node = ros_node
        self._urdf_file = urdf_file
        self.p1 = self.p2 = self.p3 = self.p4 = None
        self._viewer = None  # viser state, see init_viewer()
        self._messages = None
        # Set True to auto-record every execute() run (see RECORD_TOPICS /
        # execute(record=...)). Per-call `record=` overrides this.
        self.record = False
        # Seconds to keep `ros2 bag record` running after the trajectory
        # finishes publishing, before SIGINT -- so a manual disturbance done
        # right after the motion (e.g. pushing the end-effector) lands in the
        # same bag. Per-call `record_linger=` overrides this. Ctrl-C during
        # the linger stops it early.
        self.record_linger = 0.0

        print("Loading HPP model …")
        self._setup_model()
        print("Building constraint graph …")
        self._setup_graph()
        print("Orchestrator ready.  Call plan() to start.\n")

    # ── Model setup ───────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_robot_urdf(timeout: float = 10.0) -> str:
        """Return the URDF string from /robot_description (transient_local topic)."""
        from std_msgs.msg import String

        node = rclpy.create_node("_hpp_urdf_fetcher")
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        urdf_str = None

        def _cb(msg):
            nonlocal urdf_str
            urdf_str = msg.data

        node.create_subscription(String, "/robot_description", _cb, qos)
        t0 = time.time()
        while urdf_str is None and time.time() - t0 < timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_node()

        if urdf_str is None:
            raise RuntimeError(
                f"Timed out after {timeout}s waiting for /robot_description. "
                "Is the simulation running?"
            )
        return urdf_str

    @staticmethod
    def _strip_sc_capsules(urdf_str: str) -> str:
        """Remove the '_link_sc' simplified capsules from the URDF.

        These capsules exist for the MPC's soft, distance-based collision
        avoidance and are deliberately oversized for that purpose — they are
        not meant to be hard pass/fail collision geometry for the HPP
        planner, which should check against the real link geometry instead.
        """
        root = ET.fromstring(urdf_str)
        for link in list(root.findall("link")):
            if link.get("name", "").endswith("_link_sc"):
                root.remove(link)
        for joint in list(root.findall("joint")):
            child = joint.find("child")
            if child is not None and child.get("link", "").endswith("_link_sc"):
                root.remove(joint)
        return ET.tostring(root, encoding="unicode")

    def _setup_model(self):
        if self._urdf_file:
            with open(self._urdf_file) as _uf:
                urdf_str = _uf.read()
            print(f"OFFLINE mode: robot model from {self._urdf_file} (no robot access).")
        else:
            urdf_str = self._fetch_robot_urdf()
        urdf_str = self._strip_sc_capsules(urdf_str)
        self._urdf_str = urdf_str  # for the viser viewer
        _tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        _tmp.write(urdf_str)
        _tmp.close()

        # Inject 3 cm deburring tool collision geometry at the EE.
        # The tool extends along -Z of gripper_right_tool_holder (= into the hole at grasp).
        # At qg: tool tip is 1 cm inside the hole face.  At qpg: tool tip is 3 cm outside.
        _tool_snippet = (
            '<link name="deburring_tool">'
            '<collision><origin xyz="0 0 -0.015" rpy="0 0 0"/>'
            '<geometry><cylinder radius="0.008" length="0.03"/></geometry>'
            "</collision></link>"
            '<joint name="deburring_tool_joint" type="fixed">'
            '<parent link="gripper_right_tool_holder"/>'
            '<child link="deburring_tool"/>'
            '<origin xyz="0 0 0" rpy="0 0 0"/>'
            "</joint>"
        )
        with open(_tmp.name, "r") as _f:
            _urdf_with_tool = _f.read().replace(
                "</robot>", _tool_snippet + "\n</robot>"
            )
        with open(_tmp.name, "w") as _f:
            _f.write(_urdf_with_tool)

        robot = Device("tiago_pro")
        # anchor = fixed base, only arm joints have DOF
        urdf.loadModel(
            robot,
            0,
            "tiago_pro",
            "anchor",
            f"file://{_tmp.name}",
            ROBOT_SRDF,
            pin.SE3.Identity(),
        )
        os.unlink(_tmp.name)
        urdf.loadModel(
            robot,
            0,
            "ground",
            "anchor",
            GROUND_URDF,
            GROUND_SRDF,
            pin.SE3.Identity(),
        )
        urdf.loadModel(
            robot,
            0,
            "pylone",
            "freeflyer",
            PYLONE_URDF,
            PYLONE_SRDF,
            pin.SE3.Identity(),
        )

        self.robot = robot
        model = robot.model()
        self.model = model

        def _idx(name):
            return model.joints[model.getJointId(name)].idx_q

        self._left_arm_idx = _idx("tiago_pro/arm_left_1_joint")
        self._right_arm_idx = _idx("tiago_pro/arm_right_1_joint")
        self._pylone_idx = _idx("pylone/root_joint")

        with open(_PYLONE_POSE_FILE) as _pf:
            _pc = yaml.safe_load(_pf)
        _px = _pc["pylone_x"]
        _py = _pc["pylone_y"]
        _pz = _pc["pylone_z"]
        _pq = _pc.get("pylone_quat", [0.0, 0.0, 0.0, 1.0])

        self._set_pylone_bounds(_px, _py, _pz)

        _handle = robot.handles()[HANDLE_NAME]
        _R = np.array(
            [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        )  # Rx(-90°): handle Z = world +Y
        _handle.localPosition = pin.SE3(_R, _handle.localPosition.translation)
        _handle.mask = [True, True, True, True, True, True]
        _handle.approachingDirection = np.array([0, 0, 1])

        li = self._left_arm_idx
        ri = self._right_arm_idx
        pi = self._pylone_idx

        self.q_init = pin.neutral(model).copy()
        self._left_arm_lock_values = list(LEFT_ARM_TUCK)
        self.q_init[li : li + 7] = LEFT_ARM_TUCK
        self.q_init[ri : ri + 7] = RIGHT_ARM_TUCK
        self.q_init[pi : pi + 3] = [_px, _py, _pz]
        self.q_init[pi + 3 : pi + 7] = _pq

        self._pin_data = model.createData()
        ee_frame_name = "tiago_pro/gripper_right_tool_holder"
        self._ee_frame_id = model.getFrameId(ee_frame_name)
        if self._ee_frame_id == model.nframes:
            raise RuntimeError(f"Frame '{ee_frame_name}' not found in model")

    def _set_pylone_bounds(self, x, y, z, margin: float = 0.001):
        """Lock pylone position with tight bounds so HPP cannot move it."""
        self.robot.setJointBounds(
            "pylone/root_joint",
            [
                x - margin,
                x + margin,
                y - margin,
                y + margin,
                z - margin,
                z + margin,
                -float("Inf"),
                float("Inf"),
                -float("Inf"),
                float("Inf"),
                -float("Inf"),
                float("Inf"),
                -float("Inf"),
                float("Inf"),
            ],
        )

    # ── Constraint graph setup ─────────────────────────────────────────────────

    def _setup_graph(self):
        robot = self.robot
        model = self.model

        problem = Problem(robot)
        graph = Graph("robot", robot, problem)
        factory = ConstraintGraphFactory(graph)
        graph.maxIterations(40)
        graph.errorThreshold(1e-3)
        factory.setGrippers(["tiago_pro/gripper"])
        factory.setObjects(["pylone"], [[HANDLE_NAME]], [[]])
        factory.generate()

        self._transition_approach = graph.getTransition(
            f"tiago_pro/gripper > {HANDLE_NAME} | f_01"
        )
        self._transition_insert = graph.getTransition(
            f"tiago_pro/gripper > {HANDLE_NAME} | f_12"
        )

        _cts = ComparisonTypes()
        _cts[:] = [ComparisonType.EqualToZero]

        def _lock(joint_name, value):
            j = model.joints[model.getJointId(joint_name)]
            if j.nq == 2 and j.nv == 1:
                locked_val = np.array([np.cos(value), np.sin(value)])
            else:
                locked_val = np.array([value])
            return LockedJoint(robot, joint_name, locked_val, _cts)

        locked = []
        locked.append(_lock("tiago_pro/torso_lift_joint", 0.0))
        for wheel in [
            "wheel_front_left_joint",
            "wheel_front_right_joint",
            "wheel_rear_left_joint",
            "wheel_rear_right_joint",
        ]:
            locked.append(_lock(f"tiago_pro/{wheel}", 0.0))
        for i, val in enumerate(self._left_arm_lock_values):
            locked.append(_lock(f"tiago_pro/arm_left_{i + 1}_joint", val))
        for name in [
            "gripper_left_finger_joint",
            "gripper_left_inner_finger_left_joint",
            "gripper_left_fingertip_left_joint",
            "gripper_left_inner_finger_right_joint",
            "gripper_left_fingertip_right_joint",
            "gripper_left_outer_finger_right_joint",
            "gripper_right_tool_mount_joint",
        ]:
            locked.append(_lock(f"tiago_pro/{name}", 0.0))
        locked.append(_lock("tiago_pro/head_1_joint", 0.0))
        locked.append(_lock("tiago_pro/head_2_joint", 0.0))

        graph.addNumericalConstraintsToGraph(locked)

        sm = SecurityMargins(problem, factory, ["tiago_pro", "pylone"], robot)
        sm.setSecurityMarginBetween("tiago_pro", "pylone", 0.02)
        sm.apply()

        # Insert: no margin at all (gripper enters the hole)
        for jname in model.names:
            if "tiago_pro" in jname:
                graph.setSecurityMarginForTransition(
                    self._transition_insert, jname, "pylone/root_joint", float("-inf")
                )

        # Approach: gripper/tool need to get closer than the 2 cm global margin to reach qpg.
        # Keep 2 cm for arm joints; use 1 cm for gripper/tool so the path stays finite.
        for jname in model.names:
            if "tiago_pro" in jname and (
                "gripper_right" in jname or "deburring_tool" in jname
            ):
                graph.setSecurityMarginForTransition(
                    self._transition_approach, jname, "pylone/root_joint", 0.01
                )

        graph.initialize()

        self.problem = problem
        self.graph = graph

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(self, max_attempts: int = 50) -> bool:
        """Generate qpg (collision-free), qg, and plan p1, p2, p3, p4."""
        shooter = self.problem.configurationShooter()
        qpg = None
        n_ik_fail = 0
        n_collision_fail = 0
        collision_pair_counts = {}
        for i in range(max_attempts):
            q = shooter.shoot()
            res, q_cand, err = self.graph.generateTargetConfig(
                self._transition_approach, self.q_init, q
            )
            if not res:
                n_ik_fail += 1
                continue
            pv = self._transition_approach.pathValidation()
            res, report = pv.validateConfiguration(q_cand)
            if not res:
                n_collision_fail += 1
                key = str(report)
                collision_pair_counts[key] = collision_pair_counts.get(key, 0) + 1
                continue
            qpg = q_cand
            print(f"  qpg found at attempt {i}, err={err:.2e}")
            break

        if qpg is None:
            print(f"Failed to find collision-free qpg in {max_attempts} attempts.")
            print(f"  IK/reachability failures: {n_ik_fail}")
            print(f"  Collision failures:       {n_collision_fail}")
            for key, count in sorted(
                collision_pair_counts.items(), key=lambda kv: -kv[1]
            ):
                print(f"    [{count}x] {key}")
            return False

        self.qpg = qpg

        res, qg, err = self.graph.generateTargetConfig(
            self._transition_insert, qpg, qpg
        )
        print(f"  qg: res={res}, err={err:.2e}")
        if not res:
            print("Failed to generate qg.")
            return False

        self.problem.constraintGraph(self.graph)
        planner = TransitionPlanner(self.problem)
        planner.maxIterations(5000)

        planner.setTransition(self._transition_approach)
        q_goal = np.zeros((1, self.robot.configSize()), order="F")
        q_goal[0, :] = qpg
        print("Planning p1 (approach) …")
        p1 = planner.planPath(self.q_init, q_goal, True)
        print("  p1 found.")

        shortcut = RandomShortcut(self.problem)
        spline_opt = SplineGradientBased_bezier3(self.problem)

        try:
            for i in range(3):
                p1_new = shortcut.optimize(p1)
                tr_before = p1.timeRange()
                tr_after = p1_new.timeRange()
                dt = (tr_before.second - tr_before.first) - (
                    tr_after.second - tr_after.first
                )
                p1 = p1_new
                print(
                    f"  p1 shortcut pass {i + 1}/3: {tr_after.second - tr_after.first:.2f} s  (−{dt:.2f} s)"
                )
                if dt < 1e-3:
                    break
        except Exception as e:
            print(f"  p1 shortcut failed: {e}")
        try:
            p1 = spline_opt.optimize(p1)
            tr = p1.timeRange()
            print(f"  p1 spline: {tr.second - tr.first:.2f} s")
        except Exception as e:
            print(f"  p1 spline optimisation failed: {e}")

        planner.setTransition(self._transition_insert)
        q_goal[0, :] = qg
        print("Planning p2 (insertion) …")
        p2 = planner.planPath(qpg, q_goal, True)
        print("  p2 found.")

        try:
            p2 = shortcut.optimize(p2)
            tr = p2.timeRange()
            print(f"  p2 shortcut: {tr.second - tr.first:.2f} s")
        except Exception as e:
            print(f"  p2 shortcut failed: {e}")
        try:
            p2 = spline_opt.optimize(p2)
            tr = p2.timeRange()
            print(f"  p2 spline: {tr.second - tr.first:.2f} s")
        except Exception as e:
            print(f"  p2 spline optimisation failed: {e}")

        p3 = p2.reverse()
        print("  p3 ready (retraction, reversed from optimised p2).")
        p4 = p1.reverse()
        print("  p4 ready (retreat, reversed from optimised p1).")

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.qpg = qpg
        self.qg = qg
        return True

    # ── Path sampling ─────────────────────────────────────────────────────────

    def _extract_active_q(self, q_full):
        """Extract the 7 right-arm joint positions from a full HPP config."""
        q = np.array(q_full)
        ri = self._right_arm_idx
        return q[ri : ri + 7].copy()

    def _active_velocity(self, q1, q2, dt):
        """Finite-difference velocity for the 7 right-arm joints."""
        return (q2 - q1) / dt

    def _sample_path(self, path):
        tr = path.timeRange()
        t_min, t_max = tr.first, tr.second
        n = max(2, int((t_max - t_min) * TIME_SCALE / DT))
        times = np.linspace(t_min, t_max, n)
        q_list = [self._extract_active_q(path.eval(t)[0]) for t in times]
        q_arr = np.array(q_list)

        dq_list = [
            self._active_velocity(q_arr[i], q_arr[i + 1], DT)
            for i in range(len(q_arr) - 1)
        ]
        dq_list.append(dq_list[-1])
        dq_arr = np.array(dq_list)

        ddq_list = [(dq_arr[i + 1] - dq_arr[i]) / DT for i in range(len(dq_arr) - 1)]
        ddq_list.append(ddq_list[-1])
        ddq_arr = np.array(ddq_list)

        return q_arr, dq_arr, ddq_arr

    def _fk_ee(self, q_arm: np.ndarray) -> pin.SE3:
        q_full = pin.neutral(self.model)
        ri = self._right_arm_idx
        q_full[ri : ri + 7] = q_arm
        pin.forwardKinematics(self.model, self._pin_data, q_full)
        pin.updateFramePlacements(self.model, self._pin_data)
        return self._pin_data.oMf[self._ee_frame_id].copy()

    def _ik_ee(
        self, T_target: pin.SE3, q_seed: np.ndarray, iters: int = 60, tol: float = 1e-5
    ) -> np.ndarray:
        """Damped least-squares IK for the 7 right-arm joints placing
        _ee_frame_id at T_target. q_seed / return are the 7 arm values. Used
        for a tiny (~cm) perturbation of a known config — converges fast."""
        q_full = pin.neutral(self.model)
        ri = self._right_arm_idx  # config (nq) index
        vi = self.model.joints[
            self.model.getJointId("tiago_pro/arm_right_1_joint")
        ].idx_v  # velocity (nv) index — differs from ri if a multi-dof joint (pylone
        #          free-flyer) precedes the arm in the model
        q_lo = self.model.lowerPositionLimit[ri : ri + 7]
        q_hi = self.model.upperPositionLimit[ri : ri + 7]
        q = np.clip(np.array(q_seed, dtype=float), q_lo, q_hi)
        for _ in range(iters):
            q_full[ri : ri + 7] = q
            pin.forwardKinematics(self.model, self._pin_data, q_full)
            pin.updateFramePlacements(self.model, self._pin_data)
            T_cur = self._pin_data.oMf[self._ee_frame_id]
            err = pin.log6(T_cur.actInv(T_target)).vector  # 6, LOCAL to T_cur
            if np.linalg.norm(err) < tol:
                break
            J = pin.computeFrameJacobian(
                self.model, self._pin_data, q_full, self._ee_frame_id, pin.LOCAL
            )[:, vi : vi + 7]
            q = np.clip(
                q + np.linalg.solve(J.T @ J + 1e-6 * np.eye(7), J.T @ err), q_lo, q_hi
            )
        return q

    def _build_msg(self, q, dq, ddq, msg_id):
        """Build one MpcInput. Cost weights are the global `weights:` block —
        the SAME for every waypoint of every phase (p1/p2/p3/p4 and the deep
        insert). The tool-tip target is always FK(q), so the joint-space
        reference and the Cartesian target are always consistent."""
        msg = MpcInput()
        msg.id = msg_id
        msg.q = q.tolist()
        msg.qdot = dq.tolist()
        msg.qddot = ddq.tolist()
        msg.robot_effort = np.zeros(7).tolist()
        msg.w_q = W_Q.tolist()
        msg.w_qdot = W_QDOT.tolist()
        msg.w_qddot = W_QDDOT.tolist()
        msg.w_robot_effort = W_EFFORT.tolist()
        msg.w_collision_avoidance = W_COLLISION
        msg.w_nullspace_pos = W_NULLSPACE_POS

        T_ee = self._fk_ee(q)
        quat = pin.Quaternion(T_ee.rotation)
        ee_input = MpcEEInput()
        ee_input.frame_id = "gripper_right_tool_holder"
        ee_input.pose.position.x = float(T_ee.translation[0])
        ee_input.pose.position.y = float(T_ee.translation[1])
        ee_input.pose.position.z = float(T_ee.translation[2])
        ee_input.pose.orientation.x = float(quat.x)
        ee_input.pose.orientation.y = float(quat.y)
        ee_input.pose.orientation.z = float(quat.z)
        ee_input.pose.orientation.w = float(quat.w)
        ee_input.w_pose = list(np.concatenate([W_FRAME_TRANS, W_FRAME_ROT]))

        # DAMSoftContactAugmentedFwdDynamics.update() asserts forces[frame_id]
        # exists on every reference point — carry a zero-weight entry for the
        # contact frame. No force setpoint / cost / constraint in this
        # iteration (see ocp_definition_file.yaml comment).
        force_input = MpcEEInput()
        force_input.frame_id = FORCE_FRAME_ID
        force_input.pose.orientation.w = 1.0  # valid unit quat
        msg.ee_inputs = [ee_input, force_input]
        return msg

    def _append_deep_insert(self, msgs: list, idx: int) -> int:
        """Continue p2 INTO the part: after the insertion path reaches qg
        (tool on the nominal surface), interpolate q in joint space from qg
        to qg_deep — the config, IK-solved, that puts the tool PRESS_DEPTH
        further along the approach direction. Ramp in over PRESS_RAMP_S,
        dwell PRESS_HOLD_S, ramp back to qg.

        This replaces the earlier Cartesian-target march: q and the tool
        target are now always consistent (no state_reg vs frame-tracking
        tug-of-war), and it's a plain joint-space reference with the SAME
        cost weights as every other phase. With a wall the real contact
        stops the arm short of qg_deep — the residual position error is the
        push. Force handling is deferred (see ocp_definition_file.yaml)."""
        q_g = np.array(msgs[-1].q)
        if self.qpg is None:
            raise RuntimeError("_append_deep_insert needs self.qpg — run plan().")
        T_g = self._fk_ee(q_g)
        p_pre = self._fk_ee(self._extract_active_q(self.qpg)).translation
        approach = T_g.translation - p_pre
        n = np.linalg.norm(approach)
        if n < 1e-6:
            raise RuntimeError("p2 approach vector is ~zero — cannot deep-insert.")
        approach /= n  # world frame, the direction p2 came in — no sign guess

        T_deep = pin.SE3(T_g.rotation, T_g.translation + PRESS_DEPTH * approach)
        q_deep = self._ik_ee(T_deep, q_g)
        residual = np.linalg.norm(self._fk_ee(q_deep).translation - T_deep.translation)
        print(
            f"  deep insert: +{PRESS_DEPTH * 1e3:.0f} mm along {np.round(approach, 2)}, "
            f"IK residual {residual * 1e3:.2f} mm, "
            f"joint move {np.round(np.rad2deg(q_deep - q_g), 1)} deg"
        )

        ramp_n = max(1, round(PRESS_RAMP_S / DT))
        hold_n = max(1, round(PRESS_HOLD_S / DT))
        pre_n = max(0, round(POST_P2_HOLD_S / DT))
        alphas = np.concatenate(
            [
                np.zeros(pre_n),  # dwell at qg (p2 done, tool on the surface)
                np.linspace(0.0, 1.0, ramp_n, endpoint=False),  # push in
                np.ones(hold_n),  # dwell at qg_deep
                np.linspace(1.0, 0.0, ramp_n, endpoint=False),  # back to qg
            ]
        )
        print(
            f"    schedule: {pre_n} at qg / {ramp_n} in / {hold_n} at qg_deep / "
            f"{ramp_n} out  = {len(alphas)} wp ({len(alphas) * DT:.1f}s)"
        )
        dq_zero = np.zeros(7)
        ddq_zero = np.zeros(7)
        for a in alphas:
            q_i = q_g + a * (q_deep - q_g)
            msgs.append(self._build_msg(q_i, dq_zero, ddq_zero, idx))
            idx += 1
        return idx

    def _build_messages(self, paths: list, n_hold: int = 200) -> list:
        """
        paths : list of (path, label) tuples to concatenate.
        n_hold: number of hold waypoints appended at the end.
        """
        msgs = []
        idx = 0
        for path, label in paths:
            q_arr, dq_arr, ddq_arr = self._sample_path(path)
            print(f"  {label}: {len(q_arr)} waypoints")
            for q, dq, ddq in zip(q_arr, dq_arr, ddq_arr):
                msgs.append(self._build_msg(q, dq, ddq, idx))
                idx += 1
            if "insertion" in label:
                idx = self._append_deep_insert(msgs, idx)
        q_final = msgs[-1].q
        dq_zero = np.zeros(len(msgs[-1].qdot)).tolist()
        ddq_zero = np.zeros(len(msgs[-1].qddot)).tolist()
        for _ in range(n_hold):
            msg = self._build_msg(
                np.array(q_final), np.array(dq_zero), np.array(ddq_zero), idx
            )
            msgs.append(msg)
            idx += 1
        print(f"  {len(msgs)} MpcInput messages total ({n_hold} hold points appended).")
        return msgs

    # ── Execution ─────────────────────────────────────────────────────────────

    def _start_bag(self, tag: str = ""):
        """Spawn `ros2 bag record` for RECORD_TOPICS, scoped to one execute()
        call. Best-effort: returns None (and never raises) if `ros2` is
        missing or the recorder fails to start, so a run is never blocked."""
        if shutil.which("ros2") is None:
            print("  record: `ros2` not on PATH — skipping bag.")
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(_RUNS_DIR, f"{stamp}_{tag}" if tag else stamp)
        os.makedirs(_RUNS_DIR, exist_ok=True)
        proc = subprocess.Popen(
            ["ros2", "bag", "record", "-o", out, *RECORD_TOPICS],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.0)  # let discovery + subscriptions settle before publishing
        if proc.poll() is not None:
            print("  record: `ros2 bag record` exited immediately — skipping bag.")
            return None
        print(f"  record: {out}")
        return proc, out

    def _stop_bag(self, handle) -> None:
        if handle is None:
            return
        proc, out = handle
        time.sleep(1.5)  # let the last sensor/force/control messages land
        proc.send_signal(signal.SIGINT)  # ros2 bag finalises the DB on SIGINT
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"  record: wrote {out}")
        print(f"  plot:   python3 {_PLOT_SCRIPT} {out}")

    def execute(self, paths=None, record=None, record_linger=None):
        """
        Sample and publish MpcInput messages to the controller.

        paths  : list of Path objects to execute in sequence.
                 Defaults to [p1, p2, p3, p4].
        record : True (or a str tag) wraps the run in a `ros2 bag record` of
                 RECORD_TOPICS under plot/runs/<timestamp>[_tag]/ — feeds
                 plot/plot_force_profile.py directly, no second terminal.
                 None (default) falls back to self.record.
        record_linger : seconds to keep recording after the trajectory is
                 done publishing, before stopping the bag (Ctrl-C ends it
                 early). Use it to capture a manual disturbance right after
                 the motion. None (default) falls back to self.record_linger.
        """
        if self._urdf_file:
            print("OFFLINE mode: execute() is disabled (no robot).")
            return
        if self.p1 is None:
            print("No path available — run plan() first.")
            return

        _labels = {
            id(self.p1): "p1 (approach)",
            id(self.p2): "p2 (insertion)",
            id(self.p3): "p3 (retraction)",
            id(self.p4): "p4 (retreat)",
        }
        if paths is None:
            paths = [self.p1, self.p2, self.p3, self.p4]
        named = [(p, _labels.get(id(p), f"path_{i + 1}")) for i, p in enumerate(paths)]

        _reload_weights()  # pick up edits to hpp_orchestrator_params.yaml without a restart
        print("Sampling trajectories …")
        self._messages = self._build_messages(named)

        if self._ros_node is None:
            self._ros_node = _TrajectoryPublisherNode(self._messages)
        else:
            self._ros_node._messages = self._messages
            self._ros_node._idx = 0
            self._ros_node._done = False

        do_record = self.record if record is None else record
        tag = do_record if isinstance(do_record, str) else ""
        bag = self._start_bag(tag) if do_record else None

        print("Publishing trajectory …")
        # Explicit deadline loop: publish one point every DT of *wall* time.
        # perf_counter accumulator so slightly-long iterations self-correct and
        # the average rate stays at 1/DT (the rclpy-timer version drifted to
        # ~half rate — see _TrajectoryPublisherNode docstring).
        period = DT
        next_t = time.perf_counter()
        late = 0
        try:
            while not self._ros_node._done:
                self._ros_node._publish_next()
                rclpy.spin_once(self._ros_node, timeout_sec=0.0)
                next_t += period
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    late += 1
                    next_t = time.perf_counter()  # fell behind, resync
        except KeyboardInterrupt:
            print("\nExecution interrupted.")
        else:
            linger = self.record_linger if record_linger is None else record_linger
            if bag is not None and linger > 0:
                print(f"  record: trajectory done — still recording {linger:.0f}s "
                      f"(Ctrl-C to stop now).")
                try:
                    time.sleep(linger)
                except KeyboardInterrupt:
                    print("\n  record: linger stopped early.")
        finally:
            if late:
                print(f"  publish loop fell behind on {late} iteration(s)")
            self._stop_bag(bag)

    # ── Pose comparison ───────────────────────────────────────────────────────

    @staticmethod
    def _read_arm_from_dynamic(djs_msg, side: str) -> np.ndarray:
        """Extract arm joint positions from a DynamicJointState message.

        Prefers absolute_position (output shaft encoder); falls back to
        position (motor encoder) if absolute_position is unavailable.
        """
        result = np.zeros(7)
        for i in range(1, 8):
            name = f"arm_{side}_{i}_joint"
            for j, jname in enumerate(djs_msg.joint_names):
                if jname == name:
                    iv = djs_msg.interface_values[j]
                    imap = dict(zip(iv.interface_names, iv.values))
                    result[i - 1] = imap.get(
                        "absolute_position", imap.get("position", 0.0)
                    )
                    break
        return result

    def _read_robot_config(self, timeout: float = 5.0):
        """Return a full HPP config vector from the current robot joint states.

        Uses absolute_position (output shaft encoder) via
        _read_arm_from_dynamic(), falling back to position (motor encoder)
        if absolute_position is unavailable for a joint.
        """
        _own_node = False
        if self._ros_node is None:
            self._ros_node = rclpy.create_node("hpp_read_config_node")
            _own_node = True

        djs_state = [None]
        sub = self._ros_node.create_subscription(
            DynamicJointState,
            "/joint_torque_state_broadcaster/dynamic_joint_states",
            lambda m: djs_state.__setitem__(0, m),
            10,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            if djs_state[0] is not None:
                break
        self._ros_node.destroy_subscription(sub)

        if _own_node:
            self._ros_node.destroy_node()
            self._ros_node = None

        if djs_state[0] is None:
            raise RuntimeError("Timeout reading robot state from dynamic_joint_states.")

        q = pin.neutral(self.model).copy()
        ri = self._right_arm_idx
        q[ri : ri + 7] = self._read_arm_from_dynamic(djs_state[0], "right")
        return q

    def compare_pose(self, q_ref=None, timeout: float = 5.0):
        """
        Compare a reference configuration with the current robot state,
        at the gripper_right_tool_holder frame.

        q_ref : HPP Path object, full config vector, or None (defaults to qg).
                Examples:
                  o.compare_pose()      # vs qg (insertion goal)
                  o.compare_pose(o.p1)  # vs end of p1
        """
        if self._urdf_file:
            print("OFFLINE mode: compare_pose() is disabled (no robot).")
            return
        if self._ros_node is None:
            print("No ROS node available — run execute() first.")
            return
        if q_ref is None:
            if self.qg is None:
                print("No qg available — run plan() first.")
                return
            q_ref = np.array(self.qg)
        elif hasattr(q_ref, "timeRange"):
            q_ref = np.array(q_ref.eval(q_ref.timeRange().second)[0])

        print("Reading current robot state …")
        try:
            q_actual = self._read_robot_config(timeout)
        except RuntimeError as e:
            print(f"compare_pose: {e}")
            return

        data_ref = self.model.createData()
        data_act = self.model.createData()
        pin.forwardKinematics(self.model, data_ref, q_ref)
        pin.updateFramePlacements(self.model, data_ref)
        pin.forwardKinematics(self.model, data_act, q_actual)
        pin.updateFramePlacements(self.model, data_act)

        T_ref = data_ref.oMf[self._ee_frame_id]
        T_act = data_act.oMf[self._ee_frame_id]
        delta = T_ref.inverse() * T_act
        pos_err_mm = np.linalg.norm(delta.translation) * 1000.0
        rot_err_deg = np.degrees(np.linalg.norm(pin.log3(delta.rotation)))

        ri = self._right_arm_idx
        print(f"\n{'=' * 58}")
        print("  Pose comparison  (reference vs actual robot state)")
        print(f"{'=' * 58}")
        print(f"  EE planned [m] : {np.round(T_ref.translation, 4)}")
        print(f"  EE actual  [m] : {np.round(T_act.translation, 4)}")
        print(f"  Position error : {pos_err_mm:.1f} mm")
        print(f"  Rotation error : {rot_err_deg:.2f} °")
        print("\n  Per-joint error — right arm [rad / °]:")
        for i in range(7):
            e = q_ref[ri + i] - q_actual[ri + i]
            print(
                f"    arm_right_{i + 1}_joint : {e:+.4f} rad  ({np.degrees(e):+.2f}°)"
            )
        print(f"{'=' * 58}\n")

    # ── Visualisation ─────────────────────────────────────────────────────────


    # ── Robot state sync ──────────────────────────────────────────────────────

    def sync_from_robot(self, timeout: float = 5.0):
        """Update q_init from the current robot joint state (arm only).

        Uses absolute_position (output shaft encoder) via
        _read_arm_from_dynamic(), falling back to position (motor encoder)
        if absolute_position is unavailable for a joint.
        """
        if self._urdf_file:
            print("OFFLINE mode: sync_from_robot() is disabled (no robot).")
            return
        if self._ros_node is None:
            self._ros_node = rclpy.create_node("hpp_sync_node")
            _own_node = True
        else:
            _own_node = False

        djs_state = [None]
        sub = self._ros_node.create_subscription(
            DynamicJointState,
            "/joint_torque_state_broadcaster/dynamic_joint_states",
            lambda msg: djs_state.__setitem__(0, msg),
            10,
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            if djs_state[0] is not None:
                break

        self._ros_node.destroy_subscription(sub)
        if _own_node:
            self._ros_node.destroy_node()
            self._ros_node = None

        if djs_state[0] is None:
            print("sync_from_robot: timeout — could not receive dynamic_joint_states")
            return

        ri = self._right_arm_idx
        li = self._left_arm_idx

        right_arm = self._read_arm_from_dynamic(djs_state[0], "right")
        left_arm = self._read_arm_from_dynamic(djs_state[0], "left")
        self.q_init[ri : ri + 7] = right_arm
        self.q_init[li : li + 7] = left_arm

        self._left_arm_lock_values = left_arm.tolist()
        print("  Rebuilding constraint graph with synced left arm …")
        self._setup_graph()

        print(
            f"sync_from_robot: "
            f"right_arm={np.round(right_arm, 3).tolist()}  "
            f"left_arm={np.round(left_arm, 3).tolist()}"
        )

    # ── Pylone pose update ────────────────────────────────────────────────────

    def reload_pylone_pose(self) -> None:
        """Read pylone pose from config/pylone_pose.yaml and update q_init."""
        if not os.path.exists(_PYLONE_POSE_FILE):
            print(f"No pylone pose file found at {_PYLONE_POSE_FILE}.")
            return
        with open(_PYLONE_POSE_FILE) as f:
            cfg = yaml.safe_load(f)
        t = [cfg["pylone_x"], cfg["pylone_y"], cfg["pylone_z"]]
        q = cfg.get("pylone_quat", [0.0, 0.0, 0.0, 1.0])
        self.update_pylone_pose(t, q)

    def update_pylone_pose(self, t: list, q: list = None) -> None:
        """Update pylone pose in q_init and display in Viser if open.

        t: [x, y, z] in base_link frame
        q: [qx, qy, qz, qw] orientation (default: identity)
        """
        t = np.array(t)
        q = np.array(q) if q is not None else np.array([0.0, 0.0, 0.0, 1.0])
        self._set_pylone_bounds(t[0], t[1], t[2])
        pi = self._pylone_idx
        self.q_init[pi : pi + 3] = t
        self.q_init[pi + 3 : pi + 7] = q
        print(
            f"Pylone pose updated: t={np.round(t, 4).tolist()}, q={np.round(q, 4).tolist()}."
        )
        if self._viewer is not None:
            self._place_pylone()
        else:
            print("Call o.init_viewer() to visualize.")

    # ── Visualisation (viser) ─────────────────────────────────────────────────
    # demo 07 uses pyhpp_viser, which is not available in the nix devshell: same
    # idea with viser + yourdfpy on the robot's own URDF.

    def init_viewer(self, port: int = 8080) -> None:
        """Start viser (http://localhost:PORT): the robot and the pylone."""
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf

        if self._viewer is not None:
            print("Viewer already running.")
            return

        def _resolve(fname):
            if fname.startswith("package://"):
                from ament_index_python.packages import get_package_share_directory

                pkg, _, path = fname[len("package://") :].partition("/")
                return f"{get_package_share_directory(pkg)}/{path}"
            return fname

        with tempfile.NamedTemporaryFile(suffix=".urdf", mode="w") as f:
            f.write(self._urdf_str)
            f.flush()
            urdf_y = yourdfpy.URDF.load(f.name, filename_handler=_resolve)
        server = viser.ViserServer(port=port)
        vis = ViserUrdf(server, urdf_y, load_collision_meshes=False)
        self._viewer = {
            "server": server,
            "vis": vis,
            "urdf": urdf_y,
            "actuated": list(urdf_y.actuated_joint_names),
        }
        self._place_pylone()
        self.view(self.q_init)
        print(f"Viser viewer ready: http://localhost:{port}.  Use o.view(q) or o.play(path).")

    def _place_pylone(self) -> None:
        """(Re)draw the pylone mesh at the pose stored in q_init. That pose is given in
        base_link; viser's scene is in the URDF root frame, hence the transform."""
        import trimesh

        v = self._viewer
        pi = self._pylone_idx
        t, qx = self.q_init[pi : pi + 3], self.q_init[pi + 3 : pi + 7]  # xyz, (x, y, z, w)
        T_base_pyl = pin.SE3(pin.Quaternion(qx[3], qx[0], qx[1], qx[2]).toRotationMatrix(), t)
        root = v["urdf"].base_link
        T_root_base = pin.SE3(
            np.array(v["urdf"].get_transform("base_link", root))[:3, :3],
            np.array(v["urdf"].get_transform("base_link", root))[:3, 3],
        )
        T = T_root_base * T_base_pyl
        quat = pin.Quaternion(T.rotation)
        mesh = trimesh.load(os.path.join(_PKG_DIR, "urdf", "visual", "pylone.stl"))
        v["server"].scene.add_mesh_trimesh(
            "/pylone",
            mesh,
            position=tuple(T.translation),
            wxyz=(quat.w, quat.x, quat.y, quat.z),
        )

    def view(self, q=None) -> None:
        """Show a full HPP configuration (default q_init) in viser; prints the EE pose."""
        if self._viewer is None:
            self.init_viewer()
        q = np.array(q) if q is not None else self.q_init
        cfg = []
        for name in self._viewer["actuated"]:
            jn = f"tiago_pro/{name}"
            if self.model.existJointName(jn):
                j = self.model.joints[self.model.getJointId(jn)]
                cfg.append(q[j.idx_q] if j.nq == 1 else 0.0)
            else:
                cfg.append(0.0)
        self._viewer["vis"].update_cfg(np.array(cfg))
        T = self._fk_ee(self._extract_active_q(q))
        rpy = pin.rpy.matrixToRpy(T.rotation)
        print(f"EE position (xyz) [m]: {np.round(T.translation, 4).tolist()}")
        print(f"EE rotation (rpy) [°]: {np.round(np.degrees(rpy), 1).tolist()}")

    def play(self, path, n: int = 100, dt: float = 0.05) -> None:
        """Animate an HPP path in viser by sampling n configurations."""
        if self._viewer is None:
            self.init_viewer()
        t0, tf = path.timeRange().first, path.timeRange().second
        for i in range(n):
            self.view_quiet(path.eval(t0 + i * (tf - t0) / (n - 1))[0])
            time.sleep(dt)

    def view_quiet(self, q) -> None:
        """Like view() without the EE print (used by play())."""
        import io
        from contextlib import redirect_stdout

        with redirect_stdout(io.StringIO()):
            self.view(q)

    # ── Controller activation ─────────────────────────────────────────────────

    def activate_lfc(self) -> None:
        """Switch arm_right_controller → LFC + JSE. Call this when ready to move."""
        if self._urdf_file:
            print("OFFLINE mode: activate_lfc() is disabled (no robot).")
            return
        print("Activating LFC controllers …")
        subprocess.run(
            [
                "ros2",
                "control",
                "switch_controllers",
                "--deactivate",
                "arm_right_controller",
                "--activate",
                "linear_feedback_controller",
                "joint_state_estimator",
            ],
            check=True,
        )
        print("LFC controllers active.")

    def deactivate_lfc(self) -> None:
        """Switch LFC + JSE → arm_right_controller. Call this to return to position control."""
        if self._urdf_file:
            print("OFFLINE mode: deactivate_lfc() is disabled (no robot).")
            return
        print("Deactivating LFC controllers …")
        subprocess.run(
            [
                "ros2",
                "control",
                "switch_controllers",
                "--deactivate",
                "linear_feedback_controller",
                "joint_state_estimator",
                "--activate",
                "arm_right_controller",
            ],
            check=True,
        )
        print("arm_right_controller active.")

    # ── Combined ──────────────────────────────────────────────────────────────

    def plan_and_execute(self, max_attempts: int = 50):
        """Plan then immediately execute."""
        if self.plan(max_attempts=max_attempts):
            self.execute()
