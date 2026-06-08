"""
HPP orchestrator for the TIAGo Pro deburring demo.

HPP plans a full-body trajectory (base + arm).  At execution time:
  • Base trajectory  → base_trajectory_follower node (odom feedback → cmd_vel)
  • Arm trajectory   → fixed-base MPC (agimus_controller, 7 DoF)

Usage (IPython):
    o = Orchestrator()
    o.plan()
    o.execute()            # base + arm simultaneously
    o.execute([o.p1])      # approach only

Robot state:
    o.sync_from_robot()    # update q_init from live robot state
    o.compare_pose()       # compare planned qg vs actual pose

Visualisation (Viser):
    o.init_viewer()
    o.view(q)
    o.play(o.p1)

Pylone:
    o.reload_pylone_pose()
    o.update_pylone_pose(t, q)

Mocap (Qualisys):
    o.connect_mocap()
    o.localize_pylone_from_mocap()
    o.compare_mocap()
    o.update_mocap_frames()

LFC:
    o.activate_lfc()
    o.deactivate_lfc()
"""

import os
import subprocess
import sys
import glob
import time

import numpy as np
import pinocchio as pin
import yaml

for _p in sorted(
    glob.glob("/home/gepetto/ros2_ws/install/*/lib/python3*/site-packages")
    + glob.glob("/home/gepetto/agimus_deps_ws/install/*/lib/python3*/site-packages")
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pyhpp.manipulation import TransitionPlanner
from pyhpp.core import RandomShortcut, SplineGradientBased_bezier3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from agimus_msgs.msg import MpcInput, MpcEEInput
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState

try:
    from .moving_deburring_problem import DeburringProblem, RIGHT_ARM_TUCK, _PYLONE_POSE_FILE
except ImportError:
    from moving_deburring_problem import DeburringProblem, RIGHT_ARM_TUCK, _PYLONE_POSE_FILE

_PKG_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_CFG_FILE = os.path.join(_PKG_DIR, "config", "hpp_orchestrator_params.yaml")
with open(_CFG_FILE) as _f:
    _cfg = yaml.safe_load(_f)

DT         = _cfg["trajectory"]["dt"]
TIME_SCALE = _cfg["trajectory"]["time_scale"]

_w          = _cfg["weights"]
W_Q         = np.array(_w["w_q"])
W_QDOT      = np.array(_w["w_qdot"])
W_QDDOT     = np.array(_w["w_qddot"])
W_EFFORT    = np.array(_w["w_effort"])
W_COLLISION   = _w["w_collision"]
W_FRAME_TRANS = np.array(_w["w_frame_trans"])
W_FRAME_ROT   = np.array(_w["w_frame_rot"])


class _TrajectoryPublisherNode(Node):
    """Publishes arm MpcInput messages and the base nav_msgs/Path."""

    def __init__(self, arm_msgs: list, base_path: Path):
        super().__init__("hpp_trajectory_publisher")
        self._arm_msgs  = arm_msgs
        self._idx       = 0
        self._done      = False

        qos_arm = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_path = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._arm_pub  = self.create_publisher(MpcInput, "mpc_input", qos_arm)
        self._base_pub = self.create_publisher(Path, "/base_trajectory", qos_path)

        self._base_pub.publish(base_path)
        self._timer = self.create_timer(DT, self._publish_next)
        self.get_logger().info(
            f"Publishing {len(self._arm_msgs)} arm waypoints at {1/DT:.0f} Hz …"
        )

    def _publish_next(self):
        if self._idx >= len(self._arm_msgs):
            if not self._done:
                self.get_logger().info("Arm trajectory fully published.")
                self._done = True
            self._timer.cancel()
            return
        self._arm_pub.publish(self._arm_msgs[self._idx])
        self._idx += 1


class Orchestrator:
    """Interactive orchestrator for HPP deburring planning and MPC execution."""

    def __init__(self, ros_node: Node = None):
        self._ros_node = ros_node
        self.p1 = self.p2 = self.p3 = None
        self._messages = None

        urdf_str = self._fetch_robot_urdf()
        self._prob = DeburringProblem(urdf_str)

        # Expose HPP objects directly for interactive use
        self.robot  = self._prob.robot
        self.model  = self._prob.model
        self.problem = self._prob.problem
        self.graph   = self._prob.graph
        self.q_init  = self._prob.q_init

        print("Orchestrator ready.  Call plan() to start.\n")

    # ── URDF fetch ────────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_robot_urdf(timeout: float = 10.0) -> str:
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
                f"Timed out after {timeout}s waiting for /robot_description."
            )
        return urdf_str

    # ── Planning ──────────────────────────────────────────────────────────────

    def plan(self, max_attempts: int = 50) -> bool:
        """Plan p1 (approach, base free), p2 (insertion, base fixed), p3 (retraction).
        Call sync_from_robot() first if needed."""
        prob = self._prob

        # ── Phase 1 : base free — find qpg, plan p1 ──────────────────────────
        prob.unlock_base_in_graph()
        shooter = prob.problem.configurationShooter()

        qpg = None
        for i in range(max_attempts):
            q = shooter.shoot()
            res, q_cand, err = prob.graph.generateTargetConfig(
                prob.transition_approach, prob.q_init, q
            )
            if not res:
                continue
            pv = prob.transition_approach.pathValidation()
            res, _ = pv.validateConfiguration(q_cand)
            if not res:
                continue
            qpg = q_cand
            print(f"  qpg found at attempt {i}, err={err:.2e}")
            break

        if qpg is None:
            print(f"Failed to find collision-free qpg in {max_attempts} attempts.")
            return False

        prob.problem.constraintGraph(prob.graph)
        planner    = TransitionPlanner(prob.problem)
        planner.maxIterations(1000)
        shortcut   = RandomShortcut(prob.problem)
        spline_opt = SplineGradientBased_bezier3(prob.problem)

        planner.setTransition(prob.transition_approach)
        q_goal = np.zeros((1, self.robot.configSize()), order='F')
        q_goal[0, :] = qpg
        print("Planning p1 (approach, base free) …")
        p1 = planner.planPath(prob.q_init, q_goal, True)
        print("  p1 found.")

        try:
            for i in range(3):
                p1_new = shortcut.optimize(p1)
                dt = (p1.timeRange().second - p1.timeRange().first) - \
                     (p1_new.timeRange().second - p1_new.timeRange().first)
                p1 = p1_new
                print(f"  p1 shortcut {i+1}/3: {p1.timeRange().second - p1.timeRange().first:.2f}s  (−{dt:.2f}s)")
                if dt < 1e-3:
                    break
        except Exception as e:
            print(f"  p1 shortcut failed: {e}")
        try:
            p1 = spline_opt.optimize(p1)
            print(f"  p1 spline: {p1.timeRange().second - p1.timeRange().first:.2f}s")
        except Exception as e:
            print(f"  p1 spline failed: {e}")

        # ── Phase 2 : base locked at qpg — find qg, plan p2 ──────────────────
        prob.lock_base_in_graph(qpg)

        res, qg, err = prob.graph.generateTargetConfig(
            prob.transition_insert, qpg, qpg
        )
        print(f"  qg: res={res}, err={err:.2e}")
        if not res:
            prob.unlock_base_in_graph()
            print("Failed to generate qg.")
            return False

        prob.problem.constraintGraph(prob.graph)
        planner    = TransitionPlanner(prob.problem)
        planner.maxIterations(1000)
        shortcut   = RandomShortcut(prob.problem)
        spline_opt = SplineGradientBased_bezier3(prob.problem)

        planner.setTransition(prob.transition_insert)
        q_goal[0, :] = qg
        print("Planning p2 (insertion, base fixed) …")
        p2 = planner.planPath(qpg, q_goal, True)
        print("  p2 found.")
        try:
            p2 = shortcut.optimize(p2)
            print(f"  p2 shortcut: {p2.timeRange().second - p2.timeRange().first:.2f}s")
        except Exception as e:
            print(f"  p2 shortcut failed: {e}")
        try:
            p2 = spline_opt.optimize(p2)
            print(f"  p2 spline: {p2.timeRange().second - p2.timeRange().first:.2f}s")
        except Exception as e:
            print(f"  p2 spline failed: {e}")

        # ── Restore base-free graph ───────────────────────────────────────────
        prob.unlock_base_in_graph()
        self.graph   = prob.graph
        self.problem = prob.problem

        p3 = p2.reverse()
        print("  p3 ready (retraction).")

        self.p1  = p1
        self.p2  = p2
        self.p3  = p3
        self.qpg = qpg
        self.qg  = qg
        return True

    # ── Path sampling ─────────────────────────────────────────────────────────

    def _extract_arm_pose(self, q_full: np.ndarray) -> np.ndarray:
        """Extract the 7 right-arm joint positions from a full HPP config."""
        ri = self._prob.right_arm_idx
        return np.array(q_full)[ri:ri+7]

    def _extract_base_pose(self, q_full: np.ndarray) -> tuple:
        """Extract (x, y, theta) from a full HPP config."""
        bi = self._prob.base_idx
        q  = np.array(q_full)
        return q[bi], q[bi+1], np.arctan2(q[bi+3], q[bi+2])

    def _sample_path(self, path):
        """
        Sample a HPP path and return arm arrays + base waypoints.

        Returns
        -------
        q_full_list : list of full HPP config arrays
        arm_q   : (n, 7)  arm joint positions
        arm_dq  : (n, 7)  arm velocities (finite diff)
        arm_ddq : (n, 7)  arm accelerations (finite diff)
        base_poses : list of (x, y, theta)
        """
        tr = path.timeRange()
        t_min, t_max = tr.first, tr.second
        n = max(2, int((t_max - t_min) * TIME_SCALE / DT))
        times = np.linspace(t_min, t_max, n)

        q_full_list = [np.array(path.eval(t)[0]) for t in times]
        arm_q = np.array([self._extract_arm_pose(q) for q in q_full_list])

        arm_dq = np.zeros_like(arm_q)
        for i in range(len(arm_q) - 1):
            arm_dq[i] = (arm_q[i+1] - arm_q[i]) / DT
        arm_dq[-1] = arm_dq[-2]

        arm_ddq = np.zeros_like(arm_q)
        for i in range(len(arm_dq) - 1):
            arm_ddq[i] = (arm_dq[i+1] - arm_dq[i]) / DT
        arm_ddq[-1] = arm_ddq[-2]

        base_poses = [self._extract_base_pose(q) for q in q_full_list]

        return q_full_list, arm_q, arm_dq, arm_ddq, base_poses

    def _build_arm_msg(self, q_full: np.ndarray, arm_q, arm_dq, arm_ddq, msg_id: int) -> MpcInput:
        msg = MpcInput()
        msg.id                    = msg_id
        msg.q                     = arm_q.tolist()
        msg.qdot                  = arm_dq.tolist()
        msg.qddot                 = arm_ddq.tolist()
        msg.robot_effort          = np.zeros(7).tolist()
        msg.w_q                   = W_Q.tolist()
        msg.w_qdot                = W_QDOT.tolist()
        msg.w_qddot               = W_QDDOT.tolist()
        msg.w_robot_effort        = W_EFFORT.tolist()
        msg.w_collision_avoidance = W_COLLISION

        T_ee = self._prob.fk_ee_rel_base(q_full)
        quat = pin.Quaternion(T_ee.rotation)
        ee_input = MpcEEInput()
        ee_input.frame_id = "gripper_right_tool_holder"
        ee_input.pose.position.x    = float(T_ee.translation[0])
        ee_input.pose.position.y    = float(T_ee.translation[1])
        ee_input.pose.position.z    = float(T_ee.translation[2])
        ee_input.pose.orientation.x = float(quat.x)
        ee_input.pose.orientation.y = float(quat.y)
        ee_input.pose.orientation.z = float(quat.z)
        ee_input.pose.orientation.w = float(quat.w)
        ee_input.w_pose = list(np.concatenate([W_FRAME_TRANS, W_FRAME_ROT]))
        msg.ee_inputs = [ee_input]
        return msg

    def _build_messages(self, paths: list, n_hold: int = 200):
        """
        Build arm MpcInput messages and a nav_msgs/Path for the base.

        Parameters
        ----------
        paths   : list of (path, label) tuples
        n_hold  : hold points appended at end for MPC convergence

        Returns
        -------
        arm_msgs   : list[MpcInput]
        base_path  : nav_msgs/Path
        """
        arm_msgs   = []
        base_poses = []
        idx = 0

        for path, label in paths:
            q_full_list, arm_q, arm_dq, arm_ddq, bp = self._sample_path(path)
            print(f"  {label}: {len(arm_q)} waypoints")
            for q_full, aq, adq, addq in zip(q_full_list, arm_q, arm_dq, arm_ddq):
                arm_msgs.append(self._build_arm_msg(q_full, aq, adq, addq, idx))
                idx += 1
            base_poses.extend(bp)

        # Hold points (arm stays at final position)
        q_final  = np.array(arm_msgs[-1].q)
        dq_zero  = np.zeros(7)
        ddq_zero = np.zeros(7)
        q_full_final = q_full_list[-1]
        for _ in range(n_hold):
            arm_msgs.append(self._build_arm_msg(q_full_final, q_final, dq_zero, ddq_zero, idx))
            base_poses.append(base_poses[-1])
            idx += 1

        print(f"  {len(arm_msgs)} arm waypoints total ({n_hold} hold points appended).")

        # Build nav_msgs/Path
        base_path = Path()
        base_path.header.frame_id = "odom"
        for i, (x, y, theta) in enumerate(base_poses):
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = float(np.sin(theta / 2.0))
            ps.pose.orientation.w = float(np.cos(theta / 2.0))
            base_path.poses.append(ps)

        return arm_msgs, base_path

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, paths=None):
        """
        Execute the trajectory: base follower + arm MPC simultaneously.

        paths : list of Path objects.  Defaults to [p1, p2, p3].
        Examples:
            o.execute([o.p1])          # approach only
            o.execute([o.p1, o.p2])    # approach + insertion
        """
        if self.p1 is None:
            print("No path available — run plan() first.")
            return

        _labels = {id(self.p1): "p1 (approach)",
                   id(self.p2): "p2 (insertion)",
                   id(self.p3): "p3 (retraction)"}
        if paths is None:
            paths = [self.p1, self.p2, self.p3]
        named = [(p, _labels.get(id(p), f"path_{i+1}")) for i, p in enumerate(paths)]

        print("Sampling trajectories …")
        arm_msgs, base_path = self._build_messages(named)

        if self._ros_node is None:
            self._ros_node = _TrajectoryPublisherNode(arm_msgs, base_path)
        else:
            # Republish base path and restart arm publisher
            self._ros_node._arm_msgs = arm_msgs
            self._ros_node._idx      = 0
            self._ros_node._done     = False
            self._ros_node._base_pub.publish(base_path)
            self._ros_node._timer = self._ros_node.create_timer(
                DT, self._ros_node._publish_next
            )

        print("Publishing trajectory …")
        try:
            while not self._ros_node._done:
                rclpy.spin_once(self._ros_node, timeout_sec=0.0)
                time.sleep(DT)
        except KeyboardInterrupt:
            print("\nExecution interrupted.")

    def plan_and_execute(self, max_attempts: int = 50):
        """Plan then immediately execute."""
        if self.plan(max_attempts):
            self.execute()

    # ── Visualisation ─────────────────────────────────────────────────────────

    def init_viewer(self, open: bool = True):
        from pyhpp_viser import Viewer
        self._viewer = Viewer(self.robot)
        self._viewer.initViewer(open=open, loadModel=True)
        self._viewer.setProblem(self.problem)
        self._viewer.setGraph(self.graph)
        self._viewer(self.q_init)
        print("Viser viewer ready.")

    def view(self, q=None):
        if not hasattr(self, "_viewer"):
            self.init_viewer()
        self._viewer(q if q is not None else self.q_init)

    def play(self, path, n=100, dt=0.05):
        """Play a path in Viser."""
        import time as _time
        if not hasattr(self, "_viewer"):
            self.init_viewer()
        try:
            self._viewer.loadPath(path)
        except Exception:
            pass
        t0 = path.timeRange().first
        tf = path.timeRange().second
        for i in range(n):
            t = t0 + i * (tf - t0) / (n - 1)
            self._viewer(path.eval(t)[0])
            _time.sleep(dt)

    # ── Robot state sync ──────────────────────────────────────────────────────

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
                    result[i - 1] = imap.get("absolute_position",
                                             imap.get("position", 0.0))
                    break
        return result

    def sync_from_robot(self, timeout: float = 5.0):
        """Update q_init from the current robot state (dynamic_joint_states + odom).

        Uses absolute_position (output shaft encoder) when available,
        falls back to position (motor encoder) otherwise.
        """
        if self._ros_node is None:
            self._ros_node = rclpy.create_node("hpp_sync_node")
            _own_node = True
        else:
            _own_node = False

        djs_state  = [None]
        odom_state = [None]

        djs_sub = self._ros_node.create_subscription(
            DynamicJointState,
            "/joint_torque_state_broadcaster/dynamic_joint_states",
            lambda msg: djs_state.__setitem__(0, msg), 10
        )
        odom_sub = self._ros_node.create_subscription(
            Odometry, "/odom",
            lambda msg: odom_state.__setitem__(0, msg), 10
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            if djs_state[0] is not None and odom_state[0] is not None:
                break

        self._ros_node.destroy_subscription(djs_sub)
        self._ros_node.destroy_subscription(odom_sub)
        if _own_node:
            self._ros_node.destroy_node()
            self._ros_node = None

        if djs_state[0] is None or odom_state[0] is None:
            missing = (["dynamic_joint_states"] if djs_state[0]  is None else []) + \
                      (["/odom"]                if odom_state[0] is None else [])
            print(f"sync_from_robot: timeout — could not receive {', '.join(missing)}")
            return

        odom  = odom_state[0]
        bx    = odom.pose.pose.position.x
        by    = odom.pose.pose.position.y
        q_o   = odom.pose.pose.orientation
        siny  = 2.0 * (q_o.w * q_o.z + q_o.x * q_o.y)
        cosy  = 1.0 - 2.0 * (q_o.y * q_o.y + q_o.z * q_o.z)
        theta = np.arctan2(siny, cosy)

        right_arm = self._read_arm_from_dynamic(djs_state[0], "right")
        left_arm  = self._read_arm_from_dynamic(djs_state[0], "left")

        bi = self._prob.base_idx
        ri = self._prob.right_arm_idx
        li = self._prob.left_arm_idx

        self.q_init[bi]      = bx
        self.q_init[bi + 1]  = by
        self.q_init[bi + 2]  = np.cos(theta)
        self.q_init[bi + 3]  = np.sin(theta)
        self.q_init[ri:ri+7] = right_arm
        self.q_init[li:li+7] = left_arm

        self._left_arm_lock_values = left_arm.tolist()
        print("  Rebuilding constraint graph with synced left arm …")
        self._prob.rebuild_graph(left_arm.tolist())
        self.problem = self._prob.problem
        self.graph   = self._prob.graph

        print(
            f"sync_from_robot: base=({bx:.3f}, {by:.3f}, θ={np.degrees(theta):.1f}°)"
            f"  right_arm={np.round(right_arm, 3).tolist()}"
        )

    # ── Pose comparison ───────────────────────────────────────────────────────

    def _read_robot_config(self, timeout: float = 5.0) -> np.ndarray:
        """Return a full HPP config vector from the current robot state.

        Uses absolute_position (output shaft encoder) when available,
        falls back to position (motor encoder) otherwise.
        """
        if self._ros_node is None:
            self._ros_node = rclpy.create_node("hpp_read_config_node")
            _own = True
        else:
            _own = False

        djs_state  = [None]
        odom_state = [None]
        djs_sub = self._ros_node.create_subscription(
            DynamicJointState,
            "/joint_torque_state_broadcaster/dynamic_joint_states",
            lambda m: djs_state.__setitem__(0, m), 10)
        od_sub = self._ros_node.create_subscription(
            Odometry, "/odom", lambda m: odom_state.__setitem__(0, m), 10)

        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            if djs_state[0] is not None and odom_state[0] is not None:
                break

        self._ros_node.destroy_subscription(djs_sub)
        self._ros_node.destroy_subscription(od_sub)
        if _own:
            self._ros_node.destroy_node()
            self._ros_node = None

        if djs_state[0] is None or odom_state[0] is None:
            raise RuntimeError("Timeout reading robot state from ROS.")

        odom  = odom_state[0]
        bx    = odom.pose.pose.position.x
        by    = odom.pose.pose.position.y
        q_o   = odom.pose.pose.orientation
        siny  = 2.0 * (q_o.w * q_o.z + q_o.x * q_o.y)
        cosy  = 1.0 - 2.0 * (q_o.y**2 + q_o.z**2)
        theta = np.arctan2(siny, cosy)

        bi = self._prob.base_idx
        li = self._prob.left_arm_idx
        ri = self._prob.right_arm_idx
        q  = pin.neutral(self.model).copy()
        q[bi], q[bi+1] = bx, by
        q[bi+2], q[bi+3] = np.cos(theta), np.sin(theta)
        q[li:li+7] = self._read_arm_from_dynamic(djs_state[0], "left")
        q[ri:ri+7] = self._read_arm_from_dynamic(djs_state[0], "right")
        return q

    def compare_pose(self, q_ref=None, timeout: float = 5.0):
        """Compare a reference configuration (default: qg) with the current robot state."""
        if q_ref is None:
            if not hasattr(self, "qg") or self.qg is None:
                print("No qg available — run plan() first.")
                return
            q_ref = self.qg

        print("Reading current robot state …")
        try:
            q_actual = self._read_robot_config(timeout)
        except RuntimeError as e:
            print(f"compare_pose: {e}")
            return

        model = self.model
        data_ref = model.createData()
        data_act = model.createData()
        pin.forwardKinematics(model, data_ref, q_ref)
        pin.updateFramePlacements(model, data_ref)
        pin.forwardKinematics(model, data_act, q_actual)
        pin.updateFramePlacements(model, data_act)

        frame_id = model.getFrameId("tiago_pro/arm_right_tool_link")
        offset   = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.15]))
        T_ref    = data_ref.oMf[frame_id] * offset
        T_act    = data_act.oMf[frame_id] * offset

        delta       = T_ref.inverse() * T_act
        pos_err_mm  = np.linalg.norm(delta.translation) * 1000.0
        rot_err_deg = np.degrees(np.linalg.norm(pin.log3(delta.rotation)))

        bi = self._prob.base_idx
        ri = self._prob.right_arm_idx
        bx_ref, by_ref = q_ref[bi], q_ref[bi+1]
        bx_act, by_act = q_actual[bi], q_actual[bi+1]
        yaw_ref = np.arctan2(q_ref[bi+3], q_ref[bi+2])
        yaw_act = np.arctan2(q_actual[bi+3], q_actual[bi+2])
        dyaw = np.arctan2(np.sin(yaw_ref - yaw_act), np.cos(yaw_ref - yaw_act))

        print(f"\n{'='*58}")
        print(f"  Pose comparison  (reference vs actual robot state)")
        print(f"{'='*58}")
        print(f"\n  Base error:")
        print(f"    x   : {bx_ref - bx_act:+.4f} m")
        print(f"    y   : {by_ref - by_act:+.4f} m")
        print(f"    yaw : {np.degrees(dyaw):+.2f}°")
        print(f"  Gripper planned [m] : {np.round(T_ref.translation, 4)}")
        print(f"  Gripper actual  [m] : {np.round(T_act.translation, 4)}")
        print(f"  Position error      : {pos_err_mm:.1f} mm")
        print(f"  Rotation error      : {rot_err_deg:.2f} °")
        print(f"\n  Per-joint error — right arm [rad / °]:")
        for i in range(7):
            e = q_ref[ri+i] - q_actual[ri+i]
            print(f"    arm_right_{i+1}_joint : {e:+.4f} rad  ({np.degrees(e):+.2f}°)")
        print(f"{'='*58}\n")

    # ── Pylone pose update ────────────────────────────────────────────────────

    def reload_pylone_pose(self) -> None:
        self._prob.reload_pylone_pose()
        if hasattr(self, "_viewer"):
            self._viewer(self.q_init)

    def update_pylone_pose(self, t: list, q: list = None) -> None:
        """Update pylone pose.  t=[x,y,z] in HPP world frame."""
        self._prob.update_pylone_pose(t, q)
        if hasattr(self, "_viewer"):
            self._viewer(self.q_init)
        print(f"Pylone pose updated: t={np.round(np.array(t), 4).tolist()}")
        if not hasattr(self, "_viewer"):
            print("Call o.init_viewer() to visualize.")

    # ── Mocap (Qualisys) ─────────────────────────────────────────────────────

    _QUALISYS_IP    = "140.93.1.100"
    _MOCAP_BODIES   = {"pylone": 0, "tiago_endEffector": 1, "tiago_base": 2}
    _MOCAP_BASE_IDX = 2

    def connect_mocap(self, ip: str = _QUALISYS_IP) -> None:
        _scripts_dir = os.path.join(_PKG_DIR, "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from qualisys import QualisysClient
        self._qc = QualisysClient(ip=ip, bodies=self._MOCAP_BODIES)
        time.sleep(1.0)
        print(f"Mocap connected to {ip}.")

    def disconnect_mocap(self) -> None:
        if hasattr(self, "_qc"):
            self._qc.stop()
            del self._qc
            print("Mocap disconnected.")

    def _mocap_se3(self, idx: int) -> pin.SE3:
        pos  = self._qc.getPositions()[idx]
        quat = self._qc.getOrientationQuats()[idx]
        return pin.XYZQUATToSE3(np.concatenate([pos, quat]))

    def compare_mocap(self, timeout: float = 5.0) -> None:
        """Compare mocap poses vs robot FK relative to base_link / tiago_base."""
        if not hasattr(self, "_qc"):
            print("No mocap client — call connect_mocap() first.")
            return

        print("Reading current robot state …")
        try:
            q_actual = self._read_robot_config(timeout)
        except RuntimeError as e:
            print(f"compare_mocap: {e}")
            return

        T_mocap_base    = self._mocap_se3(self._MOCAP_BASE_IDX)
        T_rel_ee_mocap  = T_mocap_base.inverse() * self._mocap_se3(1)
        T_rel_pyl_mocap = T_mocap_base.inverse() * self._mocap_se3(0)

        T_rel_ee_robot  = self._prob.fk_ee_rel_base(q_actual)

        pi = self._prob.pylone_idx
        T_pyl_world  = pin.XYZQUATToSE3(
            np.concatenate([self.q_init[pi:pi+3], self.q_init[pi+3:pi+7]])
        )
        T_base_world = self._prob.base_se3_from_q(q_actual)
        T_rel_pyl_robot = T_base_world.inverse() * T_pyl_world

        def _breakdown(T_a, T_b):
            delta = T_a.inverse() * T_b
            return delta.translation * 1e3, np.degrees(pin.utils.matrixToRpy(delta.rotation))

        def _print_body(label, T_mocap, T_robot):
            t_m   = T_mocap.translation
            t_r   = T_robot.translation
            rpy_m = np.degrees(pin.utils.matrixToRpy(T_mocap.rotation))
            rpy_r = np.degrees(pin.utils.matrixToRpy(T_robot.rotation))
            dt_mm, drpy = _breakdown(T_mocap, T_robot)
            norm_t = np.linalg.norm(dt_mm)
            norm_r = np.linalg.norm(drpy)
            flag   = "✓" if norm_t < 20 and norm_r < 5 else "!"
            print(f"\n  {label}")
            print(f"  {'':4s}{'':12s}  {'x':>9s}  {'y':>9s}  {'z':>9s}")
            print(f"  {'':4s}{'mocap [m]':12s}  {t_m[0]:>+9.4f}  {t_m[1]:>+9.4f}  {t_m[2]:>+9.4f}")
            print(f"  {'':4s}{'robot [m]':12s}  {t_r[0]:>+9.4f}  {t_r[1]:>+9.4f}  {t_r[2]:>+9.4f}")
            print(f"  {'':4s}{'Δ [mm]':12s}  {dt_mm[0]:>+9.2f}  {dt_mm[1]:>+9.2f}  {dt_mm[2]:>+9.2f}  (|Δ|={norm_t:.1f} mm)  {flag}")
            print(f"  {'':4s}{'':12s}  {'roll':>9s}  {'pitch':>9s}  {'yaw':>9s}")
            print(f"  {'':4s}{'mocap [°]':12s}  {rpy_m[0]:>+9.2f}  {rpy_m[1]:>+9.2f}  {rpy_m[2]:>+9.2f}")
            print(f"  {'':4s}{'robot [°]':12s}  {rpy_r[0]:>+9.2f}  {rpy_r[1]:>+9.2f}  {rpy_r[2]:>+9.2f}")
            print(f"  {'':4s}{'Δ [°]':12s}  {drpy[0]:>+9.2f}  {drpy[1]:>+9.2f}  {drpy[2]:>+9.2f}  (|Δ|={norm_r:.2f}°)  {flag}")

        print(f"\n{'='*66}")
        print(f"  Mocap vs Robot — poses relative to base_footprint / tiago_base")
        print(f"{'='*66}")
        _print_body("End effector  (tiago_endEffector ↔ gripper_right_tool_holder)",
                    T_rel_ee_mocap, T_rel_ee_robot)
        print(f"\n  {'-'*62}")
        _print_body("Pylone  (mocap ↔ pose localisée)", T_rel_pyl_mocap, T_rel_pyl_robot)
        print(f"\n{'='*66}\n")

    def localize_pylone_from_mocap(self) -> None:
        """Set the pylone pose from the current mocap reading (requires sync_from_robot first)."""
        if not hasattr(self, "_qc"):
            print("No mocap client — call connect_mocap() first.")
            return

        T_mocap_base = self._mocap_se3(self._MOCAP_BASE_IDX)
        T_rel        = T_mocap_base.inverse() * self._mocap_se3(0)
        T_base_world = self._prob.base_se3_from_q(self.q_init)
        T_pyl_world  = T_base_world * T_rel

        t    = T_pyl_world.translation.tolist()
        qpin = pin.Quaternion(T_pyl_world.rotation)
        q    = [float(qpin.x), float(qpin.y), float(qpin.z), float(qpin.w)]

        self.update_pylone_pose(t, q)

        result = {
            "pylone_x":    round(t[0], 4),
            "pylone_y":    round(t[1], 4),
            "pylone_z":    round(t[2], 4),
            "pylone_quat": [round(v, 6) for v in q],
        }
        with open(_PYLONE_POSE_FILE, "w") as _f:
            yaml.dump(result, _f, default_flow_style=False)
        print(f"Pylone pose saved to {_PYLONE_POSE_FILE}")

    def update_mocap_frames(self) -> None:
        """Display EE and pylone mocap frames in the Viser viewer."""
        if not hasattr(self, "_qc"):
            print("No mocap client — call connect_mocap() first.")
            return
        if not hasattr(self, "_viewer"):
            print("No viewer — call init_viewer() first.")
            return

        T_mocap_base = self._mocap_se3(self._MOCAP_BASE_IDX)
        T_rel_ee     = T_mocap_base.inverse() * self._mocap_se3(1)
        T_rel_pyl    = T_mocap_base.inverse() * self._mocap_se3(0)
        T_base_world = self._prob.base_se3_from_q(self.q_init)
        T_ee_world   = T_base_world * T_rel_ee
        T_pyl_world  = T_base_world * T_rel_pyl

        def _se3_to_viser(T):
            return T.translation, pin.Quaternion(T.rotation).coeffs()[[3, 0, 1, 2]]

        viser_server = self._viewer.viewer
        if not hasattr(self, "_mocap_viser_frames"):
            self._mocap_viser_frames = {
                "ee":     viser_server.scene.add_frame("mocap/ee",    show_axes=True, axes_length=0.15, axes_radius=0.006),
                "pylone": viser_server.scene.add_frame("mocap/pylone", show_axes=True, axes_length=0.15, axes_radius=0.006),
            }
            print("Mocap frames created: 'mocap/ee' and 'mocap/pylone'.")

        pos, wxyz = _se3_to_viser(T_ee_world)
        self._mocap_viser_frames["ee"].position = pos
        self._mocap_viser_frames["ee"].wxyz     = wxyz
        pos, wxyz = _se3_to_viser(T_pyl_world)
        self._mocap_viser_frames["pylone"].position = pos
        self._mocap_viser_frames["pylone"].wxyz     = wxyz

    # ── Controller activation ─────────────────────────────────────────────────

    def activate_lfc(self) -> None:
        """Switch arm_right_controller → LFC + JSE."""
        subprocess.run([
            "ros2", "control", "switch_controllers",
            "--deactivate", "arm_right_controller",
            "--activate", "linear_feedback_controller", "joint_state_estimator",
        ], check=True)
        print("LFC controllers active.")

    def deactivate_lfc(self) -> None:
        """Switch LFC + JSE → arm_right_controller."""
        subprocess.run([
            "ros2", "control", "switch_controllers",
            "--deactivate", "linear_feedback_controller", "joint_state_estimator",
            "--activate", "arm_right_controller",
        ], check=True)
        print("arm_right_controller active.")
