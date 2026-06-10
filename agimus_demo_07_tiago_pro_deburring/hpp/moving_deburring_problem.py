"""
HPP problem definition for the TIAGo Pro deburring demo.

Encapsulates the HPP robot model, environment (pylone, ground) and constraint
graph.  Free of any ROS or execution logic — import this from the orchestrator.

Usage:
    problem = DeburringProblem(urdf_str)
    problem.update_pylone_pose([x, y, z], [qx, qy, qz, qw])
"""

import os
import tempfile

import numpy as np
import pinocchio as pin
import yaml

from pyhpp.manipulation import Device, urdf
from pyhpp.manipulation import Graph, Problem
from pyhpp.manipulation.constraint_graph_factory import ConstraintGraphFactory
from pyhpp.manipulation.security_margins import SecurityMargins
from pyhpp.constraints import ComparisonType, ComparisonTypes, LockedJoint

_HPP_DIR  = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR  = os.path.join(_HPP_DIR, "..")

ROBOT_SRDF  = os.path.join(_HPP_DIR, "tiago_pro.srdf")
PYLONE_SRDF = os.path.join(_HPP_DIR, "pylone.srdf")
PYLONE_URDF = os.path.join(_PKG_DIR, "urdf", "pylone.urdf")
GROUND_SRDF = os.path.join(_HPP_DIR, "ground.srdf")
GROUND_URDF = os.path.join(_PKG_DIR, "urdf", "ground.urdf")

_PYLONE_POSE_FILE = os.path.join(_PKG_DIR, "config", "pylone_pose.yaml")

_CFG_FILE = os.path.join(_PKG_DIR, "config", "hpp_orchestrator_params.yaml")
with open(_CFG_FILE) as _f:
    _cfg = yaml.safe_load(_f)

HANDLE_NAME = _cfg["handle"]["name"]

LEFT_ARM_TUCK  = _cfg["tuck"]["left_arm"]
RIGHT_ARM_TUCK = _cfg["tuck"]["right_arm"]


class DeburringProblem:
    """
    HPP model + constraint graph for the TIAGo Pro deburring task.

    Attributes
    ----------
    robot, model, problem, graph : HPP objects
    q_init : initial full HPP configuration vector
    base_idx, left_arm_idx, right_arm_idx, pylone_idx : idx_q in HPP model
    pin_data : pinocchio Data for FK
    ee_frame_id : pinocchio frame id for gripper_right_tool_holder
    transition_approach, transition_insert : HPP transitions
    """

    def __init__(self, urdf_str: str):
        print("Loading HPP model …")
        self._setup_model(urdf_str)
        print("Building constraint graph …")
        self._setup_graph()

    # ── Model setup ───────────────────────────────────────────────────────────

    def _setup_model(self, urdf_str: str) -> None:
        _tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        _tmp.write(urdf_str)
        _tmp.close()

        robot = Device("tiago_pro")
        urdf.loadModel(
            robot, 0, "tiago_pro", "planar",
            f"file://{_tmp.name}",
            ROBOT_SRDF,
            pin.SE3.Identity(),
        )
        os.unlink(_tmp.name)
        urdf.loadModel(robot, 0, "ground", "anchor", GROUND_URDF, GROUND_SRDF, pin.SE3.Identity())
        urdf.loadModel(robot, 0, "pylone", "freeflyer", PYLONE_URDF, PYLONE_SRDF, pin.SE3.Identity())

        self.robot = robot
        self.model = robot.model()

        def _idx(name):
            return self.model.joints[self.model.getJointId(name)].idx_q

        self.base_idx       = _idx("tiago_pro/root_joint")
        self.left_arm_idx   = _idx("tiago_pro/arm_left_1_joint")
        self.right_arm_idx  = _idx("tiago_pro/arm_right_1_joint")
        self.pylone_idx     = _idx("pylone/root_joint")

        robot.setJointBounds("tiago_pro/root_joint", [
            -3.0, 3.0, -3.0, 3.0,
            -float("Inf"), float("Inf"), -float("Inf"), float("Inf"),
        ])

        with open(_PYLONE_POSE_FILE) as _pf:
            _pc = yaml.safe_load(_pf)
        _px = _pc["pylone_x"]
        _py = _pc["pylone_y"]
        _pz = _pc["pylone_z"]
        _pq = _pc.get("pylone_quat", [0.0, 0.0, 0.0, 1.0])

        self._set_pylone_bounds(_px, _py, _pz)

        _handle = robot.handles()[HANDLE_NAME]
        # Align our handle Z with the SRDF handle X (insertion axis),
        # so that EE Z = insertion direction for any handle choice.
        _srdf_x = _handle.localPosition.rotation[:, 0]
        _z = _srdf_x / np.linalg.norm(_srdf_x)
        _tmp = np.array([1., 0., 0.]) if abs(_z[0]) < 0.9 else np.array([0., 1., 0.])
        _y = np.cross(_z, _tmp); _y /= np.linalg.norm(_y)
        _x = np.cross(_y, _z)
        _R = np.column_stack([_x, _y, _z])
        _handle.localPosition = pin.SE3(_R, _handle.localPosition.translation)
        _handle.mask = [True, True, True, True, True, True]
        _handle.approachingDirection = np.array([0, 0, 1])

        li = self.left_arm_idx
        ri = self.right_arm_idx
        pi = self.pylone_idx

        self.q_init = pin.neutral(self.model).copy()
        self._left_arm_lock_values = list(LEFT_ARM_TUCK)
        self.q_init[li:li+7] = LEFT_ARM_TUCK
        self.q_init[ri:ri+7] = RIGHT_ARM_TUCK
        self.q_init[pi:pi+3] = [_px, _py, _pz]
        self.q_init[pi+3:pi+7] = _pq

        self.pin_data = self.model.createData()
        ee_frame_name = "tiago_pro/gripper_right_tool_holder"
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        if self.ee_frame_id == self.model.nframes:
            raise RuntimeError(f"Frame '{ee_frame_name}' not found in model")

    # ── Constraint graph setup ─────────────────────────────────────────────────

    def _setup_graph(self, base_lock_q: np.ndarray = None) -> None:
        robot = self.robot
        model = self.model

        problem = Problem(robot)
        graph   = Graph("robot", robot, problem)
        factory = ConstraintGraphFactory(graph)
        graph.maxIterations(40)
        graph.errorThreshold(1e-3)
        factory.setGrippers(["tiago_pro/gripper"])
        factory.setObjects(["pylone"], [[HANDLE_NAME]], [[]])
        factory.generate()

        self.transition_approach = graph.getTransition(
            f"tiago_pro/gripper > {HANDLE_NAME} | f_01"
        )
        self.transition_insert = graph.getTransition(
            f"tiago_pro/gripper > {HANDLE_NAME} | f_12"
        )
        self.transition_free = graph.getTransition("Loop | f")

        _cts1 = ComparisonTypes()
        _cts1[:] = [ComparisonType.EqualToZero]

        _cts3 = ComparisonTypes()
        _cts3[:] = [ComparisonType.EqualToZero,
                    ComparisonType.EqualToZero,
                    ComparisonType.EqualToZero]

        def _lock(joint_name, value):
            j = model.joints[model.getJointId(joint_name)]
            if j.nq == 2 and j.nv == 1:
                locked_val = np.array([np.cos(value), np.sin(value)])
            else:
                locked_val = np.array([value])
            return LockedJoint(robot, joint_name, locked_val, _cts1)

        locked = []

        if base_lock_q is not None:
            bi = self.base_idx
            base_val = np.array([base_lock_q[bi],     base_lock_q[bi + 1],
                                 base_lock_q[bi + 2], base_lock_q[bi + 3]])
            locked.append(LockedJoint(robot, "tiago_pro/root_joint", base_val, _cts3))

        locked.append(_lock("tiago_pro/torso_lift_joint", 0.0))
        for wheel in ["wheel_front_left_joint", "wheel_front_right_joint",
                      "wheel_rear_left_joint",  "wheel_rear_right_joint"]:
            locked.append(_lock(f"tiago_pro/{wheel}", 0.0))
        for i, val in enumerate(self._left_arm_lock_values):
            locked.append(_lock(f"tiago_pro/arm_left_{i+1}_joint", val))
        for name in ["gripper_left_finger_joint",
                     "gripper_left_inner_finger_left_joint",
                     "gripper_left_fingertip_left_joint",
                     "gripper_left_inner_finger_right_joint",
                     "gripper_left_fingertip_right_joint",
                     "gripper_left_outer_finger_right_joint",
                     "gripper_right_tool_mount_joint"]:
            locked.append(_lock(f"tiago_pro/{name}", 0.0))
        locked.append(_lock("tiago_pro/head_1_joint", 0.0))
        locked.append(_lock("tiago_pro/head_2_joint", 0.0))

        graph.addNumericalConstraintsToGraph(locked)

        sm = SecurityMargins(problem, factory, ["tiago_pro", "pylone"], robot)
        sm.setSecurityMarginBetween("tiago_pro", "pylone", 0.02)
        sm.apply()

        for jname in model.names:
            if "tiago_pro" in jname:
                graph.setSecurityMarginForTransition(
                    self.transition_insert, jname, "pylone/root_joint", float("-inf")
                )

        graph.initialize()

        self.problem = problem
        self.graph   = graph

    # ── Base locking via constraint graph ─────────────────────────────────────

    def lock_base_in_graph(self, q_hpp: np.ndarray) -> None:
        """Rebuild constraint graph with base locked at q_hpp's position."""
        self._base_lock_q = np.array(q_hpp)
        self._setup_graph(base_lock_q=self._base_lock_q)

    def unlock_base_in_graph(self) -> None:
        """Rebuild constraint graph with base free."""
        self._base_lock_q = None
        self._setup_graph()

    # ── Pylone pose ────────────────────────────────────────────────────────────

    def _set_pylone_bounds(self, x: float, y: float, z: float, margin: float = 0.001) -> None:
        self.robot.setJointBounds("pylone/root_joint", [
            x - margin, x + margin,
            y - margin, y + margin,
            z - margin, z + margin,
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
        ])

    def update_pylone_pose(self, t: list, q: list = None) -> None:
        """Update pylone pose in q_init.  t=[x,y,z] in HPP world frame."""
        t = np.array(t)
        q = np.array(q) if q is not None else np.array([0., 0., 0., 1.])
        self._set_pylone_bounds(t[0], t[1], t[2])
        pi = self.pylone_idx
        self.q_init[pi:pi+3]   = t
        self.q_init[pi+3:pi+7] = q

    def reload_pylone_pose(self) -> None:
        """Re-read pylone pose from config/pylone_pose.yaml and update q_init."""
        if not os.path.exists(_PYLONE_POSE_FILE):
            print(f"No pylone pose file at {_PYLONE_POSE_FILE}.")
            return
        with open(_PYLONE_POSE_FILE) as f:
            cfg = yaml.safe_load(f)
        self.update_pylone_pose(
            [cfg["pylone_x"], cfg["pylone_y"], cfg["pylone_z"]],
            cfg.get("pylone_quat", [0.0, 0.0, 0.0, 1.0]),
        )

    def list_transitions(self) -> None:
        """Print all transition names in the constraint graph (for debugging)."""
        names = self.graph.getTransitionNames()
        print(f"Transitions ({len(names)}):")
        for name in names:
            print(f"  {name}")

    def rebuild_graph(self, left_arm_values: list) -> None:
        """Rebuild the constraint graph with updated locked left-arm values."""
        self._left_arm_lock_values = left_arm_values
        self._setup_graph(base_lock_q=getattr(self, "_base_lock_q", None))

    # ── FK helpers ────────────────────────────────────────────────────────────

    def base_se3_from_q(self, q: np.ndarray) -> pin.SE3:
        """Extract base SE3 (in HPP world frame) from a full HPP config vector."""
        bi = self.base_idx
        cos_t, sin_t = q[bi + 2], q[bi + 3]
        R = np.array([[cos_t, -sin_t, 0.0],
                      [sin_t,  cos_t, 0.0],
                      [0.0,    0.0,   1.0]])
        return pin.SE3(R, np.array([q[bi], q[bi + 1], 0.0]))

    def fk_ee_rel_base(self, q_full: np.ndarray) -> pin.SE3:
        """FK of gripper_right_tool_holder relative to base_link (for fixed-base MPC)."""
        pin.forwardKinematics(self.model, self.pin_data, q_full)
        pin.updateFramePlacements(self.model, self.pin_data)
        T_base_world = self.base_se3_from_q(q_full)
        return T_base_world.inverse() * self.pin_data.oMf[self.ee_frame_id]

    def fk_ee_world(self, q_full: np.ndarray) -> pin.SE3:
        """FK of gripper_right_tool_holder in HPP world frame."""
        pin.forwardKinematics(self.model, self.pin_data, q_full)
        pin.updateFramePlacements(self.model, self.pin_data)
        return self.pin_data.oMf[self.ee_frame_id].copy()
