"""
HPP pick-and-drop orchestrator for TIAGo Pro — fixed base (left arm only).

The robot picks a configurable T-LESS object from a table at 74.5 cm height
using the left arm (pal-pro-gripper), carries it to a configurable drop zone,
and releases it into a box positioned above the table there (urdf/box.urdf,
config box.{x,y,z}). The box is NOT part of the HPP scene while p1/p2/p2b/p3
plan and execute — it's simply absent from self.robot/self.model (see
_load_urdf_models) until add_box_to_scene() adds it, right after the base
has navigated to the drop zone, at which point p_place/p4/p5 are planned
(see plan_place()) and do collision-check against it for real. Both planning
and execution are split into matching pick/place entry points: plan_pick()
and execute_pick() cover p1/p2/p2b/p3 (box-less scene); plan_place() and
execute_place() cover navigating to the drop zone, adding the box to the
scene, and p_place/p4/p5 (box present). execute() (auto_gripper mode) just
calls execute_pick() then execute_place() in sequence; either half is also
callable on its own.

Interactive usage (IPython):
    o = Orchestrator()
    o.plan_pick()         # plan p1 (approach) + p2 (grasp) + p2b (retract) + p3 (carry)
    o.execute()           # run full sequence: execute_pick() (p1 → close gripper →
                           # p2b (retract) → p3) then execute_place() (navigate →
                           # add box to scene → plan_place() + run p_place →
                           # open gripper → p4 (release) → p5 (return) →
                           # navigate back to initial point, box removed again)
    o.plan_and_execute()  # plan_pick() then execute()

Phase labels:
    p1 — approach  : free arm motion to pre-grasp pose
    p2 — grasp     : close-in motion until gripper contacts object
    p2b — retract  : short pull-back, with the object grasped, to that handle's
                      own SRDF clearance distance (the same offset already used
                      to place the pre-grasp waypoint qpg) before the arm makes
                      its large motion to the carry pose in p3
    p3 — carry     : arm moves to a transport pose (CARRY_ARM_CFG) with object grasped
    p_place        : arm moves from the transport pose to the drop zone, planned and
                      executed after the base has navigated there and the box has
                      been added to the scene (add_box_to_scene). The target arm
                      config is not hardcoded — it's solved via IK (_find_drop_config)
                      so the object ends up BOX_CLEARANCE above box.{x,y,z}'s rim,
                      keeping its grasped orientation. Moving the box in the config
                      moves where p_place ends up; no manual arm-config retuning needed.
    p4 — release   : arm retreats from drop zone (planned from q_drop when carry exists)
    p5 — return    : arm moves back to the transport pose (CARRY_ARM_CFG), empty-handed,
                      before the base navigates back to the initial point

execute_pick() closes the gripper after p2, retracts to the grasped handle's
own clearance distance (p2b), then moves the arm to the transport pose (p3).
execute_place() then sends the base to NAV_TARGET_{X,Y,YAW} via the
navigate_to_pose action, adds the drop-zone box to the HPP scene and plans
p_place/p4/p5 against it (plan_place()), moves the arm on to the drop pose
(p_place), opens the gripper and runs p4 (retract) followed by p5 (back to
the carry pose). Once p5 has run (or been skipped), the base navigates back
to NAV_INITIAL_{X,Y,YAW} (removing the box from the scene again) so a new
cycle can start from the same place, arm already tucked in the carry pose.
Pass auto_gripper=False to stream all phases without gripper/navigation commands.
"""

import os
import re
import subprocess
import sys
import glob
import copy
from contextlib import contextmanager
import tempfile
import time
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

from pyhpp.manipulation import Device, urdf
from pyhpp.manipulation import Graph, GraphRandomShortcut, Problem, TransitionPlanner
from pyhpp.manipulation.constraint_graph_factory import ConstraintGraphFactory
from pyhpp.manipulation.security_margins import SecurityMargins
from pyhpp.constraints import ComparisonType, ComparisonTypes, LockedJoint
from pyhpp.core import SplineGradientBased_bezier3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from action_msgs.msg import GoalStatus
from agimus_msgs.msg import MpcInput, MpcEEInput
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from vision_msgs.msg import Detection2DArray


# ── File paths ────────────────────────────────────────────────────────────────

_HPP_DIR   = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR   = os.path.join(_HPP_DIR, "..")

ROBOT_SRDF  = os.path.join(_HPP_DIR, "tiago_pro.srdf")
GROUND_SRDF = os.path.join(_HPP_DIR, "ground.srdf")
GROUND_URDF = os.path.join(_PKG_DIR, "urdf", "ground.urdf")
TABLE_SRDF  = os.path.join(_HPP_DIR, "table.srdf")
TABLE_URDF  = os.path.join(_PKG_DIR, "urdf", "table.urdf")
TABLE_OFFSET_X = 0.85   # shift toward robot (m)
BOX_SRDF    = os.path.join(_HPP_DIR, "box.srdf")
BOX_URDF    = os.path.join(_PKG_DIR, "urdf", "box.urdf")

_CFG_FILE = os.path.join(_PKG_DIR, "config", "hpp_orchestrator_params.yaml")
with open(_CFG_FILE) as _f:
    _cfg = yaml.safe_load(_f)

DT         = _cfg["trajectory"]["dt"]
TIME_SCALE = _cfg["trajectory"]["time_scale"]
CARRY_TIME_SCALE = _cfg["trajectory"].get("carry_time_scale", TIME_SCALE)

_obj_cfg    = _cfg["object"]
_DEFAULT_OBJ_NAME = _obj_cfg["name"]
OBJECT_DATASET = _obj_cfg.get("dataset", "tless")


def _happypose_class_id(obj_name: str, dataset: str) -> str:
    num_part = obj_name.rsplit("_", 1)[-1]
    return f"{dataset}-obj_{int(num_part):06d}"


_CLASS_ID_RE = re.compile(r"^(?P<dataset>.+)-obj_(?P<num>\d+)$")


def _obj_name_from_class_id(class_id: str, dataset: str):
    """Reverse of _happypose_class_id, e.g. 'tless-obj_000023' -> 'obj_23'.
    Returns None if class_id doesn't match the expected dataset/format."""
    m = _CLASS_ID_RE.match(class_id)
    if not m or m.group("dataset") != dataset:
        return None
    return f"obj_{int(m.group('num')):02d}"


_OBJ_ASSET_RE = re.compile(r"^(obj_\d+)\.(srdf|urdf)$")


def _list_available_objects() -> list:
    """Names of objects that ship BOTH a .srdf (in hpp/) and .urdf (in urdf/)
    asset — this is what "objects present in the hpp folder" resolves to."""
    def _names(base_dir, ext):
        names = set()
        for fname in os.listdir(base_dir):
            m = _OBJ_ASSET_RE.match(fname)
            if m and m.group(2) == ext:
                names.add(m.group(1))
        return names

    srdf_names = _names(_HPP_DIR, "srdf")
    urdf_names = _names(os.path.join(_PKG_DIR, "urdf"), "urdf")
    return sorted(srdf_names & urdf_names)


def _resolve_object_asset_path(base_dir: str, extension: str, obj_name: str) -> str:
    object_basename = obj_name.rsplit("/", 1)[-1]
    path = os.path.join(base_dir, f"{object_basename}{extension}")
    if os.path.exists(path):
        return path

    raise FileNotFoundError(
        f"Configured object '{obj_name}' expects asset '{path}', but it was not found. "
        f"Available objects: {', '.join(_list_available_objects())}"
    )


def _patch_viser_tab_group_remove_bug() -> None:
    """viser's GuiTabGroupHandle.remove() marks itself removed before removing
    its child tabs, but each child tab.remove() writes back to the (now
    "removed") parent's _tab_labels/_tab_icons_html/_tab_handles props, which
    raises "Cannot assign to '_tab_labels' on a removed GuiTabGroupHandle."
    This hits us every time pyhpp_viser.Viewer.initViewer() calls
    viewer.gui.reset() to swap in a newly detected object's model. Patch the
    method so children are removed before the parent is marked removed."""
    from viser._gui_handles import GuiTabGroupHandle
    from viser._messages import GuiRemoveMessage

    if getattr(GuiTabGroupHandle.remove, "_agimus_patched", False):
        return

    def remove(self) -> None:
        if self._impl.removed:
            import warnings
            warnings.warn(
                f"Attempted to remove an already removed {type(self).__name__}.",
                stacklevel=2,
            )
            return
        for tab in tuple(self._tab_handles):
            tab.remove()
        self._impl.removed = True
        gui_api = self._impl.gui_api
        gui_api._websock_interface.queue_message(GuiRemoveMessage(self._impl.uuid))
        parent = gui_api._container_handle_from_uuid[self._impl.parent_container_id]
        parent._children.pop(self._impl.uuid)

    remove._agimus_patched = True
    GuiTabGroupHandle.remove = remove


OBJ_INIT_POS = np.array([_obj_cfg["x"], _obj_cfg["y"], _obj_cfg["z"]])

_box_cfg = _cfg["box"]
BOX_POS  = np.array([_box_cfg["x"], _box_cfg["y"], _box_cfg["z"]])
BOX_CLEARANCE = _box_cfg.get("clearance", 0.05)
# Bottom thickness + wall height from box.urdf -- keep in sync there.
BOX_WALL_TOP_OFFSET = 0.003 + 0.08

LEFT_ARM_TUCK  = _cfg["tuck"]["left_arm"]
RIGHT_ARM_TUCK = _cfg["tuck"]["right_arm"]
CARRY_ARM_CFG  = np.array(_cfg["carry"]["arm_config"])

_nav_target_cfg = _cfg.get("nav", {}).get("target_pose", {})
NAV_TARGET_FRAME = _nav_target_cfg.get("frame_id", "map")
NAV_TARGET_X   = _nav_target_cfg.get("x", 0.0)
NAV_TARGET_Y   = _nav_target_cfg.get("y", 0.0)
NAV_TARGET_YAW = _nav_target_cfg.get("yaw", 0.0)

_nav_initial_cfg = _cfg.get("nav", {}).get("initial_pose", {})
NAV_INITIAL_FRAME = _nav_initial_cfg.get("frame_id", "map")
NAV_INITIAL_X   = _nav_initial_cfg.get("x", 0.0)
NAV_INITIAL_Y   = _nav_initial_cfg.get("y", 0.0)
NAV_INITIAL_YAW = _nav_initial_cfg.get("yaw", 0.0)

NAVIGATE_TO_POSE_ACTION = "navigate_to_pose"

_w          = _cfg["weights"]
W_Q         = np.array(_w["w_q"])
W_QDOT      = np.array(_w["w_qdot"])
W_QDDOT     = np.array(_w["w_qddot"])
W_EFFORT    = np.array(_w["w_effort"])
W_COLLISION = _w["w_collision"]
W_FRAME_TRANS = np.array(_w["w_frame_trans"])
W_FRAME_ROT   = np.array(_w["w_frame_rot"])

GRIPPER_OPEN_POSITION = 0.07     # m, fingertip travel used as the HPP planning target
HPP_FIXED_JOINT_EPS = 1e-6        # +/- bound width used to "freeze" a joint for HPP planning
ACTIVE_ARM_JOINT_SHRINK_RATIO = 0.95  # HPP plans within this fraction of each active-arm joint's real range
TABLE_COLLISION_MAX_LIFT = 0.10   # max upward correction (m) before giving up
TABLE_COLLISION_STEP     = 0.005  # z increment per collision-check iteration (m)
# Each gripper is driven in HPP by a single open/close scalar (0=closed,
# GRIPPER_OPEN_POSITION=open), but every finger joint in the URDF has its own
# range and sign. These multipliers convert that one scalar into a per-joint
# target so the whole gripper closes as one rigid mechanism.
LEFT_GRIPPER_HPP_JOINT_MULTIPLIERS = (
    ("gripper_left_finger_joint", 1.0),
    ("gripper_left_left_finger_joint", 1.0),
    ("gripper_left_finger_right_joint", 0.22),
    ("gripper_left_inner_finger_left_joint", -8.28),
    ("gripper_left_inner_finger_right_joint", -8.28),
    ("gripper_left_outer_finger_left_joint", -8.28),
    ("gripper_left_outer_finger_right_joint", -8.28),
    ("gripper_left_fingertip_left_joint", 8.28),
    ("gripper_left_fingertip_right_joint", 8.28),
)
RIGHT_GRIPPER_HPP_JOINT_MULTIPLIERS = (
    ("gripper_right_finger_joint", 1.0),
    ("gripper_right_left_finger_joint", 1.0),
    ("gripper_right_finger_right_joint", 0.22),
    ("gripper_right_inner_finger_left_joint", -8.28),
    ("gripper_right_inner_finger_right_joint", -8.28),
    ("gripper_right_outer_finger_left_joint", -8.28),
    ("gripper_right_outer_finger_right_joint", -8.28),
    ("gripper_right_fingertip_left_joint", 8.28),
    ("gripper_right_fingertip_right_joint", 8.28),
)
LEFT_GRIPPER_RELEASE_SERVICE = "/gripper_left_grasper_srv/release"
LEFT_GRIPPER_GRASP_SERVICE = "/gripper_left_grasper_srv/grasp"
DEFAULT_FIXED_JOINT_VALUES = (
    ("torso_lift_joint", 0.25),
    ("wheel_front_left_joint", 0.0),
    ("wheel_front_right_joint", 0.0),
    ("wheel_rear_left_joint", 0.0),
    ("wheel_rear_right_joint", 0.0),
    ("head_1_joint", 0.0),
    ("head_2_joint", 0.0),
)
# Thresholds for warning that the robot's actual state has drifted from a
# planned path's start point (see _warn_if_robot_far_from_path_start) —
# usually a sign that plan_pick() was run against a stale q_init.
PATH_START_EE_WARN_MM = 100.0
PATH_START_JOINT_WARN_RAD = 0.35
# Publishing the last trajectory point only means the reference has been
# sent — the torque-controlled arm can still be catching up to it. These
# bound how long _wait_for_arm_settled waits for /joint_states to confirm
# the arm actually reached a path's final configuration before the next
# phase (e.g. a gripper action) begins. The check is on end-effector pose,
# not per-joint error: the MPC/OCP weights below prioritize Cartesian
# tracking over exact joint posture (w_frame_trans/w_frame_rot >> w_q), so
# a redundant joint can settle into a different null-space posture than HPP
# planned — per-joint error then stays large even once the end-effector
# (what actually matters for approach/grasp) has arrived.
ARM_SETTLE_EE_POS_TOLERANCE_MM = 15.0  # mm, max end-effector position error to consider the arm "arrived"
ARM_SETTLE_EE_ROT_TOLERANCE_DEG = 5.0  # deg, max end-effector orientation error to consider the arm "arrived"
ARM_SETTLE_TIMEOUT = 10.0           # s, max extra wait for /joint_states to confirm arrival
# agimus_controller_node's trajectory buffer needs >= ocp.horizon_size (50)
# points queued or it pads the tail with a repeated (stale) point every
# cycle rather than skipping the solve; publishing at exactly its own 100 Hz
# consumption rate leaves that buffer with ~zero margin, so it was observed
# oscillating right on that floor continuously, degrading the OCP's horizon
# every cycle for big, fast motions like p3. Burst-publish this many
# messages up front (no DT pacing) before switching to the normal paced
# loop, so the buffer starts with real headroom instead of teetering on the
# minimum — 100 matches agimus_controller_node's own "comfortable startup"
# threshold (buffer_has_enough_data(2) == 2x horizon_size at node init).
PUBLISH_BURST_COUNT = 100
# /joint_states may be published with different QoS settings depending on
# the source node, so we subscribe with all three and take whichever message
# arrives first.
JOINT_STATE_QOS_PROFILES = (
    QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    ),
    QoSProfile(
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.RELIABLE,
    ),
    QoSProfile(
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    ),
)

# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Interactive orchestrator for HPP pick-and-drop planning and MPC execution.
    Fixed-base version: uses the left arm (7 DOF) with pal-pro-gripper only.

    Usage (in IPython):
        o = Orchestrator()
        o.plan_pick()         # generates p1, p2, p2b, p3
        o.execute()           # runs full sequence with automatic gripper control;
                               # plan_place() runs internally after navigation
        o.execute([o.p1])     # run approach only (no auto-gripper)
    """

    def __init__(self, ros_node: Node = None):
        self._ros_node = ros_node
        self.p1 = self.p2 = self.p_retract = self.p3 = self.p_place = self.p4 = self.p5 = None
        self.qpg = self.qg = self.q_retract = self.q_carry = self.q_drop = None
        self._messages = None
        self._last_hold_msg = None
        self._last_executed_q = None
        self._last_executed_label = None
        self._next_msg_id = 0
        self._fixed_joint_values = dict(DEFAULT_FIXED_JOINT_VALUES)
        self._obj_name = _DEFAULT_OBJ_NAME
        # The drop-zone box isn't part of the HPP scene until
        # add_box_to_scene() adds it, right after the base has navigated
        # there (see plan_place) — p1/p2/p2b/p3 plan a scene that
        # genuinely has no box in it.
        self._box_loaded = False
        self._latest_joint_state_map = {}
        self._joint_state_subs = []
        self._seed_candidates = []
        # Grasping isn't reliable in simulation — set this to False to skip
        # _check_object_grasped()'s abort when the topic is technically
        # published but always reports no grasp (e.g. Gazebo testing).
        self.enforce_grasp_check = True
        # Navigation isn't always reliable/available in simulation — set this
        # to False to skip actually sending navigate_to_pose goals and
        # pretend the base arrived immediately (e.g. when testing the rest
        # of the cycle without a working nav2 stack).
        self.enforce_navigation = True

        print("Loading HPP model …")
        self._setup_model()
        print("Building constraint graph …")
        self._setup_graph()
        print("Orchestrator ready.  Call plan_pick() to start.\n")

    # ── Model setup ───────────────────────────────────────────────────────────

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

    # ── Namespace resolution ──────────────────────────────────────────────
    # Joint/frame/gripper names may or may not carry a "tiago_pro/" prefix,
    # depending on whether they come from the loaded Pinocchio model, the
    # HPP gripper registry, or a live /joint_states message. The helpers
    # below try both the prefixed and unprefixed form (and, failing that,
    # any name in the model ending in "/<candidate>") before giving up.

    @staticmethod
    def _name_candidates(name: str) -> list:
        candidates = [name]
        if "/" in name:
            candidates.append(name.split("/", 1)[1])
        else:
            candidates.append(f"tiago_pro/{name}")
        return list(dict.fromkeys(candidates))

    def _resolve_model_name(
        self,
        name: str,
        resolver,
        upper_bound: int,
        available_names,
        kind: str,
    ) -> str:
        candidates = self._name_candidates(name)
        for candidate in candidates:
            item_id = resolver(candidate)
            if item_id < upper_bound:
                return candidate

        suffix_matches = [
            item_name for item_name in available_names
            if any(item_name.endswith(f"/{candidate}") for candidate in candidates)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        raise KeyError(f"{kind} '{name}' not found in loaded model.")

    def _resolve_joint_name(self, name: str) -> str:
        return self._resolve_model_name(
            name,
            self.model.getJointId,
            len(self.model.joints),
            [str(joint_name) for joint_name in self.model.names],
            "Joint",
        )

    def _try_resolve_joint_name(self, name: str):
        try:
            return self._resolve_joint_name(name)
        except KeyError:
            return None

    def _resolve_frame_name(self, name: str) -> str:
        return self._resolve_model_name(
            name,
            self.model.getFrameId,
            self.model.nframes,
            [frame.name for frame in self.model.frames],
            "Frame",
        )

    def _resolve_arm_joint_names(self, side: str, required: bool = True) -> list:
        names = []
        for i in range(1, 8):
            joint_name = self._try_resolve_joint_name(f"arm_{side}_{i}_joint")
            if joint_name is None:
                if not names and not required:
                    return []
                raise KeyError(
                    f"Could not resolve the full {side} arm chain in the loaded model."
                )
            names.append(joint_name)
        return names

    def _resolve_hpp_gripper_name(self, name: str) -> str:
        candidates = []
        if "/" in name:
            candidates.append(name)
            candidates.append(name.split("/", 1)[1])
        else:
            candidates.append(f"tiago_pro/{name}")
            candidates.append(name)

        try:
            grippers = self.robot.grippers()
            if hasattr(grippers, "keys"):
                available = [str(gripper_name) for gripper_name in grippers.keys()]
            else:
                available = []
        except Exception:
            available = []

        if available:
            for candidate in candidates:
                if candidate in available:
                    return candidate

            suffix_matches = [
                gripper_name for gripper_name in available
                if any(gripper_name.endswith(f"/{candidate}") for candidate in candidates)
            ]
            if len(suffix_matches) == 1:
                return suffix_matches[0]

        return candidates[0]

    def _lookup_namespaced_value(self, values: dict, name: str):
        candidates = self._name_candidates(name)
        for candidate in candidates:
            if candidate in values:
                return values[candidate]

        suffix_matches = [
            value for item_name, value in values.items()
            if any(str(item_name).endswith(f"/{candidate}") for candidate in candidates)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        return None

    def _joint_state_position(self, js_map: dict, joint_name: str, default: float = 0.0) -> float:
        value = self._lookup_namespaced_value(js_map, joint_name)
        if value is None:
            return default
        return value

    def _arm_joint_state(self, js_map: dict, side: str):
        joint_names = getattr(self, f"_{side}_arm_joint_names")
        if not joint_names:
            return None
        return np.array([
            self._joint_state_position(js_map, joint_name, 0.0)
            for joint_name in joint_names
        ])

    def _set_joint_configuration_value(self, q: np.ndarray, joint_name: str, value: float) -> None:
        """Write `value` into q at `joint_name`'s slot, in whatever encoding
        Pinocchio uses for that joint type."""
        resolved_name = self._resolve_joint_name(joint_name)
        joint = self.model.joints[self.model.getJointId(resolved_name)]
        if joint.nq == 2 and joint.nv == 1:
            # Continuous (unbounded revolute) joints are stored as [cos, sin]
            # in the configuration vector, not as a raw angle, so Pinocchio
            # can represent them without a +/-pi wraparound discontinuity.
            q[joint.idx_q:joint.idx_q + 2] = [np.cos(value), np.sin(value)]
        else:
            q[joint.idx_q:joint.idx_q + joint.nq] = [value]

    def _exclude_joint_from_hpp_planning(self, joint_name: str, value: float) -> None:
        """Freeze a scalar joint at the model level so HPP does not plan over it."""
        resolved_name = self._resolve_joint_name(joint_name)
        joint = self.model.joints[self.model.getJointId(resolved_name)]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(
                f"Cannot freeze joint '{resolved_name}' with bounds "
                f"(nq={joint.nq}, nv={joint.nv})."
            )
        value = float(value)
        self.robot.setJointBounds(
            resolved_name,
            [value - HPP_FIXED_JOINT_EPS, value + HPP_FIXED_JOINT_EPS],
        )

    def _shrink_active_arm_joint_range(self, ratio: float = ACTIVE_ARM_JOINT_SHRINK_RATIO) -> None:
        """
        Reduce each active-arm joint's planning range to `ratio` of its full
        URDF range, centered on the same midpoint (mirrors
        hpp.corbaserver.robot.shrinkJointRange — pyhpp's Device has no
        equivalent, and no getJointBounds to read current bounds from, so
        this is reimplemented against Pinocchio's own model limits). Updates
        both self.robot.setJointBounds (what HPP's planner/graph actually
        samples and enforces) and self.model.lowerPositionLimit/
        upperPositionLimit (read directly by _find_drop_config's own IK-seed
        bounds check) so the two stay consistent. Keeps the arm away from its
        hard joint limits, since HPP's own pathValidation doesn't reliably
        catch out-of-bounds solutions for this redundant 7-DOF arm.
        """
        arm_idx = self._active_arm_idx
        self._active_arm_shrunk_bounds = []
        for i, joint_name in enumerate(self._active_arm_joint_names):
            idx_q = arm_idx + i
            lower = float(self.model.lowerPositionLimit[idx_q])
            upper = float(self.model.upperPositionLimit[idx_q])
            half_width = 0.5 * ratio * (upper - lower)
            mean = 0.5 * (upper + lower)
            new_lower, new_upper = mean - half_width, mean + half_width
            self.robot.setJointBounds(joint_name, [new_lower, new_upper])
            self.model.lowerPositionLimit[idx_q] = new_lower
            self.model.upperPositionLimit[idx_q] = new_upper
            self._active_arm_shrunk_bounds.append((new_lower, new_upper))

    def _project_into_active_arm_range(self, q: np.ndarray, tol: float = 1e-3) -> np.ndarray:
        """Clamp the active arm's 7 joint values in `q` into their shrunk
        planning range (mirrors pyhpp's missing projectInJointRange) —
        needed because a live robot pose read from /joint_states can sit in
        the outer range _shrink_active_arm_joint_range no longer considers
        valid for planning."""
        arm_idx = self._active_arm_idx
        for i, (lower, upper) in enumerate(self._active_arm_shrunk_bounds):
            q[arm_idx + i] = np.clip(q[arm_idx + i], lower + tol, upper - tol)
        return q

    def _gripper_hpp_targets(self, position: float, joint_multipliers: tuple) -> dict:
        targets = {}
        for joint_name, multiplier in joint_multipliers:
            resolved_name = self._try_resolve_joint_name(joint_name)
            if resolved_name is None or resolved_name in targets:
                continue
            targets[resolved_name] = float(multiplier * position)
        return targets

    def _sync_fixed_joint_values(self, js_map: dict) -> dict:
        synced = {}
        for joint_name in self._fixed_joint_values:
            value = self._lookup_namespaced_value(js_map, joint_name)
            if value is None:
                continue
            value = float(value)
            self._fixed_joint_values[joint_name] = value
            synced[joint_name] = value
        return synced

    def _configuration_from_joint_state(self, js_map: dict, q_seed=None) -> np.ndarray:
        q = self.q_init.copy() if q_seed is None else np.array(q_seed, copy=True)

        for joint_name, default_value in self._fixed_joint_values.items():
            try:
                joint_value = self._joint_state_position(js_map, joint_name, default_value)
                self._set_joint_configuration_value(q, joint_name, joint_value)
            except KeyError:
                continue

        left_arm = self._arm_joint_state(js_map, "left")
        if left_arm is not None:
            q[self._left_arm_idx:self._left_arm_idx + 7] = left_arm

        if self._right_arm_idx is not None:
            right_arm = self._arm_joint_state(js_map, "right")
            if right_arm is not None:
                q[self._right_arm_idx:self._right_arm_idx + 7] = right_arm

        return q

    def _as_full_configuration(self, q_ref) -> np.ndarray:
        """Accept either a full robot configuration or just the 7 active-arm
        values, and return a full configuration either way — lets callers
        like compare_pose() pass either form."""
        q_ref = np.array(q_ref, copy=True)
        if q_ref.ndim != 1:
            raise ValueError(
                f"Expected a 1-D configuration, got shape {q_ref.shape}."
            )

        if q_ref.shape[0] == self.robot.configSize():
            return q_ref

        if q_ref.shape[0] == len(self._active_arm_joint_names):
            q_full = self.q_init.copy()
            arm_idx = self._active_arm_idx
            q_full[arm_idx:arm_idx + q_ref.shape[0]] = q_ref
            return q_full

        raise ValueError(
            f"Unsupported configuration size {q_ref.shape[0]} for compare_pose()."
        )

    def _apply_locked_defaults_to_q(self, q: np.ndarray) -> None:
        """Fill in everything that ISN'T actively planned: fixed base/torso/
        head joints, the excluded gripper-finger joints, the active arm's
        home posture, and the other arm's tucked/locked posture. Used to
        build q_init and to re-seed it when the robot state is resynced."""
        for joint_name, value in self._fixed_joint_values.items():
            try:
                self._set_joint_configuration_value(q, joint_name, value)
            except KeyError:
                pass
        for joint_name, value in self._gripper_joint_targets.items():
            self._set_joint_configuration_value(q, joint_name, value)
        for joint_name, value in zip(self._active_arm_joint_names, self._active_arm_home):
            self._set_joint_configuration_value(q, joint_name, value)
        for joint_name, value in zip(self._other_arm_joint_names, self._other_arm_lock_values):
            self._set_joint_configuration_value(q, joint_name, value)

    def _setup_model(self):
        """Build self.robot/model from scratch (or rebuild it when the
        detected object changes — see update_object_pose_from_happypose)."""
        self._load_urdf_models()
        self._resolve_joint_topology()
        self._shrink_active_arm_joint_range()

        self._set_obj_bounds(*OBJ_INIT_POS)
        self._set_table_bounds()

        self.q_init = pin.neutral(self.model).copy()
        obj_idx = self._obj_idx
        table_idx = self._table_idx
        self._active_arm_home = list(LEFT_ARM_TUCK)
        self._other_arm_lock_values = (
            list(RIGHT_ARM_TUCK) if self._other_arm_idx is not None else []
        )
        # Init: active arm at home, graph-locked joints on-manifold, and the
        # HPP-excluded gripper fingers fixed to the Gazebo-controlled open pose.
        self._apply_locked_defaults_to_q(self.q_init)
        self.q_init[obj_idx:obj_idx+3] = OBJ_INIT_POS
        self.q_init[obj_idx+3:obj_idx+7] = [0., 0., 0., 1.]  # identity quaternion
        self.q_init[table_idx:table_idx+3] = [TABLE_OFFSET_X, 0, 0]
        self.q_init[table_idx+3:table_idx+7] = [0., 0., 0., 1.]  # identity quaternion

        self._pin_data = self.model.createData()
        self._ee_frame_id = self.model.getFrameId(self._ee_frame_name)
        if self._ee_frame_id == self.model.nframes:
            raise RuntimeError(f"Frame '{self._ee_frame_name}' not found in model")

    def _load_urdf_models(self):
        """Load the robot (from the live /robot_description), ground, table,
        and the configured object's URDF/SRDF pair into a fresh HPP Device."""
        urdf_str = self._fetch_robot_urdf()
        _tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        _tmp.write(urdf_str)
        _tmp.close()

        robot = Device("tiago_pro")
        urdf.loadModel(
            robot, 0, "tiago_pro", "anchor",
            f"file://{_tmp.name}",
            ROBOT_SRDF,
            pin.SE3.Identity(),
        )
        os.unlink(_tmp.name)

        urdf.loadModel(
            robot, 0, "ground", "anchor",
            GROUND_URDF, GROUND_SRDF,
            pin.SE3.Identity(),
        )
        urdf.loadModel(
            robot, 0, "table", "freeflyer",
            TABLE_URDF, TABLE_SRDF,
            pin.SE3.Identity(),
        )
        # Drop-zone box: not part of the scene at all until
        # add_box_to_scene() sets self._box_loaded and rebuilds — it's
        # irrelevant during p1/p2/p2b/p3 (near the pick location, not the
        # drop zone), so it simply isn't modeled then. It never moves, so
        # once loaded it's a fixed "anchor" (like ground) placed directly
        # at BOX_POS, with no bounds/LockedJoint needed.
        if self._box_loaded:
            urdf.loadModel(
                robot, 0, "box", "anchor",
                BOX_URDF, BOX_SRDF,
                pin.SE3(np.eye(3), BOX_POS),
            )
        self._obj_srdf = _resolve_object_asset_path(_HPP_DIR, ".srdf", self._obj_name)
        self._obj_urdf = _resolve_object_asset_path(
            os.path.join(_PKG_DIR, "urdf"), ".urdf", self._obj_name
        )
        urdf.loadModel(
            robot, 0, self._obj_name, "freeflyer",
            self._obj_urdf, self._obj_srdf,
            pin.SE3(np.eye(3), np.array([0, 0, 0])),
        )

        self.robot = robot
        self.model = robot.model()

        # Grasp candidates come straight from whatever handles are defined
        # in the object's SRDF, so this works unmodified for any T-LESS
        # object — no per-object handle list to maintain in config.
        self._handle_names = sorted(
            entry.key() for entry in robot.handles()
            if entry.key().startswith(f"{self._obj_name}/")
        )

    def _resolve_joint_topology(self):
        """Resolve gripper/frame/arm joint names in the loaded model, build
        the HPP-excluded gripper-finger targets, and record the active
        (left) vs other (right) arm indices. The left arm is the only one
        used for picking — see module docstring."""
        model = self.model

        def _idx(name):
            joint_name = self._resolve_joint_name(name)
            return model.joints[model.getJointId(joint_name)].idx_q

        self._gripper_name = self._resolve_hpp_gripper_name("gripper_left")
        self._ee_frame_msg_name = "arm_left_tool_link"
        self._ee_frame_name = self._resolve_frame_name(self._ee_frame_msg_name)
        self._left_arm_joint_names = self._resolve_arm_joint_names("left")
        self._right_arm_joint_names = self._resolve_arm_joint_names(
            "right", required=False
        )
        self._gripper_joint_targets = {}
        self._gripper_joint_targets.update(
            self._gripper_hpp_targets(
                GRIPPER_OPEN_POSITION, LEFT_GRIPPER_HPP_JOINT_MULTIPLIERS
            )
        )
        self._gripper_joint_targets.update(
            self._gripper_hpp_targets(
                GRIPPER_OPEN_POSITION, RIGHT_GRIPPER_HPP_JOINT_MULTIPLIERS
            )
        )
        # Gripper fingers are driven directly by the real gripper mechanism
        # (see open_gripper/close_gripper) rather than planned by HPP, so
        # freeze them here at their open target.
        for joint_name, value in self._gripper_joint_targets.items():
            self._exclude_joint_from_hpp_planning(joint_name, value)
        self._left_arm_idx  = _idx(self._left_arm_joint_names[0])
        self._right_arm_idx = (
            _idx(self._right_arm_joint_names[0])
            if self._right_arm_joint_names else None
        )
        self._obj_idx       = _idx(f"{self._obj_name}/root_joint")
        self._table_idx     = _idx("table/root_joint")

        # Use LEFT arm for picking (has pal-pro-gripper with actual gripper mechanism)
        self._active_arm_idx = self._left_arm_idx
        # Tangent-space (velocity) index for the arm's first joint -- NOT the
        # same as _active_arm_idx (a config-space index): earlier joints in
        # the tree with nq != nv (e.g. any multi-DOF joint before the arm)
        # make idx_q and idx_v diverge. Needed for Jacobian-based IK
        # (_solve_ee_ik) -- confirmed empirically to differ (idx_q=9 vs
        # idx_v=5 on this model), so never assume they coincide.
        self._active_arm_idx_v = model.joints[
            model.getJointId(self._resolve_joint_name(self._left_arm_joint_names[0]))
        ].idx_v
        self._active_arm_side = "left"
        self._active_arm_joint_names = self._left_arm_joint_names
        self._other_arm_idx = self._right_arm_idx
        self._other_arm_side = "right" if self._right_arm_joint_names else None
        self._other_arm_joint_names = self._right_arm_joint_names

    def _set_obj_bounds(self, x, y, z, margin: float = 1.0):
        """Lock object position with tight bounds so HPP cannot move it freely."""
        self.robot.setJointBounds(f"{self._obj_name}/root_joint", [
            x - margin, x + margin,
            y - margin, y + margin,
            z - margin, z + margin,
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
        ])

    def _set_table_bounds(self, margin: float = 0.001):
        """Give the table's freeflyer root joint a tight but finite
        translation box, so the raw configuration shooter (which requires
        genuinely bounded limits to sample from) can draw from it; actual
        pinning — including rotation, which bounds cannot constrain on a
        freeflyer — is done by the LockedJoint added in `_locked_joints()`."""
        x = TABLE_OFFSET_X
        self.robot.setJointBounds("table/root_joint", [
            x - margin, x + margin,
            -margin, margin,
            -margin, margin,
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
            -float("Inf"), float("Inf"),
        ])

    # ── Constraint graph setup ─────────────────────────────────────────────────

    def _setup_graph(self):
        robot  = self.robot
        model  = self.model

        problem = Problem(robot)
        graph   = Graph("robot", robot, problem)
        factory = ConstraintGraphFactory(graph)
        graph.maxIterations(40)
        graph.errorThreshold(1e-3)
        factory.setGrippers([self._gripper_name])
        factory.setObjects([self._obj_name], [self._handle_names], [[]])
        factory.generate()

        self._build_grasp_candidates(graph)

        locked = self._locked_joints(robot, model)
        graph.addNumericalConstraintsToGraph(locked)

        # Robot/object collision margin is 0 by default; grasp/carry
        # transitions below additionally disable it outright so the gripper
        # can actually close in on the handle. The box, when loaded (see
        # add_box_to_scene — it's absent during p1/p2/p2b/p3), gets a real
        # margin here too, applying to every transition including "carry",
        # which by the time the box exists is only ever used again for
        # p_place/p4/p5.
        margin_entities = ["tiago_pro", self._obj_name, "table"]
        if self._box_loaded:
            margin_entities.append("box")
        security_margins = SecurityMargins(problem, factory, margin_entities, robot)
        security_margins.setSecurityMarginBetween("tiago_pro", self._obj_name, 0.0)
        security_margins.setSecurityMarginBetween("tiago_pro", "table", 0.05)
        if self._box_loaded:
            security_margins.setSecurityMarginBetween("tiago_pro", "box", 0.0)
        security_margins.apply()

        # The active gripper must approach within a few centimeters of the
        # table surface to grasp objects resting on it, so exempt just the
        # gripper itself (wrist joint + fingers — the palm/housing geometry
        # is rigidly fixed to, and thus merged into, the wrist joint's body)
        # from the table margin above; the rest of the arm (e.g. the elbow)
        # keeps the 5cm clearance.
        gripper_table_exempt_joints = [
            self._resolve_joint_name("arm_left_7_joint"),
            *(name for name in self._gripper_joint_targets if "gripper_left" in name),
        ]
        for edge in graph.getTransitions():
            for jname in gripper_table_exempt_joints:
                graph.setSecurityMarginForTransition(edge, jname, "table/root_joint", 0.0)

        # Disable collision between robot and object in grasp/carry transitions
        # so the gripper can actually reach the handle — for every handle.
        for cand in self._grasp_candidates:
            if cand["grasp"] is None:
                continue
            for jname in model.names:
                if jname and "/" not in jname:
                    graph.setSecurityMarginForTransition(
                        cand["grasp"], jname,
                        f"{self._obj_name}/root_joint", float("-inf"),
                    )

        graph.initialize()

        self.problem = problem
        self.problem.addConfigValidation("CollisionValidation")
        self.graph   = graph

    def _selected_transition_names(self) -> dict:
        """Snapshot the names of the currently-selected self._transition_*
        attributes, so they can be re-resolved by name (stable across
        rebuilds) against a freshly rebuilt self.graph — see
        _reresolve_selected_transitions."""
        return {
            attr: getattr(self, attr).name()
            for attr in (
                "_transition_approach", "_transition_grasp", "_transition_carry",
                "_transition_release", "_transition_return",
            )
            if getattr(self, attr, None) is not None
        }

    def _reresolve_selected_transitions(self, selected_names: dict) -> None:
        """Restore self._transition_* (captured via
        _selected_transition_names before a _setup_graph() rebuild) by
        looking them up by name in the new self.graph — a rebuild resets
        self._transition_* to the first grasp candidate
        (_build_grasp_candidates' default), discarding whatever
        _select_best_handle() (or a previous call to this method) picked."""
        for attr, name in selected_names.items():
            try:
                setattr(self, attr, self.graph.getTransition(name))
            except Exception as e:
                print(f"  WARNING: could not re-resolve {attr} ('{name}') after rebuilding the graph: {e}")

    def _build_grasp_candidates(self, graph):
        """Build one (approach, grasp, carry, release) transition quadruple
        per object handle. Vision can't tell us which handle is actually
        reachable (e.g. the object detected flipped ~180 deg), so plan_pick()
        tries each candidate and picks whichever one actually plans — see
        _select_best_handle(). Defaults self._transition_* to the first
        candidate; _select_best_handle() overrides this once actual
        reachability is known."""
        gripper = self._gripper_name

        self._grasp_candidates = []
        for handle in self._handle_names:
            transition_approach = graph.getTransition(f"{gripper} > {handle} | f_01")
            transition_grasp = graph.getTransition(f"{gripper} > {handle} | f_12")
            # Loop edge at grasped state — used for the carry phase.
            carry_edge = self._find_carry_edge(graph, transition_grasp)
            self._grasp_candidates.append({
                "name": handle,
                "approach": transition_approach,
                "grasp": transition_grasp,
                "carry": carry_edge,
                # Reverse edge from grasped back to pre-grasp — used for
                # release at drop zone.
                "release": self._find_release_edge(graph, transition_grasp),
                # Pre-grasp → free edge (reverse of approach) — p4/release
                # lands in the pre-grasp state, not free; this brings the arm
                # into free before projecting it onto the carry pose.
                "return": self._find_return_edge(graph, transition_approach),
            })

        default = self._grasp_candidates[0]
        self._transition_approach = default["approach"]
        self._transition_grasp    = default["grasp"]
        self._transition_carry    = default["carry"]
        self._transition_release  = default["release"]
        self._transition_return   = default["return"]

    def _locked_joints(self, robot, model):
        """Build the LockedJoint list for everything HPP should treat as
        fixed while planning: the fixed base/torso/head joints, the
        inactive arm's tucked posture, and the static table."""
        _cts = ComparisonTypes()
        _cts[:] = [ComparisonType.EqualToZero]

        def _lock(joint_name, value):
            joint_name = self._resolve_joint_name(joint_name)
            j = model.joints[model.getJointId(joint_name)]
            if j.nq == 2 and j.nv == 1:
                locked_val = np.array([np.cos(value), np.sin(value)])
            else:
                locked_val = np.array([value])
            return LockedJoint(robot, joint_name, locked_val, _cts)

        def _lock_full_config(joint_name, reference_config):
            """Lock a joint to a full reference configuration rather than a
            single scalar value — needed for a freeflyer, whose rotation
            part `setJointBounds` cannot pin (Pinocchio always samples a
            freeflyer's orientation uniformly over SO(3), ignoring whatever
            bounds are set on its quaternion components)."""
            joint_name = self._resolve_joint_name(joint_name)
            j = model.joints[model.getJointId(joint_name)]
            cts = ComparisonTypes()
            cts[:] = [ComparisonType.EqualToZero] * j.nv
            return LockedJoint(robot, joint_name, np.array(reference_config), cts)

        locked = []
        for joint_name, value in self._fixed_joint_values.items():
            try:
                locked.append(_lock(joint_name, value))
            except KeyError:
                pass
        # Lock the inactive arm when the robot description exposes one.
        for joint_name, val in zip(self._other_arm_joint_names, self._other_arm_lock_values):
            locked.append(_lock(joint_name, val))
        locked.append(_lock_full_config(
            "table/root_joint", [TABLE_OFFSET_X, 0, 0, 0., 0., 0., 1.]
        ))
        # The box, once loaded (see add_box_to_scene), is a fixed "anchor"
        # placed directly at BOX_POS at load time (_load_urdf_models) — it
        # has no DOF and needs no LockedJoint, unlike table/root_joint
        # above (a "freeflyer" that's merely pinned in place).

        return locked

    @staticmethod
    def _find_transition_between(graph, source_state, target_state, found_label: str):
        for name in graph.getTransitionNames():
            try:
                edge = graph.getTransition(name)
                edge_source, edge_target = graph.getNodesConnectedByTransition(edge)
            except Exception:
                continue
            if edge_source == source_state and edge_target == target_state:
                print(f"  {found_label} edge found: '{name}'")
                return edge
        return None

    @staticmethod
    def _find_carry_edge(graph, grasp_transition):
        """Return the self-loop transition attached to the grasped state."""
        try:
            _, grasped_state = graph.getNodesConnectedByTransition(grasp_transition)
        except Exception:
            grasped_state = None

        if grasped_state is not None:
            edge = Orchestrator._find_transition_between(
                graph, grasped_state, grasped_state, "Carry"
            )
            if edge is not None:
                return edge

        print("  WARNING: carry edge not found — p3 will be skipped.")
        return None

    @staticmethod
    def _find_release_edge(graph, grasp_transition):
        """Return the grasped → pre-grasp release transition (reverse of grasp)."""
        try:
            pre_grasp_state, grasped_state = graph.getNodesConnectedByTransition(
                grasp_transition
            )
        except Exception:
            print("  WARNING: release edge not found — p4 will use reversed p2.")
            return None

        edge = Orchestrator._find_transition_between(
            graph, grasped_state, pre_grasp_state, "Release"
        )
        if edge is not None:
            return edge

        print("  WARNING: release edge not found — p4 will use reversed p2.")
        return None

    @staticmethod
    def _find_return_edge(graph, approach_transition):
        """Return the self-loop transition attached to the free state
        (approach_transition's source). p4/release lands the arm in the
        pre-grasp state, which is strictly more constrained than free, so
        q_predrop is already a valid free-state configuration too; this
        self-loop is what actually lets generateTargetConfig move it to a
        different configuration (the carry pose) — unlike the pre-grasp →
        free relaxation edge, which just hands q_predrop back unchanged
        since it already trivially satisfies free's (empty) constraints."""
        try:
            free_state, _ = graph.getNodesConnectedByTransition(approach_transition)
        except Exception:
            free_state = None

        if free_state is not None:
            edge = Orchestrator._find_transition_between(
                graph, free_state, free_state, "Return"
            )
            if edge is not None:
                return edge

        print("  WARNING: return-to-carry edge not found — p5 will be skipped.")
        return None

    # ── Planning ──────────────────────────────────────────────────────────────

    @staticmethod
    def _path_seconds(path) -> float:
        tr = path.timeRange()
        return tr.second - tr.first

    @staticmethod
    def _is_iteration_limit_error(exc: Exception) -> bool:
        return "Maximal number of iterations reached" in str(exc)

    def _optimize_path(
        self,
        path,
        label: str,
        shortcut,
        spline_opt,
        shortcut_passes: int = 3,
        use_spline: bool = True,
    ):
        optimize_start = time.perf_counter()
        try:
            for i in range(max(0, shortcut_passes)):
                p_new = shortcut.optimize(path)
                dt = self._path_seconds(path) - self._path_seconds(p_new)
                path = p_new
                print(
                    f"  {label} shortcut {i+1}/{shortcut_passes}: "
                    f"{self._path_seconds(path):.2f} s  (−{dt:.2f} s)"
                )
                if dt < 1e-3:
                    break
        except Exception as e:
            print(f"  {label} shortcut failed: {e}")
        if use_spline:
            try:
                path = spline_opt.optimize(path)
                print(f"  {label} spline: {self._path_seconds(path):.2f} s")
            except Exception as e:
                print(f"  {label} spline failed: {e}")
        print(
            f"  {label} optimization time: "
            f"{time.perf_counter() - optimize_start:.2f} s"
        )
        return path

    def _plan_transition_path(
        self,
        planner,
        q_goal,
        transition,
        q_start,
        q_target,
        label: str,
        reset_roadmap: bool = True,
    ):
        planner.setTransition(transition)
        q_goal[0, :] = q_target
        print(f"Planning {label} …")
        plan_start = time.perf_counter()
        path = planner.planPath(q_start, q_goal, reset_roadmap)
        short_label = label.split()[0]
        print(
            f"  {short_label} found in {time.perf_counter() - plan_start:.2f} s "
            f"({self._path_seconds(path):.2f} s path)."
        )
        return path

    def _generate_valid_config(self, transition, q_from, q_seed, validator=None):
        """Project q_seed onto `transition`'s manifold from q_from, then run
        collision validation on the result. Returns (generated, valid, q,
        err): `generated` is whether the projection itself converged, `valid`
        is whether the projected config also passed collision validation
        (always False when not generated), and `q` is that config (None
        unless both are True). `err` is the projection's own error estimate —
        returned even when `generated` is False, since some callers report it
        in diagnostics. Pass `validator` to reuse an already-created
        pathValidation (e.g. across many samples in a loop) instead of
        creating a new one on every call."""
        generated, q_candidate, err = self.graph.generateTargetConfig(
            transition, q_from, q_seed
        )
        if not generated:
            self._last_invalid_config_report = None
            return False, False, None, err
        validator = validator or transition.pathValidation()
        valid, report = validator.validateConfiguration(q_candidate)
        # Stashed for callers that want to explain a failure to the user
        # (e.g. tuck_arm()) without changing this method's return
        # signature, which every other caller already unpacks as a 4-tuple.
        self._last_invalid_config_report = None if valid else report
        return True, valid, (q_candidate if valid else None), err

    def _sample_grasp_candidate(
        self, approach_transition, grasp_transition,
        shooter, approach_validator, grasp_validator,
    ):
        """Try one random sample against a given (approach, grasp) transition
        pair. Returns (ok, qpg, qg, err, qg_err); ok is False (with the rest
        None) if the pre-grasp/grasp config couldn't be generated or is in
        collision — no path planning is attempted here."""
        q = shooter.shoot()
        _generated, ok, qpg, err = self._generate_valid_config(
            approach_transition, self.q_init, q, approach_validator
        )
        if not ok:
            return False, None, None, None, None

        _generated, ok, qg, qg_err = self._generate_valid_config(
            grasp_transition, qpg, qpg, grasp_validator
        )
        if not ok:
            return False, None, None, None, None

        return True, qpg, qg, err, qg_err

    def _find_approach_path(self, planner, q_goal, max_attempts: int, seed_candidates=None):
        """Sample pre-grasp/grasp config pairs until one both exists
        collision-free (_sample_grasp_candidate) AND is actually reachable by
        a planned path from q_init — a candidate can be valid in isolation
        but still unreachable within the planner's iteration budget, so on
        that specific failure we keep sampling instead of giving up.

        `seed_candidates` (already-validated (qpg, qg, err, qg_err) tuples
        from _select_best_handle's handle-scoring pass, for the winning
        handle) are tried first — they cost nothing extra to obtain and let
        us skip straight to path planning instead of resampling from scratch.
        The roadmap is only reset before the first candidate that actually
        reaches planPath; later retries against the same transition/start
        reuse the tree grown from q_init instead of re-exploring from
        scratch."""
        shooter = self.problem.configurationShooter()
        approach_validator = self._transition_approach.pathValidation()
        grasp_validator = self._transition_grasp.pathValidation()
        valid_candidate_count = 0
        search_start = time.perf_counter()
        seed_candidates = list(seed_candidates or [])

        def candidates():
            for qpg, qg, err, qg_err in seed_candidates:
                yield True, qpg, qg, err, qg_err
            for _ in range(max_attempts):
                yield self._sample_grasp_candidate(
                    self._transition_approach, self._transition_grasp,
                    shooter, approach_validator, grasp_validator,
                )

        for attempt, (ok, qpg, qg, err, qg_err) in enumerate(candidates()):
            if not ok:
                continue

            valid_candidate_count += 1
            source = "seed" if attempt < len(seed_candidates) else "fresh"
            print(
                f"  qpg candidate ({source}) at attempt {attempt + 1}, err={err:.2e}, "
                f"search={time.perf_counter() - search_start:.2f} s"
            )

            try:
                p1 = self._plan_transition_path(
                    planner, q_goal, self._transition_approach,
                    self.q_init, qpg, "p1 (approach)",
                    reset_roadmap=(valid_candidate_count == 1),
                )
            except RuntimeError as exc:
                # Only swallow the "gave up searching" case; any other
                # RuntimeError is a real error and should propagate.
                if not self._is_iteration_limit_error(exc):
                    raise
                print(
                    f"  p1 hit the planner iteration limit for qpg candidate {attempt + 1} "
                    "after trying to connect from q_init; sampling another candidate."
                )
                continue

            return p1, qpg, qg, qg_err

        elapsed = time.perf_counter() - search_start
        if valid_candidate_count == 0:
            print(
                "Failed to find collision-free qpg/qg pair in "
                f"{len(seed_candidates) + max_attempts} attempts "
                f"({len(seed_candidates)} seed + {max_attempts} fresh) ({elapsed:.2f} s)."
            )
        else:
            print(
                f"Found {valid_candidate_count} collision-free qpg/qg candidate(s), "
                "but none could be connected from q_init within the planner iteration budget "
                f"after {elapsed:.2f} s."
            )
        return None, None, None, None

    def _plan_carry(self, qg, qpg, fallback_p4):
        """Plan p2b (retract to handle clearance) and p3 (carry to
        transport pose), if a carry edge exists and both configs/paths are
        reachable. p_place/p4 (real release at the drop zone)/p5 are
        planned later, after navigation — see plan_place(). Every
        failure point degrades gracefully: no carry edge, no reachable
        carry config, or a planning exception all leave p3 as None and
        keep `fallback_p4` (the reversed p2 computed by the caller) as the
        release path, since without an actual carry there's nothing to
        navigate to or place."""
        p_retract = None
        p3 = None
        if self._transition_carry is None:
            return p_retract, p3, fallback_p4

        planner = TransitionPlanner(self.problem)
        planner.maxIterations(1000)
        shortcut = GraphRandomShortcut(self.problem)
        spline_opt = SplineGradientBased_bezier3(self.problem)
        q_goal = np.zeros((1, self.robot.configSize()), order='F')

        q_carry = self._find_carry_config(qg)
        if q_carry is None:
            return p_retract, p3, fallback_p4

        carry_start = qg
        q_retract = self._find_retract_config(qg, qpg)
        if q_retract is not None:
            try:
                p_retract = self._plan_transition_path(
                    planner, q_goal, self._transition_carry,
                    qg, q_retract, "p2b (retract to handle clearance)"
                )
                p_retract = self._optimize_path(p_retract, "p2b", shortcut, spline_opt)
                self.q_retract = q_retract
                carry_start = q_retract
            except Exception as e:
                print(f"  p2b (retract) planning failed: {e}.  Carrying straight from qg.")
                p_retract = None
                self.q_retract = None
        else:
            print("  No retract config found — carrying straight from qg.")

        try:
            p3 = self._plan_transition_path(
                planner, q_goal, self._transition_carry,
                carry_start, q_carry, "p3 (carry to transport pose)"
            )
            p3 = self._optimize_path(p3, "p3", shortcut, spline_opt)
            self.q_carry = q_carry
        except Exception as e:
            print(f"  p3 planning failed: {e}.  Skipping carry phase.")
            return p_retract, None, fallback_p4

        return p_retract, p3, None

    def _set_box_loaded(self, loaded: bool) -> None:
        """Add or remove the drop-zone box from the HPP scene by flipping
        self._box_loaded and rebuilding self.robot/model/graph around it
        (see _setup_model/_setup_graph), then restoring the robot's real
        current joint state (sync_from_robot) and the previously selected
        grasp transitions (_reresolve_selected_transitions) — a rebuild
        resets both. Idempotent: a no-op if the box is already in the
        requested state. See add_box_to_scene/remove_box_from_scene."""
        if self._box_loaded == loaded:
            return
        verb = "Adding" if loaded else "Removing"
        prep = "to" if loaded else "from"
        print(f"  {verb} drop-zone box {prep} the HPP scene …")
        selected_names = self._selected_transition_names()
        self._box_loaded = loaded
        self._setup_model()
        # _setup_model() resets q_init's arm slots to LEFT_ARM_TUCK
        # defaults — sync_from_robot() (which also rebuilds self.graph)
        # overwrites them with the robot's real, current pose.
        if not self.sync_from_robot():
            print(
                f"  WARNING: could not sync from robot while {verb.lower()} "
                "the box — q_init keeps its post-rebuild default posture."
            )
            self._setup_graph()
        self._reresolve_selected_transitions(selected_names)
        if hasattr(self, "_viewer"):
            self.init_viewer(open=False)

    def add_box_to_scene(self) -> None:
        """Add the drop-zone box to the HPP scene. Called automatically
        from navigate_to_drop_zone_and_add_box(), right after the base has
        finished navigating there, and again defensively from
        plan_place() — but public and safe to call on its own too (e.g.
        interactively), since it's idempotent. Before this runs, the box
        simply isn't part of self.robot/self.model at all (see
        _load_urdf_models), so p1/p2/p2b/p3 plan a scene that genuinely
        has no box in it."""
        self._set_box_loaded(True)

    def remove_box_from_scene(self) -> None:
        """Remove the drop-zone box from the HPP scene again — the
        mirror of add_box_to_scene(). Called automatically once the base
        has navigated back to the initial point (execute_place) and
        defensively at the start of plan_pick(), so a new pick cycle
        always plans p1/p2/p2b/p3
        against a box-less scene regardless of whatever state a previous
        cycle (or an aborted one) left behind — the box's fixed pose is
        only meaningful while the base is actually at the drop zone (see
        module docstring: the whole HPP scene is base-relative). Public
        and idempotent, so also safe to call on its own."""
        self._set_box_loaded(False)

    def plan_place(self) -> bool:
        """Plan p_place (transport pose to drop zone), p4 (release/
        retreat) and p5 (return to carry pose, empty-handed, before
        navigating back to the initial point), against the current grasp
        (self.qg, self.q_carry) once the drop-zone box has been added to
        the scene. Called from execute_place() right after
        navigate_to_drop_zone_and_add_box() succeeds (also calls
        add_box_to_scene() itself — idempotent — so this can be called
        on its own too). Returns False only on a hard
        failure (no reachable drop config at all) — p4/p5 degrade
        gracefully to None on their own failures without failing the
        whole call, same as before."""
        self.add_box_to_scene()

        self.p_place = None
        self.p4 = None
        self.p5 = None

        planner = TransitionPlanner(self.problem)
        planner.maxIterations(1000)
        shortcut = GraphRandomShortcut(self.problem)
        spline_opt = SplineGradientBased_bezier3(self.problem)
        q_goal = np.zeros((1, self.robot.configSize()), order='F')

        q_drop = self._find_drop_config(self.qg)
        if q_drop is None:
            return False

        try:
            p_place = self._plan_transition_path(
                planner, q_goal, self._transition_carry,
                self.q_carry, q_drop, "p_place (transport pose to drop zone)"
            )
            self.p_place = self._optimize_path(p_place, "p_place", shortcut, spline_opt)
            self.q_drop = q_drop
        except Exception as e:
            print(f"  p_place planning failed: {e}.  Skipping place phase.")
            return False

        q_predrop = self._find_predrop_config(q_drop)
        if q_predrop is None:
            print("  No pre-drop config — skipping retreat at drop zone.")
            self.q_predrop = None
            return True

        try:
            p4 = self._plan_transition_path(
                planner, q_goal, self._transition_release,
                q_drop, q_predrop, "p4 (release at drop zone)"
            )
            self.p4 = self._optimize_path(p4, "p4", shortcut, spline_opt)
            self.q_predrop = q_predrop
        except Exception as ep4:
            print(f"  p4 planning failed: {ep4}.  Skipping retreat at drop zone.")
            self.q_predrop = None
            return True

        if self._transition_return is None:
            print("  No return edge — skipping return to carry pose.")
            self.q_return = None
            return True

        q_return = self._find_return_config(q_predrop)
        if q_return is None:
            print("  No return config — skipping return to carry pose.")
            self.q_return = None
            return True

        try:
            p5 = self._plan_transition_path(
                planner, q_goal, self._transition_return,
                q_predrop, q_return, "p5 (return to carry pose)"
            )
            self.p5 = self._optimize_path(p5, "p5", shortcut, spline_opt)
            self.q_return = q_return
        except Exception as ep5:
            print(f"  p5 planning failed: {ep5}.  Skipping return to carry pose.")
            self.q_return = None

        return True

    def _select_best_handle(self, trial_attempts: int = 30) -> None:
        """Try every candidate handle (see _setup_graph) with a small budget
        of IK/collision-only samples (no path planning) and switch
        self._transition_* to whichever handle is actually reachable —
        vision can't tell us which one is (e.g. the object may be detected
        flipped ~180 deg), so this decides based on observed success rate
        instead of assuming a single canonical orientation.

        The winning handle's successful samples are kept (self._seed_candidates)
        so _find_approach_path can try them directly instead of resampling
        from scratch for the handle we already know is reachable."""
        shooter = self.problem.configurationShooter()
        scores = []
        for cand in self._grasp_candidates:
            approach_validator = cand["approach"].pathValidation()
            grasp_validator = cand["grasp"].pathValidation()
            samples = [
                self._sample_grasp_candidate(
                    cand["approach"], cand["grasp"],
                    shooter, approach_validator, grasp_validator,
                )
                for _ in range(trial_attempts)
            ]
            valid = [(qpg, qg, err, qg_err) for ok, qpg, qg, err, qg_err in samples if ok]
            cand["_valid_samples"] = valid
            scores.append(len(valid))
            print(f"  Handle '{cand['name']}': {len(valid)}/{trial_attempts} reachable samples.")

        chosen = self._grasp_candidates[scores.index(max(scores))]
        print(f"  Selected handle: '{chosen['name']}'.")
        self._transition_approach = chosen["approach"]
        self._transition_grasp    = chosen["grasp"]
        self._transition_carry    = chosen["carry"]
        self._transition_release  = chosen["release"]
        self._transition_return   = chosen["return"]
        self._seed_candidates     = chosen["_valid_samples"]

    def plan_pick(
        self,
        max_attempts: int = 100,
        approach_shortcut_passes: int = 1,
        approach_use_spline: bool = False,
        sync_with_robot: bool = True,
        sync_timeout: float = 3.0,
    ) -> bool:
        """
        Plan pick-and-drop trajectory.

        Generates:
            p1 — approach  (free  → pre-grasp)
            p2 — grasp     (pre-grasp → grasped)
            p2b — retract  (grasped loop: qg → q_retract, projecting qpg's arm
                            posture onto the carry manifold)  [skipped if no
                            carry edge, or no reachable retract config]
            p3 — carry     (grasped loop: q_retract (or qg if p2b was skipped)
                            → q_carry)      [skipped if no carry edge]
            p4 — release   (q_drop → pre-drop via release edge; falls back to reversed p2
                            if there's no carry at all)
            p5 — return    (pre-drop → free via return edge, then onto carry pose,
                            empty-handed) [skipped if p4 falls back to reversed p2,
                            or no return edge]

        p_place, p4 (the real release-at-drop-zone version) and p5 are NOT
        planned here — the drop-zone box isn't even part of the HPP scene
        yet (see _load_urdf_models/add_box_to_scene), since it's only
        relevant once the base has actually navigated there. They're
        planned by plan_place(), called from execute_place() right after
        navigate_to_drop_zone_and_add_box() succeeds.

        Removes the box from the scene first (remove_box_from_scene,
        idempotent) so p1/p2/p2b/p3 always plan against a box-less scene
        even if a previous cycle (or an aborted one) left it loaded —
        execute_place() normally already removes it once the base has
        navigated back to the initial point.

        By default the approach phase uses lighter post-processing than the later
        phases so a valid p1 is returned sooner. Set
        approach_shortcut_passes=3 and approach_use_spline=True to recover the
        previous fully optimized behavior for p1.

        When sync_with_robot=True, q_init and the fixed planning joints are first
        synchronized from /joint_states so HPP plans in the same torso/head/base
        posture that Gazebo is currently executing.
        """
        self.remove_box_from_scene()

        if sync_with_robot:
            self.sync_from_robot(timeout=sync_timeout)

        self.problem.constraintGraph(self.graph)
        self._select_best_handle()

        planner = TransitionPlanner(self.problem)
        planner.maxIterations(1000)
        shortcut   = GraphRandomShortcut(self.problem)
        spline_opt = SplineGradientBased_bezier3(self.problem)
        q_goal = np.zeros((1, self.robot.configSize()), order='F')

        p1, qpg, qg, qg_err = self._find_approach_path(
            planner, q_goal, max_attempts, seed_candidates=self._seed_candidates
        )
        if p1 is None:
            return False

        print(f"  qg: res=True, err={qg_err:.2e}")
        p1 = self._optimize_path(
            p1,
            "p1",
            shortcut,
            spline_opt,
            shortcut_passes=approach_shortcut_passes,
            use_spline=approach_use_spline,
        )

        p2 = self._plan_transition_path(
            planner, q_goal, self._transition_grasp, qpg, qg, "p2 (grasp)"
        )
        p2 = self._optimize_path(p2, "p2", shortcut, spline_opt)

        p4 = p2.reverse()
        print("  p4 ready (release, reversed p2).")

        p_retract, p3, p4 = self._plan_carry(qg, qpg, p4)

        self.p1       = p1
        self.p2       = p2
        self.p_retract = p_retract
        self.p3       = p3
        self.p_place  = None
        self.p4       = p4
        self.p5       = None
        self.qpg = qpg
        self.qg  = qg
        return True

    def _find_carry_config(self, qg):
        """
        Project CARRY_ARM_CFG onto the grasped carry manifold — the transport
        pose the arm holds while the base navigates, reached right after
        grasping and held until the arm moves on to the drop configuration.
        """
        q_seed = np.array(qg).copy()
        arm_idx = self._active_arm_idx
        q_seed[arm_idx:arm_idx+7] = CARRY_ARM_CFG

        if self._transition_carry is None:
            return q_seed

        generated, valid, q_carry, err = self._generate_valid_config(
            self._transition_carry, qg, q_seed
        )
        if not generated:
            print(
                f"  Failed to project carry config onto carry manifold (err={err:.2e}) — skipping p3."
            )
            return None
        if not valid:
            print("  Projected carry config is not valid for carry — skipping p3.")
            return None
        return q_carry

    def _find_retract_config(self, qg, qpg):
        """
        Project the pre-grasp arm posture (qpg) onto the grasped carry
        manifold — retracts the just-grasped object straight back off the
        handle by that handle's own SRDF clearance distance (the same
        offset HPP already used to place qpg before grasping) before the
        arm makes its large motion to the carry pose in p3.
        """
        q_seed = np.array(qg).copy()
        arm_idx = self._active_arm_idx
        q_seed[arm_idx:arm_idx+7] = np.array(qpg)[arm_idx:arm_idx+7]

        if self._transition_carry is None:
            return q_seed

        generated, valid, q_retract, err = self._generate_valid_config(
            self._transition_carry, qg, q_seed
        )
        if not generated:
            print(
                f"  Failed to project retract config onto carry manifold (err={err:.2e}) — skipping retract phase."
            )
            return None
        if not valid:
            print("  Projected retract config is not valid for carry — skipping retract phase.")
            return None
        return q_retract

    def _find_drop_config(self, qg, max_attempts: int = 100):
        """
        Solve for an arm configuration that places the grasped object
        BOX_CLEARANCE above the drop-zone box's rim (BOX_POS + wall height,
        from BOX_WALL_TOP_OFFSET), keeping the object's grasped orientation
        unchanged. The target depends entirely on the box's configured
        pose, not on any hardcoded arm configuration — move the box in the
        config and p_place follows it, no manual retuning needed.

        The object is rigidly attached to the gripper once grasped, so its
        pose relative to the tool frame (the "grasp offset") is fixed for a
        given handle; composing the desired object target with the inverse
        of that offset gives the tool-frame IK target (_solve_ee_ik). A
        7-DOF arm reaching a 6-DOF target has a 1-parameter family of
        solutions, and not every member of that family is collision-free
        (e.g. against the box itself) or within joint limits — HPP's own
        pathValidation was observed NOT to catch out-of-bounds joint values
        here, so bounds are checked explicitly. Try q_carry's own arm
        config first (nearest already-known-valid pose), then qg's, then
        random arm configs within joint limits, validating each against
        _transition_carry until one succeeds.

        Called from plan_place(), after add_box_to_scene() has
        rebuilt the graph with the box present, so _transition_carry's own
        collision checking already accounts for it — q_drop is where the
        object actually ends up relative to the drop-zone box, so its
        validity must account for the box.
        """
        transition = self._transition_carry
        if transition is None:
            return None

        T_ee_g = self._fk_ee_at(qg)
        T_obj_g = self._obj_pose_at(qg)
        grasp_offset = T_ee_g.inverse() * T_obj_g

        obj_target_pos = BOX_POS + np.array(
            [0, 0, BOX_WALL_TOP_OFFSET + BOX_CLEARANCE]
        )
        obj_target = pin.SE3(T_obj_g.rotation, obj_target_pos)
        tool_target = obj_target * grasp_offset.inverse()

        arm_idx = self._active_arm_idx
        lower = self.model.lowerPositionLimit[arm_idx:arm_idx+7]
        upper = self.model.upperPositionLimit[arm_idx:arm_idx+7]

        def seed_arms():
            if self.q_carry is not None:
                yield np.array(self.q_carry)[arm_idx:arm_idx+7]
            yield np.array(qg)[arm_idx:arm_idx+7]
            rng = np.random.default_rng()
            for _ in range(max_attempts):
                yield rng.uniform(lower, upper)

        for seed_arm in seed_arms():
            q_seed = np.array(qg).copy()
            q_seed[arm_idx:arm_idx+7] = seed_arm
            q_ik = self._solve_ee_ik(tool_target, q_seed)
            if q_ik is None:
                continue
            arm_sol = q_ik[arm_idx:arm_idx+7]
            if np.any(arm_sol < lower) or np.any(arm_sol > upper):
                continue

            generated, valid, q_drop, _err = self._generate_valid_config(
                transition, qg, q_ik
            )
            if not generated or not valid:
                continue
            return q_drop

        print(
            f"  Failed to find a drop config above the box in {max_attempts} "
            "attempts — skipping place phase."
        )
        return None

    def _find_predrop_config(self, q_drop, max_attempts: int = 100):
        """Find a valid pre-grasp config at the drop zone for the p4 release motion."""
        if self._transition_release is None:
            return None

        release_validator = self._transition_release.pathValidation()

        # First try the same deterministic projection pattern used for grasp:
        # project q_drop through the reverse edge onto the pre-grasp manifold.
        deterministic_seeds = [np.array(q_drop).copy()]
        if self.qpg is not None:
            q_seed = np.array(q_drop).copy()
            arm_idx = self._active_arm_idx
            q_seed[arm_idx:arm_idx+7] = np.array(self.qpg)[arm_idx:arm_idx+7]
            deterministic_seeds.append(q_seed)

        for idx, q_seed in enumerate(deterministic_seeds, start=1):
            generated, valid, q_cand, err = self._generate_valid_config(
                self._transition_release, q_drop, q_seed, release_validator
            )
            if not generated:
                continue
            if valid:
                print(
                    f"  q_predrop found from deterministic seed {idx}, err={err:.2e}"
                )
                return q_cand

        shooter = self.problem.configurationShooter()
        for attempt in range(max_attempts):
            q = shooter.shoot()
            generated, valid, q_cand, err = self._generate_valid_config(
                self._transition_release, q_drop, q, release_validator
            )
            if not generated:
                continue
            if valid:
                print(f"  q_predrop found at attempt {attempt + 1}, err={err:.2e}")
                return q_cand
        print(f"  WARNING: no valid pre-drop config found in {max_attempts} attempts.")
        return None

    def _find_return_config(self, q_predrop):
        """
        Project CARRY_ARM_CFG onto the free manifold, reachable from
        q_predrop (the pre-grasp state p4/release lands in) via the
        pre-grasp → free edge — after releasing the object, bring the arm
        back to the same transport pose it held during carry, so the base
        navigates back to the initial point with the arm in a known, safe
        posture instead of wherever p4/release left it.
        """
        q_seed = np.array(q_predrop).copy()
        arm_idx = self._active_arm_idx
        q_seed[arm_idx:arm_idx+7] = CARRY_ARM_CFG

        if self._transition_return is None:
            return q_seed

        generated, valid, q_return, err = self._generate_valid_config(
            self._transition_return, q_predrop, q_seed
        )
        if not generated:
            print(
                f"  Failed to project return config onto free manifold (err={err:.2e}) — skipping p5."
            )
            return None
        if not valid:
            print("  Projected return config is not valid — skipping p5.")
            return None
        return q_return

    # ── Path sampling ─────────────────────────────────────────────────────────

    def _sample_path(self, path, time_scale: float = TIME_SCALE):
        """Resample an HPP path at DT intervals (sped up by `time_scale`) into
        arm-only position/velocity/acceleration arrays for the MPC
        controller. HPP paths only give positions, so velocity/acceleration
        are derived by finite differences; the last row of each is just
        repeated once to keep all three arrays the same length."""
        tr = path.timeRange()
        t_min, t_max = tr.first, tr.second
        n = max(2, int((t_max - t_min) * time_scale / DT))
        times = np.linspace(t_min, t_max, n)
        arm_idx = self._active_arm_idx
        q_arr = np.array([np.asarray(path.eval(t)[0])[arm_idx:arm_idx+7] for t in times])
        dq_arr = np.diff(q_arr, axis=0) / DT
        dq_arr = np.vstack([dq_arr, dq_arr[-1]])
        ddq_arr = np.diff(dq_arr, axis=0) / DT
        ddq_arr = np.vstack([ddq_arr, ddq_arr[-1]])

        return q_arr, dq_arr, ddq_arr

    def _fk_ee(self, q_arm: np.ndarray) -> pin.SE3:
        # agimus_controller/LFC only tracks the active arm joints; using the full
        # HPP planning state here would inject the torso lift into the EE pose and
        # shift the controller reference about 300 mm upward.
        q_full = pin.neutral(self.model)
        arm_idx = self._active_arm_idx
        q_full[arm_idx:arm_idx+7] = q_arm
        pin.forwardKinematics(self.model, self._pin_data, q_full)
        pin.updateFramePlacements(self.model, self._pin_data)
        return self._pin_data.oMf[self._ee_frame_id].copy()

    def _fk_ee_at(self, q_full: np.ndarray) -> pin.SE3:
        """True forward kinematics at a FULL planning configuration (torso
        lift, base, etc. included as-is) — unlike _fk_ee, which deliberately
        zeroes the torso for MPC-message purposes. Needed wherever the real
        world-frame EE pose matters, e.g. IK targets computed from BOX_POS
        (also expressed in that same, torso-included frame)."""
        pin.forwardKinematics(self.model, self._pin_data, np.array(q_full))
        pin.updateFramePlacements(self.model, self._pin_data)
        return self._pin_data.oMf[self._ee_frame_id].copy()

    def _obj_pose_at(self, q_full: np.ndarray) -> pin.SE3:
        """Object pose read directly out of a full configuration's own
        freeflyer DOFs (no forward kinematics needed — it's a raw config
        value, not derived from the kinematic chain)."""
        q_full = np.array(q_full)
        obj_idx = self._obj_idx
        t = q_full[obj_idx:obj_idx+3]
        x, y, z, w = q_full[obj_idx+3:obj_idx+7]
        return pin.SE3(pin.Quaternion(w, x, y, z).matrix(), t)

    def _solve_ee_ik(
        self, target: pin.SE3, q_full_seed: np.ndarray,
        max_iters: int = 300, dt: float = 0.3, damp: float = 1e-6,
        tol: float = 1e-5,
    ):
        """Damped-least-squares IK for the 7 active-arm joints, moving only
        the EE frame (torso/base/other-arm/etc. held fixed at q_full_seed's
        values) to `target` (base_link frame). Returns the full config with
        the arm slice updated, or None if it didn't converge within
        max_iters — never raises, since callers try many seeds and expect
        failures. 7 DOF into a 6D target is redundant (a 1-parameter family
        of solutions), so different seeds generally converge to different
        joint configs for the same target pose — try several and validate
        each, don't rely on a single seed."""
        q_full = np.array(q_full_seed).copy()
        arm_idx = self._active_arm_idx
        arm_idx_v = self._active_arm_idx_v
        for _ in range(max_iters):
            T_cur = self._fk_ee_at(q_full)
            err = pin.log6(target.actInv(T_cur)).vector
            if np.linalg.norm(err) < tol:
                return q_full
            J = pin.computeFrameJacobian(
                self.model, self._pin_data, q_full, self._ee_frame_id,
                pin.ReferenceFrame.LOCAL,
            )
            J_arm = J[:, arm_idx_v:arm_idx_v+7]
            dq_arm = J_arm.T @ np.linalg.solve(
                J_arm @ J_arm.T + damp * np.eye(6), -err
            )
            q_full[arm_idx:arm_idx+7] += dt * dq_arm
        return None

    def _build_msg(self, q, dq, ddq, msg_id):
        """Build one MpcInput: joint-space targets/weights for the 7 active
        arm joints, plus a Cartesian end-effector target/weights (computed
        via forward kinematics) that the MPC controller blends in for
        task-space tracking."""
        msg = MpcInput()
        msg.id           = msg_id
        msg.q            = q.tolist()
        msg.qdot         = dq.tolist()
        msg.qddot        = ddq.tolist()
        msg.robot_effort = np.zeros(7).tolist()
        msg.w_q                   = W_Q.tolist()
        msg.w_qdot                = W_QDOT.tolist()
        msg.w_qddot               = W_QDDOT.tolist()
        msg.w_robot_effort        = W_EFFORT.tolist()
        msg.w_collision_avoidance = W_COLLISION

        T_ee = self._fk_ee(q)
        quat = pin.Quaternion(T_ee.rotation)
        ee_input = MpcEEInput()
        ee_input.frame_id = self._ee_frame_msg_name
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

    @staticmethod
    def _ensure_mpc_publisher(node: Node):
        pub = getattr(node, "_hpp_mpc_input_pub", None)
        if pub is None:
            qos = QoSProfile(depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
            pub = node.create_publisher(MpcInput, "mpc_input", qos)
            setattr(node, "_hpp_mpc_input_pub", pub)
        return pub

    @staticmethod
    def _ensure_grasper_client(node: Node, service_name: str):
        """Cache one Empty client per service name on `node`, mirroring
        _ensure_mpc_publisher — needed because _call_grasper_service now runs
        against the persistent execution node instead of a fresh throwaway
        node per call, so a naive create_client() would leak a new client
        object on every gripper action."""
        attr = "_grasper_client_" + service_name.replace("/", "_")
        client = getattr(node, attr, None)
        if client is None:
            client = node.create_client(Empty, service_name)
            setattr(node, attr, client)
        return client

    @staticmethod
    def _ensure_nav_action_client(node: Node) -> ActionClient:
        """Cache one NavigateToPose ActionClient on `node`, mirroring
        _ensure_grasper_client — avoids leaking a new client per navigate call."""
        client = getattr(node, "_nav_action_client", None)
        if client is None:
            client = ActionClient(node, NavigateToPose, NAVIGATE_TO_POSE_ACTION)
            setattr(node, "_nav_action_client", client)
        return client

    @contextmanager
    def _borrow_ros_node(self, name: str):
        if self._ros_node is not None:
            yield self._ros_node
            return

        node = rclpy.create_node(name)
        try:
            yield node
        finally:
            node.destroy_node()

    def _ensure_joint_state_cache(self, node: Node) -> None:
        """Subscribe to /joint_states exactly once (across all QoS profiles,
        since publishers vary) and keep self._latest_joint_state_map updated
        for the lifetime of `node`. Replaces the old pattern of every waiter
        (_wait_for_arm_settled, _joint_state_map, …) creating and destroying
        its own subscriptions per call, which paid fresh DDS discovery latency
        at every phase boundary and every gripper action."""
        if self._joint_state_subs:
            return

        def _cb(msg):
            self._latest_joint_state_map.update(dict(zip(msg.name, msg.position)))

        self._joint_state_subs = [
            node.create_subscription(JointState, "/joint_states", _cb, profile)
            for profile in JOINT_STATE_QOS_PROFILES
        ]

    def _execution_node(self) -> Node:
        if self._ros_node is None:
            self._ros_node = rclpy.create_node("hpp_trajectory_publisher")
        self._ensure_joint_state_cache(self._ros_node)
        return self._ros_node

    @staticmethod
    def _wait_for_topic_message(node: Node, msg_type, topic: str, timeout: float, qos=10):
        message = [None]
        qos_profiles = qos if isinstance(qos, (list, tuple)) else (qos,)
        subs = [
            node.create_subscription(
                msg_type, topic, lambda msg: message.__setitem__(0, msg), profile
            )
            for profile in qos_profiles
        ]
        try:
            deadline = time.time() + timeout
            while time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if message[0] is not None:
                    return message[0]
        finally:
            for sub in subs:
                node.destroy_subscription(sub)
        return None

    def _joint_state_map(self, timeout: float):
        node = self._execution_node()
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if self._latest_joint_state_map:
                return dict(self._latest_joint_state_map)
        return None

    def _publish_hold_reference(self, node: Node) -> bool:
        """Re-publish the last trajectory setpoint (with a fresh id) so the
        MPC controller keeps seeing a live reference while we're blocked
        waiting on something else (a service call, a future). Returns False
        if execute() hasn't run yet and there's nothing to hold."""
        if self._last_hold_msg is None:
            return False

        hold_msg = copy.deepcopy(self._last_hold_msg)
        hold_msg.id = self._next_msg_id
        self._next_msg_id += 1
        self._ensure_mpc_publisher(node).publish(hold_msg)
        self._last_hold_msg = hold_msg
        return True

    def _spin_future_with_hold(self, node: Node, future) -> None:
        """Spin `node` until `future` resolves, publishing a hold reference
        at DT intervals in the meantime — used while waiting on ROS service
        calls (e.g. gripper open/close) so the controller doesn't go quiet
        mid-sequence. Gripper close/open calls can legitimately block for a
        few seconds (contact-detection window server-side), so print
        progress periodically rather than going silent the whole time."""
        next_hold_time = time.monotonic()
        start = time.monotonic()
        next_progress = start + 1.0
        while not future.done():
            if self._last_hold_msg is not None:
                now = time.monotonic()
                if now >= next_hold_time:
                    self._publish_hold_reference(node)
                    next_hold_time = now + DT
                    continue
                timeout_sec = max(0.0, min(0.1, next_hold_time - now))
            else:
                timeout_sec = 0.1
            rclpy.spin_once(node, timeout_sec=timeout_sec)
            now = time.monotonic()
            if now >= next_progress:
                print(f"  ... waiting for gripper service response (elapsed {now - start:.1f}s)")
                next_progress = now + 1.0

    def _build_messages(self, paths: list, n_hold: int = 200) -> list:
        """Sample each (path, label) pair into MpcInput messages, then
        append n_hold extra messages at the final position with zero
        velocity/acceleration. Without this padding, the controller would
        stop receiving new setpoints the instant the path ends and could
        lose tracking; the hold points give it a stable target to settle
        into instead."""
        msgs = []
        idx = self._next_msg_id
        for path, label in paths:
            time_scale = CARRY_TIME_SCALE if (path is self.p_retract or path is self.p3 or path is self.p_place or path is self.p5) else TIME_SCALE
            q_arr, dq_arr, ddq_arr = self._sample_path(path, time_scale=time_scale)
            print(f"  {label}: {len(q_arr)} waypoints")
            for q, dq, ddq in zip(q_arr, dq_arr, ddq_arr):
                msgs.append(self._build_msg(q, dq, ddq, idx))
                idx += 1
        if not msgs:
            return []
        q_final  = msgs[-1].q
        dq_zero  = np.zeros(len(msgs[-1].qdot)).tolist()
        ddq_zero = np.zeros(len(msgs[-1].qddot)).tolist()
        for _ in range(n_hold):
            msg = self._build_msg(
                np.array(q_final), np.array(dq_zero), np.array(ddq_zero), idx
            )
            msgs.append(msg)
            idx += 1
        self._next_msg_id = idx
        print(f"  {len(msgs)} MpcInput messages total ({n_hold} hold points appended).")
        return msgs

    def _path_labels(self) -> dict:
        labels = {
            id(self.p1): "p1 (approach)",
            id(self.p2): "p2 (grasp)",
        }
        if self.p_retract is not None:
            labels[id(self.p_retract)] = "p2b (retract to handle clearance)"
        if self.p3 is not None:
            labels[id(self.p3)] = "p3 (carry)"
        if self.p_place is not None:
            labels[id(self.p_place)] = "p_place (place at drop zone)"
        if self.p4 is not None:
            labels[id(self.p4)] = "p4 (release)"
        if self.p5 is not None:
            labels[id(self.p5)] = "p5 (return to carry)"
        return labels

    def _named_paths(self, paths: list) -> list:
        labels = self._path_labels()
        return [
            (path, labels.get(id(path), f"path_{index + 1}"))
            for index, path in enumerate(paths)
            if path is not None
        ]

    def _planned_paths(self) -> list:
        return [
            path for path in [
                self.p1, self.p2, self.p_retract, self.p3, self.p_place, self.p4, self.p5
            ]
            if path is not None
        ]

    def _warn_if_robot_far_from_path_start(
        self,
        named_paths: list,
        timeout: float = 1.0,
    ) -> None:
        if not named_paths:
            return

        js_map = self._joint_state_map(timeout)
        if js_map is None:
            return

        q_actual = self._configuration_from_joint_state(js_map, q_seed=self.q_init)

        path, label = named_paths[0]
        q_start = np.array(path.eval(path.timeRange().first)[0], copy=True)
        arm_idx = self._active_arm_idx
        joint_err = np.max(np.abs(q_start[arm_idx:arm_idx+7] - q_actual[arm_idx:arm_idx+7]))

        data_start = self.model.createData()
        data_actual = self.model.createData()
        pin.forwardKinematics(self.model, data_start, q_start)
        pin.updateFramePlacements(self.model, data_start)
        pin.forwardKinematics(self.model, data_actual, q_actual)
        pin.updateFramePlacements(self.model, data_actual)
        delta = data_start.oMf[self._ee_frame_id].inverse() * data_actual.oMf[self._ee_frame_id]
        pos_err_mm = np.linalg.norm(delta.translation) * 1000.0

        if (
            pos_err_mm < PATH_START_EE_WARN_MM
            and joint_err < PATH_START_JOINT_WARN_RAD
        ):
            return

        print(
            f"WARNING: robot is {pos_err_mm:.1f} mm from the start of {label} "
            f"(max joint delta {joint_err:.3f} rad). "
            "This path was likely planned from a stale q_init; rerun plan_pick() "
            "after sync_from_robot() before executing it."
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    def _wait_for_arm_settled(self, node: Node, timeout: float = ARM_SETTLE_TIMEOUT) -> bool:
        """Poll /joint_states until the active arm's actual end-effector pose
        converges to the just-published path's final waypoint, or `timeout`
        elapses. Publishing the last trajectory message only guarantees the
        reference was sent, not that the (torque-controlled) arm has
        physically caught up to it — this confirms it did before the caller
        moves on to the next phase (e.g. a gripper action). Checked via EE
        pose rather than per-joint error — see the ARM_SETTLE_EE_* comment.

        Hold references are re-published at DT-spaced wall-clock intervals
        (mirroring _spin_future_with_hold), rather than once per spin_once
        iteration — spin_once's cadence here depends on /joint_states
        arrival plus this loop's own FK work, which can run slower than the
        controller's consumption rate and starve the MPC input buffer."""
        if self._last_executed_q is None:
            return True
        q_target = np.asarray(self._last_executed_q)
        data_target = self.model.createData()
        pin.forwardKinematics(self.model, data_target, q_target)
        pin.updateFramePlacements(self.model, data_target)
        T_target = data_target.oMf[self._ee_frame_id]
        data_actual = self.model.createData()

        last_err = (float("nan"), float("nan"))
        start = time.monotonic()
        deadline = start + timeout
        next_progress = start + 1.0
        next_hold_time = start
        while time.monotonic() < deadline:
            if self._last_hold_msg is not None:
                now = time.monotonic()
                if now >= next_hold_time:
                    self._publish_hold_reference(node)
                    next_hold_time = now + DT
                    continue
                timeout_sec = max(0.0, min(0.1, next_hold_time - now))
            else:
                timeout_sec = 0.1
            rclpy.spin_once(node, timeout_sec=timeout_sec)
            if self._latest_joint_state_map:
                q_actual = self._configuration_from_joint_state(
                    self._latest_joint_state_map, q_seed=self.q_init
                )
                pin.forwardKinematics(self.model, data_actual, q_actual)
                pin.updateFramePlacements(self.model, data_actual)
                delta = T_target.inverse() * data_actual.oMf[self._ee_frame_id]
                pos_err_mm = np.linalg.norm(delta.translation) * 1000.0
                rot_err_deg = np.degrees(np.linalg.norm(pin.log3(delta.rotation)))
                last_err = (pos_err_mm, rot_err_deg)
                if (
                    pos_err_mm <= ARM_SETTLE_EE_POS_TOLERANCE_MM
                    and rot_err_deg <= ARM_SETTLE_EE_ROT_TOLERANCE_DEG
                ):
                    return True
            now = time.monotonic()
            if now >= next_progress:
                pos_err_mm, rot_err_deg = last_err
                print(
                    f"  ... arm settling: EE error {pos_err_mm:.1f} mm / {rot_err_deg:.1f}°"
                    f" (elapsed {now - start:.1f}/{timeout:.1f}s)"
                )
                next_progress = now + 1.0

        pos_err_mm, rot_err_deg = last_err
        print(
            f"WARNING: arm did not settle at the path's final end-effector pose "
            f"within {timeout}s (EE error {pos_err_mm:.1f} mm / {rot_err_deg:.1f}°)."
        )
        return False

    def _execute_paths(self, named_paths: list, n_hold: int = 200) -> bool:
        """Build and publish MpcInput messages for the given (path, label)
        pairs. Returns True only if the trajectory streamed to completion and
        the arm was confirmed (via /joint_states) to have settled at its
        final waypoint — callers must check this before treating the motion
        as done (e.g. before firing a gripper command)."""
        named_paths = [(path, label) for path, label in named_paths if path is not None]
        if not named_paths:
            print("No paths to execute.")
            return True

        self._warn_if_robot_far_from_path_start(named_paths)
        print("Sampling trajectories …")
        messages = self._build_messages(named_paths, n_hold=n_hold)
        self._messages = messages
        self._last_hold_msg = copy.deepcopy(messages[-1])
        last_path, last_label = named_paths[-1]
        self._last_executed_q = np.array(
            last_path.eval(last_path.timeRange().second)[0], copy=True
        )
        self._last_executed_label = last_label

        node = self._execution_node()
        pub = self._ensure_mpc_publisher(node)
        node.get_logger().info(
            f"Publishing {len(messages)} trajectory points at {1/DT:.0f} Hz …"
        )

        completed = True
        try:
            # Burst the first PUBLISH_BURST_COUNT messages with no DT
            # pacing, giving agimus_controller_node's trajectory buffer
            # real headroom before settling into the steady paced rate —
            # see PUBLISH_BURST_COUNT's comment. Consumption there is
            # timer-driven at its own fixed rate regardless of how fast we
            # hand it messages, so arriving early only builds lookahead
            # margin; it doesn't make the robot move any faster.
            burst_count = min(PUBLISH_BURST_COUNT, len(messages))
            for msg in messages[:burst_count]:
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)

            publish_start = time.monotonic()
            for i, msg in enumerate(messages[burst_count:]):
                pub.publish(msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                # Deadline-based pacing (mirrors _wait_for_arm_settled /
                # _spin_future_with_hold): sleeping a fixed DT here, after
                # doing the publish+spin work, would systematically
                # undershoot 1/DT Hz by however long that work took each
                # iteration — compounding over a long phase into a real
                # drain of the MPC's trajectory buffer.
                target_time = publish_start + (i + 1) * DT
                sleep_time = target_time - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            node.get_logger().info("Trajectory fully published.")
        except KeyboardInterrupt:
            print("\nExecution interrupted.")
            completed = False

        if not completed:
            return False
        return self._wait_for_arm_settled(node)

    def _execute_phase(self, title: str, path, label: str, n_hold: int) -> bool:
        print(f"\n=== {title} ===")
        return self._execute_paths([(path, label)], n_hold=n_hold)

    @staticmethod
    def _abort_sequence(step: str) -> bool:
        print(f"\n*** ABORTING sequence: {step} did not complete/settle — "
              "see WARNING above. Fix the issue (e.g. sync_from_robot()) "
              "and re-run. ***")
        return False

    def execute_pick(self) -> bool:
        """Execute p1 (approach) through p3 (carry to transport pose),
        with automatic gripper control: opens the gripper, runs p1/p2,
        closes the gripper once p2 has settled and verifies the grasp,
        then runs p2b (retract, if planned) and p3 (carry, if planned).
        Aborts and returns False at the first step that fails to
        settle/complete — in particular, never calls close_gripper()
        unless Phase 2 was confirmed to have actually reached the grasp
        pose. Called by execute() (auto_gripper mode); also callable on
        its own."""
        print("\n=== Opening gripper ===")
        if not self.open_gripper():
            return self._abort_sequence("initial gripper open")

        if not self._execute_phase("Phase 1: Approach", self.p1, "p1 (approach)", 50):
            return self._abort_sequence("Phase 1 (approach)")
        if not self._execute_phase("Phase 2: Grasp close-in", self.p2, "p2 (grasp)", 50):
            return self._abort_sequence(
                "Phase 2 (grasp) — arm not settled at grasp pose, refusing to close gripper"
            )

        print("\n=== Closing gripper ===")
        if not self.close_gripper():
            return self._abort_sequence("gripper close")
        if not self._check_object_grasped():
            return self._abort_sequence(
                "grasp verification — object not detected in gripper, refusing to carry"
            )

        if self.p_retract is not None:
            if not self._execute_phase(
                "Phase 2b: Retract to handle clearance", self.p_retract, "p_retract (retract)", 50
            ):
                return self._abort_sequence("Phase 2b (retract)")

        if self.p3 is not None:
            if not self._execute_phase("Phase 3: Carry to transport pose", self.p3, "p3 (carry)", 50):
                return self._abort_sequence("Phase 3 (carry)")

        return True

    def execute_place(self) -> bool:
        """Navigate to the drop zone, add the drop-zone box to the HPP
        scene (navigate_to_drop_zone_and_add_box), plan p_place/p4/p5
        (plan_place) and execute them, open the gripper, run p4/p5, then
        navigate back to the initial point and remove the box from the
        scene again (ready for a new cycle). Aborts and returns False at
        the first step that fails to settle/complete. Called by
        execute() (auto_gripper mode) right after execute_pick(); also
        callable on its own (e.g. to redo just the place/release leg
        against an already-carried object)."""
        print("\n=== Navigating to drop zone ===")
        if not self.navigate_to_drop_zone_and_add_box():
            return self._abort_sequence("navigation to drop zone")

        if self.p3 is not None:
            # p_place/p4/p5 are only planned here, against the robot's
            # real post-navigation state, now that the drop-zone box is
            # part of the scene (see navigate_to_drop_zone_and_add_box).
            if not self.plan_place():
                return self._abort_sequence("planning place/release after navigation")

        if self.p_place is not None:
            if not self._execute_phase("Phase 3b: Place at drop zone", self.p_place, "p_place (place)", 50):
                return self._abort_sequence("Phase 3b (place)")

        print("\n=== Opening gripper ===")
        if not self.open_gripper():
            return self._abort_sequence("post-carry gripper open")

        if self.p4 is not None:
            if not self._execute_phase("Phase 4: Release / retreat", self.p4, "p4 (release)", 200):
                return self._abort_sequence("Phase 4 (release)")
        else:
            print("\n=== Phase 4: Release / retreat skipped ===")

        if self.p5 is not None:
            if not self._execute_phase("Phase 5: Return to carry pose", self.p5, "p5 (return to carry)", 50):
                return self._abort_sequence("Phase 5 (return to carry pose)")
        else:
            print("\n=== Phase 5: Return to carry pose skipped ===")

        print("\n=== Navigating back to initial point ===")
        if not self.navigate_to_initial_pose():
            return self._abort_sequence("navigation back to initial point")

        # The box's fixed pose is only meaningful while the base is
        # actually at the drop zone (the HPP scene is base-relative — see
        # module docstring) — remove it now that we've navigated away, so
        # the next cycle's plan_pick() finds a box-less scene, same as this one.
        self.remove_box_from_scene()

        print("\n=== Pick-and-drop complete ===")
        return True

    def execute(self, paths=None, auto_gripper: bool = True) -> bool:
        """
        Sample and publish MpcInput messages to the controller.

        paths : list of Path objects to execute in sequence.
                When None and auto_gripper=True (default), runs the full
                pick-and-drop sequence automatically, closing the gripper
                after p2, retracting to the grasped handle's own clearance
                (p2b, when available), then opening the gripper after p3
                (or p2/p2b if no carry phase), moving the arm back to the
                carry pose (p5) once retract (p4) has run, then navigating
                the base back to the initial point. When None and
                auto_gripper=False, all phases are streamed as a single
                continuous trajectory without gripper/navigation commands.
        auto_gripper : if True (default) and paths is None, gripper and
                       navigation are handled automatically between phases
                       — see execute_pick()/execute_place(), each callable
                       on its own too.

        Returns True only if every phase (and, in auto_gripper mode, every
        gripper/navigation action) completed and settled; False if the
        sequence was aborted partway through.
        """
        if self.p1 is None:
            print("No path available — run plan_pick() first.")
            return False

        self._next_msg_id = 0
        self._last_hold_msg = None
        self._last_executed_q = None
        self._last_executed_label = None

        if paths is not None:
            return self._execute_paths(self._named_paths(paths))

        if auto_gripper:
            if not self.execute_pick():
                return False
            return self.execute_place()
        return self._execute_paths(self._named_paths(self._planned_paths()))

    # ── Robot state sync ──────────────────────────────────────────────────────

    def sync_from_robot(self, timeout: float = 5.0) -> bool:
        """Update q_init and locked planning joints from the current robot state."""
        js_map = self._joint_state_map(timeout)
        if js_map is None:
            print("sync_from_robot: timeout — could not receive /joint_states")
            return False

        self._sync_fixed_joint_values(js_map)

        right_arm = self._arm_joint_state(js_map, "right")
        left_arm = self._arm_joint_state(js_map, "left")
        if left_arm is not None:
            self._active_arm_home = left_arm.tolist()
        if self._right_arm_idx is not None and right_arm is not None:
            self._other_arm_lock_values = right_arm.tolist()
        self.q_init = self._configuration_from_joint_state(js_map, q_seed=self.q_init)
        self.q_init = self._project_into_active_arm_range(self.q_init, tol=1e-3)
        print("  Rebuilding constraint graph with synced robot state …")
        self._setup_graph()
        status = [f"torso={self._fixed_joint_values['torso_lift_joint']:.3f}"]
        if left_arm is not None:
            status.append(f"left={np.round(left_arm, 3).tolist()}")
        if right_arm is not None:
            status.append(f"right={np.round(right_arm, 3).tolist()}")
        print(f"sync_from_robot: {'  '.join(status)}")
        return True

    def tuck_arm(self, sync_with_robot: bool = True, sync_timeout: float = 3.0) -> bool:
        """
        Recovery utility: send the active arm to its tuck configuration
        (LEFT_ARM_TUCK) from wherever it currently is, independent of any
        in-progress plan. Call this after an aborted/failed sequence to
        reset the arm to a known-safe pose before restarting the demo
        (plan_pick()/plan_and_execute()).
        Clears any previously planned phases (p1..p5, p_retract, p_place,
        p4, p5) — they're stale relative to the arm's new pose.
        """
        if sync_with_robot and not self.sync_from_robot(timeout=sync_timeout):
            print("tuck_arm: could not sync from robot — using last-known q_init.")

        q_seed = np.array(self.q_init).copy()
        arm_idx = self._active_arm_idx
        q_seed[arm_idx:arm_idx+7] = LEFT_ARM_TUCK

        transition = self._transition_return
        if transition is None:
            print("tuck_arm: no free-state self-loop available — cannot plan tuck motion.")
            return False

        generated, valid, q_tuck, err = self._generate_valid_config(
            transition, self.q_init, q_seed
        )
        if not generated or not valid:
            report = self._last_invalid_config_report
            detail = f" — {report}" if report is not None else ""
            print(
                f"tuck_arm: failed to project tuck config onto free state "
                f"(err={err:.2e}){detail}."
            )
            return False

        planner = TransitionPlanner(self.problem)
        planner.maxIterations(1000)
        shortcut = GraphRandomShortcut(self.problem)
        spline_opt = SplineGradientBased_bezier3(self.problem)
        q_goal = np.zeros((1, self.robot.configSize()), order='F')
        try:
            p_tuck = self._plan_transition_path(
                planner, q_goal, transition,
                self.q_init, q_tuck, "recovery (tuck arm)"
            )
        except Exception as e:
            print(f"tuck_arm: path planning failed: {e}")
            return False
        p_tuck = self._optimize_path(p_tuck, "tuck", shortcut, spline_opt)

        self.p1 = self.p2 = self.p_retract = self.p3 = self.p_place = self.p4 = self.p5 = None
        self.qpg = self.qg = self.q_retract = self.q_carry = self.q_drop = None

        self._next_msg_id = 0
        self._last_hold_msg = None
        self._last_executed_q = None
        self._last_executed_label = None

        return self._execute_phase("Recovery: tuck arm", p_tuck, "tuck (arm to tuck pose)", 200)

    # ── Gripper control ──────────────────────────────────────────────────────

    def _call_grasper_service(
        self,
        service_name: str,
        action_label: str,
        timeout: float = 5.0,
    ) -> bool:
        """Call an Empty gripper service (open/close), keeping the MPC
        controller fed with hold references both while waiting for the
        service to become available and while the call itself is pending —
        gripper actuation happens mid-sequence in execute(), so the arm
        trajectory reference must not go stale during either wait. Runs
        against the persistent execution node (not a throwaway one) so the
        shared /joint_states cache is available from the very first gripper
        call, not just once a phase has executed."""
        node = self._execution_node()
        client = self._ensure_grasper_client(node, service_name)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if client.wait_for_service(timeout_sec=min(0.2, DT)):
                break
            self._publish_hold_reference(node)
        else:
            print(f"Gripper service '{service_name}' is not available.")
            return False

        future = client.call_async(Empty.Request())
        self._spin_future_with_hold(node, future)

        if future.cancelled():
            print(f"Gripper {action_label} request to '{service_name}' was cancelled.")
            return False

        exc = future.exception()
        if exc is not None:
            print(f"Gripper {action_label} failed via '{service_name}': {exc}")
            return False

        print(f"Left gripper {action_label} via '{service_name}'.")
        return True

    def open_gripper(self, timeout: float = 5.0) -> bool:
        """Open the left gripper in Gazebo. The grasper service itself blocks
        until the physical motion completes (or its own timeout elapses), so
        its return value is the completion signal — see gripper_grasper_srv.py."""
        return self._call_grasper_service(
            LEFT_GRIPPER_RELEASE_SERVICE, "open", timeout=timeout
        )

    def close_gripper(self, timeout: float = 5.0) -> bool:
        """Close the left gripper in Gazebo. The grasper service itself blocks
        until the physical motion completes (or its own timeout elapses), so
        its return value is the completion signal — see gripper_grasper_srv.py."""
        return self._call_grasper_service(
            LEFT_GRIPPER_GRASP_SERVICE, "close", timeout=timeout
        )

    # ── Navigation ───────────────────────────────────────────────────────────

    def _navigate_to(
        self, frame_id: str, x: float, y: float, yaw: float, label: str,
        timeout: float = 10.0,
    ) -> bool:
        """Send the base to (x, y, yaw) in frame_id via the navigate_to_pose
        action, keeping the MPC controller fed with hold references while
        waiting — mirrors _call_grasper_service. `timeout` only bounds
        waiting for the action server to become available; once the goal is
        sent, this blocks until the navigation result arrives (no separate
        deadline), since the real drive time depends on distance. `label`
        is only used for the printed progress/result messages."""
        if not self.enforce_navigation:
            print(f"  enforce_navigation is False — faking navigation to {label}.")
            return True

        node = self._execution_node()
        client = self._ensure_nav_action_client(node)

        deadline = time.time() + timeout
        while time.time() < deadline:
            if client.server_is_ready():
                break
            rclpy.spin_once(node, timeout_sec=min(0.2, DT))
            self._publish_hold_reference(node)
        else:
            print(f"Navigation action server '{NAVIGATE_TO_POSE_ACTION}' is not available.")
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = frame_id
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = float(np.sin(yaw / 2.0))
        goal_msg.pose.pose.orientation.w = float(np.cos(yaw / 2.0))

        print(f"  Sending navigation goal ({label}): x={x}, y={y}, "
              f"yaw={yaw} (frame='{frame_id}')")
        send_goal_future = client.send_goal_async(goal_msg)
        self._spin_future_with_hold(node, send_goal_future)

        exc = send_goal_future.exception()
        if exc is not None:
            print(f"Navigation goal request failed: {exc}")
            return False

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            print("Navigation goal was rejected.")
            return False

        print("  Navigation goal accepted, waiting for the base to arrive …")
        result_future = goal_handle.get_result_async()
        self._spin_future_with_hold(node, result_future)

        exc = result_future.exception()
        if exc is not None:
            print(f"Navigation request failed: {exc}")
            return False

        status = result_future.result().status
        success = status == GoalStatus.STATUS_SUCCEEDED
        print(f"Navigation to {label} {'succeeded' if success else 'failed'} (status={status}).")
        return success

    def navigate_to_drop_zone(self, timeout: float = 10.0) -> bool:
        """Send the base to NAV_TARGET_{X,Y,YAW} via the navigate_to_pose action."""
        return self._navigate_to(
            NAV_TARGET_FRAME, NAV_TARGET_X, NAV_TARGET_Y, NAV_TARGET_YAW,
            "drop zone", timeout=timeout,
        )

    def navigate_to_drop_zone_and_add_box(self, timeout: float = 10.0) -> bool:
        """Send the base to the drop zone (navigate_to_drop_zone) and, once
        it has actually arrived, add the drop-zone box to the HPP scene
        (add_box_to_scene) — the box is otherwise absent from the scene
        the whole time p1/p2/p2b/p3 plan and execute (see module
        docstring). Returns False without touching the scene if navigation
        itself fails. Called from execute_place(); also callable
        on its own to inspect/debug the scene switch independently of
        planning p_place/p4/p5 (plan_place)."""
        if not self.navigate_to_drop_zone(timeout=timeout):
            return False
        self.add_box_to_scene()
        return True

    def navigate_to_initial_pose(self, timeout: float = 10.0) -> bool:
        """Send the base back to NAV_INITIAL_{X,Y,YAW} via the
        navigate_to_pose action, so a new pick-and-drop cycle can start
        from the same place."""
        return self._navigate_to(
            NAV_INITIAL_FRAME, NAV_INITIAL_X, NAV_INITIAL_Y, NAV_INITIAL_YAW,
            "initial point", timeout=timeout,
        )

    def _check_object_grasped(self, timeout: float = 2.0) -> bool:
        """If /gripper_left_grasper_srv/is_grasped has an active publisher,
        read it to confirm close_gripper() actually made contact with the
        object — gripper_grasper_srv.py's grasp_cb reports success
        unconditionally on its own timeout even with no contact detected, so
        the service call succeeding doesn't by itself mean anything was
        grasped. Where nothing publishes this topic, the signal isn't
        available in the current environment, so skip the check rather than
        block on it. Set self.enforce_grasp_check = False to bypass entirely
        (e.g. in simulation, where grasping/contact isn't reliable so the
        topic can be published but always report no grasp)."""
        if not self.enforce_grasp_check:
            print("  enforce_grasp_check is False — skipping grasp verification.")
            return True

        topic = "/gripper_left_grasper_srv/is_grasped"
        node = self._execution_node()
        deadline = time.time() + 1.0
        while time.time() < deadline and node.count_publishers(topic) == 0:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.count_publishers(topic) == 0:
            print(f"  {topic} has no publisher — skipping grasp verification.")
            return True

        msg = self._wait_for_topic_message(node, Bool, topic, timeout)
        if msg is None:
            print(f"WARNING: no response from {topic} within {timeout}s.")
            return False
        if not msg.data:
            print("WARNING: gripper did not detect a grasped object.")
        return msg.data

    # ── Object pose update ────────────────────────────────────────────────────

    @staticmethod
    def _rotation_from_xyzw(q) -> np.ndarray:
        """q is [x, y, z, w] — the quaternion component order used by ROS
        messages (geometry_msgs/Quaternion) and by this module's own pose
        tuples, as opposed to Pinocchio/Eigen's [w, x, y, z]."""
        x, y, z, w = np.asarray(q, dtype=float)
        norm = np.linalg.norm([x, y, z, w])
        if norm == 0.0:
            raise ValueError("Quaternion norm is zero.")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ])

    @classmethod
    def _se3_from_translation_quaternion(cls, t, q) -> pin.SE3:
        return pin.SE3(cls._rotation_from_xyzw(q), np.asarray(t, dtype=float))

    @classmethod
    def _pose_to_se3(cls, pose) -> pin.SE3:
        """Convert common HappyPose/ROS/numpy pose containers to pin.SE3."""
        if isinstance(pose, pin.SE3):
            return pose.copy()

        if isinstance(pose, dict):
            for key in ("TCO", "T_CO", "camera_T_object", "matrix", "T"):
                if key in pose:
                    return cls._pose_to_se3(pose[key])
            t = pose.get("translation", pose.get("t"))
            q = pose.get("quaternion", pose.get("q"))
            if t is not None and q is not None:
                return cls._se3_from_translation_quaternion(t, q)

        if hasattr(pose, "pose"):
            return cls._pose_to_se3(pose.pose)
        if hasattr(pose, "transform"):
            return cls._pose_to_se3(pose.transform)
        if hasattr(pose, "position") and hasattr(pose, "orientation"):
            t = [pose.position.x, pose.position.y, pose.position.z]
            q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            return cls._se3_from_translation_quaternion(t, q)
        if hasattr(pose, "translation") and hasattr(pose, "rotation"):
            t = [pose.translation.x, pose.translation.y, pose.translation.z]
            q = [pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w]
            return cls._se3_from_translation_quaternion(t, q)

        arr = np.asarray(pose, dtype=float)
        if arr.shape == (4, 4):
            return pin.SE3(arr[:3, :3], arr[:3, 3])
        if arr.shape == (7,):
            return cls._se3_from_translation_quaternion(arr[:3], arr[3:])

        raise ValueError(
            "Expected a 4x4 transform, [x, y, z, qx, qy, qz, qw], "
            "a dict with TCO/translation/quaternion, or a ROS Pose/Transform."
        )

    @staticmethod
    def _pose_frame_id(pose):
        if isinstance(pose, dict):
            return pose.get("frame_id", pose.get("camera_frame", pose.get("frame")))
        header = getattr(pose, "header", None)
        if header is not None:
            return header.frame_id
        return None

    def _lookup_transform_as_se3(
        self,
        target_frame: str,
        source_frame: str,
        timeout: float,
    ) -> pin.SE3:
        from rclpy.duration import Duration
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener

        last_error = None
        with self._borrow_ros_node("hpp_happypose_tf_lookup") as node:
            tf_buffer = Buffer()
            TransformListener(tf_buffer, node)
            deadline = time.time() + timeout
            while time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                try:
                    transform = tf_buffer.lookup_transform(
                        target_frame,
                        source_frame,
                        Time(),
                        timeout=Duration(seconds=0.1),
                    )
                    return self._pose_to_se3(transform.transform)
                except Exception as exc:
                    last_error = exc

        raise RuntimeError(
            f"Could not transform HappyPose pose from '{source_frame}' to "
            f"'{target_frame}' within {timeout}s: {last_error}"
        )

    def update_object_pose(self, t: list, q: list = None) -> None:
        """Update object pose in q_init (and Viser if open).

        t: [x, y, z] in base_link frame
        q: [qx, qy, qz, qw] orientation (default: identity)
        """
        t = np.array(t)
        q = np.array(q) if q is not None else np.array([0., 0., 0., 1.])
        self._set_obj_bounds(t[0], t[1], t[2])
        obj_idx = self._obj_idx
        self.q_init[obj_idx:obj_idx+3] = t
        self.q_init[obj_idx+3:obj_idx+7] = q
        if hasattr(self, "_viewer"):
            self._viewer(self.q_init)
        print(f"Object pose updated: t={np.round(t, 4).tolist()}, q={np.round(q, 4).tolist()}")

    def _collision_involves_object_and_table(self, report) -> bool:
        names = str(report).lower()
        return "table" in names and self._obj_name.lower() in names

    def _resolve_table_collision(self, t: list, q: list) -> list:
        """If (t, q) puts the object in collision with the table, lift it
        along z (keeping x/y and orientation) until clear. Returns t
        unchanged if there's no table collision, or if it can't be cleared
        within TABLE_COLLISION_MAX_LIFT (a warning is printed in that case)."""
        q_candidate = np.array(self.q_init, copy=True)
        obj_idx = self._obj_idx

        def set_z(z):
            q_candidate[obj_idx:obj_idx + 3] = [t[0], t[1], z]
            q_candidate[obj_idx + 3:obj_idx + 7] = q

        set_z(t[2])
        ok, report = self.problem.isConfigValid(q_candidate)
        if ok:
            return t
        if not self._collision_involves_object_and_table(report):
            print("  WARNING: detected object pose is invalid but not due to a "
                  "table collision; leaving pose as detected.")
            return t

        # Stepping z upward and re-checking collision is simpler and more
        # robust than solving for the exact clearance height directly: the
        # collision checker doesn't expose a penetration depth, only a
        # boolean valid/invalid per config.
        z = t[2]
        n_steps = int(TABLE_COLLISION_MAX_LIFT / TABLE_COLLISION_STEP)
        for _ in range(n_steps):
            z += TABLE_COLLISION_STEP
            set_z(z)
            ok, report = self.problem.isConfigValid(q_candidate)
            if ok:
                print(f"  Object raised {z - t[2]:.3f} m to clear the table.")
                return [t[0], t[1], z]
            if not self._collision_involves_object_and_table(report):
                break  # a different collision appeared; stop chasing it

        print(f"  WARNING: could not clear table collision within "
              f"{TABLE_COLLISION_MAX_LIFT} m; using original detected z.")
        return t

    def _detect_present_object(self, timeout: float = 3.0):
        """Read one /happypose/detections message (no class filter yet —
        object identity isn't known before the model is built) and return
        (obj_name, pose, frame_id) for the highest-confidence detection that
        matches a shipped T-LESS asset, or None if no such detection arrives
        within timeout (caller falls back to the configured default)."""
        with self._borrow_ros_node("hpp_object_detect") as node:
            msg = self._wait_for_topic_message(
                node, Detection2DArray, "/happypose/detections", timeout
            )
        if msg is None:
            return None

        available = set(_list_available_objects())
        candidates = []
        for d in msg.detections:
            if not d.results:
                continue
            obj_name = _obj_name_from_class_id(
                d.results[0].hypothesis.class_id, OBJECT_DATASET
            )
            if obj_name in available:
                candidates.append((d.results[0].hypothesis.score, obj_name, d))
        if not candidates:
            return None

        _, obj_name, best = max(candidates, key=lambda c: c[0])
        return obj_name, best.results[0].pose.pose, best.header.frame_id

    def _reload_object_model(self, detected_name: str) -> None:
        """Switch the loaded object to `detected_name` and rebuild the HPP
        model/graph around it — called from update_object_pose_from_happypose
        when vision detects a different object than the one currently
        loaded. Any in-progress plan is invalidated by this since it
        references the old model."""
        print(
            f"  Object changed: '{self._obj_name}' -> '{detected_name}'. "
            "Reloading HPP model …"
        )
        self._obj_name = detected_name
        self._setup_model()
        self._setup_graph()
        # Stale Path/config objects reference the old problem/graph —
        # drop them so execute()/compare_pose() can't use them by accident.
        self.p1 = self.p2 = self.p_retract = self.p3 = self.p_place = self.p4 = self.p5 = None
        self.qpg = self.qg = self.q_retract = self.q_carry = self.q_drop = None
        if hasattr(self, "_viewer"):
            # init_viewer() rebuilds the Viewer against the new
            # self.robot/problem/graph and reloads its mesh geometry —
            # just re-pushing q_init to the old Viewer only moves the
            # previous object's already-loaded mesh, it never swaps it.
            self.init_viewer(open=False)

    def update_object_pose_from_happypose(
        self,
        happypose_pose=None,
        camera_frame: str = None,
        base_T_camera=None,
        base_frame: str = "base_footprint",
        timeout: float = 5.0,
        detections_timeout: float = 5.0,
    ) -> pin.SE3:
        """Update the object pose from a HappyPose camera_T_object estimate.

        If happypose_pose is not given, the latest /happypose/detections
        message is read and the highest-confidence detection (across ALL
        shipped T-LESS assets, not just the currently loaded one) is used
        instead. If that detection names a different object than the one
        currently loaded, the HPP model/graph is rebuilt for it — this is
        the single place object identity is (re)detected; Orchestrator()
        itself just loads the configured fallback object at construction.
        Otherwise happypose_pose may be a 4x4 TCO matrix,
        [x, y, z, qx, qy, qz, qw], a dict containing TCO/translation/quaternion,
        or a ROS Pose/Transform.
        If base_T_camera is not passed, camera_frame is looked up in TF.

        detections_timeout/timeout are generous (5s) because both waits race
        ROS2 discovery of a freshly created node's subscriptions (the
        /happypose/detections publisher and the TF broadcasters); on a busy
        system discovery can take longer than the previous 1s default,
        which is what caused intermittent "frame does not exist" failures.
        Nothing is written to q_init until both reads succeed — a timeout or
        an unrecognized object raises RuntimeError instead of updating the pose.
        Returns the resulting base_T_object transform.
        """
        if happypose_pose is None:
            detected = self._detect_present_object(detections_timeout)
            if detected is None:
                raise RuntimeError(
                    f"No /happypose/detections message received within {detections_timeout}s."
                )
            detected_name, happypose_pose, detected_frame = detected
            camera_frame = camera_frame or detected_frame

            if detected_name != self._obj_name:
                self._reload_object_model(detected_name)

        camera_T_object = self._pose_to_se3(happypose_pose)

        if base_T_camera is None:
            camera_frame = camera_frame or self._pose_frame_id(happypose_pose)
            if not camera_frame:
                raise ValueError(
                    "camera_frame is required when base_T_camera is not provided."
                )
            base_T_camera = self._lookup_transform_as_se3(
                base_frame, camera_frame, timeout
            )
        else:
            base_T_camera = self._pose_to_se3(base_T_camera)

        base_T_object = base_T_camera * camera_T_object
        quat = pin.Quaternion(base_T_object.rotation)
        q = [quat.x, quat.y, quat.z, quat.w]
        t = self._resolve_table_collision(base_T_object.translation, q)
        self.update_object_pose(t, q)
        base_T_object = pin.SE3(base_T_object.rotation, np.array(t))
        return base_T_object

    # ── Pose comparison ───────────────────────────────────────────────────────

    def compare_pose(self, q_ref=None, timeout: float = 5.0):
        """Compare a reference configuration vs the current robot state at the EE."""
        ref_label = None
        if q_ref is None:
            if self._last_executed_q is not None:
                q_ref = np.array(self._last_executed_q, copy=True)
                ref_label = self._last_executed_label or "last executed path endpoint"
            elif self.qg is not None:
                q_ref = np.array(self.qg, copy=True)
                ref_label = "qg (default grasp reference)"
            else:
                print("No reference pose available — run plan_pick() or pass q_ref explicitly.")
                return
        elif hasattr(q_ref, "timeRange"):
            q_ref = np.array(q_ref.eval(q_ref.timeRange().second)[0], copy=True)
            ref_label = "path endpoint"
        else:
            q_ref = self._as_full_configuration(q_ref)
            ref_label = "provided configuration"

        q_ref = self._as_full_configuration(q_ref)

        js_map = self._joint_state_map(timeout)
        if js_map is None:
            print("compare_pose: timeout reading /joint_states")
            return

        q_actual = self._configuration_from_joint_state(js_map, q_seed=self.q_init)
        arm_idx = self._active_arm_idx

        data_ref = self.model.createData()
        data_act = self.model.createData()
        pin.forwardKinematics(self.model, data_ref, q_ref)
        pin.updateFramePlacements(self.model, data_ref)
        pin.forwardKinematics(self.model, data_act, q_actual)
        pin.updateFramePlacements(self.model, data_act)

        T_ref = data_ref.oMf[self._ee_frame_id]
        T_act = data_act.oMf[self._ee_frame_id]
        delta = T_ref.inverse() * T_act
        pos_err_mm  = np.linalg.norm(delta.translation) * 1000.0
        rot_err_deg = np.degrees(np.linalg.norm(pin.log3(delta.rotation)))

        print(f"\n{'='*58}")
        print(f"  Pose comparison  (reference vs actual robot state)")
        print(f"{'='*58}")
        print(f"  Reference       : {ref_label}")
        print(f"  EE planned [m] : {np.round(T_ref.translation, 4)}")
        print(f"  EE actual  [m] : {np.round(T_act.translation, 4)}")
        print(f"  Position error : {pos_err_mm:.1f} mm")
        print(f"  Rotation error : {rot_err_deg:.2f} °")
        # Per-joint error = reference config minus the live robot state, for
        # each of the 7 active-arm joints — helps pinpoint which joint is
        # off when the aggregate EE error above looks large.
        print(f"\n  Per-joint error — {self._active_arm_side} arm [rad / °]:")
        for i in range(7):
            e = q_ref[arm_idx + i] - q_actual[arm_idx + i]
            print(
                f"    arm_{self._active_arm_side}_{i+1}_joint : "
                f"{e:+.4f} rad  ({np.degrees(e):+.2f}°)"
            )
        print(f"{'='*58}\n")

    # ── Visualisation ─────────────────────────────────────────────────────────

    def init_viewer(self, open: bool = True):
        from pyhpp_viser import Viewer
        _patch_viser_tab_group_remove_bug()
        self._viewer = Viewer(self.robot)
        self._viewer.initViewer(open=open, loadModel=True)
        self._viewer.setProblem(self.problem)
        self._viewer.setGraph(self.graph)
        self._viewer(self.q_init)
        print("Viser viewer ready.  Use o.view(q) or o.play(path).")

    def view(self, q=None):
        if not hasattr(self, "_viewer"):
            self.init_viewer()
        self._viewer(q if q is not None else self.q_init)

    def play(self, path, n: int = 100, dt: float = 0.05):
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

    # ── Controller activation ─────────────────────────────────────────────────

    def activate_lfc(self) -> None:
        """Switch the active arm controller to LFC + JSE (torque control)."""
        controller_name = f"arm_{self._active_arm_side}_controller"
        print("Activating LFC controllers …")
        subprocess.run(
            [
                "ros2", "control", "switch_controllers",
                "--deactivate", controller_name,
                "--activate",
                "linear_feedback_controller",
                "joint_state_estimator",
            ],
            check=True,
        )
        print("LFC controllers active.")

    def deactivate_lfc(self) -> None:
        """Return the active arm to its position controller."""
        controller_name = f"arm_{self._active_arm_side}_controller"
        print("Deactivating LFC controllers …")
        subprocess.run(
            [
                "ros2", "control", "switch_controllers",
                "--deactivate",
                "linear_feedback_controller",
                "joint_state_estimator",
                "--activate", controller_name,
            ],
            check=True,
        )
        print(f"{controller_name} active.")

    # ── Combined ──────────────────────────────────────────────────────────────

    def plan_and_execute(self, max_attempts: int = 100) -> bool:
        if self.plan_pick(max_attempts=max_attempts):
            return self.execute()
        return False
