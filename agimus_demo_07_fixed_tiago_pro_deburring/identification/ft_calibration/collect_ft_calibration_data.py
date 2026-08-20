#!/usr/bin/env python3
"""
F/T sensor calibration data collector for Tiago Pro — automatic version.

Same pattern as `collect_calibration_data.py` (the Figaroh kinematic
calibration collector): step through a list of arm configurations, settle,
record a sample, move on. The two differ in what "ground truth" they need:

  - collect_calibration_data.py identifies *kinematic* offsets, so it needs
    an external reference (mocap EE pose) to compare against the model's FK.
  - This script identifies *inertial* parameters of the pal-atc tool (mass,
    center of mass) plus the F/T sensor's zero-offset bias. Gravity is a
    known, fixed direction — no external tracking system needed. It just
    needs the raw wrench and the joint state at each pose.

Reads poses from `ft_calibration_poses.yaml` (generate_ft_calibration_poses.py,
same folder) — D-optimal for THIS regressor (10 unknowns: mass, m_com(3),
bias(6)), collision-checked against the mounted deburring tool + the floor.
Do not reuse ../optimal_configs.yaml (the Figaroh kinematic poses): its
collision model predates the tool being mounted and doesn't know about it
(see project_demo07_force_feedback_scoping memory) — the arm can graze the
floor or itself with the real tool attached even on a "collision-free" pose
from that file.

⚠️ Run this AFTER the Figaroh kinematic calibration is applied (or at least
with a reasonably accurate URDF): the gravity direction in the sensor frame
at each pose is computed via FK, so a mis-calibrated kinematic chain would
bias the mass/CoM estimate downstream.

Prerequisites:
  - arm_right_controller and torso_controller active (NOT agimus_controller)
  - play_motion2 running (for the horizontal_right via-pose)
  - force_torque_sensor_broadcaster publishing the raw wrench (confirmed
    topic: /ft_sensor_right_controller/wrench, see
    project_demo07_force_feedback_scoping memory)

Usage:
    python3 identification/ft_calibration/collect_ft_calibration_data.py
    python3 identification/ft_calibration/collect_ft_calibration_data.py \\
        --configs identification/ft_calibration/ft_calibration_poses.yaml \\
        --output  identification/ft_calibration/ft_calibration_samples.csv \\
        --wrench-topic /ft_sensor_right_controller/wrench \\
        --duration 5.0
"""

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import WrenchStamped
from play_motion2_msgs.action import PlayMotion2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

_IDENTIFICATION = Path(__file__).parent
_CONFIGS_DEFAULT = _IDENTIFICATION / "ft_calibration_poses.yaml"
_OUTPUT_DEFAULT = _IDENTIFICATION / "ft_calibration_samples.csv"

ARM_JOINTS = [
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "arm_right_5_joint",
    "arm_right_6_joint",
    "arm_right_7_joint",
]
TORSO_JOINTS = ["torso_lift_joint"]
ACTIVE_JOINTS = TORSO_JOINTS + ARM_JOINTS

_ARM_ACTION = "/arm_right_controller/follow_joint_trajectory"
_TORSO_ACTION = "/torso_controller/follow_joint_trajectory"
_PLAY_MOTION_ACTION = "play_motion2"
_VIA_MOTION_NAME = "horizontal_right"

_SETTLE_S = 5.0
_AVERAGE_WINDOW_S = 1.0  # average wrench samples received during this trailing window
_FRESHNESS_S = 0.5
_STILL_THRESHOLD = 0.005  # rad/s
_FROZEN_WRENCH_EPS = 1e-6  # bit-identical wrench across poses => topic likely stuck


# ── Utility ───────────────────────────────────────────────────────────────────


def _djs_value(djs_msg, jname, interface="absolute_position"):
    """Look up an interface value for a joint in a DynamicJointState message.

    When interface='absolute_position', falls back to 'position' if
    unavailable (e.g. torso_lift_joint). Other interfaces (e.g. 'velocity')
    are looked up as-is.
    """
    for candidate in (jname, f"tiago_pro/{jname}"):
        try:
            idx = djs_msg.joint_names.index(candidate)
        except ValueError:
            continue
        iv = djs_msg.interface_values[idx]
        imap = dict(zip(iv.interface_names, iv.values))
        if interface == "absolute_position":
            return imap.get("absolute_position", imap.get("position"))
        return imap.get(interface)
    return None


def _print_error_table(target_vals, current_vals):
    print(f"\n  {'Joint':<30} {'Target':>9} {'Current':>9} {'Error':>9}")
    print("  " + "─" * 62)
    for jname, tgt, cur in zip(ACTIVE_JOINTS, target_vals, current_vals):
        err = np.degrees(cur - tgt)
        flag = "✓" if abs(err) < 1.0 else ("~" if abs(err) < 3.0 else "✗")
        print(
            f"  {jname:<30} {np.degrees(tgt):>8.2f}° {np.degrees(cur):>8.2f}°  {err:>+7.2f}°  {flag}"
        )


def _make_trajectory(joint_names, positions, duration_sec):
    traj = JointTrajectory()
    traj.joint_names = joint_names
    pt = JointTrajectoryPoint()
    pt.positions = [float(p) for p in positions]
    pt.velocities = [0.0] * len(positions)
    secs = int(duration_sec)
    nsecs = int((duration_sec - secs) * 1e9)
    pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
    traj.points = [pt]
    return traj


# ── ROS 2 node ────────────────────────────────────────────────────────────────


class FTCalibrationCollector(Node):
    def __init__(self, output_path, wrench_topic, duration_sec):
        super().__init__("ft_calibration_collector")
        self._output_path = output_path
        self._duration = duration_sec
        self._samples = []
        self._last_target = None
        self._last_wrench_mean = None  # guard against a stuck wrench topic

        self._lock = threading.Lock()
        self._js_msg: DynamicJointState | None = None
        self._js_last_rx = None  # this node's clock, not the msg's header.stamp
        self._wrench_buf = deque()  # (recv_time_s, wrench[6]) trailing window

        self.create_subscription(
            DynamicJointState,
            "/joint_torque_state_broadcaster/dynamic_joint_states",
            self._js_cb,
            10,
        )
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_cb, 10)

        self._arm_client = ActionClient(self, FollowJointTrajectory, _ARM_ACTION)
        self._torso_client = ActionClient(self, FollowJointTrajectory, _TORSO_ACTION)
        self._play_motion_client = ActionClient(self, PlayMotion2, _PLAY_MOTION_ACTION)

    def _js_cb(self, msg):
        with self._lock:
            self._js_msg = msg
            self._js_last_rx = self.get_clock().now()

    def _wrench_cb(self, msg):
        w = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ]
        )
        now = time.monotonic()
        with self._lock:
            self._wrench_buf.append((now, w))
            cutoff = now - _AVERAGE_WINDOW_S
            while self._wrench_buf and self._wrench_buf[0][0] < cutoff:
                self._wrench_buf.popleft()

    def _is_fresh(self, last_rx):
        """True if `last_rx` (this node's clock) is recent — deliberately not
        the message's header.stamp, see collect_calibration_data.py."""
        now = self.get_clock().now()
        return (now - last_rx).nanoseconds * 1e-9 < _FRESHNESS_S

    def _is_still(self, js):
        for jname in ACTIVE_JOINTS:
            vel = _djs_value(js, jname, interface="velocity") or 0.0
            if abs(vel) > _STILL_THRESHOLD:
                return False
        return True

    def _wait_for_servers(self, timeout=10.0):
        self.get_logger().info("Waiting for action servers ...")
        if not self._arm_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"Action server not available: {_ARM_ACTION}")
        if not self._torso_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"Action server not available: {_TORSO_ACTION}")
        if not self._play_motion_client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"Action server not available: {_PLAY_MOTION_ACTION}")
        self.get_logger().info("Action servers ready.")

    def _send_goal(self, client, joint_names, positions, duration_sec):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = _make_trajectory(joint_names, positions, duration_sec)
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            self.get_logger().warn("Goal send timed out — is the controller running?")
            return False
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Goal rejected!")
            return False
        result_future = handle.get_result_async()
        timeout = duration_sec + 15.0
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().warn(
                f"Goal execution timed out after {timeout:.0f}s — skipping."
            )
            return False
        return result_future.result().status == GoalStatus.STATUS_SUCCEEDED

    def _run_play_motion(self, motion_name, skip_planning=False, timeout=60.0):
        goal = PlayMotion2.Goal()
        goal.motion_name = motion_name
        goal.skip_planning = skip_planning
        future = self._play_motion_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done():
            self.get_logger().warn(f"play_motion2 '{motion_name}' send timed out.")
            return False
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn(f"play_motion2 '{motion_name}' goal rejected.")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done():
            self.get_logger().warn(f"play_motion2 '{motion_name}' execution timed out.")
            return False
        error = result_future.result().result.error
        if error:
            self.get_logger().warn(f"play_motion2 '{motion_name}' failed: {error}")
            return False
        return True

    def move_to(self, target_vals):
        torso_vals = [target_vals[ACTIVE_JOINTS.index(j)] for j in TORSO_JOINTS]
        arm_vals = [target_vals[ACTIVE_JOINTS.index(j)] for j in ARM_JOINTS]

        self.get_logger().info(f"Via pose: {_VIA_MOTION_NAME} ...")
        self._run_play_motion(_VIA_MOTION_NAME)

        self.get_logger().info("Moving torso ...")
        ok_t = self._send_goal(
            self._torso_client, TORSO_JOINTS, torso_vals, self._duration
        )
        self.get_logger().info("Moving arm ...")
        ok_a = self._send_goal(self._arm_client, ARM_JOINTS, arm_vals, self._duration)
        return ok_t and ok_a

    def wait_and_record(self):
        self.get_logger().info(f"Settling ({_SETTLE_S:.1f}s) ...")
        time.sleep(_SETTLE_S)

        with self._lock:
            js = self._js_msg
            js_last_rx = self._js_last_rx

        if js is None:
            self.get_logger().warn(
                "No dynamic_joint_states received — skipping sample."
            )
            return False
        if js_last_rx is None or not self._is_fresh(js_last_rx):
            self.get_logger().warn("dynamic_joint_states is stale — skipping.")
            return False
        if not self._is_still(js):
            self.get_logger().warn("Robot still moving — waiting 1s more ...")
            time.sleep(1.0)
            with self._lock:
                js = self._js_msg
            if not self._is_still(js):
                self.get_logger().warn("Still moving — skipping sample.")
                return False

        with self._lock:
            wrench_samples = [w for _, w in self._wrench_buf]
        if not wrench_samples:
            self.get_logger().warn(
                "No wrench samples in the trailing window — skipping "
                "(is the wrench topic name correct?)."
            )
            return False
        wrench_mean = np.mean(wrench_samples, axis=0)
        wrench_std = np.std(wrench_samples, axis=0)

        if self._last_wrench_mean is not None and np.allclose(
            wrench_mean, self._last_wrench_mean, atol=_FROZEN_WRENCH_EPS
        ):
            self.get_logger().error(
                "Wrench mean identical to previous sample — topic may be "
                "frozen! Skipping."
            )
            return False
        self._last_wrench_mean = wrench_mean

        sample = {"fx": wrench_mean[0], "fy": wrench_mean[1], "fz": wrench_mean[2]}
        sample.update({"tx": wrench_mean[3], "ty": wrench_mean[4], "tz": wrench_mean[5]})
        for jname in ACTIVE_JOINTS:
            val = _djs_value(js, jname)
            if val is None:
                self.get_logger().warn(f"Joint '{jname}' missing — skipping.")
                return False
            sample[jname] = val

        self._samples.append(sample)
        self.get_logger().info(
            f"[+] sample #{len(self._samples)}  "
            f"f=({wrench_mean[0]:+.2f}, {wrench_mean[1]:+.2f}, {wrench_mean[2]:+.2f}) N  "
            f"(std {np.linalg.norm(wrench_std[:3]):.3f} N over "
            f"{len(wrench_samples)} samples)"
        )

        if self._last_target is not None:
            cur_vals = [
                _djs_value(js, jname) or 0.0 for jname in ACTIVE_JOINTS
            ]
            _print_error_table(self._last_target, cur_vals)

        return True

    def save(self):
        if not self._samples:
            print("No samples to save.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(self._output_path)), exist_ok=True)
        fieldnames = ["fx", "fy", "fz", "tx", "ty", "tz"] + ACTIVE_JOINTS
        with open(self._output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._samples)
        print(f"\nSaved {len(self._samples)} samples → {self._output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Automatic F/T sensor calibration data collection for Tiago Pro."
    )
    parser.add_argument("--configs", default=str(_CONFIGS_DEFAULT))
    parser.add_argument("--output", default=str(_OUTPUT_DEFAULT))
    parser.add_argument(
        "--wrench-topic", default="/ft_sensor_right_controller/wrench"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Trajectory duration per config in seconds (default: 5)",
    )
    parsed, ros_args = parser.parse_known_args()

    with open(parsed.configs) as f:
        cfg_data = yaml.safe_load(f)
    all_configs = cfg_data["calibration_joint_configurations"]
    print(f"Loaded {len(all_configs)} configurations from {parsed.configs}.")
    print(f"Wrench topic: {parsed.wrench_topic}")

    rclpy.init(args=ros_args)
    node = FTCalibrationCollector(parsed.output, parsed.wrench_topic, parsed.duration)
    node._wait_for_servers()

    spin_done = threading.Event()

    def _spin():
        while not spin_done.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)

    threading.Thread(target=_spin, daemon=True).start()

    time.sleep(1.0)

    print("\nStarting automated collection. Ctrl+C to abort.\n")

    try:
        idx = 0
        while idx < len(all_configs):
            target_vals = all_configs[idx]

            print(f"\n{'=' * 60}")
            print(f"Configuration {idx + 1}/{len(all_configs)}")
            print(f"{'=' * 60}")
            for jname, val in zip(ACTIVE_JOINTS, target_vals):
                print(f"  {jname:<30}  {np.degrees(val):>+8.2f}°")

            node._last_target = target_vals

            ok = node.move_to(target_vals)
            if not ok:
                print("  [warn] motion failed — skipping.")
                idx += 1
                continue

            recorded = node.wait_and_record()
            idx += 1
            if not recorded:
                print("  [warn] recording failed — skipping.")

    except KeyboardInterrupt:
        pass

    spin_done.set()
    node.save()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
