#!/usr/bin/env python3
"""
Publish a MuJoCo MJCF model built from the live /robot_description URDF, on
/mujoco_robot_description (the topic mujoco_ros2_control's MujocoSystemInterface
waits on when the URDF's <hardware> has no `mujoco_model` param).

Why not the package's own `make_mjcf_from_robot_description.py`: it pip-installs
trimesh/obj2mjcf/coacd/mujoco into a venv at first run (network + wheels pinned
for a different Python than this nix env's) -- and is documented as "highly
experimental". MuJoCo already reads URDF natively (STL meshes, which is all
pal_sea_arm_description ships), so this only needs to:
  - resolve package:// URIs to absolute paths (the plugin parses the XML string
    with no base directory),
  - drop the xacro's <mujoco> block (a PAL-converter-format tag, not MuJoCo's
    native URDF <mujoco><compiler/>),
  - add one torque <motor> per actuated joint (ros2_control `effort` command
    interfaces), limited to the URDF effort limit.

Usage (see mujoco_smoketest.launch.py): runs alongside robot_state_publisher;
exits once the model is published (transient_local keeps it available to the
late-joining plugin subscriber).
"""
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

_PACKAGE_URI = re.compile(r"package://([^/]+)/")


def urdf_to_mjcf(urdf: str, discard_visual: bool = True, add_motors: bool = True) -> str:
    """URDF string -> MJCF string, by default with a torque motor on every non-fixed joint
    and no visual geoms (what the simulation needs). view_mujoco.py keeps the visuals and
    skips the motors."""
    urdf = _PACKAGE_URI.sub(lambda m: get_package_share_directory(m.group(1)) + "/", urdf)

    root = ET.fromstring(urdf)
    for mj in root.findall("mujoco"):
        root.remove(mj)
    effort_limit = {}
    for joint in root.findall("joint"):
        if joint.get("type") in ("revolute", "prismatic", "continuous"):
            limit = joint.find("limit")
            effort_limit[joint.get("name")] = (
                float(limit.get("effort")) if limit is not None and limit.get("effort") else 0.0
            )
    # Native URDF import knobs: keep every link a body (fusestatic off, so
    # frames such as the tool holder stay addressable), and let MuJoCo fix
    # URDF inertias that violate its triangle inequality.
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(
        mj,
        "compiler",
        balanceinertia="true",
        discardvisual="true" if discard_visual else "false",
        fusestatic="false",
    )

    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for name, effort in (effort_limit.items() if add_motors else ()):
        act = spec.add_actuator()
        act.name = name
        act.target = name
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.gear[0] = 1.0
        if effort > 0.0:
            act.ctrllimited = True
            act.ctrlrange = [-effort, effort]
    spec.compile()  # fail here (with MuJoCo's own message) rather than in the plugin
    return spec.to_xml()


class MjcfPublisher(Node):
    def __init__(self):
        super().__init__("mjcf_publisher")
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(String, "/mujoco_robot_description", latched)
        self._sub = self.create_subscription(
            String, "/robot_description", self._on_urdf, latched
        )
        self.done = False

    def _on_urdf(self, msg: String):
        if self.done:
            return
        try:
            mjcf = urdf_to_mjcf(msg.data)
        except Exception as e:  # noqa: BLE001 -- report and abort, launch must not hang silently
            self.get_logger().fatal(f"URDF -> MJCF conversion failed: {e}")
            sys.exit(1)
        self._pub.publish(String(data=mjcf))
        self.get_logger().info(f"Published MJCF ({len(mjcf)} bytes) on /mujoco_robot_description")
        self.done = True


def main():
    rclpy.init()
    node = MjcfPublisher()
    # Keep spinning after publishing: a transient_local publisher must stay
    # alive for late subscribers (the plugin comes up after this node).
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
