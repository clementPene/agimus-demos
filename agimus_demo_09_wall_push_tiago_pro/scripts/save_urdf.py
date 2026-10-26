#!/usr/bin/env python3
"""
Save the robot's URDF from the /robot_description topic to a file, so the demo can
run offline (orchestrator_node.py --offline) and be reused elsewhere (MuJoCo...).

Read-only: it only subscribes to /robot_description. Run it with the robot's
ROS_DOMAIN_ID, from the tiago_pro_nix devshell.

Usage:
    python3 scripts/save_urdf.py
    python3 scripts/save_urdf.py --output /path/to/tiago_pro.urdf
"""

import argparse
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

_DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "urdf" / "tiago_pro.urdf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default=str(_DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=float, default=20.0, help="seconds to wait for the topic")
    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = Node("urdf_saver")
    received = []
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    node.create_subscription(String, "/robot_description", received.append, qos)

    node.get_logger().info("Waiting for /robot_description ...")
    deadline = time.time() + parsed.timeout
    while not received and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not received:
        node.get_logger().error(f"No /robot_description after {parsed.timeout}s -- right ROS_DOMAIN_ID / network?")
        return 1

    out = Path(parsed.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(received[0].data)
    node.get_logger().info(f"URDF saved to {out} ({len(received[0].data)} bytes)")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
