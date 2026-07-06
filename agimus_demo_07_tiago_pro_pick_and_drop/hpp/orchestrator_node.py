#!/usr/bin/env python3
"""
Entry point for the HPP pick-and-drop orchestrator (TIAGo Pro, fixed base, left arm only).

Run after sourcing ros2_config.sh, ros2_ws/install/setup.bash, and hpp_config.sh.

Drops into an IPython shell with the orchestrator pre-loaded:

    o.plan()              # plan approach + grasp + carry + release
    o.execute()           # publish trajectory to MPC controller
    o.plan_and_execute()  # both
"""

from orchestrator import Orchestrator
import rclpy
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


rclpy.init()

o = Orchestrator()

banner = (
    "\n"
    "╔══════════════════════════════════════════════════════════╗\n"
    "║   TIAGo Pro — HPP Pick-and-Drop Orchestrator             ║\n"
    "╠══════════════════════════════════════════════════════════╣\n"
    "║  o.sync_from_robot()         — sync q_init from robot state    ║\n"
    "║  o.open_gripper()            — open the left gripper in Gazebo ║\n"
    "║  o.close_gripper()           — close the left gripper in Gazebo║\n"
    "║  o.update_object_pose(t, q)  — update obj position in q_init   ║\n"
    "║  o.update_object_pose_from_happypose() — move obj from vision  ║\n"
    "║  o.activate_lfc()            — switch to torque control        ║\n"
    "║  o.deactivate_lfc()          — switch back to position control ║\n"
    "║  o.plan()                    — run HPP planner (p1–p4)         ║\n"
    "║  o.execute()                 — publish full trajectory to MPC  ║\n"
    "║  o.execute([o.p1])           — publish approach only           ║\n"
    "║  o.execute([o.p1, o.p2])     — approach + grasp only           ║\n"
    "║  o.plan_and_execute()        — plan then execute               ║\n"
    "║  o.compare_pose()            — compare qg vs actual robot pose ║\n"
    "║  o.init_viewer()             — open Viser viewer               ║\n"
    "║  o.view(q)                   — show config in Viser            ║\n"
    "║  o.play(path)                — animate a path in Viser         ║\n"
    "╚══════════════════════════════════════════════════════════╝\n"
    "\n"
    "Phases:\n"
    "  p1 — approach : arm moves to pre-grasp pose\n"
    "  p2 — grasp    : arm closes in on object\n"
    "  p3 — carry    : arm carries object to drop zone  (None if no carry edge)\n"
    "  p4 — release  : arm retracts (reverse of p2)\n"
    "\n"
    "  ⚠  Close the physical gripper between p2 and p3.\n"
    "  ⚠  Open the physical gripper after p3 (or p2 if p3 is None).\n"
)

try:
    import IPython
    IPython.embed(banner1=banner, user_ns={"o": o, "rclpy": rclpy})
except ImportError:
    import code
    code.interact(banner=banner, local={"o": o, "rclpy": rclpy})

if o._ros_node is not None:
    o._ros_node.destroy_node()
rclpy.shutdown()
