#!/usr/bin/env python3
"""
Entry point for the HPP pick-and-drop orchestrator (TIAGo Pro, fixed base, left arm only).

Run after sourcing ros2_config.sh, ros2_ws/install/setup.bash, and hpp_config.sh.

Drops into an IPython shell with the orchestrator pre-loaded:

    o.plan_pick()         # plan approach + grasp + carry (p1-p3)
    o.execute_pick()      # execute p1-p3: gripper close, grasp check, carry
    o.execute_place()     # navigate to drop zone, add box, plan_place(),
                           # execute p_place/p4/p5, navigate back, remove box
    o.execute()           # execute_pick() then execute_place()
    o.plan_place()        # plan place/release on its own (normally called
                           # by execute_place() itself, right after navigation)
    o.add_box_to_scene()  # manually add the drop-zone box to the HPP scene
    o.remove_box_from_scene()  # manually remove it again
    o.plan_and_execute()  # plan_pick() then execute()
    o.tuck_arm()          # recovery: send arm to tuck pose after a failure
"""

import rclpy
import sys
import os

# Defensive fallback: Python normally puts the running script's own directory
# on sys.path automatically, which is what lets the bare `import orchestrator`
# below succeed. Adding it explicitly here covers run contexts (e.g. some
# `ros2 run` wrappers) where that auto-add doesn't happen.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import Orchestrator

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
    "║  o.enforce_grasp_check = False — skip grasp verify (sim)       ║\n"
    "║  o.enforce_navigation = False — fake navigation (sim)          ║\n"
    "║  o.update_object_pose(t, q)  — update obj position in q_init   ║\n"
    "║  o.update_object_pose_from_happypose() — move obj from vision  ║\n"
    "║  o.activate_lfc()            — switch to torque control        ║\n"
    "║  o.deactivate_lfc()          — switch back to position control ║\n"
    "║  o.plan_pick()                — run HPP planner (p1–p3)        ║\n"
    "║  o.execute_pick()            — execute p1–p3 (gripper + carry) ║\n"
    "║  o.execute_place()           — nav + box + plan_place() + run  ║\n"
    "║                                 p_place/p4/p5, nav back         ║\n"
    "║  o.execute()                 — execute_pick() + execute_place()║\n"
    "║  o.execute([o.p1])           — publish approach only           ║\n"
    "║  o.execute([o.p1, o.p2])     — approach + grasp only           ║\n"
    "║  o.plan_and_execute()        — plan_pick() then execute()      ║\n"
    "║  o.navigate_to_drop_zone_and_add_box() — nav + switch scene    ║\n"
    "║  o.plan_place()              — plan p_place/p4/p5 on its own   ║\n"
    "║  o.add_box_to_scene()        — manually add box to HPP scene   ║\n"
    "║  o.remove_box_from_scene()   — manually remove box from scene  ║\n"
    "║  o.navigate_to_initial_pose()— send base back to initial point ║\n"
    "║  o.tuck_arm()                — recovery: send arm to tuck pose ║\n"
    "║  o.compare_pose()            — compare qg vs actual robot pose ║\n"
    "║  o.init_viewer()             — open Viser viewer               ║\n"
    "║  o.view(q)                   — show config in Viser            ║\n"
    "║  o.play(path)                — animate a path in Viser         ║\n"
    "╚══════════════════════════════════════════════════════════╝\n"
    "\n"
    "Phases:\n"
    "  p1  — approach : arm moves to pre-grasp pose\n"
    "  p2  — grasp    : arm closes in on object\n"
    "  p2b — retract  : pulls back to the grasped handle's own clearance\n"
    "                   distance before the big move to the carry pose\n"
    "  p3  — carry    : arm moves to the transport pose, object grasped\n"
    "                   (None if no carry edge)\n"
    "  p_place        : arm moves from the transport pose to the drop zone —\n"
    "                   planned by execute_place() itself, after the base has\n"
    "                   navigated there and the drop-zone box has been\n"
    "                   added to the HPP scene (None right after plan_pick())\n"
    "  p4  — release  : arm retreats from the drop zone (reverse of p2 if\n"
    "                   there was no carry at all — planned by execute_place()\n"
    "                   alongside p_place otherwise)\n"
    "  p5  — return   : arm returns to carry pose, empty-handed\n"
    "  → navigate back to initial point (after p5) — ready for a new cycle\n"
    "\n"
    "  ⚠  Close the physical gripper between p2 and p3.\n"
    "  ⚠  Open the physical gripper after p3 (or p2 if p3 is None).\n"
    "  ⚠  Sequence aborted or arm in an unknown pose? Call o.tuck_arm() to\n"
    "     reset it to the tuck configuration before replanning.\n"
)

# Prefer IPython for a nicer interactive shell (tab-completion, history);
# fall back to the stdlib `code` module if it isn't installed.
try:
    import IPython
    IPython.embed(banner1=banner, user_ns={"o": o, "rclpy": rclpy})
except ImportError:
    import code
    code.interact(banner=banner, local={"o": o, "rclpy": rclpy})

# o._ros_node is only created lazily (by plan_pick()/execute()/etc.), so it may
# still be None here if the user never triggered ROS communication.
if o._ros_node is not None:
    o._ros_node.destroy_node()
rclpy.shutdown()
