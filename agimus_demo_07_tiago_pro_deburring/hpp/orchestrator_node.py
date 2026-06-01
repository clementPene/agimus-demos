#!/usr/bin/env python3
"""
Entry point for the HPP deburring orchestrator — TIAGo Pro full body.

Run after sourcing ros2_config.sh, ros2_ws/install/setup.bash, and hpp_config.sh.

Usage:
    python3 orchestrator_node.py

Drops into an IPython shell with the orchestrator pre-loaded as `o`.

── Planning ──────────────────────────────────────────────────────────────────
    o.plan()                  plan HPP (approach p1, insert p2, retract p3)
    o.execute()               publish trajectory to MPC
    o.execute([o.p1])         execute approach only
    o.execute([o.p1, o.p2])   execute approach + insertion
    o.plan_and_execute()      plan then execute

── Robot state ───────────────────────────────────────────────────────────────
    o.sync_from_robot()       update q_init from live Gazebo/robot state
    o.compare_pose()          compare planned qg vs actual robot pose

── Visualisation (Viser) ────────────────────────────────────────────────────
    o.init_viewer()           open Viser viewer in browser
    o.view(q)                 show a configuration
    o.play(o.p1)              animate a path  (path, n=100, dt=0.05)

── Pylone pose ───────────────────────────────────────────────────────────────
    o.reload_pylone_pose()    reload from config/pylone_pose.yaml
    o.update_pylone_pose(t, q)  set pylone pose [x,y,z], [qx,qy,qz,qw]

── Mocap (Qualisys) ──────────────────────────────────────────────────────────
    o.connect_mocap()         connect to Qualisys (IP 140.93.1.100)
    o.disconnect_mocap()      close connection
    o.localize_pylone_from_mocap()  update pylone pose from mocap + save yaml
    o.compare_mocap()         compare FK vs mocap for EE and pylone
    o.update_mocap_frames()   refresh mocap frames in Viser viewer

── LFC (linear feedback controller) ─────────────────────────────────────────
    o.activate_lfc()          activate linear feedback controller
    o.deactivate_lfc()        deactivate linear feedback controller
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from orchestrator import Orchestrator

rclpy.init()

o = Orchestrator()

banner = (
    "\n"
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║   TIAGo Pro (full body) — HPP Deburring Orchestrator          ║\n"
    "╠════════════════════════════════════════════════════════════════╣\n"
    "║  Planning                                                      ║\n"
    "║    o.plan()                  HPP planner (p1 / p2 / p3)       ║\n"
    "║    o.execute()               publish trajectory to MPC        ║\n"
    "║    o.execute([o.p1])         approach only                    ║\n"
    "║    o.plan_and_execute()      plan then execute                ║\n"
    "╠════════════════════════════════════════════════════════════════╣\n"
    "║  Robot state                                                   ║\n"
    "║    o.sync_from_robot()       sync q_init from Gazebo/robot    ║\n"
    "║    o.compare_pose()          planned vs actual gripper pose    ║\n"
    "╠════════════════════════════════════════════════════════════════╣\n"
    "║  Visualisation                                                 ║\n"
    "║    o.init_viewer()           open Viser viewer                ║\n"
    "║    o.view(q)                 show configuration               ║\n"
    "║    o.play(o.p1)              animate a path                   ║\n"
    "╠════════════════════════════════════════════════════════════════╣\n"
    "║  Pylone                                                        ║\n"
    "║    o.reload_pylone_pose()    reload from pylone_pose.yaml     ║\n"
    "║    o.update_pylone_pose(t,q) set pylone pose manually         ║\n"
    "╠════════════════════════════════════════════════════════════════╣\n"
    "║  Mocap (Qualisys)                                              ║\n"
    "║    o.connect_mocap()         connect (IP 140.93.1.100)        ║\n"
    "║    o.localize_pylone_from_mocap()  update & save pylone pose  ║\n"
    "║    o.compare_mocap()         FK vs mocap (EE + pylone)        ║\n"
    "║    o.update_mocap_frames()   refresh frames in Viser          ║\n"
    "╠════════════════════════════════════════════════════════════════╣\n"
    "║  LFC                                                           ║\n"
    "║    o.activate_lfc()          activate linear feedback ctrl     ║\n"
    "║    o.deactivate_lfc()        deactivate                       ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n"
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
