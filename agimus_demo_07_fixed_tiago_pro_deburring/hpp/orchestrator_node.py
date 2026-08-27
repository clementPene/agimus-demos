#!/usr/bin/env python3
"""
Entry point for the HPP deburring orchestrator.

Run after sourcing ros2_config.sh, ros2_ws/install/setup.bash, and hpp_config.sh.

Usage:
    python3 orchestrator_node.py

Drops into an IPython shell with the orchestrator pre-loaded:

    o.plan()              # plan with HPP (approach + insert + retract)
    o.execute()           # publish trajectory to MPC controller
    o.plan_and_execute()  # both

Recording & force plots
    o.execute(record=True)          # wrap the run in a `ros2 bag record`
    o.execute([o.p2], record="nowall")   # tag it; p2 also appends the guarded-move press
    o.record = True                 # ...or auto-record every execute() this session

    Bags land in  <pkg>/plot/runs/<timestamp>[_tag]/  (RECORD_TOPICS in
    orchestrator.py: /sensor_with_force, /mpc_input, /control, /ocp_x0,
    /ocp_solve_time, /mpc_debug). execute() prints the ready-to-run plot
    command when it finishes:

        python3 plot/plot_force_profile.py plot/runs/<timestamp>[_tag]

    -> 3-panel figure (f_z measured vs target / |dq| / |u|), the real-robot
    analogue of tiago_pro_force_mpc_sim/closed_loop_mujoco.py.
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
    "╔════════════════════════════════════════════════════════════════════╗\n"
    "║   TIAGo Pro — HPP Deburring Orchestrator                          ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Setup                                                            ║\n"
    "║    o.sync_from_robot()            — sync q_init from robot state  ║\n"
    "║    o.reload_pylone_pose()         — reload pylone pose from yaml  ║\n"
    "║    o.activate_lfc()              — switch to torque control       ║\n"
    "║    o.deactivate_lfc()            — switch back to pos. control    ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Mocap (Qualisys)                                                 ║\n"
    "║    o.connect_mocap()             — connect to Qualisys server     ║\n"
    "║    o.disconnect_mocap()          — stop mocap subprocess          ║\n"
    "║    o.localize_pylone_from_mocap() — set pylone pose from mocap    ║\n"
    "║    o.compare_mocap()             — mocap vs robot FK (EE+pylone)  ║\n"
    "║    o.update_mocap_frames()       — live mocap/ee + mocap/pylone   ║\n"
    "║                                    frames in Viser                ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Planning & execution                                             ║\n"
    "║    o.plan()                      — run HPP planner                ║\n"
    "║    o.execute()                   — publish full trajectory to MPC ║\n"
    "║    o.execute([o.p1])             — publish p1 only (approach)     ║\n"
    "║    o.execute([o.p2])             — p2 + its guarded-move press    ║\n"
    "║    o.plan_and_execute()          — plan then execute              ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Recording & force plots                                          ║\n"
    "║    o.execute(record=True)        — wrap run in `ros2 bag record`  ║\n"
    "║    o.execute([o.p2], record='x') — tag the bag 'x'                ║\n"
    "║    o.record = True               — auto-record every execute()    ║\n"
    "║    bags -> <pkg>/plot/runs/<timestamp>[_tag]/                     ║\n"
    "║    then: python3 plot/plot_force_profile.py plot/runs/<...>       ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Diagnostics                                                      ║\n"
    "║    o.compare_pose()              — qg vs actual robot FK          ║\n"
    "║    o.compare_pose(o.p1)          — end of p1 vs actual            ║\n"
    "║    o.init_viewer()               — open Viser viewer              ║\n"
    "║    o.view(q)                     — show config in Viser           ║\n"
    "║    o.play(path)                  — animate a path in Viser        ║\n"
    "╚════════════════════════════════════════════════════════════════════╝\n"
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
