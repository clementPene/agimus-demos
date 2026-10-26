#!/usr/bin/env python3
"""
Entry point for the wall-push demo HPP orchestrator.

Run from the tiago_pro_nix devshell (same ROS_DOMAIN_ID as the robot), after the
bringup launch is up:

    python3 hpp/orchestrator_node.py                 # on the robot
    python3 hpp/orchestrator_node.py --offline       # no robot: model from urdf/tiago_pro.urdf

At start-up it does, in this order: sync from the robot, open viser
(http://localhost:8080), plan (HPP). Then it drops into an IPython shell with the
orchestrator pre-loaded as `o`, with p1..p4 ready:

    o.play(o.p1)          # animate a path in viser
    o.activate_lfc()      # switch to torque control
    o.execute([o.p1])     # publish to the MPC; o.execute() = p1..p4
    o.sync_from_robot(); o.plan()   # redo after the robot moved
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from orchestrator import Orchestrator

_DEFAULT_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "urdf", "tiago_pro.urdf")

_parser = argparse.ArgumentParser(description="Wall-push HPP orchestrator")
_parser.add_argument(
    "--offline",
    nargs="?",
    const=_DEFAULT_URDF,
    metavar="URDF",
    help="no robot: load the model from a saved URDF (default urdf/tiago_pro.urdf, see "
    "scripts/save_urdf.py). sync, execute and LFC switching are disabled; plan() starts "
    "from the yaml tuck pose.",
)
_args, _ros_args = _parser.parse_known_args()

rclpy.init(args=_ros_args)

o = Orchestrator(urdf_file=os.path.abspath(_args.offline) if _args.offline else None)


def _startup(o) -> None:
    """sync -> viser -> plan. Each step is reported but never stops the shell from opening."""
    synced = False
    if _args.offline:
        print("\nOFFLINE: no sync -- planning from the yaml tuck pose.")
        synced = True  # plan from q_init as it is
    else:
        try:
            o.sync_from_robot()
            synced = True
        except Exception as e:  # noqa: BLE001
            print(f"\n!! sync_from_robot failed: {e}")
    try:
        o.init_viewer()
    except Exception as e:  # noqa: BLE001
        print(f"\n!! init_viewer failed: {e}")
    if not synced:
        print("\n!! plan() skipped: q_init was not synced from the robot (it would plan from the yaml tuck pose).")
        print("   Fix the connection, then: o.sync_from_robot(); o.plan()")
        return
    try:
        if not o.plan():
            print("\n!! plan() found no solution.")
    except Exception as e:  # noqa: BLE001
        print(f"\n!! plan() failed: {e}")


_startup(o)

banner = (
    "\n"
    "╔════════════════════════════════════════════════════════════════════╗\n"
    "║   TIAGo Pro — Wall-push HPP Orchestrator                          ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Done at start-up: sync_from_robot, init_viewer, plan             ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Setup                                                            ║\n"
    "║    o.sync_from_robot()            — re-sync q_init (then re-plan) ║\n"
    "║    o.reload_pylone_pose()         — reload pylone pose from yaml  ║\n"
    "║    o.activate_lfc()               — switch to torque control      ║\n"
    "║    o.deactivate_lfc()             — switch back to pos. control   ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Planning & execution                                             ║\n"
    "║    o.plan()                       — run HPP planner               ║\n"
    "║    o.execute()                    — publish p1..p4 to the MPC     ║\n"
    "║    o.execute([o.p1])              — p1 only (approach)            ║\n"
    "║    o.execute([o.p2])              — p2 + its push into the part   ║\n"
    "║    o.plan_and_execute()           — plan then execute             ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Recording & force plots                                          ║\n"
    "║    o.execute(record=True)         — wrap run in `ros2 bag record` ║\n"
    "║    o.execute([o.p2], record='x')  — tag the bag 'x'               ║\n"
    "║    bags -> <pkg>/plot/runs/<timestamp>[_tag]/                     ║\n"
    "║    then: python3 plot/plot_force_profile.py plot/runs/<...>       ║\n"
    "╠════════════════════════════════════════════════════════════════════╣\n"
    "║  Diagnostics                                                      ║\n"
    "║    o.compare_pose()               — qg vs actual robot FK         ║\n"
    "║    o.init_viewer()                — open viser viewer             ║\n"
    "║    o.view(q)                      — show config in viser          ║\n"
    "║    o.play(path)                   — animate a path in viser       ║\n"
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
