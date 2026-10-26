#!/usr/bin/env python3
"""
Load the WHOLE TIAGo Pro (the saved real-robot URDF, urdf/tiago_pro.urdf) in MuJoCo and
look at it. Offline: no ROS, no robot.

MuJoCo reads the URDF natively (all meshes are STL). Every link becomes a body, the
base (base_footprint) is fixed to the world, the 30 movable joints are unactuated.

By default the model is shown KINEMATICALLY (mj_forward only, no time stepping): with no
actuators the whole robot would otherwise sag under gravity. --physics steps the
simulation to see exactly that.

    python3 scripts/view_mujoco.py                     # interactive viewer, visual meshes
    python3 scripts/view_mujoco.py --collision         # also show the collision meshes
    python3 scripts/view_mujoco.py --physics           # step the physics (the arms will fall)
    python3 scripts/view_mujoco.py --png /tmp/r.png    # render one image, no window
    python3 scripts/view_mujoco.py --save-mjcf /tmp/tiago_pro.xml   # keep the MJCF (absolute mesh paths)

Joint positions: --set arm_right_1_joint=0.5 ... (radians / metres); default all zero.
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from publish_mjcf import urdf_to_mjcf  # noqa: E402

_DEFAULT_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "urdf", "tiago_pro.urdf")


def build_model(urdf_path: str):
    with open(urdf_path) as f:
        mjcf = urdf_to_mjcf(f.read(), discard_visual=False, add_motors=False)
    return mjcf, mujoco.MjModel.from_xml_string(mjcf)


def set_joints(model, data, assignments):
    for a in assignments or []:
        name, _, val = a.partition("=")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit(f"unknown joint '{name}'")
        data.qpos[model.jnt_qposadr[jid]] = float(val)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=_DEFAULT_URDF)
    ap.add_argument("--collision", action="store_true", help="also show the collision geoms (group 0)")
    ap.add_argument("--physics", action="store_true", help="step the simulation (unactuated: it sags)")
    ap.add_argument("--set", nargs="*", metavar="JOINT=VALUE", help="initial joint positions")
    ap.add_argument("--png", metavar="FILE", help="render one image and exit (no window)")
    ap.add_argument("--save-mjcf", metavar="FILE", help="write the generated MJCF")
    args = ap.parse_args()

    mjcf, model = build_model(os.path.abspath(args.urdf))
    if args.save_mjcf:
        with open(args.save_mjcf, "w") as f:
            f.write(mjcf)
        print(f"MJCF written to {args.save_mjcf}")
    data = mujoco.MjData(model)
    set_joints(model, data, args.set)
    mujoco.mj_forward(model, data)
    print(f"Loaded: {model.nbody} bodies, {model.njnt} joints, {model.ngeom} geoms, "
          f"{model.nmesh} meshes, {model.body_mass.sum():.1f} kg")

    # Visual meshes are group 1, collision meshes group 0 (MuJoCo's URDF importer).
    show_groups = [1] + ([0] if args.collision else [])
    center = data.subtree_com[1].copy()

    if args.png:
        os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = mujoco.Renderer(model, height=480, width=640)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for g in show_groups:
            opt.geomgroup[g] = 1
        cam = mujoco.MjvCamera()
        cam.lookat[:] = center
        cam.distance, cam.azimuth, cam.elevation = 3.2, 145, -15
        renderer.update_scene(data, camera=cam, scene_option=opt)
        import PIL.Image

        PIL.Image.fromarray(renderer.render()).save(args.png)
        print(f"Image written to {args.png}")
        return 0

    from mujoco import viewer as mjviewer

    with mjviewer.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[:] = 0
        for g in show_groups:
            viewer.opt.geomgroup[g] = 1
        viewer.cam.lookat[:] = center
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 145, -15
        print("Viewer open. Close the window (or Ctrl-C) to quit.")
        while viewer.is_running():
            if args.physics:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep if args.physics else 0.03)
    return 0


if __name__ == "__main__":
    sys.exit(main())
