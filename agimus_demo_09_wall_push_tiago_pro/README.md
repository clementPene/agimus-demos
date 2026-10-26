# Demo 09 — TIAGo Pro Wall Push (Fixed Base, HPP + MPC + Force Feedback)

The right arm approaches a "wall", pushes into it, and comes back, guarded by a
hard contact-force limit. The wall is the **demo-07 pylone at the same pose**
(`config/pylone_pose.yaml`), with the same handle and hole
(`config/hpp_orchestrator_params.yaml` `handle:`), so `p1`..`p4` are the ones of
[`agimus_demo_07_fixed_tiago_pro_deburring`](../agimus_demo_07_fixed_tiago_pro_deburring),
simplified: no mocap, no calibration scripts.

- **p1** approach (HPP) to the pre-grasp `qpg`
- **p2** insertion, then the push into the part (`contact:` block: `press_depth_m`, ramps, holds)
- **p3** retraction (reverse of p2), **p4** retreat (reverse of p1)

The motion is planned by an interactive **HPP orchestrator** (IPython shell, like demo 07) and
played through the MPC: the orchestrator publishes `MpcInput` messages at 100 Hz.

---

## Architecture

```
hpp/orchestrator_node.py  (IPython shell, `o`)
  o.plan()      HPP: qpg, qg, p1..p4
  o.execute()   publishes MpcInput on /mpc_input
        └─> agimus_controller (whole-body MPC, force-feedback OCP)
              └─> linear_feedback_controller (LFC)
                    └─> joint torque commands → robot
force_sensor_filter.py: raw F/T wrench -> gravity-compensated, filtered,
contact-gated Sensor.contacts, feeding agimus_controller's x0
```

**Active DOFs:** right arm only (7). Left arm, torso, wheels, grippers, head and base are locked.

**This is a guarded move, not force tracking**: the OCP (`config/ocp_definition_file.yaml`)
tracks the Cartesian reference and, once `force_sensor_filter.py` reports contact, a hard
`|f| ≤ 15 N` box constraint caps the push force.

### Key files

| File | Role |
|---|---|
| `launch/bringup.launch.py` | Robot bringup, LFC, MPC, force filter, pylone (no orchestrator unless `use_hpp_bridge:=true`) |
| `hpp/orchestrator.py` / `orchestrator_node.py` | HPP planning + MPC publishing (`o`) |
| `config/hpp_orchestrator_params.yaml` | Handle, tuck poses, MPC weights, push (`contact:`) |
| `config/pylone_pose.yaml` | Pylone pose (copied from demo 07) |
| `config/agimus_controller_params.yaml` | MPC node parameters (pylone collision pairs) |
| `config/ocp_definition_file.yaml` | OCP costs + `|f|` box constraint (copied from demo 07) |
| `scripts/force_sensor_filter.py` | F/T wrench → filtered, gated `Sensor.contacts` |
| `scripts/save_urdf.py` | Saves the robot's `/robot_description` to `urdf/tiago_pro.urdf` (read-only) |
| `urdf/tiago_pro.urdf` | The real robot's URDF, saved 2026-09-24 (offline use, MuJoCo) |
| `scripts/view_mujoco.py` | Loads the whole saved robot in MuJoCo and shows it (offline) |
| `plot/plot_force_profile.py` | Force / velocity / torque plot of a recorded run |

`config/` and `urdf/` (pylone, ground, environment) are **copies** of demo 07's: keep them in sync by
hand if demo 07's OCP/MPC settings change.

---

## Orchestrator

From the tiago_pro_nix devshell (`nix develop`), on the robot's `ROS_DOMAIN_ID`, with the bringup
launch running:

```bash
python3 hpp/orchestrator_node.py
```

| Call | What it does |
|---|---|
| `o.sync_from_robot()` | `q_init` from the robot's joint state (both arms). **Do this before `plan()`**, otherwise it plans from the yaml tuck pose |
| `o.plan()` | HPP: `qpg`, `qg`, `p1`..`p4` (offline, nothing is sent) |
| `o.init_viewer()` / `o.view(q)` / `o.play(path)` | viser on `http://localhost:8080`: robot URDF + pylone; animate a path |
| `o.activate_lfc()` / `o.deactivate_lfc()` | Switch `arm_right_controller` <-> LFC + joint_state_estimator |
| `o.execute([o.p1])` | Sample and publish to the MPC (`o.execute()` = p1..p4). `record=True` wraps it in a `ros2 bag record` |

At start-up it syncs from the robot, opens viser and plans (`p1`..`p4` are ready when the shell opens).

**Offline mode** — no robot needed: `python3 hpp/orchestrator_node.py --offline` loads the model from the saved
`urdf/tiago_pro.urdf` and plans from the yaml tuck pose. `sync_from_robot()`, `execute()`, `compare_pose()` and the
LFC switches are disabled there, so nothing can be sent to a robot by mistake. Refresh the saved URDF with
`python3 scripts/save_urdf.py` (on the robot's `ROS_DOMAIN_ID`). The saved URDF is the real hardware one: its three
`ros2_control` blocks use PAL's `robot_control/RobotControl` plugin (to be replaced for MuJoCo); all 42 distinct
meshes are STL (MuJoCo-compatible) and resolve in the devshell.

**Whole robot in MuJoCo** — `python3 scripts/view_mujoco.py` converts the saved URDF with MuJoCo's native URDF import
(73 bodies, 30 joints, 122 geoms, 85 kg, base fixed to the world, joints unactuated) and opens the MuJoCo viewer.
The model is shown kinematically (`--physics` steps it: with no actuators it sags). Useful options: `--set
arm_right_2_joint=0.9 ...`, `--collision` (also show collision meshes), `--png FILE` (headless render),
`--save-mjcf FILE` (MJCF with absolute mesh paths).

Unlike demo 07, the viewer uses `viser` + `yourdfpy` on the robot's own URDF (`pyhpp_viser` is not
available in the nix devshell). HPP itself (`pyhpp`) is, and needs `omni_base_description` 2.17.0 as pinned
in `workspace.repos` (the nix-provided 2.18.0 renamed a mesh the URDF still references).

---

## Bringup

```bash
ros2 launch agimus_demo_09_wall_push_tiago_pro bringup.launch.py robot_ip:=<ip>          # real robot
ros2 launch agimus_demo_09_wall_push_tiago_pro bringup.launch.py use_gazebo:=true         # simulation
```

| Argument | Default | Description |
|---|---|---|
| `use_hpp_bridge` | `false` | Open the orchestrator in an xterm (needs the devshell environment, inherited from this shell) |
| `use_force_feedback` | `true` | Launch `force_sensor_filter.py` and switch the OCP's `sensor` input to `sensor_with_force`. `false` disables the `|f|` box constraint (it is gated on contact detection): only for free-space tests |
| `use_mpc_debugger` | `false` | Launch the MPC debugger |

---

## Real robot

> [!CAUTION]
> Before starting, make sure the robot is in a safe position with sufficient movement space for the
> right arm. Ensure all spectators are at a safe distance and **the operator can quickly reach the
> Emergency Button** at every step below.

The set-up is the one of demo 07 (LFC deployed on the robot, network interface in
`cyclone_config.xml`, LFC/JSE config copied to `/tmp` on the robot and the dev machine, laser removed
from the URDF): see [demo 07's real-robot procedure](../agimus_demo_07_fixed_tiago_pro_deburring/README_real_robot.md),
steps 1–2.

Suggested order (the same discipline as demo 07): plan and look first, then move.

1. Bringup (`use_force_feedback:=true`), then `python3 hpp/orchestrator_node.py`.
2. `o.sync_from_robot()`, `o.plan()`, `o.init_viewer()`, `o.play(o.p1)`: check the approach in viser.
3. `o.activate_lfc()`, then `o.execute([o.p1])` alone (approach, free space).
4. Only then `o.execute([o.p2])` (insertion + push), and the full `o.execute()`.

Bags go to `plot/runs/<timestamp>[_tag]/`; `python3 plot/plot_force_profile.py plot/runs/<...>` plots them.

**Not yet run on the real robot as demo 09** (2026-09-24): this is demo 07's flow and configuration in a new
package. The orchestrator has been checked offline against the real robot's `/robot_description` and joint
state (init, `sync_from_robot()`, `plan()`, 2046 `MpcInput` messages built, viewer) but nothing has been
published.

---

## MuJoCo pipeline smoke test (work in progress)

`launch/mujoco_smoketest.launch.py` + `urdf/mujoco_smoketest/` + `config/mujoco_smoketest/` are a separate,
decoupled smoke test of MuJoCo <-> ros2_control <-> `linear_feedback_controller` <-> `agimus_controller`
(standalone PAL arm, no tool, no F/T sensor, no wall). Not the demo itself.

- **`mujoco_ros2_control` load bug: fixed** in the workspace `flake.nix`. The cause was not missing
  sources: nixpkgs' `mujoco` ships `libsimulate.a` and its headers, but the archive is built with slim LTO
  (empty `.text`), which the package's CMake links "directly" and the linker silently drops, leaving
  `mujoco::PlatformUIAdapter` undefined. The override compiles `simulate` from source instead
  (`rosOverrideAttrs.mujoco-ros2-control`).
- The plugin needs an MJCF on `/mujoco_robot_description`: `scripts/publish_mjcf.py` builds it from the live
  URDF (MuJoCo's native URDF import + one torque motor per joint).
- The LFC must be spawned `--inactive` and activated together with the joint_state_estimator in a single
  `switch_controllers` call (chained controllers), as the launch file does.
- The local LFC and `linear_feedback_controller_msgs` (branches in `workspace.repos`) must be built in the
  workspace: the nix-provided ones predate `Control.next_states`.
- **Open**: the full loop runs but the arm is unstable. It falls onto its joint limits before the LFC engages,
  then chatters. Not investigated further yet.

Use another `ROS_DOMAIN_ID` (and unset `CYCLONEDDS_URI`) for local tests: the default ones reach the real
robot over WireGuard.
