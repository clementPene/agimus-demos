#!/usr/bin/env python3
"""
Generate calibration poses for the F/T sensor regression (calibrate_ft_sensor.py).

Same collision-checking scaffolding as Figaroh's generate_optimal_configs.py
(random pool within joint limits, reject on collision, D-optimal selection
via Detmax) — but:

  1. adapted to the criterion that actually matters here: maximize the rank
     / conditioning of the *F/T regressor* (calibrate_ft_sensor.py's
     `_regressor_row`, 10 unknowns: mass, m_com(3), bias(6)) instead of
     Figaroh's kinematic-parameter regressor. Checked ahead of time (see
     project_demo07_force_feedback_scoping memory) that optimal_configs.yaml
     already gives rank 10/10, condition ~10.2 — this script targets the
     same quality with far fewer poses (10 unknowns vs Figaroh's ~38
     kinematic parameters need much less redundancy than 51 samples).
  2. accounts for the deburring tool mounted on gripper_right_tool_holder,
     which optimal_configs.yaml's collision model does NOT know about (only
     the pal-atc coupler itself has real collision geometry upstream of the
     tool_holder — see project_demo07_force_feedback_scoping memory). The
     tool is added as a simple cylinder primitive (--tool-length /
     --tool-diameter, default 40x40mm, confirmed by Clément 2026-08-20).
  3. adds a ground-plane collision check (arm/tool vs floor) that neither
     this nor the original Figaroh script had — self-collision alone
     doesn't catch "wrist swings close to the floor".

Usage:
    python3 identification/generate_ft_calibration_poses.py \\
        --urdf /path/to/tiago_pro_local.urdf \\
        --srdf /path/to/tiago_pro.srdf \\
        --n-configs 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

try:
    import hppfcl as fcl
except ImportError:
    import coal as fcl

ACTIVE_JOINTS = [
    "torso_lift_joint",
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "arm_right_5_joint",
    "arm_right_6_joint",
    "arm_right_7_joint",
]

_MEASUREMENT_FRAME = "wrist_right_ft_sensor_link"
_MOUNT_FRAME = "gripper_right_tool_holder"
_RANGE_FRACTION = 0.85
_GROUND_Z = -0.005  # top surface 5mm below base_footprint's z=0, avoids touching the base itself
_GROUND_HALF_EXTENT = 3.0  # 6x6m floor, generous
_GROUND_THICKNESS = 0.02

# geometry-object name substrings relevant to a floor check (avoid flagging
# wheels/base, which legitimately sit at/near z=0)
_GROUND_RELEVANT_SUBSTRINGS = (
    "torso", "arm_right", "wrist_right", "gripper_right", "deburring_tool",
)
# the tool's own rigid mounting chain — always "touching" by construction,
# not a real collision
_TOOL_MOUNT_CHAIN_LINKS = (
    "gripper_right_tool_holder",
    "gripper_right_tool_mount",
    "gripper_right_pal_atc_base_link",
    "wrist_right_ft_tool_link",
    "wrist_right_ft_sensor_link",
    "wrist_right_ft_sensor_base_link",
)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _regressor_row(g_local: np.ndarray) -> np.ndarray:
    """6x10 regressor block, see calibrate_ft_sensor.py for the derivation."""
    A = np.zeros((6, 10))
    A[0:3, 0] = g_local
    A[0:3, 4:7] = np.eye(3)
    A[3:6, 1:4] = -_skew(g_local)
    A[3:6, 7:10] = np.eye(3)
    return A


def _get_joint_limits(model, joint_names, fraction=_RANGE_FRACTION):
    lbs, ubs = [], []
    for jname in joint_names:
        jid = model.getJointId(jname)
        if jid >= model.njoints:
            raise ValueError(f"Joint '{jname}' not found in model.")
        idx_q = model.joints[jid].idx_q
        center = (model.upperPositionLimit[idx_q] + model.lowerPositionLimit[idx_q]) / 2
        half = (model.upperPositionLimit[idx_q] - model.lowerPositionLimit[idx_q]) / 2 * fraction
        lbs.append(center - half)
        ubs.append(center + half)
    return np.array(lbs), np.array(ubs)


def _q_from_active(model, active_vals, joint_names):
    q = pin.neutral(model)
    for jname, val in zip(joint_names, active_vals):
        jid = model.getJointId(jname)
        if jid < model.njoints and model.joints[jid].nq == 1:
            q[model.joints[jid].idx_q] = val
    return q


def _add_tool_geometry(model, collision_model, mount_frame, length, diameter):
    frame_id = model.getFrameId(mount_frame)
    if frame_id >= len(model.frames):
        raise SystemExit(f"Mount frame '{mount_frame}' not found in URDF.")
    frame = model.frames[frame_id]
    # tool extends along +z from the mount frame, centered at half its length
    local_offset = pin.SE3(np.eye(3), np.array([0.0, 0.0, length / 2.0]))
    placement = frame.placement * local_offset
    geom = pin.GeometryObject(
        "deburring_tool",
        frame.parentJoint,
        placement,
        fcl.Cylinder(diameter / 2.0, length),
    )
    idx = collision_model.addGeometryObject(geom)
    print(
        f"Added tool cylinder (Ø{diameter*1000:.0f}mm x {length*1000:.0f}mm) "
        f"rigidly mounted on '{mount_frame}' (geometry id {idx})."
    )
    return idx


def _add_ground_geometry(collision_model):
    placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, _GROUND_Z - _GROUND_THICKNESS / 2.0]))
    geom = pin.GeometryObject(
        "ground_plane",
        0,  # universe joint
        placement,
        fcl.Box(
            2 * _GROUND_HALF_EXTENT, 2 * _GROUND_HALF_EXTENT, _GROUND_THICKNESS
        ),
    )
    idx = collision_model.addGeometryObject(geom)
    print(f"Added ground plane at z={_GROUND_Z*1000:.0f}mm (geometry id {idx}).")
    return idx


def _setup_collisions(model, collision_model, srdf_text, tool_length, tool_diameter):
    tool_idx = _add_tool_geometry(model, collision_model, _MOUNT_FRAME, tool_length, tool_diameter)
    ground_idx = _add_ground_geometry(collision_model)

    collision_model.addAllCollisionPairs()
    pin.removeCollisionPairsFromXML(model, collision_model, srdf_text, verbose=False)
    print(f"Collision pairs after SRDF removal: {len(collision_model.collisionPairs)}")

    names = [go.name for go in collision_model.geometryObjects]

    def _remove_pair(i, j):
        pair = pin.CollisionPair(i, j)
        if collision_model.existCollisionPair(pair):
            collision_model.removeCollisionPair(pair)

    # tool vs its own rigid mounting chain: always touching by construction
    for link_name in _TOOL_MOUNT_CHAIN_LINKS:
        for i, name in enumerate(names):
            if link_name in name:
                _remove_pair(tool_idx, i)

    # ground vs everything except the right-arm chain + tool (avoid wheel/base noise)
    for i, name in enumerate(names):
        if i in (tool_idx, ground_idx):
            continue
        if not any(s in name for s in _GROUND_RELEVANT_SUBSTRINGS):
            _remove_pair(ground_idx, i)

    print(f"Collision pairs after tool-mount + ground pruning: {len(collision_model.collisionPairs)}")
    return tool_idx, ground_idx


def _run_detmax(sub_mats, n_choose, seed=0):
    import random

    rng = random.Random(seed)
    pool = list(range(len(sub_mats)))

    def crit(indices):
        M = sum(sub_mats[i] for i in indices)
        try:
            d = float(np.linalg.det(M))
            return d ** (1 / M.shape[0]) if d > 0 else 0.0
        except Exception:
            return 0.0

    cur = rng.sample(pool, n_choose)
    remaining = [i for i in pool if i not in cur]

    for _ in range(200):
        best_add, best_crit = None, crit(cur)
        for k in remaining:
            c = crit(cur + [k])
            if c > best_crit:
                best_crit, best_add = c, k
        if best_add is None:
            break
        cur.append(best_add)
        remaining.remove(best_add)

        worst_rm, worst_crit = None, float("inf")
        for j in cur:
            c = crit([x for x in cur if x != j])
            if c < worst_crit:
                worst_crit, worst_rm = c, j
        if worst_rm is not None and worst_rm != best_add:
            cur.remove(worst_rm)
            remaining.append(worst_rm)
        else:
            break

    return sorted(cur)


def _pkg_dirs(robot_description_dir):
    """Same two-level package_dirs expansion as Figaroh's
    generate_optimal_configs.py — needed for pinocchio to resolve
    package://<pkg>/... mesh URIs nested under robot_description/."""
    root = Path(robot_description_dir)
    dirs = [str(root)]
    if not root.is_dir():
        return dirs
    for p in root.iterdir():
        if p.is_dir():
            dirs.append(str(p))
            for sub in p.iterdir():
                if sub.is_dir():
                    dirs.append(str(sub))
    return dirs


def _run_viser_review(
    model, visual_model, collision_model, data,
    pool_active, chosen, tool_idx, ground_idx,
    tool_length, tool_diameter, active_joints,
):
    """Step through the selected poses in Viser: robot from its real visual
    meshes, the deburring tool and floor as simple primitives (they have no
    URDF meshes — same geometry used for the collision check)."""
    import trimesh
    import viser
    import viser.transforms as vtf

    def _load_mesh(path):
        try:
            mesh = trimesh.load(path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            return mesh
        except Exception:
            return None

    print("\nStarting Viser at http://localhost:8080 ...")
    print("Controls: Enter -> next pose | q -> quit")
    server = viser.ViserServer()
    time.sleep(0.5)

    visual_data = pin.GeometryData(visual_model)
    collision_data = pin.GeometryData(collision_model)
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateGeometryPlacements(model, data, visual_model, visual_data)

    print("Loading robot meshes into Viser ...")
    mesh_handles = {}
    for i, geom_obj in enumerate(visual_model.geometryObjects):
        mesh = _load_mesh(geom_obj.meshPath)
        if mesh is None:
            continue
        T = visual_data.oMg[i]
        vertices = np.array(mesh.vertices, dtype=np.float32) * geom_obj.meshScale
        faces = np.array(mesh.faces, dtype=np.uint32)
        handle = server.scene.add_mesh_simple(
            f"robot/{geom_obj.name}",
            vertices=vertices,
            faces=faces,
            position=T.translation,
            wxyz=vtf.SO3.from_matrix(T.rotation).wxyz,
            color=(0.8, 0.8, 0.8),
        )
        mesh_handles[i] = handle
    print(f"Loaded {len(mesh_handles)} mesh objects.")

    # deburring tool — dynamic, same cylinder used for collision checking
    tool_mesh = trimesh.creation.cylinder(
        radius=tool_diameter / 2.0, height=tool_length, sections=24
    )
    tool_handle = server.scene.add_mesh_simple(
        "tool/deburring_tool",
        vertices=np.array(tool_mesh.vertices, dtype=np.float32),
        faces=np.array(tool_mesh.faces, dtype=np.uint32),
        color=(0.9, 0.35, 0.1),
    )

    # floor — static, same slab used for collision checking
    ground_placement = collision_model.geometryObjects[ground_idx].placement
    ground_mesh = trimesh.creation.box(
        extents=[2 * _GROUND_HALF_EXTENT, 2 * _GROUND_HALF_EXTENT, _GROUND_THICKNESS]
    )
    server.scene.add_mesh_simple(
        "ground/floor",
        vertices=np.array(ground_mesh.vertices, dtype=np.float32),
        faces=np.array(ground_mesh.faces, dtype=np.uint32),
        position=ground_placement.translation,
        wxyz=vtf.SO3.from_matrix(ground_placement.rotation).wxyz,
        color=(0.55, 0.55, 0.6),
        opacity=0.35,
    )

    for idx, pool_idx in enumerate(chosen):
        active_vals = pool_active[pool_idx]
        q = _q_from_active(model, active_vals, active_joints)
        pin.forwardKinematics(model, data, q)
        pin.updateGeometryPlacements(model, data, visual_model, visual_data)
        pin.updateGeometryPlacements(model, data, collision_model, collision_data)

        for i, handle in mesh_handles.items():
            T = visual_data.oMg[i]
            handle.position = T.translation
            handle.wxyz = vtf.SO3.from_matrix(T.rotation).wxyz

        T_tool = collision_data.oMg[tool_idx]
        tool_handle.position = T_tool.translation
        tool_handle.wxyz = vtf.SO3.from_matrix(T_tool.rotation).wxyz

        print(f"\n{'─' * 50}")
        print(f"Pose {idx + 1}/{len(chosen)}")
        for jname, val in zip(active_joints, active_vals):
            print(f"  {jname:<30} {np.degrees(val):+7.2f}°")
        print("Press Enter for next pose, 'q' to quit.")
        line = input().strip().lower()
        if line == "q":
            break

    print("\nViser review done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--srdf", required=True)
    parser.add_argument(
        "--robot-description-dir",
        default=None,
        help="Directory containing the mesh packages (default: 'robot_description' "
        "next to --urdf, matching the figaroh_tiagoPro layout).",
    )
    parser.add_argument("--pool-size", type=int, default=3000)
    parser.add_argument("--n-configs", type=int, default=20)
    parser.add_argument("--tool-length", type=float, default=0.04, help="meters (default 40mm)")
    parser.add_argument("--tool-diameter", type=float, default=0.04, help="meters (default 40mm)")
    parser.add_argument("--gravity", nargs=3, type=float, default=[0.0, 0.0, 9.81])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", "-o",
        default=str(Path(__file__).parent / "ft_calibration_poses.yaml"),
    )
    parser.add_argument(
        "--no-viser", action="store_true",
        help="Skip the interactive Viser review of the selected poses.",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.urdf} ...")
    model = pin.buildModelFromUrdf(args.urdf)
    rd_dir = args.robot_description_dir or str(Path(args.urdf).parent / "robot_description")
    pkg_dirs = _pkg_dirs(rd_dir)
    try:
        _, collision_model, visual_model = pin.buildModelsFromUrdf(args.urdf, package_dirs=pkg_dirs)
    except ValueError as e:
        raise SystemExit(
            f"Geometry loading failed ({e}) — need real collision meshes to proceed "
            f"(tried package_dirs under {rd_dir}; pass --robot-description-dir if it "
            "lives elsewhere)."
        )
    data = model.createData()

    srdf_text = Path(args.srdf).read_text()
    tool_idx, ground_idx = _setup_collisions(
        model, collision_model, srdf_text, args.tool_length, args.tool_diameter
    )
    collision_data = pin.GeometryData(collision_model)

    frame_id = model.getFrameId(_MEASUREMENT_FRAME)
    if frame_id >= len(model.frames):
        raise SystemExit(f"Measurement frame '{_MEASUREMENT_FRAME}' not found.")

    lbs, ubs = _get_joint_limits(model, ACTIVE_JOINTS)
    rng = np.random.default_rng(args.seed)
    g_world = np.array(args.gravity)

    print(f"\nSampling {args.pool_size} collision-free candidates "
          f"(self + tool@{args.tool_diameter*1000:.0f}x{args.tool_length*1000:.0f}mm + ground) ...")
    pool_active, pool_A = [], []
    n_tried = 0
    n_ground_hits = 0
    while len(pool_active) < args.pool_size:
        n_tried += 1
        active_vals = rng.uniform(lbs, ubs)
        q = _q_from_active(model, active_vals, ACTIVE_JOINTS)
        pin.computeCollisions(model, model.createData(), collision_model, collision_data, q, False)
        results = collision_data.collisionResults
        if any(r.isCollision() for r in results):
            if any(
                r.isCollision()
                and ground_idx in (collision_model.collisionPairs[i].first, collision_model.collisionPairs[i].second)
                for i, r in enumerate(results)
            ):
                n_ground_hits += 1
            continue
        pin.forwardKinematics(model, data, q)
        T = pin.updateFramePlacement(model, data, frame_id)
        g_local = T.rotation.T @ g_world
        pool_active.append(active_vals)
        pool_A.append(_regressor_row(g_local))
        if n_tried % 2000 == 0:
            print(f"  ... {len(pool_active)}/{args.pool_size} valid, {n_tried} tried "
                  f"({n_ground_hits} rejected for ground contact)")

    print(f"Found {args.pool_size} valid configs out of {n_tried} tried "
          f"({n_ground_hits} rejected for ground contact).")

    sub_mats = [A.T @ A for A in pool_A]
    n_choose = max(args.n_configs, 12)  # >=12 for margin over the 10 unknowns
    print(f"\nSelecting {n_choose} D-optimal configs (criterion: F/T regressor, 10 params) ...")
    chosen = _run_detmax(sub_mats, n_choose, seed=args.seed)

    A_final = np.vstack([pool_A[i] for i in chosen])
    sv = np.linalg.svd(A_final, compute_uv=False)
    rank = np.linalg.matrix_rank(A_final)
    print(f"\nSelected {len(chosen)} configs — regressor rank {rank}/10, "
          f"condition number {sv[0]/sv[-1]:.2f}")

    output_data = {
        "calibration_joint_names": ACTIVE_JOINTS,
        "calibration_joint_configurations": [pool_active[i].tolist() for i in chosen],
        "tool_length_m": args.tool_length,
        "tool_diameter_m": args.tool_diameter,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(output_data, f, sort_keys=False, default_flow_style=False)
    print(f"Saved to {args.output}")

    if not args.no_viser:
        try:
            _run_viser_review(
                model, visual_model, collision_model, data,
                pool_active, chosen, tool_idx, ground_idx,
                args.tool_length, args.tool_diameter, ACTIVE_JOINTS,
            )
        except ImportError as e:
            print(f"\nViser/trimesh not available ({e}) — skipping visual review.")


if __name__ == "__main__":
    main()
