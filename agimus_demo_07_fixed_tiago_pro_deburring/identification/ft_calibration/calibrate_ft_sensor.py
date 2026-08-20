#!/usr/bin/env python3
"""
Regression step for the F/T sensor calibration — companion to
collect_ft_calibration_data.py.

Identifies the pal-atc tool's mass and center of mass, plus the F/T
sensor's zero-offset bias, from the (joint state, raw wrench) samples
collected at N static poses. Same family of method as Figaroh's kinematic
calibration (`run_calibration.py`): multi-pose static regression, model
prediction vs measurement. The parameterization here happens to be fully
LINEAR though, unlike Figaroh's SE3 log-map residual, so a plain
`numpy.linalg.lstsq` is enough — no iterative nonlinear solver, no local
minima to fight.

Why it's linear: naively, torque = com_xyz × (mass · g_local) is bilinear
in (mass, com_xyz) — a product of two unknowns. The standard trick (same
one used for rigid-body inertial parameter identification in general) is
to identify the *first moment of mass* m_com = mass · com_xyz as a single
compound unknown instead of com_xyz directly:

    f_lin(i) = mass · g_local(i)                    + bias_lin
    f_ang(i) = m_com × g_local(i)                    + bias_ang
             = -skew(g_local(i)) · m_com             + bias_ang

Stacked over all 6*N equations this is linear in
x = [mass, m_com_x, m_com_y, m_com_z, bias_lin(3), bias_ang(3)] (10 unknowns).
com_xyz is recovered afterwards as m_com / mass.

Note on `bias`: this fitted bias captures whatever the gravity model above
does not explain (sensor zero-offset, but also any residual model error,
e.g. from the calibration_measurement_frame rotation offset which this
script does NOT identify — left at zero, see module docstring in
force_sensor_filter.py). force_sensor_filter.py re-estimates a fresh bias
at every startup anyway (bias_measurement_samples), so this fitted bias is
mainly useful as a sanity check, not something to hardcode.

⚠️ Run collect_ft_calibration_data.py AFTER the Figaroh kinematic
calibration (or with an already-decent URDF) — g_local at each pose comes
from FK, so kinematic error leaks into the mass/CoM estimate.

Usage:
    python3 identification/calibrate_ft_sensor.py \\
        --urdf /path/to/tiago_pro_local.urdf \\
        --data identification/ft_calibration_samples.csv
"""

import argparse

import numpy as np
import pandas as pd
import pinocchio as pin


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def _q_from_row(model, row, active_joints):
    q = pin.neutral(model)
    for jname in active_joints:
        val = float(row[jname])
        for candidate in (jname, f"tiago_pro/{jname}"):
            try:
                jid = model.getJointId(candidate)
                if jid < model.njoints and model.joints[jid].nq == 1:
                    q[model.joints[jid].idx_q] = val
                    break
            except Exception:
                pass
    return q


def _regressor_row(g_local: np.ndarray) -> np.ndarray:
    """6x10 regressor block for one sample, columns:
    [mass, m_com_x, m_com_y, m_com_z, bias_fx, bias_fy, bias_fz, bias_tx, bias_ty, bias_tz]
    """
    A = np.zeros((6, 10))
    A[0:3, 0] = g_local
    A[0:3, 4:7] = np.eye(3)
    A[3:6, 1:4] = -_skew(g_local)
    A[3:6, 7:10] = np.eye(3)
    return A


def main():
    parser = argparse.ArgumentParser(description="F/T sensor mass/CoM/bias regression.")
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--measurement-frame", default="wrist_right_ft_sensor_link")
    parser.add_argument(
        "--gravity", nargs=3, type=float, default=[0.0, 0.0, 9.81], metavar=("GX", "GY", "GZ")
    )
    parser.add_argument(
        "--threshold-margin",
        type=float,
        default=4.0,
        help="Suggested contact threshold = margin x residual force std norm (default: 4).",
    )
    args = parser.parse_args()

    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()
    frame_id = model.getFrameId(args.measurement_frame)
    if frame_id >= len(model.frames):
        raise SystemExit(f"Frame '{args.measurement_frame}' not found in {args.urdf}")

    df = pd.read_csv(args.data)
    active_joints = [c for c in df.columns if c not in ("fx", "fy", "fz", "tx", "ty", "tz")]
    print(f"Loaded {len(df)} samples, {len(active_joints)} active joints.")

    g_world = np.array(args.gravity)
    A_rows, b_rows = [], []
    for _, row in df.iterrows():
        q = _q_from_row(model, row, active_joints)
        pin.forwardKinematics(model, data, q)
        T_frame = pin.updateFramePlacement(model, data, frame_id)
        g_local = T_frame.rotation.T @ g_world
        A_rows.append(_regressor_row(g_local))
        b_rows.append(
            np.array([row.fx, row.fy, row.fz, row.tx, row.ty, row.tz])
        )

    A = np.vstack(A_rows)  # (6N, 10)
    b = np.concatenate(b_rows)  # (6N,)

    x, residuals_ss, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    mass = x[0]
    m_com = x[1:4]
    com_xyz = m_com / mass
    bias = x[4:10]

    pred = A @ x
    resid = (b - pred).reshape(-1, 6)
    force_rmse = np.sqrt(np.mean(resid[:, :3] ** 2, axis=0))
    torque_rmse = np.sqrt(np.mean(resid[:, 3:] ** 2, axis=0))
    force_resid_norms = np.linalg.norm(resid[:, :3], axis=1)

    print(f"\nRank of regressor: {rank}/10", "(deficient — add more/more varied poses)" if rank < 10 else "(full rank, good)")
    print(f"\nmass            = {mass:.5f} kg")
    print(f"com_xyz         = [{com_xyz[0]:+.5f}, {com_xyz[1]:+.5f}, {com_xyz[2]:+.5f}] m")
    print(f"bias (force)    = [{bias[0]:+.4f}, {bias[1]:+.4f}, {bias[2]:+.4f}] N   (sanity check only — see docstring)")
    print(f"bias (torque)   = [{bias[3]:+.4f}, {bias[4]:+.4f}, {bias[5]:+.4f}] Nm")
    print(f"\nResidual RMSE   force  = [{force_rmse[0]:.4f}, {force_rmse[1]:.4f}, {force_rmse[2]:.4f}] N")
    print(f"Residual RMSE   torque = [{torque_rmse[0]:.4f}, {torque_rmse[1]:.4f}, {torque_rmse[2]:.4f}] Nm")

    suggested_threshold = args.threshold_margin * np.std(force_resid_norms)
    print(
        f"\nSuggested contact_lower/upper_threshold ≈ {suggested_threshold:.2f} N "
        f"({args.threshold_margin:.0f}x residual force-norm std {np.std(force_resid_norms):.3f} N) "
        "— tune with real contact/no-contact data, this is only a starting point."
    )

    print("\n" + "─" * 70)
    print("Paste into the force_sensor_filter.py launch command:")
    print("─" * 70)
    print(
        f"--ros-args "
        f"-p com_mass:={mass:.5f} "
        f'-p com_xyz:="[{com_xyz[0]:.5f},{com_xyz[1]:.5f},{com_xyz[2]:.5f}]" '
        f"-p contact_lower_threshold:={suggested_threshold:.2f} "
        f"-p contact_upper_threshold:={suggested_threshold * 1.5:.2f}"
    )


if __name__ == "__main__":
    main()
