#!/usr/bin/env python3
"""Plot the contact-force profile of a deburring-demo run, testbed-style.

Real-robot analogue of tiago_pro_force_mpc_sim/closed_loop_mujoco.py's
3-panel figure (f_z / |dq| / |u| vs time). Reads a ROS 2 bag instead of a
MuJoCo log so the p2 contact phase can be compared 1:1 against the MuJoCo
testbed's "current best result".

Easiest: let the orchestrator record for you —

    o.execute([o.p2], record="nowall")   # orchestrator.py; bag -> plot/runs/<ts>_nowall/

or record by hand during a p1..p4 run (bringup.launch.py, use_force_feedback:=true):

    ros2 bag record -o force_run \\
        /sensor_with_force /mpc_input /control /ocp_solve_time /mpc_debug

Then:

    python3 plot_force_profile.py runs/20260827_120000_nowall    # -> ...png beside the bag
    python3 plot_force_profile.py force_run --out foo.png
    python3 plot_force_profile.py force_run --t0 12 --t1 30       # zoom on the hold

Signals
    f_z measured : /sensor_with_force  contacts[FT frame].wrench.force.z   (filtered, grav-comp)
    f_z target   : /mpc_input          ee_inputs[FT frame].force.force.z   (orchestrator ramp)
    contact flag : /sensor_with_force  contacts[FT frame].active           (hysteresis detector)
    |dq|         : /sensor_with_force  joint_state.velocity  -> L2 norm
    |u|          : /control            feedforward           -> L2 norm
"""

import argparse
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rclpy.serialization import deserialize_message

from agimus_msgs.msg import MpcInput
from linear_feedback_controller_msgs.msg import Control, Sensor

FT_FRAME_DEFAULT = "wrist_right_ft_sensor_link"

# topic -> message class, for the topics this script reads
_TYPES = {
    "/sensor_with_force": Sensor,
    "/mpc_input": MpcInput,
    "/control": Control,
}


def _iter_bag(bag: Path):
    """Yield (topic, deserialized_msg, t_ns) for the topics in _TYPES.

    Prefers a direct sqlite3 read (no rosbag2_py needed — works in the plain
    tiago_pro_nix devshell); falls back to rosbag2_py for .mcap bags.
    """
    db3 = sorted(bag.glob("*.db3"))
    if db3:
        con = sqlite3.connect(f"file:{db3[0]}?mode=ro", uri=True)
        try:
            names = {
                tid: name
                for tid, name in con.execute("SELECT id, name FROM topics")
            }
            for tid, t_ns, data in con.execute(
                "SELECT topic_id, timestamp, data FROM messages ORDER BY timestamp"
            ):
                topic = names.get(tid)
                cls = _TYPES.get(topic)
                if cls is not None:
                    yield topic, deserialize_message(bytes(data), cls), t_ns
        finally:
            con.close()
        return

    if list(bag.glob("*.mcap")):
        import rosbag2_py

        r = rosbag2_py.SequentialReader()
        r.open(
            rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
        while r.has_next():
            topic, raw, t_ns = r.read_next()
            cls = _TYPES.get(topic)
            if cls is not None:
                yield topic, deserialize_message(raw, cls), t_ns
        return

    raise SystemExit(f"No .db3 or .mcap in {bag}")


def _contact(sensor_msg, frame):
    for c in sensor_msg.contacts:
        if c.name == frame:
            return c
    return None


def _ee_force_z(mpc_msg, frame):
    for ee in mpc_msg.ee_inputs:
        if ee.frame_id == frame:
            return ee.force.force.z
    return None


def load(bag: Path, frame: str):
    log = {k: [] for k in ("t_f", "f_meas", "active", "t_dq", "dq", "t_u", "u", "t_des", "f_des")}
    t_start = None
    for topic, m, t_ns in _iter_bag(bag):
        if t_start is None:
            t_start = t_ns
        t = (t_ns - t_start) * 1e-9

        if topic == "/sensor_with_force":
            c = _contact(m, frame)
            if c is not None:
                log["t_f"].append(t)
                log["f_meas"].append(c.wrench.force.z)
                log["active"].append(bool(c.active))
            if len(m.joint_state.velocity):
                log["t_dq"].append(t)
                log["dq"].append(float(np.linalg.norm(m.joint_state.velocity)))

        elif topic == "/mpc_input":
            fz = _ee_force_z(m, frame)
            if fz is not None:
                log["t_des"].append(t)
                log["f_des"].append(fz)

        elif topic == "/control":
            ff = list(m.feedforward.data)
            if ff:
                log["t_u"].append(t)
                log["u"].append(float(np.linalg.norm(ff)))

    return {k: np.asarray(v) for k, v in log.items()}


def _shade_active(ax, t, active):
    """Grey band wherever the contact detector says active=True."""
    if len(t) == 0:
        return
    a = active.astype(bool)
    edges = np.flatnonzero(np.diff(a.astype(int)))
    starts = [0] if a[0] else []
    starts += [e + 1 for e in edges if not a[e] and a[e + 1]]
    stops = [e + 1 for e in edges if a[e] and not a[e + 1]]
    if a[-1]:
        stops.append(len(a) - 1)
    for s, e in zip(starts, stops):
        ax.axvspan(t[s], t[e], color="0.85", lw=0, zorder=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path)
    p.add_argument("--out", default=None, help="output PNG (default: <bag>.png)")
    p.add_argument("--force-frame", default=FT_FRAME_DEFAULT)
    p.add_argument("--t0", type=float, default=None, help="crop start (s)")
    p.add_argument("--t1", type=float, default=None, help="crop end (s)")
    args = p.parse_args()

    d = load(args.bag, args.force_frame)
    if len(d["t_f"]) == 0:
        raise SystemExit(
            f"No contact '{args.force_frame}' on /sensor_with_force in {args.bag} "
            "— was force_sensor_filter.py running (use_force_feedback:=true)?"
        )

    out = args.out or str(args.bag.parent / f"{args.bag.name}.png")
    lo = args.t0 if args.t0 is not None else 0.0
    hi = args.t1 if args.t1 is not None else max(d["t_f"][-1], d["t_u"][-1] if len(d["t_u"]) else 0)

    # steady-state force = mean of the last 20% of the in-window samples
    m = (d["t_f"] >= lo) & (d["t_f"] <= hi)
    fw = d["f_meas"][m]
    steady = float(np.mean(fw[-max(1, len(fw) // 5):])) if len(fw) else float("nan")
    des_win = d["f_des"][(d["t_des"] >= lo) & (d["t_des"] <= hi)] if len(d["t_des"]) else np.array([])
    target = des_win[np.argmax(np.abs(des_win))] if len(des_win) else float("nan")
    print(
        f"f_z: final {fw[-1]:.2f} N | steady (last 20%) {steady:.2f} N | "
        f"peak {fw[np.argmax(np.abs(fw))]:.2f} N | target ~{target:.1f} N"
    )

    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    _shade_active(ax[0], d["t_f"], d["active"])
    ax[0].plot(d["t_f"], d["f_meas"], lw=1.0, label="measured (filtered)")
    if len(d["t_des"]):
        ax[0].step(d["t_des"], d["f_des"], where="post", color="r", ls="--", lw=1.0, label="target f_des")
    ax[0].set_ylabel("f_z  (N)")
    ax[0].legend(loc="upper left")
    ax[0].set_title(
        f"{args.bag.name}   —   grey = contact detector active   —   steady f_z ≈ {steady:.1f} N"
    )

    _shade_active(ax[1], d["t_f"], d["active"])
    ax[1].plot(d["t_dq"], d["dq"], lw=1.0)
    ax[1].set_ylabel("|dq|  (rad/s)")

    _shade_active(ax[2], d["t_f"], d["active"])
    ax[2].plot(d["t_u"], d["u"], lw=1.0)
    ax[2].set_ylabel("|u| feedforward  (N·m)")
    ax[2].set_xlabel("t  (s)")

    ax[0].set_xlim(lo, hi)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
