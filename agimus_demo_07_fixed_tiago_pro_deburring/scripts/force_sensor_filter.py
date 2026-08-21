#!/usr/bin/env python3
"""
Force sensor filter node — Python port of the Franka `ft_calibration_filter`
(agimus_demos_controllers/ft_calibration_filter.cpp) for the TIAGo Pro right
arm ATI F/T sensor.

The `force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster` already
running on the robot (ati_controller.yaml, frame_id `wrist_right_ft_sensor_link`)
publishes a raw, uncompensated wrench. Neither the LFC node nor
`agimus_controller_node` populate `Sensor.contacts` from it — this node closes
that gap:

  1. subscribes to the raw wrench and the arm joint state,
  2. removes the tool weight (gravity compensation via FK + CoM/mass params)
     and a startup bias,
  3. low-pass filters the result (2nd order Butterworth, per axis),
  4. detects contact with hysteresis,
  5. republishes the `Sensor` message from the LFC with `.contacts` filled —
     this is what `OCPCrocoForceFeedbackGeneric` reads by frame name
     (`pt.point.forces[frame_id]`, see project_demo07_force_feedback_scoping memory).

`com_mass`/`com_xyz` (pal-atc tool weight/CoM) calibrated on robot 2026-08-20
(identification/ft_calibration/, 21 poses) and set as defaults below.
⚠️ Not a precision calibration — URDF not yet corrected by the Figaroh
kinematic calibration, and no real contact/no-contact sweep was done for the
thresholds (raised with margin above the worst residual observed instead —
see project_demo07_force_feedback_scoping memory). Both worth revisiting.

Raw wrench topic name confirmed on robot 2026-08-20:
`/ft_sensor_right_controller/wrench` (the default below).

Subscriptions:
    /robot_description                                    (std_msgs/String, transient local)
    /joint_torque_state_broadcaster/dynamic_joint_states   (control_msgs/DynamicJointState) — absolute_position
    <wrench_topic>                                         (geometry_msgs/WrenchStamped) — raw ATI wrench
    sensor                                                 (linear_feedback_controller_msgs/Sensor) — from LFC

Publications:
    sensor_with_force (linear_feedback_controller_msgs/Sensor) — contacts filled
"""

import glob
import os
import sys
import tempfile
import numpy as np
import pinocchio as pin

for _p in sorted(
    glob.glob("/home/gepetto/ros2_ws/install/*/lib/python3*/site-packages")
    + glob.glob("/home/gepetto/agimus_deps_ws/install/*/lib/python3*/site-packages")
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402

from control_msgs.msg import DynamicJointState  # noqa: E402
from geometry_msgs.msg import Vector3, Wrench, WrenchStamped  # noqa: E402
from linear_feedback_controller_msgs.msg import Contact, Sensor  # noqa: E402
from std_msgs.msg import String  # noqa: E402


def _skew(v: np.ndarray) -> np.ndarray:
    """3x3 cross-product (skew-symmetric) matrix of a 3-vector."""
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


class Butterworth2:
    """2nd order IIR low-pass filter: y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2]
    - a1 y[n-1] - a2 y[n-2], with a = [a1, a2] (a0 == 1) and b = [b0, b1, b2].

    NOTE: this is *not* a literal port of agimus_demos_controllers'
    ButterworthFilter (butterworth_filter.hpp) — that implementation
    overwrites its y[n-1] history with the just-computed y[n] before the
    next call (`filtered_[0] = filtered_[1]` runs after `filtered_[1]` has
    already been updated), which breaks the 2nd-order recursion. This class
    keeps x/y history correctly instead."""

    def __init__(self, a: np.ndarray, b: np.ndarray):
        self._a = np.asarray(a, dtype=float)  # [a1, a2]
        self._b = np.asarray(b, dtype=float)  # [b0, b1, b2]
        self._x_hist = np.zeros(2)  # [x[n-2], x[n-1]]
        self._y_hist = np.zeros(2)  # [y[n-2], y[n-1]]

    def update(self, observation: float) -> float:
        y_n = (
            self._b[0] * observation
            + self._b[1] * self._x_hist[1]
            + self._b[2] * self._x_hist[0]
            - self._a[0] * self._y_hist[1]
            - self._a[1] * self._y_hist[0]
        )
        self._x_hist = np.array([self._x_hist[1], observation])
        self._y_hist = np.array([self._y_hist[1], y_n])
        return y_n


class ContactDetector:
    """Hysteresis-based contact detection on the norm of masked force axes,
    direct port of agimus_demos_controllers/contact_detector.hpp."""

    def __init__(
        self,
        axis_mask: str,
        lower_threshold: float,
        upper_threshold: float,
        hysteresis_samples: int,
    ):
        self._mask = np.array([1.0 if a in axis_mask else 0.0 for a in "xyz"])
        self._lower = lower_threshold
        self._upper = upper_threshold
        self._hysteresis_samples = hysteresis_samples
        self._last_in_contact = False
        self._in_contact = False
        self._samples_since_switch = 0

    def update(self, force_linear: np.ndarray) -> bool:
        thresh = self._lower if self._last_in_contact else self._upper
        self._in_contact = float(np.linalg.norm(force_linear * self._mask)) > thresh
        if (
            self._last_in_contact != self._in_contact
            and self._samples_since_switch > self._hysteresis_samples
        ):
            self._last_in_contact = self._in_contact
            self._samples_since_switch = 0
        else:
            self._samples_since_switch = min(
                self._samples_since_switch + 1, 5 * self._hysteresis_samples
            )
        return self._last_in_contact


class ForceSensorFilterNode(Node):
    def __init__(self):
        super().__init__("force_sensor_filter")

        # ── Parameters (mirror ft_calibration_filter_parameters.yaml) ──────────
        self.declare_parameter("wrench_topic", "/ft_sensor_right_controller/wrench")
        self.declare_parameter("measurement_frame_id", "wrist_right_ft_sensor_link")
        self.declare_parameter("contact_name", "wrist_right_ft_sensor_link")
        self.declare_parameter("gravity_vector", [0.0, 0.0, 9.81])
        self.declare_parameter("bias_measurement_samples", 50)
        self.declare_parameter("calibration_measurement_frame_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("calibration_measurement_frame_rpy", [0.0, 0.0, 0.0])
        # From identification/ft_calibration/calibrate_ft_sensor.py, 21 poses,
        # 2026-08-20 (URDF not recalibrated by Figaroh — see module docstring
        # and project_demo07_force_feedback_scoping memory for residuals/caveats).
        self.declare_parameter("com_xyz", [0.00039, 0.00088, 0.05880])
        self.declare_parameter("com_mass", 1.36619)
        self.declare_parameter("filter_cutoff_hz", 18.0)
        self.declare_parameter("filter_sample_rate_hz", 1000.0)
        # Raised with margin above the worst no-contact residual observed on
        # robot (2026-08-20, post-calibration): 0.69N / 2.56N at 2 tested
        # poses, on an URDF not yet corrected by the Figaroh kinematic
        # calibration. The regression's own suggestion (2.21N) was BELOW that
        # 2.56N no-contact residual — would have false-positived. Still not a
        # real tuning (no actual contact/no-contact sweep done), and no
        # hysteresis gap before this change (lower == upper == 5.0). See
        # project_demo07_force_feedback_scoping memory.
        # "z" = wrist_right_ft_sensor_link's own local z (raw sensor-frame
        # reading, no rotation applied here) — the tool's pushing axis,
        # confirmed with Clément 2026-08-21, matches enabled_directions in
        # ocp_definition_file.yaml (1D contact, ref: LOCAL). Was "xyz" (norm
        # over all 3 axes) — narrowing to z drops x/y calibration noise from
        # the detector, it no longer needs to reflect what the OCP tracks.
        self.declare_parameter("contact_axis_mask", "z")
        self.declare_parameter("contact_lower_threshold", 4.0)
        self.declare_parameter("contact_upper_threshold", 6.0)
        self.declare_parameter("contact_hysteresis_samples", 5)

        p = self.get_parameter
        self._measurement_frame = p("measurement_frame_id").value
        self._contact_name = p("contact_name").value
        self._g = np.array(p("gravity_vector").value)
        self._bias_n = p("bias_measurement_samples").value
        calib_xyz = np.array(p("calibration_measurement_frame_xyz").value)
        calib_rpy = np.array(p("calibration_measurement_frame_rpy").value)
        self._calibration = pin.SE3(pin.rpy.rpyToMatrix(calib_rpy), calib_xyz)
        self._com_xyz = np.array(p("com_xyz").value)
        self._com_mass = p("com_mass").value

        cutoff = p("filter_cutoff_hz").value
        fs = p("filter_sample_rate_hz").value
        a, b = self._butter2(cutoff, fs)
        self._filters = [Butterworth2(a, b) for _ in range(6)]
        self._contact_detector = ContactDetector(
            p("contact_axis_mask").value,
            p("contact_lower_threshold").value,
            p("contact_upper_threshold").value,
            p("contact_hysteresis_samples").value,
        )

        if self._com_mass == 0.0:
            self.get_logger().warn(
                "com_mass is 0.0 — gravity compensation disabled until the "
                "pal-atc tool is calibrated (see module docstring)."
            )

        # ── State ────────────────────────────────────────────────────────────
        self._model = None
        self._data = None
        self._frame_id = None
        self._q_current: np.ndarray | None = None
        self._bias: np.ndarray | None = None
        self._bias_samples: list[np.ndarray] = []
        self._filtered_wrench: np.ndarray | None = None  # [fx,fy,fz,tx,ty,tz]
        self._in_contact: bool = False

        # ── ROS I/O ──────────────────────────────────────────────────────────
        qos_latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String, "/robot_description", self._urdf_cb, qos_latched
        )
        self.create_subscription(
            DynamicJointState,
            "/joint_torque_state_broadcaster/dynamic_joint_states",
            self._js_cb,
            10,
        )
        self.create_subscription(
            WrenchStamped, p("wrench_topic").value, self._wrench_cb, 10
        )
        _qos_sensor = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Sensor, "sensor", self._sensor_cb, _qos_sensor)
        self._pub = self.create_publisher(Sensor, "sensor_with_force", _qos_sensor)

        self.get_logger().info(
            "Force sensor filter ready — waiting for robot_description …"
        )

    @staticmethod
    def _butter2(
        cutoff_hz: float, sample_rate_hz: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """2nd order Butterworth low-pass design (bilinear transform),
        returns (a=[a1,a2], b=[b0,b1,b2]) matching ButterworthFilter's
        convention (a0 == 1, implicit)."""
        from scipy.signal import butter

        b, a = butter(2, cutoff_hz, fs=sample_rate_hz)
        return a[1:], b

    # ── Model loading (same pattern as mocap_mpc_corrector.py) ────────────────

    def _urdf_cb(self, msg: String) -> None:
        if self._model is not None:
            return
        with tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w") as f:
            f.write(msg.data)
            tmp = f.name
        model = pin.buildModelFromUrdf(tmp)
        os.unlink(tmp)
        frame_id = None
        for i, frame in enumerate(model.frames):
            if frame.name == self._measurement_frame:
                frame_id = i
                break
        if frame_id is None:
            self.get_logger().error(
                f"Frame '{self._measurement_frame}' not found in model."
            )
            return
        self._model = model
        self._data = model.createData()
        self._frame_id = frame_id
        self.get_logger().info(
            f"Model loaded — measurement frame: '{self._measurement_frame}' "
            f"(id={frame_id})"
        )

    def _js_cb(self, msg: DynamicJointState) -> None:
        """Same absolute_position-with-fallback pattern as mocap_mpc_corrector.py."""
        if self._model is None:
            return
        q = pin.neutral(self._model)
        for i, jname_raw in enumerate(msg.joint_names):
            iv = msg.interface_values[i]
            imap = dict(zip(iv.interface_names, iv.values))
            val = imap.get("absolute_position", imap.get("position"))
            if val is None:
                continue
            for jname in (f"tiago_pro/{jname_raw}", jname_raw):
                try:
                    jid = self._model.getJointId(jname)
                    if jid < self._model.njoints and self._model.joints[jid].nq == 1:
                        q[self._model.joints[jid].idx_q] = val
                        break
                except Exception:
                    pass
        self._q_current = q

    # ── Wrench processing ───────────────────────────────────────────────────

    def _gravity_wrench(self) -> np.ndarray | None:
        """Wrench felt at the measurement frame due to the tool weight,
        expressed in that frame — same formula as ft_calibration_filter.cpp:
            f_lin = m · R_calib^T · R_frame^T · g_world
            f_ang = skew(com_xyz) · f_lin
        """
        if self._model is None or self._q_current is None:
            return None
        pin.forwardKinematics(self._model, self._data, self._q_current)
        T_frame = pin.updateFramePlacement(self._model, self._data, self._frame_id)
        f_lin = (
            self._com_mass * self._calibration.rotation.T @ T_frame.rotation.T @ self._g
        )
        f_ang = _skew(self._com_xyz) @ f_lin
        return np.concatenate([f_lin, f_ang])

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        raw = np.array(
            [
                msg.wrench.force.x,
                msg.wrench.force.y,
                msg.wrench.force.z,
                msg.wrench.torque.x,
                msg.wrench.torque.y,
                msg.wrench.torque.z,
            ]
        )
        f_gravity = self._gravity_wrench()
        if f_gravity is None:
            return  # model/state not ready yet

        if self._bias is None:
            self._bias_samples.append(raw - f_gravity)
            if len(self._bias_samples) >= self._bias_n:
                self._bias = np.mean(self._bias_samples, axis=0)
                self.get_logger().info(
                    f"Bias computed from {self._bias_n} samples: "
                    f"force={np.round(self._bias[:3], 2)} N, "
                    f"torque={np.round(self._bias[3:], 2)} Nm"
                )
            return  # freeze output during bias collection, like the C++ filter

        f_out = raw - self._bias - f_gravity
        f_out = np.array([flt.update(v) for flt, v in zip(self._filters, f_out)])
        self._filtered_wrench = f_out
        self._in_contact = self._contact_detector.update(f_out[:3])

    # ── Sensor augmentation ─────────────────────────────────────────────────

    def _sensor_cb(self, msg: Sensor) -> None:
        if self._filtered_wrench is not None:
            w = self._filtered_wrench
            msg.contacts = list(msg.contacts) + [
                Contact(
                    active=self._in_contact,
                    name=self._contact_name,
                    wrench=Wrench(
                        force=Vector3(x=float(w[0]), y=float(w[1]), z=float(w[2])),
                        torque=Vector3(x=float(w[3]), y=float(w[4]), z=float(w[5])),
                    ),
                )
            ]
        self._pub.publish(msg)


def main(args=None):
    import signal

    rclpy.init(args=args)
    node = ForceSensorFilterNode()

    def _shutdown(sig, frame):
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
