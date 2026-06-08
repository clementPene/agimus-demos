"""
Base trajectory follower for the TIAGo Pro deburring demo.

Subscribes to /base_trajectory (nav_msgs/Path) and /odom, publishes /cmd_vel
using a simple P controller with waypoint advancement.

On the real robot, replace /odom with a mocap-based pose topic for better accuracy.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


class BaseTrajectoryFollower(Node):

    def __init__(self):
        super().__init__("base_trajectory_follower")

        self.declare_parameter("lookahead_dist",  0.08)
        self.declare_parameter("kp_linear",       1.5)
        self.declare_parameter("kp_angular",      2.0)
        self.declare_parameter("max_linear_vel",  0.4)
        self.declare_parameter("max_angular_vel", 0.8)

        self._lookahead  = self.get_parameter("lookahead_dist").value
        self._kp_lin     = self.get_parameter("kp_linear").value
        self._kp_ang     = self.get_parameter("kp_angular").value
        self._max_lin    = self.get_parameter("max_linear_vel").value
        self._max_ang    = self.get_parameter("max_angular_vel").value

        self._waypoints  = []   # list of (x, y, theta)
        self._current_idx = 0
        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0
        self._odom_ready = False

        qos_path = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Path,    "/base_trajectory", self._path_cb,  qos_path)
        self.create_subscription(Odometry, "/odom",           self._odom_cb,  10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_timer(0.05, self._control_loop)   # 20 Hz

        self.get_logger().info("Base trajectory follower ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _path_cb(self, msg: Path) -> None:
        if not msg.poses:
            return
        self._waypoints = []
        for ps in msg.poses:
            x = ps.pose.position.x
            y = ps.pose.position.y
            qz = ps.pose.orientation.z
            qw = ps.pose.orientation.w
            theta = 2.0 * math.atan2(qz, qw)
            self._waypoints.append((x, y, theta))
        self._current_idx = 0
        self.get_logger().info(f"Received path with {len(self._waypoints)} waypoints.")

    def _odom_cb(self, msg: Odometry) -> None:
        self._x   = msg.pose.pose.position.x
        self._y   = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self._yaw = 2.0 * math.atan2(qz, qw)
        self._odom_ready = True

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if not self._odom_ready or not self._waypoints:
            return

        # Advance to the next waypoint when close enough
        while self._current_idx < len(self._waypoints) - 1:
            tx, ty, _ = self._waypoints[self._current_idx]
            dist = math.hypot(tx - self._x, ty - self._y)
            if dist < self._lookahead:
                self._current_idx += 1
            else:
                break

        tx, ty, ttheta = self._waypoints[self._current_idx]

        # Stop when at the last waypoint and close enough
        if (self._current_idx == len(self._waypoints) - 1
                and math.hypot(tx - self._x, ty - self._y) < self._lookahead
                and abs(self._angle_diff(ttheta, self._yaw)) < 0.05):
            self._publish_zero()
            return

        # Position error in world frame → body frame
        dx_world = tx - self._x
        dy_world = ty - self._y
        cos_yaw  = math.cos(self._yaw)
        sin_yaw  = math.sin(self._yaw)
        dx_body  =  cos_yaw * dx_world + sin_yaw * dy_world
        dy_body  = -sin_yaw * dx_world + cos_yaw * dy_world

        dtheta = self._angle_diff(ttheta, self._yaw)

        vx    = np.clip(self._kp_lin * dx_body,  -self._max_lin, self._max_lin)
        vy    = np.clip(self._kp_lin * dy_body,  -self._max_lin, self._max_lin)
        omega = np.clip(self._kp_ang * dtheta,   -self._max_ang, self._max_ang)

        twist = Twist()
        twist.linear.x  = vx
        twist.linear.y  = vy
        twist.angular.z = omega
        self._cmd_pub.publish(twist)

    def _publish_zero(self) -> None:
        self._cmd_pub.publish(Twist())

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed shortest angular difference a - b in [-pi, pi]."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))


def main(args=None):
    rclpy.init(args=args)
    node = BaseTrajectoryFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
