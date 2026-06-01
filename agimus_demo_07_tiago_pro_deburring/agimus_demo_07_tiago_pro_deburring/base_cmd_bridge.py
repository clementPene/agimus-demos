"""Bridge node: forwards base velocity commands from the MPC to the mobile base controller.

Clamps linear and angular velocities to safe limits so that the simulation
does not saturate the mobile base controller even when the OCP produces large
feed-forward values (e.g. during initial convergence).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist

MAX_LINEAR_VEL  = 0.3   # m/s
MAX_ANGULAR_VEL = 0.5   # rad/s


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class BaseCmdBridge(Node):
    def __init__(self):
        super().__init__("base_cmd_bridge")
        qos_in  = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_out = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(Twist, "/cmd_vel", qos_out)
        self._sub = self.create_subscription(Twist, "/base_cmd_vel", self._callback, qos_in)

    def _callback(self, msg: Twist) -> None:
        out = Twist()
        out.linear.x  = _clamp(msg.linear.x,  MAX_LINEAR_VEL)
        out.linear.y  = _clamp(msg.linear.y,  MAX_LINEAR_VEL)
        out.angular.z = _clamp(msg.angular.z, MAX_ANGULAR_VEL)
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = BaseCmdBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
