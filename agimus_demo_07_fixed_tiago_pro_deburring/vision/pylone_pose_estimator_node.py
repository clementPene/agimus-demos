#!/usr/bin/env python3
"""
Persistent ROS2 node: on-demand MegaPose pose estimation for the pylone.

Exposes:
  - Service /vision_pylone/estimate (std_srvs/Trigger): captures one camera
    frame, runs MegaPose inference against the pre-calibrated 'pylone'
    happypose example, composes the result with the base_link -> camera TF,
    and publishes it.
  - Topic /vision_pylone/pose (geometry_msgs/PoseStamped, TRANSIENT_LOCAL):
    last estimated pylone pose in base_frame (default base_link).
  - TF base_frame -> pylone_vision_estimate, broadcast once per estimate.

MegaPose is not a detector: the 2D bbox is NOT re-detected on each call.
It reuses the bbox from the last manual calibration done with
make_megapose_example.py, stored in
$HAPPYPOSE_DATA_DIR/examples/<label>/object_data.json. Recalibrate with
make_megapose_example.py if the camera/pylone setup changes significantly.

Inference takes ~20-30s on a small GPU (validated on a T400 4GB) — this is
why the interface is request/response, not a streamed topic. Run this node
in the vision_cuda container (GPU + happypose); it is reachable from the
control container over plain ROS2/DDS since both devcontainers run with
--network host.

If GPU memory usage grows across repeated /vision_pylone/estimate calls
(watch with nvidia-smi), restart this node — earlier debugging in this repo
found that Panda3D's GLX renderer can leave GPU memory pinned after a
process exits; whether that also accumulates within one long-lived process
across many in-process inferences hasn't been stress-tested yet.

Usage:
    python3 pylone_pose_estimator_node.py
    python3 pylone_pose_estimator_node.py --label pylone \
        --image-topic /head_front_camera/color/image_raw \
        --info-topic /head_front_camera/color/camera_info \
        --base-frame base_link
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_srvs.srv import Trigger
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformBroadcaster,
    TransformListener,
)

from happypose.pose_estimators.megapose.scripts.run_inference_on_example import (
    run_inference,
    setup_pose_estimator,
)
from happypose.toolbox.inference.example_inference_utils import (
    load_detections,
    load_observation_example,
    make_example_object_dataset,
)
from happypose.toolbox.inference.types import ObservationTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_IMAGE_TOPIC = "/head_front_camera/color/image_raw"
DEFAULT_INFO_TOPIC = "/head_front_camera/color/camera_info"
DEFAULT_BASE_FRAME = "base_link"
CHILD_FRAME = "pylone_vision_estimate"
DEFAULT_LABEL = "pylone"


def image_msg_to_bgr(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = data.reshape(msg.height, msg.width, 3)
        if msg.encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def _transform_msg_to_se3(t) -> pin.SE3:
    xyz = [t.translation.x, t.translation.y, t.translation.z]
    quat = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
    return pin.XYZQUATToSE3(np.array(xyz + quat))


def _se3_to_transform_stamped(T, stamp, parent, child) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(T.translation[0])
    msg.transform.translation.y = float(T.translation[1])
    msg.transform.translation.z = float(T.translation[2])
    q = pin.Quaternion(T.rotation)
    msg.transform.rotation.x = float(q.x)
    msg.transform.rotation.y = float(q.y)
    msg.transform.rotation.z = float(q.z)
    msg.transform.rotation.w = float(q.w)
    return msg


def _se3_to_pose_stamped(T, stamp, frame) -> PoseStamped:
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame
    msg.pose.position.x = float(T.translation[0])
    msg.pose.position.y = float(T.translation[1])
    msg.pose.position.z = float(T.translation[2])
    q = pin.Quaternion(T.rotation)
    msg.pose.orientation.x = float(q.x)
    msg.pose.orientation.y = float(q.y)
    msg.pose.orientation.z = float(q.z)
    msg.pose.orientation.w = float(q.w)
    return msg


class PylonePoseEstimatorNode(Node):

    def __init__(self, args):
        super().__init__("pylone_pose_estimator")
        self._args = args

        data_dir = os.environ.get("HAPPYPOSE_DATA_DIR")
        if not data_dir:
            raise RuntimeError("Set HAPPYPOSE_DATA_DIR before starting this node.")
        self._example_dir = Path(data_dir) / "examples" / args.label
        if not os.path.exists(os.path.join(self._example_dir, "object_data.json")):
            raise RuntimeError(
                f"No calibrated bbox at {self._example_dir}/object_data.json — "
                "run make_megapose_example.py once first."
            )

        self.get_logger().info(f"Loading MegaPose model ({args.model}) …")
        object_dataset = make_example_object_dataset(self._example_dir)
        self._pose_estimator, self._model_info = setup_pose_estimator(
            args.model, object_dataset)
        self.get_logger().info("Model loaded.")

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        pose_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._pose_pub = self.create_publisher(PoseStamped, "/vision_pylone/pose", pose_qos)

        self._srv = self.create_service(Trigger, "/vision_pylone/estimate", self._on_estimate)

        self.get_logger().info("Ready. Call /vision_pylone/estimate to trigger an estimate.")

    def _capture_once(self, timeout: float = 10.0):
        image_msg = [None]
        info_msg = [None]
        sub_img = self.create_subscription(
            Image, self._args.image_topic,
            lambda m: image_msg.__setitem__(0, m), qos_profile_sensor_data)
        sub_info = self.create_subscription(
            CameraInfo, self._args.info_topic,
            lambda m: info_msg.__setitem__(0, m), qos_profile_sensor_data)
        # Reentrant spin_once: this runs inside the /vision_pylone/estimate
        # service callback, itself dispatched by the outer rclpy.spin(node)
        # in main(). Nesting spin_once like this to wait for a couple of
        # one-shot messages from within a service callback is a standard,
        # supported rclpy idiom (single-threaded executor processes one
        # ready item per call); it's the same "wait for one message" pattern
        # already used by capture_image.py's OneShotCapture, just triggered
        # from a service instead of a script's main().
        deadline = time.time() + timeout
        while time.time() < deadline and (image_msg[0] is None or info_msg[0] is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub_img)
        self.destroy_subscription(sub_info)
        if image_msg[0] is None or info_msg[0] is None:
            raise RuntimeError(
                f"Timeout waiting for {self._args.image_topic} / {self._args.info_topic}")
        return image_msg[0], info_msg[0]

    def _on_estimate(self, request, response):
        try:
            t0 = time.time()
            image_msg, info_msg = self._capture_once()

            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self._args.base_frame,
                    info_msg.header.frame_id,
                    rclpy.time.Time.from_msg(image_msg.header.stamp),
                    timeout=rclpy.duration.Duration(seconds=5.0),
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                response.success = False
                response.message = (
                    f"TF lookup {self._args.base_frame}->{info_msg.header.frame_id} "
                    f"failed: {e}"
                )
                return response
            T_base_camera = _transform_msg_to_se3(tf_msg.transform)

            # Overwrite the calibrated example's image + intrinsics with the
            # fresh capture; keep the calibrated bbox (object_data.json) and
            # mesh untouched.
            img = image_msg_to_bgr(image_msg)
            cv2.imwrite(os.path.join(self._example_dir, "image_rgb.png"), img)
            K = np.array(info_msg.k).reshape(3, 3)
            camera_data = {
                "K": K.tolist(),
                "resolution": [info_msg.height, info_msg.width],
            }
            with open(os.path.join(self._example_dir, "camera_data.json"), "w") as f:
                json.dump(camera_data, f)

            rgb, _, cam_data = load_observation_example(self._example_dir, load_depth=False)
            observation = ObservationTensor.from_numpy(rgb, None, cam_data.K).to(device)
            detections = load_detections(self._example_dir).to(device)

            output = run_inference(self._pose_estimator, self._model_info, observation, detections)
            labels = list(output.infos["label"])
            idx = labels.index(self._args.label)
            pose = output.poses.numpy()[idx]
            T_camera_pylone = pin.SE3(pose[:3, :3], pose[:3, 3])

            T_base_pylone = T_base_camera * T_camera_pylone

            stamp = self.get_clock().now().to_msg()
            self._tf_broadcaster.sendTransform(_se3_to_transform_stamped(
                T_base_pylone, stamp, self._args.base_frame, CHILD_FRAME))
            self._pose_pub.publish(_se3_to_pose_stamped(
                T_base_pylone, stamp, self._args.base_frame))

            dt = time.time() - t0
            response.success = True
            response.message = f"Estimated pylone pose in {dt:.1f}s"
            self.get_logger().info(response.message)
        except Exception as e:
            self.get_logger().error(f"Estimation failed: {e}")
            response.success = False
            response.message = str(e)
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--model", default="megapose-1.0-RGB")
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--info-topic", default=DEFAULT_INFO_TOPIC)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = PylonePoseEstimatorNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
