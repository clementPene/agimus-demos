#!/usr/bin/env python3
"""
One-shot capture of the head camera: saves image_rgb.png + camera_data.json
(happypose format) into the output directory.

Run on a machine that sees the robot topics.

Usage:
    python3 capture_image.py
    python3 capture_image.py --image-topic /head_front_camera/color/image_raw \
        --info-topic /head_front_camera/color/camera_info --output ./capture_01
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class OneShotCapture(Node):

    def __init__(self, image_topic, info_topic):
        super().__init__('megapose_capture')
        self.image_msg = None
        self.info_msg = None
        self.create_subscription(
            Image, image_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data)

    def _image_cb(self, msg):
        self.image_msg = msg

    def _info_cb(self, msg):
        self.info_msg = msg

    def done(self):
        return self.image_msg is not None and self.info_msg is not None


def image_msg_to_bgr(msg):
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('rgb8', 'bgr8'):
        img = data.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    raise ValueError(f'Unsupported encoding: {msg.encoding}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-topic',
                        default='/head_front_camera/color/image_raw')
    parser.add_argument('--info-topic',
                        default='/head_front_camera/color/camera_info')
    parser.add_argument('--output', default='.',
                        help='Output directory (created if needed)')
    parser.add_argument('--timeout', type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = OneShotCapture(args.image_topic, args.info_topic)

    start = node.get_clock().now()
    while rclpy.ok() and not node.done():
        rclpy.spin_once(node, timeout_sec=0.1)
        elapsed = (node.get_clock().now() - start).nanoseconds * 1e-9
        if elapsed > args.timeout:
            missing = []
            if node.image_msg is None:
                missing.append(args.image_topic)
            if node.info_msg is None:
                missing.append(args.info_topic)
            node.get_logger().error(
                f'Timeout, nothing received on: {missing}. '
                'Check topic names with: ros2 topic list | grep -i camera')
            rclpy.shutdown()
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    img = image_msg_to_bgr(node.image_msg)
    image_path = os.path.join(args.output, 'image_rgb.png')
    cv2.imwrite(image_path, img)

    K = np.array(node.info_msg.k).reshape(3, 3)
    camera_data = {
        'K': K.tolist(),
        'resolution': [node.info_msg.height, node.info_msg.width],
    }
    camera_path = os.path.join(args.output, 'camera_data.json')
    with open(camera_path, 'w') as f:
        json.dump(camera_data, f, indent=2)

    print(f'Saved {image_path} ({node.image_msg.width}x{node.image_msg.height})')
    print(f'Saved {camera_path}')
    rclpy.shutdown()


if __name__ == '__main__':
    main()
