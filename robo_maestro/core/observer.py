#!/usr/bin/env python3
"""
camera_utils_ros2.py
ROS 2 replacement for the old rospy utilities.
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, Image, JointState
from tf_transformations import euler_from_quaternion
import tf2_ros

from robo_maestro.utils.constants import ROBOT_BASE_FRAME


# ---------------------------------------------------------------------------
# Convenience helper – rclpy does NOT have rospy.wait_for_message
# ---------------------------------------------------------------------------
def wait_for_message(node: Node, msg_type, topic: str, timeout: float | None = None):
    """
    Block until a single message of `msg_type` arrives on `topic`.

    Parameters
    ----------
    node : rclpy.node.Node
    msg_type : ROS 2 message class
    topic : str
    timeout : float | None
        Seconds to wait; None means “forever”.

    Returns
    -------
    msg_type instance
    """
    future: Future = Future()

    def _cb(msg):
        if not future.done():
            future.set_result(msg)
        sub.destroy()

    sub = node.create_subscription(msg_type, topic, _cb, 10)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)

    if not future.done():
        raise TimeoutError(f"No message on {topic} within {timeout} s")
    return future.result()


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------
class Camera:
    def __init__(self, node: Node, topic: str):
        self.node = node
        self._topic = topic
        self._info_topic = os.path.join(os.path.dirname(topic), "camera_info")
        try:
            self.info: CameraInfo = wait_for_message(
                    node,
                    CameraInfo,
                    self._info_topic,
                    timeout = 2,
            )
        except TimeoutError as err:
            raise RuntimeError(
                f"[Camera] No `sensor_msgs/CameraInfo` received on "
                f"'{self._info_topic}' within 2s. "
                "Is the camera driver running and is the topic name correct?"
                ) from err
        self.intrinsics = self.info_to_intrinsics()

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def info_to_intrinsics(self):
        """
        Build an OpenCV‑style intrinsics dict from CameraInfo.
        NB: In ROS 2 the array is `k` (lower‑case) rather than `K`.
        """
        m = self.info
        return dict(
            height=m.height,
            width=m.width,
            fx=m.k[0],
            fy=m.k[4],
            ppx=m.k[2],
            ppy=m.k[5],
            K=np.asarray(m.k).reshape(3, 3),
        )

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def record_image(self, timeout: float | None = None,
                     dtype=np.uint8) -> np.ndarray:
        msg: Image = wait_for_message(self.node, Image,
                                      self._topic, timeout=timeout)
        data = np.frombuffer(msg.data, dtype=dtype)
        return data.reshape((msg.height, msg.width, -1))


# ---------------------------------------------------------------------------
class TFRecorder:
    def __init__(self, node: Node,
                 source_frame: str,
                 target_frame: str):
        self.node = node
        self._source_frame = source_frame
        self._target_frame = target_frame

        # Buffer caches 10 s to ride out small TF gaps
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def record_tf(self, timeout: float = 4.0, now: bool = False):
        """
        Return a geometry_msgs/TransformStamped like tf2lookup in ROS 1.
        """
        try:
            stamp: Time = self.node.get_clock().now() if now else Time()
            return self.tf_buffer.lookup_transform(
                self._source_frame,
                self._target_frame,
                stamp,
                timeout=Duration(seconds=timeout),
            )
        except tf2_ros.TransformException as e:
            self.node.get_logger().error(f"TF lookup failed: {e}")
            raise

# ---------------------------------------------------------------------------
class CameraPose(Camera):
    """
    Camera that also returns its pose w.r.t. ROBOT_BASE_FRAME.
    """
    def __init__(self, node: Node, topic: str, camera_frame: str):
        super().__init__(node, topic)
        self.tf_recorder = TFRecorder(node, ROBOT_BASE_FRAME, camera_frame)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def record_image(self, dtype=np.uint8):
        img = super().record_image(dtype=dtype)
        tf = self.tf_recorder.record_tf()

        pos = tf.transform.translation
        rot = tf.transform.rotation

        cam_pos = [pos.x, pos.y, pos.z]
        cam_euler = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])

        return img, (cam_pos, cam_euler)

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def get_pose(self):
        tf = self.tf_recorder.record_tf()

        pos = tf.transform.translation
        rot = tf.transform.rotation

        return (
            [pos.x, pos.y, pos.z],
            euler_from_quaternion([rot.x, rot.y, rot.z, rot.w]),
        )


# ---------------------------------------------------------------------------
class JointStateRecorder:
    def __init__(self, node: Node, topic: str = "/joint_states"):
        self.node = node
        self._topic = topic

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def record_state(self, timeout: float | None = None):
        msg: JointState = wait_for_message(self.node, JointState,
                                           self._topic, timeout=timeout)
        return dict(
            joint_position=msg.position,
            joint_names=msg.name,
            joint_velocity=msg.velocity,
        )


# ---------------------------------------------------------------------------
# Quick smoke‑test / example usage
# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = rclpy.create_node("camera_and_tf_recorder")

    cam = CameraPose(node,
                     topic="/camera/color/image_raw",
                     camera_frame="camera_link")

    img, (pos, rpy) = cam.record_image(timeout=2.0)
    node.get_logger().info(f"Camera pose: {pos} m, {rpy} rad")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
