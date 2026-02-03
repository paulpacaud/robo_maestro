#!/usr/bin/env python3
"""
ROS 2 node – read and print the current EEF pose of the robot.

Usage:
ros2 launch robo_maestro read_eef_pose.launch.py use_sim_time:=false

"""

import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from moveit.planning import MoveItPy

from robo_maestro.core.observer import TFRecorder
from robo_maestro.utils.constants import EEF_FRAME, ROBOT_BASE_FRAME
from robo_maestro.utils.logger import log_info, log_error, log_success


def main():
    rclpy.init()
    node = Node("read_eef_pose")

    arm = node.declare_parameter("arm", "left").get_parameter_value().string_value

    log_info("Initializing MoveItPy ...")
    ur = MoveItPy(node_name="moveit_py")
    time.sleep(5)

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    eef_frame = EEF_FRAME[arm]
    eef_tf_recorder = TFRecorder(node, ROBOT_BASE_FRAME, eef_frame)

    log_info(f"Reading EEF pose: {ROBOT_BASE_FRAME} -> {eef_frame} ...")

    try:
        eef_tf = eef_tf_recorder.record_tf().transform
        eef_pos = np.array(
            [
                eef_tf.translation.x,
                eef_tf.translation.y,
                eef_tf.translation.z,
            ]
        )
        eef_quat = np.array(
            [
                eef_tf.rotation.x,
                eef_tf.rotation.y,
                eef_tf.rotation.z,
                eef_tf.rotation.w,
            ]
        )

        log_success(
            f"EEF pose ({arm} arm):\n"
            f"  position:    {eef_pos}\n"
            f"  quaternion:  {eef_quat}"
        )
    except Exception as e:
        log_error(f"Failed to read EEF pose: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
