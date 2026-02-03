#!/usr/bin/env python3
"""
ROS 2 node – send the robot to a given 8-dim EEF pose and exit.

The pose is (x, y, z, qx, qy, qz, qw, gripper).

Usage:
ros2 launch robo_maestro control_robot_eef.launch.py use_sim_time:=false

ros2 launch robo_maestro control_robot_eef.launch.py \
    action:="-0.25,0,0.3,-7.21e-05,0.999,-4.89e-05,-6.22e-03,0" \
    use_sim_time:=false
"""

import threading

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from robo_maestro.core.robot import Robot
from robo_maestro.utils.constants import WORKSPACE, DEFAULT_ROBOT_ACTION
from robo_maestro.utils.logger import log_info, log_error, log_success


def main():
    rclpy.init()
    node = Node(
        "control_robot_eef",
        allow_undeclared_parameters=True,
        automatically_declare_parameters_from_overrides=True,
    )

    # Parse action from string parameter "x,y,z,qx,qy,qz,qw,grip"
    try:
        action_str = node.get_parameter("action").get_parameter_value().string_value
    except rclpy.exceptions.ParameterNotDeclaredException:
        action_str = ""

    if action_str:
        try:
            action = np.array([float(v) for v in action_str.split(",")])
        except ValueError:
            log_error(f"Could not parse action string: '{action_str}'")
            node.destroy_node()
            rclpy.shutdown()
            return
        if action.shape[0] != 8:
            log_error(
                f"Action must have exactly 8 dimensions "
                f"(x,y,z,qx,qy,qz,qw,grip), got {action.shape[0]}"
            )
            node.destroy_node()
            rclpy.shutdown()
            return
    else:
        log_info("No action parameter provided, using DEFAULT_ROBOT_ACTION.")
        action = np.array(DEFAULT_ROBOT_ACTION)

    log_info(f"Target EEF pose: {action}")

    # Spin thread (needed for TF, gripper client, etc.)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # Instantiate Robot with no cameras
        robot = Robot(WORKSPACE["left"], cam_list=[], node=node)

        # Execute
        robot.go_to_pose(action.tolist())
        log_success(f"Robot moved to pose: {action}")
    except Exception as e:
        log_error(f"Failed to move robot: {e}")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
