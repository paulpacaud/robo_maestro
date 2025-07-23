#!/usr/bin/env python3
"""
A script to outline the fundamentals of the moveit_py motion planning API.
"""

import time

# generic ros libraries
import rclpy
from rclpy.logging import get_logger

# moveit python library
from moveit.planning import MoveItPy
from geometry_msgs.msg import PoseStamped
import numpy as np
# A simple function to plan and execute a motion

def plan_and_execute(
    robot,
    planning_component,
    logger,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    logger.info("Planning trajectory")
    plan_result = planning_component.plan()

    # execute the plan if it is valid trajectory
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)


def main():

    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("run_policy")

    # instantiate MoveItPy instance and get planning component
    ur = MoveItPy(node_name="moveit_py")
    left_arm = ur.get_planning_component("left_arm")

    logger.info("MoveItPy instance created")

    ###########################################################################
    # Plan - set goal state with PoseStamped message
    # - set the pose orientation and position, x, y, z coordinates, quaternions
    # - "base_link" = reference frame for the pose
    # - "left_tool0" = link to which the pose is applied, here the left arm end effector
    ###########################################################################

    left_arm.set_start_state_to_current_state()

    left_pose_goal = PoseStamped()
    left_pose_goal.header.frame_id = "base_link"
    left_pose_goal.pose.orientation.w = 0.0
    left_pose_goal.pose.orientation.x = 0.5
    left_pose_goal.pose.orientation.y = 0.0
    left_pose_goal.pose.orientation.z = 0.0
    left_pose_goal.pose.position.x = 0.0
    left_pose_goal.pose.position.y = 0.191
    left_pose_goal.pose.position.z = 1.001
    left_arm.set_goal_state(pose_stamped_msg=left_pose_goal, pose_link="left_tool0")

    plan_and_execute(ur, left_arm, logger, sleep_time=3.0)


if __name__ == "__main__":
    main()