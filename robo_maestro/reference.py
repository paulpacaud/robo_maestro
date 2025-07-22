#!/usr/bin/env python3
"""
A script to outline the fundamentals of the moveit_py motion planning API.
"""

import time

# generic ros libraries
import rclpy
from rclpy.logging import get_logger

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    MultiPipelinePlanRequestParameters,
)
import numpy as np
from geometry_msgs.msg import PoseStamped
# A simple function to plan and execute a motion

def plan_and_execute(
    robot,
    planning_component,
    logger,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """Helper function to plan and execute a motion."""
    # plan to goal
    logger.info("Planning trajectory")
    if multi_plan_parameters is not None:
        # multiple pipelines are used to plan the trajectory
        # the planning component will use the first pipeline that returns a valid trajectory
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        # single pipeline is used to plan the trajectory
        # the planning component will use the pipeline specified in the parameters
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        # no pipeline is specified, the planning component will use the default pipeline set in the srdf
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