from types import SimpleNamespace
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.robot_state import RobotState

import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
import numpy as np
from robo_maestro.core.observer import TFRecorder, JointStateRecorder, CameraPose
from robo_maestro.utils.constants import *
import time

from robo_maestro.utils.logger import log_info, log_error, log_warn


class Robot:
    def __init__(self, workspace, cam_list, node=None, use_sim_time=False):
        arm = 'left'
        self.eef_frame = EEF_FRAME[arm]
        self.workspace = workspace
        self.use_sim_time = use_sim_time
        self.node = node

        self._eef_tf_recorder = TFRecorder(self.node, ROBOT_BASE_FRAME, self.eef_frame)
        self._links_tf_recorder = {
            link: TFRecorder(self.node, ROBOT_BASE_FRAME, link)
            for link in ROBOT_LINKS[arm]
        }
        self.joints_state_recorder = JointStateRecorder(self.node)

        # Cameras
        self.cam_list = cam_list
        self.cameras = {}
        self.depth_cameras = {}

        for cam_name in cam_list:
            self.cameras[cam_name] = CameraPose(
                node=self.node,
                topic=f"/camera/{cam_name}/color/image_raw",
                camera_frame=CAM_TF_TOPIC[cam_name]
            )
            self.depth_cameras[cam_name] = CameraPose(
                node=self.node,
                topic=f"/camera/{cam_name}/aligned_depth_to_color/image_raw",
                camera_frame=CAM_TF_TOPIC[cam_name]
            )

        # robot controller
        self.ur = MoveItPy(node_name="moveit_py")
        time.sleep(5)
        self.left_arm = self.ur.get_planning_component("left_arm")
        self.left_gripper = self.ur.get_planning_component("left_gripper")

        # vanilla plan params
        self.plan_params = PlanRequestParameters(self.ur, "left_arm")
        self.plan_params.planning_pipeline = "ompl"
        self.plan_params.planner_id = "RRTConnect" # TODO: to change - not the best choice as it is very good for cluttered envs, but not for open spaces
        self.plan_params.max_velocity_scaling_factor = 0.25
        self.plan_params.max_acceleration_scaling_factor = 0.25

        # cartesian_only plan params
        self.lin_plan_params = PlanRequestParameters(self.ur, "left_arm")
        self.lin_plan_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.lin_plan_params.planner_id = "LIN"
        self.lin_plan_params.max_velocity_scaling_factor = 0.15
        self.lin_plan_params.max_acceleration_scaling_factor = 0.15

        self.gripper_state = 0
        log_info("Robot initialized with workspace")

        self._wait_for_controllers()
        # self.reset()
        self.eef_pose()

    def _wait_for_controllers(self, timeout=30.0):
        """Wait for trajectory controllers to be ready."""
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory

        log_info("Waiting for trajectory controllers...")

        # Controllers to check
        controllers = [
            '/left_joint_trajectory_controller/follow_joint_trajectory',
        ]

        for controller in controllers:
            client = ActionClient(self.node, FollowJointTrajectory, controller)
            if not client.wait_for_server(timeout_sec=timeout):
                log_error(f"Controller {controller} not available after {timeout}s")
                raise RuntimeError(f"Controller {controller} not available")
            log_info(f"Controller {controller} is ready")
            client.destroy()

    def _plan_gripper(self, planning_component):
        group_name = planning_component.planning_group_name
        log_info(f"Gripper planning for {group_name}, using joint space planning only")

        valid_trajectory = planning_component.plan(
            single_plan_parameters=self.plan_params
        )

        if not valid_trajectory:
            log_error(f"Gripper planning failed for {group_name}")
            return None

        return valid_trajectory

    def _plan_cartesian(self, planning_component, max_iterations=TRY_PLANNING_MAX_ITER):
        group_name = planning_component.planning_group_name

        for iteration in range(max_iterations):
            log_info(f"Trying cartesian planning for {group_name}, iteration {iteration + 1}")
            valid_trajectory = planning_component.plan(
                single_plan_parameters=self.lin_plan_params
            )

            if valid_trajectory:
                log_info(f"Cartesian planning succeeded for {group_name} after {iteration + 1} iterations")
                return valid_trajectory

        return None

    def _plan_joint_space(self, planning_component):
        group_name = planning_component.planning_group_name
        log_info(f"Trying joint space planning for {group_name}")

        valid_trajectory = planning_component.plan(
            single_plan_parameters=self.plan_params
        )

        if not valid_trajectory:
            log_error(f"Joint space planning failed for {group_name}")
            return None

        return valid_trajectory

    def _plan(self, planning_component, cartesian_only):
        group_name = planning_component.planning_group_name
        log_info(f"Planning trajectory for {group_name}")

        if "gripper" in group_name:
            return self._plan_gripper(planning_component)

        valid_trajectory = self._plan_cartesian(planning_component)

        if valid_trajectory:
            return valid_trajectory

        if cartesian_only:
            log_error(f"cartesian_only is True, returning None")
            return None

        return self._plan_joint_space(planning_component)

    def _execute(self, robot, robot_trajectory, group_name, sleep_time=0.0):
        """Helper function to execute a planned trajectory."""
        log_info(f"Executing plan for {group_name}")
        success = robot.execute(robot_trajectory, controllers=[])

        time.sleep(sleep_time)

        if not success:
            log_error(f"Execution of plan for {group_name} failed")
            return False

        log_info(f"Plan execution for {group_name} succeeded")
        return success

    def _plan_and_execute(self, robot, planning_component, cartesian_only, sleep_time=0.0):
        """Helper function to plan and execute a motion."""
        valid_trajectory = self._plan(planning_component, cartesian_only)
        if not valid_trajectory:
            return False

        group_name = planning_component.planning_group_name
        robot_trajectory = valid_trajectory.trajectory
        success = self._execute(robot, robot_trajectory, group_name, sleep_time)

        return success

    def eef_pose(self):
        eef_tf = self._eef_tf_recorder.record_tf().transform
        eef_pose = [np.array([
            eef_tf.translation.x,
            eef_tf.translation.y,
            eef_tf.translation.z
        ]),
            np.array([
                eef_tf.rotation.x,
                eef_tf.rotation.y,
                eef_tf.rotation.z,
                eef_tf.rotation.w
            ]),
            self.gripper_state
        ]

        log_info(
            f"current EEF: pos={eef_pose[0]}, quat={eef_pose[1]}, gripper_open={bool(eef_pose[2])}"
        )
        return eef_pose

    def joints_state(self):
        return self.joints_state_recorder.record_state()

    def links_pose(self):
        links_poses = {}
        for link_name, link_tf_recorder in self._links_tf_recorder.items():
            link_tf = link_tf_recorder.record_tf().transform
            link_pose = np.array([
                link_tf.translation.x,
                link_tf.translation.y,
                link_tf.translation.z,
                link_tf.rotation.x,
                link_tf.rotation.y,
                link_tf.rotation.z,
                link_tf.rotation.w
            ])

            links_poses[link_name] = link_pose
        return links_poses


    def _limit_position(self, position):
        new_position = []
        for i, coord in enumerate(position):
            new_coord = min(max(coord, self.workspace[0][i]), self.workspace[1][i])
            new_position.append(new_coord)

        if not np.array_equal(position, new_position):
            log_info(f"Safety _limit_position() function called and limited  the pos to: {new_position}")
        return new_position

    def go_to_pose(self, action, cartesian_only=False):
        log_info(f"target action: \n {action}")

        target_pos   = self._limit_position(action[:3])
        target_quat  = action[3:7]
        target_grip = action[7]

        self.left_arm.set_start_state_to_current_state()
        goal = PoseStamped()
        goal.header.frame_id = ROBOT_BASE_FRAME
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = map(float, target_pos)
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = map(float,
                                                                                                                 target_quat)
        self.left_arm.set_goal_state(pose_stamped_msg=goal, pose_link="left_gripper_grasp_frame")
        success_arm = self._plan_and_execute(
            self.ur,
            self.left_arm,
            cartesian_only,
            sleep_time=3.0
        )

        # Gripper
        # target_grip_str = "open" if target_grip == 0 else "close"
        # self.left_gripper.set_goal_state(configuration_name=target_grip_str)
        # if self.use_sim_time:
        #     success_gripper = True
        #     log_info("Skipping gripper execution in simulation")
        # else:
        #     success_gripper = self._plan_and_execute(
        #         self.ur,
        #         self.left_gripper,
        #         cartesian_only=False,
        #         sleep_time=3.0
        #     )
        # self.gripper_state = 0 if target_grip_str == "open" else 1
        success_gripper = True

        return success_arm and success_gripper

    def reset(self):
        log_info("resetting Robot")
        success = self.go_to_pose(DEFAULT_ROBOT_ACTION)
        if not success:
            log_error("Moving the robot failed during reset")
            raise RuntimeError("Moving the robot failed")

        return success

    def _joint_space_plan(self, target_pos, target_quat, gripper_state, cartesian_only):
        """
        Default joint-space plan+execute for arm and gripper.
        """
