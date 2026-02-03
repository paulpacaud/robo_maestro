from types import SimpleNamespace
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.robot_state import RobotState
from ur_msgs.srv import SetIO
import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger
from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
import numpy as np
from robo_maestro.core.observer import TFRecorder, JointStateRecorder, CameraPose
from robo_maestro.utils.constants import *
import time

from robo_maestro.utils.logger import log_info, log_error, log_warn, log_success


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

        # Gripper Subscription
        self.gripper_client = self.node.create_client(
            SetIO, "/left_io_and_status_controller/set_io"
        )

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
                topic=f"/camera/{cam_name}/depth/image_raw",
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
        log_success("Robot initialized with workspace")

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
            log_success(f"Controller {controller} is ready")
            client.destroy()

    def open_gripper(self, low_force_mode=False, wait=True):  # OK
        """Open the gripper.

        Keyword Arguments:
            low_force_mode {bool} -- use the low force mode (5N instead of 40N) (default: {False})
            wait {bool} -- wait until the end of the movement (default: {False})

        Raises:
            ROSException: timeout exception
        """
        log_info('Opening gripper...')
        self.gripper_move(0, low_force_mode, wait)
        self.gripper_state = 0

    def close_gripper(self, low_force_mode=False, wait=True):  # OK
        """Close the gripper.

        Keyword Arguments:
            low_force_mode {bool} -- use the low force mode (5N instead of 40N) (default: {False})
            wait {bool} -- wait until the end of the movement (default: {False})

        Raises:
            ROSException: timeout exception
        """
        log_info('Closing gripper...')
        self.gripper_move(1, low_force_mode, wait)
        self.gripper_state = 1

    def gripper_move(self, target, low_force_mode=False, wait=True):  # OK
        self._set_digital_out(1, 17, 1 if low_force_mode else 0)
        self._set_digital_out(1, 16, target)
        time.sleep(6) # the low-level controller does not wait for the gripper to reach the target position, so we need to wait here
        log_success(f'Gripper moved to position {target} with low force mode {low_force_mode}.')

    def _set_digital_out(self, fun, pin, state):  # OK
        req = SetIO.Request()
        req.fun = fun
        req.pin = pin
        req.state = float(state)
        self.future = self.gripper_client.call_async(req)
        log_info("sending request to set digital out")
        return self.future.result()

    def _execute_gripper(self, state):
        if state=="open":
            self.open_gripper()
        else:
            self.close_gripper()


    def _plan_cartesian(self, planning_component, max_iterations=TRY_PLANNING_MAX_ITER):
        group_name = planning_component.planning_group_name

        for iteration in range(max_iterations):
            log_info(f"Trying cartesian planning for {group_name}, iteration {iteration + 1}")
            valid_trajectory = planning_component.plan(
                single_plan_parameters=self.lin_plan_params
            )

            if valid_trajectory:
                log_success(f"Cartesian planning succeeded for {group_name} after {iteration + 1} iterations")
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

        log_success(f"Plan execution for {group_name} succeeded")
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
            f"current EEF: pos={eef_pose[0]}, quat={eef_pose[1]}, gripper={eef_pose[2]}"
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
            log_warn(f"Safety _limit_position() function called and limited the pos to: {new_position}")
        return new_position

    def go_to_pose(self, action, cartesian_only=False, sleep_time=3.0):
        """
        For the arm, we will first try cartesian planning, and fall back to vanilla RTT motion planner if it fails.
        For the gripper, we will just execute the open/close command (skipped if already in target state).
        """
        log_info(f"target action: {', '.join(map(str, action))}")

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
            sleep_time=sleep_time
        )

        # Gripper — only actuate if state changed
        target_grip_str = "open" if target_grip == 0 else "close"
        target_grip_int = 0 if target_grip_str == "open" else 1
        if target_grip_int == self.gripper_state:
            success_gripper = True
        elif self.use_sim_time:
            success_gripper = True
            log_info("Skipping gripper execution in simulation")
        else:
            self._execute_gripper(target_grip_str)
            success_gripper = True
        self.gripper_state = target_grip_int

        return success_arm and success_gripper

    def reset(self):
        log_info("resetting Robot")
        success = self.go_to_pose(DEFAULT_ROBOT_ACTION)
        if not success:
            log_error("Moving the robot failed during reset")
            raise RuntimeError("Moving the robot failed")

        log_success("Robot reset successfully")
        return success