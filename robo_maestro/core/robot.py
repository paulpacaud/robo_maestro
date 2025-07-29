from types import SimpleNamespace
from moveit.core.robot_trajectory import RobotTrajectory
from moveit.core.robot_state import RobotState

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetCartesianPath
from rclpy.logging import get_logger
from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy
import numpy as np
from robo_maestro.core.observer import TFRecorder, JointStateRecorder, CameraPose
from robo_maestro.utils.constants import *
import time

from robo_maestro.utils.logger import log_info, log_error, log_warn

class CartesianClient(Node):
    def __init__(self):
        super().__init__("cartesian_client")
        self.cli = self.create_client(GetCartesianPath,
                                      "/compute_cartesian_path")
        self.cli.wait_for_service()

    def plan(self, group: str, waypoints: list[Pose],
             eef_step: float = 0.01, jump_thresh: float = 0.0):
        req = GetCartesianPath.Request()
        req.group_name      = group
        req.header.frame_id = "prl_ur5_base"
        req.waypoints       = waypoints
        req.max_step        = eef_step
        req.jump_threshold  = jump_thresh
        req.avoid_collisions = True     # or False
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.error_code.val == resp.error_code.SUCCESS and resp.fraction > 0.0:
            return resp.solution            # moveit_msgs/RobotTrajectory
        else:
            log_error(f"Cartesian planning failed")
            raise RuntimeError(f"Cartesian planning failed: "
                               f"{resp.error_code.val}, fraction={resp.fraction}")


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
        self.left_arm = self.ur.get_planning_component("left_arm")
        self.left_gripper = self.ur.get_planning_component("left_gripper")

        self.gripper_state = 0
        log_info("Robot initialized with workspace")
        self.cartesian_client = CartesianClient()
        self.reset()

    def _plan_and_execute(
            self,
            robot,
            planning_component,
            sleep_time=0.0,
    ):
        """Helper function to plan and execute a motion."""
        group_name = planning_component.planning_group_name
        log_info(f"Planning trajectory for {group_name}")
        plan_result = planning_component.plan()

        # execute the plan if it is valid trajectory
        if plan_result:
            log_info(f"Executing plan for {group_name}")
            robot_trajectory = plan_result.trajectory
            success = robot.execute(robot_trajectory, controllers=[])
        else:
            log_error(f"Planning failed for {group_name}")
            success = False

        time.sleep(sleep_time)

        log_info(f"Plan execution for {group_name} {'succeeded' if success else 'failed'}")
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

        if position != new_position:
            log_info(f"Safety _limit_position() function called and limited  the pos to: {new_position}")
        return new_position

    def go_to_pose(self, action, cartesian=False):
        """
        Move to `action`. If `cartesian=True`, force straight-line Cartesian path.
        """
        log_info(f"target action: \n {action}")

        target_pos   = self._limit_position(action[:3])
        target_quat  = action[3:7]

        # if cartesian:
        #     # success = self._joint_space_plan(target_pos, target_quat)
        #     plan = self._cartesian_plan(target_pos, target_quat)
        #     if plan:
        #         log_info("Executing Cartesian path")
        #         success = self.ur.execute(plan.trajectory, controllers=[])
        #     else:
        #         log_warn("Cartesian planning failed; falling back to joint-space")
        #         success = self._joint_space_plan(target_pos, target_quat)
        # else:
        #     success = self._joint_space_plan(target_pos, target_quat)
        success = self._joint_space_plan(target_pos, target_quat)

        return success

    def reset(self):
        log_info("resetting Robot")
        success = self.go_to_pose(DEFAULT_ROBOT_ACTION, cartesian=True)
        if not success:
            log_error("Moving the robot failed during reset")
            raise RuntimeError("Moving the robot failed")

        return success

    def _cartesian_plan(self, target_pos, target_quat):
        """Return a PlanSolution‑like shim whose .trajectory
           is a moveit.core.robot_trajectory.RobotTrajectory."""
        wp = Pose()
        wp.position.x, wp.position.y, wp.position.z = map(float, target_pos)
        wp.orientation.x, wp.orientation.y, wp.orientation.z, wp.orientation.w = map(float, target_quat)

        traj_msg = self.cartesian_client.plan("left_arm", [wp])

        robot_model = self.ur.get_robot_model()
        traj_core = RobotTrajectory(robot_model)
        current_state = self.ur.get_current_state()  # RobotState
        traj_core.set_robot_trajectory_msg(current_state, traj_msg)  #  copy in

        return SimpleNamespace(successful=True, trajectory=traj_core)

    def _joint_space_plan(self, target_pos, target_quat):
        """
        Default joint-space plan+execute for arm and gripper.
        """
        # Arm
        self.left_arm.set_start_state_to_current_state()
        goal = PoseStamped()
        goal.header.frame_id = ROBOT_BASE_FRAME
        goal.pose.position.x, goal.pose.position.y, goal.pose.position.z = map(float, target_pos)
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = map(float,
                                                                                                                 target_quat)
        self.left_arm.set_goal_state(pose_stamped_msg=goal, pose_link="left_tool0")
        success_arm = self._plan_and_execute(self.ur, self.left_arm, sleep_time=3.0)

        # Gripper
        gripper_state = "open" if self.gripper_state == 0 else "close"
        self.left_gripper.set_goal_state(configuration_name=gripper_state)
        if self.use_sim_time:
            success_gripper = True
            log_info("Skipping gripper execution in simulation")
        else:
            success_gripper = self._plan_and_execute(self.ur, self.left_gripper, sleep_time=3.0)
        self.gripper_state = 1 if gripper_state == "open" else 0

        return success_arm and success_gripper