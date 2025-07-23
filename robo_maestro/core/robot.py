import rclpy
from rclpy.logging import get_logger
from geometry_msgs.msg import PoseStamped
from moveit.planning import MoveItPy
import numpy as np
from robo_maestro.core.observer import TFRecorder, JointStateRecorder, CameraPose
from robo_maestro.utils.constants import *
import time


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
        success = robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)

    return success


class Robot:
    def __init__(self, workspace, cam_list):
        arm = 'left'
        self.eef_frame = EEF_FRAME[arm]
        self.workspace = workspace
        camera_and_tf_recorder_node = rclpy.create_node("camera_and_tf_recorder")

        self._eef_tf_recorder = TFRecorder(camera_and_tf_recorder_node, ROBOT_BASE_FRAME, self.eef_frame)
        self._links_tf_recorder = {link: TFRecorder(camera_and_tf_recorder_node, ROBOT_BASE_FRAME, link) for link in ROBOT_LINKS[arm]}
        self.joints_state_recorder = JointStateRecorder(camera_and_tf_recorder_node)

        # Cameras
        self.cam_list = cam_list
        self.cameras = {}
        self.depth_cameras = {}

        for cam_name in cam_list:
            self.cameras[cam_name] = CameraPose(
                node=camera_and_tf_recorder_node,
                topic=f"/{cam_name}/color/image_raw",
                camera_frame=CAM_TF_TOPIC[cam_name]
            )
            self.depth_cameras[cam_name] = CameraPose(
                node=camera_and_tf_recorder_node,
                topic=f"{cam_name}/aligned_depth_to_color/image_raw",
                camera_frame=CAM_TF_TOPIC[cam_name]
            )

        # robot controller
        self.ur = MoveItPy(node_name="moveit_py")
        self.left_arm = self.ur.get_planning_component("left_arm")
        self.logger = get_logger("run_policy")

        self.go_to_pose(DEFAULT_ROBOT_ACTION)
        self.gripper_state = 0


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
        return new_position

    def go_to_pose(self, action):
        gripper_pos = action[:3]
        gripper_pos = self._limit_position(gripper_pos)

        gripper_quat = action[3:7]

        self.left_arm.set_start_state_to_current_state()
        left_pose_goal = PoseStamped()
        left_pose_goal.header.frame_id = "base_link"
        left_pose_goal.pose.position.x = gripper_pos[0]
        left_pose_goal.pose.position.y = gripper_pos[1]
        left_pose_goal.pose.position.z = gripper_pos[2]
        left_pose_goal.pose.orientation.x = gripper_quat[0]
        left_pose_goal.pose.orientation.y = gripper_quat[1]
        left_pose_goal.pose.orientation.z = gripper_quat[2]
        left_pose_goal.pose.orientation.w = gripper_quat[3]
        self.left_arm.set_goal_state(pose_stamped_msg=left_pose_goal, pose_link="left_tool0")

        success = plan_and_execute(self.ur, self.left_arm, self.logger, sleep_time=3.0)
        self.gripper_state = action[7]

        # TODO: add safety checks against jumps in the trajectory

        return success
