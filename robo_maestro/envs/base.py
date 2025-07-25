"""
Base Gymnasium environment for RoboMaestro.
At its core:
# reset(): -> obs
sets the environment in an initial state, beginning an episode

# step(action) -> (obs, reward, terminated, truncated, info)
executes a selected action in the environment, and moves the simulation forward
we deal with a real robot, so success is always False and reward is always 0
limited by Truncation (max steps limit)

# _get_obs() -> obs
visualizes the current state of the environment
"""
import gymnasium as gym
import rclpy

from robo_maestro.core.robot import Robot
from robo_maestro.core.tf import *
from robo_maestro.utils.constants import *
from robo_maestro.utils.helpers import *


class BaseEnv(gym.Env):
    def __init__(self, cam_list: list[str], node=None, use_sim_time=False):
        # Accept node and use_sim_time parameters
        self.node = node
        self.use_sim_time = use_sim_time

        # Robot init - pass node and use_sim_time
        self.robot = Robot(WORKSPACE["left"], cam_list, node, use_sim_time)

        # Camera init
        self.cam_list = cam_list
        self.cam_info = {}
        for cam_name in cam_list:
            if cam_name in CAM_INFO:
                self.cam_info[f"info_{cam_name}"] = CAM_INFO[cam_name]
            self.cam_info[f"intrinsics_{cam_name}"] = (
                self.robot.cameras[cam_name].intrinsics)

    def reset(self):
        print("Returning to home config")
        # might need to stop_current_movement(), to see later

        success = self.robot.go_to_pose(DEFAULT_ROBOT_ACTION)

        if not success:
            raise RuntimeError("Moving the robot to default position failed")

        obs = self._get_obs()

        return obs, {}

    def step(self, action):
        success = self.robot.go_to_pose(action)

        if not success:
            raise RuntimeError("Moving the robot failed")

        obs = self._get_obs()
        return (
            obs,
            0,
            False,
            False,
            {}
        )

    def _get_obs(self, sync_record=False):
        obs = {}

        gripper_pose = self.robot.eef_pose()
        obs["gripper_pos"] = np.array(gripper_pose[0])
        obs["gripper_quat"] = np.array(gripper_pose[1])
        obs["gripper_state"] = np.array(gripper_pose[2])

        # Sensors
        for cam_name in self.robot.cam_list:
            cam = self.robot.cameras[cam_name]

            obs[f"rgb_{cam_name}"], cam_pose = cam.record_image(dtype=np.uint8)

            if f"info_{cam_name}" in self.cam_info:
                obs[f"info_{cam_name}"] = self.cam_info[f"info_{cam_name}"]
                obs[f"info_{cam_name}"]["pos"] = CAM_INFO[cam_name]["pos"]
                obs[f"info_{cam_name}"]["euler"] = CAM_INFO[cam_name]["euler"]
            else:
                obs[f"info_{cam_name}"] = dict()
                obs[f"info_{cam_name}"]["pos"] = cam_pose[0]
                obs[f"info_{cam_name}"]["euler"] = cam_pose[1]

            # depth
            depth_cam = self.robot.depth_cameras[cam_name]
            depth, _ = depth_cam.record_image(dtype=np.uint16)
            depth = depth.astype(np.float32).squeeze(-1) / 1000
            obs[f"depth_{cam_name}"] = depth

            # point cloud
            info_cam = obs[f"info_{cam_name}"]
            cam_pos = info_cam["pos"]
            cam_euler = info_cam["euler"]
            world_T_cam = pos_euler_to_hom(cam_pos, cam_euler)
            intrinsics = self.cam_info[f"intrinsics_{cam_name}"]
            pcd = depth_to_pcd(depth, intrinsics)
            pcd = transform_pcd(pcd, world_T_cam)
            obs[f"pcd_{cam_name}"] = pcd

            # gripper attention
            K = self.cam_info[f"intrinsics_{cam_name}"]["K"]
            gr_px = project(gripper_pose[:2], np.linalg.inv(world_T_cam), K)
            gr_x, gr_y = gr_px[0], gr_px[1]
            obs[f"gripper_uv_{cam_name}"] = [gr_x, gr_y]

        obs["robot_info"] = self.robot.links_pose()
        return obs


gym.register(
    id="RealRobot-BaseEnv",
    entry_point=BaseEnv,
    max_episode_steps=300,
)
