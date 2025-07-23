"""
bridge between a remote policy inference server (genrobot3d) and the Paris‑Lab UR5, wrapping all real‑robot concerns inside RoboMaestro’s environment stack.
1) defines a gym environment for the real robot: Gym -> BaseEnv -> Robot
2) get obs from the env
3) preprocess the obs to match the input of the remote policy inference server
4) query the policy over http and get the action + new cache
5) execute the action on the real robot with env.move()
"""

import tap
import gymnasium as gym
import json
import os
import robo_maestro.envs
from ament_index_python.packages import get_package_share_directory
from robo_maestro.utils.helpers import *
from robo_maestro.utils.constants import *
import rclpy
from easydict import EasyDict
import numpy as np
from copy import deepcopy
import msgpack
import requests
import msgpack_numpy
msgpack_numpy.patch()

class Arguments(tap.Tap):
    device: str = "cuda"
    num_demos: int = 10
    image_size: int = 256
    cam_list: list[str] = ["bravo_camera","charlie_camera","alpha_camera"]
    arm: str = "left"
    env_name: str = "RealRobot-Pick-v0"
    arch: str = "ptv3"
    save_obs_outs_dir: str = None
    checkpoint: str = None
    taskvar: str = "real_push_buttons+0"
    instr: str = None
    use_sem_ft: bool = False
    ip: str = "127.0.0.1"
    port: int = 8001
    episode_id: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._server_addr = None

    @property
    def server_addr(self):
        if self._server_addr is None:
            self._server_addr = f"http://{self.ip}:{self.port}"
        return self._server_addr


class PolicyServer:
    def __init__(self, server_addr):
        self.server_addr = server_addr

    def predict(self, batch):
        data = msgpack_numpy.packb(batch)
        response = requests.post(f"{self.server_addr}/predict", data=data)
        output = msgpack_numpy.unpackb(response._content)
        action = output['action']
        cache = output['cache']

        print('action', action)
        print("cache:", cache)

        return action, cache

    def mock_predict(self, batch):
        """
        Mock prediction function to simulate server response.
        """
        print("Mock prediction called with batch:", batch)

        # Simulate a slight change from DEFAULT_ROBOT_ACTION
        action = np.array(DEFAULT_ROBOT_ACTION, dtype=np.float32)
        action[0] += 0.05
        action[1] += 0.05
        action[2] += 0.05
        action[-1] = 1 if action[-1] == 0 else 0
        cache = {}

        print('action', action)
        print("cache:", cache)

        return action, cache

class TaskEvaluator:
    def __init__(self, env, task, variation, instructions, episode_id, links_bbox, cam_list, image_size, server_addr):
        self.env = env
        self.task = task
        self.variation = variation
        self.instructions = instructions
        self.episode_id = episode_id
        self.links_bbox = links_bbox
        self.cam_list = cam_list
        self.image_size = image_size
        self.save_path = os.path.join(DATA_DIR, 'run_policy_experiments', self.task + "+" + self.variation, f"episode_{self.episode_id}")
        os.makedirs(self.save_path, exist_ok=True)

        self.policy_server = PolicyServer(server_addr)

    def execute_step(self, step_id, obs, keystep_real, cache):
        if not rclpy.ok():
            return None
            
        if step_id == 0:
            cache = None

        if cache is not None and isinstance(cache, dict):
            cache = EasyDict(cache)

        batch = {
            'task_str': self.task,
            'variation': self.variation,
            'step_id': step_id,
            'obs_state_dict': keystep_real,
            'episode_id': self.episode_id,
            'instructions': self.instructions,
            'cache': cache,
        }

        action, cache = self.policy_server.predict(batch)

        print('action', action)
        print("cache:", cache)

        keystep_real["action"] = action

        np.save(os.path.join(self.save_path, f"{step_id}.npy"), keystep_real)

        pos = deepcopy(action[:3]).astype(np.double)
        quat = deepcopy(action[3:7]).astype(np.double)
        open_gripper = action[7] >= 0.5

        print('action', action)
        print("Predicting step:", step_id, f"Open gripper {open_gripper}")

        print("Position:", pos)
        print("Quaternion:", quat)
        print("Open gripper:", open_gripper)
        print("Rotation:", quat_to_euler(quat, True))
        if not rclpy.ok():
            return None
        obs, _, _, _, _ = self.env.step([pos, quat, open_gripper])

        keystep_real = process_keystep(
            obs, 
            links_bbox=self.links_bbox,
            cam_list=self.cam_list,
            crop_size=self.image_size
        )

        if not rclpy.ok():
            return None
        
        return keystep_real, cache

def main():
    args = Arguments().parse_args(known_only=True)

    taskvars_instructions = json.load(open(
        os.path.join(CODE_DIR, 'assets/taskvars_instructions.json')
    ))
    task, variation = args.taskvar.split("+")
    instructions = taskvars_instructions[args.taskvar]
    links_bbox_file_path = os.path.join(CODE_DIR, 'assets/real_robot_bbox_info.pkl')
    with open(links_bbox_file_path, "rb") as f:
        links_bbox = pkl.load(f)

    env = gym.make('RealRobot-BaseEnv',
       cam_list=args.cam_list
    )

    obs, info = env.reset()

    keystep_real = process_keystep(
        obs,
        links_bbox=links_bbox,
        cam_list=args.cam_list,
        crop_size=args.image_size
    )
    evaluator = TaskEvaluator(env, task, variation, instructions, args.episode_id, links_bbox, args.cam_list, args.image_size, args.server_addr)
    cache = None

    for step_id in range(MAX_STEPS):
        keystep_real, cache = evaluator.execute_step(step_id, obs, keystep_real, cache)
        if keystep_real is None:
            break


if __name__ == "__main__":
    try:
        rclpy.init()
        main()
        rclpy.shutdown()
    except KeyboardInterrupt:
        rclpy.shutdown()