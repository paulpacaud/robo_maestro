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
from robo_maestro.utils.logger import log_info, log_warn, log_error, log_debug, log_success
import rclpy
from rclpy.node import Node
from easydict import EasyDict
import numpy as np
import msgpack
import requests
import msgpack_numpy
msgpack_numpy.patch()

class Arguments(tap.Tap):
    cam_list: list[str] = ["bravo_camera","charlie_camera","alpha_camera"]
    arm: str = "left"
    taskvar: str = "real_put_fruit_in_box+0"
    ip: str = "127.0.0.1"
    port: int = 8002
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
    def __init__(self, server_addr, save_path):
        self.server_addr = server_addr
        self.save_path = save_path

    def predict(self, batch):
        log_info(f"Policy server at {self.server_addr}/predict called with batch {batch.keys()}")

        # 1) pack the batch
        data = msgpack_numpy.packb(batch)

        # 2) save it to disk
        save_batch_dir = os.path.join(self.save_path, "batch")
        os.makedirs(save_batch_dir, exist_ok=True)
        filename = os.path.join(
            save_batch_dir,
            f"step-{batch['step_id']}.msgpack"
        )
        with open(filename, "wb") as f:
            f.write(data)
        log_info(f"Saved batch to {filename}")

        # 3) send it on its way
        response = requests.post(f"{self.server_addr}/predict", data=data)
        output = msgpack_numpy.unpackb(response._content)
        action = output['action']
        cache = output['cache']

        log_info(f"Received action of shape {action.shape} and cache from server")

        self._display_cache(action, cache)
        return action, cache

    def mock_predict(self, batch, step_id):
        """
        Mock prediction function to simulate server response.
        """
        log_info(f"Mock prediction called with batch {batch.keys()}")

        # randomly pick MOCK_ROBOT_ACTION_1 or MOCK_ROBOT_ACTION_2
        if step_id % 2 == 0:
            action = np.array(MOCK_ROBOT_ACTION_1, dtype=np.float32)
        else:
            action = np.array(MOCK_ROBOT_ACTION_2, dtype=np.float32)

        cache = {}

        self._display_cache(action, cache)
        return action, cache

    def _display_cache(self, action, cache):
        log_info(f"cache['last_plan']: {cache.get('last_plan', 'N/A')}")
        log_info(f"highlevel_step_id: {cache.get('highlevel_stepid', 'N/A')}")
        log_info(f"highlevel_plans: {cache.get('highlevel_plans', 'N/A')}")


class TaskEvaluator:
    def __init__(self, env, task, variation, instructions, episode_id, links_bbox, cam_list, server_addr):
        self.env = env
        self.task = task
        self.variation = variation
        self.instructions = instructions
        self.episode_id = episode_id
        self.links_bbox = links_bbox
        self.cam_list = cam_list
        self.save_path = os.path.join(DATA_DIR, 'run_policy_experiments', self.task + "+" + self.variation, f"episode_{self.episode_id}")

        self.policy_server = PolicyServer(server_addr, self.save_path)

    def execute_step(self, step_id, keystep_real, cache):
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

        #action, cache = self.policy_server.predict(batch)
        action, cache = self.policy_server.mock_predict(batch, step_id)

        keystep_real["action"] = action

        save_steps_dir = os.path.join(self.save_path, "keysteps")
        os.makedirs(save_steps_dir, exist_ok=True)
        np.save(os.path.join(save_steps_dir, f"{step_id}.npy"), keystep_real)

        if not rclpy.ok():
            return None
        obs, _, _, _, _ = self.env.step(action)

        keystep_real = process_keystep(
            obs, 
            links_bbox=self.links_bbox,
            cam_list=self.cam_list,
        )

        if not rclpy.ok():
            return None
        
        return keystep_real, cache


class RunPolicyNode(Node):
    def __init__(self):
        super().__init__(
            'run_policy_node',
             allow_undeclared_parameters=True,
             automatically_declare_parameters_from_overrides=True
        )

        self.args = Arguments().parse_args(known_only=True)
        self.setup_environment()

    def setup_environment(self):
        # Load task configurations
        taskvars_instructions = json.load(open(
            os.path.join(get_package_share_directory('robo_maestro'),
                         'assets', 'taskvars_instructions.json')
        ))

        self.task, self.variation = self.args.taskvar.split("+")
        self.instructions = taskvars_instructions[self.args.taskvar]

        # Load bbox info
        links_bbox_file_path = os.path.join(
            get_package_share_directory('robo_maestro'),
            'assets', 'real_robot_bbox_info.pkl'
        )
        with open(links_bbox_file_path, "rb") as f:
            self.links_bbox = pkl.load(f)

        # Create environment
        use_sim_time = self.get_parameter('use_sim_time').value
        env = gym.make('RealRobot-BaseEnv',
                            cam_list=self.args.cam_list,
                            node=self,
                            use_sim_time=use_sim_time,
                            disable_env_checker = True, # skip need for defining action and observation spaces
                            )
        self.env = env.unwrapped if hasattr(env, 'unwrapped') else env

        # Initialize evaluator
        self.evaluator = TaskEvaluator(
            self.env, self.task, self.variation,
            self.instructions, self.args.episode_id,
            self.links_bbox, self.args.cam_list,
            self.args.server_addr
        )

    def run(self):
        obs, info = self.env.reset()

        keystep_real = process_keystep(
            obs,
            links_bbox=self.links_bbox,
            cam_list=self.args.cam_list,
        )

        cache = None
        for step_id in range(MAX_STEPS):
            log_info(f"Step {step_id}/{MAX_STEPS-1}")
            if not rclpy.ok():
                break

            rclpy.spin_once(self, timeout_sec=0.0)

            keystep_real, cache = self.evaluator.execute_step(
                step_id, keystep_real, cache
            )
            if keystep_real is None or cache is None:
                log_error("Environment step failed, keystep_real is None or cache is None, exiting.")
                break

        log_success(f"Last step completed, max steps reached")


def main():
    rclpy.init()
    node = RunPolicyNode()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()