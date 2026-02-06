"""
bridge between a remote policy inference server (genrobot3d) and the Paris‑Lab UR5, wrapping all real‑robot concerns inside RoboMaestro’s environment stack.
1) defines a gym environment for the real robot: Gym -> BaseEnv -> Robot
2) get obs from the env
3) preprocess the obs to match the input of the remote policy inference server
4) query the policy over http and get the action + new cache
5) execute the action on the real robot with env.move()

connect to cleps:
ssh -N -v -L 18000:gpu017:18000 cleps -i ~/.ssh/jz_rsa

on the docker:
ros2 launch robo_maestro run_policy_pointact.launch.py use_sim_time:=false taskvar:="ur5_put_grapes_and_banana_in_plates" port:=18000
"""

import tap
import gymnasium as gym
import json
import os
import robo_maestro.envs
from ament_index_python.packages import get_package_share_directory
from robo_maestro.schemas import ObsStateDict
from robo_maestro.utils.helpers import *
from robo_maestro.utils.constants import *
from robo_maestro.utils.logger import (
    log_info,
    log_warn,
    log_error,
    log_debug,
    log_success,
)
import rclpy
from rclpy.node import Node
from easydict import EasyDict
import numpy as np
import msgpack
import requests
import msgpack_numpy

msgpack_numpy.patch()

from robo_maestro.utils.server_client import PolicyClient


class Arguments(tap.Tap):
    cam_list: list[str] = [
        "echo_camera",
    ]  # ["echo_camera","foxtrot_camera","golf_camera"]
    arm: str = "left"
    taskvar: str = "ur5_put_grapes_and_banana_in_plates"
    ip: str = "127.0.0.1"
    port: int = 17000
    episode_id: int = 0
    mock: bool = False


class TaskEvaluator:
    def __init__(
        self,
        env,
        task,
        variation,
        instructions,
        episode_id,
        links_bbox,
        cam_list,
        ip,
        port,
        mock=False,
    ):
        self.env = env
        self.task = task
        self.variation = variation
        self.instructions = instructions
        self.episode_id = episode_id
        self.links_bbox = links_bbox
        self.cam_list = cam_list
        self.mock = mock
        ts = __import__("datetime").datetime.now().strftime("%m%dT%H%M%S")
        self.save_path = os.path.join(
            DATA_DIR,
            "run_policy_experiments",
            self.task + "+" + self.variation + "_" + ts,
            f"episode_{self.episode_id}",
        )

        if self.mock:
            log_info("Running in MOCK mode — no policy server required")
        else:
            self.policy_client = PolicyClient(ip, port)
            is_server_running = False
            while not is_server_running:
                is_server_running = self.policy_client.ping()
            print(f"Server is running on host {ip} port {port}")

    def mock_predict(self, batch, step_id):
        """
        Mock prediction function to simulate server response.
        """
        log_info(f"Mock prediction called with batch {batch.keys()}")

        if step_id % 2 == 0:
            action = np.array(MOCK_ROBOT_ACTION_1, dtype=np.float32)
        else:
            action = np.array(MOCK_ROBOT_ACTION_2, dtype=np.float32)

        cache = {}
        return action, cache

    def execute_step(self, step_id: int, keystep_real: ObsStateDict, cache):
        if not rclpy.ok():
            return None

        if step_id == 0:
            cache = None

        if cache is not None and isinstance(cache, dict):
            cache = EasyDict(cache)

        batch = {
            "task_str": self.task,
            "variation": self.variation,
            "step_id": step_id,
            "obs_state_dict": keystep_real.model_dump(),
            "episode_id": self.episode_id,
            "instructions": self.instructions,
            "cache": cache,
        }

        if self.mock:
            action, cache = self.mock_predict(batch, step_id)
        else:
            options = {
                "pred_rot_type": "euler",
            }
            outputs = self.policy_client.get_action(batch, options)
            action = np.array(outputs["action"], dtype=np.float64).flatten()[:8]

        # Save batch to log dir
        batch_data = msgpack_numpy.packb(batch)
        batch_dir = os.path.join(self.save_path, "batch")
        os.makedirs(batch_dir, exist_ok=True)
        batch_file = os.path.join(batch_dir, f"step-{step_id}.msgpack")
        with open(batch_file, "wb") as f:
            f.write(batch_data)
        log_info(f"Batch: {batch.keys()}")
        log_info(
            f"Task: {self.task}, Variation: {self.variation}, Step ID: {step_id}, Episode ID: {self.episode_id}"
        )
        log_info(f"Instruction: {self.instructions}")
        log_info(f"Saved batch to {batch_file}")

        # Save keystep with the predicted action
        keystep_with_action = keystep_real.model_dump()
        keystep_with_action["action"] = action
        save_steps_dir = os.path.join(self.save_path, "keysteps")
        os.makedirs(save_steps_dir, exist_ok=True)
        np.save(os.path.join(save_steps_dir, f"{step_id}.npy"), keystep_with_action)

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
            "run_policy_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.args = Arguments().parse_args(known_only=True)
        self.setup_environment()

    def setup_environment(self):
        # Load task configurations
        taskvars_instructions = json.load(
            open(
                os.path.join(
                    get_package_share_directory("robo_maestro"),
                    "assets",
                    "taskvars_instructions.json",
                )
            )
        )

        log_info(f"Keys in taskvars_instructions: {list(taskvars_instructions.keys())}")
        log_info(f"Using taskvar: {self.args.taskvar}")

        # self.task, self.variation = self.args.taskvar.split("+")
        self.task = self.args.taskvar
        self.variation = "0"
        self.instructions = taskvars_instructions[self.args.taskvar]

        # Load bbox info
        links_bbox_file_path = os.path.join(
            get_package_share_directory("robo_maestro"),
            "assets",
            "real_robot_bbox_info.pkl",
        )
        with open(links_bbox_file_path, "rb") as f:
            self.links_bbox = pkl.load(f)

        # Create environment
        use_sim_time = self.get_parameter("use_sim_time").value
        env = gym.make(
            "RealRobot-BaseEnv",
            cam_list=self.args.cam_list,
            node=self,
            use_sim_time=use_sim_time,
            disable_env_checker=True,  # skip need for defining action and observation spaces
        )
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env

        # Initialize evaluator
        self.evaluator = TaskEvaluator(
            self.env,
            self.task,
            self.variation,
            self.instructions,
            self.args.episode_id,
            self.links_bbox,
            self.args.cam_list,
            self.args.ip,
            self.args.port,
            mock=self.args.mock,
        )

    def run(self):
        obs, info = self.env.reset()

        keystep_real: ObsStateDict = process_keystep(
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

            result = self.evaluator.execute_step(step_id, keystep_real, cache)
            if result is None:
                log_error("Environment step failed, exiting.")
                break
            keystep_real, cache = result

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
