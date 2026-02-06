"""
LeRobot gRPC bridge for UR5 — sends observations to a remote LeRobot policy
server via gRPC and executes the returned actions on the real robot.

Usage (inside Docker):
ros2 launch robo_maestro run_policy_lerobot.launch.py \
    policy_type:=pi0 \
    pretrained_name_or_path:=/home/ppacaud/data/lerobot/models/pi0_multitasks_3tasks_ur5_20260206_004448-ckpt10k \
    task:="put the grapes in the yellow plate, then put the banana in the pink plate" port:=8002

ros2 launch robo_maestro run_policy_lerobot.launch.py \
    policy_type:=groot \
    pretrained_name_or_path:=/home/ppacaud/data/lerobot/models/groot1.5_multitasks_3tasks_ur5_20260206_010732-ckpt10k \
    task:="put the grapes in the yellow plate, then put the banana in the pink plate" port:=8002

Requires an SSH tunnel (or direct connection) to the policy server:
    ssh -N -L 8080:127.0.0.1:8080 ppacaud@dgx-station.paris.inria.fr

# On dgx-station
CUDA_VISIBLE_DEVICES=3 TORCH_COMPILE_DISABLE=1 python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 --port=8002
"""

import json
import os
import pickle
import time

import grpc
import gymnasium as gym
import numpy as np
import tap
import torch
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

import robo_maestro.envs  # noqa: F401 — registers gym env
import robo_maestro.lerobot_bridge  # registers sys.modules aliases
from robo_maestro.lerobot_bridge.compat import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from robo_maestro.lerobot_bridge import services_pb2, services_pb2_grpc
from robo_maestro.lerobot_bridge.transport_utils import (
    grpc_channel_options,
    python_object_to_bytes,
    bytes_to_python_object,
    send_bytes_in_chunks,
)
from robo_maestro.utils.constants import DATA_DIR, MAX_STEPS
from robo_maestro.utils.logger import log_info, log_warn, log_error, log_success


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------


class Arguments(tap.Tap):
    cam_list: list[str] = ["echo_camera"]
    arm: str = "left"
    taskvar: str = "ur5_close_drawer"
    task: str = ""  # natural language instruction (VLA)
    ip: str = "127.0.0.1"
    port: int = 8080
    episode_id: int = 0
    policy_type: str = "pi0"
    pretrained_name_or_path: str = ""
    policy_device: str = "cuda"
    actions_per_chunk: int = 50
    rename_map: str = "{}"  # JSON string for obs key renaming


# ---------------------------------------------------------------------------
# gRPC client
# ---------------------------------------------------------------------------


class LeRobotPolicyClient:
    """Wraps a gRPC channel + AsyncInferenceStub."""

    def __init__(self, ip: str, port: int):
        self.address = f"{ip}:{port}"
        self.channel = None
        self.stub = None
        self.action_chunk_cache: list[TimedAction] = []
        self.cache_index: int = 0

    def connect(self, policy_config: RemotePolicyConfig):
        """Open channel, handshake, and send policy configuration."""
        log_info(f"Connecting to LeRobot policy server at {self.address} ...")
        self.channel = grpc.insecure_channel(self.address, grpc_channel_options())
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        # Handshake
        self.stub.Ready(services_pb2.Empty())
        log_info("Connected (Ready handshake OK)")

        # Send policy instructions
        config_bytes = python_object_to_bytes(policy_config)
        self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=config_bytes))
        log_info("Sent RemotePolicyConfig to server")

    def send_observation(self, timed_obs: TimedObservation):
        """Stream a single observation (chunked) to the server."""
        obs_bytes = pickle.dumps(timed_obs)
        obs_iter = send_bytes_in_chunks(
            obs_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Obs",
            silent=True,
        )
        response = self.stub.SendObservations(obs_iter)
        # Force completion — stream_unary may return a future-like object
        if hasattr(response, "result"):
            response.result()

    def get_actions(self, max_retries: int = 5, retry_delay: float = 1.0) -> list[TimedAction]:
        """Block until the server returns an action chunk, with retries."""
        for attempt in range(max_retries):
            actions_msg = self.stub.GetActions(services_pb2.Empty())
            if len(actions_msg.data) > 0:
                return bytes_to_python_object(actions_msg.data)
            log_info(f"GetActions returned empty (attempt {attempt + 1}/{max_retries}), waiting {retry_delay}s ...")
            time.sleep(retry_delay)
        return []

    def get_next_action(self) -> np.ndarray | None:
        """
        Get the next action from the cached chunk, or None if cache is empty.
        Returns an 8-D numpy array ready for execution.
        """
        if self.cache_index < len(self.action_chunk_cache):
            timed_action = self.action_chunk_cache[self.cache_index]
            self.cache_index += 1
            action = lerobot_actions_to_ur5([timed_action])
            log_info(f"Using cached action {self.cache_index}/{len(self.action_chunk_cache)}")
            return action
        return None

    def needs_new_observation(self) -> bool:
        """Returns True if the action cache is exhausted and a new observation is needed."""
        return self.cache_index >= len(self.action_chunk_cache)

    def refill_action_cache(self) -> bool:
        """
        Request new actions from server and reset cache.
        Returns True if successful, False otherwise.
        """
        timed_actions = self.get_actions()
        if not timed_actions:
            log_error("Failed to get actions after retries")
            return False
        self.action_chunk_cache = timed_actions
        self.cache_index = 0
        log_info(f"Received action chunk with {len(timed_actions)} actions")
        return True

    def close(self):
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            log_info("gRPC channel closed")


# ---------------------------------------------------------------------------
# Feature / observation helpers
# ---------------------------------------------------------------------------


def build_ur5_lerobot_features(
    cam_list: list[str],
    image_shape: tuple[int, int, int] = (480, 640, 3),
) -> dict[str, dict]:
    """Build the ``lerobot_features`` dict expected by RemotePolicyConfig.
    Values must be plain dicts (not PolicyFeature) because the server's
    ``build_dataset_frame`` accesses them with ``ft["dtype"]``, ``ft["names"]``, etc.
    """
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": _STATE_KEYS,
        },
    }
    if len(cam_list) > 1:
        raise NotImplementedError("Multiple cameras not yet supported in this example")
    features["observation.images.front_image"] = {
        "dtype": "image",
        "shape": image_shape,
        "names": ["height", "width", "channels"],
    }
    return features


_STATE_KEYS = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_state"]


def ur5_obs_to_raw_observation(
    obs: dict,
    cam_list: list[str],
    task_str: str,
) -> dict:
    """Convert a robo_maestro observation dict to a LeRobot RawObservation."""
    raw_obs: dict = {}

    # State: gripper_pos(3) + gripper_quat(4) + gripper_state(1) → named keys
    gripper_state = obs["gripper_state"]
    state_vec = np.concatenate(
        [
            obs["gripper_pos"],
            obs["gripper_quat"],
            np.atleast_1d(gripper_state)[:1],
        ]
    )
    for key, val in zip(_STATE_KEYS, state_vec):
        raw_obs[key] = float(val)

    # Images — key must match feature key "observation.images.front_image" after prefix strip
    raw_obs["front_image"] = obs[f"rgb_{cam_list[0]}"]  # (H, W, 3) uint8

    # Language instruction
    if task_str:
        raw_obs["task"] = task_str

    return raw_obs


def lerobot_actions_to_ur5(timed_actions: list[TimedAction]) -> np.ndarray:
    """Extract the first action from a chunk and return an 8-D numpy array."""
    action_tensor = timed_actions[0].get_action()
    if isinstance(action_tensor, torch.Tensor):
        action_tensor = action_tensor.cpu().numpy()
    return np.asarray(action_tensor, dtype=np.float64).flatten()[:8]


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------


class RunPolicyLeRobotNode(Node):
    def __init__(self):
        super().__init__(
            "run_policy_lerobot_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.args = Arguments().parse_args(known_only=True)
        self.setup_environment()

    def setup_environment(self):
        args = self.args

        # --- Task instruction ---
        if args.task:
            self.task_str = args.task
        else:
            taskvars_path = os.path.join(
                get_package_share_directory("robo_maestro"),
                "assets",
                "taskvars_instructions.json",
            )
            with open(taskvars_path) as f:
                taskvars_instructions = json.load(f)
            self.task_str = taskvars_instructions[args.taskvar]

        # --- Gym environment ---
        use_sim_time = self.get_parameter("use_sim_time").value
        env = gym.make(
            "RealRobot-BaseEnv",
            cam_list=args.cam_list,
            node=self,
            use_sim_time=use_sim_time,
            disable_env_checker=True,
        )
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env

        # --- Save directory ---
        ts = __import__("datetime").datetime.now().strftime("%m%dT%H%M%S")
        self.save_path = os.path.join(
            DATA_DIR,
            "run_policy_lerobot_experiments",
            f"{args.taskvar}+0_{ts}",
            f"episode_{args.episode_id}",
        )

        # --- gRPC client ---
        self.client = LeRobotPolicyClient(args.ip, args.port)

    def run(self):
        args = self.args

        # Reset environment and get first observation
        obs, _ = self.env.reset()
        log_info("Environment reset done")

        # Determine image shape from first camera observation
        first_cam = args.cam_list[0]
        image_shape = obs[f"rgb_{first_cam}"].shape  # (H, W, 3)

        # Build features and policy config
        lerobot_features = build_ur5_lerobot_features(args.cam_list, image_shape)
        rename_map = json.loads(args.rename_map)
        policy_config = RemotePolicyConfig(
            policy_type=args.policy_type,
            pretrained_name_or_path=args.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=args.actions_per_chunk,
            device=args.policy_device,
            rename_map=rename_map,
        )

        # Connect to policy server
        self.client.connect(policy_config)
        log_info("Policy server configured, starting control loop")

        timestep = 0
        for step_id in range(MAX_STEPS):
            if not rclpy.ok():
                break
            rclpy.spin_once(self, timeout_sec=0.0)

            log_info(f"Step {step_id}/{MAX_STEPS - 1}")

            # Convert current observation (needed for both sending and saving)
            raw_obs = ur5_obs_to_raw_observation(obs, args.cam_list, self.task_str)

            # Send new observation only if action cache is exhausted
            if self.client.needs_new_observation():
                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    timestep=timestep,
                    observation=raw_obs,
                )
                self.client.send_observation(timed_obs)
                log_info(f"Sent observation #{timestep}")

                # Refill action cache
                if not self.client.refill_action_cache():
                    log_error("Failed to refill action cache, skipping step")
                    continue
            else:
                log_info("Using cached actions (no new observation needed)")

            # Get next action from cache
            action = self.client.get_next_action()
            if action is None:
                log_error("Action cache unexpectedly empty, skipping step")
                continue

            log_info(f"Action: {action}")

            # Save step data
            self._save_step(step_id, raw_obs, action)

            # Execute
            if not rclpy.ok():
                break
            obs, _, _, _, _ = self.env.step(action)

            timestep += 1

        self.client.close()
        log_success("Control loop finished")

    def _save_step(self, step_id: int, raw_obs: dict, action: np.ndarray):
        """Persist observation and action for later analysis."""
        step_dir = os.path.join(self.save_path, "steps")
        os.makedirs(step_dir, exist_ok=True)
        np.save(
            os.path.join(step_dir, f"{step_id}.npy"),
            {"raw_obs": raw_obs, "action": action},
            allow_pickle=True,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    rclpy.init()
    node = RunPolicyLeRobotNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
