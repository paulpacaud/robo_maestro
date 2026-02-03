#!/usr/bin/env python3
"""
ROS 2 node – Joystick teleop + keystep data collection.

Teleoperate the robot with a DualSense PS5 controller (via pygame),
record keysteps at desired poses, save episodes to LMDB, and manage
the dataset session.

Usage:
ros2 launch robo_maestro collect_dataset.launch.py use_sim_time:=false

# By default this uses task="my_task", var=0, cam_list=["foxtrot_camera"].
# To customize, pass CLI args after '--':
ros2 launch robo_maestro collect_dataset.launch.py \
    use_sim_time:=false \
    task:=test_task \
    var:=0 \
    debug:=true

"""

import pickle as pkl
import threading
import time
from pathlib import Path

import gymnasium as gym
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pygame
import rclpy
import torch
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

import robo_maestro.envs  # noqa: F401 – registers gym envs
from robo_maestro.utils.constants import DATA_DIR, DEFAULT_ROBOT_ACTION
from robo_maestro.utils.helpers import crop_center, resize
from robo_maestro.utils.logger import log_error, log_info, log_success, log_warn

msgpack_numpy.patch()


# ---------------------------------------------------------------------------
# Dataset – LMDB + msgpack episode storage
# ---------------------------------------------------------------------------
class Dataset:
    """Stores keystep episodes in LMDB, matching the legacy data format."""

    def __init__(
        self,
        output_dir: str,
        camera_list: list[str],
        crop_size: int | None = None,
        links_bbox: dict | None = None,
    ):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.lmdb_env = lmdb.open(str(output_dir), map_size=int(1024**4))
        self.data: list[dict] = []
        self.episode_idx = 0
        self.camera_list = camera_list
        self.crop_size = crop_size
        self.links_bbox = links_bbox

    # -- per-episode helpers ------------------------------------------------

    def reset(self):
        self.data = []

    def add_keystep(self, obs: dict):
        gripper_pos = obs["gripper_pos"]
        gripper_quat = obs["gripper_quat"]
        gripper_pose = np.concatenate([gripper_pos, gripper_quat])
        gripper_state = not obs["gripper_state"]

        rgb, pc, depth = [], [], []
        gripper_uv: dict = {}
        for cam_name in self.camera_list:
            rgb.append(torch.from_numpy(obs[f"rgb_{cam_name}"]))
            pc.append(torch.from_numpy(obs[f"pcd_{cam_name}"]))
            depth.append(torch.from_numpy(obs[f"depth_{cam_name}"]))
            gripper_uv[cam_name] = obs[f"gripper_uv_{cam_name}"]

        rgb = torch.stack(rgb)
        pc = torch.stack(pc)
        depth = torch.stack(depth)
        action = np.concatenate([gripper_pose, np.array([int(gripper_state)])], axis=-1)

        if self.crop_size:
            rgb = rgb.permute(0, 3, 1, 2)
            pc = pc.permute(0, 3, 1, 2)
            depth = depth.permute(0, 3, 1, 2)

            rgb, ratio = resize(rgb, self.crop_size, im_type="rgb")
            pc, _ = resize(pc, self.crop_size, im_type="pc")
            depth, _ = resize(depth, self.crop_size, im_type="pc")
            rgb, start_x, start_y = crop_center(rgb, self.crop_size, self.crop_size)
            pc, start_x, start_y = crop_center(pc, self.crop_size, self.crop_size)
            depth, start_x, start_y = crop_center(depth, self.crop_size, self.crop_size)

            rgb = rgb.permute(0, 2, 3, 1)
            pc = pc.permute(0, 2, 3, 1)
            depth = depth.permute(0, 2, 3, 1)

            for cam_name, uv in gripper_uv.items():
                gripper_uv[cam_name] = [
                    int(uv[0] * ratio) - start_x,
                    int(uv[1] * ratio) - start_y,
                ]

        robot_info = obs["robot_info"]
        bbox_info: dict = {}
        pose_info: dict = {}
        for link_name, link_pose in robot_info.items():
            pose_info[f"{link_name}_pose"] = link_pose
            bbox_info[f"{link_name}_bbox"] = self.links_bbox[link_name]

        keystep = {
            "rgb": rgb,
            "pc": pc,
            "depth": depth,
            "gripper_uv": gripper_uv,
            "action": action,
            "bbox_info": bbox_info,
            "pose_info": pose_info,
        }
        self.data.append(keystep)

    # -- episode lifecycle --------------------------------------------------

    def save(self):
        rgbs, pcs, depths = [], [], []
        gripper_uv_list: list = []
        actions: list = []
        bbox_info_list: list = []
        pose_info_list: list = []

        for ks in self.data:
            rgbs.append(ks["rgb"])
            pcs.append(ks["pc"])
            depths.append(ks["depth"])
            gripper_uv_list.append(ks["gripper_uv"])
            actions.append(ks["action"])
            bbox_info_list.append(ks["bbox_info"])
            pose_info_list.append(ks["pose_info"])

        outs = {
            "rgb": torch.stack(rgbs).numpy().astype(np.uint8),
            "pc": torch.stack(pcs).float().numpy(),
            "depth": torch.stack(depths).float().numpy(),
            "gripper_uv": gripper_uv_list,
            "action": np.stack(actions).astype(np.float32),
            "bbox_info": bbox_info_list,
            "pose_info": pose_info_list,
        }

        txn = self.lmdb_env.begin(write=True)
        txn.put(
            f"episode{self.episode_idx}".encode("ascii"),
            msgpack.packb(outs),
        )
        txn.commit()
        self.episode_idx += 1
        self.reset()

    def done(self):
        self.lmdb_env.close()


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class CollectDatasetNode(Node):
    def __init__(self):
        super().__init__(
            "collect_dataset",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Read configuration from ROS2 parameters (set by launch file)
        def _str_param(name, default):
            try:
                return self.get_parameter(name).get_parameter_value().string_value or default
            except rclpy.exceptions.ParameterNotDeclaredException:
                return default

        def _double_param(name, default):
            try:
                v = self.get_parameter(name).get_parameter_value()
                # Launch args arrive as strings; try double first, fall back to parsing string
                if v.type == 3:  # PARAMETER_DOUBLE
                    return v.double_value
                return float(v.string_value)
            except (rclpy.exceptions.ParameterNotDeclaredException, ValueError):
                return default

        def _int_param(name, default):
            try:
                v = self.get_parameter(name).get_parameter_value()
                if v.type == 2:  # PARAMETER_INTEGER
                    return v.integer_value
                return int(v.string_value)
            except (rclpy.exceptions.ParameterNotDeclaredException, ValueError):
                return default

        def _bool_param(name, default):
            try:
                v = self.get_parameter(name).get_parameter_value()
                if v.type == 1:  # PARAMETER_BOOL
                    return v.bool_value
                return v.string_value.lower() in ("true", "1", "yes")
            except (rclpy.exceptions.ParameterNotDeclaredException, ValueError):
                return default

        cam_str = _str_param("cam_list", "foxtrot_camera")
        self.cam_list = [c.strip() for c in cam_str.split(",")]
        self.task = _str_param("task", "my_task")
        self.var = _int_param("var", 0)
        self.data_dir = _str_param("data_dir", DATA_DIR)
        self.pos_step = _double_param("pos_step", 0.02)
        self.rot_step = _double_param("rot_step", 5.0)
        self.crop_size = _int_param("crop_size", 256)
        self.debug = _bool_param("debug", False)

        self.gripper_state = 0  # 0 = open, 1 = closed
        self.running = True

        log_info(
            f"Config: task={self.task} var={self.var} cam_list={self.cam_list} "
            f"pos_step={self.pos_step} rot_step={self.rot_step} debug={self.debug}"
        )

    # -- setup --------------------------------------------------------------

    def setup_environment(self):
        """Create gym env, load bbox, init dataset and joystick."""
        log_info("Setting up environment ...")

        env = gym.make(
            "RealRobot-BaseEnv",
            cam_list=self.cam_list,
            node=self,
            use_sim_time=False,
            disable_env_checker=True,
        )
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env

        # Load bounding-box info for robot links
        import importlib.resources as pkg_resources

        bbox_path = Path(
            pkg_resources.files("robo_maestro") / "assets" / "real_robot_bbox_info.pkl"
        )
        with open(bbox_path, "rb") as f:
            self.links_bbox = pkl.load(f)

        output_dir = Path(self.data_dir) / f"{self.task}+{self.var}"
        self.dataset = Dataset(
            str(output_dir),
            camera_list=self.cam_list,
            crop_size=self.crop_size,
            links_bbox=self.links_bbox,
        )

        # Pygame / joystick init
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            log_error("No joystick detected! Connect a DualSense PS5 controller.")
            raise RuntimeError("No joystick detected")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        log_success(f"Joystick detected: {self.joy.get_name()}")

        log_success("Environment ready. Use joystick to teleoperate.")

    # -- main loop ----------------------------------------------------------

    def run(self):
        """Synchronous polling loop (runs in main thread)."""
        deadzone = 0.15

        while self.running:
            pygame.event.pump()

            # --- Discrete button events (one-shot) ---
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    self._handle_button(event.button)
                elif event.type == pygame.JOYHATMOTION:
                    hat = event.value  # (x, y)
                    if hat[1] == 1:  # D-pad up
                        self._reset_to_home()

            if not self.running:
                break

            # --- Continuous axes (teleop) ---
            self._read_and_apply_teleop(deadzone)

    # -- button dispatch ----------------------------------------------------

    def _handle_button(self, button: int):
        if self.debug:
            log_info(f"Button pressed: {button}")

        if button == 0:  # Cross – record keystep
            self._record_keystep()
        elif button == 1:  # Circle – close gripper
            self.gripper_state = 1
            log_info("Gripper → CLOSED")
        elif button == 2:  # Triangle – open gripper
            self.gripper_state = 0
            log_info("Gripper → OPEN")
        elif button == 3:  # Square – save episode
            self._save_episode()
        elif button == 8:  # Share – discard episode
            self._discard_episode()
        elif button == 9:  # Options – finish session
            self._finish_session()

    # -- teleop -------------------------------------------------------------

    def _read_and_apply_teleop(self, deadzone: float):
        # Read axes
        ax_lx = self.joy.get_axis(0)  # left stick X  → Y pos
        ax_ly = self.joy.get_axis(1)  # left stick Y  → X pos
        ax_rx = self.joy.get_axis(2)  # right stick X → Z rot (yaw)
        ax_ry = self.joy.get_axis(3)  # right stick Y → Z pos

        # Rotation buttons
        rot_dx = 0.0
        rot_dy = 0.0
        rot_dz = 0.0

        if self.joy.get_button(4):  # L1 → X-rot negative
            rot_dx = -self.rot_step
        if self.joy.get_button(5):  # R1 → X-rot positive
            rot_dx = self.rot_step
        if self.joy.get_button(6):  # L2 click → Y-rot negative
            rot_dy = -self.rot_step
        if self.joy.get_button(7):  # R2 click → Y-rot positive
            rot_dy = self.rot_step

        # Apply deadzone
        def dz(v):
            return v if abs(v) > deadzone else 0.0

        ax_lx = dz(ax_lx)
        ax_ly = dz(ax_ly)
        ax_rx = dz(ax_rx)
        ax_ry = dz(ax_ry)

        # Z-rotation from right stick X
        if abs(ax_rx) > 0:
            rot_dz = -ax_rx * self.rot_step

        has_motion = (
            ax_lx != 0
            or ax_ly != 0
            or ax_ry != 0
            or rot_dx != 0
            or rot_dy != 0
            or rot_dz != 0
        )

        if not has_motion:
            time.sleep(0.05)
            return

        if self.debug:
            log_info(
                f"Axes: lx={ax_lx:.2f} ly={ax_ly:.2f} "
                f"rx={ax_rx:.2f} ry={ax_ry:.2f} | "
                f"rot: dx={rot_dx:.1f} dy={rot_dy:.1f} dz={rot_dz:.1f}"
            )

        # Current EEF pose
        eef = self.env.robot.eef_pose()
        current_pos = eef[0]  # np.array([x, y, z])
        current_quat = eef[1]  # np.array([qx, qy, qz, qw])

        # Position delta
        new_pos = current_pos.copy()
        new_pos[0] += -ax_ly * self.pos_step  # left stick Y → X (forward/back)
        new_pos[1] += ax_lx * self.pos_step  # left stick X → Y (left/right)
        new_pos[2] += -ax_ry * self.pos_step  # right stick Y → Z (up/down)

        # Orientation delta (local-frame rotation)
        R_current = Rotation.from_quat(current_quat)  # [qx, qy, qz, qw]
        R_delta = Rotation.from_euler("xyz", [rot_dx, rot_dy, rot_dz], degrees=True)
        R_new = R_current * R_delta  # local-frame rotation
        new_quat = R_new.as_quat()  # [qx, qy, qz, qw]

        # Build 8D action
        action = [
            float(new_pos[0]),
            float(new_pos[1]),
            float(new_pos[2]),
            float(new_quat[0]),
            float(new_quat[1]),
            float(new_quat[2]),
            float(new_quat[3]),
            float(self.gripper_state),
        ]

        log_info(f"Teleop → go_to_pose({action})")
        self.env.robot.go_to_pose(action)

    # -- keystep / episode actions ------------------------------------------

    def _record_keystep(self):
        log_info("Recording keystep ...")
        obs = self.env._get_obs()
        self.dataset.add_keystep(obs)
        log_success(f"Keystep {len(self.dataset.data)} recorded")

    def _save_episode(self):
        if not self.dataset.data:
            log_warn("No keysteps to save – record at least one keystep first.")
            return
        ep = self.dataset.episode_idx
        log_info(f"Saving episode {ep} ({len(self.dataset.data)} keysteps) ...")
        self.dataset.save()
        log_success(f"Episode {ep} saved. Resetting robot ...")
        self.env.robot.reset()
        log_success("Robot reset. Ready for next episode.")

    def _discard_episode(self):
        n = len(self.dataset.data)
        self.dataset.reset()
        log_warn(f"Episode discarded ({n} keysteps dropped). Resetting robot ...")
        self.env.robot.reset()
        log_success("Robot reset. Ready for next episode.")

    def _reset_to_home(self):
        log_info("Resetting robot to home pose ...")
        self.env.robot.reset()
        log_success("Robot at home pose.")

    def _finish_session(self):
        log_info("Finishing session ...")
        self.dataset.done()
        log_success(
            f"Session finished. {self.dataset.episode_idx} episode(s) saved to "
            f"{self.dataset.output_dir}"
        )
        self.running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = CollectDatasetNode()

    # NOTE: Do NOT start the spin thread before setup_environment().
    # Robot/Camera init uses rclpy.spin_until_future_complete() internally
    # (via wait_for_message), which needs to temporarily spin the node.
    # If the node is already in an executor, that call conflicts.
    # After setup completes, we start the background spin thread for
    # continuous TF/camera updates during the teleop loop.

    try:
        node.setup_environment()

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        node.run()
    except KeyboardInterrupt:
        log_warn("KeyboardInterrupt – shutting down.")
    except Exception as e:
        log_error(f"Fatal error: {e}")
        raise
    finally:
        pygame.quit()
        if "executor" in locals():
            executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
