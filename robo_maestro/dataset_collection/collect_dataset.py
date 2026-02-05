#!/usr/bin/env python3
"""
ROS 2 node -- Joystick teleop + keystep data collection.

Teleoperate the robot with a DualSense PS5 controller (via pygame), record keysteps at desired poses,
save episodes to LMDB, and manage the dataset session.

The task instruction is loaded automatically from robo_maestro/utils/taskvars_instructions.json using the key "{task}+{var}".

A meta.json file is saved in the output directory (<data_dir>/<task>+<var>/) with task info, camera intrinsics and extrinsics.

Usage
-----
# Required args: task, cam_list, start_episode_id.
ros2 launch robo_maestro collect_dataset.launch.py \
    use_sim_time:=false \
    task:=ur5_stack_yellow_onto_pink_cup\
    cam_list:=echo_camera \
    start_episode_id:=0

# Multiple cameras: pass a comma-separated string (no spaces).
ros2 launch robo_maestro collect_dataset.launch.py \\
    use_sim_time:=false \\
    task:=put_fruits_in_plate \\
    cam_list:=foxtrot_camera,echo_camera,golf_camera \\
    start_episode_id:=0

# Append to an existing dataset (e.g. 5 episodes already collected):
ros2 launch robo_maestro collect_dataset.launch.py \\
    use_sim_time:=false \\
    task:=put_fruits_in_plate \\
    cam_list:=foxtrot_camera \\
    start_episode_id:=5

# Optional args (shown with defaults):
ros2 launch robo_maestro collect_dataset.launch.py \\
    use_sim_time:=false \\
    task:=put_fruits_in_plate \\
    cam_list:=foxtrot_camera \\
    start_episode_id:=0 \\
    pos_step:=0.02 \\
    rot_step:=5.0 \\
    crop_size:=256 \\
    debug:=true
"""

"""
DualSense PS5 Controller Mapping
---------------------------------

Movement (analog sticks -- continuous, proportional)

  Stick input               Robot effect          Detail
  -------------------------------------------------------------------------
  Left stick  up/down       Y position (L / R)    Push up   = robot left
  Left stick  left/right    X position (fwd/back) Push left = robot forward
  Right stick up/down       Z position (up/down)  Push up   = robot up
  Right stick left/right    Z rotation (yaw)      Push right = CW

  Step size: pos_step (default 2 cm), deadzone: 0.20

Rotation (bumpers / triggers -- hold to rotate, 5 deg/step)

  L1 (hold)   Roll  -X rotation
  R1 (hold)   Roll  +X rotation
  L2 (hold)   Pitch -Y rotation
  R2 (hold)   Pitch +Y rotation

Gripper (one-shot, immediate)

  Circle      Close gripper
  Triangle    Open gripper

Data collection (one-shot)

  Cross        Record keystep at current pose
  D-pad up     Reset robot to home pose
  D-pad down   Undo last keystep
  D-pad left   Save episode to LMDB + reset robot
  D-pad right  Undo last saved episode (delete from LMDB)
  Share        Discard entire episode + reset robot
  Options      Finish session and exit
  Square

DualSense axis mapping (Linux hid_playstation driver)

  Axis 0: Left stick X     Axis 1: Left stick Y
  Axis 2: L2 trigger        Axis 3: Right stick X
  Axis 4: Right stick Y     Axis 5: R2 trigger

  Axes 2/5 rest at -1.0 (not used; rotation via button events).
"""

import json
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
from robo_maestro.core.tf import pos_euler_to_hom
from robo_maestro.schemas import (
    CameraConfig,
    CameraExtrinsics,
    CameraIntrinsics,
    CollectedDatasetMeta,
    GembenchKeystep,
    pack_keysteps,
)
from robo_maestro.utils.constants import DATA_DIR, DEFAULT_ROBOT_ACTION
from robo_maestro.utils.helpers import crop_center, resize
from robo_maestro.utils.logger import log_error, log_info, log_success, log_warn

msgpack_numpy.patch()


# ---------------------------------------------------------------------------
# Dataset – LMDB + msgpack episode storage
# ---------------------------------------------------------------------------
class Dataset:
    """Stores keystep episodes in LMDB."""

    def __init__(
        self,
        output_dir: str,
        camera_list: list[str],
        start_episode_id: int = 0,
        crop_size: int | None = None,
        links_bbox: dict | None = None,
    ):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.lmdb_env = lmdb.open(str(output_dir), map_size=int(1024**4))
        self.data: list[GembenchKeystep] = []
        self.camera_list = camera_list
        self.crop_size = crop_size
        self.links_bbox = links_bbox

        # Validate start_episode_id against existing episodes
        self._validate_start_episode_id(start_episode_id)
        self.episode_idx = start_episode_id

    def _validate_start_episode_id(self, start_episode_id: int):
        """Check that start_episode_id is exactly the next available index."""
        with self.lmdb_env.begin() as txn:
            existing_keys = {key.decode() for key, _ in txn.cursor()}
        # Find the expected next id: number of existing episodes
        existing_count = len(existing_keys)

        if start_episode_id < existing_count:
            log_error(
                f"start_episode_id={start_episode_id} already exists in "
                f"{self.output_dir} (dataset has {existing_count} episodes: "
                f"episode0..episode{existing_count - 1}). "
                f"Use start_episode_id:={existing_count} to append."
            )
            raise RuntimeError(
                f"start_episode_id={start_episode_id} conflicts with existing data"
            )
        if start_episode_id > existing_count:
            log_error(
                f"start_episode_id={start_episode_id} leaves a gap — "
                f"dataset has {existing_count} episodes "
                f"(episode0..episode{existing_count - 1}). "
                f"Use start_episode_id:={existing_count} to append."
            )
            raise RuntimeError(
                f"start_episode_id={start_episode_id} leaves a gap in episode indices"
            )

    # -- per-episode helpers ------------------------------------------------

    def reset(self):
        self.data = []

    def add_keystep(self, obs: dict):
        gripper_pos = obs["gripper_pos"]
        gripper_quat = obs["gripper_quat"]
        gripper_pose = np.concatenate([gripper_pos, gripper_quat])
        gripper_state = obs["gripper_state"]

        rgb, xyz, depth = [], [], []
        for cam_name in self.camera_list:
            rgb.append(torch.from_numpy(obs[f"rgb_{cam_name}"]))
            xyz.append(torch.from_numpy(obs[f"pcd_{cam_name}"]))
            depth.append(torch.from_numpy(obs[f"depth_{cam_name}"]))

        rgb = torch.stack(rgb)
        xyz = torch.stack(xyz)
        depth = torch.stack(depth)
        action = np.concatenate(
            [gripper_pose, np.array([int(gripper_state)])], axis=-1
        ).astype(np.float32)

        if self.crop_size:
            rgb = rgb.permute(0, 3, 1, 2)
            xyz = xyz.permute(0, 3, 1, 2)
            # Depth is single-channel (N, H, W) — add channel dim for resize/crop
            if depth.dim() == 3:
                depth = depth.unsqueeze(1)  # (N, H, W) → (N, 1, H, W)
            else:
                depth = depth.permute(0, 3, 1, 2)

            rgb, ratio = resize(rgb, self.crop_size, im_type="rgb")
            xyz, _ = resize(xyz, self.crop_size, im_type="pc")
            depth, _ = resize(depth, self.crop_size, im_type="pc")
            rgb, start_x, start_y = crop_center(rgb, self.crop_size, self.crop_size)
            xyz, start_x, start_y = crop_center(xyz, self.crop_size, self.crop_size)
            depth, start_x, start_y = crop_center(depth, self.crop_size, self.crop_size)

            rgb = rgb.permute(0, 2, 3, 1)
            xyz = xyz.permute(0, 2, 3, 1)
            if depth.shape[1] == 1:
                depth = depth.squeeze(1)  # (N, 1, H, W) → (N, H, W)
            else:
                depth = depth.permute(0, 2, 3, 1)

        robot_info = obs["robot_info"]
        bbox_info: dict = {}
        pose_info: dict = {}
        for link_name, link_pose in robot_info.items():
            pose_info[f"{link_name}_pose"] = link_pose
            bbox_info[f"{link_name}_bbox"] = self.links_bbox[link_name]

        keystep = GembenchKeystep(
            rgb=rgb.numpy().astype(np.uint8),
            xyz=xyz.float().numpy(),
            depth=depth.float().numpy(),
            action=action,
            bbox_info=bbox_info,
            pose_info=pose_info,
        )
        self.data.append(keystep)

    # -- episode lifecycle --------------------------------------------------

    def save(self):
        txn = self.lmdb_env.begin(write=True)
        txn.put(
            f"episode{self.episode_idx}".encode("ascii"),
            msgpack.packb(pack_keysteps(self.data)),
        )
        txn.commit()
        self.episode_idx += 1
        self.reset()

    def undo_last_episode(self) -> bool:
        """Delete the most recently saved episode from LMDB."""
        if self.episode_idx == 0:
            return False
        self.episode_idx -= 1
        key = f"episode{self.episode_idx}".encode("ascii")
        txn = self.lmdb_env.begin(write=True)
        txn.delete(key)
        txn.commit()
        return True

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
                return (
                    self.get_parameter(name).get_parameter_value().string_value
                    or default
                )
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

        # Required parameters — fail fast if not provided
        def _require_str(name):
            try:
                v = self.get_parameter(name).get_parameter_value().string_value
                if not v:
                    raise ValueError
                return v
            except (rclpy.exceptions.ParameterNotDeclaredException, ValueError):
                raise RuntimeError(
                    f"Required parameter '{name}' not set. "
                    "Pass it via the launch file."
                )

        def _require_int(name):
            try:
                p = self.get_parameter(name)
                v = p.get_parameter_value()
                # Launch args may arrive as string, int, or double depending
                # on how ROS 2 infers the type.
                if v.type == 2:  # PARAMETER_INTEGER
                    return v.integer_value
                if v.type == 3:  # PARAMETER_DOUBLE
                    return int(v.double_value)
                if v.type == 4 and v.string_value:  # PARAMETER_STRING
                    return int(v.string_value)
                # Last resort: try the raw Parameter value
                if p.value is not None:
                    return int(p.value)
                raise ValueError
            except (
                rclpy.exceptions.ParameterNotDeclaredException,
                ValueError,
                TypeError,
            ):
                raise RuntimeError(
                    f"Required parameter '{name}' not set. "
                    "Pass it via the launch file."
                )

        self.task = _require_str("task")
        cam_str = _require_str("cam_list")
        self.cam_list = [c.strip() for c in cam_str.split(",")]
        self.start_episode_id = _require_int("start_episode_id")

        # Load task instruction from JSON
        import importlib.resources as pkg_resources

        taskvar = f"{self.task}"
        _instr_path = Path(
            pkg_resources.files("robo_maestro")
            / "assets"
            / "taskvars_instructions.json"
        )
        with open(_instr_path) as f:
            taskvars_instructions = json.load(f)
        if taskvar not in taskvars_instructions:
            log_error(
                f"Task variant '{taskvar}' not found in "
                f"{"robo_maestro/assets/taskvars_instructions.json"}. "
                f"Available: {list(taskvars_instructions.keys())}"
            )
            raise RuntimeError(f"Unknown task variant: '{taskvar}'")
        self.task_instruction = taskvars_instructions[taskvar]

        self.data_dir = _str_param("data_dir", DATA_DIR)
        self.pos_step = _double_param("pos_step", 0.04)
        self.rot_step = _double_param("rot_step", 10)
        self.crop_size = _int_param("crop_size", 256)
        self.debug = _bool_param("debug", False)

        self.gripper_state = 0  # 0 = open, 1 = closed
        self.running = True
        # Track rotation button held-state via events, not polling,
        # to avoid DualSense analog-trigger drift on L2/R2.
        self._rot_buttons_held = {4: False, 5: False, 6: False, 7: False}
        # Commanded EEF state — deltas accumulate here instead of re-reading
        # the measured pose (which may lag behind during motion).
        self._cmd_pos = None  # initialized on first teleop call
        self._cmd_quat = None
        # Speed regime: fast (3x) by default, toggle with Square button
        self._fast_mode = True
        self._active_pos_step = self.pos_step * 5.0
        self._active_rot_step = self.rot_step * 5.0

        log_info(
            f"Config: task={self.task} cam_list={self.cam_list} "
            f"pos_step={self.pos_step} rot_step={self.rot_step} debug={self.debug}"
        )
        log_info(
            f"Speed regime: FAST (pos_step={self._active_pos_step}, rot_step={self._active_rot_step})"
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

        output_dir = Path(self.data_dir) / f"{self.task}"
        self.dataset = Dataset(
            str(output_dir),
            camera_list=self.cam_list,
            start_episode_id=self.start_episode_id,
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

        self._save_meta()

        log_success("Environment ready. Use joystick to teleoperate.")

    # -- metadata -----------------------------------------------------------

    def _save_meta(self):
        """Write meta.json with task info, camera intrinsics and extrinsics."""
        cameras: dict[str, CameraConfig] = {}
        for cam_name in self.cam_list:
            intrinsics_raw = self.env.cam_info[f"intrinsics_{cam_name}"]
            cam_pose = self.env.robot.cameras[cam_name].get_pose()
            cam_pos, cam_euler = cam_pose
            world_T_cam = pos_euler_to_hom(cam_pos, cam_euler)

            cameras[cam_name] = CameraConfig(
                intrinsics=CameraIntrinsics(
                    height=int(intrinsics_raw["height"]),
                    width=int(intrinsics_raw["width"]),
                    fx=float(intrinsics_raw["fx"]),
                    fy=float(intrinsics_raw["fy"]),
                    ppx=float(intrinsics_raw["ppx"]),
                    ppy=float(intrinsics_raw["ppy"]),
                    K=intrinsics_raw["K"].tolist(),
                ),
                extrinsics=CameraExtrinsics(
                    pos=[float(v) for v in cam_pos],
                    euler=[float(v) for v in cam_euler],
                    world_T_cam=world_T_cam.tolist(),
                ),
            )

        meta = CollectedDatasetMeta(
            task=self.task,
            task_instruction=self.task_instruction,
            cam_list=self.cam_list,
            cameras=cameras,
        )

        meta_path = Path(self.dataset.output_dir) / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta.model_dump(), f, indent=2)
        log_info(f"Saved metadata to {meta_path}")

    # -- main loop ----------------------------------------------------------

    def run(self):
        """Synchronous polling loop (runs in main thread)."""
        deadzone = 0.20

        while self.running:
            pygame.event.pump()

            # --- Discrete button events (one-shot) ---
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if event.button in self._rot_buttons_held:
                        self._rot_buttons_held[event.button] = True
                    self._handle_button(event.button)
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button in self._rot_buttons_held:
                        self._rot_buttons_held[event.button] = False
                elif event.type == pygame.JOYHATMOTION:
                    hat = event.value  # (x, y)
                    if hat[1] == 1:  # D-pad up
                        self._reset_to_home()
                    elif hat[1] == -1:  # D-pad down
                        self._undo_last_keystep()
                    elif hat[0] == -1:  # D-pad left
                        self._save_episode()
                    elif hat[0] == 1:  # D-pad right
                        self._undo_last_episode()

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
            self.env.robot.close_gripper()
        elif button == 2:  # Triangle – open gripper
            self.gripper_state = 0
            log_info("Gripper → OPEN")
            self.env.robot.open_gripper()
        elif button == 3:  # Square – toggle speed regime
            self._fast_mode = not self._fast_mode
            if self._fast_mode:
                self._active_pos_step = self.pos_step * 3.0
                self._active_rot_step = self.rot_step * 3.0
                log_info(
                    f"Speed regime: FAST (pos_step={self._active_pos_step}, rot_step={self._active_rot_step})"
                )
            else:
                self._active_pos_step = self.pos_step
                self._active_rot_step = self.rot_step
                log_info(
                    f"Speed regime: SLOW (pos_step={self._active_pos_step}, rot_step={self._active_rot_step})"
                )
        elif button == 8:  # Share – discard episode
            self._discard_episode()
        elif button == 9:  # Options – finish session
            self._finish_session()

    # -- teleop -------------------------------------------------------------

    def _read_and_apply_teleop(self, deadzone: float):
        # Read axes – DualSense on Linux (hid_playstation driver):
        #   0: left stick X    1: left stick Y
        #   2: L2 trigger      3: right stick X
        #   4: right stick Y   5: R2 trigger
        # Axes 2/5 (triggers) rest at -1.0; we ignore them here since
        # rotation is handled via button events.
        ax_lx = self.joy.get_axis(0)  # left stick X  → Y pos
        ax_ly = self.joy.get_axis(1)  # left stick Y  → X pos
        ax_rx = self.joy.get_axis(3)  # right stick X → Z rot (yaw)
        ax_ry = self.joy.get_axis(4)  # right stick Y → Z pos

        # Apply deadzone
        def dz(v):
            return v if abs(v) > deadzone else 0.0

        ax_lx = dz(ax_lx)
        ax_ly = dz(ax_ly)
        ax_rx = dz(ax_rx)
        ax_ry = dz(ax_ry)

        # Rotation buttons — use event-tracked state instead of get_button()
        # polling to avoid DualSense analog trigger drift on L2/R2
        rot_dx = 0.0
        rot_dy = 0.0
        rot_dz = 0.0

        rot_step = self._active_rot_step

        if self._rot_buttons_held[4]:  # L1 → X-rot negative
            rot_dx = -rot_step
        if self._rot_buttons_held[5]:  # R1 → X-rot positive
            rot_dx = rot_step
        if self._rot_buttons_held[6]:  # L2 → Y-rot negative
            rot_dy = -rot_step
        if self._rot_buttons_held[7]:  # R2 → Y-rot positive
            rot_dy = rot_step

        # Z-rotation from right stick X
        if abs(ax_rx) > 0:
            rot_dz = -ax_rx * rot_step

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

        # Use commanded pose (not measured) to avoid oscillation from
        # reading the EEF mid-motion.  Seed from measured pose on first call.
        if self._cmd_pos is None:
            eef = self.env.robot.eef_pose()
            self._cmd_pos = eef[0].copy()
            self._cmd_quat = eef[1].copy()

        # Position delta
        new_pos = self._cmd_pos.copy()
        pos_step = self._active_pos_step
        new_pos[0] += -ax_lx * pos_step  # left stick X → X (left=fwd, right=back)
        new_pos[1] += ax_ly * pos_step  # left stick Y → Y (up=left, down=right)
        new_pos[2] += -ax_ry * pos_step  # right stick Y → Z (up/down)

        # Orientation delta (local-frame rotation)
        R_current = Rotation.from_quat(self._cmd_quat)
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
        self.env.robot.go_to_pose(action, cartesian_only=True, sleep_time=0.5)

        # Update commanded state so next delta builds on the target, not
        # a stale measured pose.
        self._cmd_pos = np.array(new_pos)
        self._cmd_quat = np.array(new_quat)

    # -- keystep / episode actions ------------------------------------------

    @property
    def _tag(self):
        """Log prefix: [task] [ep N] [ks M]"""
        ep = self.dataset.episode_idx
        ks = len(self.dataset.data)
        return f"[{self.task}] [ep {ep}] [ks {ks}]"

    def _record_keystep(self):
        log_info(f"{self._tag} Recording keystep ...")
        obs = self.env._get_obs()
        self.dataset.add_keystep(obs)
        log_success(f"{self._tag} Keystep recorded")

    def _undo_last_keystep(self):
        if not self.dataset.data:
            log_warn(f"{self._tag} No keysteps to undo.")
            return
        self.dataset.data.pop()
        log_success(f"{self._tag} Last keystep removed.")

    def _reset_commanded_pose(self):
        """Clear commanded pose so it re-seeds from measured EEF on next teleop step."""
        self._cmd_pos = None
        self._cmd_quat = None

    def _save_episode(self):
        if not self.dataset.data:
            log_warn(f"{self._tag} No keysteps to save.")
            return
        log_info(f"{self._tag} Saving episode ...")
        self.dataset.save()
        log_success(f"{self._tag} Episode saved. Resetting robot ...")
        self.env.robot.reset()
        self._reset_commanded_pose()
        log_success(f"{self._tag} Robot reset. Ready for next episode.")

    def _undo_last_episode(self):
        if self.dataset.undo_last_episode():
            log_success(f"{self._tag} Previous episode deleted from LMDB.")
        else:
            log_warn(f"{self._tag} No saved episodes to undo.")

    def _discard_episode(self):
        n = len(self.dataset.data)
        self.dataset.reset()
        log_warn(
            f"{self._tag} Episode discarded ({n} keysteps dropped). Resetting robot ..."
        )
        self.env.robot.reset()
        self._reset_commanded_pose()
        log_success(f"{self._tag} Robot reset. Ready for next episode.")

    def _reset_to_home(self):
        log_info(f"{self._tag} Resetting robot to home pose ...")
        self.env.robot.reset()
        self._reset_commanded_pose()
        log_success(f"{self._tag} Robot at home pose.")

    def _finish_session(self):
        log_info(f"{self._tag} Finishing session ...")
        self.dataset.done()
        log_success(
            f"[{self.task}] Session finished. "
            f"{self.dataset.episode_idx} episode(s) saved to {self.dataset.output_dir}"
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
