#!/usr/bin/env python3
"""
Visualize a saved policy batch: point cloud with RGB colors and
end-effector pose overlay.

Usage:
  # Load a specific .msgpack file
  python3 viz_batch.py --batch_path /path/to/step-0.msgpack

  # Load latest step from a batch directory
  python3 viz_batch.py --batch_path /path/to/batch/

  # Load a specific step from a batch directory
  python3 viz_batch.py --batch_path /path/to/batch/ --step 3
"""
import argparse
import glob
import os

import msgpack_numpy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

msgpack_numpy.patch()


def load_batch(path: str, step: int = None) -> dict:
    """Load a batch from a .msgpack file or directory."""
    if os.path.isdir(path):
        if step is not None:
            filepath = os.path.join(path, f"step-{step}.msgpack")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No file found: {filepath}")
        else:
            files = sorted(glob.glob(os.path.join(path, "*.msgpack")))
            if not files:
                raise FileNotFoundError(f"No .msgpack files found in {path}")
            filepath = files[-1]
            print(f"Loading latest batch: {filepath}")
    else:
        filepath = path

    with open(filepath, "rb") as f:
        return msgpack_numpy.unpackb(f.read(), raw=False)


def build_eef_frame(gripper: np.ndarray, size: float = 0.1) -> o3d.geometry.TriangleMesh:
    """Build a coordinate frame at the end-effector pose.

    Args:
        gripper: (8,) array — pos(3), quat_xyzw(4), state(1)
        size: length of the frame axes
    """
    pos = gripper[:3]
    quat_xyzw = gripper[3:7]

    rot = Rotation.from_quat(quat_xyzw)  # scipy uses xyzw order
    rot_matrix = rot.as_matrix()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = pos
    frame.transform(T)
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a saved policy batch (point cloud + EEF pose)."
    )
    parser.add_argument(
        "--batch_path",
        type=str,
        required=True,
        help="Path to a .msgpack file or a batch/ directory (loads latest step).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step number to load when batch_path is a directory.",
    )
    args = parser.parse_args()

    batch = load_batch(args.batch_path, args.step)

    # --- Extract data ---
    obs = batch["obs_state_dict"]
    pc_data = obs["pc"]        # (C, 256, 256, 3) float32
    rgb_data = obs["rgb"]      # (C, 256, 256, 3) uint8
    gripper = obs["gripper"]   # (8,) — pos(3), quat(4), state(1)

    # --- Print metadata ---
    print(f"Task:          {batch.get('task_str', 'N/A')}")
    print(f"Variation:     {batch.get('variation', 'N/A')}")
    print(f"Step:          {batch.get('step_id', 'N/A')}")
    print(f"Episode:       {batch.get('episode_id', 'N/A')}")
    print(f"PC shape:      {pc_data.shape}")
    print(f"RGB shape:     {rgb_data.shape}")
    print(f"Gripper pos:   {gripper[:3]}")
    print(f"Gripper quat:  {gripper[3:7]}  (xyzw)")
    print(f"Gripper state: {gripper[7]}")

    # --- Build point cloud ---
    pc_points = pc_data.reshape(-1, 3)
    rgb_points = rgb_data.reshape(-1, 3)

    # Filter out zero (invalid) points
    valid_mask = np.any(pc_points != 0, axis=1)
    pc_points = pc_points[valid_mask]
    rgb_points = rgb_points[valid_mask]

    print(f"Valid points:  {pc_points.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb_points.astype(np.float64) / 255.0)

    # --- Build EEF coordinate frame ---
    eef_frame = build_eef_frame(gripper, size=0.1)

    # --- Origin coordinate frame ---
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    # --- Visualize ---
    o3d.visualization.draw_geometries(
        [pcd, eef_frame, origin_frame],
        window_name="Batch Visualization — Point Cloud + EEF Pose",
        width=1024,
        height=768,
    )


if __name__ == "__main__":
    main()
