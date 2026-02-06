#!/usr/bin/env python3
"""
Visualize episodes saved by run_policy_pointact.py (policy evaluation format).

The policy eval format stores keysteps as individual .npy files (pickled dicts)
with keys: rgb, pc, gripper, arm_links_info, action.

Usage:
    # Print info
    python3 -m robo_maestro.dev_tools.viz_policy_eval \
        ~/docker_shared/data/run_policy_experiments/ur5_put_grapes_and_banana_in_plates+0_0205T125947/episode_0

    # Visualize with Open3D point cloud viewer (arrow keys to navigate)
    python3 -m robo_maestro.dev_tools.viz_policy_eval \
        ~/docker_shared/data/run_policy_experiments/ur5_close_drawer+0/episode_0 \
        --viz
"""

import argparse
import glob
import os
import sys

import numpy as np


def load_keysteps(episode_dir: str) -> list[dict]:
    """Load all keystep .npy files from an episode directory."""
    keysteps_dir = os.path.join(episode_dir, "keysteps")
    if not os.path.isdir(keysteps_dir):
        print(f"No keysteps/ directory found in {episode_dir}")
        sys.exit(1)

    files = sorted(
        glob.glob(os.path.join(keysteps_dir, "*.npy")),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0]),
    )

    keysteps = []
    for f in files:
        data = np.load(f, allow_pickle=True).item()
        keysteps.append(data)
    return keysteps


def print_info(keysteps: list[dict], episode_dir: str):
    """Print summary info for the episode."""
    print(f"Episode: {episode_dir}")
    print(f"  keysteps: {len(keysteps)}\n")

    if keysteps:
        ks0 = keysteps[0]
        for field in ("rgb", "pc", "action", "gripper"):
            if field in ks0:
                arr = np.asarray(ks0[field])
                print(f"  {field:>8s}: shape={arr.shape}  dtype={arr.dtype}")
        print()

    for i, ks in enumerate(keysteps):
        action = np.asarray(ks["action"])
        pos = action[:3]
        quat = action[3:7]
        grip = action[7] if len(action) > 7 else "N/A"
        print(
            f"  keystep {i}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
            f"quat=[{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]  "
            f"gripper={grip}"
        )
    print()


def visualize_episode(keysteps: list[dict], episode_dir: str):
    """Interactive point cloud viewer -- use left/right arrow keys to navigate keysteps."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed. Install with: pip install open3d")
        sys.exit(1)

    n_keysteps = len(keysteps)
    if n_keysteps == 0:
        print("No keysteps to visualize")
        return

    state = {"idx": 0, "geometries": []}

    def _make_geometries(idx):
        ks = keysteps[idx]
        # pc and rgb have shape (1, H, W, 3) in policy eval format
        points = np.asarray(ks["pc"]).reshape(-1, 3)
        colors = np.asarray(ks["rgb"]).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
        points = points[valid]
        colors = colors[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        # Current EEF pose (blue)
        gripper = np.asarray(ks["gripper"])
        eef_pos = gripper[:3]
        eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        eef_sphere.translate(eef_pos.astype(np.float64))
        eef_sphere.paint_uniform_color([0.0, 0.0, 1.0])

        # Target action (red)
        action = np.asarray(ks["action"])
        action_pos = action[:3]
        action_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        action_sphere.translate(action_pos.astype(np.float64))
        action_sphere.paint_uniform_color([1.0, 0.0, 0.0])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        return [pcd, eef_sphere, action_sphere, frame]

    def _print_keystep_info(idx):
        ks = keysteps[idx]
        gripper = np.asarray(ks["gripper"])
        eef_pos = gripper[:3]
        action = np.asarray(ks["action"])
        act_pos = action[:3]
        act_grip = action[7] if len(action) > 7 else "N/A"
        print(
            f"  keystep {idx}/{n_keysteps - 1} | \n"
            f"eef={gripper[0]:+.3f}, {gripper[1]:+.3f}, {gripper[2]:+.3f}, {gripper[3]:+.3f}, {gripper[4]:+.3f}, {gripper[5]:+.3f}, {gripper[6]:+.3f}, {gripper[7]:+.3f} | \n"
            f"action={action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}, {action[3]:+.3f}, {action[4]:+.3f}, {action[5]:+.3f}, {action[6]:+.3f}, {act_grip}"
        )

    def _switch_keystep(vis, new_idx):
        if new_idx < 0 or new_idx >= n_keysteps:
            return False
        for g in state["geometries"]:
            vis.remove_geometry(g, reset_bounding_box=False)
        state["idx"] = new_idx
        state["geometries"] = _make_geometries(new_idx)
        for g in state["geometries"]:
            vis.add_geometry(g, reset_bounding_box=False)
        _print_keystep_info(new_idx)
        return False

    def _on_right(vis):
        return _switch_keystep(vis, state["idx"] + 1)

    def _on_left(vis):
        return _switch_keystep(vis, state["idx"] - 1)

    state["geometries"] = _make_geometries(0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Policy Eval ({n_keysteps} keysteps) — arrow keys to navigate",
        width=1280,
        height=720,
    )
    for g in state["geometries"]:
        vis.add_geometry(g)

    # GLFW key codes: Right=262, Left=263
    vis.register_key_callback(262, _on_right)
    vis.register_key_callback(263, _on_left)

    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_front([-1.0, 2.0, 1.0])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(0.5)

    _print_keystep_info(0)
    print("  Use left/right arrow keys to navigate keysteps. Close window to exit.")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize policy evaluation episodes (npy format)"
    )
    parser.add_argument(
        "episode_dir", help="Path to the episode directory (containing keysteps/)"
    )
    parser.add_argument(
        "--viz", action="store_true", help="Visualize point clouds interactively"
    )
    args = parser.parse_args()

    keysteps = load_keysteps(args.episode_dir)

    if not keysteps:
        print(f"No keysteps found in {args.episode_dir}")
        sys.exit(1)

    print_info(keysteps, args.episode_dir)

    if args.viz:
        visualize_episode(keysteps, args.episode_dir)


if __name__ == "__main__":
    main()
