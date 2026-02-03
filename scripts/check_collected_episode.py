#!/usr/bin/env python3
"""
Inspect and visualize episodes saved by collect_dataset.py.

1) Print data info (shapes, dtypes, actions, etc.)
2) Visualize keystep point clouds interactively with Open3D
   (left/right arrow keys to navigate keysteps)

Usage:
    # Print info for all episodes in a dataset
    python3 scripts/check_collected_episode.py /home/ppacaud/docker_shared/data/put_fruits_in_plates+0

    # Visualize episode 0 (navigate keysteps with arrow keys)
    python3 scripts/check_collected_episode.py /home/ppacaud/docker_shared/data/put_fruits_in_plates+0 --viz --episode 0
"""

import argparse
import sys

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()


def load_episodes(db_path: str) -> dict[str, dict]:
    """Load all episodes from an LMDB dataset."""
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    episodes = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            name = key.decode("ascii")
            data = msgpack.unpackb(value, raw=False)
            episodes[name] = data
    env.close()
    return episodes


def print_info(episodes: dict[str, dict]):
    """Print summary info for every episode."""
    print(f"Dataset contains {len(episodes)} episode(s)\n")

    for name in sorted(episodes.keys()):
        ep = episodes[name]
        print(f"=== {name} ===")
        print(f"  Keys: {list(ep.keys())}")

        for key in ("rgb", "xyz", "depth", "action"):
            arr = ep.get(key)
            if arr is not None:
                arr = np.asarray(arr)
                print(f"  {key:>8s}: shape={arr.shape}  dtype={arr.dtype}")

        n_keysteps = len(ep["action"]) if "action" in ep else "?"
        print(f"  keysteps: {n_keysteps}")

        # Actions per keystep
        if "action" in ep:
            actions = np.asarray(ep["action"])
            for i in range(len(actions)):
                pos = actions[i, :3]
                quat = actions[i, 3:7]
                grip = actions[i, 7] if actions.shape[1] > 7 else "N/A"
                print(
                    f"    keystep {i}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
                    f"quat=[{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]  "
                    f"gripper={grip}"
                )

        # Bbox info
        if "bbox_info" in ep:
            for i, bbox in enumerate(ep["bbox_info"]):
                print(f"    keystep {i} bbox_info: {bbox}")

        # Pose info
        if "pose_info" in ep:
            for i, pose in enumerate(ep["pose_info"]):
                print(f"    keystep {i} pose_info: {pose}")

        print()


def visualize_episode(ep_name: str, ep: dict):
    """Interactive point cloud viewer – use left/right arrow keys to navigate keysteps."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed. Install with: pip install open3d")
        sys.exit(1)

    xyz_all = np.asarray(ep["xyz"])
    rgb_all = np.asarray(ep["rgb"])
    actions = np.asarray(ep["action"])
    n_keysteps = len(xyz_all)

    if n_keysteps == 0:
        print(f"{ep_name} has no keysteps")
        return

    state = {"idx": 0, "geometries": []}

    def _make_geometries(idx):
        points = xyz_all[idx].reshape(-1, 3)
        colors = rgb_all[idx].reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
        points = points[valid]
        colors = colors[valid]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        pos = actions[idx][:3]
        eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        eef_sphere.translate(pos.astype(np.float64))
        eef_sphere.paint_uniform_color([1.0, 0.0, 0.0])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        return [pcd, eef_sphere, frame]

    def _print_keystep_info(idx):
        action = actions[idx]
        pos = action[:3]
        grip = action[7] if len(action) > 7 else "N/A"
        print(
            f"  {ep_name} | keystep {idx}/{n_keysteps - 1} | "
            f"pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}] | "
            f"gripper={grip}"
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

    # Build initial geometries
    state["geometries"] = _make_geometries(0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{ep_name} ({n_keysteps} keysteps) — arrow keys to navigate",
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
        description="Inspect and visualize collected episodes from LMDB"
    )
    parser.add_argument("db_path", help="Path to the LMDB dataset directory")
    parser.add_argument(
        "--viz", action="store_true", help="Visualize point clouds interactively"
    )
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index (default: 0)"
    )
    args = parser.parse_args()

    episodes = load_episodes(args.db_path)

    if not episodes:
        print(f"No episodes found in {args.db_path}")
        sys.exit(1)

    print_info(episodes)

    if args.viz:
        ep_key = f"episode{args.episode}"
        if ep_key not in episodes:
            print(f"Episode '{ep_key}' not found. Available: {sorted(episodes.keys())}")
            sys.exit(1)
        visualize_episode(ep_key, episodes[ep_key])


if __name__ == "__main__":
    main()
