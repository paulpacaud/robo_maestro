#!/usr/bin/env python3
"""
Load a training dataset and visualize its point clouds cropped to the workspace bounding box.

Navigate keysteps with left/right arrow keys.

Usage:
    python3 -m robo_maestro.dev_tools.check_workspace_bbox /path/to/lmdb_dataset
    python3 -m robo_maestro.dev_tools.check_workspace_bbox /home/ppacaud/docker_shared/data/ur5_put_grapes_and_banana_in_plates --episode 2
"""

import argparse
import sys

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

from robo_maestro.schemas import GembenchKeystep, unpack_keysteps

msgpack_numpy.patch()

# ── Workspace bounding box ──────────────────────────────────────────────────
X_BBOX = (-0.55, 0.15)
Y_BBOX = (-0.45, 0.45)
Z_BBOX = (-0.01, 0.75)


def load_episodes(db_path: str) -> dict[str, list[GembenchKeystep]]:
    """Load all episodes from an LMDB dataset."""
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    episodes: dict[str, list[GembenchKeystep]] = {}
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            name = key.decode("ascii")
            data = msgpack.unpackb(value, raw=False)
            episodes[name] = unpack_keysteps(data)
    env.close()
    return episodes


def crop_to_workspace(points: np.ndarray, colors: np.ndarray):
    """Keep only points inside the workspace bounding box."""
    mask = (
        (points[:, 0] >= X_BBOX[0])
        & (points[:, 0] <= X_BBOX[1])
        & (points[:, 1] >= Y_BBOX[0])
        & (points[:, 1] <= Y_BBOX[1])
        & (points[:, 2] >= Z_BBOX[0])
        & (points[:, 2] <= Z_BBOX[1])
    )
    return points[mask], colors[mask]


def visualize_episode(ep_name: str, keysteps: list[GembenchKeystep]):
    """Interactive point-cloud viewer with workspace bbox cropping."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed. Install with: pip install open3d")
        sys.exit(1)

    n_keysteps = len(keysteps)
    if n_keysteps == 0:
        print(f"{ep_name} has no keysteps")
        return

    state = {"idx": 0, "geometries": []}

    def _make_bbox_lineset():
        """Create a wireframe box showing the workspace bounds."""
        x0, x1 = X_BBOX
        y0, y1 = Y_BBOX
        z0, z1 = Z_BBOX
        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # vertical
        ]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([0.0, 1.0, 0.0])
        return ls

    def _make_geometries(idx):
        ks = keysteps[idx]
        points = ks.xyz.reshape(-1, 3)
        colors = ks.rgb.reshape(-1, 3)

        # Remove invalid points
        valid = np.isfinite(points).all(axis=1) & (np.abs(points).sum(axis=1) > 1e-6)
        points = points[valid]
        colors = colors[valid]

        # Crop to workspace
        points, colors = crop_to_workspace(points, colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        # Current EEF pose (blue sphere)
        eef_pos = ks.action[:3]
        eef_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        eef_sphere.translate(eef_pos.astype(np.float64))
        eef_sphere.paint_uniform_color([0.0, 0.0, 1.0])

        geoms = [pcd, eef_sphere, _make_bbox_lineset()]

        # Target action = next keystep (red sphere)
        if idx + 1 < n_keysteps:
            target_pos = keysteps[idx + 1].action[:3]
            target_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            target_sphere.translate(target_pos.astype(np.float64))
            target_sphere.paint_uniform_color([1.0, 0.0, 0.0])
            geoms.append(target_sphere)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        geoms.append(frame)

        return geoms

    def _print_keystep_info(idx):
        ks = keysteps[idx]
        eef_pos = ks.action[:3]
        grip = ks.action[7] if len(ks.action) > 7 else "N/A"
        info = (
            f"  {ep_name} | keystep {idx}/{n_keysteps - 1} | "
            f"eef=[{eef_pos[0]:+.4f}, {eef_pos[1]:+.4f}, {eef_pos[2]:+.4f}] | "
            f"gripper={grip}"
        )
        if idx + 1 < n_keysteps:
            target_pos = keysteps[idx + 1].action[:3]
            info += f" | target=[{target_pos[0]:+.4f}, {target_pos[1]:+.4f}, {target_pos[2]:+.4f}]"
        else:
            info += " | target=N/A (last keystep)"
        print(info)

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
        window_name=f"{ep_name} ({n_keysteps} keysteps) — workspace bbox crop",
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
    print(
        f"  Workspace bbox: X={X_BBOX}, Y={Y_BBOX}, Z={Z_BBOX}\n"
        f"  Use left/right arrow keys to navigate keysteps. Close window to exit."
    )
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize collected episodes cropped to the workspace bounding box"
    )
    parser.add_argument("db_path", help="Path to the LMDB dataset directory")
    parser.add_argument(
        "--episode", type=int, default=0, help="Episode index (default: 0)"
    )
    args = parser.parse_args()

    episodes = load_episodes(args.db_path)

    if not episodes:
        print(f"No episodes found in {args.db_path}")
        sys.exit(1)

    print(f"Dataset contains {len(episodes)} episode(s): {sorted(episodes.keys())}")

    ep_key = f"episode{args.episode}"
    if ep_key not in episodes:
        print(f"Episode '{ep_key}' not found. Available: {sorted(episodes.keys())}")
        sys.exit(1)

    visualize_episode(ep_key, episodes[ep_key])


if __name__ == "__main__":
    main()
