#!/usr/bin/env python3
"""
Mock a call to the RoboMaestro policy server using a saved batch file.
This script loads the first .msgpack from a specified folder and either sends it
over HTTP to the real policy server or invokes the mock_predict method locally.
"""
import glob
import os
import argparse
import msgpack_numpy as m
import requests
import numpy as np
import open3d as o3d

# Ensure numpy support for msgpack
m.patch()


def load_batch(path: str) -> dict:
    """Load a batch from a .msgpack file."""
    with open(path, "rb") as f:
        return m.unpackb(f.read(), raw=False)


def visualize_pc_with_action(pc_data, rgb_data, action_pos):
    """
    Visualize point cloud with RGB colors and action position as a red sphere.

    Args:
        pc_data: Point cloud data, shape (C, H, W, 3)
        rgb_data: RGB data, shape (C, H, W, 3)
        action_pos: Action position [x, y, z]
    """
    # Reshape point cloud and RGB data if needed
    if len(pc_data.shape) == 4:
        # Assuming shape is (num_cameras, height, width, 3)
        # Flatten and combine all camera point clouds
        pc_points = pc_data.reshape(-1, 3)
        rgb_points = rgb_data.reshape(-1, 3)
    else:
        pc_points = pc_data
        rgb_points = rgb_data

    # Remove invalid points (e.g., zeros from depth camera)
    valid_mask = np.any(pc_points != 0, axis=1)
    pc_points = pc_points[valid_mask]
    rgb_points = rgb_points[valid_mask]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points)

    # Normalize RGB values to [0, 1] range
    colors = rgb_points.astype(np.float32) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create a red sphere for the action position
    action_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    action_sphere.translate(action_pos)
    action_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color

    # Create coordinate frame at origin for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    # Visualize
    o3d.visualization.draw_geometries(
        [pcd, action_sphere, coord_frame],
        window_name="Point Cloud with RGB Colors and Predicted Action (Red Sphere)",
        width=1024,
        height=768
    )


def main():
    parser = argparse.ArgumentParser(
        description="Mock a call to the policy server using a batch file."
    )
    parser.add_argument(
        "--batch_folder",
        type=str,
        default="/home/ppacaud/docker_shared/data/run_policy_experiments/real_hungry+0/episode_0/batch",
        help="Path to the batch folder containing .msgpack files",
    )
    parser.add_argument(
        "--server_addr",
        type=str,
        default="127.0.0.1:8002",
        help="Policy server address (host:port)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the point cloud and action",
    )
    args = parser.parse_args()

    # Find .msgpack files
    files = sorted(glob.glob(os.path.join(args.batch_folder, "*.msgpack")))
    if not files:
        print(f"No .msgpack files found in {args.batch_folder}")
        return

    batch_file = files[0]
    print(f"Using batch file: {batch_file}")

    batch = load_batch(batch_file)
    print("Batch keys:", list(batch.keys()))

    step_id = batch.get("step_id")

    # Extract point cloud and RGB data for visualization
    obs = batch.get("obs_state_dict", {})
    pc_data = obs.get("pc")
    rgb_data = obs.get("rgb")

    if pc_data is not None:
        print(f"Point cloud shape: {pc_data.shape}")
    else:
        print("No point cloud data found in batch")

    if rgb_data is not None:
        print(f"RGB data shape: {rgb_data.shape}")
    else:
        print("No RGB data found in batch")

    # Send to the real policy server over HTTP
    data = m.packb(batch)
    url = f"http://{args.server_addr}/predict"
    print(f"Sending POST to {url} ...")
    try:
        response = requests.post(url, data=data)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return

    if not response.ok:
        print(f"Server returned status code {response.status_code}")
        print(response.text)
        return

    output = m.unpackb(response.content, raw=False)
    action = output.get("action")
    cache = output.get("cache")

    print("\n=== Policy Server Response ===")
    print("Action:", action)
    print("Cache:", cache)

    # Visualize if requested and data is available
    if args.visualize and pc_data is not None and rgb_data is not None and action is not None:
        action_pos = action[:3]  # First 3 elements are x, y, z
        print(f"\nVisualizing point cloud with action position: {action_pos}")
        visualize_pc_with_action(pc_data, rgb_data, action_pos)
    elif args.visualize:
        missing = []
        if pc_data is None:
            missing.append("point cloud")
        if rgb_data is None:
            missing.append("RGB")
        if action is None:
            missing.append("action")
        print(f"\nCannot visualize: missing {', '.join(missing)} data")


if __name__ == "__main__":
    main()