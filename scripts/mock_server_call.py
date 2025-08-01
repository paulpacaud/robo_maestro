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

# Ensure numpy support for msgpack
m.patch()

def load_batch(path: str) -> dict:
    """Load a batch from a .msgpack file."""
    with open(path, "rb") as f:
        return m.unpackb(f.read(), raw=False)


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


if __name__ == "__main__":
    main()
