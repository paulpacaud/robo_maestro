#!/usr/bin/env python3
import glob
import os
import msgpack_numpy
import pprint

# ensure numpy support
msgpack_numpy.patch()

def load_batch(path):
    with open(path, "rb") as f:
        return msgpack_numpy.unpackb(f.read(), raw=False)

def inspect_one(path):
    batch = load_batch(path)
    print(f"\n=== {os.path.basename(path)} ===")
    print(f" Task:     {batch.get('task_str')}+{batch.get('variation')}")
    print(f" Episode:  {batch.get('episode_id')}")
    print(f" Step ID:  {batch.get('step_id')}")
    print(f" Keys:     {list(batch.keys())}\n")

    obs = batch['obs_state_dict']
    print(" → obs_state_dict keys:", list(obs.keys()))
    # eg. show shapes or types
    if 'rgb' in obs:
        print("    rgb shape:", obs['rgb'].shape, "dtype:", obs['rgb'].dtype)
    if 'pc' in obs:
        print("    pointcloud shape:", obs['pc'].shape)
    if 'gripper' in obs:
        print("    gripper pose+state:", obs['gripper'])
    if 'arm_links_info' in obs:
        bbox_info, pose_info = obs['arm_links_info']
        print("    #links in bbox_info:", len(bbox_info))
    print()

def main():
    folder = "/home/ppacaud/docker_shared/data/run_policy_experiments/real_hungry+0/episode_0/batch"
    files = sorted(glob.glob(os.path.join(folder, "*.msgpack")))
    if not files:
        print("No .msgpack files found in", folder)
        return

    for p in files:
        inspect_one(p)

if __name__ == "__main__":
    main()
