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
    cache = batch.get('cache', None)
    print(f"\n=== {os.path.basename(path)} ===")
    print(f" Task:     {batch.get('task_str')}+{batch.get('variation')}")
    print(f" Episode:  {batch.get('episode_id')}")
    print(f" Step ID:  {batch.get('step_id')}")
    print(f" Keys:     {list(batch.keys())}\n")
    if cache is not None:
        # ['valid_actions', 'highlevel_plans', 'ret_objs', 'grasped_obj_name', 'prev_ee_pose', 'subtask_start_pose', 'is_retrying', 'highlevel_step_id', 'subtask_retry_id', 'highlevel_step_id_without_release', 'subtask_images', 'last_action', 'last_plan', 'history']
        print(f"cache keys: {list(cache.keys())}")
        print(f" cache['valid_actions']:", cache.get('valid_actions', 'N/A'))
        print(f" cache['highlevel_plans']:", cache.get('highlevel_plans', 'N/A'))
        print(f" cache['ret_objs']:", cache.get('ret_objs', 'N/A'))
        print(f" cache['grasped_obj_name']:", cache.get('grasped_obj_name', 'N/A'))
        print(f" cache['prev_ee_pose']:", cache.get('prev_ee_pose', 'N/A'))
        print(f" cache['subtask_start_pose']:", cache.get('subtask_start_pose', 'N/A'))
        print(f" cache['is_retrying']:", cache.get('is_retrying', 'N/A'))
        print(f" cache['highlevel_step_id']:", cache.get('highlevel_step_id', 'N/A'))
        print(f" cache['subtask_retry_id']:", cache.get('subtask_retry_id', 'N/A'))
        print(f" cache['highlevel_step_id_without_release']:", cache.get('highlevel_step_id_without_release', 'N/A'))
        print(f" cache['subtask_images']:", cache.get('subtask_images', 'N/A'))
        print(f" cache['last_action']:", cache.get('last_action', 'N/A'))
        print(f" cache['last_plan']:", cache.get('last_plan', 'N/A'))
        print(f" cache['history']:", cache.get('history', 'N/A'))


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
