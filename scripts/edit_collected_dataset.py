#!/usr/bin/env python3
"""
Remove one or more keysteps from an episode in an LMDB dataset
produced by collect_dataset.py.

The original database is modified IN PLACE – a backup of the episode
is printed to stdout before any change so you can verify.

Usage examples:

    # Remove the penultimate keystep (index -2) of episode 0
    python3 scripts/edit_collected_dataset.py \
        /home/ppacaud/docker_shared/data/put_fruits_in_plates+0 \
        --episode 0 --keystep -2

    # Remove keystep 3 of episode 2
    python3 scripts/edit_collected_dataset.py \
        /home/ppacaud/docker_shared/data/put_fruits_in_plates+0 \
        --episode 2 --keystep 3

    # Dry-run (show what would happen without writing)
    python3 scripts/edit_collected_dataset.py \
        /home/ppacaud/docker_shared/data/put_fruits_in_plates+0 \
        --episode 0 --keystep -2 --dry-run
"""

import argparse
import sys

import lmdb
import msgpack
import msgpack_numpy
import numpy as np

msgpack_numpy.patch()

# Keys whose values are numpy arrays indexed along the first axis (keystep axis)
ARRAY_KEYS = ("rgb", "xyz", "depth", "action")
# Keys whose values are plain Python lists indexed per keystep
LIST_KEYS = ("bbox_info", "pose_info")


def load_episode(db_path: str, episode_key: str) -> dict:
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    with env.begin() as txn:
        raw = txn.get(episode_key.encode("ascii"))
    env.close()
    if raw is None:
        return None
    return msgpack.unpackb(raw, raw=False)


def save_episode(db_path: str, episode_key: str, episode: dict):
    env = lmdb.open(str(db_path), map_size=int(1024**4))
    with env.begin(write=True) as txn:
        txn.put(
            episode_key.encode("ascii"),
            msgpack.packb(episode),
        )
    env.close()


def list_episodes(db_path: str) -> list[str]:
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    keys = []
    with env.begin() as txn:
        for key, _ in txn.cursor():
            keys.append(key.decode("ascii"))
    env.close()
    return sorted(keys)


def print_episode_summary(episode_key: str, ep: dict):
    """Print a compact summary of the episode."""
    for key in ARRAY_KEYS:
        arr = ep.get(key)
        if arr is not None:
            arr = np.asarray(arr)
            print(f"  {key:>8s}: shape={arr.shape}  dtype={arr.dtype}")

    if "action" in ep:
        actions = np.asarray(ep["action"])
        n = len(actions)
        print(f"  keysteps: {n}")
        for i in range(n):
            pos = actions[i, :3]
            quat = actions[i, 3:7]
            grip = actions[i, 7] if actions.shape[1] > 7 else "N/A"
            print(
                f"    keystep {i}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
                f"quat=[{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]  "
                f"gripper={grip}"
            )


def remove_keystep(ep: dict, idx: int) -> dict:
    """Return a new episode dict with the given keystep index removed."""
    out = {}
    for key in ARRAY_KEYS:
        if key in ep:
            arr = np.asarray(ep[key])
            out[key] = np.delete(arr, idx, axis=0)
    for key in LIST_KEYS:
        if key in ep:
            lst = list(ep[key])
            del lst[idx]
            out[key] = lst
    # Preserve any other keys unchanged
    for key in ep:
        if key not in out:
            out[key] = ep[key]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Remove a keystep from a collected episode (LMDB dataset)"
    )
    parser.add_argument("db_path", help="Path to the LMDB dataset directory")
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index (e.g. 0 for episode0)",
    )
    parser.add_argument(
        "--keystep",
        type=int,
        required=True,
        help="Keystep index to remove. Supports negative indexing "
        "(e.g. -2 for penultimate)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying the dataset",
    )
    args = parser.parse_args()

    episode_key = f"episode{args.episode}"

    # --- Load ----------------------------------------------------------------
    ep = load_episode(args.db_path, episode_key)
    if ep is None:
        available = list_episodes(args.db_path)
        print(f"Episode '{episode_key}' not found. Available: {available}")
        sys.exit(1)

    actions = np.asarray(ep["action"])
    n_keysteps = len(actions)

    # Resolve negative index
    keystep_idx = args.keystep
    if keystep_idx < 0:
        keystep_idx = n_keysteps + keystep_idx
    if keystep_idx < 0 or keystep_idx >= n_keysteps:
        print(
            f"Keystep index {args.keystep} is out of range for {n_keysteps} keysteps "
            f"(valid range: 0..{n_keysteps - 1} or -{n_keysteps}..-1)"
        )
        sys.exit(1)

    # --- Show before ----------------------------------------------------------
    print(f"=== BEFORE: {episode_key} ({n_keysteps} keysteps) ===")
    print_episode_summary(episode_key, ep)
    print()

    action = actions[keystep_idx]
    pos = action[:3]
    grip = action[7] if len(action) > 7 else "N/A"
    print(
        f"Keystep to remove: {keystep_idx} "
        f"(pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}], gripper={grip})"
    )
    print()

    # --- Remove ---------------------------------------------------------------
    new_ep = remove_keystep(ep, keystep_idx)

    print(f"=== AFTER: {episode_key} ({n_keysteps - 1} keysteps) ===")
    print_episode_summary(episode_key, new_ep)
    print()

    if args.dry_run:
        print("Dry-run mode – no changes written.")
        return

    # --- Confirm --------------------------------------------------------------
    answer = input("Write changes? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    # --- Save -----------------------------------------------------------------
    save_episode(args.db_path, episode_key, new_ep)
    print(f"Done. Keystep {keystep_idx} removed from {episode_key}.")


if __name__ == "__main__":
    main()
