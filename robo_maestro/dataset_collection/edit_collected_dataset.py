#!/usr/bin/env python3
"""
Remove one or more keysteps from an episode in an LMDB dataset
produced by collect_dataset.py.

The original database is modified IN PLACE – a backup of the episode
is printed to stdout before any change so you can verify.

Usage examples:

    # Remove the penultimate keystep (index -2) of episode 0
    python3 robo_maestro/dataset_collection/edit_collected_dataset.py \
        /home/ppacaud/docker_shared/data/ur5_put_grapes_and_banana_in_plates \
        --episode 0 --keystep -2

    # Remove keystep 3 of episode 2
    python3 robo_maestro/dataset_collection/edit_collected_dataset.py \
        /home/ppacaud/docker_shared/data/ur5_put_grapes_and_banana_in_plates \
        --episode 2 --keystep 3

    # Dry-run (show what would happen without writing)
    python3 robo_maestro/dataset_collection/edit_collected_dataset.py \
        /home/ppacaud/docker_shared/data/ur5_put_grapes_and_banana_in_plates \
        --episode 0 --keystep -2 --dry-run
"""

import argparse
import sys

import lmdb
import msgpack
import msgpack_numpy

from robo_maestro.schemas import GembenchKeystep, pack_keysteps, unpack_keysteps

msgpack_numpy.patch()


def load_episode(db_path: str, episode_key: str) -> list[GembenchKeystep] | None:
    env = lmdb.open(str(db_path), readonly=True, lock=False)
    with env.begin() as txn:
        raw = txn.get(episode_key.encode("ascii"))
    env.close()
    if raw is None:
        return None
    return unpack_keysteps(msgpack.unpackb(raw, raw=False))


def save_episode(db_path: str, episode_key: str, keysteps: list[GembenchKeystep]):
    env = lmdb.open(str(db_path), map_size=int(1024**4))
    with env.begin(write=True) as txn:
        txn.put(
            episode_key.encode("ascii"),
            msgpack.packb(pack_keysteps(keysteps)),
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


def print_episode_summary(episode_key: str, keysteps: list[GembenchKeystep]):
    """Print a compact summary of the episode."""
    if not keysteps:
        print("  (empty)")
        return

    ks0 = keysteps[0]
    print(
        f"  rgb:    shape=({len(keysteps)}, {', '.join(str(s) for s in ks0.rgb.shape)})  dtype={ks0.rgb.dtype}"
    )
    print(
        f"  xyz:    shape=({len(keysteps)}, {', '.join(str(s) for s in ks0.xyz.shape)})  dtype={ks0.xyz.dtype}"
    )
    print(
        f"  depth:  shape=({len(keysteps)}, {', '.join(str(s) for s in ks0.depth.shape)})  dtype={ks0.depth.dtype}"
    )
    print(
        f"  action: shape=({len(keysteps)}, {ks0.action.shape[0]})  dtype={ks0.action.dtype}"
    )
    print(f"  keysteps: {len(keysteps)}")

    for i, ks in enumerate(keysteps):
        pos = ks.action[:3]
        quat = ks.action[3:7]
        grip = ks.action[7] if len(ks.action) > 7 else "N/A"
        print(
            f"    keystep {i}: pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
            f"quat=[{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]  "
            f"gripper={grip}"
        )


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
    keysteps = load_episode(args.db_path, episode_key)
    if keysteps is None:
        available = list_episodes(args.db_path)
        print(f"Episode '{episode_key}' not found. Available: {available}")
        sys.exit(1)

    n_keysteps = len(keysteps)

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
    print_episode_summary(episode_key, keysteps)
    print()

    action = keysteps[keystep_idx].action
    pos = action[:3]
    grip = action[7] if len(action) > 7 else "N/A"
    print(
        f"Keystep to remove: {keystep_idx} "
        f"(pos=[{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}], gripper={grip})"
    )
    print()

    # --- Remove ---------------------------------------------------------------
    new_keysteps = list(keysteps)
    del new_keysteps[keystep_idx]

    print(f"=== AFTER: {episode_key} ({len(new_keysteps)} keysteps) ===")
    print_episode_summary(episode_key, new_keysteps)
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
    save_episode(args.db_path, episode_key, new_keysteps)
    print(f"Done. Keystep {keystep_idx} removed from {episode_key}.")


if __name__ == "__main__":
    main()
