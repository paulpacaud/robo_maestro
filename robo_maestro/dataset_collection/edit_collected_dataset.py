#!/usr/bin/env python3
"""
Remove one or more keysteps from an episode, or remove an entire episode
from an LMDB dataset produced by collect_dataset.py.

The original database is modified IN PLACE – a backup of the episode
is printed to stdout before any change so you can verify.

Usage examples:

    # Remove the penultimate keystep (index -2) of episode 0
    python3 -m robo_maestro.dataset_collection.edit_collected_dataset \
        /home/ppacaud/docker_shared/data/ur5_stack_yellow_onto_pink_cup \
        --episode 0 --keystep -2

    # Remove keystep 3 of episode 2
    python3 -m robo_maestro.dataset_collection.edit_collected_dataset \
        /home/ppacaud/docker_shared/data/ur5_put_grapes_and_banana_in_plates \
        --episode 2 --keystep 3

    # Remove entire episode 3 and re-index subsequent episodes
    python3 -m robo_maestro.dataset_collection.edit_collected_dataset \
        /home/ppacaud/docker_shared/data/ur5_stack_yellow_onto_pink_cup \
        --episode 2

    # Dry-run (show what would happen without writing)
    python3 -m robo_maestro.dataset_collection.edit_collected_dataset \
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


def delete_episode(db_path: str, episode_key: str):
    env = lmdb.open(str(db_path), map_size=int(1024**4))
    with env.begin(write=True) as txn:
        txn.delete(episode_key.encode("ascii"))
    env.close()


def rename_episode(db_path: str, old_key: str, new_key: str):
    env = lmdb.open(str(db_path), map_size=int(1024**4))
    with env.begin(write=True) as txn:
        raw = txn.get(old_key.encode("ascii"))
        if raw is not None:
            txn.put(new_key.encode("ascii"), raw)
            txn.delete(old_key.encode("ascii"))
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
        description="Remove a keystep from an episode, or remove an entire "
        "episode (with re-indexing) from an LMDB dataset",
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
        default=None,
        help="Keystep index to remove. Supports negative indexing "
        "(e.g. -2 for penultimate). If omitted, the entire episode "
        "is removed and subsequent episodes are re-indexed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying the dataset",
    )
    args = parser.parse_args()

    episode_key = f"episode{args.episode}"
    all_episodes = list_episodes(args.db_path)

    if episode_key not in all_episodes:
        print(f"Episode '{episode_key}' not found. Available: {all_episodes}")
        sys.exit(1)

    # =========================================================================
    # Mode 1: Remove an entire episode and re-index
    # =========================================================================
    if args.keystep is None:
        print(f"=== Episodes in dataset ({len(all_episodes)} total) ===")
        for ep_key in all_episodes:
            ks = load_episode(args.db_path, ep_key)
            print(f"  {ep_key}: {len(ks)} keysteps")
        print()

        # Figure out which episodes need to be shifted
        ep_idx = args.episode
        episodes_to_shift = [
            k
            for k in all_episodes
            if k.startswith("episode") and int(k[len("episode") :]) > ep_idx
        ]
        episodes_to_shift.sort(key=lambda k: int(k[len("episode") :]))

        print(f"Will remove: {episode_key}")
        if episodes_to_shift:
            renames = [
                f"  {old} -> episode{int(old[len('episode'):]) - 1}"
                for old in episodes_to_shift
            ]
            print("Will re-index:")
            print("\n".join(renames))
        print()

        print(f"=== After removal: {len(all_episodes) - 1} episodes ===")
        for ep_key_iter in all_episodes:
            if ep_key_iter == episode_key:
                continue
            old_idx = int(ep_key_iter[len("episode") :])
            new_idx = old_idx - 1 if old_idx > ep_idx else old_idx
            ks = load_episode(args.db_path, ep_key_iter)
            print(f"  episode{new_idx}: {len(ks)} keysteps")
        print()

        if args.dry_run:
            print("Dry-run mode – no changes written.")
            return

        answer = input("Write changes? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

        # Delete the target episode
        delete_episode(args.db_path, episode_key)

        # Rename subsequent episodes in ascending order to avoid collisions
        for old_key in episodes_to_shift:
            old_idx = int(old_key[len("episode") :])
            new_key = f"episode{old_idx - 1}"
            rename_episode(args.db_path, old_key, new_key)

        print(
            f"Done. Removed {episode_key} and re-indexed "
            f"{len(episodes_to_shift)} subsequent episode(s)."
        )
        return

    # =========================================================================
    # Mode 2: Remove a single keystep from an episode
    # =========================================================================
    keysteps = load_episode(args.db_path, episode_key)
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
