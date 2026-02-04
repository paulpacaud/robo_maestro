"""
The GemBench format stores demos as keystep-based episodes in LMDB, serialized with msgpack (+msgpack_numpy).

Data layout in LMDB
-------------------
    key:   b"episode{idx}"  (ASCII)
    value: msgpack.packb(pack_keysteps(keysteps))
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class GembenchKeystep(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # arbitrary_types_allowed=True tells Pydantic to accept field types it doesn't know how to validate natively — in this case NDArray (numpy arrays). Without it, Pydantic would raise an error at model definition because it doesn'thave a built-in validator for numpy.ndarray

    rgb: NDArray[np.uint8]  # (C, H, W, 3)
    xyz: NDArray[np.float32]  # (C, H, W, 3)
    depth: NDArray[np.float32]  # (C, H, W)
    action: NDArray[np.float32]  # (8,) [x, y, z, qx, qy, qz, qw, gripper_state]
    bbox_info: dict[
        str, Any
    ]  # "{link}_bbox" -> [x_min, x_max, y_min, y_max, z_min, z_max]
    pose_info: dict[str, Any]  # "{link}_pose" -> [x, y, z, qx, qy, qz, qw]


def pack_keysteps(keysteps: list[GembenchKeystep]) -> dict:
    """Stack keysteps into the LMDB on-disk format (struct-of-arrays)."""
    return {
        "rgb": np.stack([ks.rgb for ks in keysteps]).astype(np.uint8),
        "xyz": np.stack([ks.xyz for ks in keysteps]).astype(np.float32),
        "depth": np.stack([ks.depth for ks in keysteps]).astype(np.float32),
        "action": np.stack([ks.action for ks in keysteps]).astype(np.float32),
        "bbox_info": [ks.bbox_info for ks in keysteps],
        "pose_info": [ks.pose_info for ks in keysteps],
    }


def unpack_keysteps(data: dict) -> list[GembenchKeystep]:
    """Split a stacked LMDB dict back into individual keysteps."""
    rgb = np.asarray(data["rgb"]).astype(np.uint8)
    xyz = np.asarray(data["xyz"]).astype(np.float32)
    depth = np.asarray(data["depth"]).astype(np.float32)
    action = np.asarray(data["action"]).astype(np.float32)
    bbox_info = list(data["bbox_info"])
    pose_info = list(data["pose_info"])
    return [
        GembenchKeystep(
            rgb=rgb[i],
            xyz=xyz[i],
            depth=depth[i],
            action=action[i],
            bbox_info=bbox_info[i],
            pose_info=pose_info[i],
        )
        for i in range(len(action))
    ]
