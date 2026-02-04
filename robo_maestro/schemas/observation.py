from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class ObsStateDict(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rgb: NDArray[np.uint8]  # (C, H, W, 3)
    pc: NDArray[np.float32]  # (C, H, W, 3)
    gripper: NDArray  # (8,) [x, y, z, qx, qy, qz, qw, gripper_state]
    arm_links_info: tuple[dict[str, Any], dict[str, Any]]  # (bbox_info, pose_info)
