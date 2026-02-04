from pydantic import BaseModel


class CameraIntrinsics(BaseModel):
    height: int
    width: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    K: list[list[float]]  # 3x3 intrinsic matrix


class CameraExtrinsics(BaseModel):
    pos: list[float]  # (3,) [x, y, z] metres
    euler: list[float]  # (3,) [roll, pitch, yaw] radians
    world_T_cam: list[list[float]]  # (4, 4) homogeneous transform


class CameraConfig(BaseModel):
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
