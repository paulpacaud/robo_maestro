from pydantic import BaseModel

from robo_maestro.schemas.camera import CameraConfig


class CollectedDatasetMeta(BaseModel):
    task: str
    task_instruction: str | list[str]
    cam_list: list[str]
    cameras: dict[str, CameraConfig]
