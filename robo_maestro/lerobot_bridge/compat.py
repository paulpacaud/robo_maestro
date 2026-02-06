"""Minimal copies of lerobot dataclasses for pickle-compatible gRPC communication.

These classes mirror their lerobot counterparts exactly, with __module__ overridden
so that pickle serialization writes the correct class paths that the LeRobot policy
server can resolve.

No external dependencies beyond stdlib + torch.
"""

from dataclasses import dataclass, field
from enum import Enum

import torch


# --- From lerobot/configs/types.py ---

class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"


FeatureType.__module__ = "lerobot.configs.types"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]


PolicyFeature.__module__ = "lerobot.configs.types"


# --- From lerobot/async_inference/helpers.py ---

Action = torch.Tensor
RawObservation = dict[str, torch.Tensor]


@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


TimedData.__module__ = "lerobot.async_inference.helpers"


@dataclass
class TimedAction(TimedData):
    action: Action

    def get_action(self):
        return self.action


TimedAction.__module__ = "lerobot.async_inference.helpers"


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False

    def get_observation(self):
        return self.observation


TimedObservation.__module__ = "lerobot.async_inference.helpers"


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, dict]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)


RemotePolicyConfig.__module__ = "lerobot.async_inference.helpers"
