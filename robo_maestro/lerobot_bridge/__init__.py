"""LeRobot bridge package — self-contained gRPC client for LeRobot policy servers.

Sets up sys.modules aliases so that pickle (de)serialization is compatible with
the LeRobot server, which imports from ``lerobot.async_inference.helpers`` and
``lerobot.configs.types``.

Import order matters:
  1. Register services_pb2 under ``lerobot.transport.services_pb2`` first.
  2. Then import services_pb2_grpc (which does a relative import of services_pb2).
  3. Then register compat classes under ``lerobot.async_inference.helpers``
     and ``lerobot.configs.types``.
"""

import sys
import types

# --- 1. Ensure parent module stubs exist in sys.modules ---
for _mod_path in [
    "lerobot",
    "lerobot.transport",
    "lerobot.async_inference",
    "lerobot.configs",
]:
    if _mod_path not in sys.modules:
        sys.modules[_mod_path] = types.ModuleType(_mod_path)

# --- 2. Register protobuf module under lerobot.transport ---
from . import services_pb2  # noqa: E402

sys.modules["lerobot.transport.services_pb2"] = services_pb2

# --- 3. Import gRPC stubs (depends on services_pb2 being registered) ---
from . import services_pb2_grpc  # noqa: E402, F401

# --- 4. Register compat classes for pickle resolution ---
from . import compat  # noqa: E402

sys.modules["lerobot.async_inference.helpers"] = compat
sys.modules["lerobot.configs.types"] = compat
