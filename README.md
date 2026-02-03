# RoboMaestro

**RoboMaestro** is a ROS 2 Python package for controlling the **Paris Robotics Lab's bimanual UR5 ("Mantis")** system.  
It acts as a bridge between **remote policy inference servers** (e.g., CLEPS/GenRobot3D) and the real or simulated robot, encapsulating all robot-specific concerns such as:

- Observation gathering (RGB-D, point clouds, robot state)
- Action execution (MoveIt motion planning + UR5 controllers)
- Gripper control
- Integration with remote policy inference over HTTP

The package provides:
- A **ROS 2 launch interface** for starting the control loop
- A **Gymnasium-compatible environment** (`RealRobot-BaseEnv`)
- Utilities for **camera, TF, and point cloud processing**
- Support for both **real hardware** and **Gazebo simulation**

---

## Features

- **Real and simulated execution** via `use_sim_time` parameter
- Automatic MoveIt configuration loading
- Synchronous data acquisition from multiple RGB-D cameras
- End-effector and link pose acquisition through TF
- Point cloud generation and transformation
- Workspace safety constraints on Cartesian motions
- Gripper actuation (low-force or normal mode)
- Modular policy server integration via HTTP (`PolicyServer`)
- Mock mode for debugging without a remote policy server

---

## Installation

### Prerequisites
- ROS 2 Humble or later (Jazzy recommended for Docker)
- MoveIt 2 installed and working with the UR5
- `colcon` build system
- Python 3.10+
- Docker (if using provided dockerized ROS environment)
- UR5 bimanual setup (real or simulated)
- Cameras configured (Realsense or Orbbec)

### Clone and Build
```bash
cd ~/share/ws/src
git clone https://github.com/inria-paris-robotics-lab/prl_ur5_ros2.git
```

Add to the Dockerfile robo_maestro's dependencies:
```Dockerfile
########## ROBO MAESTRO DEPENDENCIES ##########
# Install robo_maetro's python dependencies in docker
sudo apt-get update \
&& sudo apt-get install -y python3-pip \
&& sudo python3 -m pip install --break-system-packages --ignore-installed \
torch torchvision gymnasium typed-argument-parser \
msgpack-numpy easydict scipy open3d requests \
transforms3d setuptools
pip3 install --break-system-packages "setuptools==70.0.0"

# Ensure the wheel we just installed shadows the APT copy
PYPLIB=$(python3 -c "import sysconfig,sys;print(sysconfig.get_paths()['platlib'])") \
 && echo "export PYTHONPATH=${PYPLIB}:\$PYTHONPATH" >> sudo /etc/profile.d/pythonpath.sh

# Make it available in non‑interactive shells (docker exec, launch files)
PYTHONPATH=${PYPLIB}:${PYTHONPATH}
########## ROBO MAESTRO DEPENDENCIES ##########
```


```

# Include robo_maestro in the same workspace:
git clone <this_repo_url>
cd ~/share/ws
colcon build --symlink-install --packages-skip robotiq_ft_sensor_hardware
source install/setup.bash
```

---

## Configuration

### Planner Configuration
Planner parameters are stored in:
```
robo_maestro/config/planner.yaml
```
This file defines available pipelines (`ompl`, `pilz_industrial_motion_planner`, `chomp`) and tuning parameters for each.

### Camera Info
Known camera positions/orientations are hardcoded in:
```
robo_maestro/utils/constants.py
```
For unknown cameras, RoboMaestro will query TF at runtime.

### Network & Robot Configuration
Ensure the `prl_ur5_robot_configuration/config/standart_setup.yaml` file matches your real or simulated robot IPs.

---

## Usage

### Launch the Robot (Real or Sim)
```bash
# Simulation:
ros2 launch prl_ur5_run sim.launch.py activate_cameras:=true

# Real hardware:
ros2 launch prl_ur5_run real.launch.py activate_cameras:=true use_sim_time:=false
```

### Run RoboMaestro with Policy
```bash
ros2 launch robo_maestro run_policy.launch.py use_sim_time:=false
```

You can override policy parameters:
```bash
ros2 run robo_maestro run_policy     --cam_list bravo_camera charlie_camera     --arm left     --taskvar real_put_fruit_in_box+0     --ip 127.0.0.1     --port 8002
```

---

## Architecture

```
robo_maestro/
├── launch/
│   └── run_policy.launch.py        # Launches RunPolicyNode with MoveIt configs
├── robo_maestro/
│   ├── run_policy.py               # Main ROS node for policy execution
│   ├── envs/base.py                 # RealRobot-BaseEnv Gym environment
│   ├── core/
│   │   ├── robot.py                 # Robot class, MoveIt interface, gripper control
│   │   ├── observer.py              # Camera/TF/JointState recorders
│   │   └── tf.py                    # Pose & point cloud transforms
│   ├── utils/                       # Constants, logging, helpers
│   └── assets/                      # Task instructions, bbox data
```

**Data Flow:**
1. **`RunPolicyNode`** creates the Gym env (`BaseEnv`) with `Robot`.
2. **`Robot`** connects to MoveIt, cameras, TF, and gripper controllers.
3. **Observation loop**:
   - Capture RGB, depth, point clouds
   - Get gripper/link poses
4. **Policy inference**:
   - Send processed keystep to remote policy server
   - Receive and execute action (pose + gripper state)
5. **Repeat until max steps or termination**

---

## Development Notes

- **Simulation Mode** (`use_sim_time:=true`): Skips gripper actuation, relies on Gazebo time.
- **Workspace Safety**: Cartesian positions are clamped to prevent collisions/out-of-reach targets.
- **Mock Mode**: `PolicyServer.mock_predict()` alternates between two pre-set actions for testing.

---

## Troubleshooting

### ROS 2
- `Package 'prl_ur5_run' not found`: Forgot to `source install/setup.bash`.
- No camera topics: Restart docker container or check camera USB connection.
- UR5 unreachable: Verify Ethernet cabling and IP addresses in config.

### RoboMaestro
- Policy server connection refused: Check `--ip` and `--port` match the SSH-forwarded CLEPS port.
- Gripper not responding: Verify IO controller `/left_io_and_status_controller/set_io` is active.