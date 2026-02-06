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
# Installation
cf INSTALL.md

# Troubleshooting
cf ERROR_CATCH.md

# Contribute
To add a launch, update setup.py, add the launch to the ./launch folder, then run `colcon build --symlink-install --packages-select robo_maestro`, source ROS, and test

# Usage
## Launch the Robot (Real or Sim)
```bash
# Real hardware:
ros2 launch prl_ur5_run real.launch.py activate_cameras:=true use_sim_time:=false

# Simulation:
ros2 launch prl_ur5_run sim.launch.py activate_cameras:=true
```

## Run RoboMaestro with Policy
```bash
ros2 launch robo_maestro run_policy.launch.py use_sim_time:=false
```

You can override policy parameters:
```bash
ros2 run robo_maestro run_policy     --cam_list bravo_camera charlie_camera     --arm left     --taskvar real_put_fruit_in_box+0     --ip 127.0.0.1     --port 8002
```

## Dev tools

```bash
# Read End-Effector Pose
ros2 launch robo_maestro read_eef_pose.launch.py use_sim_time:=false

# Control End-Effector Pose without cameras or policy
ros2 launch robo_maestro control_robot_eef.launch.py use_sim_time:=false

# Collect a keystep-based dataset
ros2 launch robo_maestro collect_dataset.launch.py \
    use_sim_time:=false \
    task:=put_fruits_in_plates\
    var:=0 \
    cam_list:=echo_camera \
    start_episode_id:=0
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


