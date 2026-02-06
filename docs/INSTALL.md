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
colcon build --symlink-install --packages-select robo_maestro
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