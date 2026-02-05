cd Projects/prl_ur5_ros2/docker-ros2/
./start_docker.bash robomaestro ~/docker_shared
cd ~/share/ws

sudo apt-get install -y python3-pip \
&& sudo python3 -m pip install --break-system-packages --ignore-installed \
torch torchvision gymnasium typed-argument-parser \
msgpack-numpy easydict scipy open3d requests \
transforms3d setuptools lmdb pygame pydantic
pip3 install --break-system-packages "setuptools==70.0.0"

PYPLIB=$(python3 -c "import sysconfig,sys;print(sysconfig.get_paths()['platlib'])") \
 && echo "export PYTHONPATH=${PYPLIB}:\$PYTHONPATH" >> sudo /etc/profile.d/pythonpath.sh
PYTHONPATH=${PYPLIB}:${PYTHONPATH}


tmux new -s robomaestro
> Open 3 terminals, in each, run:

cd ~/share/ws
source install/setup.bash

1st terminal: ros2 launch prl_ur5_run real.launch.py activate_cameras:=true use_sim_time:=false
2nd terminal: 

