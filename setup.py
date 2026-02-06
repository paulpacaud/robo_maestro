"""
defines the package metadata
defines executable offline_scripts, i.e. available ROS2 nodes
tells colcon how to install the package and where
list other python packages required by this package (akin to requirements.txt but versions of dependencies have flexible ranges like >=1.0.0,<2.0.0)
"""

from setuptools import find_packages, setup
import os
from glob import glob

package_name = "robo_maestro"

setup(
    name="robo_maestro",
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    # data_files=[
    #     ('share/ament_index/resource_index/packages',
    #         ['resource/' + package_name]),
    #     ('share/' + package_name, ['package.xml']),
    # ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "assets"),
            glob(os.path.join("robo_maestro/assets", "*")),
        ),
    ],
    install_requires=[
        "setuptools",
        "torch",
        "torchvision",
        "gymnasium",
        "typed-argument-parser",
        "msgpack-numpy",
        "easydict",
        "scipy",
        "open3d",
        "requests",
        "pygame",
        "lmdb",
        "pydantic>=2.0",
        "grpcio>=1.50.0",
        "protobuf>=4.0",
    ],
    zip_safe=True,
    maintainer="Paul Pacaud",
    maintainer_email="paul.pacaud@inria.fr",
    description="robo_maestro: Python Package to interact with Paris Robotics Lab UR5",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "run_policy = robo_maestro.run_policy:main",
            "run_policy_lerobot = robo_maestro.run_policy_lerobot:main",
            "run_policy_pointact = robo_maestro.run_policy_pointact:main",
            "read_eef_pose = robo_maestro.dev_tools.read_eef_pose:main",
            "control_robot_eef = robo_maestro.dev_tools.control_robot_in_EEF_pose:main",
            "collect_dataset = robo_maestro.dataset_collection.collect_dataset:main",
        ],
    },
)
