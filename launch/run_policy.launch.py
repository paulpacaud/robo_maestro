"""
A launch file for running the motion planning python api tutorial
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import os


def launch_setup(context):
    use_sim_time = LaunchConfiguration("use_sim_time", default=True)
    taskvar = LaunchConfiguration("taskvar")
    port = LaunchConfiguration("port")
    # Here we built the moveit_config object
    # This object is used to load the robot description and the planner
    moveit_config = (
        MoveItConfigsBuilder(robot_name="mantis", package_name="prl_ur5_moveit")
        .robot_description(file_path="config/mantis.urdf.xacro")
        .moveit_cpp(
            file_path=get_package_share_directory("robo_maestro")
            + "/config/planner.yaml"
        )
        .to_moveit_configs()
    )

    # Here we launch the executable
    # This executable is the run_policy.py file warn to specify if the robot is in simulation or not via the use_sim_time parameter
    moveit_py_node = Node(
        name="run_policy",
        package="robo_maestro",
        executable="run_policy",
        output="both",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
        arguments=["--taskvar", taskvar, "--port", port, "--ros-args", "--log-level", "WARN"],
    )

    return [
        moveit_py_node,
    ]


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "taskvar",
            default_value="None",
            description="taskvar as defined in the json file",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "port",
            default_value="8080",
            description="server port",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
