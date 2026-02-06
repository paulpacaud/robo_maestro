"""
A launch file for running the pointact policy inference node
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
import os

def launch_setup(context):
    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    mock = LaunchConfiguration('mock').perform(context).lower() == 'true'
    taskvar = LaunchConfiguration('taskvar')
    ip = LaunchConfiguration('ip')
    port = LaunchConfiguration('port')

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

    # Build arguments list
    arguments = ["--taskvar", taskvar, "--ip", ip, "--port", port]
    if mock:
        arguments = ["--mock"] + arguments
    arguments.extend(["--ros-args", "--log-level", "WARN"])

    moveit_py_node = Node(
        name="run_policy_pointact",
        package="robo_maestro",
        executable="run_policy_pointact",
        output="both",
        parameters=[moveit_config.to_dict(), {'use_sim_time': use_sim_time}],
        arguments=arguments,
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
            "mock",
            default_value="false",
            description="Use mock policy predictions instead of a real server",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "taskvar",
            default_value="ur5_put_grapes_and_banana_in_plates",
            description="Task variant (used for instruction lookup)",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "ip",
            default_value="127.0.0.1",
            description="Policy server IP address",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "port",
            default_value="17000",
            description="Policy server port",
        )
    )
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
