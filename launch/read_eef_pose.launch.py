"""
Launch file for reading the EEF pose of the robot.
Mirrors the run_policy launch: loads the MoveIt robot description
so that all TF frames (prl_ur5_base, left_gripper_grasp_frame, etc.)
are available, then runs the read_eef_pose node.
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

    moveit_config = (
        MoveItConfigsBuilder(robot_name="mantis", package_name="prl_ur5_moveit")
        .robot_description(file_path="config/mantis.urdf.xacro")
        .moveit_cpp(
            file_path=get_package_share_directory("robo_maestro")
            + "/config/planner.yaml"
        )
        .to_moveit_configs()
    )

    read_eef_node = Node(
        name="read_eef_pose",
        package="robo_maestro",
        executable="read_eef_pose",
        output="both",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
    )

    return [read_eef_node]


def generate_launch_description():
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
