"""
Launch file for controlling the robot via an 8-dim EEF pose.
Mirrors the read_eef_pose launch: loads the MoveIt robot description
so that all TF frames and planning are available, then runs the
control_robot_eef node.
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def launch_setup(context):
    use_sim_time = LaunchConfiguration("use_sim_time", default=True)
    action = LaunchConfiguration("action")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="mantis", package_name="prl_ur5_moveit")
        .robot_description(file_path="config/mantis.urdf.xacro")
        .moveit_cpp(
            file_path=get_package_share_directory("robo_maestro")
            + "/config/planner.yaml"
        )
        .to_moveit_configs()
    )

    control_node = Node(
        name="control_robot_eef",
        package="robo_maestro",
        executable="control_robot_eef",
        output="both",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time, "action": action},
        ],
    )

    return [control_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),
        DeclareLaunchArgument(
            "action",
            default_value="",
            description="8-dim EEF pose: x,y,z,qx,qy,qz,qw,grip",
        ),
    ]
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
