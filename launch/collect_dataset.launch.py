"""
Launch file for joystick teleop + keystep data collection.

Loads the MoveIt robot description (mantis / prl_ur5_moveit) so that
TF frames and motion planning are available, then runs the
collect_dataset node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def launch_setup(context):
    use_sim_time = LaunchConfiguration("use_sim_time")
    task = LaunchConfiguration("task")
    var = LaunchConfiguration("var")
    data_dir = LaunchConfiguration("data_dir")
    cam_list = LaunchConfiguration("cam_list")
    pos_step = LaunchConfiguration("pos_step")
    rot_step = LaunchConfiguration("rot_step")
    crop_size = LaunchConfiguration("crop_size")
    debug = LaunchConfiguration("debug")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="mantis", package_name="prl_ur5_moveit")
        .robot_description(file_path="config/mantis.urdf.xacro")
        .moveit_cpp(
            file_path=get_package_share_directory("robo_maestro")
            + "/config/planner.yaml"
        )
        .to_moveit_configs()
    )

    collect_node = Node(
        name="collect_dataset",
        package="robo_maestro",
        executable="collect_dataset",
        output="both",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": use_sim_time,
                "task": task,
                "var": var,
                "data_dir": data_dir,
                "cam_list": cam_list,
                "pos_step": pos_step,
                "rot_step": rot_step,
                "crop_size": crop_size,
                "debug": debug,
            },
        ],
    )

    return [collect_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),
        DeclareLaunchArgument(
            "task",
            default_value="my_task",
            description="Task name for the dataset",
        ),
        DeclareLaunchArgument(
            "var",
            default_value="0",
            description="Task variant index",
        ),
        DeclareLaunchArgument(
            "data_dir",
            default_value="/home/ros/share/data",
            description="Root directory for dataset storage",
        ),
        DeclareLaunchArgument(
            "cam_list",
            default_value="foxtrot_camera",
            description="Comma-separated camera names",
        ),
        DeclareLaunchArgument(
            "pos_step",
            default_value="0.02",
            description="Position step size in metres",
        ),
        DeclareLaunchArgument(
            "rot_step",
            default_value="5.0",
            description="Rotation step size in degrees",
        ),
        DeclareLaunchArgument(
            "crop_size",
            default_value="256",
            description="Image crop/resize target size",
        ),
        DeclareLaunchArgument(
            "debug",
            default_value="false",
            description="Print joystick debug info",
        ),
    ]
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
