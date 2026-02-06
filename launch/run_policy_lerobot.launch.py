"""Launch file for run_policy_lerobot — LeRobot gRPC bridge for UR5."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


def launch_setup(context):
    use_sim_time = LaunchConfiguration("use_sim_time", default=True)
    taskvar = LaunchConfiguration("taskvar")
    task = LaunchConfiguration("task")
    ip = LaunchConfiguration("ip")
    port = LaunchConfiguration("port")
    policy_type = LaunchConfiguration("policy_type")
    pretrained_name_or_path = LaunchConfiguration("pretrained_name_or_path")
    policy_device = LaunchConfiguration("policy_device")
    actions_per_chunk = LaunchConfiguration("actions_per_chunk")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="mantis", package_name="prl_ur5_moveit")
        .robot_description(file_path="config/mantis.urdf.xacro")
        .moveit_cpp(
            file_path=get_package_share_directory("robo_maestro")
            + "/config/planner.yaml"
        )
        .to_moveit_configs()
    )

    moveit_py_node = Node(
        name="run_policy_lerobot",
        package="robo_maestro",
        executable="run_policy_lerobot",
        output="both",
        parameters=[moveit_config.to_dict(), {"use_sim_time": use_sim_time}],
        arguments=[
            "--taskvar", taskvar,
            "--task", task,
            "--ip", ip,
            "--port", port,
            "--policy_type", policy_type,
            "--pretrained_name_or_path", pretrained_name_or_path,
            "--policy_device", policy_device,
            "--actions_per_chunk", actions_per_chunk,
            "--ros-args", "--log-level", "WARN",
        ],
    )

    return [moveit_py_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        ),
        DeclareLaunchArgument(
            "taskvar",
            default_value="ur5_close_drawer",
            description="Task variant (used for instruction lookup)",
        ),
        DeclareLaunchArgument(
            "task",
            default_value="",
            description="Natural language task instruction (overrides taskvar lookup)",
        ),
        DeclareLaunchArgument(
            "ip",
            default_value="127.0.0.1",
            description="Policy server IP",
        ),
        DeclareLaunchArgument(
            "port",
            default_value="8080",
            description="Policy server port",
        ),
        DeclareLaunchArgument(
            "policy_type",
            default_value="pi0",
            description="Policy type (pi0, act, diffusion, ...)",
        ),
        DeclareLaunchArgument(
            "pretrained_name_or_path",
            default_value="",
            description="Path or HF Hub name for the pretrained model",
        ),
        DeclareLaunchArgument(
            "policy_device",
            default_value="cuda",
            description="Device for policy inference on the server",
        ),
        DeclareLaunchArgument(
            "actions_per_chunk",
            default_value="50",
            description="Number of actions per inference chunk",
        ),
    ]
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
