from pathlib import Path

import numpy as np

# PATHS
DATA_DIR = "/home/ros/share/data"
TASKVARS_INSTRUCTIONS_PATH = Path(__file__).parent / "taskvars_instructions.json"

# WORKSPACE PARAMETERS
# The first sub-array [-0.50, -0.54, 0.01] is the minimum allowed position: x, y, z.
# The second sub-array [0.0, 0.54, 0.75] is the maximum allowed position: x, y, z.
WORKSPACE = {
    "left": np.array([[-0.50, -0.54, 0.015], [0.0, 0.54, 0.75]]),
    "right": np.array([[0.295, -0.16, 0.00], [0.695, 0.175, 0.2]]),
}

# ROBOT CONTROL PARAMETERS
DEFAULT_ROBOT_ACTION = [
    -0.17,
    -0.05,
    0.25,
    -0.01592672,
    0.99976501,
    0.00417359,
    0.01410128,
    0,
]
# MOCK_ROBOT_ACTION_1 = [-0.25,  0.2,  0.65, -7.21724635e-05,  9.99980612e-01, -4.89975834e-05, -6.22642981e-03, 1]
MOCK_ROBOT_ACTION_1 = [
    -0.344795868,
    0.174202271,
    0.07,
    -0.258819045,
    -0.965925826,
    5.91458986e-17,
    1.58480958e-17,
    1,
]
# MOCK_ROBOT_ACTION_2 = [-0.25,  -0.1,  0.5, -7.21724635e-05,  9.99980612e-01, -4.89975834e-05, -6.22642981e-03, 0]
# MOCK_ROBOT_ACTION_2 = [-1.76405452e-01,  9.35083255e-03,  0.1, -2.58819045e-01, -9.65925826e-01,  5.91458986e-17,  1.58480958e-17,  7.58555225e-02]
MOCK_ROBOT_ACTION_2 = [
    -0.5,
    0.3,
    0.1,
    -0.258819045,
    -0.965925826,
    5.91458986e-17,
    1.58480958e-17,
    0,
]

# corresponds to an orientation of [pi, 0, pi] in euler angles. The gripper is open (0)
# the quaternions must satisfy the unit quaternion constraint (the sum of squares should equal 1).
EEF_FRAME = {"left": "left_gripper_grasp_frame", "right": "right_gripper_grasp_frame"}
MAX_VELOCITY_SCALING_FACTOR = 0.2
MAX_ACCELERATION_SCALING_FACTOR = 0.2
PLANNING_TIME = 2.0
COMMAND_ROS_TOPIC = {
    "left": "/left_arm/scaled_pos_joint_traj_controller/command",
    "right": "/right_arm/scaled_pos_joint_traj_controller/command",
}
EEF_STEPS = 0.01
JUMP_THRESHOLD = 0.0
Q_VEL_THRESHOLD = 2  # rad/s
TRY_PLANNING_MAX_ITER = 10
ROBOT_BASE_FRAME = "prl_ur5_base"
ROBOT_LINKS = {
    "left": [
        "left_base_link",
        "left_shoulder_link",
        "left_upper_arm_link",
        "left_forearm_link",
        "left_wrist_1_link",
        "left_wrist_2_link",
        "left_wrist_3_link",
        "left_ft300_mounting_plate",
        "left_ft300_sensor",
        "left_gripper_body",
        "left_gripper_bracket",
        "left_gripper_finger_1_finger_tip",
        "left_gripper_finger_1_flex_finger",
        "left_gripper_finger_1_safety_shield",
        "left_gripper_finger_1_truss_arm",
        "left_gripper_finger_1_moment_arm",
        "left_gripper_finger_2_finger_tip",
        "left_gripper_finger_2_flex_finger",
        "left_gripper_finger_2_safety_shield",
        "left_gripper_finger_2_truss_arm",
        "left_gripper_finger_2_moment_arm",
    ],
    "right": ["right_base_link"],
}

# POLICY PARAMETERS
MAX_STEPS = 4

# CAMERA PARAMETERS
# Camera poses are now obtained from TF at runtime rather than hardcoded.
CAM_INFO = {}
CAM_TF_TOPIC = {
    "echo_camera": "echo_camera_color_optical_frame",
    "foxtrot_camera": "foxtrot_camera_color_optical_frame",
    "golf_camera": "golf_camera_color_optical_frame",
}
