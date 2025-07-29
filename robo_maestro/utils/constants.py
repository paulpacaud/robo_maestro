import numpy as np

# PATHS
DATA_DIR = "/home/ros/share/data"

# WORKSPACE PARAMETERS
WORKSPACE = {
    "left": np.array([[-0.50, -0.54, 0.01], [0.0, 0.54, 0.75]]),
    "right": np.array([[0.295, -0.16, 0.00], [0.695, 0.175, 0.2]]),
}


# ROBOT CONTROL PARAMETERS
DEFAULT_ROBOT_ACTION = [-0.25,  0,  0.5, -7.21724635e-05,  9.99980612e-01, -4.89975834e-05, -6.22642981e-03, 0]
MOCK_ROBOT_ACTION_1 = [-0.25,  0.2,  0.65, -7.21724635e-05,  9.99980612e-01, -4.89975834e-05, -6.22642981e-03, 0]
MOCK_ROBOT_ACTION_2 = [-0.25,  -0.1,  0.5, -7.21724635e-05,  9.99980612e-01, -4.89975834e-05, -6.22642981e-03, 0]
# corresponds to an orientation of [pi, 0, pi] in euler angles. The gripper is open (0)
# the quaternions must satisfy the unit quaternion constraint (the sum of squares should equal 1).
EEF_FRAME = {"left": "left_gripper_grasp_frame", "right": "right_gripper_grasp_frame"}
MAX_VELOCITY_SCALING_FACTOR = 0.2
MAX_ACCELERATION_SCALING_FACTOR = 0.2
PLANNING_TIME = 2.0
COMMAND_ROS_TOPIC = {"left": "/left_arm/scaled_pos_joint_traj_controller/command", "right": "/right_arm/scaled_pos_joint_traj_controller/command"}
EEF_STEPS = 0.01
JUMP_THRESHOLD = 0.0
Q_VEL_THRESHOLD = 2  # rad/s
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
    "right": [
        "right_base_link"
    ]
}

# POLICY PARAMETERS
MAX_STEPS = 5

# CAMERA PARAMETERS
BRAVO_INFO = {
    "pos": np.array([-0.192736, 0.835309,0.700992]),
    "euler": np.array([-2.227, 0.011, 3.142]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}
CHARLIE_INFO = {
    "pos": np.array([-0.670188, -0.558114, 0.597616]),
    "euler": np.array([-2.355, -0.100,  -0.517]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}
ALPHA_INFO = {
    "pos": np.array([0.131421, -0.626980, 0.818875]),
    "euler": np.array([-2.467, 0.026, 0.475]),
    "fovy": 42.5,
    "height": 480,
    "width": 640,
}
CAM_INFO = {"bravo_camera": BRAVO_INFO, "charlie_camera": CHARLIE_INFO, "alpha_camera": ALPHA_INFO}
CAM_TF_TOPIC = {"bravo_camera": "bravo_camera_color_optical_frame", "charlie_camera": "charlie_camera_color_optical_frame", "alpha_camera": "alpha_camera_color_optical_frame"}
