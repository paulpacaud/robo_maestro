# camera calibration
you need one terminal to launch the setup (to launch the cameras)
and one another terminal to launch the calib.
The process is to run the calib on each camera (change `camera_name:=echo_camera`), and copy paste the camera pose to the camera config file.
The calib is capricious, you will need to try a few times

terminal1: 
`ros2 launch robo_maestro run_policy.launch.py use_sim_time:=false`

terminal2:
`ros2 launch prl_ur5_calibration calibrate_external_camera.launch.py camera_name:=echo_camera camera_topic:=/color namespace_camera:=/camera`

# Update the camera extrinsics in the config file:
vim ~/share/ws/src/prl_ur5_robot_configuration/config/fixed_cameras/dataset_collection.yaml

# How to check that the cameras have not moved and the previous calibration is still valid
Rerun the calib launch, and compare the new pose with the one in the config file. If they are close, the previous calibration is still valid. If they are very different, you may need to update the config file with the new pose.
For the same camera pose, running several times the calib, I observe a variability of +/- 0.01 m in translation and 0.05 rad in rotation.