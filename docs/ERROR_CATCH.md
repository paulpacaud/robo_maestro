## Launch/Startup Errors

### `Package 'prl_ur5_run' not found`
**Cause:** You forgot to source the workspace after building.
**Fix:**
```bash
source ~/share/ws/install/setup.bash
```

### `Controller not available after X seconds`
**Cause:** The MoveIt trajectory controller has not started.
**Fix:**
- Ensure the `prl_ur5_run` launch file started the correct controllers.
- For real robots, check the UR controller on the teach pendant.

### `No message on <topic> within timeout`
**Cause:** Camera or TF data is not being published.
**Fix:**
- Check `ros2 topic list` for expected topics.
- Restart camera drivers or reconnect USB.
- For TF issues, ensure all required robot_state_publisher and static_transform_publisher nodes are running.

---

## Policy Server Errors

### `ConnectionRefusedError` when sending batch
**Cause:** Policy server is not reachable.
**Fix:**
- Verify SSH tunnel to CLEPS server:
```bash
ssh -N -L 8002:gpu017:8002 cleps -i ~/.ssh/jz_rsa
```
- Ensure `--ip` matches `localhost` and port matches the forwarded port.

### `requests.exceptions.Timeout`
**Cause:** Policy server too slow or unresponsive.
**Fix:**
- Increase server resources (`salloc` with more CPU/GPU).
- Reduce image/point cloud sizes to speed up transmission.

---

## Robot Execution Errors

### `"Failed to move the robot"`
**Cause:** Motion planning or execution failed.
**Fix:**
- Check if target pose is outside `WORKSPACE` bounds.
- Verify MoveIt collision environment is correct.
- Try joint-space planning by disabling `cartesian_only`.

### `Gripper moved to position ...` but no movement observed
**Cause:** Incorrect IO pin or controller not active.
**Fix:**
- Verify `/left_io_and_status_controller/set_io` exists:
```bash
ros2 service list | grep set_io
```
- Check teach pendant IO configuration.

---

## Camera/Point Cloud Errors

### `ValueError: cannot reshape array`
**Cause:** Depth or RGB data size mismatch.
**Fix:**
- Ensure correct `dtype` is passed to `record_image` (`np.uint8` for RGB, `np.uint16` for depth).
- If hardware setup changed, verify `CameraInfo` intrinsics.

---

## Simulation-Specific Issues

### Gripper not actuating in sim
**Explanation:** In `use_sim_time` mode, gripper execution is skipped by design.
**Fix:** This is expected. Use MoveIt gripper component if you want simulated gripper control.

## TF Lookup Errors

### `"prl_ur5_base" passed to lookupTransform argument target_frame does not exist`
**Cause:** The node using `TFRecorder` / `TransformListener` is not being spun, so `/tf` and `/tf_static` subscription callbacks never fire and the TF buffer stays empty. In ROS 2 Jazzy, `TransformListener` defaults to `spin_thread=False`.
**Fix:**
- Spin the node in a background thread before creating `TFRecorder`:
```python
from rclpy.executors import SingleThreadedExecutor
import threading

executor = SingleThreadedExecutor()
executor.add_node(node)
spin_thread = threading.Thread(target=executor.spin, daemon=True)
spin_thread.start()
```
- Shut down the executor cleanly in the `finally` block (before `node.destroy_node()`).
- If the frame genuinely doesn't exist, check that `robot_state_publisher` and any required `static_transform_publisher` nodes are running:
```bash
ros2 topic echo /tf_static --once
ros2 run tf2_tools view_frames
```

### `ConnectivityException: not part of the same tree` during `Robot.__init__`
**Cause:** The node was added to a background `SingleThreadedExecutor` **before** `gym.make()` / `Robot.__init__` ran. Internally, `Camera.__init__` calls `wait_for_message()` → `rclpy.spin_until_future_complete(node, ...)`, which needs to temporarily spin the node in its own executor. If the node is already in an executor, this conflicts: the TF buffer never fills properly and the subsequent `eef_pose()` call finds a disconnected TF tree.
**Fix:**
- Start the background spin thread **after** environment setup completes, not before. `Robot.__init__` handles its own spinning via `rclpy.spin_until_future_complete()` during camera init, which also populates the TF buffer.
```python
# WRONG — conflicts with wait_for_message() inside Robot.__init__
executor = SingleThreadedExecutor()
executor.add_node(node)
spin_thread = threading.Thread(target=executor.spin, daemon=True)
spin_thread.start()
node.setup_environment()  # gym.make() fails here

# CORRECT — let Robot.__init__ handle its own spinning
node.setup_environment()  # gym.make() succeeds
executor = SingleThreadedExecutor()
executor.add_node(node)
spin_thread = threading.Thread(target=executor.spin, daemon=True)
spin_thread.start()
```
- This matches the pattern used in `run_policy.py`, which does not use a background spin thread during setup.


---

## General Debug Tips

- **Check ROS graph**: `ros2 node list` and `ros2 topic list`
- **Inspect data**: `ros2 topic echo` and RViz visualization
- **Mock mode**: Switch to `mock_predict()` for isolating robot vs. policy issues
- **Logging**: RoboMaestro logs include call chain for easier tracing

---

## Emergency Stop
If the robot behaves unexpectedly:
1. Hit the physical **E-STOP** button immediately.
2. Investigate logs and last commanded action.

### ROS 2
- `Package 'prl_ur5_run' not found`: Forgot to `source install/setup.bash`.
- No camera topics: Restart docker container or check camera USB connection.
- UR5 unreachable: Verify Ethernet cabling and IP addresses in config.

### RoboMaestro
- Policy server connection refused: Check `--ip` and `--port` match the SSH-forwarded CLEPS port.
- Gripper not responding: Verify IO controller `/left_io_and_status_controller/set_io` is active.

# Data collection
Pb: No Joystick detected
Solution: restart the docker once the joystick is connected