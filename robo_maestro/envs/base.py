"""
Base Gymnasium environment for RoboMaestro.
At its core:
# reset(): -> obs
sets the environment in an initial state, beginning an episode

# step(action) -> (obs, reward, success, msg)
executes a selected action in the environment, and moves the simulation forward
we deal with a real robot, so success is always False and reward is always 0
limited by Truncation (max steps limit)

# render() -> obs
visualizes the current state of the environment
"""