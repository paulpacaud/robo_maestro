# Push / Pull huggingface
hf upload paulpacaud/ur5_put_grapes_and_banana_in_plates /home/ppacaud/docker_shared/data/ur5_put_grapes_and_banana_in_plates --repo-type dataset
hf download paulpacaud/ur5_put_grapes_and_banana_in_plates --repo-type dataset --local-dir ur5_put_grapes_and_banana_in_plates

# put fruits in plates

11 keysteps
1: start
2: approach the grape
3: down, close gripper

4: lift the grape
5: approach the yellow plate
6: down, open gripper

7: approach banana
8: down, close gripper
9: lift banana
10: approach the pink plate
11: down, open gripper



grasp:
HOME>8 Z Down
move to::
> 8 Z UP
release
3 DOWN
