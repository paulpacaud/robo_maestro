# Push / Pull huggingface
huggingface-cli upload paulpacaud/put_fruits_in_plates_0 /home/ppacaud/docker_shared/data/put_fruits_in_plates+0 --repo-type dataset
hf download paulpacaud/put_fruits_in_plates_0 --repo-type dataset --local-dir put_fruits_in_plates_0

put fruits in plates

11 keysteps
1: start
2: approach the grape
3: close gripper
4: lift the grape
5: approach the yellow plate
6: open gripper
7: approach banana
8: close gripper
9: lift banana
10: approach the pink plate
11: open gripper



grasp:
HOME>8 Z Down
move to::
> 8 Z UP
release
3 DOWN
