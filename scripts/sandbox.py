import numpy as np

from robo_maestro.utils.logger import log_info

action = np.array([-3.44795868e-01,  1.74202271e-01,  2.00817175e-02, -2.58819045e-01, -9.65925826e-01,  5.91458986e-17,  1.58480958e-17,  3.61067792e-02])

log_info(f"target action: {', '.join(map(str, round(action.tolist(),3)))}")