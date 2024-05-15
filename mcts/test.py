import sys
sys.path.append("../")
sys.path.append("build/")

import time
import mcts_module
import numpy as np

dt = 0.04
horizon = 16
simulations = 100
legs = 4
use_value_function = False
use_action_policy = False
only_imitation_learning = False

state = {
    'position': np.zeros(3),
    'linear_velocity': np.zeros(3),
    'orientation': np.zeros(3),
    'angular_velocity': np.zeros(3),
    'swing_time': np.zeros(legs),
    'stance_time': np.zeros(legs),
    'contact': 15,
    'foot_FL': np.zeros(3),
    'foot_FR': np.zeros(3),
    'foot_RL': np.zeros(3),
    'foot_RR': np.zeros(3)
}

reference = {
    'ref_position': np.zeros(3),
    'ref_linear_velocity': np.zeros(3),
    'ref_orientation': np.zeros(3),
    'ref_angular_velocity': np.zeros(3),
    'ref_foot_FL': np.zeros(3),
    'ref_foot_FR': np.zeros(3),
    'ref_foot_RL': np.zeros(3),
    'ref_foot_RR': np.zeros(3)
}


obj = mcts_module.MCTS(dt, horizon, simulations, legs, False, False, False)

obj.set_current_state(state, reference)


# Run MCTS routine
max_iter = 100
fixed_contacts = []
for x in range(2):
    start = time.time()
    obj.run(max_iter, fixed_contacts)
    print("MCTS runtime: ", time.time() - start)

# print("MPC contact sequence:\n", contact_sequence)