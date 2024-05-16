import sys
sys.path.append("../")
sys.path.append("build/")

import time
import mcts_module
import numpy as np

dt = 0.04
horizon = 16
simulations = 10
legs = 4
use_value_function = False
use_action_policy = False
only_imitation_learning = False

state =  {'position': np.array([-0.018,  0.001,  0.357]), 
          'linear_velocity': np.array([-0.001,  0.   ,  0.   ]), 
          'orientation': np.array([-0.008,  0.001, -0.   ]), 
          'angular_velocity': np.array([-0.001,  0.   , -0.   ]), 
          'foot_FL': np.array([0.229, 0.126, 0.026]), 
          'foot_FR': np.array([ 0.229, -0.126,  0.026]), 
          'foot_RL': np.array([-0.25 ,  0.127,  0.025]), 
          'foot_RR': np.array([-0.25 , -0.127,  0.025])}

reference =  {'ref_position': np.array([0.  , 0.  , 0.35]), 
              'ref_linear_velocity': np.array([0., 0., 0.]), 
              'ref_orientation': np.array([0., 0., 0.]), 
              'ref_angular_velocity': np.array([0., 0., 0.]), 
              'ref_foot_FL': np.array([[0.221, 0.152, 0.   ]]), 
              'ref_foot_FR': np.array([[ 0.221, -0.15 ,  0.   ]]), 
              'ref_foot_RL': np.array([[-0.258,  0.153,  0.   ]]), 
              'ref_foot_RR': np.array([[-0.259, -0.149,  0.   ]])}

contact = 15
swing_time = np.zeros(legs)
stance_time = np.zeros(legs) + 0.5


obj = mcts_module.MCTS(dt, horizon, simulations, legs, False, False, False)

obj.set_current_state(state, reference, contact, swing_time, stance_time)


# Run MCTS routine
max_iter = 100
fixed_contacts = []
start = time.time()
obj.run(max_iter, fixed_contacts)
print("MCTS runtime: ", time.time() - start)

# print("MPC contact sequence:\n", contact_sequence)