"""
    Monte-Carlo Tree Search (MCTS) for gait adaptation
    to be used in conjuction with an MPC module.
"""

import sys

import copy
import numpy as np
from typing import Tuple, List
from functools import reduce

sys.path.append('./../optimization')
sys.path.append('./../optimization/robust')
sys.path.append('./../optimization/input_rates')
sys.path.append('./../optimization/integrator')

from centroidal_nmpc_input_rates import Acados_NMPC_InputRates


class MCTS:
    """
        Monte-Carlo Tree Search (MCTS) class.
    """
    def __init__(self, mcts_params: dict):
        """
            Class constructor where the MCTS variables are set.
        """
        # MCTS tree
        self.tree = []

        # Number of legs to the system
        self.legs = mcts_params["legs"]

        # MCTS dt
        self.tree_dt = mcts_params["tree_dt"]

        # MPC optimizer user to evaluate MCTS sequences
        self.controller = self.get_ocp_solver()

        # Decimal to binary contact mapping
        self.decimal_to_binary_map = self.get_contact_map()

        # MCTS horizon
        self.tree_horizon = mcts_params["tree_horizon"]

        # Maximum allowed iterations to the MCTS
        self.max_iterations = mcts_params["max_iterations"]

        # Number of simulations for the node evaluation
        self.tree_simulations = mcts_params["tree_simulations"]

        # Whether to use the action policy to choose which node to expand
        self.use_action_policy = mcts_params["use_action_policy"]

        # Whether to use the value function to approximate node cost
        self.use_value_function = mcts_params["use_value_function"]
        
        # Leg's constraints
        self.max_leg_swing_time = mcts_params["max_leg_swing_time"]
        self.min_leg_swing_time = mcts_params["min_leg_swing_time"]
        self.min_leg_stance_time = mcts_params["min_leg_stance_time"]
        
        # Convergence threshold
        self.convergence_threshold = mcts_params["convergence_threshold"]

        # CUrrent state of the system
        self.current_state = {
            "contact": 15,
            "R_op": np.ones((3,3)),
            "R_ref": np.ones((3,3)),
            "x_op": np.ones((24, 1)),
            "u_op": np.ones((24, 1)),
            "x_ref": np.ones((24, 1)),
            "user_input": np.zeros((4,1)),
            "swing_time": np.zeros((4,1))
        }


    def decimal_to_binary(self, decimal_contact: int) -> str:
        """
            Convert a decimal contact to binary.
        """
        return np.binary_repr(decimal_contact, width=self.legs)


    def binary_to_decimal(self, binary_contact: np.ndarray) -> int:
        """
            Convert a binary contact to a decimal.
        """
        return reduce(lambda a,b: 2*a+b, binary_contact)


    def set_current_state(self, new_current_state: dict):
        """
            Update the current state of the system.
        """
        self.current_state = new_current_state


    def is_leaf_node(self, node_idx: int) -> bool:
        """
            Determine if a given node is a leaf in the tree.
        """
        level = 0
        idx = node_idx

        while idx != 0:
            idx = self.tree[idx].parent_
            level += 1

        return level == self.tree_horizon


    def get_ocp_solver(self) -> Acados_NMPC_InputRates:
        """
            Initialize Acados OCP solver for the MCTS.
        """
        acados_controller = Acados_NMPC_InputRates(horizon = self.tree_horizon,
                                                   dt = self.tree_dt,
                                                   use_RTI = False,
                                                   use_warm_start = False,
                                                   use_integrators = False,
                                                   use_foothold_constraints = True,
                                                   use_static_stability = False,
                                                   use_zmp_stability = True,
                                                   use_capture_point_stability = False,
                                                   use_input_prediction=True)
        return acados_controller


    def get_contact_map(self) -> dict:
        """
            Compute dictionary for decimal to binary contact mapping.
        """
        contact_map = {}

        for x in range(2**self.legs):
            contact_map[x] = self.decimal_to_binary(x)

        return contact_map


    def get_node_contact_sequence(self, node_idx: int) -> np.ndarray:
        """
            Retrieve the contact sequence up until the given node idx.
        """
        idx = node_idx
        node_contact_sequence = []

        while idx != 0:
            node_contact_sequence.append(self.tree[idx]["contact"])
            idx = self.tree[idx]["parent"]

        return node_contact_sequence[::-1]
    
    
    def decimal_to_binary_sequence(self, decimal_sequence: list) -> np.ndarray:
        """
            Convert a decimal sequence to a binary one.
        """
        binary_sequence = np.ones((self.legs, self.tree_horizon))
        
        for idx, decimal_contact in enumerate(decimal_sequence):
            binary_sequence[:, idx] = self.con

    
    def check_leg_constraint_violation(self, leg_contact: int, leg_swing_time: float, leg_stance_time: float) -> int:
        """
            Compute if a given leg violates the set constraints.
        """
        # Minimum swing time violation
        if not leg_contact and leg_swing_time < self.min_leg_swing_time:
            return 0
        # Maximum swing time violation
        elif not leg_contact and leg_swing_time >= self.max_leg_swing_time:
            return 1
        # Minimum stance time violation
        elif leg_contact and leg_stance_time < self.min_leg_stance_time:
            return 2
        # No violation
        else:
            return 3

        
    def compute_children(self, node: dict):
        """
            Computes which out of the 16 possible contacts are 
            feasible based on the set swing/stance constraints.
        """
        # Add only a subset of the 16 contact choices
        for contact in range(2**16):
            if contact in [5, 6, 7, 9, 10, 11, 13, 14, 15]:
                node["children"].append(contact)

        # Check for single leg constraint violation
        for leg in range(self.legs):
            violation = self.check_leg_constraint_violation(self.decimal_to_binary_map(node["contact"])[leg], node["swing_time"], node["stance_time"])

            # Minimum swing time violation
            if violation == 0:
                for contact in node["children"]:
                    if self.decimal_to_binary_map[contact][leg]:
                        node["children"].remove(contact)
            # Maximum swing time violation or minimum stance time violation
            elif violation == 1 or violation == 2:
                for contact in node["children"]:
                    if not self.decimal_to_binary_map[contact][leg]:
                        node["children"].remove(contact)


    def simulation_policy(self, node_idx: int):
        """
            Simulate a node's missing contact sequence 
            by performing random rollouts.
        """
        # Storage for our contact sequences
        contact_sequences = []

        # Iterate over recently expanded nodes
        for x in range(node_idx, len(self.tree)):
            node_swing_times = copy.deepcopy(self.tree[x]["swing_times"])
            node_stance_times = copy.deepcopy(self.tree[x]["stance_times"])
            node_contact_sequence = copy.deepcopy(self.get_node_contact_sequence(x))

            for _ in range(len(node_contact_sequence), self.tree_horizon):

            



    def propagate_cost(self, start_idx: int):
        """
            TBD
        """
        return


    def expansion_policy(self, node_idx: int) -> int:
        """
            In the expansion policy, we add to the tree
            the children of the given node, as long as
            it has feasible contacts to it.
        """
        # Expansion is not allowed if the node is a terminal node or if it does not have children
        if self.is_leaf_node(node_idx) or len(self.tree[node_idx]["children"]) == 0:
            return node_idx
        
        # Starting node for the simulation process
        simulation_node_idx = len(self.tree)

        # Add the node's children to the tree
        for contact in self.tree[node_idx]["children"]:
            self.add_node_to_tree(node_idx, contact)
        
        # Clear expanded contacts
        self.tree[node_idx]["children"].clear()

        # Return node idx from which to start simulating
        return simulation_node_idx


    def mcts_converged(self, node_idx: int) -> bool:
        """
            The procedure checks if the MCTS algorithm converged.
            We define convergence if the tree policy selection
            chooses a terminal node for a certain amount of times.
        """
        if self.is_leaf_node(node_idx) and self.tree[node_idx]["tree_policy_selection"] == self.convergence_threshold:
            return True
        else:
            return False


    def tree_policy(self) -> int:
        """
            Select node within the tree with the lowest cost.
        """
        min_cost = 10e10
        min_cost_idx = None

        for idx, node in enumerate(self.tree):
            if node["cost"] < min_cost:
                min_cost = node["cost"]
                min_cost_idx = idx
        
        return min_cost_idx


    def scale_mcts_contact_sequence(self,
                                    mpc_dt: float,
                                    mcts_contact_sequence: np.ndarray) -> np.ndarray:
        """
            TBD
        """
        return np.ndarray


    def run(self, max_iterations: int, fixed_contacts: list) -> np.ndarray:
        """
            Run the MCTS algorithm until convergence.
        """
        # Clear MCTS tree
        self.tree.clear()

        # Temp variables
        iteration = 0
        current_node_idx = 0
        exit_mcts_procedure = False

        # Create root node
        root_node = {}
        root_node["cost"] = 0
        root_node["parent"] = 0
        root_node["contact"] = self.current_state["contact"]
        root_node["children"] = []
        root_node["swing_time"] = self.current_state["swing_time"]
        root_node["stance_time"] = self.current_state["stance_time"]
        root_node["number_of_visits"] = 0
        root_node["tree_policy_selection"] = 0

        # Set fixed contacts
        if len(fixed_contacts) == 0:
            # Compute feasible next contacts for the root node
            self.compute_children(root_node)

            # Push root node to the tree
            self.tree.append(root_node)
        else:
            # Push root node to the tree
            self.tree.append(root_node)

            for contact in fixed_contacts:
                # Create fixed child node
                child_node = {}
                child_node["cost"] = 0
                child_node["parent"] = len(self.tree) - 1
                child_node["contact"] = contact
                child_node["children"] = []
                child_node["swing_time"] = (self.tree[len(self.tree)]["swing_time"] + self.tree_dt) * (1 - self.decimal_to_binary_map[contact])
                child_node["stance_time"] = (self.tree[len(self.tree)]["stance_time"] + self.tree_dt) * self.decimal_to_binary_map[contact]
                child_node["number_of_visits"] = 0
                child_node["tree_policy_selection"] = 0

                # Push child node to the tree
                self.tree.append(child_node)
            
            # Compute children for the last added node
            self.compute_children(self.tree[-1])

            # Increase pointer
            current_node_idx += 1

        # Expand root node
        current_node_idx = self.expansion_policy(current_node_idx)

        # Simulate all newly added children
        self.simulation_policy(current_node_idx)

        # Backpropagate costs for the simulate nodes
        self.propagate_cost(current_node_idx)

        # Main loop
        while not exit_mcts_procedure:
            # Choose best node to expand
            best_node_idx = self.tree_policy()

            # Check if MCTS converged
            exit_mcts_procedure = self.mcts_converged(best_node_idx)
            if exit_mcts_procedure:
                print(f"MCTS converged at iteration {iteration}.")
                print(f"Best node information: \n\n {self.tree[best_node_idx]}")
                continue

            # Expand best node
            current_node_idx = self.expansion_policy(best_node_idx)

            # Simulate expanded nodes
            self.simulation_policy(current_node_idx)

            # Backpropagate costs for the simulated nodes
            self.propagate_cost(current_node_idx)

            if iteration > self.max_iterations:
                min_cost = 10e10

                for idx, node in enumerate(self.tree):
                    if self.is_leaf_node(idx) and node["cost"] < min_cost:
                        min_cost = node["cost"]
                        current_node_idx = idx

                print(f"MCTS maximum iteration reached: {iteration}.")
                print(f"Best node information: \n\n {self.tree[current_node_idx]}")
            
            # Keep track of how many iterations so far
            iteration += 1

        # Scale compute MCTS contact sequence to match MPC dt
        mpc_contact_sequence = self.scale_mcts_contact_sequence(0.04, self.tree_dt, )

# Example Usage:
# mcts_instance = MCTS(tree_dt=0.1, legs=4, tree_steps=10, simulations=5, use_value_function=True, use_action_policy=True)
# mcts_instance.init(...)