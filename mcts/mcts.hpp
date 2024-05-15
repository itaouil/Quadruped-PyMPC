#pragma once

#include <map>
#include <thread>
#include "random"
#include <memory>
#include <iostream>
#include <filesystem>
#include <Eigen/Dense>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
// #include <torch/script.h>
#include "structs/node.hpp"
#include "structs/state.hpp"

namespace py = pybind11;

class MCTS
{
public:
    /**
     * Constructor.
     *
     * @param dt
     * @param n_leg
     * @param n_step
     * @param par
     * @param useValueFunction
     */
    MCTS(float dt, int horizon, int simulations, int legs, bool use_value_function, bool use_action_policy, bool only_imitation_learning);

    /**
     * Start the MCTS cycle.
     *
     * @param max_iter
     * @param fixed_contacts
     * @return the MCTS contact sequence to be fed to the MPC
     */
    Eigen::MatrixXi run(int max_iter, const std::vector<int>& fixed_contacts);

    /**
     * Update the MCTS with the current robot
     * state and references for the upcoming
     * computations.
     *
     * @param state
     * @param reference
     */
    void setCurrentState(const py::dict state, const py::dict reference);
                         
    /**
     * Bundle and return the values used
     * a posteriori for the training process
     * of the value and action policies.
     *
     * @return tuple of system state and MCTS tree
     */
    std::tuple<Eigen::Matrix3d, Eigen::Matrix3d,
            Eigen::VectorXd, Eigen::VectorXd,
            Eigen::Vector4d, float, std::vector<std::tuple < int, float, int, int, std::vector < float>>>> getMCTSData();

private:
    // Whether to print debug info or not
    bool m_debug{false};

    // Touchdown positions
    Eigen::Matrix<float,3,4> m_touchdowns;

    // Whether to use policies or not
    bool m_use_action_policy;
    bool m_use_value_function;
    bool m_only_imitation_learning;

    // Current state of the system
    State current_state_{};

    // The full tree in the MCTS
    std::vector<Node> tree_;

    // Value and action policies used
    // within the MCTS computation to
    // improve computational speed
    // torch::jit::script::Module m_action_policy;
    // torch::jit::script::Module m_value_function;

    // Python embedded variables
    py::dict m_state;
    py::dict m_reference;
    py::object m_nmpc_instance;
    bool m_python_initialized{false};

    // Contact constant
    float m_rc{2};

    // Robot height reference
    float m_height_reference{};

    // Number of legs
    int m_legs{4};

    // MCTS parameters
    int m_tree_horizon{10};

    // MCTS discretization value
    float m_tree_dt{0.05};

    // MCTS simulations per simulated node
    int m_simulations{5};

    // MCTS computational time constrain flags
    float m_current_routine_time;
    float m_max_routine_time{30};
    bool m_constrain_routine_time{false};
    std::chrono::_V2::steady_clock::time_point m_start_routine_time;

    // Swinging times parameters
    float m_max_swing_time{0.4};
    float m_min_swing_time{0.4};
    float m_min_stance_time{0.2};

    // Number of computations
    std::vector<int> m_first_sequence;
    std::vector<int> m_second_sequence;

    // Whether action policy was used for expansion or not
    bool m_action_policy_expansion{false};

    // Full Stance Cost
    float m_full_stance_cost{0};

    // Mersenne Twister pseudo-random generator
    std::mt19937 m_mt;

    // MCTS exit flag
    bool m_exit_flag{false};

    // Int to binary contact map
    std::map<int, Eigen::VectorXi> m_contact_map{};

    // Int to Int contact map for action policy
    std::map<int, int> m_action_policy_contact_map{};

    // Leg contact additional cost
    std::vector<float> m_contact_temp_cost{};

    /**
     *  Searches in the tree the node with
     *  the lowest LCB value.
     *
     * @return the node idx with the lowest cost
     */
    int treePolicy();

    /**
     * Convert decimal contact to binary.
     *
     * @param n
     * @return binary contact
     */
    Eigen::VectorXi decimal2Binary(int n);

    /**
     * Checks if a node is a leaf
     * node in the tree or not.
     *
     * @param node_idx
     * @return true if leaf node, otherwise false
     */
    bool isLeafNode(const int &node_idx);

    /**
     * Creates new children for the parent
     * node based on the feasible contacts
     * of the parent node.
     *
     * @param parent_idx
     * @return the first node idx added in the expansion process
     */
    int expansionPolicy(const int parent_idx);

    /**
     * Create a new child node in the tree.
     */
    void addNode(const int parent_idx, const int contact);

    /**
     * Compute and return the rollout costs.
     *
     * @param rollouts
     * @return rollout costs
     */
    std::vector<float> solveOCPs(const std::vector<Eigen::MatrixXi> &rollouts);

    /**
     * Computes the average cost of the
     * simulated node based on the computed
     * QPs.
     *
     * @param node_idx
     */
    void simulationPolicy(const int node_idx);

    /**
     * Updates the cost of the parents
     * of the simulated node.
     *
     * @param node_idx
     */
    void backPropagation(const int &node_idx);

    /**
     * Convert binary contact to decimal.
     *
     * @param vec
     * @return decimal contact
     */
    int binary2Decimal(const Eigen::VectorXi &vec);

    /**
     * Computes the contact sequence starting
     * from the root node up until the given
     * node idx.
     *
     * @param node_idx
     * @return
     */
    std::vector<int> getCurrentSeq(const int node_idx);

    /**
     * Compute the best action to follows using the
     * trained action policy.
     *
     * @param p_idx_sim
     * @return best contact to place in the future sequence step
     */
    int computeNodeActionPolicyRollout(const int p_idx_sim);

    /**
     * Compute the QP cost estimate using the trained
     * value function.
     *
     * @param p_idx_sim
     * @return cost output from the value function network
     */
    float computeNodeValueFunctionCost(const int p_idx_sim);

    /**
     * Converts a decimal contact sequence
     * to a binary contact matrix.
     *
     * @param vec
     * @return binary contact matrix
     */
    Eigen::MatrixXi state2Binary(const std::vector<int> &vec);

    /**
     * Determines if the MCTS cycle
     * can be quit or not.
     *
     * @param current_idx
     * @param last_idx
     * @return MCTS cycle ended or not
     */
    bool exitPolicy(const int &current_idx, const int &last_idx);

    /**
     * Computes if given a new contact and the
     * current swing time, the former leads to
     * a swing time violation of the swing constraints.
     *
     * @param contact
     * @param swing_time
     * @param stance_time
     * @return type of violation (0: min swing violation, 1: max swing violation, 2: min stance violation, 3: no violation)
     */
    int constraintViolation(const int &contact, const float &swing_time, const float &stance_time) const;

    /**
     * Converts a contact sequence with a
     * higher dynamics dt to a contact
     * sequence with a lower dynamics dt.
     *
     * @param dt1
     * @param dt2
     * @param seq_dt1
     * @return
     */
    Eigen::MatrixXi scaleConverter(float dt1, float dt2, const Eigen::MatrixXi &seq_dt1);

    /**
     * Prepare the input for the value function network.
     *
     * @param value_function_input
     * @param idx_sim
     */
    void prepareValueFunctionNetworkInput(std::vector<float> &value_function_input, const int idx_sim);

    /**
     * Prepare the input for the value function network.
     *
     * @param action_policy_input
     * @param idx_sim
     */
    void prepareActionPolicyNetworkInput(std::vector<float> &action_policy_input, const int idx_sim);

    /**
     * Compute the allowed future contacts given
     * the current contact and swing time.
     *
     * @param contact
     * @param swing_time
     * @param stance_time
     * @param possible_choice
     */
    void checkExpansion(const int &contact, 
                        const std::vector<float> &swing_time, 
                        const std::vector<float> &stance_time, 
                        std::vector<int> &possible_choice);
};