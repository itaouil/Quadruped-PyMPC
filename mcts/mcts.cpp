#include "mcts.hpp"

/**
 * Constructor.
 *
 * @param dt
 * @param horizon
 * @param simulations
 * @param legs
 * @param use_value_function
 * @param use_action_policy
 * @param only_imitation_learning
 */
MCTS::MCTS(float dt, int horizon, int simulations, int legs, bool use_value_function, bool use_action_policy, bool only_imitation_learning) :
        m_legs(legs),
        m_tree_dt(dt),
        m_tree_horizon(horizon),
        m_simulations(simulations),
        m_mt((std::random_device())()),
        m_use_action_policy(use_action_policy),
        m_use_value_function(use_value_function),
        m_only_imitation_learning(only_imitation_learning) {
    // Define mapping from decimal to binary for the contacts
    for (int idx{0}; idx < pow(2, m_legs); idx++) {
        m_contact_map[idx] = decimal2Binary(idx);
    }

    // Define mapping for the action policy output
    m_action_policy_contact_map[0] = 3;
    m_action_policy_contact_map[1] = 5;
    m_action_policy_contact_map[2] = 6;
    m_action_policy_contact_map[3] = 7;
    m_action_policy_contact_map[4] = 9;
    m_action_policy_contact_map[5] = 10;
    m_action_policy_contact_map[6] = 11;
    m_action_policy_contact_map[7] = 12;
    m_action_policy_contact_map[8] = 13;
    m_action_policy_contact_map[9] = 14;
    m_action_policy_contact_map[10] = 15;

    // // Load value function model
    // if (m_use_value_function) {
    //     try {
    //         m_value_function = torch::jit::load(
    //                 "/home/ilyass/workspace/raisim/raisim_workspace/dls_raisim/raisimGymTorch/raisimGymTorch/env/envs/mpc_aliengo/logs/data/models/mcts1.8hz/value_function.pt");
    //         m_value_function.to(at::kCPU);
    //     }
    //     catch (const c10::Error &e) {
    //         std::cerr << "Error loading the value function model\n";
    //     }
    // }

    // // Load action policy model
    // if (m_use_action_policy || m_only_imitation_learning || m_constrain_routine_time) {
    //     try {
    //         m_action_policy = torch::jit::load(
    //                 "/home/ilyass/workspace/raisim/raisim_workspace/dls_raisim/raisimGymTorch/raisimGymTorch/env/envs/mpc_aliengo/logs/data/models/mcts1.8hz/action_policy.pt");
    //         m_action_policy.to(at::kCPU);
    //     }
    //     catch (const c10::Error &e) {
    //         std::cerr << "Error loading the action policy model\n";
    //     }
    // }

    // Start the pythong interpreter
    py::gil_scoped_acquire acquire;

    if (!m_python_initialized) {
        try {
            // Add the directory containing required files
            py::module sys = py::module::import("sys");
            sys.attr("path").attr("append")("../gradient/nominal/");

            // Import the NMPC module and save the NMPC instance
            py::module nmpc_module = py::module::import("centroidal_nmpc_nominal");
            py::object nmpc_class = nmpc_module.attr("Acados_NMPC_Nominal");
            m_nmpc_instance = nmpc_class();

            m_python_initialized = true; // Mark as initialized
        } catch (const py::error_already_set &e) {
            std::cerr << "Error initializing Python: " << e.what() << std::endl;
            throw;
        }
    }
};

/**
 * MCTS routine.
 *
 * @param max_iter
 * @param fixed_contacts
 * @return the MCTS contact sequence to be fed to the MPC
 */
Eigen::MatrixXi MCTS::run(const int max_iter, const std::vector<int>& fixed_contacts) {    
    py::gil_scoped_acquire acquire;

    // Clear previous search
    tree_.clear();

    // Set MCTS routine starting time
    m_current_routine_time = 0;
    m_start_routine_time = std::chrono::steady_clock::now();

    int iter{0};
    int last_node{0};
    int current_node{0};
    m_exit_flag = false;

    // Set initial nodes
    Node root_node{};
    root_node.cost_ = 0;
    root_node.parent_ = 0;
    root_node.n_visit_ = 0;
    root_node.contact_ = current_state_.contact_;

    for (int leg{0}; leg < m_legs; leg++) {
        root_node.swing_time_.push_back(current_state_.swing_time_(leg));
        root_node.stance_time_.push_back(current_state_.stance_time_(leg));
    }

    if (fixed_contacts.empty()) {
        checkExpansion(root_node.contact_, root_node.swing_time_, root_node.stance_time_, root_node.children_);
        tree_.push_back(root_node);
    }
    else {
        tree_.push_back(root_node);

        for (int x{0}; x < fixed_contacts.size(); x++) {
            Node fixed_node{};
            fixed_node.cost_ = 0;
            fixed_node.parent_ = current_node;
            fixed_node.n_visit_ = 0;
            fixed_node.contact_ = fixed_contacts[x];
            for (int leg{0}; leg < m_legs; leg++) {
                fixed_node.swing_time_.push_back((tree_[current_node].swing_time_[leg] + m_tree_dt) * float((1 - m_contact_map[fixed_contacts[x]][leg])));
                fixed_node.stance_time_.push_back((tree_[current_node].stance_time_[leg] + m_tree_dt) * float((m_contact_map[fixed_contacts[x]][leg])));
            }

            tree_.push_back(fixed_node);

            current_node += 1;
        }

        checkExpansion(tree_[current_node].contact_, tree_[current_node].swing_time_, tree_[current_node].stance_time_, tree_[current_node].children_);
    }

    for (int x{0}; x < tree_.size(); x++) {
        std::cout << "Tree node: " << x << std::endl;
        std::cout << "Node contact: " << tree_[x].contact_ << std::endl;
        std::cout << "Node swing time: " << tree_[x].swing_time_[0] << ", " 
                                         << tree_[x].swing_time_[1] << ", "
                                         << tree_[x].swing_time_[2] << ", "
                                         << tree_[x].swing_time_[3] << std::endl;
        std::cout << "Node stance time: " << tree_[x].stance_time_[0] << ", " 
                                         << tree_[x].stance_time_[1] << ", "
                                         << tree_[x].stance_time_[2] << ", "
                                         << tree_[x].stance_time_[3] << std::endl;
        std::cout << "\n" << std::endl;
    }

    // Perform initial MCTS iteration
    last_node = expansionPolicy(current_node);
    simulationPolicy(last_node);
    backPropagation(last_node);

    // Perform MCTS iterations until convergence or max iteration/time reached
    int prev_current_node{-1};
    while (!m_exit_flag) {
        // Compute time passed since MCTS run started
        m_current_routine_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - m_start_routine_time).count();
        
        // Select node to expand
        current_node = treePolicy();

        // Check if MCTS converged
        m_exit_flag = exitPolicy(current_node, prev_current_node);
        if (m_exit_flag) {
            std::cout << "MCTS converged at iter: " << iter << std::endl;
            std::cout << "MCTS prediction cost: " << tree_[current_node].cost_ << "\n" << std::endl;
            continue;
        }

        // Perform MCTS iteration on selected node
        last_node = expansionPolicy(current_node);
        simulationPolicy(last_node);
        backPropagation(last_node);

        // Break if max iterations reached
        if (iter >= max_iter) {
            m_exit_flag = true;

            float min_cost{1e6};
            for (int idx{0}; idx < tree_.size(); idx++) {
                if (tree_[idx].cost_ < min_cost && isLeafNode(idx)) {
                    last_node = idx;
                    min_cost = tree_[idx].cost_;
                }
            }

            std::cout << "MCTS prediction cost: " << min_cost << "\n" << std::endl;
            break;
        }

        // Keep track of which node was selected previously
        prev_current_node = current_node;

        iter += 1;
    }

    // Scale the MCTS contact sequence to match MPC horizon and dt
    Eigen::MatrixXi l_contact_sequence = scaleConverter(0.04, m_tree_dt, state2Binary(getCurrentSeq(last_node)));
    std::cout << "MPC contact sequence: \n" << l_contact_sequence << std::endl;

    return l_contact_sequence;
};

/**
 * Set the current state of the system and its reference.
 *
 * @param state
 * @param reference
 */
void MCTS::setCurrentState(const py::dict state, const py::dict reference) {
    py::gil_scoped_acquire acquire;

    current_state_.contact_ = state["contact"].cast<int>();
    current_state_.swing_time_ = state["swing_time"].cast<Eigen::VectorXd>();
    current_state_.stance_time_ = state["stance_time"].cast<Eigen::VectorXd>();

    current_state_.position_ = state["position"].cast<Eigen::Vector3d>();
    current_state_.orientation_ = state["orientation"].cast<Eigen::Vector3d>();
    current_state_.linear_velocity_ = state["linear_velocity"].cast<Eigen::Vector3d>();
    current_state_.angular_velocity_ = state["angular_velocity"].cast<Eigen::Vector3d>();

    current_state_.ref_position = reference["ref_position"].cast<Eigen::Vector3d>();
    current_state_.ref_orientation_ = reference["ref_orientation"].cast<Eigen::Vector3d>();
    current_state_.ref_linear_velocity_ = reference["ref_linear_velocity"].cast<Eigen::Vector3d>();
    current_state_.ref_angular_velocity_ = reference["ref_angular_velocity"].cast<Eigen::Vector3d>();

    m_state = state;
    m_reference = reference;

    std::cout << "Current state set successfully." << std::endl;
}

/**
 * Bundle and return the values used
 * a posteriori for the training process
 * of the value and action policies.
 *
 * @return tuple of system state and MCTS tree
 */
std::tuple<Eigen::Matrix3d, Eigen::Matrix3d,
           Eigen::VectorXd, Eigen::VectorXd,
           Eigen::Vector4d, float, std::vector<std::tuple < int, float, int, int, std::vector < float>>>>

MCTS::getMCTSData() {
    // We need to unroll the Node structure
    // for the logger in Python to be able
    // to understand the types, hence why
    // we do not use Node directly in the
    // vector definition, but instead a tuple
    std::vector < std::tuple < int, float, int, int,
            std::vector < float>>> l_mcts_tree;

    // Populate MCTS tree
    for (const auto &node: tree_) {
        l_mcts_tree.push_back(std::make_tuple(node.parent_, node.cost_,
                                              node.contact_, node.n_visit_,
                                              node.swing_time_));
    }

    // // Return state-output tuple
    // return std::make_tuple(current_state_.rpy_, current_state_.ref_rpy_,
    //                        current_state_.x_op_, current_state_.swing_time_,
    //                        current_state_.user_input_, current_state_.z_ref_, 
    //                        l_mcts_tree);
};

/**
 * Call action policy to obtain action rollout.
 *
 * @param p_idx_sim
 * @return cost output from the value function network
 */
int MCTS::computeNodeActionPolicyRollout(const int p_idx_sim) {
    // // Prepare input to feed the action policy
    // std::vector<float> action_policy_input{};
    // prepareActionPolicyNetworkInput(action_policy_input, p_idx_sim);

    // // Infer best action to expand
    // std::vector <torch::jit::IValue> inputs;
    // auto options = torch::TensorOptions().dtype(at::kFloat);
    // inputs.push_back(torch::from_blob(action_policy_input.data(),
    //                                   {1, static_cast<long>(action_policy_input.size())},
    //                                   options));
    // at::Tensor output = m_action_policy.forward(inputs).toTensor();

    // // Return index whose probablity is the highest
    // return m_action_policy_contact_map[torch::argmax(output, 1).item<int>()];

    return 0;
}

/**
 * Call value function to obtain node QP estimate.
 *
 * @param p_idx_sim
 * @return cost output from the value function network
 */
float MCTS::computeNodeValueFunctionCost(const int p_idx_sim) {
    // // Prepare input to feed the value function
    // std::vector<float> value_function_input{};
    // prepareValueFunctionNetworkInput(value_function_input, p_idx_sim);

    // // Infer QP cost from value function
    // std::vector <torch::jit::IValue> inputs;
    // auto options = torch::TensorOptions().dtype(at::kFloat);
    // inputs.push_back(torch::from_blob(value_function_input.data(),
    //                                   {1, static_cast<long>(value_function_input.size())},
    //                                   options));
    // at::Tensor output = m_value_function.forward(inputs).toTensor();

    // // Returned scaled QP cost (network is trained to ouput value between 0-1)
    // return output.item<float>() * 2500;

    return 0.0;
}

/**
 * Tree search to find which node to expand next.  
 *
 * @return node idx
 */
int MCTS::treePolicy() {
    int min_cost_idx{0};
    float min_cost{1e6};

    std::uniform_real_distribution<float> l_dist_epsilon{0.0, 1.0};
    float l_epsilon = l_dist_epsilon(m_mt);

    for (int idx{1}; idx < tree_.size(); idx++) {
        // if (l_epsilon > 0.8) {
        //     if (!tree_[idx].children_.empty() && !isLeafNode(idx)) {
        //         return idx;
        //     }
        //     else {
        //         continue;
        //     }
        // }

        float lcb = tree_[idx].cost_;
        if (lcb < min_cost && !tree_[idx].children_.empty()) {
            min_cost = lcb;
            min_cost_idx = idx;
        }

        for (int leg{0}; leg < m_legs; leg++) {
            if (tree_[tree_[idx].parent_].swing_time_[leg] == m_min_swing_time && tree_[tree_[idx].parent_].contact_ == 9 || tree_[tree_[idx].parent_].contact_ == 6) {
                int step = getCurrentSeq(idx).size();

                if (step >= 10)
                    std::cout << "Step: " << getCurrentSeq(idx).size() << "  Cost: " << tree_[idx].cost_ << std::endl;
                else
                    std::cout << "Step: " << getCurrentSeq(idx).size() << "   Cost: " << tree_[idx].cost_ << std::endl;
            }
        }
    }
    // std::cout << "=============================================" << std::endl;

    return min_cost_idx;
};

/**
 * Expand node returned by tree policy.
 *
 * @param parent_idx
 * @return first node idx added to the tree
 */
int MCTS::expansionPolicy(const int parent_idx) {
    if (isLeafNode(parent_idx) || tree_[parent_idx].children_.empty()) {
        return parent_idx;
    }

    int child_idx{static_cast<int>(tree_.size())};

    if (m_only_imitation_learning) {
        int l_action_policy_contact = computeNodeActionPolicyRollout(parent_idx);
        addNode(parent_idx, l_action_policy_contact);
        tree_[parent_idx].children_.clear();
    }
    else {
        m_action_policy_expansion = false;
        std::uniform_real_distribution<float> dist_epsilon{0.0, 1.0};

        if (m_use_action_policy && dist_epsilon(m_mt) <= 0.8) {
            // Add action policy contact to the tree
            int l_action_policy_contact = computeNodeActionPolicyRollout(parent_idx);
            addNode(parent_idx, l_action_policy_contact);

            // Remove action policy contact from the parent feasible contacts if it exists
            for (int x{0}; x < tree_[parent_idx].children_.size(); x++) {
                if (tree_[parent_idx].children_[x] == l_action_policy_contact) {
                    tree_[parent_idx].children_.erase(tree_[parent_idx].children_.begin() + x);
                    break;
                }
            }

            // Add sibling contacts to be evaluated with value function
            if (m_use_value_function) {
                for (int x{0}; x < tree_[parent_idx].children_.size(); x++) 
                    addNode(parent_idx, tree_[parent_idx].children_[x]);

                tree_[parent_idx].children_.clear();
            }

            m_action_policy_expansion = true;
        }
        else {
            for (int x{0}; x < tree_[parent_idx].children_.size(); x++)
                addNode(parent_idx, tree_[parent_idx].children_[x]);
            
            tree_[parent_idx].children_.clear();
        }
    }

    return child_idx;
};

/**
 * Add new expanded node to the tree.
 */
void MCTS::addNode(const int parent_idx, const int contact) {
    Node child_node{};
    child_node.contact_ = contact;
    child_node.parent_ = parent_idx;

    for (int leg{0}; leg < m_legs; leg++) {
        child_node.stance_time_.push_back((tree_[parent_idx].stance_time_[leg] + m_tree_dt) * float((m_contact_map[child_node.contact_][leg])));
        child_node.swing_time_.push_back((tree_[parent_idx].swing_time_[leg] + m_tree_dt) * float((1 - m_contact_map[child_node.contact_][leg])));
    }

    checkExpansion(child_node.contact_, child_node.swing_time_, child_node.stance_time_, child_node.children_);

    tree_.push_back(child_node);
}

/**
 * MCTS exit condition (if leaf node is selected).
 *
 * @param current_idx
 * @param last_idx
 * @return true if exit condition reached, false otherwise
 */
bool MCTS::exitPolicy(const int &current_idx, const int &last_idx) {
    if (current_idx == last_idx && isLeafNode(current_idx))
        return true;
    else
        return false;

    // if (isLeafNode(current_idx))
    //     return true;
    // else
    //     return false;
};

/**
 * Check if given node violates the swing constraints.
 *
 * @param contact
 * @param swing_time
 * @param stance_time
 * @return true if constraint violated, false otherwise
 */
int MCTS::constraintViolation(const int &contact, const float &swing_time, const float &stance_time) const {
    if (!contact && swing_time < m_min_swing_time)
        return 0;
    else if (!contact && swing_time >= m_max_swing_time)
        return 1;
    else if (contact && stance_time < m_min_stance_time)
        return 2;
    else
        return 3;
}

/**
 * Solve OCPs in batch for the computed rollouts using ACADOS (python binding).
 *
 * @param rollouts
 */
std::vector<float> MCTS::solveOCPs(const std::vector<Eigen::MatrixXi> &rollouts) {
    try {
        // Compute costs for the rollouts
        py::object result = m_nmpc_instance.attr("ocp_batch_solver")(m_state, m_reference, rollouts);
        std::vector<float> ocp_costs = result.cast<std::vector<float>>();

        // // Print the ocp costs obtained matrix
        // for (const auto& cost : ocp_costs) {
        //     std::cout << "The cost is: " << cost << std::endl;
        // }

        return ocp_costs;
    } catch (const py::error_already_set &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

/**
 * Compute the expanded node
 *
 * @param node_idx
 */
void MCTS::simulationPolicy(const int node_idx) {
    if (m_only_imitation_learning || tree_[tree_[node_idx].parent_].children_.size() == 1) {
        tree_[node_idx].n_visit_ += 1;
        tree_[node_idx].cost_ = tree_[tree_[node_idx].parent_].cost_ - 0.1;
        return;
    }

    // Clear previous batch of evaluated contact sequences
    std::vector<Eigen::MatrixXi> l_rollouts{};
    
    bool l_first_node{true};
    
    for (int idx_sim{node_idx}; idx_sim < tree_.size(); idx_sim++) {
        // Compute value function cost using NN if this one is enabled
        float l_value_function_cost{0};
        if (m_use_value_function) {
            l_value_function_cost = computeNodeValueFunctionCost(idx_sim);

            if (m_simulations == 0 || 
                m_action_policy_expansion && !l_first_node ||
                m_constrain_routine_time && m_current_routine_time >= m_max_routine_time) {
                tree_[idx_sim].n_visit_ += 1;
                tree_[idx_sim].cost_ = l_value_function_cost;
                continue;
            }
        }

        // Cost that penalizes fast swing to stance transitions
        float l_swing_cost{0};

        // Cost that penalizes too many legs in swing
        float l_contact_cost{0};

        // Perform simulations to obtain expected average cost for the node
        for (int th{0}; th < m_simulations; th++) {
            int contact{tree_[idx_sim].contact_};
            std::vector<int> current_seq{getCurrentSeq(idx_sim)};
            std::vector<float> l_swing_time{tree_[idx_sim].swing_time_};
            std::vector<float> l_stance_time{tree_[idx_sim].stance_time_};

            // // Compute cost penalizations
            // if (th == 0) {
            //     for (int leg{0}; leg < m_legs; leg++) {
            //         int l_leg_contact_switches{0};
            //         float l_temp_leg_swing_time{0};
            //         float l_leg_total_swing_time{tree_[0].swing_time_[leg]};
            //         std::vector<float> l_parent_swing_time{tree_[0].swing_time_};

            //         for (int step{0}; step < current_seq.size(); step++) {
            //             if (m_contact_map[current_seq[step]][leg] == 1) {
            //                 if (step == 0 && l_leg_total_swing_time >= m_tree_dt || step > 0 && m_contact_map[current_seq[step - 1]][leg] == 0) {
            //                     l_leg_contact_switches += 1;

            //                     l_leg_total_swing_time += l_temp_leg_swing_time;
            //                     l_temp_leg_swing_time = 0;
            //                 }
            //             }
            //             else {
            //                 l_temp_leg_swing_time += m_tree_dt;
            //             }

            //             if (leg == 0)
            //                 l_contact_cost += (m_legs - m_contact_map[current_seq[step]].sum()) * m_rc;
            //         }

            //         if (l_leg_contact_switches > 0) {
            //             float l_mean_swing_time = l_leg_total_swing_time / l_leg_contact_switches;

            //             // std::cout << "Leg: " << leg << ". Mean swing frequency: " << l_leg_total_swing_time / l_leg_contact_switches << std::endl;

            //             if (l_mean_swing_time < m_max_swing_time)
            //                 l_swing_cost += (m_max_swing_time - l_mean_swing_time) * 50;
            //         }

            //         // std::cout << "Leg parent swing time: " << l_parent_swing_time[leg] << std::endl;
            //         // std::cout << "Leg contact sequence: " << current_seq << std::endl;
            //         // std::cout << "Leg total swing time: " << l_leg_total_swing_time << std::endl;
            //         // std::cout << "Leg contact switches: " << l_leg_contact_switches << std::endl;
            //         // std::cout << "Leg swing cost: " << l_swing_cost << std::endl;
            //      }

            //     // std::cout << "Contact cost: " << l_contact_cost << std::endl;
            //     // std::cout << "\n" << std::endl;
            // }

            for (int step{static_cast<int>(current_seq.size())}; step < m_tree_horizon; step++) {
                std::vector<int> possible_choices{};
                checkExpansion(contact, l_swing_time, l_stance_time, possible_choices);

                std::uniform_int_distribution<int> dist(0, possible_choices.size() - 1);
                contact = possible_choices[dist(m_mt)];
                current_seq.push_back(contact);

                for (int leg{0}; leg < m_legs; leg++) {
                    l_stance_time[leg] = (l_stance_time[leg] + m_tree_dt) * float((m_contact_map[contact][leg]));
                    l_swing_time[leg] = (l_swing_time[leg] + m_tree_dt) * float((1 - m_contact_map[contact][leg]));
                }
            }

            // Add simulated contact sequence among those that need to be evaluated
            l_rollouts.push_back(state2Binary(current_seq));
        }

        // // Add full-stance contact sequence to also be evaluated
        // std::vector<int> l_full_stance_seq{};
        // for (int x{0}; x < m_tree_horizon; x++) {
        //     l_full_stance_seq.push_back(15);
        // }
        // l_rollouts.push_back(l_full_stance_seq);

        // Solve the OCPs in batch
        std::vector<float> l_rollout_costs = solveOCPs(l_rollouts);

        // Compute average cost from the simulations
        float cost = 0;
        int n_visit = 0;
        for (int idx{0}; idx < l_rollout_costs.size(); idx++) {
            if (l_rollout_costs[idx] != -1) {
                n_visit += 1;
                cost += l_rollout_costs[idx];
            }
        }

        // Normalize cost w.r.t horizon
        // cost /= m_tree_horizon;

        // // Add penalization costs
        // cost += l_swing_cost;

        // std::cout << "Contact sequence: \n" << state2Binary(getCurrentSeq(idx_sim)) << std::endl;
        // std::cout << "Prediction cost: " << cost << ". Step: " << getCurrentSeq(idx_sim).size() << std::endl;
        // std::cout << "\n" << std::endl;

        // Increase counter for number of node visits
        tree_[idx_sim].n_visit_ = n_visit + 1;

        // Set cost for the node
        if (m_use_value_function) {
            tree_[idx_sim].cost_ = 0.4 * (cost / tree_[idx_sim].n_visit_) + 0.6 * l_value_function_cost;
        }
        else {
            tree_[idx_sim].cost_ = cost / tree_[idx_sim].n_visit_ + l_contact_cost;
            // std::cout << "Total node cost: " << tree_[idx_sim].cost_  << std::endl;
        }

        if (tree_[idx_sim].n_visit_ == 0) {
            tree_[idx_sim].cost_ = 1e6;
            tree_[idx_sim].n_visit_ += 1;
        }

        l_first_node = false;
    }
};

/**
 * Update the cost of the node whose
 * children where simulated before
 * and backpropagate the cost iteratively
 * up the tree until the root node is reached.
 *
 * @param node_idx
 */
void MCTS::backPropagation(const int &node_idx) {
//    std::cout << "=====BACKPROPAGATION=======" << std::endl;
    int idx{node_idx};
    float new_cost{0}; // New calculated cost
    float new_visit{0}; // number of new simulation done

    // Compute cost for the simulation of the expanded nodes
    for (int last_child{idx}; last_child < tree_.size(); last_child++) {
        new_cost = tree_[last_child].cost_ * float(tree_[last_child].n_visit_);
        new_visit = float(tree_[last_child].n_visit_);

        // Update cost and visits of the node from which the expansion happened
        tree_[tree_[last_child].parent_].cost_ =
                (tree_[tree_[last_child].parent_].cost_ * float(tree_[tree_[last_child].parent_].n_visit_) +
                 new_cost) / (tree_[tree_[last_child].parent_].n_visit_ + new_visit);
        tree_[tree_[last_child].parent_].n_visit_ += new_visit;
    }

    // Perform the backpropagation pass
    idx = tree_[idx].parent_;
    while (idx != 0) {
        // update cost
        tree_[tree_[idx].parent_].cost_ =
                (tree_[tree_[idx].parent_].cost_ * float(tree_[tree_[idx].parent_].n_visit_) + new_cost) /
                (tree_[tree_[idx].parent_].n_visit_ + new_visit);

        // update the n_visit
        tree_[tree_[idx].parent_].n_visit_ += new_visit;

        //move to the parent
        idx = tree_[idx].parent_;
    }
//    std::cout << "=======================\n\n" <<std::endl;
};

/**
 * The function given a node index
 * retraces the tree using the parent
 * indices until the root node to
 * retrieve the contact sequence for
 * the given tree path.
 *
 * @param node_idx
 * @return contact sequence of a given node index
 */
std::vector<int> MCTS::getCurrentSeq(const int node_idx) {
    int l_idx{node_idx};

    std::vector<int> contact_seq;
    while (l_idx != 0) {
        contact_seq.push_back(tree_[l_idx].contact_);
        l_idx = tree_[l_idx].parent_;
    }

    std::reverse(contact_seq.begin(), contact_seq.end());
    return contact_seq;
};

/**
 * Convert a decimal sequence to a binary matrix representation.
 *
 * @param vec
 * @return binary contact sequence
 */
Eigen::MatrixXi MCTS::state2Binary(const std::vector<int> &vec) {
    Eigen::MatrixXi contact_seq = Eigen::MatrixXi::Ones(m_legs, m_tree_horizon);

    for (int idx{0}; idx < vec.size(); idx++) {
        contact_seq.block<4, 1>(0, idx) = m_contact_map[vec[idx]];
    }

    return contact_seq;
};

/**
 * Converts a contact from a binary representation to a decimal one.
 *
 * @param vec
 * @return decimal value corresponding to the binary contact
 */
int MCTS::binary2Decimal(const Eigen::VectorXi &vec) {
    int n{0};

    for (int idx{0}; idx < vec.size(); idx++) {
        n += vec(vec.size() - 1 - idx) * pow(2, idx);
    }

    return n;
}

/**
 * Check if a tree node is a leaf node.
 *
 * @param node_idx
 * @return true if leaf node, otherwise false
 */
bool MCTS::isLeafNode(const int &node_idx) {
    int step{0};
    int idx{node_idx};

    while (idx != 0) {
        idx = tree_[idx].parent_;
        step += 1;
    }

    return step == m_tree_horizon;
}

/**
 * Converts a contact from a decimal representation to a binary one.
 *
 * @param n
 * @return binary representation of the contact in decimal
 */
Eigen::VectorXi MCTS::decimal2Binary(int n) {
    if (n >= pow(2, m_legs)) {
        Eigen::VectorXi out_vec{};
        out_vec.setZero(m_legs);
        return out_vec;
    }

    Eigen::VectorXi vec{};
    vec.setZero(m_legs);
    
    Eigen::VectorXi out_vec{};
    out_vec.setZero(m_legs);

    int i{0}, num{n};
    while (n != 0) {
        vec[i] = n % 2;
        i++;
        n = n / 2;
    }

    for (i = i - 1; i >= 0; i--) {
        out_vec[m_legs - 1 - i] = vec[i];
    }

    return out_vec;
}

/**
 * Convert the MCTS sequence to a matching MPC horizon and dt.
 *
 * @param mpc_dt
 * @param mcts_dt
 * @param mcts_seq
 * @return contact sequence matching the MPC dt and horizon
 */
Eigen::MatrixXi MCTS::scaleConverter(float mpc_dt, float mcts_dt, const Eigen::MatrixXi &mcts_seq) {
    int repetitions = static_cast<int>((mcts_dt / mpc_dt) + 0.6);

    Eigen::MatrixXi mpc_seq{};
    mpc_seq.setZero(m_legs, mcts_seq.cols() * repetitions);

    for (int column{0}; column < mcts_seq.cols(); column++)
        for (int repetition{0}; repetition < repetitions; repetition++)
            mpc_seq.block(0, column * repetitions + repetition, m_legs, 1) = mcts_seq.block(0, column, m_legs, 1);

    return mpc_seq;
}

/**
 * Prepare the input for the value function network.
 *
 * @param value_function_input
 * @param idx_sim
 */
void MCTS::prepareValueFunctionNetworkInput(std::vector<float> &value_function_input, const int idx_sim) {
    // // Z error
    // float z_error = current_state_.z_ref_ - current_state_.x_op_(2);
    // value_function_input.push_back(z_error);

    // // Rotation error
    // Eigen::Vector3d rpy_op;
    // Eigen::Vector3d rpy_ref;
    // rpy_op(0) = atan2f((float) current_state_.rpy_(2,1), (float) current_state_.rpy_(2,2));
    // rpy_op(1) = -asinf((float) current_state_.rpy_(2,0));
    // rpy_op(2) = atan2f((float) current_state_.rpy_(1,0), (float) current_state_.rpy_(0,0));
    // rpy_ref(0) = atan2f((float) current_state_.R_ref_(2,1), (float) current_state_.R_ref_(2,2));
    // rpy_ref(1) = -asinf((float) current_state_.R_ref_(2,0));
    // rpy_ref(2) = atan2f((float) current_state_.R_ref_(1,0), (float) current_state_.R_ref_(0,0));
    // value_function_input.push_back(rpy_ref(0) - rpy_op(0));
    // value_function_input.push_back(rpy_ref(1) - rpy_op(1));
    // // value_function_input.push_back(rpy_ref(2) - rpy_op(2));

    // // Linear velocity error
    // auto linear_velocity_xop = current_state_.rpy_.transpose() * current_state_.x_op_.block<3, 1>(3, 0);
    // auto linear_velocity_ref = current_state_.rpy_.transpose() * current_state_.user_input_.block<3, 1>(0, 0);
    // value_function_input.push_back(linear_velocity_ref(0) - linear_velocity_xop(0));
    // value_function_input.push_back(linear_velocity_ref(1) - linear_velocity_xop(1));
    // value_function_input.push_back(linear_velocity_ref(2) - linear_velocity_xop(2));

    // // Angular velocity error
    // auto angular_velocity_xop = current_state_.x_op_.block<3, 1>(9, 0);
    // Eigen::Vector3d angular_velocity_ref{0, 0, current_state_.user_input_(3)};
    // value_function_input.push_back(angular_velocity_ref(0) - angular_velocity_xop(0));
    // value_function_input.push_back(angular_velocity_ref(1) - angular_velocity_xop(1));
    // value_function_input.push_back(angular_velocity_ref(2) - angular_velocity_xop(2));

    // // FL foot pose
    // auto fl_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*0, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // value_function_input.push_back(fl_foot_pose_op(0));
    // value_function_input.push_back(fl_foot_pose_op(1));
    // value_function_input.push_back(fl_foot_pose_op(2));

    // // FR foot pose
    // auto fr_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*1, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // value_function_input.push_back(fr_foot_pose_op(0));
    // value_function_input.push_back(fr_foot_pose_op(1));
    // value_function_input.push_back(fr_foot_pose_op(2));

    // // RL foot pose
    // auto rl_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*2, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // value_function_input.push_back(rl_foot_pose_op(0));
    // value_function_input.push_back(rl_foot_pose_op(1));
    // value_function_input.push_back(rl_foot_pose_op(2));

    // // RR foot pose
    // auto rr_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*3, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // value_function_input.push_back(rr_foot_pose_op(0));
    // value_function_input.push_back(rr_foot_pose_op(1));
    // value_function_input.push_back(rr_foot_pose_op(2));

    // // Root swing time
    // value_function_input.push_back(current_state_.swing_time_[0]);
    // value_function_input.push_back(current_state_.swing_time_[1]);
    // value_function_input.push_back(current_state_.swing_time_[2]);
    // value_function_input.push_back(current_state_.swing_time_[3]);

    // // // Tree node contact chain
    // // std::vector<int> current_seq{getCurrentSeq(idx_sim)};
    // // for (int x{0}; x < current_seq.size(); x++) {
    // //     auto binary_value = decimal2Binary(current_seq[x]);

    // //     for (int y{0}; y < binary_value.size(); y++)
    // //         value_function_input.push_back(binary_value(y));
    // // }

    // // for (int x{0}; x < m_tree_horizon*m_legs - current_seq.size() * m_legs ; x++)
    // //     value_function_input.push_back(-1);

    // // Tree node contact chain
    // std::vector<int> current_seq{getCurrentSeq(idx_sim)};
    // for (int x{0}; x < current_seq.size(); x++)
    //     value_function_input.push_back(current_seq[x]);

    // for (int x{0}; x < m_tree_horizon - current_seq.size() * m_legs ; x++)
    //     value_function_input.push_back(-1);

    // // std::cout << "Value function input: " << std::endl;
    // // for (int x{0}; x < value_function_input.size(); x++) {
    // //     std::cout << value_function_input[x] << std::endl;
    // // }
    return;
}

/**
 * Prepare the input for the action policy network.
 *
 * @param action_policy_input
 * @param idx_sim
 */
void MCTS::prepareActionPolicyNetworkInput(std::vector<float> &action_policy_input, const int idx_sim) {
    // // Z error
    // float z_error = current_state_.z_ref_ - current_state_.x_op_(2);
    // action_policy_input.push_back(z_error);

    // // Rotation error
    // Eigen::Vector3d rpy_op;
    // Eigen::Vector3d rpy_ref;
    // rpy_op(0) = atan2f((float) current_state_.rpy_(2,1), (float) current_state_.rpy_(2,2));
    // rpy_op(1) = -asinf((float) current_state_.rpy_(2,0));
    // rpy_op(2) = atan2f((float) current_state_.rpy_(1,0), (float) current_state_.rpy_(0,0));
    // rpy_ref(0) = atan2f((float) current_state_.R_ref_(2,1), (float) current_state_.R_ref_(2,2));
    // rpy_ref(1) = -asinf((float) current_state_.R_ref_(2,0));
    // rpy_ref(2) = atan2f((float) current_state_.R_ref_(1,0), (float) current_state_.R_ref_(0,0));
    // action_policy_input.push_back(rpy_ref(0) - rpy_op(0));
    // action_policy_input.push_back(rpy_ref(1) - rpy_op(1));
    // action_policy_input.push_back(rpy_ref(2) - rpy_op(2));

    // // Linear velocity error
    // auto linear_velocity_xop = current_state_.rpy_.transpose() * current_state_.x_op_.block<3, 1>(3, 0);
    // auto linear_velocity_ref = current_state_.rpy_.transpose() * current_state_.user_input_.block<3, 1>(0, 0);
    // action_policy_input.push_back(linear_velocity_ref(0) - linear_velocity_xop(0));
    // action_policy_input.push_back(linear_velocity_ref(1) - linear_velocity_xop(1));
    // action_policy_input.push_back(linear_velocity_ref(2) - linear_velocity_xop(2));

    // // Angular velocity error
    // auto angular_velocity_xop = current_state_.x_op_.block<3, 1>(9, 0);
    // Eigen::Vector3d angular_velocity_ref{0, 0, current_state_.user_input_(3)};
    // action_policy_input.push_back(angular_velocity_ref(0) - angular_velocity_xop(0));
    // action_policy_input.push_back(angular_velocity_ref(1) - angular_velocity_xop(1));
    // action_policy_input.push_back(angular_velocity_ref(2) - angular_velocity_xop(2));

    // // // FL foot error
    // // auto fl_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*0, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // // action_policy_input.push_back(fl_foot_pose_op(0));
    // // action_policy_input.push_back(fl_foot_pose_op(1));
    // // action_policy_input.push_back(fl_foot_pose_op(2));

    // // // FR foot error
    // // auto fr_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*1, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // // action_policy_input.push_back(fr_foot_pose_op(0));
    // // action_policy_input.push_back(fr_foot_pose_op(1));
    // // action_policy_input.push_back(fr_foot_pose_op(2));

    // // // RL foot error
    // // auto rl_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*2, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // // action_policy_input.push_back(rl_foot_pose_op(0));
    // // action_policy_input.push_back(rl_foot_pose_op(1));
    // // action_policy_input.push_back(rl_foot_pose_op(2));

    // // // RR foot error
    // // auto rr_foot_pose_op = current_state_.rpy_.transpose() * (current_state_.x_op_.block<3, 1>(12+3*3, 0) - current_state_.x_op_.block<3, 1>(0, 0));
    // // action_policy_input.push_back(rr_foot_pose_op(0));
    // // action_policy_input.push_back(rr_foot_pose_op(1));
    // // action_policy_input.push_back(rr_foot_pose_op(2));

    // // Root swing time
    // action_policy_input.push_back(current_state_.swing_time_[0]);
    // action_policy_input.push_back(current_state_.swing_time_[1]);
    // action_policy_input.push_back(current_state_.swing_time_[2]);
    // action_policy_input.push_back(current_state_.swing_time_[3]);

    // // // Tree node contact chain
    // // std::vector<int> current_seq{getCurrentSeq(idx_sim)};
    // // for (int x{0}; x < current_seq.size(); x++) {
    // //     auto binary_value = decimal2Binary(current_seq[x]);

    // //     for (int y{0}; y < binary_value.size(); y++)
    // //         action_policy_input.push_back(binary_value(y));
    // // }

    // // for (int x{0}; x < (m_tree_horizon - 1) * m_legs - current_seq.size() * m_legs ; x++)
    // //     action_policy_input.push_back(-1);

    // // Tree node contact chain
    // std::vector<int> current_seq{getCurrentSeq(idx_sim)};
    // for (int x{0}; x < current_seq.size(); x++)
    //     action_policy_input.push_back(current_seq[x]);

    // for (int x{0}; x < (m_tree_horizon - 1) - current_seq.size(); x++)
    //     action_policy_input.push_back(-1);

    // // std::cout << "Action policy input: " << std::endl;
    // // for (int x{0}; x < action_policy_input.size(); x++) {
    // //     std::cout << action_policy_input[x] << std::endl;
    // // }

    return;
}

/**
 *  The function computes the children/choices
 *  of contact for a given node, so that these
 *  can be expanded in the next iteration. More
 *  precisely, only the contacts that do not violate
 *  the minimum flying time constraint for any leg
 *  are kept. The rest are all removed.
 *
 * @param contact
 * @param swing_time
 * @param stance_time
 * @param possible_choice
 */
void
MCTS::checkExpansion(const int &contact, const std::vector<float> &swing_time, const std::vector<float> &stance_time, std::vector<int> &possible_choice) {
    std::vector<int>::iterator it{};
    int range{static_cast<int>(pow(2, m_legs))};

    // Initially fill all possible choices
    // for the child node with all the system
    // combinations (2^4) for a quadruped
    // for (int idx{0}; idx < range; idx++) {
    //     if (idx == 3 ||
    //         idx == 5 ||
    //         idx == 6 ||
    //         idx == 7 ||
    //         idx == 9 ||
    //         idx == 10 ||
    //         idx == 11 ||
    //         idx == 12 ||
    //         idx == 13 ||
    //         idx == 14 ||
    //         idx == 15)
    //         possible_choice.push_back(idx);
    // }
    for (int idx{0}; idx < range; idx++) {
        if (idx == 5 ||
            idx == 6 ||
            idx == 7 ||
            idx == 9 ||
            idx == 10 ||
            idx == 11 ||
            idx == 13 ||
            idx == 14 ||
            idx == 15)
            possible_choice.push_back(idx);
    }

    // Check legs' timing violations
    for (int leg{0}; leg < m_legs; leg++) {
        auto l_violation = constraintViolation(m_contact_map[contact][leg], swing_time[leg], stance_time[leg]);

        if (l_violation == 0) {
            // Remove all possible contacts that
            // cause the foot to go in stance as
            // the minimum swing time for the leg
            // has not been reached
            for (int idx{0}; idx < range; idx++)
            {
                if (m_contact_map[idx][leg]) {
                    it = std::find(possible_choice.begin(), possible_choice.end(), idx);
                    if (it != possible_choice.end())
                    {
                        possible_choice.erase(it);
                    }
                }
            }
        }
        else if (l_violation == 1) {
            // Remove all possible contacts that
            // cause the foot to remain in swing as
            // the maximum swing time for the leg
            // has been reached
            for (int idx{0}; idx < range; idx++)
            {
                if (!m_contact_map[idx][leg]) {
                    it = std::find(possible_choice.begin(), possible_choice.end(), idx);
                    if (it != possible_choice.end())
                    {
                        possible_choice.erase(it);
                    }
                }
            }
        }
        else if (l_violation == 2) {
            // Remove all possible contacts that
            // cause the foot to go in swing as
            // the minimum stance time for the leg
            // has not been reached
            for (int idx{0}; idx < range; idx++)
            {
                if (!m_contact_map[idx][leg]) {
                    it = std::find(possible_choice.begin(), possible_choice.end(), idx);
                    if (it != possible_choice.end())
                    {
                        possible_choice.erase(it);
                    }
                }
            }
        }
    }
};