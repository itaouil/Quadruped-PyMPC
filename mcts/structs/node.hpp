#pragma once

/**
 * MCTS node structure.
 */
struct Node
{
    int parent_;
    float cost_;
    int contact_;
    int n_visit_;
    int n_selections_;
    std::vector<int> children_;
    std::vector<float> swing_time_;
    std::vector<float> stance_time_;
};