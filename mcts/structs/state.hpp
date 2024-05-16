#pragma once

/**
 * System's state structure.
 */
struct State
{
    int contact_{};
    Eigen::VectorXf swing_time_{};
    Eigen::VectorXf stance_time_{};

    Eigen::Vector3f position_{};
    Eigen::Vector3f orientation_{};
    Eigen::Vector3f linear_velocity_{};
    Eigen::Vector3f angular_velocity_{};

    Eigen::Vector3f ref_position{};
    Eigen::Vector3f ref_orientation_{};
    Eigen::Vector3f ref_linear_velocity_{};
    Eigen::Vector3f ref_angular_velocity_{};
};