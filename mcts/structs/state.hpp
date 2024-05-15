#pragma once

/**
 * System's state structure.
 */
struct State
{
    int contact_{};
    Eigen::VectorXd swing_time_{};
    Eigen::VectorXd stance_time_{};

    Eigen::Vector3d position_{};
    Eigen::Vector3d orientation_{};
    Eigen::Vector3d linear_velocity_{};
    Eigen::Vector3d angular_velocity_{};

    Eigen::Vector3d ref_position{};
    Eigen::Vector3d ref_orientation_{};
    Eigen::Vector3d ref_linear_velocity_{};
    Eigen::Vector3d ref_angular_velocity_{};
};