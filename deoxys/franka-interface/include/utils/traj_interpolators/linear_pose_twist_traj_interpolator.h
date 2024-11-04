// Copyright 2022 Yifeng Zhu

#include "base_traj_interpolator.h"
#include <Eigen/Dense>
#ifndef DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_TRAJ_INTERPOLATORS_LINEAR_POSE_TWIST_TRAJ_INTERPOLATOR_H_
#define DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_TRAJ_INTERPOLATORS_LINEAR_POSE_TWIST_TRAJ_INTERPOLATOR_H_

namespace traj_utils {
class LinearPoseTwistTrajInterpolator : public BaseTrajInterpolator {
private:
  Eigen::Vector3d p_start_;
  Eigen::Vector3d p_goal_;
  Eigen::Vector3d last_p_t_;
  Eigen::Vector3d prev_p_goal_;

  Eigen::Quaterniond q_start_;
  Eigen::Quaterniond q_goal_;
  Eigen::Quaterniond last_q_t_;
  Eigen::Quaterniond prev_q_goal_;

  Eigen::Vector3d twist_trans_start_;
  Eigen::Vector3d twist_trans_goal_;
  Eigen::Vector3d last_twist_trans_t_;
  Eigen::Vector3d prev_twist_trans_goal_;

  Eigen::Vector3d twist_rot_start_;
  Eigen::Vector3d twist_rot_goal_;
  Eigen::Vector3d last_twist_rot_t_;
  Eigen::Vector3d prev_twist_rot_goal_;

  double dt_;
  double last_time_;
  double max_time_;
  double start_time_;
  bool start_;
  bool first_goal_;

public:
  inline LinearPoseTwistTrajInterpolator()
      : dt_(0.), last_time_(0.), max_time_(1.), start_time_(0.), start_(false),
        first_goal_(true){};

  inline ~LinearPoseTwistTrajInterpolator(){};

  inline void Reset(const double &time_sec, const Eigen::Vector3d &p_start,
                    const Eigen::Quaterniond &q_start,
                    const Eigen::Vector3d &p_goal,
                    const Eigen::Quaterniond &q_goal,
                    const Eigen::Vector3d &twist_trans_start,
                    const Eigen::Vector3d &twist_rot_start,
                    const Eigen::Vector3d &twist_trans_goal,
                    const Eigen::Vector3d &twist_rot_goal,
                    const int &policy_rate,
                    const int &rate,
                    const double &traj_interpolator_time_fraction) {

    dt_ = 1. / static_cast<double>(rate);
    last_time_ = time_sec;
    max_time_ =
        1. / static_cast<double>(policy_rate) * traj_interpolator_time_fraction;
    start_time_ = time_sec;

    start_ = false;

    if (first_goal_) {
      p_start_ = p_start;
      q_start_ = q_start;

      prev_p_goal_ = p_start;
      prev_q_goal_ = q_start;

      twist_trans_start_ = twist_trans_start;
      twist_rot_start_ = twist_rot_start;

      prev_twist_trans_goal_ = twist_trans_start;
      prev_twist_rot_goal_ = twist_rot_start;
      first_goal_ = false;
      // std::cout << "First goal" << p_start << std::endl;
    } else {
      // If the goal is already set, use prev goal as the starting point of
      // interpolation.

      prev_p_goal_ = p_goal_;
      prev_q_goal_ = q_goal_;

      // TODO: this assumes that the trajectory is tracking closely. Could be dangenerous if the robot haven't reach closely to the prev goal.
      p_start_ = prev_p_goal_;
      q_start_ = prev_q_goal_;

      prev_twist_trans_goal_ = twist_trans_goal_;
      prev_twist_rot_goal_ = twist_rot_goal_;

      twist_trans_start_ = prev_twist_trans_goal_;
      twist_rot_start_ = prev_twist_rot_goal_; 
    }

    p_goal_ = p_goal;
    q_goal_ = q_goal;

    twist_trans_goal_ = twist_trans_goal;
    twist_rot_goal_ = twist_rot_goal;

    // std::cout<<"rrrrrrrrrrrrrset goal"<<p_goal_<<" pstart :" << p_start_<<std::endl;

    // Flip the sign if the dot product of quaternions is negative
    if (q_goal_.coeffs().dot(q_start_.coeffs()) < 0.0) {
      q_start_.coeffs() << -q_start_.coeffs();
    }
  };

  inline void GetNextStep(const double &time_sec, Eigen::Vector3d &p_t,
                          Eigen::Quaterniond &q_t, Eigen::Vector3d &twist_trans_t, Eigen::Vector3d &twist_rot_t) {

    if (!start_) {
      start_time_ = time_sec;
      last_p_t_ = p_start_;
      last_q_t_ = q_start_;
      last_twist_trans_t_ = twist_trans_start_;
      last_twist_rot_t_ = twist_rot_start_;
      start_ = true;
    }

    if (last_time_ + dt_ <= time_sec) {
      double t =
          std::min(std::max((time_sec - start_time_) / max_time_, 0.), 1.);
      last_p_t_ = p_start_ + t * (p_goal_ - p_start_);
      last_q_t_ = q_start_.slerp(t, q_goal_);
      last_twist_trans_t_ = twist_trans_start_ + t * (twist_trans_goal_ - twist_trans_start_);
      last_twist_rot_t_ = twist_rot_start_ + t * (twist_rot_goal_ - twist_rot_start_);
      last_time_ = time_sec;

      // std::cout<<" input time "<< time_sec<<" t_intepolation " <<t<<std::endl;
    }
    p_t = last_p_t_;
    q_t = last_q_t_;
    twist_trans_t = last_twist_trans_t_;
    twist_rot_t = last_twist_rot_t_;

    // std::cout<<"p_tttt "<<p_t<<std::endl;
    // std::cout<<"p_start"<<p_start_<<" p_goal "<<p_goal_<<std::endl;
  };
};
} // namespace traj_utils
#endif // DEOXYS_FRANKA_INTERFACE_INCLUDE_UTILS_TRAJ_INTERPOLATORS_POSE_TRAJ_INTERPOLATOR_H_
