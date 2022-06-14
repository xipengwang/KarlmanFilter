/**
 * Kalman filter implementation using Eigen. Based on the following
 * introductory paper with a little bit modifications:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Xipeng Wang
 * @date: 2022.06.14
 */

#pragma once
#include <Eigen/Dense>

/*
** x_(k+1) = A*x_k + B*(u_k + w ~ (0, Q))
** z_k = C*(x_k + v ~ (0, R))
** x_k ~ (mu_k, P_k)
*/
template <typename scalar, int StateLength, int ProcSignalLength, int MeasSignalLength>
class KalmanFilter {
public:
  KalmanFilter() = default;

  KalmanFilter(const Eigen::Matrix<scalar, StateLength, StateLength>& A,
               const Eigen::Matrix<scalar, StateLength, ProcSignalLength>& B,
               const Eigen::Matrix<scalar, ProcSignalLength, ProcSignalLength>& Q,
               const Eigen::Matrix<scalar, MeasSignalLength, StateLength>& C,
               const Eigen::Matrix<scalar, MeasSignalLength, MeasSignalLength>& R)
      : A_{A}, B_{B}, Q_{Q}, C_{C}, R_{R} {}

  Eigen::Matrix<scalar, StateLength, StateLength>& A() { return A_; };
  Eigen::Matrix<scalar, StateLength, ProcSignalLength>& B() { return B_; };
  Eigen::Matrix<scalar, ProcSignalLength, ProcSignalLength>& Q() { return Q_; };
  Eigen::Matrix<scalar, MeasSignalLength, StateLength>& C() { return C_; };
  Eigen::Matrix<scalar, MeasSignalLength, MeasSignalLength> R() { return R_; };

  constexpr int state_length() const { return StateLength; }
  constexpr int proc_signal_length() const { return ProcSignalLength; }
  constexpr int meas_signal_length() const { return MeasSignalLength; }

  void Init(const Eigen::Matrix<scalar, StateLength, 1>& mu,
            const Eigen::Matrix<scalar, StateLength, StateLength>& P) {
    mu_ = mu;
    P_ = P;
    is_initialized = true;
  }

  void Predict(const Eigen::Matrix<scalar, ProcSignalLength, 1>& u) {
    if (!is_initialized) {
      throw std::runtime_error("Filter is not initialized!");
    }

    mu_ = A_ * mu_ + B_ * u;
    P_ = A_ * P_ * A_.transpose() + B_ * Q_ * B_.transpose();
  }

  void Update(const Eigen::Matrix<scalar, MeasSignalLength, 1>& z) {
    if (!is_initialized) {
      throw std::runtime_error("Filter is not initialized!");
    }

    const auto K = P_ * C_.transpose() * (C_ * P_ * C_.transpose() + R_).inverse();
    mu_ += K * (z - C_ * mu_);
    P_ = (I_ - K * C_) * P_;
  }

private:
  bool is_initialized{false};
  const Eigen::Matrix<scalar, StateLength, StateLength> A_{
      Eigen::Matrix<scalar, StateLength, StateLength>::Identity()};
  const Eigen::Matrix<scalar, StateLength, ProcSignalLength> B_{
      Eigen::Matrix<scalar, StateLength, StateLength>::Identity()};
  const Eigen::Matrix<scalar, ProcSignalLength, ProcSignalLength> Q_{
      Eigen::Matrix<scalar, StateLength, StateLength>::Identity()};
  const Eigen::Matrix<scalar, MeasSignalLength, StateLength> C_{
      Eigen::Matrix<scalar, StateLength, StateLength>::Identity()};
  const Eigen::Matrix<scalar, MeasSignalLength, MeasSignalLength> R_{
      Eigen::Matrix<scalar, StateLength, StateLength>::Identity()};
  const Eigen::Matrix<scalar, StateLength, StateLength> I_{
      Eigen::Matrix<scalar, StateLength, StateLength>::Identity()};
  Eigen::Matrix<scalar, StateLength, 1> mu_{Eigen::Matrix<scalar, StateLength, 1>::Zero()};
  Eigen::Matrix<scalar, StateLength, StateLength> P_{
      Eigen::Matrix<scalar, StateLength, StateLength>::Identity()};
};
