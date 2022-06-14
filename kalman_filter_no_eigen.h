/**
 * Kalman filter implementation using algebra. Based on the following
 * introductory paper with a little bit modifications:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Xipeng Wang
 * @date: 2022.06.14
 */

#pragma once
#include "matrix.h"

/*
** x_(k+1) = A*x_k + B*(u_k + w ~ (0, Q))
** z_k = C*(x_k + v ~ (0, R))
** x_k ~ (mu_k, P_k)
*/
template <typename scalar, int StateLength, int ProcSignalLength, int MeasSignalLength>
class KalmanFilter {
public:
  KalmanFilter() = default;

  KalmanFilter(const algebra::Matrix<scalar, StateLength, StateLength>& A,
               const algebra::Matrix<scalar, StateLength, ProcSignalLength>& B,
               const algebra::Matrix<scalar, ProcSignalLength, ProcSignalLength>& Q,
               const algebra::Matrix<scalar, MeasSignalLength, StateLength>& C,
               const algebra::Matrix<scalar, MeasSignalLength, MeasSignalLength>& R)
      : A_{A}, B_{B}, Q_{Q}, C_{C}, R_{R} {}

  algebra::Matrix<scalar, StateLength, StateLength>& A() { return A_; };
  algebra::Matrix<scalar, StateLength, ProcSignalLength>& B() { return B_; };
  algebra::Matrix<scalar, ProcSignalLength, ProcSignalLength>& Q() { return Q_; };
  algebra::Matrix<scalar, MeasSignalLength, StateLength>& C() { return C_; };
  algebra::Matrix<scalar, MeasSignalLength, MeasSignalLength> R() { return R_; };

  constexpr int state_length() const { return StateLength; }
  constexpr int proc_signal_length() const { return ProcSignalLength; }
  constexpr int meas_signal_length() const { return MeasSignalLength; }

  void Init(const algebra::Matrix<scalar, StateLength, 1>& mu,
            const algebra::Matrix<scalar, StateLength, StateLength>& P) {
    mu_ = mu;
    P_ = P;
    is_initialized = true;
  }

  void Predict(const algebra::Matrix<scalar, ProcSignalLength, 1>& u) {
    if (!is_initialized) {
      throw std::runtime_error("Filter is not initialized!");
    }

    mu_ = A_ * mu_ + B_ * u;
    P_ = A_ * P_ * A_.transpose() + B_ * Q_ * B_.transpose();
  }

  void Update(const algebra::Matrix<scalar, MeasSignalLength, 1>& z) {
    if (!is_initialized) {
      throw std::runtime_error("Filter is not initialized!");
    }

    const auto K = P_ * C_.transpose() * (C_ * P_ * C_.transpose() + R_).inverse();
    mu_ += K * (z - C_ * mu_);
    P_ = (I_ - K * C_) * P_;
  }

private:
  bool is_initialized{false};
  const algebra::Matrix<scalar, StateLength, StateLength> A_{
      algebra::Matrix<scalar, StateLength, StateLength>::Identity()};
  const algebra::Matrix<scalar, StateLength, ProcSignalLength> B_{
      algebra::Matrix<scalar, StateLength, StateLength>::Identity()};
  const algebra::Matrix<scalar, ProcSignalLength, ProcSignalLength> Q_{
      algebra::Matrix<scalar, StateLength, StateLength>::Identity()};
  const algebra::Matrix<scalar, MeasSignalLength, StateLength> C_{
      algebra::Matrix<scalar, StateLength, StateLength>::Identity()};
  const algebra::Matrix<scalar, MeasSignalLength, MeasSignalLength> R_{
      algebra::Matrix<scalar, StateLength, StateLength>::Identity()};
  const algebra::Matrix<scalar, StateLength, StateLength> I_{
      algebra::Matrix<scalar, StateLength, StateLength>::Identity()};
  algebra::Matrix<scalar, StateLength, 1> mu_{algebra::Matrix<scalar, StateLength, 1>::Zero()};
  algebra::Matrix<scalar, StateLength, StateLength> P_{
      algebra::Matrix<scalar, StateLength, StateLength>::Identity()};
};
