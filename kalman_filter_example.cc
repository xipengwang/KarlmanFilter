/**
 * Kalman filter implementation using Eigen. Based on the following
 * introductory paper with a little bit modifications:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Xipeng Wang
 * @date: 2022.06.14
 */

#include "kalman_filter.h"

#include <Eigen/Dense>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {

  const int state_length = 3;
  const int proc_signal_length = 2;
  const int meas_signal_length = 2;

  Eigen::Matrix<double, state_length, state_length> A;
  A << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Eigen::Matrix<double, state_length, proc_signal_length> B;
  B << 1, 0, 0, 0, 1, 0;
  Eigen::Matrix<double, proc_signal_length, proc_signal_length> Q;
  Q << 1, 1, 1, 1;
  Eigen::Matrix<double, meas_signal_length, state_length> C;
  C << 1, 2, 3, 4, 5, 6;
  Eigen::Matrix<double, meas_signal_length, meas_signal_length> R;
  R << 2, 3, 4, 5;

  KalmanFilter<double, state_length, proc_signal_length, meas_signal_length> kf(A, B, Q, C, R);
  std::cout << "state length: " << kf.state_length() << std::endl;

  Eigen::Matrix<double, kf.state_length(), 1> mu;
  mu << 1, 2, 3;

  Eigen::Matrix<double, kf.state_length(), kf.state_length()> P;
  P << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  kf.Init(mu, P);

  Eigen::Matrix<double, kf.proc_signal_length(), 1> u;
  u << 1, 2;
  kf.Predict(u);

  Eigen::Matrix<double, kf.meas_signal_length(), 1> z;
  kf.Update(z);
  z << 0, 0;
  return 0;
}
