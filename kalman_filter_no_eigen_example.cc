/**
 * Kalman filter implementation using algebra. Based on the following
 * introductory paper with a little bit modifications:
 *
 *     http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
 *
 * @author: Xipeng Wang
 * @date: 2022.06.14
 */

#include "kalman_filter_no_eigen.h"
#include "matrix.h"

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {

  const int state_length = 3;
  const int proc_signal_length = 2;
  const int meas_signal_length = 2;

  algebra::Matrix<double, state_length, state_length> A{1, 0, 0, 0, 1, 0, 0, 0, 1};
  algebra::Matrix<double, state_length, proc_signal_length> B{1, 0, 0, 0, 1, 0};
  algebra::Matrix<double, proc_signal_length, proc_signal_length> Q{1, 1, 1, 1};
  algebra::Matrix<double, meas_signal_length, state_length> C{1, 2, 3, 4, 5, 6};
  algebra::Matrix<double, meas_signal_length, meas_signal_length> R{2, 3, 4, 5};

  KalmanFilter<double, state_length, proc_signal_length, meas_signal_length> kf(A, B, Q, C, R);
  std::cout << "state length: " << kf.state_length() << std::endl;

  algebra::Matrix<double, kf.state_length(), 1> mu{1, 2, 3};

  algebra::Matrix<double, kf.state_length(), kf.state_length()> P{1, 0, 0, 0, 1, 0, 0, 0, 1};

  kf.Init(mu, P);

  algebra::Matrix<double, kf.proc_signal_length(), 1> u{1, 2};
  kf.Predict(u);

  algebra::Matrix<double, kf.meas_signal_length(), 1> z{0, 0};
  kf.Update(z);

  algebra::Matrix<double, 3, 2> data{0.0, 1.0, 2.0, 2.0, 3.0, 3.0};
  std::cout << data << "\n";
  data(0, 0) = 1.0;
  std::cout << data << "\n";

  const auto data_T = data.transpose();
  const algebra::Matrix<double, 2, 3> expected_data_T{1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
  std::cout << (data_T == expected_data_T ? "true" : "false") << "\n";

  const auto data_plus = data + data;
  const algebra::Matrix<double, 3, 2> expected_data_plus{2.0, 2.0, 4.0, 4.0, 6.0, 6.0};
  std::cout << (data_plus == expected_data_plus ? "true" : "false") << "\n";

  {
    auto data_plus = data;
    data_plus += data;
    const algebra::Matrix<double, 3, 2> expected_data_plus{2.0, 2.0, 4.0, 4.0, 6.0, 6.0};
    std::cout << (data_plus == expected_data_plus ? "true" : "false") << "\n";
  }

  const auto data_minus = data - data;
  const algebra::Matrix<double, 3, 2> expected_data_minus{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::cout << (data_minus == expected_data_minus ? "true" : "false") << "\n";

  {
    auto data_minus = data;
    data_minus -= data;
    const algebra::Matrix<double, 3, 2> expected_data_plus{2.0, 2.0, 4.0, 4.0, 6.0, 6.0};
    std::cout << (data_minus == expected_data_minus ? "true" : "false") << "\n";
  }

  const auto data_multi = data * 3.0;
  const algebra::Matrix<double, 3, 2> expected_data_multi{3.0, 3.0, 6.0, 6.0, 9.0, 9.0};
  std::cout << (data_multi == expected_data_multi ? "true" : "false") << "\n";

  {
    auto new_data = data;
    const auto& data_swap_row = new_data.SwapRows(0, 2).SwapRows(0, 1);
    const algebra::Matrix<double, 3, 2> expected_data_swap_row{2.0, 2.0, 3.0, 3.0, 1.0, 1.0};
    std::cout << (data_swap_row == expected_data_swap_row ? "true" : "false") << "\n";
  }

  {
    auto new_data = data;
    const auto data_scale_row = new_data.ScaleRow(0, 1.8);
    const algebra::Matrix<double, 3, 2> expected_data_scale_row{1.8, 1.8, 2.0, 2.0, 3.0, 3.0};
    std::cout << (data_scale_row == expected_data_scale_row ? "true" : "false") << "\n";
  }

  {
    auto new_data = data;
    const auto data_scale_and_add_row = new_data.ScaleRowAndAddToOtherRow(0, 1.8, 1);
    const algebra::Matrix<double, 3, 2> expected_data_scale_and_add_row{1., 1., 3.8, 3.8, 3.0, 3.0};
    std::cout << (data_scale_and_add_row == expected_data_scale_and_add_row ? "true" : "false")
              << "\n";
  }

  if (true) {
    algebra::Matrix<double, 3, 3> data{1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 3.0};
    const auto data_inverse = data.inverse();
    // std::cout << data << "\n";
    // std::cout << data_inverse << "\n";
    std::cout << (data * data_inverse == algebra::Matrix<double, 3, 3>::Identity() ? "true"
                                                                                   : "false")
              << "\n";
  }
  return 0;
}
