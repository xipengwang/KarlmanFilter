/**
 * A simple matrix class to replace using of Eigen.
 * @author: Xipeng Wang
 * @date: 2022.06.14
 */

#pragma once
#include "assert.h"
#include <algorithm>
#include <array>
#include <functional>
#include <ostream>
#include <stdexcept>
#include <type_traits>

namespace algebra {

// Row major matrix.
template <typename scalar, int nrows, int ncols>
class Matrix {
public:
  using MatrixType = Matrix<scalar, nrows, ncols>;
  using TransposeMatrixType = Matrix<scalar, ncols, nrows>;
  using DataType = std::array<scalar, nrows * ncols>;
  using ElemType = scalar;

  Matrix() {
    static_assert(std::is_same_v<ElemType, double> || std::is_same_v<ElemType, float>,
                  "Matrix only supports double and float type!");
  };

  explicit Matrix(const std::initializer_list<ElemType>& data_list) {
    static_assert(std::is_same_v<ElemType, double> || std::is_same_v<ElemType, float>,
                  "Matrix only supports double and float type!");
    assert(data_list.size() == nrows * ncols);
    std::copy_n(data_list.begin(), nrows * ncols, data_.begin());
  }

  constexpr int num_of_rows() const { return nrows; }
  constexpr int num_of_ncols() const { return ncols; }

  DataType& data() { return data_; }
  const DataType& data() const { return data_; }

  ElemType& operator()(int row, int col) {
    Validate(row, col);
    return data_[row * ncols + col];
  }
  const ElemType& operator()(int row, int col) const {
    Validate(row, col);
    return data_[row * ncols + col];
  }

  TransposeMatrixType transpose() const {
    TransposeMatrixType ret;
    for (int row = 0; row < nrows; ++row) {
      for (int col = 0; col < ncols; ++col) {
        ret(col, row) = (*this)(row, col);
      }
    }
    return ret;
  }

  MatrixType operator+(const MatrixType& other) const {
    MatrixType ret;
    std::transform(this->data().cbegin(), this->data().cend(), other.data().cbegin(),
                   ret.data().begin(), std::plus<ElemType>{});
    return ret;
  }

  MatrixType& operator+=(const MatrixType& other) {
    std::transform(this->data().cbegin(), this->data().cend(), other.data().cbegin(),
                   this->data().begin(), std::plus<ElemType>{});
    return *this;
  }

  MatrixType operator-(const MatrixType& other) const {
    MatrixType ret;
    std::transform(this->data().cbegin(), this->data().cend(), other.data().cbegin(),
                   ret.data().begin(), std::minus<ElemType>{});
    return ret;
  }

  MatrixType& operator-=(const MatrixType& other) {
    std::transform(this->data().cbegin(), this->data().cend(), other.data().cbegin(),
                   this->data().begin(), std::minus<ElemType>{});
    return *this;
  }

  template <int new_ncols>
  Matrix<ElemType, nrows, new_ncols>
  operator*(const Matrix<ElemType, ncols, new_ncols>& other) const {
    Matrix<ElemType, nrows, new_ncols> ret;
    for (int row = 0; row < nrows; ++row) {
      for (int col = 0; col < new_ncols; ++col) {
        ElemType val{0.0};
        for (int k = 0; k < ncols; ++k) {
          val += (*this)(row, k) * other(k, col);
        }
        ret(row, col) = val;
      }
    }
    return ret;
  }

  MatrixType operator*(const ElemType other) const {
    MatrixType ret;
    std::transform(this->data_.begin(), this->data_.end(), ret.data().begin(),
                   [other](const ElemType elem) { return elem * other; });
    return ret;
  }

  static MatrixType Identity() {
    static_assert(nrows == ncols, "Not a square matrix!");
    MatrixType ret;
    for (int i = 0; i < nrows; ++i) {
      ret(i, i) = 1;
    }
    return ret;
  }

  static MatrixType Zero() {
    MatrixType ret;
    ret.data().fill(0.0);
    return ret;
  }

  bool operator==(const MatrixType& other) const {
    auto other_iter = other.data().cbegin();
    return std::all_of(data_.cbegin(), data_.cend(), [&other_iter](const ElemType elem) {
      const ElemType other_elem = *other_iter++;
      constexpr ElemType EPSILON = 1e-8;
      return fabs(elem - other_elem) < EPSILON;
    });
  }

  typename DataType::iterator row_begin(int row) {
    assert(0 <= row && row < nrows);
    return data_.begin() + row * ncols;
  }

  typename DataType::iterator row_end(int row) {
    assert(0 <= row && row < nrows);
    return data_.begin() + (row + 1) * ncols;
  }

  MatrixType& SwapRows(int row_i, int row_j) {
    assert(0 <= row_i && row_i < nrows);
    assert(0 <= row_j && row_j < nrows);

    std::swap_ranges(row_begin(row_i), row_end(row_i), row_begin(row_j));
    return *this;
  }

  MatrixType& ScaleRow(int row, ElemType scale_value) {
    assert(0 <= row && row < nrows);

    std::transform(row_begin(row), row_end(row), row_begin(row),
                   [scale_value](const ElemType elem) { return scale_value * elem; });
    return *this;
  }

  MatrixType& ScaleRowAndAddToOtherRow(int row, ElemType scale_value, int other_row) {
    assert(0 <= row && row < nrows);
    assert(0 <= other_row && other_row < nrows);

    std::transform(row_begin(row), row_end(row), row_begin(other_row), row_begin(other_row),
                   [scale_value](const ElemType elem1, const ElemType elem2) {
                     return scale_value * elem1 + elem2;
                   });
    return *this;
  }

  MatrixType inverse() const {
    auto I = MatrixType::Identity();
    auto M = *this;
    constexpr ElemType EPSILON = 1e-8;
    for (int row = 0; row < nrows; ++row) {
      ElemType pivot = M(row, row);
      if (fabs(pivot) < EPSILON) {
        bool is_singular = true;
        for (int new_row = row + 1; new_row < nrows; ++new_row) {
          if (fabs(M(new_row, new_row)) > EPSILON) {
            M.SwapRows(row, new_row);
            I.SwapRows(row, new_row);
            pivot = M(row, row);
            is_singular = false;
            break;
          };
        }
        if (is_singular) {
          throw std::runtime_error("Can't perform inversion on a singular matrix!");
        }
      }

      ElemType factor = 1.0 / pivot;
      M.ScaleRow(row, factor);
      I.ScaleRow(row, factor);
      for (int new_row = row + 1; new_row < nrows; ++new_row) {
        ElemType first_elem = M(new_row, row);
        if (fabs(first_elem) < EPSILON) {
          continue;
        }
        M.ScaleRowAndAddToOtherRow(row, -first_elem, new_row);
        I.ScaleRowAndAddToOtherRow(row, -first_elem, new_row);
      }
    }
    for (int row = nrows - 1; row > 0; --row) {
      ElemType pivot = M(row, row);
      for (int new_row = row - 1; new_row > 0; --new_row) {
        ElemType first_elem = M(new_row, row);
        if (fabs(first_elem) < EPSILON) {
          continue;
        }
        M.ScaleRowAndAddToOtherRow(row, -first_elem, new_row);
        I.ScaleRowAndAddToOtherRow(row, -first_elem, new_row);
      }
    }
    return I;
  }

private:
  void Validate(int row, int col) const {
    assert(0 <= row && row < nrows);
    assert(0 <= col && col < ncols);
  }
  DataType data_;
};

template <typename scalar, int nrows, int ncols>
std::ostream& operator<<(std::ostream& os, const Matrix<scalar, nrows, ncols>& matrix) {
  for (int row = 0; row < nrows; ++row) {
    for (int col = 0; col < ncols; ++col) {
      os << matrix(row, col) << ",";
    }
    os << "\n";
  }
  return os;
}

} // namespace algebra
