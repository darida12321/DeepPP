#pragma once
#include <Eigen/Dense>

using Eigen::Matrix;

template <int size>
struct Linalg {
  typedef Matrix<double, size, size> Matrixd;
  typedef Matrix<double, size, 1> Vectord;
};

/**
 * @brief type alias for a square matrix of doubles
 *
 * @tparam size
 */
template <int size>
using Matrixd = typename Linalg<size>::Matrixd;

/**
 * @brief typa alias for a column vector of doubles
 *
 * @tparam size
 */
template <int size>
using Vectord = typename Linalg<size>::Vectord;