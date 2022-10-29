#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * @brief The sigmoid activation function (applied componentwise)
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd sigmoid(VectorXd x) {
  return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
}

/**
 * @brief The gradient of the sigmoid activation function
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd sigmoid_derivative(VectorXd x) {
  return sigmoid(x).cwiseProduct(
      sigmoid(x).unaryExpr([](double x) { return 1 - x; }));
}

/**
 * @brief The softmax activation function
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd softmax(VectorXd x) {
  double max = x.maxCoeff();
  VectorXd shiftx = x.array() - max;
  VectorXd exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
  return exps.array() / exps.sum();
}

/**
 * @brief The gradient of the softmax activation function
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd softmax_derivative(VectorXd x) {
  MatrixXd jacobian(x.rows(), x.rows());
  for (int i = 0; i < x.rows(); i++) {
    for (int j = 0; j < x.rows(); j++) {
      if (i == j) {
        jacobian(i, j) = x(i) * (1 - x(i));
      } else {
        jacobian(i, j) = -x(i) * x(j);
      }
    }
  }
  return jacobian * sigmoid(x);
}

/**
 * @brief The RELU activation function (applied componentwise)
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd relu(VectorXd x) {
  return x.unaryExpr([](double x) { return fmax(x, 0); });
}

/**
 * @brief The gradient of the RELU activation function
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd relu_derivative(VectorXd x) {
  return x.unaryExpr([](double x) {
    if (x <= 0) return 0.0;
    return 1.0;
  });
}

/**
 * @brief The indentity function on vectors
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd linear(VectorXd x) { return x; }

/**
 * @brief The gradient of the identity function
 *
 * @param x The input vector
 * @return VectorXd
 */
inline VectorXd linear_derivative(VectorXd x) {
  return VectorXd::Ones(x.rows(), x.cols());
}
