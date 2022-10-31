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
inline MatrixXd sigmoidDerivative(VectorXd x) {
  MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
  VectorXd diag = sigmoid(x).cwiseProduct(
      sigmoid(x).unaryExpr([](double x) { return 1 - x; }));

  for (int i = 0; i < x.rows(); i++) {
    out(i, i) = diag(i);
  }
  return out;

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
inline MatrixXd softmaxDerivative(VectorXd x) {
  VectorXd y = softmax(x);
  MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
  for (int i = 0; i < x.rows(); i++) {
    for (int j = 0; j < x.rows(); j++) {
      if (i == j) {
        out(i, j) += y(i) * (1 - y(i));
      } else {
        out(i, j) += - y(i) * y(j);
      }
    }
  }
  return out;
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
inline MatrixXd reluDerivative(VectorXd x) {
  MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i, i) = x(i) < 0 ? 0 : 1;
  }
  return out;
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
inline MatrixXd linearDerivative(VectorXd x) {
  MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i, i) = 1;
  }
  return out;
}
