#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline VectorXd sigmoid(VectorXd x) {
  return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
}
inline VectorXd sigmoid_derivative(VectorXd x) {
  return sigmoid(x).cwiseProduct(
      sigmoid(x).unaryExpr([](double x) { return 1 - x; }));
}

inline VectorXd softmax(VectorXd x) {
  double max = x.maxCoeff();
  VectorXd shiftx = x.array() - max;
  VectorXd exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
  return exps.array() / exps.sum();
}
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

inline VectorXd relu(VectorXd x) {
  return x.unaryExpr([](double x) { return fmax(x, 0); });
}
inline VectorXd relu_derivative(VectorXd x) {
  return x.unaryExpr([](double x) {
    if (x <= 0) return 0.0;
    return 1.0;
  });
}

inline VectorXd linear(VectorXd x) { return x; }
inline VectorXd linear_derivative(VectorXd x) {
  return VectorXd::Ones(x.rows(), x.cols());
}

// inline double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
// inline double sigmoid_derivative(double x) { return sigmoid(x) * (1 -
// sigmoid(x)); }
//
// inline double relu(double x) { return fmax(x, 0); }
// inline double relu_derivative(double x) {
//   if (x <= 0)
//     return 0;
//   return 1;
// }
//
// inline double linear(double x) { return x; }
// inline double linear_derivative(double x) { return 1; }
