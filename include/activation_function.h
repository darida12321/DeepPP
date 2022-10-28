#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline VectorXd sigmoid(VectorXd x) {
  return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
}
inline VectorXd sigmoidDerivative(VectorXd x) {
  return sigmoid(x).cwiseProduct(
      sigmoid(x).unaryExpr([](double x) { return 1 - x; }));
}

inline VectorXd softmax(VectorXd x) {
  double max = x.maxCoeff();
  VectorXd shiftx = x.array() - max;
  VectorXd exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
  return exps.array() / exps.sum();
}
inline VectorXd softmaxDerivative(VectorXd x) {
  VectorXd y = softmax(x);
  VectorXd out = VectorXd::Zero(x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i) += y(i) * (1 - y(i));
  }
  return out*2;
}

inline VectorXd relu(VectorXd x) {
  return x.unaryExpr([](double x) { return fmax(x, 0); });
}
inline VectorXd reluDerivative(VectorXd x) {
  return x.unaryExpr([](double x) {
    if (x <= 0) return 0.0;
    return 1.0;
  });
}

inline VectorXd linear(VectorXd x) { return x; }
inline VectorXd linearDerivative(VectorXd x) {
  return VectorXd::Ones(x.rows(), x.cols());
}

