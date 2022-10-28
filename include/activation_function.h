#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline VectorXd sigmoid(VectorXd x) {
  return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
}
inline MatrixXd sigmoidDerivative(VectorXd x) {
  MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
  VectorXd diag = sigmoid(x).cwiseProduct(
      sigmoid(x).unaryExpr([](double x) { return 1 - x; }));

  for (int i = 0; i < x.rows(); i++) {
    out(i, i) = diag(i);
  }
  return out;

}

inline VectorXd softmax(VectorXd x) {
  double max = x.maxCoeff();
  VectorXd shiftx = x.array() - max;
  VectorXd exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
  return exps.array() / exps.sum();
}
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

inline VectorXd relu(VectorXd x) {
  return x.unaryExpr([](double x) { return fmax(x, 0); });
}
#include <iostream>
inline MatrixXd reluDerivative(VectorXd x) {
  MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i, i) = x(i) < 0 ? 0 : 1;
  }
  return out;
}

inline VectorXd linear(VectorXd x) { return x; }
inline MatrixXd linearDerivative(VectorXd x) {
  MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
  for (int i = 0; i < x.rows(); i++) {
    out(i, i) = 1;
  }
  return out;
}

