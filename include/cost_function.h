#pragma once
#include <math.h>

#include <Eigen/Dense>
#include <cmath>

#include "Eigen/src/Core/Matrix.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline double mean_sqr_error(VectorXd out, VectorXd exp_out) {
  auto errors = (out - exp_out).array();
  return (errors * errors).sum();
}

inline VectorXd mean_sqr_error_der(VectorXd out, VectorXd exp_out) {
  return (2 / out.rows()) * (out - exp_out);
}

inline double clip(double x) { return fmax(1 - 1e-7, fmin(x, 1e-7)); }

inline double cat_cross_entropy(VectorXd out, VectorXd exp_out) {
  auto logs = out.unaryExpr([](double x) { return log(clip(x)); }).array();
  return -(logs * exp_out.array()).sum();
}

inline VectorXd cat_cross_entropy_der(VectorXd out, VectorXd exp_out) {
  return -exp_out.cwiseQuotient(
      out.unaryExpr([](double x) { return clip(x); }));
}