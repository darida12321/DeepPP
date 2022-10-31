#pragma once
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