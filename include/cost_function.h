#pragma once
#include <math.h>

#include <Eigen/Dense>
#include <cmath>

#include "Eigen/src/Core/Matrix.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline double clip(double x) { return fmin(1 - 1e-7, fmax(x, 1e-7)); }

class CostFunction {
  public:
    virtual inline double function(VectorXd out, VectorXd exp_out) = 0;
    virtual inline VectorXd derivative(VectorXd out, VectorXd exp_out) = 0;
};

class MeanSquareError : public CostFunction {
  public:
    inline double function(VectorXd out, VectorXd exp_out) {
      auto errors = (out - exp_out).array();
      return (errors * errors).sum() / out.rows();
    }
    inline VectorXd derivative(VectorXd out, VectorXd exp_out) {
      return (2.0 / out.rows()) * (out - exp_out);
    }
};

class CategoricalCrossEntropy : public CostFunction {
  public:
    inline double function(VectorXd out, VectorXd exp_out) {
      auto logs = out.unaryExpr([](double x) { return log(clip(x)); }).array();
      return -(logs * exp_out.array()).sum();
    }
    inline VectorXd derivative(VectorXd out, VectorXd exp_out) {
      return -exp_out.cwiseQuotient(
          out.unaryExpr([](double x) { return clip(x); }));
    }
};

extern MeanSquareError mean_sqr_error;
extern CategoricalCrossEntropy cat_cross_entropy;