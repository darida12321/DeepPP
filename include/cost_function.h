#pragma once
#include <math.h>

#include <Eigen/Dense>
#include <cmath>

#include "Eigen/src/Core/Matrix.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * @brief restricts values from the closed interval [0,1] to the open interval
 * (0,1). Used in categorical cross entropy to prevent division by 0.
 *
 * @param x
 * @return double
 * @pre 0 <= x <= 1
 */
inline double clip(double x) { return fmin(1 - 1e-7, fmax(x, 1e-7)); }

class CostFunction {
 public:
  /**
   * @brief Compute the cost function based on expected and actual outputs
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double The cost
   */
  virtual inline double function(const VectorXd& out, const VectorXd& exp_out) = 0;
  /**
   * @brief Compute the gradient of the cost function
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return VectorXd Vector containing the parial derivatives of the cost
   */
  virtual inline VectorXd derivative(const VectorXd& out, const VectorXd& exp_out) = 0;
};

class MeanSquareError : public CostFunction {
 public:
  /**
   * @brief Calculate the mean square error
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double The mean square error
   */
  inline double function(const VectorXd& out, const VectorXd& exp_out) {
    auto errors = (out - exp_out).array();
    return (errors * errors).sum() / out.rows();
  }
  /**
   * @brief Calculate the gradient of the mean square error
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return VectorXd Vector containing the partial derrivatives of the
   * mean square error
   */
  inline VectorXd derivative(const VectorXd& out, const VectorXd& exp_out) {
    return (2.0 / out.rows()) * (out - exp_out);
  }
};

class CategoricalCrossEntropy : public CostFunction {
 public:
  /**
   * @brief Calculate the categorical cross entropy
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double The categorical cross entropy
   */
  inline double function(const VectorXd& out, const VectorXd& exp_out) {
    auto logs = out.unaryExpr([](double x) { return log(clip(x)); }).array();
    return -(logs * exp_out.array()).sum();
  }
  /**
   * @brief Calculate the gradient of the categorical cross entropy
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return Vectord<size> Vector containing the partial derivatives of the
   * categorical cross entropy
   */
  inline VectorXd derivative(const VectorXd& out, const VectorXd& exp_out) {
    return -exp_out.cwiseQuotient(
        out.unaryExpr([](double x) { return clip(x); }));
  }
};

extern MeanSquareError mean_sqr_error;
extern CategoricalCrossEntropy cat_cross_entropy;