#pragma once
#include <math.h>
#include <templates/linalg.h>

#include <Eigen/Dense>
#include <cmath>

namespace Template {
template <int size>
class CostFunction {
 public:
  /**
   * @brief Compute the cost function based on expected and actual outputs
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double The cost
   */
  virtual inline double function(Vectord<size> out, Vectord<size> exp_out) = 0;
  /**
   * @brief Compute the gradient of the cost function
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return Vector<size> Vector containing the parial derivatives of the cost
   */
  virtual inline Vectord<size> derivative(Vectord<size> out,
                                          Vectord<size> exp_out) = 0;
};

template <int size>
class MeanSquareError : public CostFunction<size> {
 public:
  /**
   * @brief Calculate the mean square error
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double The mean square error
   */
  inline double function(Vectord<size> out, Vectord<size> exp_out) {
    auto errors = (out - exp_out).array();
    return (errors * errors).sum() / size;
  }
  /**
   * @brief Calculate the gradient of the mean square error
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return Vectord<size> Vector containing the partial derrivatives of the
   * mean square error
   */
  inline Vectord<size> derivative(Vectord<size> out, Vectord<size> exp_out) {
    return (2.0 / size) * (out - exp_out);
  }
};

/**
 * @brief restricts values from the closed interval [0,1] to the open interval
 * (0,1). Used in categorical cross entropy to prevent division by 0.
 *
 * @param x
 * @return double
 * @pre 0 <= x <= 1
 */
inline double clip(double x) { return fmin(1 - 1e-7, fmax(x, 1e-7)); }

template <int size>
class CategoricalCrossEntropy : public CostFunction<size> {
 public:
  /**
   * @brief Calculate the categorical cross entropy
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double The categorical cross entropy
   */
  inline double function(Vectord<size> out, Vectord<size> exp_out) {
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
  inline Vectord<size> derivative(Vectord<size> out, Vectord<size> exp_out) {
    return -exp_out.cwiseQuotient(
        out.unaryExpr([](double x) { return clip(x); }));
  }
};
}  // namespace Template
