#pragma once
#include <templates/linalg.h>

#include <Eigen/Dense>
#include <cmath>

namespace Template {
template <int size>
class ActivationFunction {
  /**
   * @brief Compute the activation function
   *
   * @param x
   * @return Vectord<size>
   */
  virtual inline Vectord<size> function(Vectord<size> x) = 0;
  /**
   * @brief Compute the partial derivatives of the sctivation function
   *
   * @param x
   * @return Matrixd<size>
   */
  virtual inline Matrixd<size> derivative(Vectord<size> x) = 0;
};

template <int size>
class Sigmoid : public ActivationFunction<size> {
 public:
  /**
   * @brief The sigmoid activation function (applied componentwise)
   *
   * @param x The input vector
   * @return Vector with the same size as x
   */
  inline Vectord<size> function(Vectord<size> x) {
    return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
  }

  /**
   * @brief The gradient of the sigmoid activation function
   *
   * @param x The input vector
   * @return Matrix with dimensions n * n, where n is the size of x
   */
  inline Matrixd<size> derivative(Vectord<size> x) {
    Matrixd<size> out = Matrixd<size>::Zero();
    Vectord<size> diag = function(x).cwiseProduct(
        function(x).unaryExpr([](double x) { return 1 - x; }));

    for (int i = 0; i < x.rows(); i++) {
      out(i, i) = diag(i);
    }
    return out;
  }
};

template <int size>
class Softmax : public ActivationFunction<size> {
 public:
  /**
   * @brief The softmax activation function
   *
   * @param x The input vector
   * @return Vector with the same size as x
   */
  inline Vectord<size> function(Vectord<size> x) {
    double max = x.maxCoeff();
    Vectord<size> shiftx = x.array() - max;
    Vectord<size> exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
    return exps.array() / exps.sum();
  }

  /**
   * @brief The gradient of the softmax activation function
   *
   * @param x The input vector
   * @return Matrix of dimensions n * n, where n is the size of x
   */
  inline Matrixd<size> derivative(Vectord<size> x) {
    Vectord<size> y = function(x);
    Matrixd<size> out = Matrixd<size>::Zero();
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < x.rows(); j++) {
        if (i == j) {
          out(i, j) += y(i) * (1 - y(i));
        } else {
          out(i, j) += -y(i) * y(j);
        }
      }
    }
    return out;
  }
};

template <int size>
class Relu : public ActivationFunction<size> {
 public:
  /**
   * @brief The relu activation function
   *
   * @param x The input vector
   * @return Vector with the same size as x
   */
  inline Vectord<size> function(Vectord<size> x) {
    return x.unaryExpr([](double x) { return std::max(0.0, x); });
  }

  /**
   * @brief The gradient of the relu activation function
   *
   * @param x The input vector
   * @return Matrix of dimensions n * n, where n is the size of x
   */
  inline Matrixd<size> derivative(Vectord<size> x) {
    Matrixd<size> out = Matrixd<size>::Zero();
    for (int i = 0; i < x.rows(); i++) {
      out(i, i) = x(i) > 0 ? 1 : 0;
    }
    return out;
  }
};

template <int size>
class Linear : public ActivationFunction<size> {
 public:
  /**
   * @brief The linear activation function
   *
   * @param x The input vector
   * @return Vector with the same size as x
   */
  inline Vectord<size> function(Vectord<size> x) { return x; }

  /**
   * @brief The gradient of the linear activation function
   *
   * @param x The input vector
   * @return Matrix of dimensions n * n, where n is the size of x
   */
  inline Matrixd<size> derivative(Vectord<size> x) {
    return Matrixd<size>::Identity();
  }
};
}  // namespace Template
