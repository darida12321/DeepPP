#include <Eigen/Dense>
#include "Eigen/src/Core/Matrix.h"

// TODO add rest

namespace DeepPP {
  template <size_t N>
  struct Linear {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    /**
     * @brief The indentity function on vectors
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline Vec activation(const Vec& x) { return x; }

    /**
     * @brief The gradient of the identity function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline Mat activation_der(const Vec& x) {
      return Eigen::Matrix<double, N, N>::Identity();
    }
  };

  template <size_t N>
  struct Sigmoid {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    /**
     * @brief The sigmoid activation function (applied componentwise)
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline Vec activation(const Vec& x) {
      return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
    }

    /**
     * @brief The gradient of the sigmoid activation function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline Mat activation_der(const Vec& x) {
      Mat out = Mat::Zero();
      Vec diag = activation(x).cwiseProduct(
          activation(x).unaryExpr([](double x) { return 1 - x; }));

      for (size_t i = 0; i < N; i++) {
        out(i, i) = diag(i);
      }
      return out;
    }
  };

  template<size_t N>
  struct Softmax {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    /**
     * @brief The softmax activation function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline Vec activation(const Vec& x) {
      double max = x.maxCoeff();
      Vec shiftx = x.array() - max;
      Vec exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
      return exps.array() / exps.sum();
    }

    /**
     * @brief The gradient of the softmax activation function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline Mat activation_der(const Vec& x) {
      Vec y = activation(x);
      Mat out = Mat::Zero();
      for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
          if (i == j) {
            out(i, j) += y(i) * (1 - y(i));
          } else {
            out(i, j) += - y(i) * y(j);
          }
        }
      }
      return out;
    }
  };

  template <size_t N>
  struct Relu {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    /**
     * @brief The RELU activation function (applied componentwise)
     *
     * @param x The input vector
     * @return Vector<double, N>
     */
    inline Vec activation(const Vec& x) {
      return x.unaryExpr([](double x) { return std::max(0.0, x); });
    }

    /**
     * @brief The gradient of the RELU activation function
     *
     * @param x The input vector
     * @return Matrix<double, N, N>
     */
    inline Mat activation_der(const Vec& x) {
      Mat out = Mat::Zero();
      for (size_t i = 0; i < N; i++) {
        out(i, i) = x(i) < 0 ? 0 : 1;
      }
      return out;
    }
  };

}



