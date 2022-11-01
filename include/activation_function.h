#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class ActivationFunction {
  public: 
    virtual inline VectorXd function(VectorXd x) = 0;
    virtual inline MatrixXd derivative(VectorXd x) = 0;
};

class Sigmoid : public ActivationFunction {
  public:
    /**
     * @brief The sigmoid activation function (applied componentwise)
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline VectorXd function(VectorXd x) {
      return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
    }

    /**
     * @brief The gradient of the sigmoid activation function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline MatrixXd derivative(VectorXd x) {
      MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
      VectorXd diag = function(x).cwiseProduct(
          function(x).unaryExpr([](double x) { return 1 - x; }));

      for (int i = 0; i < x.rows(); i++) {
        out(i, i) = diag(i);
      }
      return out;
    }
};

class Softmax : public ActivationFunction {
  public: 
    /**
     * @brief The softmax activation function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline VectorXd function(VectorXd x) {
      double max = x.maxCoeff();
      VectorXd shiftx = x.array() - max;
      VectorXd exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
      return exps.array() / exps.sum();
    }

    /**
     * @brief The gradient of the softmax activation function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline MatrixXd derivative(VectorXd x) {
      VectorXd y = function(x);
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
};

class Relu : public ActivationFunction {
  public: 
    /**
     * @brief The RELU activation function (applied componentwise)
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline VectorXd function(VectorXd x) {
      return x.unaryExpr([](double x) { return fmax(x, 0); });
    }

    /**
     * @brief The gradient of the RELU activation function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline MatrixXd derivative(VectorXd x) {
      MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
      for (int i = 0; i < x.rows(); i++) {
        out(i, i) = x(i) < 0 ? 0 : 1;
      }
      return out;
    }
};

class Linear : public ActivationFunction {
  public: 
    /**
     * @brief The indentity function on vectors
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline VectorXd function(VectorXd x) { return x; }

    /**
     * @brief The gradient of the identity function
     *
     * @param x The input vector
     * @return VectorXd
     */
    inline MatrixXd derivative(VectorXd x) {
      MatrixXd out = MatrixXd::Zero(x.rows(), x.rows());
      for (int i = 0; i < x.rows(); i++) {
        out(i, i) = 1;
      }
      return out;
    }
};

extern Sigmoid sigmoid;
extern Softmax softmax;
extern Relu relu;
extern Linear linear;
