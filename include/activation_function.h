#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class ActivationFunction {
  public: 
    virtual inline VectorXd function(VectorXd x) = 0;
    virtual inline VectorXd derivative(VectorXd x) = 0;
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
    inline VectorXd derivative(VectorXd x) {
      return function(x).cwiseProduct(
          function(x).unaryExpr([](double x) { return 1 - x; }));
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
    inline VectorXd derivative(VectorXd x) {
      MatrixXd jacobian(x.rows(), x.rows());
      Sigmoid sigmoid;
      for (int i = 0; i < x.rows(); i++) {
        for (int j = 0; j < x.rows(); j++) {
          if (i == j) {
            jacobian(i, j) = x(i) * (1 - x(i));
          } else {
            jacobian(i, j) = -x(i) * x(j);
          }
        }
      }
      return jacobian * sigmoid.function(x);
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
    inline VectorXd derivative(VectorXd x) {
      return x.unaryExpr([](double x) {
        if (x <= 0) return 0.0;
        return 1.0;
      });
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
    inline VectorXd derivative(VectorXd x) {
      return VectorXd::Ones(x.rows(), x.cols());
    }
};

extern Sigmoid sigmoid;
extern Softmax softmax;
extern Relu relu;
extern Linear linear;