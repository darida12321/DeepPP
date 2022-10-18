#include <Eigen/Dense>
#include <layer.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// constructor
Layer::Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func,
             std::function<double(double)> act_func_der)
    : weights_(m), bias_(b), act_func_(act_func), act_func_der_(act_func_der) {}

VectorXd Layer::forwardProp(VectorXd in) {
  VectorXd newV = weights_ * in + bias_;
  act_derivatives_ = newV.unaryExpr(act_func_der_);
  return newV.unaryExpr(act_func_);
}

VectorXd Layer::backProp(VectorXd err) {
  // TODO: Calculate and store partial derivatives with respect to weights and
  // biases
  VectorXd propagated =
      weights_.transpose() * err.cwiseProduct(act_derivatives_);
  return propagated;
}
