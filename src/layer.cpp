#include <Eigen/Dense>
#include <layer.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// constructor
Layer::Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func,
             std::function<double(double)> act_func_der)
    : weights_(m), bias_(b), act_func_(act_func), act_func_der_(act_func_der) {}

VectorXd Layer::forwardProp(VectorXd in) {
  last_input_ = in; // record input for use in back propagation
  VectorXd newV = weights_ * in + bias_;

  // calculate derivatives of activation function for use in back propagation
  act_derivatives_ = newV.unaryExpr(act_func_der_);

  return newV.unaryExpr(act_func_);
}

// Precondition: err = dC/da {where C = cost, a = this layer's activation}
VectorXd Layer::backProp(VectorXd err, double step) {
  // calculate dC/dz where z is the activation before applying act_func
  VectorXd tmp = err.cwiseProduct(act_derivatives_);

  // calculate dC/da for previous layer
  VectorXd propagated = weights_.transpose() * tmp;

  // adjust weights and biases
  weights_ -= step * tmp * last_input_.transpose();
  bias_ -= step * tmp;

  return propagated;
}
