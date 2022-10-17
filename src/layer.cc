#include "layer.h"
#include "../Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// constructor
Layer::Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func)
    : weights_(m), bias_(b), act_func_(act_func) {}

VectorXd Layer::forwardProp(VectorXd in) {
  VectorXd newV = weights_ * in + bias_;
  newV = newV.unaryExpr(act_func_);
  return newV;
}

VectorXd Layer::backProp(VectorXd err) {
  VectorXd error = err;
  return error;
}
