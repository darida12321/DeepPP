#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer {
public:
  Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func);
  VectorXd forwardProp(VectorXd in);
  VectorXd backProp(VectorXd err);

private:
  MatrixXd weights_;
  VectorXd bias_;
  std::function<double(double)> act_func_; // activator function
};
