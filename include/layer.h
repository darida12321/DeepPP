#pragma once

#include "Eigen/Core"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer {
public:
  Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func, std::function<double(double)> act_func_der);
  VectorXd forwardProp(VectorXd in);
  VectorXd backProp(VectorXd err, double stepSize);

private:
  MatrixXd weights_;
  VectorXd bias_;
  std::function<double(double)> act_func_; // activator functionon
  std::function<double(double)>
      act_func_der_;         // activator functionon derivative
  VectorXd act_derivatives_; // for back propagation
  VectorXd last_input_;      // for back propagation
};
