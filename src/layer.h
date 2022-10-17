#pragma once

#include "../Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer {
public:
  Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func);

  VectorXd forwardProp(VectorXd in);
  VectorXd backProp(VectorXd err);
};
