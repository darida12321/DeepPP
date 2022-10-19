#pragma once

#include <Eigen/Dense>
#include <layer.h>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Network {
public:
  Network(std::vector<Layer>);
  VectorXd forwardProp(VectorXd);
  VectorXd backProp(VectorXd, double);

private:
  std::vector<Layer> layers_;
};
