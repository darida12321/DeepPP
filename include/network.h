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
  void trainOne(VectorXd, VectorXd, double);

private:
  VectorXd forwardPropAndStore(VectorXd);
  VectorXd backProp(VectorXd, double);
  std::vector<Layer> layers_;
};
