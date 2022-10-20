#pragma once

#include "Eigen/src/Core/Matrix.h"
#include <Eigen/Dense>
#include <layer.h>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Network {
public:
  Network(std::vector<Layer>);
  VectorXd forwardProp(VectorXd);
  void trainOne(VectorXd, VectorXd);

private:
  VectorXd forwardPropAndStore(VectorXd);
  VectorXd backProp(VectorXd, VectorXd, double);
  std::vector<Layer> layers_;
};
