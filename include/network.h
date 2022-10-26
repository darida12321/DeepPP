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
  double getCost(std::vector<VectorXd>, std::vector<VectorXd>);
  void train(std::vector<VectorXd>, std::vector<VectorXd>, double);
  Layer getLayer(size_t);
  std::vector<MatrixXd> getWeights();
  std::vector<VectorXd> getBiases();

private:
  VectorXd forwardPropAndStore(VectorXd);
  void backProp(VectorXd, VectorXd, double); 
  std::vector<Layer> layers_;
};
