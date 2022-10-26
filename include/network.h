#pragma once

#include <Eigen/Dense>
#include <layer.h>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Network {
public:
  Network(std::vector<MatrixXd>, std::vector<VectorXd>, std::function<VectorXd(VectorXd)>,
           std::function<VectorXd(VectorXd)>);
  Network(std::vector<int>, std::function<VectorXd(VectorXd)>,
           std::function<VectorXd(VectorXd)>);
  VectorXd forwardProp(VectorXd);
  double getCost(std::vector<VectorXd>, std::vector<VectorXd>);
  void train(std::vector<VectorXd>, std::vector<VectorXd>, double);
  std::vector<MatrixXd>& getWeights();
  std::vector<VectorXd>& getBiases();

private:
  std::vector<MatrixXd> weights_ = std::vector<MatrixXd>();
  std::vector<VectorXd> biases_ = std::vector<VectorXd>();
  std::function<VectorXd(VectorXd)> act_func_;     // activator functionon
  std::function<VectorXd(VectorXd)> act_func_der_; // activator function derivative
};
