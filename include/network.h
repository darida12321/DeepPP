#pragma once

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Network {
 public:
  Network(std::vector<MatrixXd>, std::vector<VectorXd>,
          std::vector<std::function<VectorXd(VectorXd)>>, 
          std::vector<std::function<MatrixXd(VectorXd)>>);
  Network(std::vector<int>,
          std::vector<std::function<VectorXd(VectorXd)>>, 
          std::vector<std::function<MatrixXd(VectorXd)>>); 
  VectorXd forwardProp(VectorXd);
  void train(std::vector<VectorXd>, std::vector<VectorXd>, double);

  double getCost(std::vector<VectorXd>, std::vector<VectorXd>);
  double getAccuracy(std::vector<VectorXd>, std::vector<VectorXd>);

  std::vector<MatrixXd>& getWeights();
  std::vector<VectorXd>& getBiases();

 private:
  std::vector<MatrixXd> weights_ = std::vector<MatrixXd>();
  std::vector<VectorXd> biases_ = std::vector<VectorXd>();
  std::vector<std::function<VectorXd(VectorXd)>> act_func_;
  std::vector<std::function<MatrixXd(VectorXd)>> act_func_der_;
};
