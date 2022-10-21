#pragma once

#include "Eigen/Core"
#include "Eigen/src/Core/Matrix.h"
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer {
public:
  Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func,
        std::function<double(double)> act_func_der);
  VectorXd forwardPropAndStore(VectorXd in);
  VectorXd forwardProp(VectorXd in);
  VectorXd backProp(VectorXd gradient, double stepSize);
  void applyAccumulatedChange(int sampleSize);
  MatrixXd getWeights();
  VectorXd getBias();

private:
  MatrixXd weights_;
  VectorXd bias_;
  std::function<double(double)> act_func_;     // activator functionon
  std::function<double(double)> act_func_der_; // activator function derivative
  VectorXd act_derivatives_;                   // for back propagation
  VectorXd last_input_;                        // for back propagation
  MatrixXd backprop_weight_acc_;
  VectorXd backprop_bias_acc_;
  
};
