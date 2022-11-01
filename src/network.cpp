#include <network.h>
#include <activation_function.h>

#include <functional>
#include <iostream>

#include "Eigen/Core"
#include "Eigen/src/Core/Matrix.h"

// Constructor for the layer
Network::Network(std::vector<MatrixXd> weights, std::vector<VectorXd> biases,
                 std::vector<ActivationFunction*> act_func),
                 std::function<double(VectorXd, VectorXd)> cost_func,
                 std::function<VectorXd(VectorXd, VectorXd)> cost_func_der)
    : weights_(weights),
      biases_(biases),
      act_func_(act_func),
      cost_func_(cost_func),
      cost_func_der_(cost_func_der) {}
Network::Network(std::vector<int> sizes,
                 std::vector<ActivationFunction*> act_func,
                 std::function<double(VectorXd, VectorXd)> cost_func,
                 std::function<VectorXd(VectorXd, VectorXd)> cost_func_der)
    : act_func_(act_func),
      cost_func_(cost_func),
      cost_func_der_(cost_func_der){
  for (int i = 0; i < sizes.size() - 1; i++) {
    weights_.push_back(MatrixXd::Random(sizes[i + 1], sizes[i]));
    biases_.push_back(VectorXd::Random(sizes[i + 1]));
  }
}

VectorXd Network::forwardProp(VectorXd in) {
  VectorXd curr = in;
  for (int i = 0; i < weights_.size(); i++) {
    curr = act_func_[i]->function(weights_[i] * curr + biases_[i]);
  }
  return curr;
}

void Network::train(std::vector<VectorXd> in, std::vector<VectorXd> exp_out,
                    double stepSize) {
  assert(in.size() == exp_out.size());

  // Accumulate the changes in the gradient
  std::vector<MatrixXd> backprop_weight_acc;
  std::vector<VectorXd> backprop_bias_acc;
  for (int i = 0; i < weights_.size(); i++) {
    backprop_weight_acc.push_back(weights_[i] - weights_[i]);
    backprop_bias_acc.push_back(biases_[i] - biases_[i]);
  }

  // For each data point, accumulate the changes
  for (int i = 0; i < in.size(); i++) {
    std::vector<VectorXd> a(weights_.size());

    // Forward propogation
    VectorXd prop = in[i];
    for (int j = 0; j < weights_.size(); j++) {
      VectorXd z = weights_[j] * prop + biases_[j];

      // Record data for backpropogation
      a[i] = prop;
      dadz[i] = act_func_[i]->derivative(newV);

      // Get the forward propogated value
      prop = act_func_[i]->function(newV);
    }

    // Backward propogation
    VectorXd dcda = cost_func_der_(prop, exp_out[i]);
    for (int j = weights_.size() - 1; j >= 0; j--) {
      VectorXd z = weights_[j] * a[j] + biases_[j];
      MatrixXd dadz = act_func_der_[j](z);
      VectorXd dcdz = dadz * dcda;

      // calculate dC/da for previous layer
      dcda = weights_[j].transpose() * dcdz;

      // adjust weights and biases
      backprop_weight_acc[j] -= stepSize * dcdz * a[j].transpose();
      backprop_bias_acc[j] -= stepSize * dcdz;
    }

    // Apply the accumulated changes
    if (i % 100 == 99) {
      for (int j = 0; j < weights_.size(); j++) {
        weights_[j] += backprop_weight_acc[j] / 100;
        biases_[j] += backprop_bias_acc[j] / 100;
        backprop_weight_acc[j] = weights_[j] - weights_[j];
        backprop_bias_acc[j] = biases_[j] - biases_[j];
      }
    }
  }

  // Apply the accumulated changes
  for (int i = 0; i < weights_.size(); i++) {
    weights_[i] += backprop_weight_acc[i] / in.size();
    biases_[i] += backprop_bias_acc[i] / in.size();
  }
}

// Get the cost of the function for a set of inputs
double Network::getCost(std::vector<VectorXd> in,
                        std::vector<VectorXd> exp_out) {
  double error = 0;
  for (int i = 0; i < in.size(); i++) {
    error += cost_func_(forwardProp(in[i]), exp_out[i]);
  }
  return error / in.size();
}

// TODO: This does categorical accuracy all the time.
double Network::getAccuracy(std::vector<VectorXd> in,
                            std::vector<VectorXd> exp_out) {
  double acc = 0;
  for (int i = 0; i < in.size(); i++) {
    VectorXd val = forwardProp(in[i]);
    Eigen::Index predicted, expected;
    val.maxCoeff(&predicted);
    exp_out[i].maxCoeff(&expected);
    if (predicted == expected) {
      acc++;
    }
  }
  return acc / in.size();
}

// Get a vector containing weight matrices for all layers
std::vector<MatrixXd>& Network::getWeights() { return weights_; }
std::vector<VectorXd>& Network::getBiases() { return biases_; }
