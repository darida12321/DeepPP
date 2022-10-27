#include <network.h>

#include <iostream>

#include "Eigen/Core"
#include "Eigen/src/Core/Matrix.h"

// Constructor for the layer
Network::Network(std::vector<MatrixXd> weights, std::vector<VectorXd> biases,
                 std::vector<std::function<VectorXd(VectorXd)>> act_func,
                 std::vector<std::function<VectorXd(VectorXd)>> act_func_der)
    : weights_(weights),
      biases_(biases),
      act_func_(act_func),
      act_func_der_(act_func_der) {}
Network::Network(std::vector<int> sizes,
                 std::vector<std::function<VectorXd(VectorXd)>> act_func,
                 std::vector<std::function<VectorXd(VectorXd)>> act_func_der)
    : act_func_(act_func), act_func_der_(act_func_der) {
  for (int i = 0; i < sizes.size() - 1; i++) {
    weights_.push_back(MatrixXd::Random(sizes[i + 1], sizes[i]));
    biases_.push_back(VectorXd::Random(sizes[i + 1]));
  }
}

// Propogate the values forward through the layers
VectorXd Network::forwardProp(VectorXd in) {
  VectorXd curr = in;
  for (int i = 0; i < weights_.size(); i++) {
    curr = act_func_[i](weights_[i] * curr + biases_[i]);
  }
  return curr;
}

// Get the cost of the function for a set of inputs
double Network::getCost(std::vector<VectorXd> in,
                        std::vector<VectorXd> exp_out) {
  double error = 0;
  for (int i = 0; i < in.size(); i++) {
    VectorXd diff = forwardProp(in[i]).array() - exp_out[i].array();
    error += (diff.array() * diff.array()).sum();
  }
  return error / in.size();
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
    std::vector<VectorXd> dadz(weights_.size());

    // Forward propogation
    VectorXd prop = in[i];
    for (int i = 0; i < weights_.size(); i++) {
      VectorXd newV = weights_[i] * prop + biases_[i];

      // Record data for backpropogation
      a[i] = prop;
      dadz[i] = act_func_der_[i](newV);

      // Get the forward propogated value
      prop = act_func_[i](newV);
    }

    // Backward propogation
    VectorXd dcda = 1 * (prop - exp_out[i]);
    for (int i = weights_.size() - 1; i >= 0; i--) {
      VectorXd dcdz = dcda.cwiseProduct(dadz[i]);

      // calculate dC/da for previous layer
      dcda = weights_[i].transpose() * dcdz;

      // adjust weights and biases
      backprop_weight_acc[i] -= stepSize * dcdz * a[i].transpose();
      backprop_bias_acc[i] -= stepSize * dcdz;
    }
  }

  // Apply the accumulated changes
  for (int i = 0; i < weights_.size(); i++) {
    weights_[i] += backprop_weight_acc[i] / in.size();
    biases_[i] += backprop_bias_acc[i] / in.size();
  }
}

// Get a vector containing weight matrices for all layers
std::vector<MatrixXd>& Network::getWeights() { return weights_; }

// Get a vector containing bias vectors for all layers
std::vector<VectorXd>& Network::getBiases() { return biases_; }
