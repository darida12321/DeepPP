#include "Eigen/Core"
#include "Eigen/src/Core/Matrix.h"
#include <network.h>

// Constructor for the layer
Network::Network(std::vector<Layer> layers) : layers_(layers) {}

// Propogate the values forward through the layers
VectorXd Network::forwardProp(VectorXd in) {
  VectorXd curr = in;
  for (int i = 0; i < layers_.size(); i++) {
    curr = layers_[i].forwardProp(curr);
  }
  return curr;
}

// Get the cost of the function for a set of inputs
double Network::getCost(std::vector<VectorXd> in, std::vector<VectorXd> exp_out) {
    double error = 0;
    for (int i = 0; i < in.size(); i++) {
        VectorXd diff = forwardProp(in[i]).array() - exp_out[i].array();
        error = (diff.array() * diff.array()).sum();
    }
    return error/in.size();
}

void train(std::vector<VectorXd> in, std::vector<VectorXd> exp_out) {
    for (int i = 0; i < in.size(); i++) {
    }
}


// Getter for a layer
Layer Network::getLayer(size_t i) {
    return layers_[i];
}

// Propogate the values forward and store data for backpropogation
VectorXd Network::forwardPropAndStore(VectorXd in) {
  VectorXd curr = in;
  for (int i = 0; i < layers_.size(); i++) {
    curr = layers_[i].forwardPropAndStore(curr);
  }
  return curr;
}

// Apply backpropogation to the layer
void Network::backProp(VectorXd in, VectorXd exp_out, double stepSize) {
  VectorXd gradient = 2*(forwardPropAndStore(in) - exp_out);
  for (int i = layers_.size() - 1; i >= 0; i--) {
    gradient = layers_[i].backProp(gradient, stepSize);
  }
}

