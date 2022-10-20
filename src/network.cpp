#include "Eigen/Core"
#include "Eigen/src/Core/Matrix.h"
#include <network.h>

Network::Network(std::vector<Layer> layers) : layers_(layers) {}

VectorXd Network::forwardProp(VectorXd in) {
  VectorXd curr = in;
  for (int i = 0; i < layers_.size(); i++) {
    curr = layers_[i].forwardProp(curr);
  }
  return curr;
}
VectorXd Network::forwardPropAndStore(VectorXd in) {
  VectorXd curr = in;
  for (int i = 0; i < layers_.size(); i++) {
    curr = layers_[i].forwardPropAndStore(curr);
  }
  return curr;
}
VectorXd Network::backProp(VectorXd err, double stepSize) {
  VectorXd curr = err;
  for (int i = layers_.size() - 1; i >= 0; i--) {
    curr = layers_[i].backProp(curr, stepSize);
  }
  return err;
}
VectorXd Network::trainOne(VectorXd in, VectorXd exp_out, double stepSize) {
  VectorXd err = forwardPropAndStore(in) - exp_out;
  backProp(err, stepSize);
}
