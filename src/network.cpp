#include "Eigen/Core"
#include "Eigen/src/Core/Matrix.h"
#include <network.h>

Network::Network(std::vector<Layer> layers) : layers_(layers) {}

#include <iostream>
VectorXd Network::forwardProp(VectorXd in) {
  VectorXd curr = in;
  for (int i = 0; i < layers_.size(); i++) {
    std::cout << "From " << curr;
    std::cout << " Weight: " << layers_[i].getWeights();
    std::cout << " Bias: " << layers_[i].getBias();
    curr = layers_[i].forwardProp(curr);
    std::cout << "To " << i << ": " << curr << std::endl;
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
#include <iostream>
void Network::trainOne(VectorXd in, VectorXd exp_out, double stepSize) {
  std::cout << "Input is " << in << std::endl;
  std::cout << "a3: " << forwardProp(in) << std::endl;
  VectorXd err = 2*(forwardPropAndStore(in) - exp_out);
  std::cout << "errDer" << err << std::endl;
  backProp(err, stepSize);
}
Layer Network::getLayer(size_t i) {
    return layers_[i];
}
