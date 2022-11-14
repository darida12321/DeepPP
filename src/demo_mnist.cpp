#include <templates/network.h>

#include <Eigen/Dense>
#include <iostream>

using Eigen::Matrix;
using Eigen::Vector;
using namespace Template;

int main() {
  std::cout << "Hi sussy bakas!!!" << std::endl;
  Matrix<double, 2, 2> w1{{1, 2}, {1, 4}};
  Vector<double, 2> b1{9, -1};
  Matrix<double, 2, 2> w2{{1, 2}, {1, 4}};
  Vector<double, 2> b2{9, -1};

  Network<
    MeanSquareError, 
    InputLayer<2>, 
    Layer<2, Linear>, 
    Layer<2, Softmax>
  > network;

  network.setWeights(w1, w2);
  network.setBiases(b1, b2);

  // Check forwardpropogation value
  Vector<double, 2> in{2, 4};

  Vector<double, 2> out = network.forwardProp(in);
  std::cout << "Output: " << std::endl << out << std::endl;
  return 0;
}
