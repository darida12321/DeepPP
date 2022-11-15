#include <templates/network.h>
#include <templates/mnist_imageset.h>

#include <Eigen/Dense>
#include <cstddef>
#include <iostream>
#include "Eigen/src/Core/Matrix.h"

// using Eigen::Matrix;
// using Eigen::Vector;
using namespace Template;

const size_t IMAGESIZE = IMG_WIDTH*IMG_HEIGHT;

int main() {
  // Matrix<double, 128, IMAGESIZE> w1 = Matrix<double, 128, IMAGESIZE>::Random();
  // Vector<double, 128>            b1 = Vector<double, 128>::Zero();
  // Matrix<double, 128, 128>       w2 = Matrix<double, 128, 128>::Random();
  // Vector<double, 128>            b2 = Vector<double, 128>::Zero();
  // Matrix<double, 10, 128>        w3 = Matrix<double, 10, 128>::Random();
  // Vector<double, 10>             b3 = Vector<double, 10>::Zero();

  Network<
    MeanSquareError, WeightRandom, BiasZero,
    InputLayer<IMAGESIZE>, 
    Layer<100, Relu>, 
    Layer<100, Relu>, 
    Layer<10, Softmax>
  > network;

  // network.setWeights(w1, w2, w3);
  // network.setBiases(b1, b2, b3);

  ImageSet image;
  
  auto trainImages = image.getTrainImages();
  auto trainLabels = image.getTrainLabels();
  
  double acc =
      network.getAccuracy(trainImages, trainLabels);
  std::cout << "Network accuracy: " << acc << std::endl;

std::cout << "Started training" << std::endl;
  for (int i = 0; i < 3; i++) {
    network.train(trainImages, trainLabels, 0.1);

    std::cout << "Completed round " << i << std::endl;
    double acc = network.getAccuracy(trainImages, trainLabels);
    std::cout << "Network accuracy: " << acc << std::endl;
  
  }

  return 0;
}
