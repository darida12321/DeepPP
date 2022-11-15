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
  Network<
    CategoricalCrossEntropy, WeightRandom, BiasZero,
    InputLayer<IMAGESIZE>, 
    Layer<128, Relu>, 
    Layer<128, Relu>, 
    Layer<10, Softmax>
  > network;

  // TODO test initialization plsssss
  Eigen::MatrixXd w1 = Eigen::MatrixXd::Random(128, 28 * 28);
  Eigen::MatrixXd w2 = Eigen::MatrixXd::Random(128, 128);
  Eigen::MatrixXd w3 = Eigen::MatrixXd::Random(10, 128);
  Eigen::MatrixXd w1_t(w1);
  Eigen::MatrixXd w2_t(w2);
  Eigen::MatrixXd w3_t(w3);
  network.setWeights(w1_t, w2_t, w3_t);

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
    double acc = network.getAccuracy(image.getTestImages(), image.getTestLabels());
    std::cout << "Network accuracy: " << acc << std::endl;
  
  }

  return 0;
}
