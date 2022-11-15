#include <templates/network.h>

#include <Eigen/Dense>
#include <iostream>

using Eigen::Matrix;
using Eigen::Vector;
using namespace Template;

int main() {
  Network<
    CategoricalCrossEntropy, WeightRandom, BiasZero,
    InputLayer<IMAGESIZE>, 
    Layer<128, Relu>, 
    Layer<128, Relu>, 
    Layer<10, Softmax>
  > network;

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
