#include <activation_function.h>
#include <gtest/gtest.h>
#include <image.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

int getPrediction(VectorXd out) {
  int maxIndex = 0;
  for (int i = 1; i < 10; i++) {
    if (out(i) > out(maxIndex)) {
      maxIndex = i;
    }
  }
  return maxIndex;
}
void printPrediction(VectorXd out) {
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(3);
  for (int i = 0; i < 10; i++) {
    std::cout << "  " << i << "   | ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 10; i++) {
    std::cout << out(i) << " | ";
  }
  std::cout << getPrediction(out);
}

TEST(MnistTest, ReadData) {
  MatrixXd w1 = MatrixXd::Random(10, 28*28);
  VectorXd b1 = VectorXd::Random(10);
  // Layer layer1(w1, b1, sigmoid, sigmoid_derivative);
  MatrixXd w2 = MatrixXd::Random(10, 10);
  VectorXd b2 = VectorXd::Random(10);
  // Layer layer2(w2, b2, sigmoid, sigmoid_derivative);
  MatrixXd w3 = MatrixXd::Random(10, 10);
  VectorXd b3 = VectorXd::Random(10);
  // Layer layer3(w3, b3, softmax, softmax_derivative);
  Network network(std::vector<MatrixXd>{w1, w2, w3}, std::vector<VectorXd>{b1, b2, b3},
      std::vector<std::function<VectorXd(VectorXd)>>{sigmoid, sigmoid, softmax},
      std::vector<std::function<VectorXd(VectorXd)>>{sigmoidDerivative, sigmoidDerivative, softmaxDerivative});

  ImageSet image;

  int training_size = 100;
  for (int j = 0; j < 50; j++) {
      std::vector<VectorXd> in1;
      std::vector<VectorXd> out1;
      for (int i = 0; i < training_size; i++) {
          in1.push_back(image.GetImage(j*training_size + i));
          out1.push_back(image.GetLabel(j*training_size + i));
      }

      double cost = network.getCost(in1, out1);
      std::cout << cost << std::endl;
      for (int i = 0; i < 1; i++) {
        network.train(in1, out1, 0.1);
        double cost = network.getCost(in1, out1);
        std::cout << cost << std::endl;
        // if (cost < 0.85) {
        //     break;
        // }
        for (int i = 0; i < 1; i++) {
          // std::cout << network.getWeights()[2] << std::endl;
          // std::cout << network.getBiases()[2] << std::endl;
          int index = 50000+i;
          printPrediction(network.forwardProp(image.GetImage(index)));
          std::cout << " Actual: " << getPrediction(image.GetLabel(index)) <<
          std::endl;
        }
      }

  }

  std::cout << "XXXXX THE PREDICTIONS XXXXX" << std::endl;
  for (int i = 0; i < 10; i++) {
      int index = 50000+i;
      printPrediction(network.forwardProp(image.GetImage(index)));
      std::cout << " Actual: " << getPrediction(image.GetLabel(index)) <<
      std::endl;
  }

  image.PrintImage(1000);
}
