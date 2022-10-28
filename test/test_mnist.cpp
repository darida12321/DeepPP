#include <activation_function.h>
#include <gtest/gtest.h>
// #include <image.h>
#include <network.h>
#include <fstream>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

const int TEST_SIZE = 10000;
const int TRAIN_SIZE = 60000;
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;

typedef Eigen::Vector<double, IMG_WIDTH * IMG_HEIGHT> ImgVector;
typedef Eigen::Vector<double, 10> ImgLabel;

class ImageSet {
 public:
  ImageSet(){
    std::ifstream trainingLabels("test/img/train-labels-idx1-ubyte");
    std::ifstream trainingImages("test/img/train-images-idx3-ubyte");

    trainingLabels.seekg(8, std::ios_base::beg);
    trainingImages.seekg(16, std::ios_base::beg);

    for (int i = 0; i < TRAIN_SIZE; i++) {
      char cl;
      char ci;
      trainingLabels.get(cl);
      trainLabels_.push_back(charToLabel(cl));

      ImgVector img;
      for (int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j++) {
        trainingImages.get(ci);
        img(j) = ci;
      }
      trainImages_.push_back(img);
    }
  }
  ImgVector& getImage(int index){
    return trainImages_[index]; 
  }
  ImgLabel& getLabel(int index){
    return trainLabels_[index]; 
  }
  void printImage(int index){
    ImgVector img = trainImages_[index];
    ImgLabel label = trainLabels_[index];
    for (int i = 0; i < IMG_HEIGHT; i++) {
      for (int j = 0; j < IMG_WIDTH; j++) {
        if ((unsigned int)img(IMG_WIDTH * i + j) > 128) {
          std::cout << "@@";
        } else {
          std::cout << "  ";
        }
      }
      std::cout << std::endl;
    }
  }

 private:
  std::vector<ImgLabel> trainLabels_;
  std::vector<ImgVector> trainImages_;
  ImgLabel charToLabel(char c) {
    ImgLabel label;
    for (int i = 0; i < 10; i++) {
      if (c == i) {
        label(i) = 1.0;
      } else {
        label(i) = 0.0;
      }
    }
    return label;
  }
};


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
  MatrixXd w1 = MatrixXd::Zero(128, 28*28);
  VectorXd b1 = VectorXd::Zero(128);
  MatrixXd w2 = MatrixXd::Zero(128, 128);
  VectorXd b2 = VectorXd::Zero(128);
  MatrixXd w3 = MatrixXd::Zero(10, 128);
  VectorXd b3 = VectorXd::Zero(10);
  Network network(std::vector<MatrixXd>{w1, w2, w3}, std::vector<VectorXd>{b1, b2, b3},
      std::vector<std::function<VectorXd(VectorXd)>>{relu, relu, softmax},
      std::vector<std::function<MatrixXd(VectorXd)>>{reluDerivative, reluDerivative, softmaxDerivative});

  ImageSet image;
  std::vector<VectorXd> x_train{image.getImage(0)};
  std::vector<VectorXd> y_train{image.getLabel(0)};

  VectorXd exp1{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
  VectorXd out1 = network.forwardProp(x_train[0]);

  network.train(x_train, y_train, 1);

  VectorXd exp2{{0.0998, 0.0998, 0.0998, 0.0998, 0.0998, 0.1018, 0.0998, 0.0998, 0.0998, 0.0998}};
  VectorXd out2 = network.forwardProp(x_train[0]);

  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(exp1(i), out1(i), 0.001);
    EXPECT_NEAR(exp2(i), out2(i), 0.001);
  }
}













