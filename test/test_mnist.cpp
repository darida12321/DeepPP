#include <activation_function.h>
#include <cost_function.h>
#include <gtest/gtest.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>

// MAX_TRAIN_SIZE = 60000;
// MAX_TEST_SIZE = 10000;
const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;

typedef Eigen::VectorXd ImgVector;
typedef Eigen::VectorXd ImgLabel;

class ImageSet {
 public:
  ImageSet() {
    std::ifstream trainingLabels("test/img/train-labels-idx1-ubyte");
    std::ifstream trainingImages("test/img/train-images-idx3-ubyte");
    std::ifstream testLabels("test/img/t10k-labels-idx1-ubyte");
    std::ifstream testImages("test/img/t10k-images-idx3-ubyte");

    trainingLabels.seekg(8, std::ios_base::beg);
    trainingImages.seekg(16, std::ios_base::beg);
    testLabels.seekg(8, std::ios_base::beg);
    testImages.seekg(16, std::ios_base::beg);

    for (int i = 0; i < TRAIN_SIZE; i++) {
      char cl;
      char ci;
      trainingLabels.get(cl);
      trainLabels_.push_back(charToLabel(cl));

      ImgVector img(28 * 28);
      for (int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j++) {
        trainingImages.get(ci);
        img(j) = (double)(unsigned char)ci / 256.0;
      }
      img.normalize();
      trainImages_.push_back(img);
    }

    for (int i = 0; i < TEST_SIZE; i++) {
      char cl;
      char ci;
      testLabels.get(cl);
      testLabels_.push_back(charToLabel(cl));

      ImgVector img(28 * 28);
      for (int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j++) {
        testImages.get(ci);
        img(j) = (double)(unsigned char)ci / 256.0;
      }
      img.normalize();
      testImages_.push_back(img);
    }
  }
  ImgVector& getImage(int index) { return trainImages_[index]; }
  ImgLabel& getLabel(int index) { return trainLabels_[index]; }
  void printImage(int index) {
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

  std::vector<ImgLabel> getTrainLabels() { return trainLabels_; }
  std::vector<ImgVector> getTrainImages() { return trainImages_; }
  std::vector<ImgLabel> getTestLabels() { return testLabels_; }
  std::vector<ImgVector> getTestImages() { return testImages_; }

 private:
  std::vector<ImgLabel> trainLabels_;
  std::vector<ImgVector> trainImages_;
  std::vector<ImgLabel> testLabels_;
  std::vector<ImgVector> testImages_;

  ImgLabel charToLabel(char c) {
    ImgLabel label(10);
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

TEST(MnistTest, MSEtest) {
  MatrixXd w1 = MatrixXd::Zero(128, 28 * 28);
  VectorXd b1 = VectorXd::Zero(128);
  MatrixXd w2 = MatrixXd::Zero(128, 128);
  VectorXd b2 = VectorXd::Zero(128);
  MatrixXd w3 = MatrixXd::Zero(10, 128);
  VectorXd b3 = VectorXd::Zero(10);
  Network network(std::vector<MatrixXd>{w1, w2, w3},
                  std::vector<VectorXd>{b1, b2, b3},
                  std::vector<ActivationFunction*>{&relu, &relu, &softmax},
                  mean_sqr_error, mean_sqr_error_der);

  ImageSet image;
  std::vector<VectorXd> x_train{image.getImage(0)};
  std::vector<VectorXd> y_train{image.getLabel(0)};

  VectorXd exp1{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
  VectorXd out1 = network.forwardProp(x_train[0]);

  network.train(x_train, y_train, 1);

  VectorXd exp2{{0.0998, 0.0998, 0.0998, 0.0998, 0.0998, 0.1018, 0.0998, 0.0998,
                 0.0998, 0.0998}};
  VectorXd out2 = network.forwardProp(x_train[0]);

  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(exp1(i), out1(i), 0.001);
    EXPECT_NEAR(exp2(i), out2(i), 0.001);
  }

  std::vector<VectorXd> x_test, y_test;
  for (int i = 0; i < 100; i++) {
    x_test.push_back(image.getImage(200 + i));
    y_test.push_back(image.getLabel(200 + i));
  }
  double acc = network.getAccuracy(x_test, y_test);
  EXPECT_NEAR(acc, 0.1, 0.001);
}

TEST(MnistTest, SCCtest) {
  MatrixXd w1 = MatrixXd::Zero(128, 28 * 28);
  VectorXd b1 = VectorXd::Zero(128);
  MatrixXd w2 = MatrixXd::Zero(128, 128);
  VectorXd b2 = VectorXd::Zero(128);
  MatrixXd w3 = MatrixXd::Zero(10, 128);
  VectorXd b3 = VectorXd::Zero(10);
  Network network(std::vector<MatrixXd>{w1, w2, w3},
                  std::vector<VectorXd>{b1, b2, b3},
                  std::vector<ActivationFunction*>{&relu, &relu, &softmax},
                  cat_cross_entropy, cat_cross_entropy_der);

  ImageSet image;
  std::vector<VectorXd> x_train{image.getImage(0)};
  std::vector<VectorXd> y_train{image.getLabel(0)};

  VectorXd exp1{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
  VectorXd out1 = network.forwardProp(x_train[0]);

  network.train(x_train, y_train, 1);

  VectorXd exp2{{0.0853, 0.0853, 0.0853, 0.0853, 0.0853, 0.2320, 0.0853, 0.0853,
                 0.0853, 0.0853}};
  VectorXd out2 = network.forwardProp(x_train[0]);

  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(exp1(i), out1(i), 0.001);
    EXPECT_NEAR(exp2(i), out2(i), 0.001);
  }
}

TEST(MnistTest, IntegrationTest) {
  MatrixXd w1 = MatrixXd::Random(128, 28 * 28);
  VectorXd b1 = VectorXd::Zero(128);
  MatrixXd w2 = MatrixXd::Random(128, 128);
  VectorXd b2 = VectorXd::Zero(128);
  MatrixXd w3 = MatrixXd::Random(10, 128);
  VectorXd b3 = VectorXd::Zero(10);

  Network network(std::vector<MatrixXd>{w1, w2, w3},
                  std::vector<VectorXd>{b1, b2, b3},
                  std::vector<ActivationFunction*>{&relu, &relu, &softmax},
                  cat_cross_entropy, cat_cross_entropy_der);

  ImageSet image;

  for (int i = 0; i < 300; i++) {
    network.train(image.getTrainImages(), image.getTrainLabels(), 0.1);

    double acc1 =
        network.getAccuracy(image.getTrainImages(), image.getTrainLabels());
    double acc2 =
        network.getAccuracy(image.getTestImages(), image.getTestLabels());
    std::cout << "Round " << i << " train: " << acc1 << ", test: " << acc2
              << std::endl;
  }
}
