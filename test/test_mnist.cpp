#include <activation_function.h>
#include <cost_function.h>
#include <gtest/gtest.h>
#include <network.h>
#include <mnist_imageset.h>

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>

typedef Eigen::VectorXd ImgVector;
typedef Eigen::VectorXd ImgLabel;

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
                  &mean_sqr_error);

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
                  &cat_cross_entropy);

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
                  &cat_cross_entropy);

  return;  // TODO actually do this test
  ImageSet image;

  for (int i = 0; i < 3; i++) {
    network.train(image.getTrainImages(), image.getTrainLabels(), 0.1);

    // double acc1 =
    //     network.getAccuracy(image.getTrainImages(), image.getTrainLabels());
    // double acc2 =
    //     network.getAccuracy(image.getTestImages(), image.getTestLabels());
    std::cout << "Round " << i << std::endl;
  }

  double acc =
      network.getAccuracy(image.getTrainImages(), image.getTrainLabels());
  ASSERT_GT(acc, 0.9);
}
