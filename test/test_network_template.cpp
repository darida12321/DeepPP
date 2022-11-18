#include <gtest/gtest.h>
#include <templates/network.h>
#include <templates/mnist_imageset.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::Matrix;
using Eigen::Vector;
using namespace Template;

// TODO use is_approx
TEST(NetworkTemplate, SoftMax) {
  // Create 2-2-2 neural network
  Matrix<double, 2, 2> w1{{1, 2}, {1, 4}};
  Vector<double, 2> b1{9, -1};
  Matrix<double, 2, 2> w2{{1, 2}, {1, 4}};
  Vector<double, 2> b2{9, -1};

  Network<MeanSquareError, WeightZero, BiasZero, InputLayer<2>,
          Layer<2, Linear>, Layer<2, Softmax>>
      network;
  network.setWeights(w1, w2);
  network.setBiases(b1, b2);

  // Check forwardpropogation value
  Vector<double, 2> in1{2, 4};
  Vector<double, 2> in2{1, 1};

  Vector<double, 2> out1 = network.forwardProp(in1);
  Vector<double, 2> out2 = network.forwardProp(in2);

  EXPECT_NEAR(out1(0), 0, 0.001);
  EXPECT_NEAR(out1(1), 1, 0.001);
  EXPECT_NEAR(out2(0), 0.8807, 0.001);
  EXPECT_NEAR(out2(1), 0.1192, 0.001);

  // Check backpropogation value
  Vector<double, 2> t_in1{1.0, 1.0};
  Vector<double, 2> t_out1{0.6, 0.4};
  Vector<double, 2> t_in2{3.0, 5.0};
  Vector<double, 2> t_out2{0.2, 0.8};
  std::vector<Vector<double, 2>> input{t_in1, t_in2};
  std::vector<Vector<double, 2>> output{t_out1, t_out2};
  network.train(input, output, 1);

  Matrix<double, 2, 2> w1_final{{1, 2}, {1.059, 4.058}};
  Vector<double, 2> b1_final{9, -0.941};
  Matrix<double, 2, 2> w2_final{{0.646, 1.882}, {1.354, 4.118}};
  Vector<double, 2> b2_final{8.971, -0.971};

  EXPECT_TRUE(w1_final.isApprox(network.getWeight<0>(), 0.001));
  EXPECT_TRUE(w2_final.isApprox(network.getWeight<1>(), 0.001));
  EXPECT_TRUE(b1_final.isApprox(network.getBias<0>(), 0.001));
  EXPECT_TRUE(b2_final.isApprox(network.getBias<1>(), 0.001));
}

TEST(NetworkTemplate, TestInitialisation) {
  Network<MeanSquareError, WeightOnes, BiasZero, InputLayer<2>,
          Layer<2, Linear>, Layer<2, Softmax>>
      network;

  Matrix<double, 2, 2> w1{{1, 1}, {1, 1}};
  Vector<double, 2> b1{0, 0};
  Matrix<double, 2, 2> w2{{1, 1}, {1, 1}};
  Vector<double, 2> b2{0, 0};

  EXPECT_TRUE(w1.isApprox(network.getWeight<0>(), 0.001));
  EXPECT_TRUE(w2.isApprox(network.getWeight<1>(), 0.001));
  EXPECT_TRUE(b1.isApprox(network.getBias<0>(), 0.001));
  EXPECT_TRUE(b2.isApprox(network.getBias<1>(), 0.001));
}

TEST(NetworkTemplate, TestMnistSingle) {
  const size_t IMAGESIZE = IMG_WIDTH*IMG_HEIGHT;
  Network<
    CategoricalCrossEntropy, WeightZero, BiasZero,
    InputLayer<IMAGESIZE>, 
    Layer<128, Relu>, 
    Layer<128, Relu>, 
    Layer<10, Softmax>
  > network;

  ImageSet image;
  std::vector<Vector<double, IMAGESIZE>> x_train{image.getImage(0)};
  std::vector<Vector<double, 10>> y_train{image.getLabel(0)};

  Vector<double, 10> exp1{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
  Vector<double, 10> out1 = network.forwardProp(x_train[0]);

  network.train(x_train, y_train, 1);

  Vector<double, 10> exp2{{0.0853, 0.0853, 0.0853, 0.0853, 0.0853, 0.2320, 0.0853, 0.0853,
                 0.0853, 0.0853}};
  Vector<double, 10> out2 = network.forwardProp(x_train[0]);

  for (int i = 0; i < 10; i++) {
    EXPECT_NEAR(exp1(i), out1(i), 0.001);
    EXPECT_NEAR(exp2(i), out2(i), 0.001);
  }
}

TEST(NetworkTemplate, Fooooo) { // TODO delete
  // const size_t IMAGESIZE = IMG_WIDTH*IMG_HEIGHT;
  // Network<
  //   CategoricalCrossEntropy, WeightZero, BiasZero,
  //   InputLayer<IMAGESIZE>, 
  //   Layer<128, Relu>, 
  //   Layer<128, Relu>, 
  //   Layer<10, Softmax>
  // > network;
  //
  //
  // ImageSet image;
  //
  // auto x_train{image.getTrainImages()};
  // auto y_train{image.getTrainLabels()};
  // // x_train.resize(100);
  // // y_train.resize(100);
  // x_train.resize(99);
  // y_train.resize(99);
  //
  // network.train(x_train, y_train, 0.1);
  // network.train(x_train, y_train, 0.1);
  // network.train(x_train, y_train, 0.1);
  // network.train(x_train, y_train, 0.1);
  // network.train(x_train, y_train, 0.1);
  // double acc = network.getAccuracy(image.getTestImages(), image.getTestLabels());
  // std::cout << "Acc: " << acc << std::endl;
}
