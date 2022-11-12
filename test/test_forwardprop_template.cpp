#include <gtest/gtest.h>
#include <templates/template_test.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::Matrix;
using Eigen::Vector;

TEST(LayerForwardPropTemplate, SoftMax) {
  // Create 2-2 neural network
  Matrix<double, 2, 2> w1;
  w1 << 1, 2, 1, 4;
  Vector<double, 2> b1;
  b1 << 9, -1;
  Matrix<double, 2, 2> w2;
  w2 << 1, 2, 1, 4;
  Vector<double, 2> b2;
  b2 << 9, -1;

  Network<MeanSquareError, Layer<2, 2, Linear>, Layer<2, 2, Softmax> > network;
  network.setWeights(w1, w2);
  network.setBiases(b1, b2);

  // Check forwardpropogation value
  Vector<double, 2> in1;
  in1 << 2, 4;
  Vector<double, 2> in2;
  in2 << 1, 1;

  Vector<double, 2> out1 = network.forwardProp(in1);
  Vector<double, 2> out2 = network.forwardProp(in2);

  EXPECT_NEAR(out1(0), 0, 0.001);
  EXPECT_NEAR(out1(1), 1, 0.001);
  EXPECT_NEAR(out2(0), 0.8807, 0.001);
  EXPECT_NEAR(out2(1), 0.1192, 0.001);
}
