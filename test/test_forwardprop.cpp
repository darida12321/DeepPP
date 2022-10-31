#include <activation_function.h>
#include <gtest/gtest.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(LayerForwardProp, Linear) {
  // Create 2-2 neural network
  MatrixXd w1(2, 2);
  w1 << 1, 2, 3, 4;
  VectorXd b1(2);
  b1 << 0, -1;
  MatrixXd w2(2, 2);
  w2 << 2, -4, -1, 3;
  VectorXd b2(2);
  b2 << -3, 1;
  Network network(
      std::vector<MatrixXd>{w1, w2}, std::vector<VectorXd>{b1, b2},
      std::vector<std::function<VectorXd(VectorXd)>>{linear, linear},
      std::vector<std::function<VectorXd(VectorXd)>>{linear_derivative,
                                                     linear_derivative});

  // Check forwardpropogation value
  VectorXd in1(2);
  in1 << 2, 4;
  VectorXd in2(2);
  in2 << -2, 0;

  EXPECT_NEAR(network.forwardProp(in1)(0), -67, 0.001);
  EXPECT_NEAR(network.forwardProp(in1)(1), 54, 0.001);
  EXPECT_NEAR(network.forwardProp(in2)(0), 21, 0.001);
  EXPECT_NEAR(network.forwardProp(in2)(1), -18, 0.001);

  // Check error value
  VectorXd out1(2);
  out1 << -65, 54;  // 4
  VectorXd out2(2);
  out2 << 25, -19;  // 17
  std::vector<VectorXd> in{in1, in2};
  std::vector<VectorXd> out{out1, out2};

  EXPECT_NEAR(network.getCost(in, out), 10.5, 0.001);
}

TEST(LayerForwardProp, SoftMax) {
  // Create 2-2 neural network
  MatrixXd w(2, 2);
  w << 1, 2, 1, 4;
  VectorXd b(2);
  b << 9, -1;
  Network network(
      std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
      std::vector<std::function<VectorXd(VectorXd)>>{linear, softmax},
      std::vector<std::function<VectorXd(VectorXd)>>{linear_derivative,
                                                     softmax_derivative});

  // Check forwardpropogation value
  VectorXd in1(2);
  in1 << 2, 4;
  VectorXd in2(2);
  in2 << 1, 1;

  VectorXd out1 = network.forwardProp(in1);
  VectorXd out2 = network.forwardProp(in2);

    EXPECT_NEAR(out1(0), 0, 0.001);
  EXPECT_NEAR(out1(1), 1, 0.001);
  EXPECT_NEAR(out2(0), 0.8807, 0.001);
  EXPECT_NEAR(out2(1), 0.1192, 0.001);
}
