#include <activation_function.h>
#include <gtest/gtest.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <typeinfo>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

void compareNetwork(std::vector<MatrixXd> ws, std::vector<VectorXd> bs, Network n) {
  ASSERT_EQ(ws.size(), bs.size());
  for (int i = 0; i < ws.size(); i++) {
    ASSERT_EQ(ws[i].rows(), n.getWeights()[i].rows());
    ASSERT_EQ(ws[i].cols(), n.getWeights()[i].cols());
    ASSERT_EQ(bs[i].rows(), n.getBiases()[i].rows());
    ASSERT_EQ(ws[i].rows(), bs[i].rows());

    for (int y = 0; y < ws[i].rows(); y++) {
      for (int x = 0; x < ws[i].cols(); x++) {
        EXPECT_NEAR(n.getWeights()[i](y, x), ws[i](y), 0.001);
      }
      EXPECT_NEAR(n.getBiases()[i](y), bs[i](y), 0.001);
    }
  }
}

TEST(LayerBackPropTest, Relu) {
  // Create 1-1 neural network
  MatrixXd w(2, 2); w << 1, 1, 1, 1;
  VectorXd b(2); b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
      relu, relu_derivative);

  // Create example data point
  VectorXd in(2); in << 1.0, 1.0;
  VectorXd out(2); out << 3.0, 3.0;
  std::vector<VectorXd> input{in};
  std::vector<VectorXd> output{out};

  // Train the network
  network.train(input, output, 1);

  // Expect output
  MatrixXd w1(2, 2); w1 << -7, -7, -7, -7;
  VectorXd b1(2); b1 << -7, -7;
  MatrixXd w2(2, 2); w2 << -11, -11, -11, -11;
  VectorXd b2(2); b2 << -3, -3;
  std::vector<MatrixXd> ws{w1, w2};
  std::vector<VectorXd> bs{b1, b2};
  compareNetwork(ws, bs, network);
}