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

void compareNetwork(std::vector<MatrixXd> ws, std::vector<VectorXd> bs,
                    Network n) {
  ASSERT_EQ(ws.size(), bs.size());
  for (int i = 0; i < ws.size(); i++) {
    ASSERT_EQ(ws[i].rows(), n.getWeights()[i].rows());
    ASSERT_EQ(ws[i].cols(), n.getWeights()[i].cols());
    ASSERT_EQ(bs[i].rows(), n.getBiases()[i].rows());
    ASSERT_EQ(ws[i].rows(), bs[i].rows());

    for (int y = 0; y < ws[i].rows(); y++) {
      for (int x = 0; x < ws[i].cols(); x++) {
        EXPECT_NEAR(n.getWeights()[i](y, x), ws[i](y, x), 0.001);
      }
      EXPECT_NEAR(n.getBiases()[i](y), bs[i](y), 0.001);
    }
  }
}

TEST(LayerBackPropTest, MultiInput) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                  std::vector<std::function<VectorXd(VectorXd)>>{relu, relu},
                  std::vector<std::function<VectorXd(VectorXd)>>{
                      relu_derivative, relu_derivative});

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  network.train(input, output, 1);

  // Expect output
  MatrixXd w1(2, 2);
  w1 << -39, -63, -39, -63;
  VectorXd b1(2);
  b1 << -15, -15;
  MatrixXd w2(2, 2);
  w2 << -63.5, -63.5, -54.5, -54.5;
  VectorXd b2(2);
  b2 << -7.5, -6.5;
  std::vector<MatrixXd> ws{w1, w2};
  std::vector<VectorXd> bs{b1, b2};
  compareNetwork(ws, bs, network);
}

TEST(LayerBackPropTest, Relu) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(
      std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
      std::vector<std::function<VectorXd(VectorXd)>>{linear, linear},
      std::vector<std::function<VectorXd(VectorXd)>>{linear_derivative,
                                                     linear_derivative});

  // Create example data point
  VectorXd in(2);
  in << 1.0, 1.0;
  VectorXd out(2);
  out << 3.0, 3.0;
  std::vector<VectorXd> input{in};
  std::vector<VectorXd> output{out};

  // Train the network
  network.train(input, output, 1);

  // Expect output
  MatrixXd w1(2, 2);
  w1 << -7, -7, -7, -7;
  VectorXd b1(2);
  b1 << -7, -7;
  MatrixXd w2(2, 2);
  w2 << -11, -11, -11, -11;
  VectorXd b2(2);
  b2 << -3, -3;
  std::vector<MatrixXd> ws{w1, w2};
  std::vector<VectorXd> bs{b1, b2};
  compareNetwork(ws, bs, network);
}

TEST(LayerBackPropTest, Sigmoid) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(
      std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
      std::vector<std::function<VectorXd(VectorXd)>>{sigmoid, sigmoid},
      std::vector<std::function<VectorXd(VectorXd)>>{sigmoid_derivative,
                                                     sigmoid_derivative});

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  network.train(input, output, 1);

  // Expect output
  MatrixXd w1(2, 2);
  w1 << 1.0046, 1.0047, 1.0046, 1.0047;
  VectorXd b1(2);
  b1 << 1.0046, 1.0046;
  MatrixXd w2(2, 2);
  w2 << 1.1621, 1.1621, 1.2072, 1.2073;
  VectorXd b2(2);
  b2 << 1.1645, 1.2097;
  std::vector<MatrixXd> ws{w1, w2};
  std::vector<VectorXd> bs{b1, b2};
  compareNetwork(ws, bs, network);
}

// TEST(LayerBackPropTest, SoftMax) {
//   // Create 1-1 neural network
//   MatrixXd w(2, 2); w << 1, 1, 1, 1;
//   VectorXd b(2); b << 1, 1;
//   Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
//       std::vector<std::function<VectorXd(VectorXd)>>{linear, softmax},
//       std::vector<std::function<VectorXd(VectorXd)>>{linear_derivative,
//       softmax_derivative});

//   // Create example data point
//   VectorXd in1(2); in1 << 1.0, 1.0;
//   VectorXd out1(2); out1 << 0.6, 0.4;
//   VectorXd in2(2); in2 << 3.0, 5.0;
//   VectorXd out2(2); out2 << 0.2, 0.8;
//   std::vector<VectorXd> input{in1, in2};
//   std::vector<VectorXd> output{out1, out2};

//   // Train the network
//   network.train(input, output, 1);

// Expect output
// MatrixXd w1(2, 2); w1 << 1, 1, 1, 1;
// VectorXd b1(2); b1 << 1, 1;
// MatrixXd w2(2, 2); w2 << 0.4, 0.4, 1.6, 1.6;
// VectorXd b2(2); b2 << 0.95, 1.05;
// std::vector<MatrixXd> ws{w1, w2};
// std::vector<VectorXd> bs{b1, b2};
// compareNetwork(ws, bs, network);
// }
