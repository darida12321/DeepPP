#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

class LayerForwardPropTest:public::testing::Test {
  protected:
    MatrixXd zeroMatrix_ {
      {0, 0, 0},
      {0, 0, 0},
      {0, 0, 0}
    };

    MatrixXd idMatrix_ {
      {1, 0, 0},
      {0, 1, 0},
      {0, 0, 1}
    };

    VectorXd zeroVector_ {{0, 0, 0}};

    VectorXd v1_ {{1, 1, 1}};
    VectorXd v2_ {{0.5, 0.5, 0.5}};
};

TEST_F(LayerForwardPropTest, MediumNetwork) {
    // Create 1-1 neural network
    MatrixXd w1(2, 2); w1 << 1, 2, 3, 4;
    VectorXd b1(2); b1 << 0, -1;
    Layer layer1(w1, b1, linear, linear_derivative);
    MatrixXd w2(2, 2); w2 << 2, -4, -1, 3;
    VectorXd b2(2); b2 << -3, 1;
    Layer layer2(w2, b2, linear, linear_derivative);
    Network network(std::vector<Layer>{layer1, layer2});

    // Create example data point
    VectorXd in1(2); in1 << 2, 4;
    VectorXd in2(2); in2 << -2, 0;

    // TODO Check these numbers.
    ASSERT_LT(abs(network.forwardProp(in1)(0) - -67), 0.001);
    ASSERT_LT(abs(network.forwardProp(in1)(1) - 54), 0.001);
    ASSERT_LT(abs(network.forwardProp(in2)(0) - 21), 0.001);
    ASSERT_LT(abs(network.forwardProp(in2)(1) - -18), 0.001);

    VectorXd out1(2); out1 << -65, 54; // 4
    VectorXd out2(2); out2 << 25, -19; // 17
    std::vector<VectorXd> in{in1, in2};
    std::vector<VectorXd> out{out1, out2};

    ASSERT_LT(abs(network.getCost(in, out) - 10.5), 0.001);
}

TEST_F(LayerForwardPropTest, ZeroMatrixRelu) {
  Layer layer(zeroMatrix_, zeroVector_, relu, relu_derivative);
  EXPECT_EQ(layer.forwardProp(v1_), zeroVector_);
}

TEST_F(LayerForwardPropTest, ZeroMatrixSigmoid) {
  Layer layer(zeroMatrix_, zeroVector_, sigmoid, sigmoid_derivative);
  EXPECT_EQ(layer.forwardProp(v1_), v2_);
}

TEST_F(LayerForwardPropTest, IdMatrixRelu) {
  Layer layer(idMatrix_, zeroVector_, relu, relu_derivative);
  EXPECT_EQ(layer.forwardProp(v1_), v1_);
}

TEST_F(LayerForwardPropTest, biasOffset1) {
  Layer layer(zeroMatrix_, v1_, relu, relu_derivative);
  EXPECT_EQ(layer.forwardProp(v1_), v1_);
}

TEST_F(LayerForwardPropTest, biasOffset2) {
  Layer layer(idMatrix_, v2_, relu, relu_derivative);
  EXPECT_EQ(layer.forwardProp(v2_), v1_);
}
