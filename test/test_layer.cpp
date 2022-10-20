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