#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(LayerForwardProp, MediumNetwork){
    // Create 2-2 neural network
    MatrixXd w1(2, 2); w1 << 1, 2, 3, 4;
    VectorXd b1(2); b1 << 0, -1;
    MatrixXd w2(2, 2); w2 << 2, -4, -1, 3;
    VectorXd b2(2); b2 << -3, 1;
    Network network(
      std::vector<MatrixXd>{w1, w2},
      std::vector<VectorXd>{b1, b2},
      linear, linear_derivative
    );

    // Check forwardpropogation value
    VectorXd in1(2); in1 << 2, 4;
    VectorXd in2(2); in2 << -2, 0;

    EXPECT_NEAR(network.forwardProp(in1)(0), -67, 0.001);
    EXPECT_NEAR(network.forwardProp(in1)(1), 54, 0.001);
    EXPECT_NEAR(network.forwardProp(in2)(0), 21, 0.001);
    EXPECT_NEAR(network.forwardProp(in2)(1), -18, 0.001);

    // Check error value
    VectorXd out1(2); out1 << -65, 54; // 4
    VectorXd out2(2); out2 << 25, -19; // 17
    std::vector<VectorXd> in{in1, in2};
    std::vector<VectorXd> out{out1, out2};

    EXPECT_NEAR(network.getCost(in, out), 10.5, 0.001);
}

// class LayerForwardPropTest:public::testing::Test {
//   protected:
//     MatrixXd zeroMatrix_ {
//       {0, 0, 0},
//       {0, 0, 0},
//       {0, 0, 0}
//     };
//
//     MatrixXd idMatrix_ {
//       {1, 0, 0},
//       {0, 1, 0},
//       {0, 0, 1}
//     };
//
//     VectorXd zeroVector_ {{0, 0, 0}};
//
//     VectorXd v1_ {{1, 1, 1}};
//     VectorXd v2_ {{0.5, 0.5, 0.5}};
// };
// TEST_F(LayerForwardPropTest, ZeroMatrixRelu) {
//   Layer layer(zeroMatrix_, zeroVector_, relu, relu_derivative);
//   EXPECT_EQ(layer.forwardProp(v1_), zeroVector_);
// }
//
// TEST_F(LayerForwardPropTest, ZeroMatrixSigmoid) {
//   Layer layer(zeroMatrix_, zeroVector_, sigmoid, sigmoid_derivative);
//   EXPECT_EQ(layer.forwardProp(v1_), v2_);
// }
//
// TEST_F(LayerForwardPropTest, IdMatrixRelu) {
//   Layer layer(idMatrix_, zeroVector_, relu, relu_derivative);
//   EXPECT_EQ(layer.forwardProp(v1_), v1_);
// }
//
// TEST_F(LayerForwardPropTest, biasOffset1) {
//   Layer layer(zeroMatrix_, v1_, relu, relu_derivative);
//   EXPECT_EQ(layer.forwardProp(v1_), v1_);
// }
//
// TEST_F(LayerForwardPropTest, biasOffset2) {
//   Layer layer(idMatrix_, v2_, relu, relu_derivative);
//   EXPECT_EQ(layer.forwardProp(v2_), v1_);
// }
