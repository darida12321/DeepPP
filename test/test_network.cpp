#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

// using Network::Network;

class NetworkForwardPropTest:public::testing::Test {
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

    Layer idLayer_(idMatrix_, v1_, relu, relu_derivative);
};

TEST_F(NetworkForwardPropTest, EmptyNetwork) {
  Network network({});
  EXPECT_EQ(network.forwardProp(v1_), v1_);
}

TEST_F(NetworkForwardPropTest, OneLayer) {
  Network network({idLayer_});
  EXPECT_EQ(network.forwardProp(v1_), idLayer_.forwardProp(v1_));
}