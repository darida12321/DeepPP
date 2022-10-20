#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Demonstrate some basic assertions.
TEST(ExpectAndAssert, Basics) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);

  EXPECT_FALSE(false);
  // EXPECT_FALSE(true);
  // EXPECT_FALSE(true);
  // ASSERT_TRUE(false);
  // EXPECT_FALSE(true);
}

TEST(Matrices, Values) {
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);

    EXPECT_EQ(m(0, 0), 3);
    EXPECT_EQ(m(1, 0), 2.5);
    EXPECT_EQ(m(0, 1), -1);
    EXPECT_EQ(m(1, 1), 1.5);
}

TEST(LayerForwardProp, ZeroMatrix) {
  MatrixXd m {
    {0, 0, 0},
    {0, 0, 0},
    {0, 0, 0}
  };
  VectorXd b {{0, 0, 0}};
  VectorXd v {{1, 1, 1}};

  Layer layer(m, b, relu, relu_derivative);
  EXPECT_EQ(layer.forwardProp(v), b);
}