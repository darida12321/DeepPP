#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;

// Demonstrate some basic assertions.
TEST(ExpectAndAssert, Basics) {
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);

  EXPECT_FALSE(false);
  EXPECT_FALSE(true);
  EXPECT_FALSE(true);
  ASSERT_TRUE(false);
  EXPECT_FALSE(true);
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

// int main() {
//   MatrixXd l1_weights(2, 2);
//   l1_weights(0, 0) = 1;
//   l1_weights(1, 0) = 0;
//   l1_weights(0, 1) = 0;
//   l1_weights(1, 1) = 1;
//
//   VectorXd l1_bias(2);
//   l1_bias(0) = 2;
//   l1_bias(1) = 3;
//
//   Layer layer1(l1_weights, l1_bias, relu);
//
//   std::vector<Layer> layers{layer1};
//   Network network(layers);
//
//   VectorXd input(2);
//   input(0) = 1;
//   input(1) = 2;
//   VectorXd out = network.forwardProp(input);
//   std::cout << "Output: " << std::endl << out << std::endl;
//
//   std::cout << "I wanna die" << std::endl;
//   return 0;
// }
