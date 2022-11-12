#include <activation_function.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(ActivationFunction, Sigmoid) {
    VectorXd in; in << -2, -1, 0, 1, 2;
    VectorXd out; out << 0.119, 0.269, 0.5, 0.731, 0.881;
    
    EXPECT_TRUE(sigmoid.function(in).isApprox(out, 0.001));
}

TEST(ActivationFunction, Softmax) {
    VectorXd in; in << 1, 2, 3, 4;
    VectorXd out; out << 0.032, 0.087, 0.237, 0.644;

    EXPECT_TRUE(softmax.function(in).isApprox(out, 0.001));
}

TEST(ActivationFunction, Relu) {
    VectorXd in; in << -2, -1, 0, 1, 2;
    VectorXd out; out << 0, 0, 0, 1, 2;

    EXPECT_TRUE(relu.function(in).isApprox(out, 0.001))
}

TEST(ActivationFunction, Linear) {
    VectorXd in; in << -2, -1, 0, 1, 2;
    VectorXd out; out << -2, -1, 0, 1, 2;

    EXPECT_TRUE(linear.function(in).isApprox(out, 0.001));
}
