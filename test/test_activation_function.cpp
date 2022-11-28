#include <runtime/activation_function.h>
#include <gtest/gtest.h>
#include <templates/activation_function.h>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::Vector;
using Eigen::VectorXd;

TEST(ActivationFunction, Sigmoid) {
  VectorXd in(5);
  in << -2, -1, 0, 1, 2;
  VectorXd out(5);
  out << 0.119, 0.269, 0.5, 0.731, 0.881;

  EXPECT_TRUE(sigmoid.function(in).isApprox(out, 0.001));
}

TEST(ActivationFunction, Softmax) {
  VectorXd in(4);
  in << 1, 2, 3, 4;
  VectorXd out(4);
  out << 0.032, 0.087, 0.237, 0.644;

  EXPECT_TRUE(softmax.function(in).isApprox(out, 0.001));
}

TEST(ActivationFunction, Relu) {
  VectorXd in(5);
  in << -2, -1, 0, 1, 2;
  VectorXd out(5);
  out << 0, 0, 0, 1, 2;

  EXPECT_TRUE(relu.function(in).isApprox(out, 0.001));
}

TEST(ActivationFunction, Linear) {
  VectorXd in(5);
  in << -2, -1, 0, 1, 2;
  VectorXd out(5);
  out << -2, -1, 0, 1, 2;

  EXPECT_TRUE(linear.function(in).isApprox(out, 0.001));
}

TEST(ActivationFunction, Sigmoid_Template) {
  for (int i = 0; i < 100; i++) {
    VectorXd in = VectorXd::Random(10);
    Vector<double, 10> in_t(in);
    Template::Sigmoid<10> sigmoid_t;

    EXPECT_TRUE(sigmoid.function(in).isApprox(sigmoid_t.activation(in_t)));
    EXPECT_TRUE(
        sigmoid.derivative(in).isApprox(sigmoid_t.activation_der(in_t)));
  }
}

TEST(ActivationFunction, Softmax_Template) {
  for (int i = 0; i < 100; i++) {
    VectorXd in = VectorXd::Random(10);
    Vector<double, 10> in_t(in);
    Template::Softmax<10> softmax_t;

    EXPECT_TRUE(softmax.function(in).isApprox(softmax_t.activation(in_t)));
    EXPECT_TRUE(
        softmax.derivative(in).isApprox(softmax_t.activation_der(in_t)));
  }
}

TEST(ActivationFunction, Relu_Template) {
  for (int i = 0; i < 100; i++) {
    VectorXd in = VectorXd::Random(10);
    Vector<double, 10> in_t(in);
    Template::Relu<10> relu_t;

    EXPECT_TRUE(relu.function(in).isApprox(relu_t.activation(in_t)));
    EXPECT_TRUE(relu.derivative(in).isApprox(relu_t.activation_der(in_t)));
  }
}

TEST(ActivationFunction, Linear_Template) {
  for (int i = 0; i < 100; i++) {
    VectorXd in = VectorXd::Random(10);
    Vector<double, 10> in_t(in);
    Template::Linear<10> linear_t;

    EXPECT_TRUE(linear.function(in).isApprox(linear_t.activation(in_t)));
    EXPECT_TRUE(linear.derivative(in).isApprox(linear_t.activation_der(in_t)));
  }
}
