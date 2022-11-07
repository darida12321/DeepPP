#include <activation_function.h>
#include <gtest/gtest.h>
#include <templates/activation_function.h>
#include <templates/linalg.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(ActivationFunction, Sigmoid) {
  for (int i = 0; i < 1000; i++) {
    VectorXd v = VectorXd::Random(10);
    Vectord<10> v_t(v);
    Template::Sigmoid<10> sigmoid_t;

    EXPECT_TRUE(sigmoid.function(v).isApprox(sigmoid_t.function(v_t)));
    EXPECT_TRUE(sigmoid.derivative(v).isApprox(sigmoid_t.derivative(v_t)));
  }
}

TEST(ActivationFunction, Softmax) {
  for (int i = 0; i < 1000; i++) {
    VectorXd v = VectorXd::Random(10);
    Vectord<10> v_t(v);
    Template::Softmax<10> softmax_t;

    EXPECT_TRUE(softmax.function(v).isApprox(softmax_t.function(v_t)));
    EXPECT_TRUE(softmax.derivative(v).isApprox(softmax_t.derivative(v_t)));
  }
}

TEST(ActivationFunction, Relu) {
  for (int i = 0; i < 1000; i++) {
    VectorXd v = VectorXd::Random(10);
    Vectord<10> v_t(v);
    Template::Relu<10> relu_t;

    EXPECT_TRUE(relu.function(v).isApprox(relu_t.function(v_t)));
    EXPECT_TRUE(relu.derivative(v).isApprox(relu_t.derivative(v_t)));
  }
}

TEST(ActivationFunction, Linear) {
  for (int i = 0; i < 1000; i++) {
    VectorXd v = VectorXd::Random(10);
    Vectord<10> v_t(v);
    Template::Linear<10> linear_t;

    EXPECT_TRUE(linear.function(v).isApprox(linear_t.function(v_t)));
    EXPECT_TRUE(linear.derivative(v).isApprox(linear_t.derivative(v_t)));
  }
}