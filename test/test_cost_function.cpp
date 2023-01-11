#include <runtime/cost_function.h>
#include <gtest/gtest.h>
#include <templates/cost_function.h>

#include <Eigen/Dense>

using Eigen::Vector;
using Eigen::VectorXd;

TEST(Costcost, MeanSquareError) {
  for (int i = 0; i < 1000; i++) {
    VectorXd v = VectorXd::Random(10);
    VectorXd exp_v = VectorXd::Random(10);
    Vector<double, 10> v_t(v);
    Vector<double, 10> exp_v_t(exp_v);
    DeepPP::MeanSquareError<10> mean_sqr_error_t;

    EXPECT_NEAR(mean_sqr_error.function(v, exp_v),
                mean_sqr_error_t.cost(v_t, exp_v_t), 1e-7);
    EXPECT_TRUE(mean_sqr_error.derivative(v, exp_v).isApprox(
        mean_sqr_error_t.cost_der(v_t, exp_v_t)));
  }
}

TEST(Costcost, CategoricalCrossEntropy) {
  for (int i = 0; i < 1000; i++) {
    VectorXd v = VectorXd::Random(10);
    VectorXd exp_v = VectorXd::Random(10);
    Vector<double, 10> v_t(v);
    Vector<double, 10> exp_v_t(exp_v);
    DeepPP::CategoricalCrossEntropy<10> cat_cross_entropy_t;

    EXPECT_NEAR(cat_cross_entropy.function(v, exp_v),
                cat_cross_entropy_t.cost(v_t, exp_v_t), 1e-7);
    EXPECT_TRUE(cat_cross_entropy.derivative(v, exp_v).isApprox(
        cat_cross_entropy_t.cost_der(v_t, exp_v_t)));
  }
}
