#include <gtest/gtest.h>
#include <templates/network.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::Matrix;
using Eigen::Vector;

// TODO use is_approx
namespace Template {
  TEST(NetworkTemplate, SoftMax) {
    // Create 2-2-2 neural network
    Matrix<double, 2, 2> w1{{1, 2}, {1, 4}};
    Vector<double, 2> b1{9, -1};
    Matrix<double, 2, 2> w2{{1, 2}, {1, 4}};
    Vector<double, 2> b2{9, -1};

    Network<MeanSquareError, Layer<2, 2, Linear>, Layer<2, 2, Softmax>> network;
    network.setWeights(w1, w2);
    network.setBiases(b1, b2);

    // Check forwardpropogation value
    Vector<double, 2> in1{2, 4};
    Vector<double, 2> in2{1, 1};

    Vector<double, 2> out1 = network.forwardProp(in1);
    Vector<double, 2> out2 = network.forwardProp(in2);

    EXPECT_NEAR(out1(0), 0, 0.001);
    EXPECT_NEAR(out1(1), 1, 0.001);
    EXPECT_NEAR(out2(0), 0.8807, 0.001);
    EXPECT_NEAR(out2(1), 0.1192, 0.001);

    // Check backpropogation value
    Vector<double, 2> t_in1{1.0, 1.0};
    Vector<double, 2> t_out1{0.6, 0.4};
    Vector<double, 2> t_in2{3.0, 5.0};
    Vector<double, 2> t_out2{0.2, 0.8};
    std::vector<Vector<double, 2>> input{t_in1, t_in2};
    std::vector<Vector<double, 2>> output{t_out1, t_out2};
    network.train(input, output, 1);

    Matrix<double, 2, 2> w1_final{{1, 2}, {1.059, 4.058}};
    Vector<double, 2> b1_final{9, -0.941};
    Matrix<double, 2, 2> w2_final{{0.646, 1.882}, {1.354, 4.118}};
    Vector<double, 2> b2_final{8.971, -0.971};

    EXPECT_TRUE(w1_final.isApprox(network.getWeight<0>(), 0.001));
    EXPECT_TRUE(w2_final.isApprox(network.getWeight<1>(), 0.001));
    EXPECT_TRUE(b1_final.isApprox(network.getBias<0>(), 0.001));
    EXPECT_TRUE(b2_final.isApprox(network.getBias<1>(), 0.001));
  }
}