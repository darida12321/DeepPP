#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
#include <typeinfo>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(LayerBackPropTest, SimpleNudge){
    // Create 1-1 neural network
    MatrixXd w1{{1}};
    VectorXd b1{{0.5}};
    Layer layer1(w1, b1, linear, linear_derivative);
    Network network(std::vector<Layer>{layer1});
    
    // Create example data point
    VectorXd in1{{0.2}};
    VectorXd out1{{0.8}};
    std::vector<VectorXd> in{in1};
    std::vector<VectorXd> out{out1};

    // Train the network
    network.train(in, out, 1);
    ASSERT_LE(abs(network.getLayer(0).getWeights()(0,0) - 1.04), 0.001);
    ASSERT_LE(abs(network.getLayer(0).getBias()(0) - 0.7), 0.001);
}

TEST(LayerBackPropTest, DeepNudge){
    // Create 1-1-1 neural network
    MatrixXd w1{{1}};
    VectorXd b1{{0.5}};
    Layer layer1(w1, b1, linear, linear_derivative);
    MatrixXd w2{{2}};
    VectorXd b2{{0.3}};
    Layer layer2(w2, b2, linear, linear_derivative);
    MatrixXd w3{{0.2}};
    VectorXd b3{{0.8}};
    Layer layer3(w3, b3, linear, linear_derivative);
    Network network(std::vector<Layer>{layer1, layer2, layer3});
    
    // Create example data point
    VectorXd in1{{0.2}};
    VectorXd out1{{0.8}};
    std::vector<VectorXd> in{in1};
    std::vector<VectorXd> out{out1};

    // Train the network
    network.train(in, out, 1);
    ASSERT_LE(abs(network.getLayer(0).getWeights()(0,0) - 0.9456), 0.001); // TODO check
    ASSERT_LE(abs(network.getLayer(0).getBias()(0) - 0.228), 0.001); // TODO check
    ASSERT_LE(abs(network.getLayer(1).getWeights()(0,0) - 1.904), 0.001);
    ASSERT_LE(abs(network.getLayer(1).getBias()(0) - 0.164), 0.001); // TODO check
    ASSERT_LE(abs(network.getLayer(2).getWeights()(0,0) - -0.956), 0.001);
    ASSERT_LE(abs(network.getLayer(2).getBias()(0) - 0.12), 0.001);
}


