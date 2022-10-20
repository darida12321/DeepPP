#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::vector<VectorXd> createDataVector(std::vector<double> data) {
    std::vector<VectorXd> out;
    for (double d : data) {
        VectorXd vec(1);
        vec << d;
        out.push_back(vec);
    }
    return out;
}

TEST(LayerBackwardPropTest, SmallNetwork){
    // Create 1-1 neural network
    MatrixXd w1 = MatrixXd::Random(1, 1);
    VectorXd b1 = VectorXd::Random(1);
    Layer layer1(w1, b1, relu, relu_derivative);
    Network network(std::vector<Layer>{layer1});
    
    // Process is f(x) = x+0.3
    auto input  = createDataVector(std::vector<double>{0.2, 0.5, 0.6});
    auto output = createDataVector(std::vector<double>{0.5, 0.8, 0.9});

    // Get the number of iterations before reaching correct conclusion
    int iterations = 0;
    double error = abs(network.forwardProp(input[2])(0) - output[2](0));
    while (iterations < 100 && error > 0.01) {
        network.trainOne(input[0], output[0], 1);
        network.trainOne(input[1], output[1], 1);
        error = abs(network.forwardProp(input[2])(0) - output[2](0));
        iterations++;
    }

    std::cout << "It took " << iterations  << " iterations." << std::endl;
    ASSERT_LE(iterations, 20);
}

TEST(LayerBackwardPropTest, MediumNetwork){
    MatrixXd w1 = MatrixXd::Random(2, 2);
    VectorXd b1 = VectorXd::Random(2);
    Layer layer1(w1, b1, sigmoid, sigmoid_derivative);
    MatrixXd w2 = MatrixXd::Random(1, 2);
    VectorXd b2 = VectorXd::Random(1);
    Layer layer2(w2, b2, sigmoid, sigmoid_derivative);
    Network network(std::vector<Layer>{layer1, layer2});
    
    // Function is f(x, y) = 2x - y - 0.2
    std::vector<VectorXd> input;
    input.push_back(VectorXd(2)); input[0](0) = 0.4; input[0](1) = 0.4;
    input.push_back(VectorXd(2)); input[1](0) = 0.6; input[1](1) = 0.6;
    input.push_back(VectorXd(2)); input[2](0) = 0.5; input[2](1) = 0.4;

    std::vector<VectorXd> output;
    output.push_back(VectorXd(1)); output[0](0) = 0.2;
    output.push_back(VectorXd(1)); output[1](0) = 0.4;
    output.push_back(VectorXd(1)); output[2](0) = 0.9;


    // Get the number of iterations before reaching correct conclusion
    int iterations = 0;
    double error = abs(network.forwardProp(input[2])(0) - output[2](0));
    while (iterations < 100 && error > 0.01) {
        network.trainOne(input[0], output[0], 1);
        network.trainOne(input[1], output[1], 1);
        error = abs(network.forwardProp(input[2])(0) - output[2](0));
        iterations++;
    }

    std::cout << "Second took " << iterations  << " iterations." << std::endl;
    // ASSERT_LE(iterations, 20);
}
