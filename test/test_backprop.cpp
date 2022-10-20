#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(LayerBackwardPropTest, MediumNetwork){
    MatrixXd w1 = MatrixXd::Random(2, 2);
    VectorXd b1 = VectorXd::Random(2);
    Layer layer1(w1, b1, sigmoid, sigmoid_derivative);
    MatrixXd w2 = MatrixXd::Random(1, 2);
    VectorXd b2 = VectorXd::Random(1);
    Layer layer2(w2, b2, sigmoid, sigmoid_derivative);
    Network network(std::vector<Layer>{layer1, layer2});
    

    VectorXd input(2);
    input(0) = 0.1;
    input(1) = 0.8;
    VectorXd output(1);
    output(0) = 0.5;

    std::cout << network.forwardProp(input) << std::endl;
    std::cout << "------" << std::endl;
    network.trainOne(input, output, 10);
    network.trainOne(input, output, 10);
    network.trainOne(input, output, 10);
    network.trainOne(input, output, 10);
    network.trainOne(input, output, 10);
    network.trainOne(input, output, 10);
    network.trainOne(input, output, 10);
    network.trainOne(input, output, 10);
    std::cout << "------" << std::endl;
    std::cout << network.forwardProp(input) << std::endl;
}
