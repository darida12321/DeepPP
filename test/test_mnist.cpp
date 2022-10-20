#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <image.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;


TEST(MnistTest, ReadData) {
    ImageSet image;

    MatrixXd w1 = MatrixXd::Random(28*28, 700);
    VectorXd b1 = VectorXd::Random(700);
    Layer layer1(w1, b1, relu, relu_derivative);
    MatrixXd w2 = MatrixXd::Random(700, 400);
    VectorXd b2 = VectorXd::Random(400);
    Layer layer2(w2, b2, relu, relu_derivative);
    MatrixXd w3 = MatrixXd::Random(400, 10);
    VectorXd b3 = VectorXd::Random(10);
    Layer layer3(w3, b3, relu, relu_derivative);
    Network network(std::vector<Layer>{layer1, layer2, layer3});

    // image.PrintImage(2);
    std::cout << network.forwardProp(image.GetImage(2)) << std::endl;

    std::cout << "This is a test" << std::endl;
}
