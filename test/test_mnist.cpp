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


int getPrediction(VectorXd out) {
    int maxIndex = 0;
    for (int i = 1; i < 10; i++) {
        if (out(i) > out(maxIndex)) {
            maxIndex = i;
        }
    }
    return maxIndex;
}
void printPrediction(VectorXd out) {
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(3);
    for (int i = 0; i < 10; i++) {
        std::cout << "  " << i << "   | ";
    }std::cout << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << out(i) << " | ";
    }std::cout << getPrediction(out);
}

// TEST(MnistTest, ReadData) {
//     ImageSet image;
//
//     MatrixXd w1 = MatrixXd::Random(700, 28*28);
//     VectorXd b1 = VectorXd::Random(700);
//     Layer layer1(w1, b1, sigmoid, sigmoid_derivative);
//     MatrixXd w2 = MatrixXd::Random(400, 700);
//     VectorXd b2 = VectorXd::Random(400);
//     Layer layer2(w2, b2, sigmoid, sigmoid_derivative);
//     MatrixXd w3 = MatrixXd::Random(10, 400);
//     VectorXd b3 = VectorXd::Random(10);
//     // Layer layer3(w3, b3, relu, relu_derivative);
//     Layer layer3(w3, b3, sigmoid, sigmoid_derivative);
//     Network network(std::vector<Layer>{layer1, layer2, layer3});
//
//
//     std::cout << "XXXXX TRAINING XXXXX" << std::endl;
//     for(int i = 0; i < 300; i++) {
//         int index = i%20;
//         network.trainOne(image.GetImage(index), image.GetLabel(index), 0.05);
//         std::cout << getPrediction(image.GetLabel(index)) << " ";
//         if (i%20 == 19) { std::cout << std::endl; }
//     }std::cout << std::endl;
//
//     // image.PrintImage(1000);
//     std::cout << "XXXXX THE PREDICTIONS XXXXX" << std::endl;
//     for (int i = 0; i < 10; i++) {
//         int index = i;
//         printPrediction(network.forwardProp(image.GetImage(index)));
//         std::cout << " Actual: " << getPrediction(image.GetLabel(index)) << std::endl;
//     }
//
//
//
//     std::cout << "This is a test" << std::endl;
// }
