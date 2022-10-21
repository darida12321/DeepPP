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
//     MatrixXd w1 = MatrixXd::Random(16, 28*28);
//     VectorXd b1 = VectorXd::Random(16);
//     Layer layer1(w1, b1, sigmoid, sigmoid_derivative);
//     MatrixXd w2 = MatrixXd::Random(16, 16);
//     VectorXd b2 = VectorXd::Random(16);
//     Layer layer2(w2, b2, sigmoid, sigmoid_derivative);
//     MatrixXd w3 = MatrixXd::Random(10, 16);
//     VectorXd b3 = VectorXd::Random(10);
//     Layer layer3(w3, b3, sigmoid, sigmoid_derivative);
//     Network network(std::vector<Layer>{layer1, layer2, layer3});
//
//     ImageSet image;
//
//     int training_size = 500;
//     for (int j = 0; j < 30; j++) {
//         std::vector<VectorXd> in1;
//         std::vector<VectorXd> out1;
//         for (int i = 0; i < training_size; i++) {
//             in1.push_back(image.GetImage(j*training_size + i));
//             out1.push_back(image.GetLabel(j*training_size + i));
//         }
//
//         double cost = network.getCost(in1, out1);
//         std::cout << cost << std::endl;
//         while (true) {
//             network.train(in1, out1, 5);
//             double cost = network.getCost(in1, out1);
//             std::cout << cost << std::endl;
//             if (cost < 0.85) {
//                 break;
//             }
//         }
//
//         for (int i = 0; i < 5; i++) {
//             int index = 50000+i;
//             printPrediction(network.forwardProp(image.GetImage(index)));
//             std::cout << " Actual: " << getPrediction(image.GetLabel(index)) << std::endl;
//         }
//     }
//
//     std::cout << "XXXXX THE PREDICTIONS XXXXX" << std::endl;
//     for (int i = 0; i < 10; i++) {
//         int index = 80+i;
//         printPrediction(network.forwardProp(image.GetImage(index)));
//         std::cout << " Actual: " << getPrediction(image.GetLabel(index)) << std::endl;
//     }
//     // image.PrintImage(1000);
// }
