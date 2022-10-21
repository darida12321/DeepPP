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

// TEST(LayerBackPropTest, SimpleNudge){
//     // Create 1-1 neural network
//     MatrixXd w1{{1}};
//     VectorXd b1{{0.5}};
//     Layer layer1(w1, b1, linear, linear_derivative);
//     Network network(std::vector<Layer>{layer1});
//     
//     // Create example data point
//     VectorXd in{{0.2}};
//     VectorXd out{{0.8}};
//
//     // Train the network
//     network.trainOne(in, out, 1);
//     ASSERT_LE(abs(network.getLayer(0).getWeights()(0,0) - 1.04), 0.001);
//     ASSERT_LE(abs(network.getLayer(0).getBias()(0) - 0.7), 0.001);
// }
//
// TEST(LayerBackPropTest, DeepNudge){
//     // Create 1-1 neural network
//     MatrixXd w1{{1}};
//     VectorXd b1{{0.5}};
//     Layer layer1(w1, b1, linear, linear_derivative);
//     MatrixXd w2{{2}};
//     VectorXd b2{{0.3}};
//     Layer layer2(w2, b2, linear, linear_derivative);
//     MatrixXd w3{{0.2}};
//     VectorXd b3{{0.8}};
//     Layer layer3(w3, b3, linear, linear_derivative);
//     Network network(std::vector<Layer>{layer1, layer2, layer3});
//     
//     // Create example data point
//     VectorXd in{{0.2}};
//     VectorXd out{{0.8}};
//
//     // Train the network
//     network.trainOne(in, out, 1);
//     ASSERT_LE(abs(network.getLayer(0).getWeights()(0,0) - 0.9456), 0.001); // TODO check
//     ASSERT_LE(abs(network.getLayer(0).getBias()(0) - 0.228), 0.001); // TODO check
//     ASSERT_LE(abs(network.getLayer(1).getWeights()(0,0) - 1.904), 0.001);
//     ASSERT_LE(abs(network.getLayer(1).getBias()(0) - 0.164), 0.001); // TODO check
//     ASSERT_LE(abs(network.getLayer(2).getWeights()(0,0) - -0.956), 0.001);
//     ASSERT_LE(abs(network.getLayer(2).getBias()(0) - 0.12), 0.001);
// }


// std::vector<VectorXd> createDataVector(std::vector<double> data) {
//     std::vector<VectorXd> out;
//     for (double d : data) {
//         VectorXd vec(1);
//         vec << d;
//         out.push_back(vec);
//     }
//     return out;
// }

// TEST(LayerBackwardPropTest, SmallNetwork){
//     // Create 1-1 neural network
//     MatrixXd w1 = MatrixXd::Random(1, 1);
//     VectorXd b1 = VectorXd::Random(1);
//     Layer layer1(w1, b1, relu, relu_derivative);
//     Network network(std::vector<Layer>{layer1});
//     
//     // Process is f(x) = x+0.3
//     auto input  = createDataVector(std::vector<double>{0.2, 0.5, 0.6});
//     auto output = createDataVector(std::vector<double>{0.5, 0.8, 0.9});
//
//     // Get the number of iterations before reaching correct conclusion
//     int iterations = 0;
//     double error = abs(network.forwardProp(input[2])(0) - output[2](0));
//     while (iterations < 100 && error > 0.01) {
//         network.trainOne(input[0], output[0], 1);
//         network.trainOne(input[1], output[1], 1);
//         error = abs(network.forwardProp(input[2])(0) - output[2](0));
//         iterations++;
//     }
//
//     std::cout << "It took " << iterations  << " iterations." << std::endl;
//     ASSERT_LE(iterations, 20);
// }

// TEST(LayerBackwardPropTest, MediumNetwork){
//     MatrixXd w1 = MatrixXd::Random(2, 2);
//     VectorXd b1 = VectorXd::Random(2);
//     Layer layer1(w1, b1, sigmoid, sigmoid_derivative);
//     MatrixXd w2 = MatrixXd::Random(1, 2);
//     VectorXd b2 = VectorXd::Random(1);
//     Layer layer2(w2, b2, sigmoid, sigmoid_derivative);
//     Network network(std::vector<Layer>{layer1, layer2});
//     
//     // Function is f(x, y) = 2x - y - 0.2
//     std::vector<VectorXd> input;
//     input.push_back(VectorXd(2)); input[0](0) = 0.4; input[0](1) = 0.4;
//     input.push_back(VectorXd(2)); input[1](0) = 0.4; input[1](1) = 0.5;
//     input.push_back(VectorXd(2)); input[2](0) = 0.4; input[2](1) = 0.6;
//     input.push_back(VectorXd(2)); input[3](0) = 0.5; input[3](1) = 0.4;
//     input.push_back(VectorXd(2)); input[4](0) = 0.5; input[4](1) = 0.6;
//     input.push_back(VectorXd(2)); input[5](0) = 0.5; input[5](1) = 0.5;
//     input.push_back(VectorXd(2)); input[6](0) = 0.6; input[6](1) = 0.4;
//     input.push_back(VectorXd(2)); input[7](0) = 0.6; input[7](1) = 0.5;
//     input.push_back(VectorXd(2)); input[8](0) = 0.6; input[8](1) = 0.6;
//
//     std::vector<VectorXd> output;
//     output.push_back(VectorXd(1)); output[0](0) = 0.2;
//     output.push_back(VectorXd(1)); output[1](0) = 0.1;
//     output.push_back(VectorXd(1)); output[2](0) = 0.0;
//     output.push_back(VectorXd(1)); output[3](0) = 0.4;
//     output.push_back(VectorXd(1)); output[4](0) = 0.3;
//     output.push_back(VectorXd(1)); output[5](0) = 0.2;
//     output.push_back(VectorXd(1)); output[6](0) = 0.6;
//     output.push_back(VectorXd(1)); output[7](0) = 0.5;
//     output.push_back(VectorXd(1)); output[8](0) = 0.4;
//
//
//     // Get the number of iterations before reaching correct conclusion
//     int iterations = 0;
//     int train_i = 4;
//     double error = abs(network.forwardProp(input[2])(0) - output[2](0));
//     while (iterations < 100 && error > 0.01) {
//         for (int i = 0; i < input.size(); i++) {
//             if (i == train_i) {
//                 continue;
//             }
//             network.trainOne(input[i], output[i], 0.2);
//         }
//         std::cout << "nn(0.5, 0.4) = " << network.forwardProp(input[train_i]) << std::endl;
//         error = abs(network.forwardProp(input[train_i])(0) - output[train_i](0));
//         iterations++;
//     }
//
//     std::cout << "Second took " << iterations  << " iterations." << std::endl;
//     // ASSERT_LE(iterations, 20);
// }
