#include <activation_function.h>
#include <gtest/gtest.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <typeinfo>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(LayerBackPropTest, Relu) {
  // Create 1-1 neural network
  MatrixXd w1(2, 2); w1 << 1, 1, 1, 1;
  VectorXd b1(2); b1 << 1, 1;
  Network network(std::vector<MatrixXd>{w1, w1}, std::vector<VectorXd>{b1, b1},
      relu, relu_derivative);

  // Create example data point
  VectorXd in1(2); in1 << 1.0, 1.0;
  VectorXd out1(2); out1 << 3.0, 3.0;
  std::vector<VectorXd> in{in1};
  std::vector<VectorXd> out{out1};

  // Train the network
  network.train(in, out, 1);

  // Expect output
  EXPECT_NEAR(network.getWeights()[0](0, 0), -7, 0.001);
  EXPECT_NEAR(network.getWeights()[0](1, 0), -7, 0.001);
  EXPECT_NEAR(network.getWeights()[0](0, 1), -7, 0.001);
  EXPECT_NEAR(network.getWeights()[0](1, 1), -7, 0.001);
  EXPECT_NEAR(network.getBiases()[0](0), -7, 0.001);
  EXPECT_NEAR(network.getBiases()[0](1), -7, 0.001);
  EXPECT_NEAR(network.getWeights()[1](0, 0), -11, 0.001);
  EXPECT_NEAR(network.getWeights()[1](1, 0), -11, 0.001);
  EXPECT_NEAR(network.getWeights()[1](0, 1), -11, 0.001);
  EXPECT_NEAR(network.getWeights()[1](1, 1), -11, 0.001);
  EXPECT_NEAR(network.getBiases()[1](0), -3, 0.001);
  EXPECT_NEAR(network.getBiases()[1](1), -3, 0.001);
}

//
// TEST(LayerBackPropTest, DeepNudge){
//     // Create 1-1-1 neural network
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
//     VectorXd in1{{0.2}};
//     VectorXd out1{{0.8}};
//     std::vector<VectorXd> in{in1};
//     std::vector<VectorXd> out{out1};
//
//     // Train the network
//     network.train(in, out, 1);
//     ASSERT_LE(abs(network.getLayer(0).getWeights()(0,0) - 0.9456), 0.001); //
//     TODO check ASSERT_LE(abs(network.getLayer(0).getBias()(0) - 0.228),
//     0.001); // TODO check ASSERT_LE(abs(network.getLayer(1).getWeights()(0,0)
//     - 1.904), 0.001); ASSERT_LE(abs(network.getLayer(1).getBias()(0) -
//     0.164), 0.001); // TODO check
//     ASSERT_LE(abs(network.getLayer(2).getWeights()(0,0) - -0.956), 0.001);
//     ASSERT_LE(abs(network.getLayer(2).getBias()(0) - 0.12), 0.001);
// }
//
// TEST(LayerBackPropTest, MultiNeuronNudge){
//     // Create 1-1-1 neural network
//     MatrixXd w1(2, 2); w1 << 1, 0.7, 0.5, -1;
//     VectorXd b1(2); b1 << 0.5, 0;
//     Layer layer1(w1, b1, linear, linear_derivative);
//     MatrixXd w2(2, 2); w2 << 0.8, -2, 1, 1;
//     VectorXd b2(2); b2 << 1, -0.5;
//     Layer layer2(w2, b2, linear, linear_derivative);
//     MatrixXd w3(2, 2); w3 << 0.5, -1, 0, 1;
//     VectorXd b3(2); b3 << 0, 1;
//     Layer layer3(w3, b3, linear, linear_derivative);
//     Network network(std::vector<Layer>{layer1, layer2, layer3});
//
//     // Create example data point
//     VectorXd in1{{1, 0.5}};
//     VectorXd out1{{3, 2}};
//     std::vector<VectorXd> in{in1};
//     std::vector<VectorXd> out{out1};
//
//     // Train the network
//     network.train(in, out, 1);
//
//     // Layer 2 bias
//     EXPECT_NEAR(network.getLayer(2).getBias()(0), 6.22, 0.001);
//     EXPECT_NEAR(network.getLayer(2).getBias()(1), 0.3, 0.001);
//     // Layer 2 weights
//     EXPECT_NEAR(network.getLayer(2).getWeights()(0, 0), 15.9256, 0.001);
//     EXPECT_NEAR(network.getLayer(2).getWeights()(1, 0), -1.736, 0.001);
//     EXPECT_NEAR(network.getLayer(2).getWeights()(0, 1), 7.397, 0.001);
//     EXPECT_NEAR(network.getLayer(2).getWeights()(1, 1), 0.055, 0.001);
//
//     // Layer 1 bias
//     EXPECT_NEAR(network.getLayer(1).getBias()(0), 4.110, 0.001);
//     EXPECT_NEAR(network.getLayer(1).getBias()(1), -7.420, 0.001);
//     // Layer 1 weights
//     EXPECT_NEAR(network.getLayer(1).getWeights()(0, 0), 6.554, 0.001);
//     EXPECT_NEAR(network.getLayer(1).getWeights()(1, 0), -11.802, 0.001);
//     EXPECT_NEAR(network.getLayer(1).getWeights()(0, 1), -2, 0.001);
//     EXPECT_NEAR(network.getLayer(1).getWeights()(1, 1), 1, 0.001);
// }
//
