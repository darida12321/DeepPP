#include <activation_function.h>
#include <benchmark/benchmark.h>
#include <cost_function.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

static void BM_MultiInput(benchmark::State& state) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                  std::vector<ActivationFunction*>{&relu, &relu},
                  &mean_sqr_error);

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  for (auto _ : state) {
    network.train(input, output, 1);
  }
}

BENCHMARK(BM_MultiInput);

static void BM_Relu(benchmark::State& state) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                  std::vector<ActivationFunction*>{&relu, &relu},
                  &mean_sqr_error);

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  for (auto _ : state) {
    network.train(input, output, 1);
  }
}

BENCHMARK(BM_Relu);

static void BM_Sigmoid(benchmark::State& state) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                  std::vector<ActivationFunction*>{&sigmoid, &sigmoid},
                  &mean_sqr_error);

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  for (auto _ : state) {
    network.train(input, output, 1);
  }
}

BENCHMARK(BM_Sigmoid);

// static void BM_Tanh(benchmark::State& state) {
// 	// Create 1-1 neural network
// 	MatrixXd w(2, 2); w << 1, 1, 1, 1;
// 	VectorXd b(2); b << 1, 1;
// 	Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b,
// b}, 			std::vector<std::function<VectorXd(VectorXd)>>{tanh,
// tanh},
// std::vector<std::function<VectorXd(VectorXd)>>{tanhDerivative,
// tanhDerivative});

// 	// Create example data point
// 	VectorXd in1(2); in1 << 1.0, 1.0;
// 	VectorXd out1(2); out1 << 3.0, 3.0;
// 	VectorXd in2(2); in2 << 3.0, 5.0;
// 	VectorXd out2(2); out2 << 6.0, 8.0;
// 	std::vector<VectorXd> input{in1, in2};
// 	std::vector<VectorXd> output{out1, out2};

// 	// Train the network
// 	for (auto _ : state) {
// 		network.train(input, output, 1);
// 	}
// }

// BENCHMARK(BM_Tanh);

// static void BM_SimpleNudge(benchmark::State& state) {
// 	for (auto _ : state) {
// 		MatrixXd w1 {{1}};
// 		VectorXd b1 {{0.5}};
// 		Layer layer1(w1, b1, linear, linearDerivative);
// 		Network network({layer1});
//
// 		VectorXd in1 {{0.2}};
// 		VectorXd out1{{0.8}};
// 		std::vector<VectorXd> in {in1};
// 		std::vector<VectorXd> out {out1};
//
// 		network.train(in, out, 1);
// 	}
// }
//
// BENCHMARK(BM_SimpleNudge);
//
// static void BM_DeepNudge(benchmark::State& state) {
// 	for (auto _ : state) {
// 		MatrixXd w1 {{1}};
// 		VectorXd	b1 {{0.5}};
// 		Layer layer1(w1, b1, linear, linearDerivative);
//
// 		MatrixXd w2 {{2}};
// 		VectorXd b2 {{0.3}};
// 		Layer layer2(w2, b2, linear, linearDerivative);
//
// 		MatrixXd w3 {{0.2}};
// 		VectorXd b3 {{0.8}};
// 		Layer layer3(w3, b3, linear, linearDerivative);
//
// 		Network network({layer1, layer2, layer3});
//
// 		VectorXd in1 {{0.2}};
// 		VectorXd out1 {{0.8}};
// 		std::vector<VectorXd> in {in1};
// 		std::vector<VectorXd> out {out1};
//
// 		network.train(in, out, 1);
// 	}
// }
//
// BENCHMARK(BM_DeepNudge);
