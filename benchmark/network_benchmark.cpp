#include <activation_function.h>
#include <benchmark/benchmark.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

// static void BM_SimpleNudge(benchmark::State& state) {
// 	for (auto _ : state) {
// 		MatrixXd w1 {{1}};
// 		VectorXd b1 {{0.5}};
// 		Layer layer1(w1, b1, linear, linear_derivative);
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
// 		Layer layer1(w1, b1, linear, linear_derivative);
//
// 		MatrixXd w2 {{2}};
// 		VectorXd b2 {{0.3}};
// 		Layer layer2(w2, b2, linear, linear_derivative);
//
// 		MatrixXd w3 {{0.2}};
// 		VectorXd b3 {{0.8}};
// 		Layer layer3(w3, b3, linear, linear_derivative);
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
