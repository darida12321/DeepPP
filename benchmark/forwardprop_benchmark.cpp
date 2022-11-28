#include <benchmark/benchmark.h>

#include <runtime/activation_function.h>
#include <runtime/cost_function.h>
#include <runtime/network.h>

#include "forwardprop_benchmark.h"
// #include <templates/network.h>

#include <Eigen/Dense>
#include<vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector;

// static void BM_ForwardProp(benchmark::State& state) {
//   Network network({static_cast<int>(state.range(0)), 128, 128, 10}, {&relu, &relu, &softmax}, &cat_cross_entropy);

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(state.range(0))));
//   }
// }

// BENCHMARK(BM_ForwardProp)->RangeMultiplier(2)->Range(8, 1<<10);
BENCHMARK(BM_ForwardProp<Network>)->RangeMultiplier(2)->Range(8, 1<<10);

// void BM_ForwardProp_Template_8(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<8>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Relu>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(Vector<double, 8>::Random()));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_8);

// void BM_ForwardProp_Template_16(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<16>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Softmax>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(16)));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_16);

// void BM_ForwardProp_Template_32(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<32>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Softmax>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(32)));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_32);

// void BM_ForwardProp_Template_64(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<64>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Softmax>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(64)));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_64);

// void BM_ForwardProp_Template_128(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<128>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Softmax>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(128)));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_128);

// void BM_ForwardProp_Template_256(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<256>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Softmax>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(256)));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_256);

// void BM_ForwardProp_Template_512(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<512>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Softmax>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(512)));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_512);

// void BM_ForwardProp_Template_1024(benchmark::State& state) {
//   Template::Network<
//     Template::CategoricalCrossEntropy, Template::WeightRandom, Template::BiasZero,
//     Template::InputLayer<1024>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<128, Template::Relu>,
//     Template::Layer<10, Template::Softmax>
//   > network;

//   for (auto _ : state) {
//     benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(1024)));
//   }
// }

// BENCHMARK(BM_ForwardProp_Template_1024);

BENCHMARK(Template::BM_ForwardProp_Template<8, 128, 128, 10>)->Name("BM_ForwardProp_Template/8");
BENCHMARK(Template::BM_ForwardProp_Template<16, 128, 128, 10>)->Name("BM_ForwardProp_Template/16");
BENCHMARK(Template::BM_ForwardProp_Template<32, 128, 128, 10>)->Name("BM_ForwardProp_Template/32");
BENCHMARK(Template::BM_ForwardProp_Template<64, 128, 128, 10>)->Name("BM_ForwardProp_Template/64");
BENCHMARK(Template::BM_ForwardProp_Template<128, 128, 128, 10>)->Name("BM_ForwardProp_Template/128");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 10>)->Name("BM_ForwardProp_Template/256");
BENCHMARK(Template::BM_ForwardProp_Template<512, 128, 128, 10>)->Name("BM_ForwardProp_Template/512");
BENCHMARK(Template::BM_ForwardProp_Template<1024, 128, 128, 10>)->Name("BM_ForwardProp_Template/1024");
