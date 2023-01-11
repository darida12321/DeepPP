#include <benchmark/benchmark.h>

#include <runtime/network.h>
#include <templates/network.h>

#include <Eigen/Dense>
#include<vector>

using Eigen::VectorXd;
using Eigen::Vector;

template<class NetworkClass>
void BM_InputSize(benchmark::State& state) {
  NetworkClass network;

  for (auto _ : state) {
    benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(state.range(0))));
  }
}


template<>
void BM_InputSize<Network>(benchmark::State& state) {
  // Network* sample[10];
  // for (int i = 0; i < 10; i++) {
  //   Network *network = new Network({static_cast<int>(state.range(0)), 128, 128, 10}, {&relu, &relu, &softmax}, &cat_cross_entropy);
  //   sample[i] = network;
  // }

  Network network({static_cast<int>(state.range(0)), 128, 128, 10}, {&relu, &relu, &softmax}, &cat_cross_entropy);

  for (auto _ : state) {
    benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(state.range(0))));
  }
}

template<class NetworkClass>
void BM_LayerDepth(benchmark::State& state) {
  NetworkClass network;

  for (auto _ : state) {
    benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(state.range(0))));
  }
}

template<>
void BM_LayerDepth<Network>(benchmark::State& state) {
  std::vector<int> layers;
  std::vector<ActivationFunction*> funcs;

  layers.push_back(256);
  for (int i = 0; i < static_cast<int>(state.range(0)); i++) {
    funcs.push_back(&relu);
    layers.push_back(128);
  }

  layers.push_back(10);
  funcs.push_back(&softmax);

  Network network(layers, funcs, &cat_cross_entropy);
  
  for (auto _ : state) {
    benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(256)));
  }
}


namespace Template {
  template<size_t In, size_t... Outs>
  void BM_ForwardProp_Template(benchmark::State& state) {
    Network<
      CategoricalCrossEntropy, WeightRandom, BiasZero,
      InputLayer<In>,
      Layer<Outs, Relu>...
    > network;

    for (auto _ : state) {
      benchmark::DoNotOptimize(network.forwardProp(Vector<double, In>::Random()));
    }
  }
}
