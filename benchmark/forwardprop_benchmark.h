#include <benchmark/benchmark.h>

#include <runtime/network.h>
#include <templates/network.h>

#include <Eigen/Dense>
#include<vector>

using Eigen::VectorXd;
using Eigen::Vector;

template<class NetworkClass>
void BM_ForwardProp(benchmark::State& state) {
  NetworkClass network;

  for (auto _ : state) {
    benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(state.range(0))));
  }
}

template<>
void BM_ForwardProp<Network>(benchmark::State& state) {
  Network network({static_cast<int>(state.range(0)), 128, 128, 10}, {&relu, &relu, &softmax}, &cat_cross_entropy);

  for (auto _ : state) {
    benchmark::DoNotOptimize(network.forwardProp(VectorXd::Random(state.range(0))));
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
