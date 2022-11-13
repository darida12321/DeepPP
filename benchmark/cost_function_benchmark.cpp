#include <benchmark/benchmark.h>
#include <cost_function.h>
#include <templates/cost_function.h>

using Eigen::Vector;
using Eigen::VectorXd;

static void BM_MeanSquareError(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  MeanSquareError mse;
  for (auto _ : state) {
    mse.function(v, v);
    mse.derivative(v, v);
  }
}

BENCHMARK(BM_MeanSquareError);

static void BM_Template_MeanSquareError(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::MeanSquareError<10> mse;
  for (auto _ : state) {
    mse.cost(v, v);
    mse.cost_der(v, v);
  }
}

BENCHMARK(BM_Template_MeanSquareError);

static void BM_CategoricalCrossEntropy(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  CategoricalCrossEntropy cce;
  for (auto _ : state) {
    cce.function(v, v);
    cce.derivative(v, v);
  }
}

BENCHMARK(BM_CategoricalCrossEntropy);

static void BM_Template_CategoricalCrossEntropy(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::CategoricalCrossEntropy<10> cce;
  for (auto _ : state) {
    cce.cost(v, v);
    cce.cost_der(v, v);
  }
}

BENCHMARK(BM_Template_CategoricalCrossEntropy);
