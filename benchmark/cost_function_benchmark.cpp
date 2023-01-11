#include <benchmark/benchmark.h>
#include <runtime/cost_function.h>
#include <templates/cost_function.h>

using Eigen::Vector;
using Eigen::VectorXd;

static void BM_Function_MeanSquareError(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  MeanSquareError mse;
  for (auto _ : state) {
    benchmark::DoNotOptimize(mse.function(v, v));
    benchmark::DoNotOptimize(mse.derivative(v, v));
  }
}

BENCHMARK(BM_Function_MeanSquareError);

static void BM_Function_Template_MeanSquareError(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::MeanSquareError<10> mse;
  for (auto _ : state) {
    benchmark::DoNotOptimize(mse.cost(v, v));
    benchmark::DoNotOptimize(mse.cost_der(v, v));
  }
}

BENCHMARK(BM_Function_Template_MeanSquareError);

static void BM_Function_CategoricalCrossEntropy(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  CategoricalCrossEntropy cce;
  for (auto _ : state) {
    benchmark::DoNotOptimize(cce.function(v, v));
    benchmark::DoNotOptimize(cce.derivative(v, v));
  }
}

BENCHMARK(BM_Function_CategoricalCrossEntropy);

static void BM_Function_Template_CategoricalCrossEntropy(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::CategoricalCrossEntropy<10> cce;
  for (auto _ : state) {
   benchmark::DoNotOptimize(cce.cost(v, v));
   benchmark::DoNotOptimize(cce.cost_der(v, v));
  }
}

BENCHMARK(BM_Function_Template_CategoricalCrossEntropy);
