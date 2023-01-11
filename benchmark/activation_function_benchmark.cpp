
#include <runtime/activation_function.h>
#include <benchmark/benchmark.h>
// TODO restore
#include <templates/activation_function.h>
// #include <templates/linalg.h>

using Eigen::Vector;
using Eigen::VectorXd;

static void BM_Function_Sigmoid(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Sigmoid s;
  for (auto _ : state) {
    benchmark::DoNotOptimize(s.function(v));
    benchmark::DoNotOptimize(s.derivative(v));
  }
}

BENCHMARK(BM_Function_Sigmoid);

static void BM_Function_Template_Sigmoid(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  DeepPP::Sigmoid<128> s;
  for (auto _ : state) {
    benchmark::DoNotOptimize(s.activation(v));
    benchmark::DoNotOptimize(s.activation_der(v));
  }
}

BENCHMARK(BM_Function_Template_Sigmoid);

static void BM_Function_Softmax(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Softmax s;
  for (auto _ : state) {
    benchmark::DoNotOptimize(s.function(v));
    benchmark::DoNotOptimize(s.derivative(v));
  }
}

BENCHMARK(BM_Function_Softmax);

static void BM_Template_Softmax(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  DeepPP::Softmax<128> s;
  for (auto _ : state) {
    benchmark::DoNotOptimize(s.activation(v));
    benchmark::DoNotOptimize(s.activation_der(v));
  }
}

BENCHMARK(BM_Function_Template_Softmax);

static void BM_Function_Relu(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Relu r;
  for (auto _ : state) {
    benchmark::DoNotOptimize(r.function(v));
    benchmark::DoNotOptimize(r.derivative(v));
  }
}

BENCHMARK(BM_Function_Relu);

static void BM_Template_Relu(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  DeepPP::Relu<128> r;
  for (auto _ : state) {
    benchmark::DoNotOptimize(r.activation(v));
    benchmark::DoNotOptimize(r.activation_der(v));
  }
}

BENCHMARK(BM_Function_Template_Relu);

static void BM_Function_Linear(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Linear l;
  for (auto _ : state) {
    benchmark::DoNotOptimize(l.function(v));
    benchmark::DoNotOptimize(l.derivative(v));
  }
}

BENCHMARK(BM_Function_Linear);

static void BM_Template_Linear(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  DeepPP::Linear<128> l;
  for (auto _ : state) {
    benchmark::DoNotOptimize(l.activation(v));
    benchmark::DoNotOptimize(l.activation_der(v));
  }
}

BENCHMARK(BM_Function_Template_Linear);
