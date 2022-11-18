
#include <runtime/activation_function.h>
#include <benchmark/benchmark.h>
// TODO restore
#include <templates/activation_function.h>
// #include <templates/linalg.h>

using Eigen::Vector;
using Eigen::VectorXd;

static void BM_Sigmoid(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  Sigmoid s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Sigmoid);

static void BM_Template_Sigmoid(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::Sigmoid<10> s;
  for (auto _ : state) {
    s.activation(v);
    s.activation_der(v);
  }
}

BENCHMARK(BM_Template_Sigmoid);

static void BM_Softmax(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  Softmax s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Softmax);

static void BM_Template_Softmax(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::Softmax<10> s;
  for (auto _ : state) {
    s.activation(v);
    s.activation_der(v);
  }
}

BENCHMARK(BM_Template_Softmax);

static void BM_Relu(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  Relu r;
  for (auto _ : state) {
    r.function(v);
    r.derivative(v);
  }
}

BENCHMARK(BM_Relu);

static void BM_Template_Relu(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::Relu<10> r;
  for (auto _ : state) {
    r.activation(v);
    r.activation_der(v);
  }
}

BENCHMARK(BM_Template_Relu);

static void BM_Linear(benchmark::State& state) {
  VectorXd v = VectorXd::Random(10);
  Linear l;
  for (auto _ : state) {
    l.function(v);
    l.derivative(v);
  }
}

BENCHMARK(BM_Linear);

static void BM_Template_Linear(benchmark::State& state) {
  Vector<double, 10> v = Vector<double, 10>::Random();
  Template::Linear<10> l;
  for (auto _ : state) {
    l.activation(v);
    l.activation_der(v);
  }
}

BENCHMARK(BM_Template_Linear);
