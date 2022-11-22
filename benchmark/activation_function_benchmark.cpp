
#include <runtime/activation_function.h>
#include <benchmark/benchmark.h>
// TODO restore
#include <templates/activation_function.h>
// #include <templates/linalg.h>

using Eigen::Vector;
using Eigen::VectorXd;

static void BM_Activation_Sigmoid(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Sigmoid s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Activation_Sigmoid);

static void BM_Activation_Template_Sigmoid(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  Template::Sigmoid<128> s;
  for (auto _ : state) {
    s.activation(v);
    s.activation_der(v);
  }
}

BENCHMARK(BM_Activation_Template_Sigmoid);

static void BM_Activation_Softmax(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Softmax s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Activation_Softmax);

static void BM_Activation_Template_Softmax(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  Template::Softmax<128> s;
  for (auto _ : state) {
    s.activation(v);
    s.activation_der(v);
  }
}

BENCHMARK(BM_Activation_Template_Softmax);

static void BM_Activation_Relu(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Relu r;
  for (auto _ : state) {
    r.function(v);
    r.derivative(v);
  }
}

BENCHMARK(BM_Activation_Relu);

static void BM_Activation_Template_Relu(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  Template::Relu<128> r;
  for (auto _ : state) {
    r.activation(v);
    r.activation_der(v);
  }
}

BENCHMARK(BM_Activation_Template_Relu);

static void BM_Activation_Linear(benchmark::State& state) {
  VectorXd v = VectorXd::Random(128);
  Linear l;
  for (auto _ : state) {
    l.function(v);
    l.derivative(v);
  }
}

BENCHMARK(BM_Activation_Linear);

static void BM_Activation_Template_Linear(benchmark::State& state) {
  Vector<double, 128> v = Vector<double, 128>::Random();
  Template::Linear<128> l;
  for (auto _ : state) {
    l.activation(v);
    l.activation_der(v);
  }
}

BENCHMARK(BM_Activation_Template_Linear);
