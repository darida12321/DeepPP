
#include <activation_function.h>
#include <benchmark/benchmark.h>
#include <templates/activation_function.h>
#include <templates/linalg.h>

static void BM_Sigmoid(benchmark::State& state) {
  VectorXd v(1);
  v << 1;
  Sigmoid s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Sigmoid);

static void BM_Template_Sigmoid(benchmark::State& state) {
  Vectord<1> v;
  v << 1;
  Template::Sigmoid<1> s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Template_Sigmoid);

static void BM_Softmax(benchmark::State& state) {
  VectorXd v(1);
  v << 1;
  Softmax s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Softmax);

static void BM_Template_Softmax(benchmark::State& state) {
  Vectord<1> v;
  v << 1;
  Template::Softmax<1> s;
  for (auto _ : state) {
    s.function(v);
    s.derivative(v);
  }
}

BENCHMARK(BM_Template_Softmax);

static void BM_Relu(benchmark::State& state) {
  VectorXd v(1);
  v << 1;
  Relu r;
  for (auto _ : state) {
    r.function(v);
    r.derivative(v);
  }
}

BENCHMARK(BM_Relu);

static void BM_Template_Relu(benchmark::State& state) {
  Vectord<1> v;
  v << 1;
  Template::Relu<1> r;
  for (auto _ : state) {
    r.function(v);
    r.derivative(v);
  }
}

BENCHMARK(BM_Template_Relu);

static void BM_Linear(benchmark::State& state) {
  VectorXd v(1);
  v << 1;
  Linear l;
  for (auto _ : state) {
    l.function(v);
    l.derivative(v);
  }
}

BENCHMARK(BM_Linear);

static void BM_Template_Linear(benchmark::State& state) {
  Vectord<1> v;
  v << 1;
  Template::Linear<1> l;
  for (auto _ : state) {
    l.function(v);
    l.derivative(v);
  }
}

BENCHMARK(BM_Template_Linear);
