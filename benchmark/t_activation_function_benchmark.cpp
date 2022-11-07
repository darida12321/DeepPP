
#include <benchmark/benchmark.h>
#include <templates/activation_function.h>
#include <templates/linalg.h>

namespace Template {
  static void BM_TemplatizedSigmoid(benchmark::State& state) {
    Vectord<1> v;
    v << 1;
    Sigmoid<1> s;
    for (auto _ : state) {
      s.function(v);
      s.derivative(v);
    }
  }

  BENCHMARK(BM_TemplatizedSigmoid);

  static void BM_TemplatizedSoftmax(benchmark::State& state) {
    Vectord<1> v;
    v << 1;
    Softmax<1> s;
    for (auto _ : state) {
      s.function(v);
      s.derivative(v);
    }
  }

  BENCHMARK(BM_TemplatizedSoftmax);

  static void BM_TemplatizedRelu(benchmark::State& state) {
    Vectord<1> v;
    v << 1;
    Relu<1> r;
    for (auto _ : state) {
      r.function(v);
      r.derivative(v);
    }
  }

  BENCHMARK(BM_TemplatizedRelu);

  static void BM_TemplatizedLinear(benchmark::State& state) {
    Vectord<1> v;
    v << 1;
    Linear<1> l;
    for (auto _ : state) {
      l.function(v);
      l.derivative(v);
    }
  }

  BENCHMARK(BM_TemplatizedLinear);
}
