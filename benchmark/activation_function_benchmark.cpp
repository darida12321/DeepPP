
#include <activation_function.h>
#include <benchmark/benchmark.h>
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