
#include <benchmark/benchmark.h>

#include <templates/linalg.h>
#include <templates/activation_function.h>


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