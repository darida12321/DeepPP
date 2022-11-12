#include <benchmark/benchmark.h>
#include <cost_function.h>
// TODO restore
// #include <templates/cost_function.h>
// #include <templates/linalg.h>

// static void BM_MeanSquareError(benchmark::State& state) {
//   VectorXd v = VectorXd::Random(10);
//   MeanSquareError mse;
//   for (auto _ : state) {
//     mse.function(v, v);
//     mse.derivative(v, v);
//   }
// }
//
// BENCHMARK(BM_MeanSquareError);
//
// static void BM_Template_MeanSquareError(benchmark::State& state) {
//   Vectord<10> v = Vectord<10>::Random();
//   Template::MeanSquareError<10> mse;
//   for (auto _ : state) {
//     mse.function(v, v);
//     mse.derivative(v, v);
//   }
// }
//
// BENCHMARK(BM_Template_MeanSquareError);
//
// static void BM_CategoricalCrossEntropy(benchmark::State& state) {
//   VectorXd v = VectorXd::Random(10);
//   CategoricalCrossEntropy cce;
//   for (auto _ : state) {
//     cce.function(v, v);
//     cce.derivative(v, v);
//   }
// }
//
// BENCHMARK(BM_CategoricalCrossEntropy);
//
// static void BM_Template_CategoricalCrossEntropy(benchmark::State& state) {
//   Vectord<10> v = Vectord<10>::Random();
//   Template::CategoricalCrossEntropy<10> cce;
//   for (auto _ : state) {
//     cce.function(v, v);
//     cce.derivative(v, v);
//   }
// }
//
// BENCHMARK(BM_Template_CategoricalCrossEntropy);
