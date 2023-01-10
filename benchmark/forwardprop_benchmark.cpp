#include <benchmark/benchmark.h>

#include <runtime/activation_function.h>
#include <runtime/cost_function.h>
#include <runtime/network.h>

#include "forwardprop_benchmark.h"

#include <Eigen/Dense>
#include<vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector;

BENCHMARK(BM_InputSize<Network>)->Name("BM_InputSize")->DenseRange(64, 1<<10, 64);
BENCHMARK(BM_LayerDepth<Network>)->Name("BM_LayerDepth")->DenseRange(1, 10);

// BENCHMARK(Template::BM_InputSize_Template<8, 128, 128, 10>)->Name("BM_InputSize_Template/8");
// BENCHMARK(Template::BM_InputSize_Template<16, 128, 128, 10>)->Name("BM_InputSize_Template/16");
// BENCHMARK(Template::BM_InputSize_Template<32, 128, 128, 10>)->Name("BM_InputSize_Template/32");
// BENCHMARK(Template::BM_InputSize_Template<64, 128, 128, 10>)->Name("BM_InputSize_Template/64");
// BENCHMARK(Template::BM_InputSize_Template<128, 128, 128, 10>)->Name("BM_InputSize_Template/128");
// BENCHMARK(Template::BM_InputSize_Template<256, 128, 128, 10>)->Name("BM_InputSize_Template/256");
// BENCHMARK(Template::BM_InputSize_Template<512, 128, 128, 10>)->Name("BM_InputSize_Template/512");
// BENCHMARK(Template::BM_InputSize_Template<1024, 128, 128, 10>)->Name("BM_InputSize_Template/1024");

// for (int i = 32; i <= 1024; i += 32) {
// 	BENCHMARK(Template::BM_InputSize_Template<i, 128, 128, 10>)->Name("BM_InputSize_Template/" + i);	
// }

//Templatized benchmarks across input vector size
BENCHMARK(Template::BM_ForwardProp_Template<64, 128, 128, 10>)->Name("BM_InputSize_Template/64");
BENCHMARK(Template::BM_ForwardProp_Template<128, 128, 128, 10>)->Name("BM_InputSize_Template/128");
BENCHMARK(Template::BM_ForwardProp_Template<192, 128, 128, 10>)->Name("BM_InputSize_Template/192");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 10>)->Name("BM_InputSize_Template/256");
BENCHMARK(Template::BM_ForwardProp_Template<320, 128, 128, 10>)->Name("BM_InputSize_Template/320");
BENCHMARK(Template::BM_ForwardProp_Template<384, 128, 128, 10>)->Name("BM_InputSize_Template/384");
BENCHMARK(Template::BM_ForwardProp_Template<448, 128, 128, 10>)->Name("BM_InputSize_Template/448");
BENCHMARK(Template::BM_ForwardProp_Template<512, 128, 128, 10>)->Name("BM_InputSize_Template/512");
BENCHMARK(Template::BM_ForwardProp_Template<576, 128, 128, 10>)->Name("BM_InputSize_Template/576");
BENCHMARK(Template::BM_ForwardProp_Template<640, 128, 128, 10>)->Name("BM_InputSize_Template/640");
BENCHMARK(Template::BM_ForwardProp_Template<704, 128, 128, 10>)->Name("BM_InputSize_Template/704");
BENCHMARK(Template::BM_ForwardProp_Template<768, 128, 128, 10>)->Name("BM_InputSize_Template/768");
BENCHMARK(Template::BM_ForwardProp_Template<832, 128, 128, 10>)->Name("BM_InputSize_Template/832");
BENCHMARK(Template::BM_ForwardProp_Template<896, 128, 128, 10>)->Name("BM_InputSize_Template/896");
BENCHMARK(Template::BM_ForwardProp_Template<960, 128, 128, 10>)->Name("BM_InputSize_Template/960");
BENCHMARK(Template::BM_ForwardProp_Template<1024, 128, 128, 10>)->Name("BM_InputSize_Template/1024");

//Templatized benchmarks across hidden layer depth
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 10>)->Name("BM_LayerDepth_Template/1");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 10>)->Name("BM_LayerDepth_Template/2");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/3");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/4");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/5");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/6");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 128, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/7");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 128, 128, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/8");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/9");
BENCHMARK(Template::BM_ForwardProp_Template<256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 10>)->Name("BM_LayerDepth_Template/10");