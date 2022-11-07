#include <activation_function.h>
#include <benchmark/benchmark.h>
#include <cost_function.h>
#include <network.h>

#include <Eigen/Dense>
#include <cmath>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;

static void BM_MultiInput(benchmark::State& state) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                  std::vector<ActivationFunction*>{&relu, &relu},
                  mean_sqr_error, mean_sqr_error_der);

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  for (auto _ : state) {
    network.train(input, output, 1);
  }
}

BENCHMARK(BM_MultiInput);

static void BM_Backprop_Relu(benchmark::State& state) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                  std::vector<ActivationFunction*>{&relu, &relu},
                  mean_sqr_error, mean_sqr_error_der);

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  for (auto _ : state) {
    network.train(input, output, 1);
  }
}

BENCHMARK(BM_Backprop_Relu);

static void BM_Backprop_Sigmoid(benchmark::State& state) {
  // Create 1-1 neural network
  MatrixXd w(2, 2);
  w << 1, 1, 1, 1;
  VectorXd b(2);
  b << 1, 1;
  Network network(std::vector<MatrixXd>{w, w}, std::vector<VectorXd>{b, b},
                  std::vector<ActivationFunction*>{&sigmoid, &sigmoid},
                  mean_sqr_error, mean_sqr_error_der);

  // Create example data point
  VectorXd in1(2);
  in1 << 1.0, 1.0;
  VectorXd out1(2);
  out1 << 3.0, 3.0;
  VectorXd in2(2);
  in2 << 3.0, 5.0;
  VectorXd out2(2);
  out2 << 6.0, 8.0;
  std::vector<VectorXd> input{in1, in2};
  std::vector<VectorXd> output{out1, out2};

  // Train the network
  for (auto _ : state) {
    network.train(input, output, 1);
  }
}

BENCHMARK(BM_Backprop_Sigmoid);
