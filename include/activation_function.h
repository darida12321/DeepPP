#pragma once
#include <cmath>

inline double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
inline double sigmoid_derivative(double x) { return sigmoid(x) * (1 - sigmoid(x)); }

inline double relu(double x) { return fmax(x, 0); }
inline double relu_derivative(double x) {
  if (x <= 0)
    return 0;
  return 1;
}

inline double linear(double x) { return x; }
inline double linear_derivative(double x) { return 1; }
