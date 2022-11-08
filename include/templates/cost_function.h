#pragma once
#include <math.h>
#include <templates/linalg.h>

#include <Eigen/Dense>
#include <cmath>

namespace Template {
  template <int size>
  class CostFunction {
  public:
    virtual inline double function(Vectord<size> out, Vectord<size> exp_out) = 0;
    virtual inline Vectord<size> derivative(Vectord<size> out,
                                            Vectord<size> exp_out) = 0;
  };

  template <int size>
  class MeanSquareError : public CostFunction<size> {
    public:
      inline double function(Vectord<size> out, Vectord<size> exp_out) {
        auto errors = (out - exp_out).array();
        return (errors * errors).sum() / out.rows();
      }
      inline Vectord<size> derivative(Vectord<size> out, Vectord<size> exp_out) {
        return (2.0 / out.rows()) * (out - exp_out);
      }
  };

  inline double clip(double x) { return fmin(1 - 1e-7, fmax(x, 1e-7)); }

  template <int size>
  class CategoricalCrossEntropy : public CostFunction<size> {
    public:
      inline double function(Vectord<size> out, Vectord<size> exp_out) {
        auto logs = out.unaryExpr([](double x) { return log(clip(x)); }).array();
        return -(logs * exp_out.array()).sum();
      }
      inline Vectord<size> derivative(Vectord<size> out, Vectord<size> exp_out) {
        return -exp_out.cwiseQuotient(
            out.unaryExpr([](double x) { return clip(x); }));
      }
  };
}  // namespace Template
