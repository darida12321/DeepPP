#include <Eigen/Dense>

// TODO  documentation
//TODO add other
namespace Template {
  template<int N>
  struct MeanSquareError {
    typedef Eigen::Vector<double, N> Vec;
    inline double cost(Vec out, Vec exp_out) {
      auto errors = (out - exp_out).array();
      return (errors * errors).sum() / out.rows();
    }
    inline Vec cost_der(Vec out, Vec exp_out) {
      return (2.0 / out.rows()) * (out - exp_out);
    }
  };

  template<int N>
  struct CategoricalCrossEntropy {
    typedef Eigen::Vector<double, N> Vec;
    inline double clip(double x) { return fmin(1 - 1e-7, fmax(x, 1e-7)); }

    inline double cost(Vec out, Vec exp_out) {
      auto logs = out.unaryExpr([this](double x) { return log(clip(x)); }).array();
      return -(logs * exp_out.array()).sum();
    }

    inline Vec cost_der(Vec out, Vec exp_out) {
      return -exp_out.cwiseQuotient(
          out.unaryExpr([this](double x) { return clip(x); }));
    }
  };

}
