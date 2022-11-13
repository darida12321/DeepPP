#include <Eigen/Dense>

// TODO  documentation
// TODO add rest

namespace Template {
  template <size_t N>
  struct Linear {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    inline Vec activation(Vec x) { return x; }

    inline Mat activation_der(Vec x) {
      Mat out = Mat::Zero();
      for (size_t i = 0; i < N; i++) {
        out(i, i) = 1;
      }
      return out;
    }
  };

  template <size_t N>
  struct Sigmoid {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    inline Vec activation(Vec x) {
      return x.unaryExpr([](double x) { return 1 / (1 + std::exp(-x)); });
    }

    inline Mat activation_der(Vec x) {
      Mat out = Mat::Zero();
      Vec diag = activation(x).cwiseProduct(
          activation(x).unaryExpr([](double x) { return 1 - x; }));

      for (size_t i = 0; i < N; i++) {
        out(i, i) = diag(i);
      }
      return out;
    }
  };

  template<size_t N>
  struct Softmax {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    inline Vec activation(Vec x) {
      double max = x.maxCoeff();
      Vec shiftx = x.array() - max;
      Vec exps = shiftx.unaryExpr([](double x) { return std::exp(x); });
      return exps.array() / exps.sum();
    }

    inline Mat activation_der(Vec x) {
      Vec y = activation(x);
      Mat out = Mat::Zero();
      for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
          if (i == j) {
            out(i, j) += y(i) * (1 - y(i));
          } else {
            out(i, j) += - y(i) * y(j);
          }
        }
      }
      return out;
    }
  };

  template <size_t N>
  struct Relu {
    typedef Eigen::Vector<double, N> Vec;
    typedef Eigen::Matrix<double, N, N> Mat;

    inline Vec activation(Vec x) {
      return x.unaryExpr([](double x) { return std::max(0.0, x); });
    }

    inline Mat activation_der(Vec x) {
      Mat out = Mat::Zero();
      for (size_t i = 0; i < N; i++) {
        out(i, i) = x(i) < 0 ? 0 : 1;
      }
      return out;
    }
  };

}
