#include <Eigen/Dense>

// TODO  documentation
// TODO add rest

template <int N>
struct Linear {
  typedef Eigen::Vector<double, N> Vec;
  typedef Eigen::Matrix<double, N, N> Mat;

  inline Vec activation(Vec x) { return x; }

  inline Mat activation_der(Vec x) {
    Mat out = Mat::Zero();
    for (int i = 0; i < x.rows(); i++) {
      out(i, i) = 1;
    }
    return out;
  }
};

template <int N>
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

    for (int i = 0; i < N; i++) {
      out(i, i) = diag(i);
    }
    return out;
  }
};

template<int N>
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
    for (int i = 0; i < x.rows(); i++) {
      for (int j = 0; j < x.rows(); j++) {
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

