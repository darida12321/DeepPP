#include <Eigen/Dense>

// TODO  documentation
//TODO add other

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
