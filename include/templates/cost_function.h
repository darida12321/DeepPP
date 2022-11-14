#include <Eigen/Dense>

//TODO add other
namespace Template {
  template<size_t N>
  struct MeanSquareError {
    typedef Eigen::Vector<double, N> Vec;
  /**
   * @brief Calculate the mean square error
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double
   */
    inline double cost(const Vec& out, const Vec& exp_out) {
      auto errors = (out - exp_out).array();
      return (errors * errors).sum() / N;
    }
  /**
   * @brief Calculate the gradient of the mean square error
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return Vector<double, N>
   */
    inline Vec cost_der(const Vec& out, const Vec& exp_out) {
      return (2.0 / N) * (out - exp_out);
    }
  };

  template<size_t N>
  struct CategoricalCrossEntropy {
    typedef Eigen::Vector<double, N> Vec;
    /**
     * @brief restricts values from the closed interval [0,1] to the open
     * interval (0,1). Used in to prevent division by 0.
     *
     * @param x
     * @return double
     * @pre 0 <= x <= 1
     */
    inline double clip(double x) { return fmin(1 - 1e-7, fmax(x, 1e-7)); }

  /**
   * @brief Calculate the categorical cross entropy
   *
   * @param out The actual output of the neural network
   * @param exp_out The expected output
   * @return double
   */
    inline double cost(const Vec& out, const Vec& exp_out) {
      auto logs = out.unaryExpr([this](double x) { return log(clip(x)); }).array();
      return -(logs * exp_out.array()).sum();
    }

    /**
     * @brief Calculate the gradient of the categorical cross entropy
     *
     * @param out The actual output of the neural network
     * @param exp_out The expected output
     * @return Vector<double, N>
     */
    inline Vec cost_der(const Vec& out, const Vec& exp_out) {
      return -exp_out.cwiseQuotient(
          out.unaryExpr([this](double x) { return clip(x); }));
    }
  };

}
