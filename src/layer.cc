#include "../Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer {
public:
  // constructor
  Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func)
      : weights_(m), bias_(b), act_func_(act_func) {}

  VectorXd forwardProp(VectorXd in) {
    VectorXd newV = weights_ * in + bias_;
    newV = newV.unaryExpr(act_func_);
    return newV;
  }

  VectorXd backProp(VectorXd err) {
    VectorXd error = err;
    return error;
  }

private:
  MatrixXd weights_;
  VectorXd bias_;
  std::function<double(double)> act_func_; // activator function
};
