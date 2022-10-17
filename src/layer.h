#pragma once
#include "../Eigen/Dense"

class Layer {

  Layer(MatrixXd m, VectorXd b, std::function<double(double)> act_func);

  VectorXd forwardProp(VectorXd in);

  VectorXd backProp(VectorXd err);
};