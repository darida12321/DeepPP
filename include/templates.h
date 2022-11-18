#pragma once

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

template<class WeightType = std::vector<MatrixXd>, 
          class BiasesType = std::vector<VectorXd>,
          class LayerSizesType = std::vector<int>,
          class ActFuncType = std::vector<std::function<VectorXd(VectorXd)>>,
          class ActFuncDerType = std::vector<std::function<VectorXd(VectorXd)>>>
class templates {

    Network(WeightType weights, BiasesType biases,
          ActFuncType act_func,
          ActFuncDerType act_func_der);

    Network(LayerSizesType layer_sizes,
          ActFuncType act_func,
          ActFuncDerType act_func_der);
};
