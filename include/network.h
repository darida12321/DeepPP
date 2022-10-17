#pragma once

#include <vector>
#include "layer.h"
#include "../Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Network {
public:
    Network(std::vector<Layer>);
    VectorXd forwardProp(VectorXd);
    VectorXd backProp(VectorXd);
private:
    std::vector<Layer> layers_;
};
