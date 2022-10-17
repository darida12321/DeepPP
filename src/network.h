#pragma once

#include <vector>
#include "layer.cc"
#include "../Eigen/Dense"

class Network {
public:
    Network(std::vector<Layer>);
    VectorXd forwardProp(VectorXd);
    VectorXd backProp(VectorXd);
private:
    std::vector<Layer> layers_;
};
