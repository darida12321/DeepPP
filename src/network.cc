#include "../include/network.h"

Network::Network(std::vector<Layer> layers): layers_(layers) {}

VectorXd Network::forwardProp(VectorXd in) {
    VectorXd curr = in;
    for (int i = 0; i < layers_.size(); i++) {
        curr = layers_[i].forwardProp(curr);
    }
    return curr;
}
VectorXd Network::backProp(VectorXd err) {
    return err;
}

