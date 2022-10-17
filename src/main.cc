#include <cmath>
#include <iostream>
// #include "layer.h"
#include "network.h"

double act_func(double x) {
    return 1 / (1 + std::exp(-x));
}

int main(){
    MatrixXd l1_weights(2, 2);
    l1_weights(0, 0) = 1;
    l1_weights(1, 0) = 0;
    l1_weights(0, 1) = 0;
    l1_weights(1, 1) = 1;

    VectorXd l1_bias(2);
    l1_bias(0) = 2;
    l1_bias(1) = 3;

    Layer layer1(l1_weights, l1_bias, act_func);

    std::vector<Layer> layers{
        layer1
    };
    Network network(layers);

    VectorXd input(2);
    input(0) = 1;
    input(1) = 2;
    VectorXd out = network.forwardProp(input);
    std::cout << "Output: " << std::endl << out << std::endl;

    std::cout << "I wanna die" << std::endl;
    return 0;
}


