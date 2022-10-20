#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <layer.h>
#include <network.h>
#include <image.h>
#include <activation_function.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;


TEST(MnistTest, ReadData) {
    ImageSet image;
    image.PrintImage(0);

    std::cout << "This is a test" << std::endl;
}
