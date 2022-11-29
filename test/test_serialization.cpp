#include <gtest/gtest.h>
#include <templates/network.h>

#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector;
using namespace Template;

typedef Network<MeanSquareError, WeightRandom, BiasRandom, InputLayer<2>, Layer<2, Linear>, Layer<2, Softmax>> NetworkTemplate; 

void assert_equal_networks(NetworkTemplate& n1, NetworkTemplate& n2) { 
    EXPECT_TRUE(n1.getWeight<0>().isApprox(n2.getWeight<0>(), 0.01));
    EXPECT_TRUE(n1.getWeight<1>().isApprox(n2.getWeight<1>(), 0.01));
    EXPECT_TRUE(n1.getBias<0>().isApprox(n2.getBias<0>(), 0.01));
    EXPECT_TRUE(n1.getBias<1>().isApprox(n2.getBias<1>(), 0.01));
}

TEST(Serialization, Templatized) {
   NetworkTemplate network_1;
   network_1.exportNetwork("network.deeppp");
    
   NetworkTemplate network_2;
   network_2.importNetwork("network.deeppp");
   assert_equal_networks(network_1, network_2);
}
