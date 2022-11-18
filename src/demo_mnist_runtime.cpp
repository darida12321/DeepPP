#include <network.h>
#include <activation_function.h>
#include <cost_function.h>
#include <mnist_imageset.h>

#include <Eigen/Dense>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
// #include "Eigen/src/Core/Matrix.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

const size_t IMAGESIZE = IMG_WIDTH*IMG_HEIGHT;

int main() {
	// Network network({IMAGESIZE, 128, 128, 10}, {&relu, &relu, &softmax}, &mean_sqr_error);
  MatrixXd w1 = MatrixXd::Zero(128, 28 * 28);
  VectorXd b1 = VectorXd::Zero(128);
  MatrixXd w2 = MatrixXd::Zero(128, 128);
  VectorXd b2 = VectorXd::Zero(128);
  MatrixXd w3 = MatrixXd::Zero(10, 128);
  VectorXd b3 = VectorXd::Zero(10);
  Network network(std::vector<MatrixXd>{w1, w2, w3},
                  std::vector<VectorXd>{b1, b2, b3},
                  std::vector<ActivationFunction*>{&relu, &relu, &softmax},
                  &mean_sqr_error);

	ImageSet image;

	auto trainImages = image.getTrainImages();
	auto trainLabels = image.getTrainLabels();

	double acc =
		network.getAccuracy(trainImages, trainLabels);
	std::cout << "Network accuracy: " << acc << std::endl;

	std::cout << "Started training" << std::endl;
	for (int i = 0; i < 3; i++) {
		network.train(trainImages, trainLabels, 0.1);

		std::cout << "Completed round " << i << std::endl;
		double acc =
			network.getAccuracy(trainImages, trainLabels);
		std::cout << "Network accuracy: " << acc << std::endl;
	}

	return 0;
}