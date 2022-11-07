#pragma once
#include <Eigen/Dense>

using Eigen::Matrix;

template<int size>
struct Linalg {
	typedef Matrix<double, size, size> Matrixd;
	typedef Matrix<double, size, 1> Vectord;
};

template<int size>
using Matrixd =  typename Linalg<size>::Matrixd;

template<int size>
using Vectord = typename Linalg<size>::Vectord;