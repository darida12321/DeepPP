#pragma once

#include <Eigen/Dense>
#include <vector>

const int TEST_SIZE = 10000;
const int TRAIN_SIZE = 60000;
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;

typedef Eigen::Vector<double, IMG_WIDTH * IMG_HEIGHT> ImgVector;
typedef Eigen::Vector<double, 10> ImgLabel;

class ImageSet {
 public:
  ImageSet();
  ImgVector getImage(int index);
  ImgLabel getLabel(int index);
  void printImage(int index);

 private:
  std::vector<ImgLabel> trainLabels_;
  std::vector<ImgVector> trainImages_;
  ImgLabel charToLabel(char);
};
