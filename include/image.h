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
  ImgVector GetImage(int index);
  ImgLabel GetLabel(int index);
  void PrintImage(int index);

 private:
  std::vector<ImgLabel> trainLabels_;
  std::vector<ImgVector> trainImages_;
  ImgLabel CharToLabel(char);
};
