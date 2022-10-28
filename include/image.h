#pragma once

#include <Eigen/Dense>
#include <vector>

const int TEST_SIZE = 10000;
const int TRAIN_SIZE = 60000;
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;

typedef Eigen::Vector<double, IMG_WIDTH * IMG_HEIGHT> ImgVector;
typedef Eigen::Vector<double, 10> ImgLabel;

/**
 * @brief A class representing the MNIST image set
 *
 */
class ImageSet {
 public:
  /**
   * @brief Initialise the image set
   *
   */
  ImageSet();

  /**
   * @brief Get a specific image by index
   *
   * @param index
   * @return ImgVector
   */
  ImgVector GetImage(int index);

  /**
   * @brief Get the label of a specifix image by that image's index
   *
   * @param index
   * @return ImgLabel
   */
  ImgLabel GetLabel(int index);

  /**
   * @brief Output an image as ascii art
   *
   * @param index
   */
  void PrintImage(int index);

 private:
  std::vector<ImgLabel> trainLabels_;
  std::vector<ImgVector> trainImages_;
  ImgLabel CharToLabel(char);
};
