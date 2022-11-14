#pragma once

#include <vector>
#include <Eigen/Dense>

const int TRAIN_SIZE = 60000;
const int TEST_SIZE = 10000;
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;

typedef Eigen::VectorXd ImgVector;
typedef Eigen::VectorXd ImgLabel;

class ImageSet {
 public:
    ImageSet();
    ImgVector& getImage(int index);
    ImgLabel& getLabel(int index);
    void printImage(int index);
    std::vector<ImgLabel> getTrainLabels(); 
    std::vector<ImgVector> getTrainImages(); 
    std::vector<ImgLabel> getTestLabels(); 
    std::vector<ImgVector> getTestImages(); 
  private:
    std::vector<ImgLabel> trainLabels_;
    std::vector<ImgVector> trainImages_;
    std::vector<ImgLabel> testLabels_;
    std::vector<ImgVector> testImages_;
    ImgLabel charToLabel(char c);
};