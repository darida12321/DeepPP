#pragma once

#include <vector>
#include <Eigen/Dense>

const int TEST_SIZE = 10000;
const int TRAIN_SIZE = 60000;
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;

typedef Eigen::Matrix<char, IMG_WIDTH * IMG_HEIGHT, 1> ImgVector;

class ImageSet {
    public:
        ImageSet();
        ImgVector GetImage(int index);
        int GetLabel(int index);
        void PrintImage(int index);
    private:
        std::vector<char> trainLabels_;
        std::vector<ImgVector> trainImages_;
};
