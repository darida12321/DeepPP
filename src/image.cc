#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <image.h>

ImageSet::ImageSet() {
    std::ifstream trainingLabels("../img/train-labels-idx1-ubyte");
    std::ifstream trainingImages("../img/train-images-idx3-ubyte");

    trainingLabels.seekg(8, std::ios_base::beg); 
    trainingImages.seekg(16, std::ios_base::beg);

    for (int i = 0; i < TRAIN_SIZE; i++) {
        char cl;
        char ci;
        trainingLabels.get(cl);
        trainLabels_.push_back(cl);
        
        ImgVector img;
        for (int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j++) {
            trainingImages.get(ci);
            img(j) = ci;
        }
        trainImages_.push_back(img);
    }
}


ImgVector ImageSet::GetImage(int index) {
    return trainImages_[index];
}

int ImageSet::GetLabel(int index) {
    return trainLabels_[index];
}


void ImageSet::PrintImage(int index) {
    ImgVector img = trainImages_[index];
    int label = trainLabels_[index]; 
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            if ((unsigned int) img(IMG_WIDTH * i + j) > 128) {
                std::cout << "@@";
            } else {
                std::cout << "  ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "label: " << label << std::endl;
}
