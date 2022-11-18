#include <runtime/mnist_imageset.h>

#include <fstream>
#include <iostream>

ImageSet::ImageSet() {
  std::ifstream trainingLabels("test/img/train-labels-idx1-ubyte");
  std::ifstream trainingImages("test/img/train-images-idx3-ubyte");
  std::ifstream testLabels("test/img/t10k-labels-idx1-ubyte");
  std::ifstream testImages("test/img/t10k-images-idx3-ubyte");

  trainingLabels.seekg(8, std::ios_base::beg);
  trainingImages.seekg(16, std::ios_base::beg);
  testLabels.seekg(8, std::ios_base::beg);
  testImages.seekg(16, std::ios_base::beg);

  for (int i = 0; i < TRAIN_SIZE; i++) {
    char cl;
    char ci;
    trainingLabels.get(cl);
    trainLabels_.push_back(charToLabel(cl));

    ImgVector img(28 * 28);
    for (int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j++) {
      trainingImages.get(ci);
      img(j) = (double)(unsigned char)ci / 256.0;
    }
    img.normalize();
    trainImages_.push_back(img);
  }

  for (int i = 0; i < TEST_SIZE; i++) {
    char cl;
    char ci;
    testLabels.get(cl);
    testLabels_.push_back(charToLabel(cl));

    ImgVector img(28 * 28);
    for (int j = 0; j < IMG_WIDTH * IMG_HEIGHT; j++) {
      testImages.get(ci);
      img(j) = (double)(unsigned char)ci / 256.0;
    }
    img.normalize();
    testImages_.push_back(img);
  }
}
ImgVector& ImageSet::getImage(int index) { return trainImages_[index]; }
ImgLabel& ImageSet::getLabel(int index) { return trainLabels_[index]; }
void ImageSet::printImage(int index) {
  ImgVector img = trainImages_[index];
  ImgLabel label = trainLabels_[index];
  for (int i = 0; i < IMG_HEIGHT; i++) {
    for (int j = 0; j < IMG_WIDTH; j++) {
      if ((unsigned int)img(IMG_WIDTH * i + j) > 128) {
        std::cout << "@@";
      } else {
        std::cout << "  ";
      }
    }
    std::cout << std::endl;
  }
}

std::vector<ImgLabel> ImageSet::getTrainLabels() { return trainLabels_; }
std::vector<ImgVector> ImageSet::getTrainImages() { return trainImages_; }
std::vector<ImgLabel> ImageSet::getTestLabels() { return testLabels_; }
std::vector<ImgVector> ImageSet::getTestImages() { return testImages_; }

ImgLabel ImageSet::charToLabel(char c) {
  ImgLabel label(10);
  for (int i = 0; i < 10; i++) {
    if (c == i) {
      label(i) = 1.0;
    } else {
      label(i) = 0.0;
    }
  }
  return label;
}
