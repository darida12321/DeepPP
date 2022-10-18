#include <iostream>
#include <fstream>
#include <vector>
#include "../Eigen/Dense"

class ImageSet {
    public:
        ImageSet() {
            std::ifstream trainingLabels("../img/train-labels-idx1-ubyte");
            std::ifstream trainingImages("../img/train-images-idx3-ubyte");

            trainingLabels.seekg(8, std::ios_base::beg); 
            trainingImages.seekg(16, std::ios_base::beg);

            for (int i = 0; i < TRAIN_SIZE; i++) {
                char cl;
                char ci;
                trainingLabels.get(cl);
                trainLabels_.push_back(cl);
                
                Eigen::Matrix<char, 28 * 28, 1> img;
                for (int j = 0; j < 28 * 28; j++) {
                    trainingImages.get(ci);
                    img(j) = ci;
                }
                trainImages_.push_back(img);
            }
        }

        Eigen::Matrix<char, 28 * 28, 1> GetImage(int index) {
            return trainImages_[index];
        }

        int GetLabel(int index) {
            return trainLabels_[index];
        }

        void PrintImage(int index) {
            Eigen::Matrix<char, 28 * 28, 1> img = trainImages_[index];
            int label = trainLabels_[index]; 
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    if ((unsigned int) img(28 * i + j) > 128) {
                        std::cout << "@@";
                    } else {
                        std::cout << "  ";
                    }
                }
                std::cout << std::endl;
            }
            std::cout << "label: " << label << std::endl;
        }

    private:
        static const int TEST_SIZE = 10000;
        static const int TRAIN_SIZE = 60000;

        std::vector<char> trainLabels_;
        std::vector<Eigen::Matrix<char, 28 * 28, 1>> trainImages_;
};
