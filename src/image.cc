#include <iostream>
#include <fstream>
#include <vector>
#include "../Eigen/Dense"

class ImageSet {
    public:
        ImageSet() {
            std::ifstream trainingLabels_("../img/t10k-labels-idx1-ubyte");
            std::ifstream trainingImages_("../img/t10k-labels-idx3-ubyte");

            trainingLabels_.seekg(8, std::ios_base::beg); 
            trainingImages_.seekg(16, std::ios_base::beg);

            for (int i = 0; i < TRAIN_SIZE; i++) {
                labels_.push_back(trainingLabels_.get());
                
                Eigen::Matrix<char, IMG_SIZE, 1> img;
                for (int j = 0; j < IMG_SIZE; j++) {
                    img << trainingImages_.get();
                }

            }
        }

        

    private:
        std::vector<char> labels_;

        static const int TEST_SIZE = 10000;
        static const int TRAIN_SIZE = 60000;
        static const int IMG_SIZE = 28 * 28;
};

// For test purposes only
int main() {
    return EXIT_SUCCESS;
}
