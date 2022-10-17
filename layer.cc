#include "Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class Layer {

    public:
    //constructor
    Layer(MatrixXd m, VectorXd b, std::function<double (double)> act_func) : 
          m_(m), b_ (b), act_func_ (act_func){}


    VectorXd ProcessVector(VectorXd v) {

        VectorXd newV = m_ * v + b_;
        newV = newV.unaryExpr(act_func_);
        return newV;

    }

    private:
    MatrixXd m_;
    VectorXd b_;
    std::function<double (double)> act_func_; //activator function

};