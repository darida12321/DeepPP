#pragma once
#include <activation_function.h>

#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Network {
 public:
  /**
   * @brief Construct a new Network object
   *
   * @param weights Matrices representing the weights of the connections in the
   * network
   * @param biases Bias vectors
   * @param act_func The activation function objects to be applied at each layer of the
   * network
   */
  Network(std::vector<MatrixXd> weights, std::vector<VectorXd> biases,
          std::vector<ActivationFunction*>);

  /**
   * @brief Construct a new Network object using layer sizes
   *
   * @param layer_sizes A vector containing the sizes of the layers
   * @param act_func The activation function objects to be applied at each layer of the
   * network
   */
  Network(std::vector<int> layer_sizes,
          std::vector<ActivationFunction*>);

  /**
   * @brief Propagete a vector through the network
   *
   * @param in The input vector
   * @return VectorXd The output vector
   */
  VectorXd forwardProp(VectorXd in);

  /**
   * @brief Get the cost of the function for a set of inputs
   *
   * @param in The input vector
   * @param exp_out The expected output
   * @return double The cost
   */
  double getCost(std::vector<VectorXd> in, std::vector<VectorXd> exp_out);

  /**
   * @brief Perform one trainning iteration on a given set of inputs and
   * expected outputs
   *
   * @param in Inputs
   * @param exp_out Expected outputs
   * @param stepSize The amount by which the weights and biases are to be
   * adjusted
   */
  void train(std::vector<VectorXd>, std::vector<VectorXd>, double);

  /**
   * @brief Get the weight matrices of all layers of the network
   *
   * @return std::vector<MatrixXd>& A vector containing the weight matrices
   */
  std::vector<MatrixXd>& getWeights();

  /**
   * @brief Get the bias vectors of all the layers of the network
   *
   * @return std::vector<VectorXd>& A vector containing the bias vectors
   */
  std::vector<VectorXd>& getBiases();

 private:
  std::vector<MatrixXd> weights_ = std::vector<MatrixXd>();
  std::vector<VectorXd> biases_ = std::vector<VectorXd>();
  std::vector<ActivationFunction*> act_func_;
};
