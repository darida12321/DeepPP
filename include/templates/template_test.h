#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <utility>
#include <templates/template_helpers.h>
#include <templates/activation_function.h>
#include <templates/cost_function.h>

// TODO  use std::size for stuff
// TODO  bias and weight initializer classes
// TODO: smart template stuff to remove layer size redundancy

template <int In, int Out, template <int> typename Activation>
struct Layer : Activation<Out> {
 public:
  Eigen::Matrix<double, Out, In> weight_;
  Eigen::Vector<double, Out> bias_;

  Eigen::Vector<double, Out> operator<<(Eigen::Vector<double, In> rhs) {
    return Activation<Out>::activation(bias_ + (weight_ * rhs));
  }
};



template <template <int> typename Cost, typename... Ls>
struct Network {};




template <template <int> typename CostFunction, int... Ins, int... Outs,
          template <int> typename... Activations>
struct Network<CostFunction, Layer<Ins, Outs, Activations>...>
    : CostFunction<select_last<Outs...>::elem> {

public:
  using Cost = CostFunction<select_last<Outs...>::elem>;
  using InputVector = Eigen::Vector<double, select_first<Ins...>::elem>;
  using OutputVector = Eigen::Vector<double, select_last<Outs...>::elem>;

  using MatrixList = std::tuple<Eigen::Matrix<double, Outs, Ins>...>;
  using InVectorList = std::tuple<Eigen::Vector<double, Ins>...>;
  using OutVectorList = std::tuple<Eigen::Vector<double, Outs>...>;

  static constexpr int N = sizeof...(Outs);

  // Setters
  template <typename... Weights>
  void setWeights(Weights... weights) {
    [ this, &weights... ]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(layers_).weight_ = weights), ...);
    }
    (std::make_index_sequence<N>{});
  }
  template <typename... Biases>
  void setBiases(Biases... biases) {
    [ this, &biases... ]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(layers_).bias_ = biases), ...);
    }
    (std::make_index_sequence<N>{});
  }

  // Getters
  template <int I>
  typename std::tuple_element<I, MatrixList>::type getWeight() {
    return std::get<I>(layers_).weight_;
  }
  template <int I>
  typename std::tuple_element<I, OutVectorList>::type getBias() {
    return std::get<I>(layers_).bias_;
  }

  // Forward propogation
  OutputVector forwardProp(InputVector input) {
    return [ this, &input ]<int... I>(reverse_index_sequence<I...>) {
      return (std::get<I>(layers_) << ... << input);
    }
    (make_reverse_index_sequence<N>{});
  }

  // Backpropogation
  void train(std::vector<InputVector> in, std::vector<OutputVector> exp_out,
             double stepSize) {
    // Create change accumulators
    MatrixList weight_acc;
    OutVectorList bias_acc;

    // Reset the change accumulators
    // TODO Zero function vs. Matrix::Zero creation.
    [&weight_acc, &bias_acc ]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(weight_acc) =
            std::tuple_element<I, MatrixList>::type::Zero()),
       ...);
      ((std::get<I>(bias_acc) =
            std::tuple_element<I, OutVectorList>::type::Zero()),
       ...);
    }
    (std::make_index_sequence<N>{});

    // For each data poing, accumulate the changes
    // TODO: Use std::array for compile time size
    for (int i = 0; i < in.size(); i++) {
      // Save activations
      std::tuple<InputVector, Eigen::Vector<double, Outs>...> a;
      std::get<0>(a) = in[i];

      // Forward propogation
      [ this, &a]<std::size_t... I>(std::index_sequence<I...>) {
        ((std::get<I + 1>(a) = get<I>(layers_).activation(
              std::get<I>(layers_).bias_ +
              std::get<I>(layers_).weight_ * std::get<I>(a))),
         ...);
      }
      (std::make_index_sequence<N>{});

      // Backward propogation
      [ this, &i, &exp_out, &a, &weight_acc, &bias_acc, &
        stepSize ]<int... I>(reverse_index_sequence<I...>) {
        std::tuple<InputVector, Eigen::Vector<double, Outs>...> dcda;
        // Calculate error as last element
        std::get<N>(dcda) = Cost::cost_der(std::get<N>(a), exp_out[i]);

        // dcda[i] = act_der(b[j] + w[j] * a[j]) * dcda[i+1]
        ((std::get<I>(dcda) =
              std::get<I>(layers_).weight_.transpose() *
              std::get<I>(layers_).activation_der(std::get<I>(layers_).bias_ +
                                                  std::get<I>(layers_).weight_ *
                                                      std::get<I>(a)) *
              std::get<I + 1>(dcda)),
         ...);

        // dcdz = act_der(b[i] + w[i] * a[i]) * dcda[i]
        ((std::get<I>(weight_acc) -=
          std::get<I>(layers_).activation_der(std::get<I>(layers_).bias_ +
                                              std::get<I>(layers_).weight_ *
                                                  std::get<I>(a)) *
          std::get<I + 1>(dcda) * std::get<I>(a).transpose() * stepSize,

          std::get<I>(bias_acc) -=
          std::get<I>(layers_).activation_der(std::get<I>(layers_).bias_ +
                                              std::get<I>(layers_).weight_ *
                                                  std::get<I>(a)) *
          std::get<I + 1>(dcda) * stepSize),
         ...);
      }
      (make_reverse_index_sequence<N>{});
    }

    // Add the change
    [ this, &in, &weight_acc, &
      bias_acc ]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(layers_).weight_ += std::get<I>(weight_acc) / in.size()),
       ...);
      ((std::get<I>(layers_).bias_ += std::get<I>(bias_acc) / in.size()), ...);
    }
    (std::make_index_sequence<N>{});
  }

private:
  std::tuple<Layer<Ins, Outs, Activations>...> layers_;
};



