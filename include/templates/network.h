#include <Eigen/Dense>
#include <iostream>
#include <iterator>
#include <tuple>
#include <utility>
#include <templates/template_helpers.h>
#include <templates/activation_function.h>
#include <templates/cost_function.h>

// TODO  bias and weight initializer classes
// TODO  comment stuff

// USER INTERFACE
template <size_t In>
struct InputLayer{};
template <size_t Out, template <size_t > typename Activation>
struct Layer {};

template <template <size_t > typename Cost, template <size_t, size_t> typename WeightInit, template <size_t> typename BiasInit, typename... Ls>
struct Network {};


// Distribution
template<size_t N, size_t M>
struct WeightZero {
  typedef Eigen::Matrix<double, N, M> Weight;

  inline Weight genWeight() {
    return Weight::Zero();
  }

};

template<size_t N>
struct BiasZero {
  typedef Eigen::Vector<double, N> Bias;

  inline Bias genBias() {
    return Bias::Zero();
  }

};

template<size_t N, size_t M>
struct WeightRandom {
  typedef Eigen::Matrix<double, N, M> Weight;

  inline Weight genWeight() {
    return Weight::Random();
  }

};

template<size_t N>
struct BiasRandom {
  typedef Eigen::Vector<double, N> Bias;

  inline Bias genBias() {
    return Bias::Random();
  }

};






template <size_t In, size_t Out, template <size_t > typename Activation, template <size_t, size_t> typename WeightInit, template <size_t> typename BiasInit>
struct LayerBase : Activation<Out>, WeightInit<Out, In>, BiasInit<Out> {
 public:
  Eigen::Matrix<double, Out, In> weight_ = WeightInit<Out, In>::genWeight();
  Eigen::Vector<double, Out> bias_ = BiasInit<Out>::genBias();


  Eigen::Vector<double, Out> operator<<(Eigen::Vector<double, In> rhs) {
    Eigen::Vector<double, Out> z = bias_ + (weight_ * rhs);
    return Activation<Out>::activation(z);
  }
};



// Network implementation
template <typename I, template <size_t > typename Cost, template <size_t, size_t> typename WeightInit, template <size_t> typename BiasInit, typename... Ls>
struct NetworkBase {};

template <template <size_t > typename CostFunction, template <size_t, size_t> typename WeightInit, template <size_t> typename BiasInit,
  size_t Input, size_t ... Outs, size_t... LayerIndices,
  template <size_t> typename... Activations>
struct NetworkBase<
  std::index_sequence<LayerIndices...>,
  CostFunction,
  WeightInit,
  BiasInit,
  InputLayer<Input>,
  Layer<Outs, Activations>...
> : CostFunction<select_last<Outs...>::elem> {

public:
  using Cost = CostFunction<select_last<Outs...>::elem>;
  using InputVector = Eigen::Vector<double, Input>;
  using OutputVector = Eigen::Vector<double, select_last<Outs...>::elem>;

  using MatrixList = std::tuple<Eigen::Matrix<double, Outs, intlist_element<LayerIndices, Input, Outs...>::elem>...>;
  using InVectorList = std::tuple<Eigen::Vector<double, intlist_element<LayerIndices, Input, Outs...>::elem>...>;
  using OutVectorList = std::tuple<Eigen::Vector<double, Outs>...>;

  static constexpr size_t N = sizeof...(Outs);

  // Setters
  template <typename... Weights>
  /**
   * @brief Set the weight matrices
   * 
   * @param weights 
   */
  inline void setWeights(Weights... weights) {
    [ this, &weights... ]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(layers_).weight_ = weights), ...);
    }
    (std::make_index_sequence<N>{});
  }
  template <typename... Biases>
  /**
   * @brief Set the bias vectors
   * 
   * @param biases 
   */
  inline void setBiases(Biases... biases) {
    [ this, &biases... ]<std::size_t... I>(std::index_sequence<I...>) {
      ((std::get<I>(layers_).bias_ = biases), ...);
    }
    (std::make_index_sequence<N>{});
  }

  // Getters
  template <size_t I>
  /**
   * @brief Get the weight matrix of a perticular layer
   */
  inline typename std::tuple_element<I, MatrixList>::type getWeight() {
    return std::get<I>(layers_).weight_;
  }
  template <size_t I>
  /**
   * @brief Get the bias vector of a perticular layer
   */
  inline typename std::tuple_element<I, OutVectorList>::type getBias() {
    return std::get<I>(layers_).bias_;
  }

  /**
   * @brief Perform forward propagation
   * 
   * @param input
   */
  inline OutputVector forwardProp(const InputVector& input) {
    return [ this, &input ]<size_t... I>(reverse_index_sequence<I...>) {
      return (std::get<I>(layers_) << ... << input);
    }
    (make_reverse_index_sequence<N>{});
  }

  /**
   * @brief Use backpropagation to train the network on a set of inputs and
   * expected outputs
   * 
   * @param in inputs
   * @param exp_out expected outputs
   * @param stepSize amount by which to vary the weights and biases
   */
  inline void train(const std::vector<InputVector>& in, const std::vector<OutputVector>& exp_out,
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
    std::tuple<InputVector, Eigen::Vector<double, Outs>...> a;
    for (size_t i = 0; i < in.size(); i++) {
      // Save activations
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
        stepSize ]<size_t... I>(reverse_index_sequence<I...>) {
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

  
  inline double getAccuracy(const std::vector<InputVector>& in,
                            const std::vector<OutputVector>& exp_out) {
    double acc = 0;
    for (unsigned int i = 0; i < in.size(); i++) {
      OutputVector val = forwardProp(in[i]);
      Eigen::Index predicted, expected;
      val.maxCoeff(&predicted);
      exp_out[i].maxCoeff(&expected);
      if (predicted == expected) {
        acc++;
      }
    }
    return acc / in.size();
  }

private:
  std::tuple<LayerBase<intlist_element<LayerIndices, Input, Outs...>::elem, Outs, Activations, WeightInit, BiasInit>...> layers_;
};


// template <
//   template <size_t> typename CostFunction,
//   size_t Input, size_t... Outs,
//   template <size_t> typename... Activations
// >
// struct Network<
//   CostFunction,
//   InputLayer<Input>,
//   Layer<Outs, Activations>...
// > : NetworkBase<
//     std::make_index_sequence<sizeof...(Outs)>,
//     CostFunction, 
//     WeightZero,
//     BiasZero,
//     InputLayer<Input>,
//     Layer<Outs, Activations>...
//   >
// {};


template <
  template <size_t> typename CostFunction, template <size_t, size_t> typename WeightInit, template <size_t> typename BiasInit,
  size_t Input, size_t... Outs,
  template <size_t> typename... Activations
>
struct Network<
  CostFunction,
  WeightInit,
  BiasInit,
  InputLayer<Input>,
  Layer<Outs, Activations>...
> : NetworkBase<
    std::make_index_sequence<sizeof...(Outs)>,
    CostFunction,
    WeightInit,
    BiasInit, 
    InputLayer<Input>,
    Layer<Outs, Activations>...
  >
{};









