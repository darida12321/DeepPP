#include <iostream>
#include <tuple>
#include <utility>
#include <Eigen/Dense>

using namespace std;

// Select last argument
template<int ...Args> struct select_last;
template<int A>
struct select_last<A> { static constexpr int elem = A; };
template<int A, int... Args> 
struct select_last<A,Args...>{ 
  static constexpr int elem = select_last<Args...>::elem;
};

// Select first argument
template<int ...Args> struct select_first;
template<int A, int... Args> 
struct select_first<A,Args...>{ 
  static constexpr int elem = A;
};

// Reverse index sequence
template<int... Ints> struct reverse_index_sequence {};

template<std::size_t N, int... Is>
struct make_reverse_index_sequence : 
  make_reverse_index_sequence<N - 1, Is..., N - 1> 
{};
template<int... Is>
struct make_reverse_index_sequence<0, Is...> : 
  reverse_index_sequence<Is...> 
{};


template <int In, int Out, template<int> typename Activation>
struct Layer : Activation<Out>{
public:
  Eigen::Matrix<double, Out, In> weight_;
  Eigen::Vector<double, Out> bias_;

  Eigen::Vector<double, Out> operator<<(Eigen::Vector<double, In> rhs) {
    return Activation<Out>::activation(bias_ + (weight_ * rhs));
  }
};


template <template<int> typename Cost, typename... Ls>
struct Layers {};

template <
  template<int> typename CostFunction,
  int... Ins, int... Outs,
  template<int> typename... Activations
>
struct Layers<
  CostFunction,
  Layer<Ins, Outs, Activations>...
> : CostFunction<select_last<Outs...>::elem> {
public:
  using Cost = CostFunction<select_last<Outs...>::elem>;
  using InputVector = Eigen::Vector<double, select_first<Ins...>::elem>; 
  using OutputVector = Eigen::Vector<double, select_last<Ins...>::elem>; 

  using MatrixList = std::tuple<Eigen::Matrix<double, Outs, Ins>...>;
  using InVectorList = std::tuple<Eigen::Vector<double, Ins>...>;
  using OutVectorList = std::tuple<Eigen::Vector<double, Outs>...>;

  static constexpr int N = sizeof...(Outs);

  // Setters
  template<typename... Weights>
  void setWeights(Weights... weights) {
    [this, &weights...] <std::size_t... I>
    (std::index_sequence<I...>)
    {
      ((std::get<I>(layers_).weight_ = weights) , ...);
    }(
      std::make_index_sequence<N>{}
    );
  }
  template<typename... Biases>
  void setBiases(Biases... biases) {
    [this, &biases...] <std::size_t... I>
    (std::index_sequence<I...>)
    {
      ((std::get<I>(layers_).bias_ = biases) , ...);
    }(
      std::make_index_sequence<N>{}
    );
  }

  // Getters
  template<int I>
  typename std::tuple_element<I, MatrixList>::type getWeight() {
    return std::get<I>(layers_).weight_;
  }
  template<int I>
  typename std::tuple_element<I, MatrixList>::type getBias() {
    return std::get<I>(layers_).bias_;
  }


  // Forward propogation

  OutputVector forwardProp(InputVector input) {
    return [this, &input] <int... I>
    (reverse_index_sequence<I...>)
    {
      return (std::get<I>(layers_) << ... << input);
    }(
      make_reverse_index_sequence<N>{}
    );
  }

  // Backpropogation
  void train(
      std::vector<InputVector> in, 
      std::vector<OutputVector> exp_out,
      double stepSize
  ) {
    // Create change accumulators
    MatrixList weight_acc;
    OutVectorList bias_acc;

    // Reset the change accumulators
    // TODO Zero function vs. Matrix::Zero creation.
    [&weight_acc, &bias_acc] <std::size_t... I>
    (std::index_sequence<I...>){
      ((std::get<I>(weight_acc).Zero()) , ...);
      ((std::get<I>(bias_acc).Zero()) , ...);
    }(std::make_index_sequence<N>{});

    // For each data poing, accumulate the changes
    // TODO: Use std::array for compile time size
    for (int i = 0; i < in.size(); i++) {
      // Save activations 
      std::tuple<InputVector, Eigen::Vector<double, Ins>...> a;
      std::get<0>(a) = in[i];

      // Forward propogation
      [this, &a, &weight_acc, &bias_acc] <std::size_t... I>
      (std::index_sequence<I...>){
        ((
          std::get<I+1>(a) = get<I>(layers_).activation(
            std::get<I>(layers_).bias_ + 
            std::get<I>(layers_).weight_ * std::get<I>(a)
          )
        ) , ...);
      }(std::make_index_sequence<N>{});

      // Backward propogation
      // cout << std::get<N>(a) << endl << endl;
      [this, &i, &exp_out, &a, &weight_acc, &bias_acc, &stepSize] <int... I>
      (reverse_index_sequence<I...>){
        std::tuple<InputVector, Eigen::Vector<double, Outs>...> dcda;
        // Calculate error as last element 
        std::get<N>(dcda) = Cost::cost_der(std::get<N>(a), exp_out[i]);

        // dcda[i] = act_der(b[j] + w[j] * a[j]) * dcda[i+1]
        ((
          std::get<I>(dcda) = std::get<I>(layers_).weight_.transpose() *
          std::get<I>(layers_).activation_der(
            std::get<I>(layers_).bias_ +
            std::get<I>(layers_).weight_ * std::get<I>(a)
          ) * std::get<I+1>(dcda)
        ) , ...);
        // cout << "TRAIN" << endl;
        // cout << "Error: " << endl << std::get<0>(dcda) << endl << endl;
        // cout << "Error: " << endl << std::get<1>(dcda) << endl << endl;
        // cout << "Error: " << endl << std::get<2>(dcda) << endl << endl;
        
        // dcdz = act_der(b[i] + w[i] * a[i]) * dcda[i]
        ((
          std::get<I>(weight_acc) -= std::get<I>(layers_).activation_der(
            std::get<I>(layers_).bias_ + 
            std::get<I>(layers_).weight_ * std::get<I+1>(a)
          ) * std::get<I>(dcda) * std::get<I+1>(a).transpose() * stepSize,

          std::get<I>(bias_acc) -= std::get<I>(layers_).activation_der(
            std::get<I>(layers_).bias_ + 
            std::get<I>(layers_).weight_ * std::get<I+1>(a)
          ) * std::get<I>(dcda) * stepSize
        ) , ...);

      }(make_reverse_index_sequence<N>{});
    }

    // Add the change
    [this, &weight_acc, &bias_acc] <std::size_t... I>
    (std::index_sequence<I...>){
      ((std::get<I>(layers_).weight_ += std::get<I>(weight_acc)) , ...);
      ((std::get<I>(layers_).bias_ += std::get<I>(bias_acc)) , ...);
    }(std::make_index_sequence<N>{});
  }

private:
  std::tuple<Layer<Ins, Outs, Activations>...> layers_;
};


template<int N>
struct Linear {
  inline Eigen::Vector<double, N> activation(Eigen::Vector<double, N> x) {
    return x;
  }

  inline Eigen::Matrix<double, N, N> activation_der(Eigen::Vector<double, N> x) {
    Eigen::Matrix<double, N, N> out = Eigen::Matrix<double, N, N>::Zero();
    for (int i = 0; i < x.rows(); i++) {
      out(i, i) = 1;
    }
    return out;
  }
};

template<int N>
struct MeanSquareError {
  typedef Eigen::Vector<double, N> Vec;
  inline double cost(Vec out, Vec exp_out) {
    auto errors = (out - exp_out).array();
    return (errors * errors).sum() / out.rows();
  }
  inline Vec cost_der(Vec out, Vec exp_out) {
    return (2.0 / out.rows()) * (out - exp_out);
  }
};

int main(){
  Eigen::Matrix<double, 2, 2> w1 = Eigen::Matrix<double, 2, 2>::Ones();
  Eigen::Matrix<double, 2, 2> w2 = Eigen::Matrix<double, 2, 2>::Ones();

  Eigen::Vector<double, 2> b1 = Eigen::Vector<double, 2>::Ones();
  Eigen::Vector<double, 2> b2 = Eigen::Vector<double, 2>::Ones();

  Layers<
    MeanSquareError,
    Layer<2, 2, Linear>, 
    Layer<2, 2, Linear> 
  > l;

  l.setWeights(w1, w2);
  l.setBiases(b1, b2);

  Eigen::Vector<double, 2> in1{1, 1};
  Eigen::Vector<double, 2> out1{3, 3};
  Eigen::Vector<double, 2> in2{3, 5};
  Eigen::Vector<double, 2> out2{6, 8};
  std::vector<Eigen::Vector<double, 2>> ins{in1, in2};
  std::vector<Eigen::Vector<double, 2>> outs{out1, out2};

  // cout << "FORWARD" << endl;
  // cout << l.forwardProp(in2) << endl << endl;

  cout << "WEIGHTS:" << endl;
  cout << l.getWeight<0>() << endl << endl;
  cout << l.getWeight<1>() << endl << endl;

  l.train(ins, outs, 1);

  cout << "WEIGHTS:" << endl;
  cout << l.getWeight<0>() << endl << endl;
  cout << l.getWeight<1>() << endl << endl;
}
