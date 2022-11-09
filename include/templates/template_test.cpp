#include <tuple>
#include <iostream>
#include <utility>

using namespace std;

template <int size>
struct Layer {
  char bias[size]{0};
  char weight[size*2]{0};
};

template <typename... Ls>
struct Layers {
  std::tuple<Ls...> layers;
};


template <typename... ls>
constexpr decltype(auto) print(Layers<ls...> layers){
  [] <std::size_t... I>
  (Layers<ls...> layers, std::index_sequence<I...>){
    ((cout << (int)std::get<I>(layers.layers).bias[0] << ", "), ...) << endl;
  }(std::forward<Layers<ls...>>(layers),
    std::index_sequence_for<ls...>{});
}

template <typename... ls>
constexpr int get_size(Layers<ls...> layers){
  int size = 0;

  [&size] <std::size_t... I>
  (Layers<ls...> layers, std::index_sequence<I...>){
    ((size += std::get<I>(layers.layers).bias[0]), ...);
  }(std::forward<Layers<ls...>>(layers),
    std::index_sequence_for<ls...>{});

  return size;
}



int main(){
  Layers<
    Layer<5>, 
    Layer<2>, 
    Layer<3>
  > l;

  std::get<0>(l.layers).bias[0] = 2;
  std::get<1>(l.layers).bias[0] = 3;
  std::get<2>(l.layers).bias[0] = 5;

  cout << "Sizesum: " << get_size(l) << endl;
  print(l);
}

