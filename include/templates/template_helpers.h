#include <utility>

// Select last argument
template <int... Args>
struct select_last;
template <int A>
struct select_last<A> {
  static constexpr int elem = A;
};
template <int A, int... Args>
struct select_last<A, Args...> : select_last<Args...> {};

// Select first argument
template <int... Args>
struct select_first;
template <int A, int... Args>
struct select_first<A, Args...> {
  static constexpr int elem = A;
};

// Reverse index sequence
template <int... Ints>
struct reverse_index_sequence {};

template <std::size_t N, int... Is>
struct make_reverse_index_sequence
    : make_reverse_index_sequence<N - 1, Is..., N - 1> {};
template <int... Is>
struct make_reverse_index_sequence<0, Is...> : reverse_index_sequence<Is...> {};
