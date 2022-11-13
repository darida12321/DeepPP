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

// Select index
template <int I, int... Xs>
struct intlist_element;
template <int X, int... Xs>
struct intlist_element<0, X, Xs...> {
  static constexpr int elem = X;
};
template <int I, int X, int... Xs>
struct intlist_element<I, X, Xs...> : select_last<I-1, Xs...> {};


// Reverse index sequence
template <int... Ints>
struct reverse_index_sequence {};

template <std::size_t N, int... Is>
struct make_reverse_index_sequence
    : make_reverse_index_sequence<N - 1, Is..., N - 1> {};
template <int... Is>
struct make_reverse_index_sequence<0, Is...> : reverse_index_sequence<Is...> {};
