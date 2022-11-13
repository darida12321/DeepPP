#include <cstddef>
#include <utility>

// Select last argument
template <size_t... Args>
struct select_last;
template <size_t A>
struct select_last<A> {
  static constexpr size_t elem = A;
};
template <size_t A, size_t... Args>
struct select_last<A, Args...> : select_last<Args...> {};

// Select first argument
template <size_t... Args>
struct select_first;
template <size_t A, size_t... Args>
struct select_first<A, Args...> {
  static constexpr size_t elem = A;
};

// Select index
template <size_t I, size_t... Xs>
struct intlist_element;
template <size_t X, size_t... Xs>
struct intlist_element<0, X, Xs...> {
  static constexpr size_t elem = X;
};
template <size_t I, size_t X, size_t... Xs>
struct intlist_element<I, X, Xs...> : select_last<I-1, Xs...> {};


// Reverse index sequence
template <size_t... Ints>
struct reverse_index_sequence {};

template <std::size_t N, size_t... Is>
struct make_reverse_index_sequence
    : make_reverse_index_sequence<N - 1, Is..., N - 1> {};
template <size_t... Is>
struct make_reverse_index_sequence<0, Is...> : reverse_index_sequence<Is...> {};
