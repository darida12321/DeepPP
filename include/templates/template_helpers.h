#include <cstddef>
#include <utility>

/**
 * @brief Select the last parameter from a parameter pack
 * 
 * @tparam Args template parameters to select from
 */
template <size_t... Args>
struct select_last;
template <size_t A>
struct select_last<A> {
  static constexpr size_t elem = A;
};
template <size_t A, size_t... Args>
struct select_last<A, Args...> : select_last<Args...> {};

/**
 * @brief Select the first parameter from a parameter pack
 * 
 * @tparam Args template parameters to select from
 */
template <size_t... Args>
struct select_first;
template <size_t A, size_t... Args>
struct select_first<A, Args...> {
  static constexpr size_t elem = A;
};

/**
 * @brief Select an element from an integer parameter pack
 * 
 * @tparam I index
 * @tparam Xs template parameters to select from
 */
template <size_t I, size_t... Xs>
struct intlist_element;
template <size_t X, size_t... Xs>
struct intlist_element<0, X, Xs...> {
  static constexpr size_t elem = X;
};
template <size_t I, size_t X, size_t... Xs>
struct intlist_element<I, X, Xs...> : select_last<I-1, Xs...> {};


/**
 * @brief Reverse an index sequence
 * 
 */
template <size_t... Ints>
struct reverse_index_sequence {};

template <std::size_t N, size_t... Is>
struct make_reverse_index_sequence
    : make_reverse_index_sequence<N - 1, Is..., N - 1> {};
template <size_t... Is>
struct make_reverse_index_sequence<0, Is...> : reverse_index_sequence<Is...> {};
