// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_print_hpp
#define poplibs_support_print_hpp

#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <type_traits>
#include <vector>

// Overload to print a pair
template <class A, class B>
static std::ostream &operator<<(std::ostream &os, std::pair<A, B> const &pair) {
  os << "{" << pair.first << ", " << pair.second << "}";
  return os;
}
// Overload ostream output for containers.

// Enable printing of specific containers
template <class T> struct is_container : std::false_type {};
template <class T, class A>
struct is_container<std::vector<T, A>> : std::true_type {};
template <class T, class S, class A>
struct is_container<std::set<T, S, A>> : std::true_type {};
template <class K, class T, class C, class A>
struct is_container<std::map<K, T, C, A>> : std::true_type {};

// forward declation to support nested containers
template <class Container>
typename std::enable_if<is_container<Container>::value, std::ostream &>::type
operator<<(std::ostream &os, const Container &container);

#ifdef __clang__
// -Wrange-loop-analysis does not work here because this is called with
// containers that iterate over values, *and* containers that iterate over
// references, so the range loop analysis always complains about one or the
// other.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wrange-loop-analysis"
#endif

// Print the elements in a container in the form {a,b,c,d}
template <class T>
static void printContainer(const T &container, std::ostream &os) {
  os << "{";
  bool needComma = false;
  for (const auto &x : container) {
    if (needComma)
      os << ',';
    os << x;
    needComma = true;
  }
  os << '}';
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template <class Container>
typename std::enable_if<is_container<Container>::value, std::ostream &>::type
operator<<(std::ostream &os, const Container &container) {
  printContainer(container, os);
  return os;
}

namespace poplibs_support {

template <class Container>
typename std::enable_if<is_container<Container>::value, std::string>::type
toString(const Container &container) {
  std::stringstream ss;
  printContainer(container, ss);
  return ss.str();
}

} // namespace poplibs_support

#endif // poplibs_support_print_hpp
