// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplibs_support_HashTuple_hpp
#define poplibs_support_HashTuple_hpp

#include <tuple>
#include <utility>
#include <vector>

namespace poplibs_support {
namespace hash_tuple {

template <typename TT> struct hash {
  size_t operator()(TT const &tt) const { return std::hash<TT>()(tt); }
};

template <class T> inline void hash_combine(std::size_t &seed, T const &v) {
  seed ^= hash_tuple::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename TT> struct hash<std::vector<TT>> {
  size_t operator()(const std::vector<TT> &tt) const {
    size_t hash = 0;
    for (const auto e : tt)
      hash_combine(hash, e);
    return hash;
  }
};

namespace details {
template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
  void operator()(size_t &seed, Tuple const &tuple) const {
    HashValueImpl<Tuple, Index - 1>{}(seed, tuple);
    hash_combine(seed, std::get<Index>(tuple));
  }
};
template <class Tuple> struct HashValueImpl<Tuple, 0> {
  void operator()(size_t &seed, Tuple const &tuple) const {
    hash_combine(seed, std::get<0>(tuple));
  }
};
} // namespace details

template <typename... TT> struct hash<std::tuple<TT...>> {
  size_t operator()(std::tuple<TT...> const &tt) const {
    size_t seed = 0;
    details::HashValueImpl<std::tuple<TT...>>{}(seed, tt);
    return seed;
  }
};

} // namespace hash_tuple
} // namespace poplibs_support

#endif // poplibs_support_HashTuple_hpp
