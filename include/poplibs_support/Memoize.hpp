// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplibs_support_Memoize_hpp
#define poplibs_support_Memoize_hpp

#include "poplibs_support/HashTuple.hpp"
#include <tbb/concurrent_unordered_map.h>

namespace poplibs_support {

// A simple function to memoize other functions. Any recursive calls
// with the function are non memoized
template <typename Ret, typename... Args> class Memo {
  using Key = std::tuple<typename std::remove_reference<Args>::type...>;

public:
  tbb::concurrent_unordered_map<Key, Ret, hash_tuple::hash<Key>> table;
  Ret (*fn)(Args...);

public:
  Memo(Ret (*fn)(Args...)) : fn(fn) {}
  Ret operator()(Args... args) {
    const auto key = std::make_tuple(args...);
    const auto match = table.find(key);
    if (match == table.end()) {
      auto result = fn(args...);
      auto insertRes = table.insert({key, result});
      // another thread may have updated with the same key - in which case
      // it should be with the same value
      if (insertRes.second == false)
        assert(insertRes.first->second == result);
      return result;
    } else {
      return match->second;
    }
  }
  void clearTable() { table.clear(); }
};

template <typename Ret, typename... Args>
static Memo<Ret, Args...> memoize(Ret (*fn)(Args...)) {
  return Memo<Ret, Args...>(fn);
}

} // namespace poplibs_support

#endif // poplibs_support_Memoize_hpp
