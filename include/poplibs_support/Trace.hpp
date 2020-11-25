// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_Trace_hpp
#define poplibs_support_Trace_hpp

#include <optional>
#include <poplar/Graph.hpp>

namespace poplibs_support {

template <typename F>
auto trace(poplar::Graph &graph, std::initializer_list<poplar::StringRef> names,
           const F &f) {
  using Ret = decltype(f());
  if constexpr (std::is_void_v<Ret>) {
    graph.trace(names, [&] { f(); });
  } else {
    std::optional<Ret> r;
    graph.trace(names, [&] { r = f(); });
    return std::move(*r);
  }
}

template <typename F>
auto trace(poplar::Graph &graph, poplar::StringRef name, const F &f) {
  return trace(graph, {name}, f);
}

} // namespace poplibs_support

#endif // poplibs_support_Trace_hpp
