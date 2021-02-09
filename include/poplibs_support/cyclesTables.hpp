// Copyright (c) 2017 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_cyclesTables_hpp
#define poplibs_support_cyclesTables_hpp

#include <poplar/Graph.hpp>
#include <poplar/PerfEstimateFunc.hpp>
#include <poplar/Target.hpp>
#include <poplar/VertexIntrospector.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <functional>
#include <iterator>
#include <regex>
#include <string>

/// These macros reduce the amount of boiler plate code when writing perf-
/// estimators for codelets. E.g. for a templated codelet class
/// named 'MyCodelet':
///`poplar::VertexPerfEstimate
/// MAKE_PERF_ESTIMATOR_NAME(MyCodelet)(const VertexIntrospector &vertex,
///                                     const Target &target,
///                                     args...) {
///  // compute your performance estimate
///  vertex.getFieldInfo("myField"); // access to field size info
///  target.getDataPathWidth(); // access to target info
///  return estimate;
///}`

#define MAKE_PERF_ESTIMATOR_NAME(codelet) getPerfEstimateFor##codelet
#define CYCLE_ESTIMATOR_ENTRY_NOPARAMS(ns, codelet)                            \
  poplibs::makePerfEstimatorEntry(#ns "::" #codelet,                           \
                                  MAKE_PERF_ESTIMATOR_NAME(codelet))
#define CYCLE_ESTIMATOR_ENTRY(ns, codelet, ...)                                \
  poplibs::makePerfEstimatorEntry(                                             \
      #ns "::" #codelet, MAKE_PERF_ESTIMATOR_NAME(codelet), __VA_ARGS__)

// These macros reduce boiler plate code when accessing
// the codelet fields in cycle estimators:
#define CODELET_FIELD(field) const auto field = vertex.getFieldInfo(#field)
#define CODELET_SCALAR_VAL(field, type)                                        \
  const auto field = vertex.getFieldInfo(#field).getInitialValue<type>(target);
#define CODELET_VECTOR_VALS(field, type)                                       \
  const auto field = vertex.getFieldInfo(#field).getInitialValues<type>(target);
#define CODELET_VECTOR_2D_VALS(field, type)                                    \
  const auto field =                                                           \
      vertex.getFieldInfo(#field).getInitialValues<std::vector<type>>(target);

namespace poplibs {

using PerfEstimatorTable =
    std::vector<std::pair<std::string, poplar::PerfEstimateFunc>>;

template <typename F, typename... Args>
inline std::pair<std::string, poplar::PerfEstimateFunc>
makePerfEstimatorEntry(const std::string &codelet, F f, Args &&...args) {
  using std::placeholders::_1;
  using std::placeholders::_2;
  return std::make_pair(
      poputil::templateVertex(codelet, std::forward<Args>(args)...),
      std::bind(f, _1, _2, std::forward<Args>(args)...));
}

inline void registerPerfFunctions(poplar::Graph &graph,
                                  const PerfEstimatorTable &table) {
  for (auto &kv : table) {
    graph.registerPerfEstimator(kv.first, poplar::PerfEstimateFunc(kv.second));
  }
}

inline std::uint64_t getUnpackCost(const poplar::layout::Vector layout) {
  switch (layout) {
  case poplar::layout::Vector::NotAVector:
  case poplar::layout::Vector::Span:
  case poplar::layout::Vector::OnePtr:
    return 0;
  case poplar::layout::Vector::ShortSpan:
    return 2;
  case poplar::layout::Vector::ScaledPtr32:
    // shl + add (note: doesn't include setzi that is shared among fields)
    return 2;
  case poplar::layout::Vector::ScaledPtr64:
  case poplar::layout::Vector::ScaledPtr128:
    // shl
    return 1;
  }

  throw poputil::poplibs_error("Unknown layout");
}

inline std::uint64_t getUnpackCost(const poplar::layout::VectorList layout) {
  switch (layout) {
  case poplar::layout::VectorList::NotAVector:
  case poplar::layout::VectorList::OnePtr:
  case poplar::layout::VectorList::ScaledPtr32:
  case poplar::layout::VectorList::ScaledPtr64:
  case poplar::layout::VectorList::ScaledPtr128:
    throw poputil::poplibs_error("Unknown cost for this layout");
  case poplar::layout::VectorList::DeltaN:
    return poplibs::getUnpackCost(poplar::layout::Vector::ScaledPtr32);
  case poplar::layout::VectorList::DeltaNElements:
    // (shl ; and) to get base and nA and again to get deltaN and nB then an
    // add to combine nA and nB
    return 5;
  }

  throw poputil::poplibs_error("Unknown layout");
}

inline std::uint64_t
getVectorListDeltaUnpackCost(const poplar::layout::VectorList layout) {
  switch (layout) {
  case poplar::layout::VectorList::NotAVector:
  case poplar::layout::VectorList::OnePtr:
  case poplar::layout::VectorList::ScaledPtr32:
  case poplar::layout::VectorList::ScaledPtr64:
  case poplar::layout::VectorList::ScaledPtr128:
    throw poputil::poplibs_error("Unknown cost for this layout");
  case poplar::layout::VectorList::DeltaN:
    // shl (offset), shr ; shl (n)
    return 3;
  case poplar::layout::VectorList::DeltaNElements:
    // shl (offset), and (n)
    return 2;
  }

  throw poputil::poplibs_error("Unknown layout");
}

} // end namespace poplibs

#endif // poplibs_support_cyclesTables_hpp
