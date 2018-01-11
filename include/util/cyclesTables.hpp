#ifndef __poplibs_cycles_tables__hpp__
#define __poplibs_cycles_tables__hpp__

#include <poplar/Target.hpp>
#include <poplar/VertexIntrospector.hpp>
#include <poplar/Graph.hpp>
#include <popstd/VertexTemplates.hpp>

#include <string>
#include <regex>
#include <iterator>
#include <functional>

/// These macros reduce the amount of boiler plate code when writing cycles-
/// estimators for codelets. E.g. for a templated codelet class
/// named 'MyCodelet':
///`std::uint64_t
/// MAKE_CYCLE_ESTIMATOR_NAME(MyCodelet)(const VertexIntrospector &vertex,
///                                      const Target &target,
///                                      args...) {
///  // compute your cycles estimate
///  vertex.getFieldInfo("myField"); // access to field size info
///  target.getDataPathWidth(); // access to target info
///  return estimate;
///}`

#define MAKE_CYCLE_ESTIMATOR_NAME(codelet) getCyclesEstimateFor ##codelet

#define CYCLE_ESTIMATOR_ENTRY(ns, codelet, ...) \
  poplibs::makeCycleEstimatorEntry( \
      #ns"::"#codelet, \
      MAKE_CYCLE_ESTIMATOR_NAME(codelet), \
      ## __VA_ARGS__)

// These macros reduce boiler plate code when accessing
// the codelet fields in cycle estimators:
#define CODELET_FIELD(field) \
  const auto field = vertex.getFieldInfo(#field)
#define CODELET_SCALAR_VAL(field, type) \
  const auto field = vertex.getFieldInfo(#field).getInitialValue<type>(target);
#define CODELET_VECTOR_VALS(field, type) \
  const auto field = vertex.getFieldInfo(#field).getInitialValues<type>(target);
#define CODELET_VECTOR_2D_VALS(field, type) \
  const auto field = vertex.getFieldInfo(#field) \
    .getInitialValues<std::vector<type>>(target);

namespace poplibs {

using CycleEstimatorTable = std::vector<
  std::pair<std::string, poplar::CycleEstimateFunc>>;

template <typename F, typename... Args>
inline std::pair<std::string, poplar::CycleEstimateFunc>
makeCycleEstimatorEntry(const std::string &codelet, F f, Args&&... args) {
  using std::placeholders::_1;
  using std::placeholders::_2;
  return std::make_pair(
           popstd::templateVertex(codelet, std::forward<Args>(args)...),
           std::bind(f, _1, _2, std::forward<Args>(args)...)
         );
}

inline void registerCyclesFunctions(poplar::Graph& graph,
                             const CycleEstimatorTable& table) {
  for (auto& kv : table) {
    graph.registerCycleEstimator(kv.first,
                                 poplar::CycleEstimateFunc(kv.second));
  }
}

} // end namespace poplibs

#endif
