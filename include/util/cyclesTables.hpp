#ifndef __poplibs_cycles_tables__hpp__
#define __poplibs_cycles_tables__hpp__

#include <poplar/Target.hpp>
#include <poplar/VertexIntrospector.hpp>
#include <poplar/Graph.hpp>
#include <popstd/VertexTemplates.hpp>

#include <string>
#include <regex>
#include <iterator>

// Removes spaces around commas:
inline std::string removeCommaSpaces(std::string s) {
  static const std::regex rgx("\\s*\\,\\s*");
  std::string result;
  result.reserve(s.size());
  std::regex_replace(std::back_insert_iterator<std::string>(result),
                     s.cbegin(), s.cend(),
                     rgx, ",");
  return result;
}

/// These macros reduce the amount of boiler plate code when writing cycles-
/// estimators for codelets. E.g. for a templated codelet class
/// named 'MyCodelet':
///`template <class T> MAKE_CYCLE_ESTIMATOR(MyCodelet,vertex,target) {
///  // compute your cycles estimate
///  vertex.getFieldInfo("myField"); // access to field size info
///  target.getDataPathWidth(); // access to target info
///  return estimate;
///}`
#define MAKE_CYCLE_ESTIMATOR_NAME(codelet) getCyclesEstimateFor ##codelet
#define CYCLES_PARAMS(v,t) (const poplar::VertexIntrospector& v, \
                            const poplar::Target& t)
#define MAKE_CYCLE_ESTIMATOR(codelet,v,t) \
std::uint64_t  MAKE_CYCLE_ESTIMATOR_NAME(codelet) CYCLES_PARAMS(v,t)

// The macros below facilitate construction of the static
// cycles estimator function tables. There are 3
// combinations of codelet and function templatisation:

// This macros is to be used when the codelet is templated
// but the cycle estimator need not be (i.e. cycles count is
// independent of type)
#define TYPED_CYCLE_ESTIMATOR_ENTRY(ns, codelet, TYPES...) \
  std::make_pair(popstd::templateVertex(#ns"::"#codelet, \
                                        removeCommaSpaces(#TYPES)), \
  MAKE_CYCLE_ESTIMATOR_NAME(codelet))

// This macros is to be used when both the codelet and the cycles
// estimator function are templated (identically).
#define TEMPLATE_CYCLE_ESTIMATOR_ENTRY(ns, codelet, TYPES...) \
  std::make_pair(popstd::templateVertex(#ns"::"#codelet, \
                                         removeCommaSpaces(#TYPES)), \
  MAKE_CYCLE_ESTIMATOR_NAME(codelet)<TYPES>)

// This macros is to be used when neither the codelet or cycles
// estimator function are templated.
#define CYCLE_ESTIMATOR_ENTRY(ns, codelet) \
  std::make_pair(#ns"::"#codelet, \
  MAKE_CYCLE_ESTIMATOR_NAME(codelet))

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

inline void registerCyclesFunctions(poplar::Graph& graph,
                             const CycleEstimatorTable& table) {
  for (auto& kv : table) {
    graph.registerCycleEstimator(kv.first,
                                 poplar::CycleEstimateFunc(kv.second));
  }
}

} // end namespace poplibs

#endif
