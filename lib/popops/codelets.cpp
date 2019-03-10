#include <popops/codelets.hpp>
#include "popopsCycleEstimators.hpp"
#include "reduction/CycleEstimationFunctions.hpp"
#include "reduction/ReductionVertex.hpp"
#include "poplibs_support/codelets.hpp"

namespace popops {

void addReduceCodelets(poplar::Graph &graph) {

  typedef std::vector<std::pair<poplar::Type, poplar::Type>> type_pairs;

  type_pairs fullTypes = {
    {poplar::FLOAT, poplar::FLOAT},
    {poplar::HALF, poplar::FLOAT},
    {poplar::FLOAT, poplar::HALF},
    {poplar::HALF, poplar::HALF},
    {poplar::INT, poplar::INT},
  };

  type_pairs equalTypes = {
    {poplar::FLOAT, poplar::FLOAT},
    {poplar::HALF, poplar::HALF},
    {poplar::INT, poplar::INT},
  };

  type_pairs boolTypes = {
    {poplar::BOOL, poplar::BOOL},
  };

  auto registerReduceCycleEstimators = [&](
                                       const type_pairs& types,
                                       popops::Operation operation
                                       ) {
    using std::placeholders::_1;
    using std::placeholders::_2;

    for (const auto &p : types) {
      for (bool isUpdate : {false, true}) {
        for (bool outputRegionSizeIsOne : {false, true}) {
          std::string opName = getReductionVertexOpName(operation);
          auto vertexName = getReductionVertexName(opName, p.first, p.second,
                                                   isUpdate,
                                                   outputRegionSizeIsOne);
          graph.registerCycleEstimator(
                vertexName,
                std::bind(getCycleEstimateForReduceVertex, _1, _2,
                          p.first, p.second, operation, isUpdate,
                          outputRegionSizeIsOne)
          );
        }
      }
    }
  };

  registerReduceCycleEstimators(fullTypes, popops::Operation::ADD);
  registerReduceCycleEstimators(fullTypes, popops::Operation::SQUARE_ADD);
  registerReduceCycleEstimators(fullTypes, popops::Operation::MUL);
  registerReduceCycleEstimators(equalTypes, popops::Operation::MAX);
  registerReduceCycleEstimators(equalTypes, popops::Operation::MIN);
  registerReduceCycleEstimators(boolTypes, popops::Operation::LOGICAL_AND);
  registerReduceCycleEstimators(boolTypes, popops::Operation::LOGICAL_OR);
}

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popops", "popops.gp", loc));
  poplibs::registerCyclesFunctions(graph, makeCyclesFunctionTable());
  addReduceCodelets(graph);
}

} // namespace popreduce
