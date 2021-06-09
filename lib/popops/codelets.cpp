// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poplibs_support/codelets.hpp"
#include "popopsCycleEstimators.hpp"
#include "reduction/CycleEstimationFunctions.hpp"
#include "reduction/ReductionConnection.hpp"
#include "reduction/ReductionVertex.hpp"
#include <popops/codelets.hpp>

namespace popops {

void addReduceCodelets(poplar::Graph &graph) {

  typedef std::vector<std::pair<poplar::Type, poplar::Type>> type_pairs;

  type_pairs fullTypes = {
      {poplar::FLOAT, poplar::FLOAT}, {poplar::HALF, poplar::FLOAT},
      {poplar::FLOAT, poplar::HALF},  {poplar::HALF, poplar::HALF},
      {poplar::INT, poplar::INT},
  };

  type_pairs equalTypes = {
      {poplar::FLOAT, poplar::FLOAT},
      {poplar::HALF, poplar::HALF},
      {poplar::INT, poplar::INT},
  };

  type_pairs fpTypes = {
      {poplar::FLOAT, poplar::FLOAT},
      {poplar::HALF, poplar::HALF},
      {poplar::FLOAT, poplar::HALF},
      {poplar::HALF, poplar::FLOAT},
  };

  type_pairs boolTypes = {
      {poplar::BOOL, poplar::BOOL},
  };

  auto registerReduceCycleEstimators = [&](const type_pairs &types,
                                           popops::Operation operation) {
    using std::placeholders::_1;
    using std::placeholders::_2;

    for (const auto &p : types) {
      for (bool isUpdate : {false, true}) {
        // specialisations without scaling
        for (auto specialisation :
             {ReductionSpecialisation::DEFAULT,
              ReductionSpecialisation::SCALAR_OUTPUT_REGIONS,
              ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT,
              ReductionSpecialisation::STRIDED_REDUCE,
              ReductionSpecialisation::ALL_REGIONS_CONTINUOUS}) {
          // continuous reductions do not take the specialisation
          // as a template parameter
          std::string opName = getReductionVertexOpName(operation);
          auto vertexName = getReductionVertexName(opName, p.first, p.second,
                                                   isUpdate, specialisation);
          graph.registerPerfEstimator(
              vertexName,
              std::bind(getCycleEstimateForReduceVertex, _1, _2, p.first,
                        p.second, operation, isUpdate, specialisation));
        }

        // specialisations with scaling (Only SCALAR_OUTPUT_SINGLE_INPUT
        // does not at present)
        for (auto specialisation :
             {ReductionSpecialisation::DEFAULT,
              ReductionSpecialisation::SCALAR_OUTPUT_REGIONS,
              ReductionSpecialisation::STRIDED_REDUCE,
              ReductionSpecialisation::ALL_REGIONS_CONTINUOUS}) {
          std::string opName = getReductionVertexOpName(operation);
          auto vertexName = getReductionVertexName(
              opName, p.first, p.second, isUpdate, specialisation, true);
          graph.registerPerfEstimator(
              vertexName,
              std::bind(getCycleEstimateForReduceVertex, _1, _2, p.first,
                        p.second, operation, isUpdate, specialisation));
        }
      }
    }
  };

  registerReduceCycleEstimators(fullTypes, popops::Operation::ADD);
  registerReduceCycleEstimators(fullTypes, popops::Operation::SQUARE_ADD);
  registerReduceCycleEstimators(fpTypes, popops::Operation::LOG_ADD);
  registerReduceCycleEstimators(fullTypes, popops::Operation::MUL);
  registerReduceCycleEstimators(equalTypes, popops::Operation::MAX);
  registerReduceCycleEstimators(equalTypes, popops::Operation::MIN);
  registerReduceCycleEstimators(boolTypes, popops::Operation::LOGICAL_AND);
  registerReduceCycleEstimators(boolTypes, popops::Operation::LOGICAL_OR);
}

void addCodelets(poplar::Graph &graph) {
  static poplibs::CurrentLibLocator loc;
  graph.addCodelets(poplibs::getCodeletsPath("popops", "popops.gp", loc));
  poputil::registerPerfFunctions(graph, makePerfFunctionTable());
  addReduceCodelets(graph);
}

} // namespace popops
