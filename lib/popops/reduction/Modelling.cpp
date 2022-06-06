// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "Modelling.hpp"

#include "../ExchangeEstimator.hpp"

#include "CycleEstimationFunctions.hpp"
#include "ReductionVertexDefs.hpp"

#include <poplar/Target.hpp>
#include <poplar/Type.hpp>

#include <gccs/Algorithm.hpp>

using namespace poplar;
using namespace popops::modelling;
namespace popsolver = gccs::popsolver;

namespace popops {
namespace modelling {

ReduceEstimates modelBalancedIntertileReduction(
    const Target &target, const Type &inputType, const Type &partialsType,
    const popops::Operation &operation, const bool isUpdate,
    popsolver::Model &m, const ExchangeEstimator &exchangeEstimator,
    const popsolver::Variable &mInputsPerTile,
    const popsolver::Variable &mReductionFactor,
    const std::string &debugPrefix) {
  ReduceEstimates e(m.zero());

  const auto mNeedsReduction = m.reifiedLess(m.one(), mReductionFactor);

  // Estimate exchange as an all-to-all exchange where we receive an even
  // partition of the input data on each tile.
  const auto mBytesPerInputElem = m.addConstant(target.getTypeSize(inputType));
  const auto mOutputElemsBalancedBetweenTiles =
      m.ceildiv(mInputsPerTile, mReductionFactor);
  const auto mInputElemsPerTile =
      m.product({mReductionFactor, mOutputElemsBalancedBetweenTiles});
  const auto mInputBytesPerTile =
      m.product({mInputElemsPerTile, mBytesPerInputElem});

  // We only handle estimation for a single IPU hence this does not account for
  // potential global exchange cost.
  e.cyclesBreakdown.exchange = m.product(
      {mNeedsReduction, exchangeEstimator(mInputBytesPerTile,
                                          debugPrefix + ".cycles.exchange")});

  // We make the assumption that work is split amongst workers only using
  // the number of outputs on a tile (rather than splitting the reduced
  // dimension to give a multi-stage reduction). This may be a little
  // pessimistic or optimistic in some cases. We assume the gather
  // of partials from different tiles results in a contiguous chunk
  // of partials to be reduced on each tile, resulting in some set
  // of strided reduction vertices.
  const auto numWorkerContexts = target.getNumWorkerContexts();
  const auto vectorWidth = target.getVectorWidth(partialsType);
  const auto dataPathWidth =
      target.getDataPathWidth() / (8 * target.getTypeSize(partialsType));
  e.cyclesBreakdown.compute = m.product(
      {mNeedsReduction,
       m.call<unsigned>(
           {
               mReductionFactor,
               mOutputElemsBalancedBetweenTiles,
           },
           [=](const std::vector<unsigned> &values) {
             const auto reductionFactor = values[0];
             const auto outputsPerTile = values[1];
             const auto outputVectors =
                 gccs::ceildiv(outputsPerTile, vectorWidth);
             const auto outputVectorsPerWorker =
                 gccs::ceildiv(outputVectors, numWorkerContexts);
             const auto outputsPerWorker =
                 std::min(outputsPerTile, outputVectorsPerWorker * vectorWidth);
             const auto inputsPerWorker = outputsPerWorker * reductionFactor;
             const auto maxWorkerCycles = getCyclesEstimateForStridedReduce(
                 inputsPerWorker, reductionFactor, outputsPerWorker,
                 outputsPerTile, 1, dataPathWidth, vectorWidth, partialsType,
                 inputType, operation, isUpdate);
             return popsolver::DataType{maxWorkerCycles * numWorkerContexts};
           },
           debugPrefix + ".cycles.compute")});

  e.cycles = m.sum({e.cyclesBreakdown.exchange, e.cyclesBreakdown.compute});
  return e;
}

} // end namespace modelling
} // end namespace popops
