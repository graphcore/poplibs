// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef poplin_ExchangeEstimator_hpp
#define poplin_ExchangeEstimator_hpp

#include "ConvPlan.hpp"
#include "ConvPlanTypes.hpp"
#include "poplibs_support/Algorithm.hpp"
#include <cmath>
#include <poplar/Target.hpp>
#include <popsolver/Model.hpp>

namespace poplin {

class ExchangeEstimator {
  // Exchange bytes per cycle is given as a floating point value but the
  // constaint solver only supports unsigned integer variables. To reduce
  // quantization error in the calculation of the number of cycles we multiply
  // both the divisor (exchange bytes per cycle) and the dividend (the number of
  // bytes) by this scaling factor. Larger values of the scaling factor reduce
  // the quantization error but reduce the maximum number of bytes that can
  // be exchanged before running into the limits of the data type used to store
  // it.
  constexpr static unsigned exchangeBytesScalingFactor = 16u;

public:
  ExchangeEstimator(popsolver::Model &m, const poplar::Target &target,
                    const std::vector<double> &perLevelExchangeBytesPerCycle,
                    const unsigned numLevelsOfHierarchy,
                    const std::vector<PartitionVariables> &partitionVars,
                    const Plan::LinearizeTileOrder linearizeTileOrder)
      : m(m), target(target), numLevelsOfHierarchy(numLevelsOfHierarchy) {
    for (unsigned level = 0; level != numLevelsOfHierarchy - 1; ++level) {
      const auto scaledBytesPerCycle = getScaledExchangeBytesPerCycle(
          m, perLevelExchangeBytesPerCycle[level], exchangeBytesScalingFactor);

      perLevelScaledExchangeBytesPerCycle.push_back(scaledBytesPerCycle);
      perLevelScaledExchangeBytesPerCycleVar.push_back(
          m.addConstant(scaledBytesPerCycle));
    }

    const unsigned ipuLevel = numLevelsOfHierarchy - 2;
    scaledInputElementBytesPerCycle =
        perLevelScaledExchangeBytesPerCycleVar[ipuLevel];

    // when we lay the data out on the tiles (assuming the standard linearlize
    // tile order) we make the grouped output channels the innermost dimension.
    // this means that consecutive output channels will be distributed across
    // consecutive tiles. this is advantageous because when we parallel split by
    // output channels we need to broadcast out the same input elements to these
    // tiles. therefore the tiles that receive the same input elements will be
    // next to each other and therefore part of the same super tile. this
    // enables a higher bandwidth for receiving as both tiles can receive the
    // same data in the same cycle. we teach the planner about this here so that
    // it will bias splits towards making this happen and therefore produce
    // faster convolutions. for the implementation side of this see the function
    // `linearizeConvIndices` in Convolution.cpp
    //
    // it is worth mentioning that this decision to share inputs rather than
    // weights is arbitrary -- in the future we may want to let the planner
    // decide which is the innermost dimension and therefore gets a faster
    // exchange speed.
    if (target.supportsExchangeBusSharing() &&
        linearizeTileOrder == Plan::LinearizeTileOrder::STANDARD) {
      const auto tilesPerSuperTile = target.getTilesPerSharedExchangeBus();

      // don't care about the serial split here as that does not change the
      // tiles that the input elements are mapped to.
      const auto outChanSplit = partitionVars[ipuLevel].outChanSplit.parallel;
      const auto multiplier = m.call<unsigned>(
          {outChanSplit},
          [tilesPerSuperTile](const auto &values) -> popsolver::DataType {
            return popsolver::DataType{values[0] % tilesPerSuperTile == 0 ? 2
                                                                          : 1};
          });
      scaledInputElementBytesPerCycle =
          m.product({scaledInputElementBytesPerCycle, multiplier});
    }
  }

  popsolver::Variable
  getInputElementCycles(const popsolver::Variable numInputElements,
                        const poplar::Type inputElementType,
                        const unsigned level,
                        const std::string &debugName = "") const {
    const auto scaledInputElementSize = m.addConstant(
        target.getTypeSize(inputElementType) * exchangeBytesScalingFactor);

    const auto scaledInputElementBytes =
        m.product({numInputElements, scaledInputElementSize});

    if (level + 2 == numLevelsOfHierarchy) {
      return m.ceildiv(scaledInputElementBytes, scaledInputElementBytesPerCycle,
                       debugName);
    } else {
      return m.ceildiv(scaledInputElementBytes,
                       perLevelScaledExchangeBytesPerCycleVar[level],
                       debugName);
    }
  }

  popsolver::Variable getCycles(const popsolver::Variable numElements,
                                const poplar::Type elementType,
                                const unsigned level,
                                const std::string &debugName = "") const {
    const auto scaledSize = m.addConstant(target.getTypeSize(elementType) *
                                          exchangeBytesScalingFactor);

    const auto scaledElementBytes = m.product({numElements, scaledSize});
    return m.ceildiv(scaledElementBytes,
                     perLevelScaledExchangeBytesPerCycleVar[level], debugName);
  }

  unsigned getCycles(unsigned numElements, const poplar::Type elementType,
                     unsigned level) const {
    const unsigned scaledSize =
        target.getTypeSize(elementType) * exchangeBytesScalingFactor;
    const auto scaledElementBytes = numElements * scaledSize;
    return poplibs_support::ceildiv(scaledElementBytes,
                                    perLevelScaledExchangeBytesPerCycle[level]);
  }

private:
  static unsigned getScaledExchangeBytesPerCycle(popsolver::Model &m,
                                                 double exchangeBytesPerCycle,
                                                 unsigned scaleFactor) {
    auto scaledExchangeBytesPerCycle =
        std::round(exchangeBytesPerCycle * scaleFactor);
    // Ensure scaled bytes per cycle is at least one to avoid divide by zero
    // errors.
    scaledExchangeBytesPerCycle = std::max(1.0, scaledExchangeBytesPerCycle);
    // Saturate to the half the maximum unsigned integer value (we avoid the
    // maximum value to avoid range problems with the intermediate variables
    // used to implement ceildiv).
    scaledExchangeBytesPerCycle =
        std::min(scaledExchangeBytesPerCycle,
                 static_cast<double>(std::numeric_limits<unsigned>::max() / 2));
    return static_cast<unsigned>(scaledExchangeBytesPerCycle);
  }

  popsolver::Model &m;
  const poplar::Target &target;
  unsigned numLevelsOfHierarchy;
  std::vector<unsigned> perLevelScaledExchangeBytesPerCycle;
  std::vector<popsolver::Variable> perLevelScaledExchangeBytesPerCycleVar;

  // input elements can sometimes benefit from a fast bandwidth. see comment
  // in the constructor about why this is the case.
  popsolver::Variable scaledInputElementBytesPerCycle;
};

} // namespace poplin

#endif // poplin_ExchangeEstimator_hpp
