// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_reduction_Modelling_hpp
#define popops_reduction_Modelling_hpp

#include "popops/Operation.hpp"

#include <gccs/popsolver/Model.hpp>

// Forward declarations
namespace poplar {
class Target;
class Type;
} // namespace poplar

namespace popops {
namespace modelling {

// Forward declaration
class ExchangeEstimator;

// Set of estimates given when modelling a reduction
struct ReduceEstimates {
  ReduceEstimates(const gccs::popsolver::Variable &init)
      : cycles(init), cyclesBreakdown(init) {}
  gccs::popsolver::Variable cycles;
  struct CyclesBreakdown {
    CyclesBreakdown(const gccs::popsolver::Variable &init)
        : exchange(init), compute(init) {}
    gccs::popsolver::Variable exchange;
    gccs::popsolver::Variable compute;
  } cyclesBreakdown;
};

/**
 *   Model exchange and compute cost for an intertile reduction where
 *   \p mReductionFactor tiles each have \p mInputsPerTile partials, and
 *   that data must be reduced by \p mReductionFactor to give a total
 *   of \p mInputsPerTile outputs, spread over the input tiles in a
 *   balanced way. i.e.:
 *
 *
 *       mReductionFactor
 *  +-----+-----+-----+-----+ -+-
 *  |     |     |     |     |  |
 *  |   <-+-----+-----+---  |  |
 *  |     |     |     |     |  |
 *  |     |     |     |     |  |
 *  +-----+-----+-----+-----+  |
 *  |     |     |     |     |  |
 *  |  ---+-> <-+-----+---  |  |
 *  |     |     |     |     |  |
 *  |     |     |     |     |  |
 *  +-----+-----+-----+-----+  | mInputsPerTile
 *  |     |     |     |     |  |
 *  |  ---+-----+-> <-+---  |  |
 *  |     |     |     |     |  |
 *  |     |     |     |     |  |
 *  +-----+-----+-----+-----+  |
 *  |     |     |     |     |  |
 *  |  ---+-----+-----+->   |  |
 *  |     |     |     |     |  |
 *  +-----+-----+-----+-----+ -+-
 *
 *              |
 *              |
 *              v
 *
 *  +-----+-----+-----+-----+ -+-
 *  |     |     |     |     |  |  mInputsPerTile
 *  |     |     |     |     |  | ----------------
 *  |     |     |     |     |  | mReductionFactor
 *  |     |     |     +-----+  |
 *  +-----+-----+-----+       -+-
 *
 */
ReduceEstimates modelBalancedIntertileReduction(
    const poplar::Target &target, const poplar::Type &inputType,
    const poplar::Type &partialType, const popops::Operation &operation,
    const bool isUpdate, gccs::popsolver::Model &m,
    const ExchangeEstimator &exchangeEstimator,
    const gccs::popsolver::Variable &mInputsPerTile,
    const gccs::popsolver::Variable &mReductionFactor,
    const std::string &debugPrefix = "");

} // end namespace modelling
} // end namespace popops

#endif // popops_reduction_Modelling_hpp
