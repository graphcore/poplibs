// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_ExchangeEstimator_hpp
#define popops_ExchangeEstimator_hpp

#include <string>
#include <vector>

#include <popsolver/Model.hpp>

// Forward declaration
namespace poplar {
class Target;
}

namespace popops {
namespace modelling {

class ExchangeEstimator {
public:
  ExchangeEstimator(popsolver::Model &m, const poplar::Target &target);

  // Return estimated cycles to exchange the given number of bytes via exchange
  // at the given level in the hierarchy.
  popsolver::Variable operator()(const popsolver::Variable mNumBytes,
                                 const std::string &debugName = "") const;

  // Return estimated cycles to exchange the given number of bytes via exchange
  // at the given level in the hierarchy. Information about broadcasting may
  // be given here to allow for double-width exchange.
  popsolver::Variable
  operator()(const popsolver::Variable mNumBytes,
             const popsolver::Variable mConsecutiveTilesReceivingSameData,
             const popsolver::Variable mTotalReceivingTiles,
             const std::string &debugName = "") const;

private:
  popsolver::Variable
  getCycles(const popsolver::Variable mNumBytes,
            const popsolver::Variable mConsecutiveTilesReceivingSameData,
            const popsolver::Variable mTotalReceivingTiles,
            const std::string &debugName = "") const;

  popsolver::Variable getCycles(const popsolver::Variable mNumBytes,
                                const std::string &debugName) const;

  popsolver::Model &m;
  const poplar::Target &target;
  unsigned scaledExchangeBytesPerCycle;
  popsolver::Variable scaledExchangeBytesPerCycleVar;
};

} // end namespace modelling
} // end namespace popops

#endif // popops_ExchangeEstimator_hpp
