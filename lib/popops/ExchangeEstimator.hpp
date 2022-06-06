// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef popops_ExchangeEstimator_hpp
#define popops_ExchangeEstimator_hpp

#include <gccs/popsolver/Model.hpp>

#include <string>
#include <vector>

// Forward declaration
namespace poplar {
class Target;
}

namespace popops {
namespace modelling {

class ExchangeEstimator {
public:
  ExchangeEstimator(gccs::popsolver::Model &m, const poplar::Target &target);

  // Return estimated cycles to exchange the given number of bytes via exchange
  // at the given level in the hierarchy.
  gccs::popsolver::Variable
  operator()(const gccs::popsolver::Variable mNumBytes,
             const std::string &debugName = "") const;

  // Return estimated cycles to exchange the given number of bytes via exchange
  // at the given level in the hierarchy. Information about broadcasting may
  // be given here to allow for double-width exchange.
  gccs::popsolver::Variable
  operator()(const gccs::popsolver::Variable mNumBytes,
             const gccs::popsolver::Variable mConsecutiveTilesReceivingSameData,
             const gccs::popsolver::Variable mTotalReceivingTiles,
             const std::string &debugName = "") const;

private:
  gccs::popsolver::Variable
  getCycles(const gccs::popsolver::Variable mNumBytes,
            const gccs::popsolver::Variable mConsecutiveTilesReceivingSameData,
            const gccs::popsolver::Variable mTotalReceivingTiles,
            const std::string &debugName = "") const;

  gccs::popsolver::Variable getCycles(const gccs::popsolver::Variable mNumBytes,
                                      const std::string &debugName) const;

  gccs::popsolver::Model &m;
  const poplar::Target &target;
  unsigned scaledExchangeBytesPerCycle;
  gccs::popsolver::Variable scaledExchangeBytesPerCycleVar;
};

} // end namespace modelling
} // end namespace popops

#endif // popops_ExchangeEstimator_hpp
