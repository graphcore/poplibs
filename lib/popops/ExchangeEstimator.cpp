// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#include "ExchangeEstimator.hpp"

#include <poplar/Target.hpp>

#include <cmath>

using namespace poplar;

namespace popops {
namespace modelling {

// TODO: T41384, share common estimation code between poplibs libraries.

// Exchange bytes per cycle is given as a floating point value but the
// constaint solver only supports unsigned integer variables. To reduce
// quantization error in the calculation of the number of cycles we multiply
// both the divisor (exchange bytes per cycle) and the dividend (the number of
// bytes) by this scaling factor. Larger values of the scaling factor reduce
// the quantization error but reduce the maximum number of bytes that can
// be exchanged before running into the limits of the data type used to store
// it.
static constexpr unsigned exchangeBytesScalingFactor = 16u;

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

ExchangeEstimator::ExchangeEstimator(popsolver::Model &m, const Target &target)
    : m(m), target(target) {
  const auto scaledBytesPerCycle = getScaledExchangeBytesPerCycle(
      m, target.getExchangeBytesPerCycle(), exchangeBytesScalingFactor);

  scaledExchangeBytesPerCycle = scaledBytesPerCycle;
  scaledExchangeBytesPerCycleVar = m.addConstant(scaledBytesPerCycle);
}

popsolver::Variable
ExchangeEstimator::operator()(const popsolver::Variable mNumBytes,
                              const std::string &debugName) const {
  return getCycles(mNumBytes, debugName);
}

popsolver::Variable ExchangeEstimator::operator()(
    const popsolver::Variable mNumBytes,
    const popsolver::Variable mConsecutiveTilesReceivingSameData,
    const popsolver::Variable mTotalReceivingTiles,
    const std::string &debugName) const {
  return getCycles(mNumBytes, mConsecutiveTilesReceivingSameData,
                   mTotalReceivingTiles, debugName);
}

popsolver::Variable ExchangeEstimator::getCycles(
    const popsolver::Variable mNumBytes,
    const popsolver::Variable mConsecutiveTilesReceivingSameData,
    const popsolver::Variable mTotalReceivingTiles,
    const std::string &debugName) const {

  auto mScaledBytesPerCycle = scaledExchangeBytesPerCycleVar;
  assert(target.getTilesPerSharedExchangeBus() == 2);
  if (target.supportsExchangeBusSharing() &&
      target.getTilesPerSharedExchangeBus() == 2) {

    // In general the factor by which we can speed up the exchange by sharing
    // the exchange bus is the greatest common divisor of the number of
    // consecutive tiles receiving the same data and the number of tiles
    // sharing an exchange bus. A separate special case where we can always
    // share the exchange bus is when the number of consecutive tiles
    // receiving the same data is equal the number of tiles receiving data
    // (even if that number shared no common factor with the number of tiles
    // sharing the exchange bus > 1).
    //
    // Because gcd is hard to do in popsolver and because we only ever have
    // a maximum of 2 tiles sharing an exchange bus for current architecture
    // we assume 2 tiles share an exchange bus at most and the logic below
    // reflects this and would not work for more.
    const auto tilesSharingBus = target.getTilesPerSharedExchangeBus();
    const auto mTilesSharingBus = m.addConstant(tilesSharingBus);
    const auto mZeroWhenFullBroadcast =
        m.sub(mTotalReceivingTiles, mConsecutiveTilesReceivingSameData);
    const auto mZeroWhenCanShareBusAnyway =
        m.mod(mConsecutiveTilesReceivingSameData, mTilesSharingBus);
    const auto mZeroWhenCanShareBus =
        m.product({mZeroWhenFullBroadcast, mZeroWhenCanShareBusAnyway});
    const auto mCanShareBus =
        m.sub(m.one(), m.min({m.one(), mZeroWhenCanShareBus}));
    const auto mShareFactor = m.sum({m.one(), mCanShareBus});
    mScaledBytesPerCycle = m.product({mScaledBytesPerCycle, mShareFactor});
  }

  const auto mScalingFactor = m.addConstant(exchangeBytesScalingFactor);
  const auto mScaledBytes = m.product({mNumBytes, mScalingFactor});
  return m.ceildiv(mScaledBytes, mScaledBytesPerCycle, debugName);
}

popsolver::Variable
ExchangeEstimator::getCycles(const popsolver::Variable mNumBytes,
                             const std::string &debugName = "") const {
  const auto mScaledBytesPerCycle = scaledExchangeBytesPerCycleVar;
  const auto mScalingFactor = m.addConstant(exchangeBytesScalingFactor);
  const auto mScaledBytes = m.product({mNumBytes, mScalingFactor});
  return m.ceildiv(mScaledBytes, mScaledBytesPerCycle, debugName);
}

} // end namespace modelling
} // end namespace popops
