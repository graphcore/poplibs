// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "FullyConnectedPlan.hpp"
#include "PerformanceEstimation.hpp"
#include "SparseMetaInfo.hpp"

#include "popsparse/FullyConnectedParams.hpp"

#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/exceptions.hpp"

#include "popsolver/Model.hpp"

#include "FullyConnectedOptions.hpp"
#include "FullyConnectedUtils.hpp"
#include "PlanningCacheImpl.hpp"
#include "popsparse/FullyConnected.hpp"

#include <gccs/Algorithm.hpp>

#include <map>
#include <utility>
#include <vector>

using namespace poplar;
using namespace poplibs_support;

// TODO: share this across files
using MetaInfoType = unsigned short;

namespace popsparse {

using namespace dynamic;

namespace fullyconnected {

namespace {

using MetaInfoType = unsigned short;
static const auto deviceMetaInfoType =
    poplar::equivalent_device_type<MetaInfoType>().value;

using CostBreakdown = std::vector<std::pair<std::string, Cost>>;

using CostVariables = Estimates<popsolver::Variable>;
using CostBreakdownVariables =
    std::vector<std::pair<std::string, CostVariables>>;

static Cost highestCost(popsolver::DataType::max(), popsolver::DataType::max());

// TODO: This can easily be shared along with other stuff with the
// (dense) convolution library.
class PlanningObjective {
public:
  enum Type { MINIMIZE_CYCLES, MINIMIZE_TILE_TEMP_MEMORY };

private:
  Type type;
  popsolver::DataType cyclesBound = popsolver::DataType::max();
  popsolver::DataType tileTempMemoryBound = popsolver::DataType::max();
  PlanningObjective(Type type) : type(type) {}

public:
  PlanningObjective() {}
  static PlanningObjective minimizeCycles() {
    return PlanningObjective(MINIMIZE_CYCLES);
  }
  static PlanningObjective minimizeTileTempMemory() {
    return PlanningObjective(MINIMIZE_TILE_TEMP_MEMORY);
  }
  PlanningObjective &setCyclesBound(popsolver::DataType bound) {
    assert(type != MINIMIZE_CYCLES);
    assert(*bound > 0);
    cyclesBound = bound;
    return *this;
  }
  PlanningObjective &setTileTempMemoryBound(popsolver::DataType bound) {
    assert(type != MINIMIZE_TILE_TEMP_MEMORY);
    assert(*bound > 0);
    tileTempMemoryBound = bound;
    return *this;
  }
  popsolver::DataType getCyclesBound() const { return cyclesBound; }
  popsolver::DataType getTileTempMemoryBound() const {
    return tileTempMemoryBound;
  }
  Type getType() const { return type; }
  bool lowerCost(Cost a, Cost b) const {
    bool aCyclesOutOfBounds = a.cycles >= cyclesBound;
    bool bCyclesOutOfBounds = b.cycles >= cyclesBound;
    bool aMemoryOutOfBounds = a.tempBytes >= tileTempMemoryBound;
    bool bMemoryOutOfBounds = b.tempBytes >= tileTempMemoryBound;
    switch (type) {
    case MINIMIZE_CYCLES:
      return std::tie(aCyclesOutOfBounds, aMemoryOutOfBounds, a.cycles,
                      a.tempBytes) < std::tie(bCyclesOutOfBounds,
                                              bMemoryOutOfBounds, b.cycles,
                                              b.tempBytes);
    case MINIMIZE_TILE_TEMP_MEMORY:
      return std::tie(aMemoryOutOfBounds, aCyclesOutOfBounds, a.tempBytes,
                      a.cycles) < std::tie(bMemoryOutOfBounds,
                                           bCyclesOutOfBounds, b.tempBytes,
                                           b.cycles);
    }
    POPLIB_UNREACHABLE();
  }
};

// TODO: T41384, share common estimation code between poplibs libraries.
class ExchangeEstimator {
  // Exchange bytes per cycle is given as a floating point value but the
  // constraint solver only supports unsigned integer variables. To reduce
  // quantization error in the calculation of the number of cycles we multiply
  // both the divisor (exchange bytes per cycle) and the dividend (the number of
  // bytes) by this scaling factor. Larger values of the scaling factor reduce
  // the quantization error but reduce the maximum number of bytes that can
  // be exchanged before running into the limits of the data type used to store
  // it.
  constexpr static unsigned exchangeBytesScalingFactor = 16u;

public:
  ExchangeEstimator(popsolver::Model &m, const poplar::Target &target)
      : m(m), target(target) {
    const auto scaledBytesPerCycle = getScaledExchangeBytesPerCycle(
        m, target.getExchangeBytesPerCycle(), exchangeBytesScalingFactor);

    scaledExchangeBytesPerCycle = scaledBytesPerCycle;
    scaledExchangeBytesPerCycleVar = m.addConstant(scaledBytesPerCycle);
  }

  popsolver::Variable operator()(const popsolver::Variable mNumBytes,
                                 const std::string &debugName = "") const {
    return getCycles(mNumBytes, debugName);
  }

  popsolver::Variable
  operator()(const popsolver::Variable mNumBytes,
             const popsolver::Variable mConsecutiveTilesReceivingSameData,
             const popsolver::Variable mTotalReceivingTiles,
             const std::string &debugName = "") const {
    return getCycles(mNumBytes, mConsecutiveTilesReceivingSameData,
                     mTotalReceivingTiles, debugName);
  }

  unsigned operator()(unsigned numBytes) const {
    const unsigned scalingFactor = exchangeBytesScalingFactor;
    const auto scaledElementBytes = numBytes * scalingFactor;
    return gccs::ceildiv(scaledElementBytes, scaledExchangeBytesPerCycle);
  }

private:
  popsolver::Variable
  getCycles(const popsolver::Variable mNumBytes,
            const popsolver::Variable mConsecutiveTilesReceivingSameData,
            const popsolver::Variable mTotalReceivingTiles,
            const std::string &debugName = "") const {

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

  popsolver::Variable getCycles(const popsolver::Variable mNumBytes,
                                const std::string &debugName = "") const {
    const auto mScaledBytesPerCycle = scaledExchangeBytesPerCycleVar;
    const auto mScalingFactor = m.addConstant(exchangeBytesScalingFactor);
    const auto mScaledBytes = m.product({mNumBytes, mScalingFactor});
    return m.ceildiv(mScaledBytes, mScaledBytesPerCycle, debugName);
  }

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
  unsigned scaledExchangeBytesPerCycle;
  popsolver::Variable scaledExchangeBytesPerCycleVar;
};

// Contains variables describing partitions. Only one form canonically describes
// the partitions, but it is useful to be able to store this information in
// redundant forms to avoid recomputing different forms/combinations of
// partitions all over the place.
struct PartitionVariables {
  // Partitions in each dimension at each level.
  std::vector<Vector<popsolver::Variable>> partition;
  // Product of the partitions of each dimension in each level.
  std::vector<popsolver::Variable> product;
  // Number of tile-level partitions at and below each level.
  // i.e. productByLevel[level] * productByLevel[level + 1]
  // .. * productByLevel[maxLevels]
  std::vector<popsolver::Variable> tile;
  // Cumulative product of partitions at each level and all levels
  // higher than it.
  std::vector<Vector<popsolver::Variable>> cumulative;

  PartitionVariables(
      popsolver::Model &m,
      const std::vector<Vector<popsolver::Variable>> &mPartitions)
      : partition(mPartitions), product(mPartitions.size()),
        tile(mPartitions.size() + 1), cumulative(mPartitions.size() + 1) {

    // Calculate products of partitions
    for (unsigned level = 0; level < partition.size(); ++level) {
      product[level] = m.product(partition[level].asStdVector());
    }
    // Calculate no. of tile-level partitions at each level
    tile[mPartitions.size()] = m.one();
    for (int level = partition.size() - 1; level >= 0; --level) {
      tile[level] = m.product({product[level], tile[level + 1]});
    }
    // Calculate cumulative partitions
    cumulative[0] =
        Vector<popsolver::Variable>::generate([&] { return m.one(); });
    for (unsigned level = 1; level < partition.size() + 1; ++level) {
      cumulative[level] = partition[level - 1].binaryOp(
          cumulative[level - 1],
          [&](const auto &partition, const auto &cumulativePrev) {
            return m.product({partition, cumulativePrev});
          });
    }
  }
};

} // end anonymous namespace

static popsolver::Variable getArePartitionsOnConsecutivePNs(
    popsolver::Model &m, const PartitionVariables &p,
    const PartitionToPNMapping &mapping, unsigned level, const unsigned dim) {
  const auto &order = mapping.getLinearisationOrder().asStdVector<unsigned>();
  const auto &partition = p.partition[level].asStdVector();
  // Inverse order contains the ordering of dimensions as they
  // are linearised.
  std::vector<unsigned> inverseOrder(order.size());
  std::iota(inverseOrder.begin(), inverseOrder.end(), 0);
  for (std::size_t i = 0; i < order.size(); ++i) {
    inverseOrder[order[i]] = i;
  }

  // Partitions are on consecutive tiles if the product of all
  // partitions of inner dimensions in the ordering is 1.
  std::vector<popsolver::Variable> innerPartitions = {m.one()};
  for (std::size_t i = order[dim] + 1; i < inverseOrder.size(); ++i) {
    innerPartitions.push_back(partition[inverseOrder[i]]);
  }
  const auto mInnerPartitionsProduct = m.product(innerPartitions);
  const auto mInnerPartitionsProductM1 =
      m.sub(mInnerPartitionsProduct, m.one());
  // Re-ify (mInnerPartitionsProduct == 1) to a boolean represented by 0/1
  return m.sub(m.one(), m.min({m.one(), mInnerPartitionsProductM1}));
}

// This cost covers:
// * Pre-distribution exchange i.e. exchange of dense input to the fc layer
//   potentially broadcast across partitions of X.
// * Distribution exchange i.e. exchange of buckets required to complete
//   computation assuming a perfectly uniform distribution of sparsity.
//   In practice this means the exchange cost to broadcast each bucket on
//   each PN within a SORG to all PNs within a SORG.
static std::tuple<CostVariables, popsolver::Variable, popsolver::Variable>
addDistributionExchangeCostSparseDense(
    popsolver::Model &m, const Target &target, const Type &inputType,
    const Type &deviceMetaInfoType, const Options &options,
    const ExchangeEstimator &exchangeEstimator,
    const PartitionToPNMapping &mapping,
    const std::vector<Vector<popsolver::Variable>> &mGroups,
    const Vector<popsolver::Variable> &mGrouping,
    const popsolver::Variable &mRBytesPerBucket, const PartitionVariables &p) {

  const auto mBytesPerInput = m.addConstant(target.getTypeSize(inputType));

  std::vector<popsolver::Variable> mRBytesPerTile(2), mSBytesPerTile(2);
  for (unsigned level = 0; level < 2; ++level) {
    // Bytes per-tile for the dense input at each level are given by the
    // product of number of grains of each dimension of the input, spread
    // over the tiles that will eventually compute on those bytes.
    mSBytesPerTile[level] =
        m.product({m.ceildiv(m.product({mGroups[level].groups, mGroups[level].y,
                                        mGroups[level].z}),
                             p.tile[level]),
                   mGrouping.groups, mGrouping.y, mGrouping.z, mBytesPerInput});

    // In the initial distribution we broadcast the buckets of the sparse
    // operand across partitions processing the same X and Y partitions.
    // Buckets are constrained to be of equal size on all tiles so this
    // product will not introduce any error in the calculation moving up
    // levels of the hierarchy.
    mRBytesPerTile[level] =
        m.product({p.cumulative[level].z, mRBytesPerBucket});
  }

  // Estimate exchange for the initial distribution. We exchange input
  // operands s and r to the tiles that will process them during the first
  // compute step.
  //
  // Exchange cycles are calculated by finding the critical path for send
  // /receive of data. In this case the exchange will multi-cast data from
  // each tile within a particular set of partitions to all tiles in that
  // particular partition. The critical path then is the sending of each
  // chunk of data on each tile in series due to not being able to receive
  // on all tiles in parallel.
  //
  // Exchange temporary memory is more complex as this is dependent on the
  // need to gather the input operands into contiguous memory as part of the
  // exchange or not.
  //
  // There are 2 special cases:
  //
  // The first occurs when there is no broadcast
  // of data and we assume that inputs are allocated such that they are
  // already resident on each tile. There is no exchange and no temporary
  // memory requirement for these inputs in this case.
  //
  // TODO: The second case occurs when the data is only being
  // multi-cast to one other tile and/ we don't need to gather the data
  // into one contiguous region. In this case we can simultaneously
  // send/receive from both tiles in each set. This doesn't affect
  // single-IPU planning.
  popsolver::Variable mCycles, mTempBytes;
  popsolver::Variable mSTempBytesAfterExchange = m.zero(),
                      mRTempBytesAfterExchange = m.zero();
  unsigned level = 0;
  // If this is the last level then we need to gather the operand
  // S as this needs to be contiguous on-tile. TODO: We don't
  // need to gather at other levels so current estimation of temp
  // memory is exaggerated.
  const auto mSBytesAreExchanged =
      m.min({m.one(), m.sub(mSBytesPerTile[level + 1], mSBytesPerTile[level])});
  const auto mSBytesToSendReceivePerTile =
      m.product({mSBytesAreExchanged, mSBytesPerTile[level + 1]});
  const auto mSTempBytes = mSBytesToSendReceivePerTile;
  const auto mSBytesToSendReceive =
      m.product({mSBytesToSendReceivePerTile, p.tile[level + 1]});

  const auto mRBytesAreExchanged =
      m.min({m.one(), m.sub(mRBytesPerTile[level + 1], mRBytesPerTile[level])});
  const auto mRBytesToSendReceive = m.product(
      {mRBytesAreExchanged, p.tile[level + 1], mRBytesPerTile[level + 1]});
  // Because we never need to gather R temporary memory at any stage is
  // just the difference between the bytes for original locations of
  // buckets at level 0 and the current level.
  const auto mRTempBytes = m.sub(mRBytesPerTile[level + 1], mRBytesPerTile[0]);

  // Using our knowledge of how the source and destination of the exchange
  // will be laid out to allow the exchange estimator to account for the
  // possibility of exchange bus sharing between tiles during the broadcast
  // of information.
  //
  // We choose a tile to process this partition based on the flattened
  // index into a 3D array with shape {y,z,x}. This means that 2
  // partitions of x will be on neighbouring tiles and input S could be
  // broadcast. Alternatively if there is only 1 partition of x then
  // 2 partitions of z will be on neighbouring tiles.
  const auto mXPartitionsOnConsecutiveTiles =
      getArePartitionsOnConsecutivePNs(m, p, mapping, level, 1 /* X */);
  const auto mSConsecutiveTilesReceivingSameData = m.max(
      {m.one(),
       m.product({mXPartitionsOnConsecutiveTiles, p.partition[level].x})});
  const auto mZPartitionsOnConsecutiveTiles =
      getArePartitionsOnConsecutivePNs(m, p, mapping, level, 3 /* Z */);
  const auto mRConsecutiveTilesReceivingSameData = m.max(
      {m.one(),
       m.product({mZPartitionsOnConsecutiveTiles, p.partition[level].z})});

  const auto mSExchangeCycles =
      exchangeEstimator(mSBytesToSendReceive,
                        mSConsecutiveTilesReceivingSameData, p.product[level]);
  const auto mRExchangeCycles =
      exchangeEstimator(mRBytesToSendReceive,
                        mRConsecutiveTilesReceivingSameData, p.product[level]);
  mCycles = m.sum({mSExchangeCycles, mRExchangeCycles});

  mTempBytes = m.sum({mSTempBytesAfterExchange, mSTempBytes, mRTempBytes});
  mSTempBytesAfterExchange = mSTempBytes;
  mRTempBytesAfterExchange = mRTempBytes;

  CostVariables mCost(mCycles, mTempBytes);
  return std::make_tuple(mCost, mSTempBytesAfterExchange,
                         mRTempBytesAfterExchange);
}

/** Account for the cost of broadcasting/rearranging
 *  inputs/output gradients
 */
static CostVariables addPreDistributionExchangeCostDenseDense(
    popsolver::Model &m, const Options &options,
    const ExchangeEstimator &exchangeEstimator,
    const PartitionToPNMapping &mapping,
    const std::vector<popsolver::Variable> &mQGradBytesPerTile,
    const std::vector<popsolver::Variable> &mSBytesPerTile,
    popsolver::Variable &mQGradTempBytesAfterExchange,
    popsolver::Variable &mSTempBytesAfterExchange,
    const PartitionVariables &p) {

  // TODO: Add cost for exchanging meta-info when mapping order
  // does not match forward pass

  popsolver::Variable mCycles, mTempBytes;
  // Assuming the temporary memory for these operands first appears here.
  mQGradTempBytesAfterExchange = m.zero(), mSTempBytesAfterExchange = m.zero();
  unsigned level = 0;
  const auto mQGradBytesAreExchanged =
      m.min({m.one(),
             m.sub(mQGradBytesPerTile[level + 1], mQGradBytesPerTile[level])});
  const auto mQGradBytesToSendReceivePerTile =
      m.product({mQGradBytesAreExchanged, mQGradBytesPerTile[level + 1]});
  const auto mQGradTempBytes = mQGradBytesToSendReceivePerTile;
  const auto mQGradBytesToSendReceive =
      m.product({mQGradBytesToSendReceivePerTile, p.tile[level + 1]});

  const auto mSBytesAreExchanged =
      m.min({m.one(), m.sub(mSBytesPerTile[level + 1], mSBytesPerTile[level])});
  const auto mSBytesToSendReceivePerTile =
      m.product({mSBytesAreExchanged, mSBytesPerTile[level + 1]});
  const auto mSTempBytes = mSBytesToSendReceivePerTile;
  const auto mSBytesToSendReceive =
      m.product({mSBytesToSendReceivePerTile, p.tile[level + 1]});

  const auto mXPartitionsOnConsecutiveTiles =
      getArePartitionsOnConsecutivePNs(m, p, mapping, level, 1 /* X */);
  const auto mYPartitionsOnConsecutiveTiles =
      getArePartitionsOnConsecutivePNs(m, p, mapping, level, 2 /* Y */);
  const auto mQGradConsecutiveTilesReceivingSameData = m.max(
      {m.one(),
       m.product({mYPartitionsOnConsecutiveTiles, p.partition[level].y})});
  const auto mSConsecutiveTilesReceivingSameData = m.max(
      {m.one(),
       m.product({mXPartitionsOnConsecutiveTiles, p.partition[level].x})});

  // There should be as much data as the number of z partitions as there
  // is no reduction stage following this.
  // This assumes we have to move operands on-tile - we only cycle
  // operands between tiles z partitions - 1 times.
  const auto mQGradExchangeCycles = exchangeEstimator(
      mQGradBytesToSendReceive, mQGradConsecutiveTilesReceivingSameData,
      p.product[level]);
  const auto mSExchangeCycles =
      exchangeEstimator(mSBytesToSendReceive,
                        mSConsecutiveTilesReceivingSameData, p.product[level]);

  mCycles = m.sum({mQGradExchangeCycles, mSExchangeCycles});

  mTempBytes = m.sum({mQGradTempBytesAfterExchange, mSTempBytesAfterExchange,
                      mQGradTempBytes, mSTempBytes});
  mQGradTempBytesAfterExchange = mQGradTempBytes;
  mSTempBytesAfterExchange = mSTempBytes;

  return CostVariables(mCycles, mTempBytes);
}

static std::tuple<popsolver::Variable, popsolver::Variable>
addGradWExchangeAndComputeTempBytesCost(
    popsolver::Model &m, const Options &options, const Type &inputType,
    const bool exchangeBuckets,
    const popsolver::Variable &mRGradPartialBytesPerTile,
    const popsolver::Variable &mRMetaInfoBytesPerTile,
    const popsolver::Variable &mQGradBytesPerTile,
    const popsolver::Variable &mSBytesPerTile, const PartitionVariables &p) {
  popsolver::Variable mTempBytes;
  const auto mNeedsCast =
      m.addConstant(inputType != options.partialsType ? 1u : 0u);
  const auto mRemainingPartialBytes =
      m.product({mNeedsCast, mRGradPartialBytesPerTile});
  if (exchangeBuckets) {
    // If we exchange buckets, the peak temp memory during exchange and
    // compute phases is 2x the size of the partials & meta-info and
    // whatever temporary storage is required for the dense operands
    // s/q-grad.
    const auto mRGradBytesPerTile =
        m.sum({mRGradPartialBytesPerTile, mRMetaInfoBytesPerTile});
    mTempBytes = m.sum({mQGradBytesPerTile, mSBytesPerTile,
                        m.product({mRGradBytesPerTile, m.addConstant(2u)})});
  } else {
    // When exchanging inputs, the temporary memory is given by the memory
    // needed for partials (if needed) and 2x the size of the buffers for
    // dense operands s/q-grad.
    const auto mInputBytes = m.sum({mQGradBytesPerTile, mSBytesPerTile});
    mTempBytes = m.sum(
        {mRemainingPartialBytes, m.product({mInputBytes, m.addConstant(2u)})});
  }
  return std::make_tuple(mTempBytes, mRemainingPartialBytes);
}

/** Account for the cost of exchange in the distribution phase.
 *  The cost of exchange for this phase is any exchange required
 *  assuming a perfectly uniform sparsity pattern. This boils down
 *  to the cost of exchange required to complete one full cycle
 *  around the Z dimension.
 */
static popsolver::Variable addDistributionExchangeCycleCostDenseDense(
    popsolver::Model &m, const Options &options,
    const ExchangeEstimator &exchangeEstimator,
    const PartitionToPNMapping &mapping, const bool exchangeBuckets,
    const popsolver::Variable &mRGradBytesPerTile,
    const popsolver::Variable &mQGradBytesPerTile,
    const popsolver::Variable &mSBytesPerTile, const PartitionVariables &p) {

  popsolver::Variable mCycles = m.zero();
  unsigned level = 0;
  const auto mZPartitionsM1 = m.sub(p.partition[level].z, m.one());
  const auto mNeedsExchange = m.min({mZPartitionsM1, m.one()});
  if (exchangeBuckets) {
    const auto mBytesToSendReceivePerTile =
        mRGradBytesPerTile; // Non-zero value partials and meta-info.
    const auto mBytesToSendReceive = m.product(
        {mNeedsExchange, mBytesToSendReceivePerTile, p.tile[level + 1]});
    // We don't do any broadcasting when exchanging buckets hence no
    // calculation of consecutive tiles like below.
    const auto mExchangeCycles = m.product(
        {exchangeEstimator(mBytesToSendReceive), p.partition[level].z});
    mCycles = mExchangeCycles;
  } else {
    const auto mQGradBytesToSendReceivePerTile =
        m.product({mNeedsExchange, mQGradBytesPerTile, p.tile[level + 1]});
    const auto mSBytesToSendReceivePerTile =
        m.product({mNeedsExchange, mSBytesPerTile, p.tile[level + 1]});
    const auto mQGradBytesToSendReceive =
        m.product({mQGradBytesToSendReceivePerTile, p.tile[level + 1]});
    const auto mSBytesToSendReceive =
        m.product({mSBytesToSendReceivePerTile, p.tile[level + 1]});

    const auto mZPartitionsOnConsecutiveTiles =
        getArePartitionsOnConsecutivePNs(m, p, mapping, level, 3 /* Z */);
    const auto mConsecutiveTilesReceivingSameData = m.max(
        {m.one(),
         m.product({mZPartitionsOnConsecutiveTiles, p.partition[level].z})});

    const auto mQGradExchangeCycles =
        m.product({exchangeEstimator(mQGradBytesToSendReceive,
                                     mConsecutiveTilesReceivingSameData,
                                     p.product[level]),
                   p.partition[level].z});
    const auto mSExchangeCycles =
        m.product({exchangeEstimator(mSBytesToSendReceive,
                                     mConsecutiveTilesReceivingSameData,
                                     p.product[level]),
                   p.partition[level].z});
    mCycles = m.sum({mQGradExchangeCycles, mSExchangeCycles});
  }

  return mCycles;
}

static std::tuple<unsigned, unsigned> getNumGroupsGivenUniformSparsityPattern(
    const double nzRatio, const unsigned xGroups, const unsigned yGroups) {
  const double pGroupIsZero = 1.0 - nzRatio;
  const double pXGroupHasAllZeroGroups = std::pow(pGroupIsZero, yGroups);
  const double pXGroupHasNonZeroGroup = 1.0 - pXGroupHasAllZeroGroups;
  const unsigned totalNonZeroGroups = std::ceil(xGroups * yGroups * nzRatio);
  const unsigned xNonZeroGroups = std::ceil(xGroups * pXGroupHasNonZeroGroup);
  const unsigned yNonZeroGroups =
      gccs::ceildiv(totalNonZeroGroups, xNonZeroGroups);
  return std::make_tuple(xNonZeroGroups, yNonZeroGroups);
}

static inline unsigned getNumConvUnits(const Target &target,
                                       bool floatActivations,
                                       bool floatPartial) {
  if (floatActivations) {
    return target.getFp32InFp32OutConvUnitsPerTile();
  } else {
    return floatPartial ? target.getFp16InFp32OutConvUnitsPerTile()
                        : target.getFp16InFp16OutConvUnitsPerTile();
  }
}

static std::tuple<CostVariables, popsolver::Variable>
addDistributionComputeCostSparseDense(
    popsolver::Model &m, const Target &target, const Type &inputType,
    const double &nzRatio, const Options &options, const OnTileMethod &method,
    const Vector<popsolver::Variable> &mGroups,
    const Vector<popsolver::Variable> &mGrouping,
    const Vector<popsolver::Variable> &mCumulativePartitions,
    const popsolver::Variable &mSTempBytes,
    const popsolver::Variable &mRTempBytes) {

  // TODO: Padding estimates etc...

  const auto mPartialsPerTile =
      m.product({mGroups.groups, mGroups.x, mGroups.z, mGrouping.groups,
                 mGrouping.x, mGrouping.z});

  const auto numWorkers = target.getNumWorkerContexts();
  const auto partialsType = options.partialsType;
  const unsigned numConvUnits =
      getNumConvUnits(target, inputType == FLOAT, partialsType == FLOAT);
  const auto mBytesPerPartial = m.addConstant(target.getTypeSize(partialsType));
  const auto mNumBucketsPerTile = mCumulativePartitions.z;
  const auto mCycles = m.call<unsigned>(
      {mPartialsPerTile, mNumBucketsPerTile, mGroups.x, mGroups.y, mGroups.z,
       mGrouping.x, mGrouping.y, mGrouping.z},
      [=](const std::vector<unsigned> &values) -> popsolver::DataType {
        const auto partialsPerTile = values[0];
        const auto numBuckets = values[1];
        const auto xGroups = values[2];
        const auto yGroups = values[3];
        const auto zGroups = values[4];
        const auto xGrouping = values[5];
        const auto yGrouping = values[6];
        const auto zGrouping = values[7];
        const auto partialsPerWorker =
            gccs::ceildiv(partialsPerTile, numWorkers);
        std::uint64_t cycles = zeroPartialsCycles(partialsPerWorker, numWorkers,
                                                  options.partialsType == FLOAT,
                                                  xGrouping * yGrouping > 0);

        unsigned xNonZeroGroups, yNonZeroGroups;
        std::tie(xNonZeroGroups, yNonZeroGroups) =
            getNumGroupsGivenUniformSparsityPattern(nzRatio, xGroups, yGroups);

        std::vector<Tile> workerTiles;
        switch (method) {
        case OnTileMethod::Forward:
        case OnTileMethod::GradA:
          assert(xGrouping * yGrouping == 1);
          workerTiles =
              splitTileBetweenWorkers(xNonZeroGroups, zGroups, numWorkers);
          break;
        case OnTileMethod::Transpose:
          assert(xGrouping * yGrouping == 1);
          // Transpose vertex does its work split at runtime
          workerTiles = splitTileBetweenWorkers(1, 1, numWorkers);
        case OnTileMethod::ForwardAMPBlock:
        case OnTileMethod::TransposeAMPBlock:
          // We may only split Z amongst workers for forward/grad-a AMP
          // codelets
          workerTiles = splitTileBetweenWorkers(1, zGroups, numWorkers);
          break;
        default:
          throw poputil::poplibs_error("Unhandled OnTileMethod");
        }

        std::uint64_t maxMulCycles = 0;
        for (const auto &workerTile : workerTiles) {
          const unsigned workerXGroups = workerTile.getRows().size();
          const unsigned workerZGroups = workerTile.getColumns().size();
          const unsigned workerZElems = workerZGroups * zGrouping;
          const unsigned numY = yNonZeroGroups * yGrouping;

          // Because we are assuming best case with perfectly uniform
          // distribution of sparsity over the dense sparse of R, there should
          // be a perfect distribution of sub-groups over buckets such that each
          // bucket only contains elements of 1 sub-group.
          constexpr auto numSubGroupsPerBucket = 1u;

          std::uint64_t mulCycles = 0;
          switch (method) {
          case OnTileMethod::Forward:
            mulCycles = sparseDenseElementwiseMultiply(
                numBuckets, numBuckets, numSubGroupsPerBucket, workerXGroups,
                workerZElems,
                std::vector<unsigned>({static_cast<unsigned>(numY)}),
                inputType == FLOAT, partialsType == FLOAT, numWorkers);
            break;
          case OnTileMethod::GradA:
            mulCycles = sparseDenseGradAElementwiseMultiply(
                numBuckets, numBuckets, numSubGroupsPerBucket, workerXGroups,
                workerZElems,
                std::vector<unsigned>({static_cast<unsigned>(numY)}),
                inputType == FLOAT, partialsType == FLOAT, numWorkers);
            break;
          case OnTileMethod::Transpose:
            // The transpose method divides the work along the X-dimension.
            mulCycles = sparseDenseTransposeElementwiseMultiply(
                numBuckets, numBuckets, numSubGroupsPerBucket, numY,
                zGroups * zGrouping, std::vector<unsigned>({xGroups}),
                inputType == FLOAT, partialsType == FLOAT, numWorkers);
            break;
          case OnTileMethod::ForwardAMPBlock: {
            constexpr bool retainX = true;
            mulCycles = sparseDenseBlockMultiply(
                numBuckets, numBuckets, numSubGroupsPerBucket, xNonZeroGroups,
                workerZElems, xGrouping, yGrouping, {yNonZeroGroups},
                inputType == FLOAT, partialsType == FLOAT, numWorkers,
                numConvUnits, retainX);
            break;
          }
          case OnTileMethod::TransposeAMPBlock: {
            constexpr bool retainX = false;
            mulCycles = sparseDenseBlockMultiply(
                numBuckets, numBuckets, numSubGroupsPerBucket, yNonZeroGroups,
                workerZElems, yGrouping, xGrouping, {xNonZeroGroups},
                inputType == FLOAT, partialsType == FLOAT, numWorkers,
                numConvUnits, retainX);
            break;
          }
          default:
            throw poputil::poplibs_error("Unhandled method when planning");
          }

          maxMulCycles = std::max(maxMulCycles, mulCycles);
        }
        cycles += maxMulCycles;

        return popsolver::DataType{cycles};
      });

  // The temporary memory during this operation is the temporary memory for
  // both the inputs, and the memory for partial outputs. Memory for partial
  // outputs is only temporary if there is a cast or reduction to be done
  // later on.
  const auto mNeedsCast = m.addConstant(inputType != partialsType ? 1u : 0u);
  const auto mNeedsReduction = m.sub(mCumulativePartitions.z, m.one());
  const auto mNeedsCastOrReduction =
      m.min({m.one(), m.sum({mNeedsCast, mNeedsReduction})});

  const auto mPartialsTempBytes =
      m.product({mNeedsCastOrReduction, mPartialsPerTile, mBytesPerPartial});
  const auto mTempBytes = m.sum({mSTempBytes, mRTempBytes, mPartialsTempBytes});
  return std::make_tuple(CostVariables(mCycles, mTempBytes),
                         mPartialsTempBytes);
}

static std::pair<popsolver::Variable, popsolver::Variable>
rearrangeDenseCost(popsolver::Model &m, const Target &target,
                   const Type &dataType, const popsolver::Variable &mXOrYGroups,
                   const popsolver::Variable mXOrYGrouping,
                   const popsolver::Variable &mZGroups,
                   const popsolver::Variable &mZGrouping) {
  const auto numWorkers = target.getNumWorkerContexts();

  // TODO: add padding cost once we support padding.
  const auto calculateRearrangeCycles =
      [=](const std::vector<unsigned> &values) {
        const auto numXOrYGroups = values[0];
        const auto blockSizeXY = values[1];
        const auto numZ = values[2];
        const auto cycles = getBlockTransposeGradWCycles(
            dataType == FLOAT, blockSizeXY, numXOrYGroups, numZ, numWorkers);
        return popsolver::DataType{cycles};
      };

  const auto mCycles = m.call<unsigned>(
      {mXOrYGroups, mXOrYGrouping, m.product({mZGroups, mZGrouping})},
      calculateRearrangeCycles);
  const auto mBytesPerInput = m.addConstant(target.getTypeSize(dataType));

  const auto mTransposedBytes = m.product(
      {mBytesPerInput, mXOrYGroups, mXOrYGrouping, mZGroups, mZGrouping});
  return std::make_pair(mCycles, mTransposedBytes);
}

/** Account for the cost of compute in the distribution phase.
 *  The cost of compute for this phase is any compute required
 *  assuming a perfectly uniform sparsity pattern. This boils
 *  down to the cost of compute in one full partition of X/Y.
 */
static popsolver::Variable addDistributionComputeCycleCostDenseDense(
    popsolver::Model &m, const Target &target, const Type &inputType,
    const double &nzRatio, const Options &options, const OnTileMethod &method,
    const Vector<popsolver::Variable> &mGroups,
    const Vector<popsolver::Variable> &mGrouping,
    const Vector<popsolver::Variable> &mCumulativePartitions,
    const popsolver::Variable &mSparseGroups,
    const popsolver::Variable &mElemsPerSparseGroup) {
  // TODO: Handle groups for vertex cycle estimates properly
  const auto mPartialsPerTile =
      m.product({mSparseGroups, mElemsPerSparseGroup});
  const auto &partialsType = options.partialsType;
  const unsigned numConvUnits =
      getNumConvUnits(target, inputType == FLOAT, partialsType == FLOAT);
  const auto numWorkers = target.getNumWorkerContexts();
  auto mCycles = m.call<unsigned>(
      {mPartialsPerTile, mGroups.x, mGroups.y, mGroups.z, mGrouping.x,
       mGrouping.y, mGrouping.z, mCumulativePartitions.z},
      [=](const std::vector<unsigned> &values) -> popsolver::DataType {
        const auto partialsPerTile = values[0];
        const auto xGroups = values[1];
        const auto yGroups = values[2];
        const auto zGroups = values[3];
        const auto xGrouping = values[4];
        const auto yGrouping = values[5];
        const auto zGrouping = values[6];
        const auto numZPartitions = values[7];
        const auto partialsPerWorker =
            gccs::ceildiv(partialsPerTile, numWorkers);

        std::uint64_t cycles = zeroPartialsCycles(partialsPerWorker, numWorkers,
                                                  partialsType == FLOAT,
                                                  xGrouping * yGrouping > 1);

        unsigned xNonZeroGroups, yNonZeroGroups;
        // Divide the number of xGroups by Z partition as we always split
        // rows first.
        const auto xGroupsPerZSplit = gccs::ceildiv(xGroups, numZPartitions);
        std::tie(xNonZeroGroups, yNonZeroGroups) =
            getNumGroupsGivenUniformSparsityPattern(nzRatio, xGroupsPerZSplit,
                                                    yGroups);
        unsigned nonZeroGroups = xNonZeroGroups * yNonZeroGroups;
        const auto groupsPerWorker = gccs::ceildiv(nonZeroGroups, numWorkers);
        const auto numUsedWorkers =
            method == OnTileMethod::GradWAMPBlock
                ? 1
                : gccs::ceildiv(nonZeroGroups, groupsPerWorker);

        const auto numZ = zGroups * zGrouping;
        std::uint64_t maxMulCycles = 0;
        for (unsigned worker = 0; worker < numUsedWorkers; ++worker) {
          std::vector<unsigned> numYThisWorker;
          unsigned numXGroupsThisWorker;
          if (method == OnTileMethod::GradWAMPBlock) {
            numYThisWorker.push_back(yNonZeroGroups);
            numXGroupsThisWorker = xNonZeroGroups;
          } else {
            auto startGroup = worker * groupsPerWorker;
            auto endGroup =
                std::min(nonZeroGroups, (worker + 1) * groupsPerWorker);
            numXGroupsThisWorker = gccs::ceildiv(endGroup, yNonZeroGroups) -
                                   gccs::floordiv(startGroup, yNonZeroGroups);
            numYThisWorker.reserve(numXGroupsThisWorker);
            while (startGroup != endGroup) {
              const auto numYGroupsForXGroup =
                  std::min(endGroup, startGroup + yNonZeroGroups) - startGroup;
              numYThisWorker.emplace_back(numYGroupsForXGroup);
              startGroup += numYGroupsForXGroup;
            }
          }

          constexpr auto numBuckets = 1u;
          constexpr auto numSubGroupsPerBucket = 1u;

          const auto numXThisWorker = numXGroupsThisWorker * xGrouping;
          std::uint64_t mulCycles = 0;
          switch (method) {
          case OnTileMethod::GradW:
            mulCycles = sparseDenseGradWElementwiseMultiply(
                numBuckets, numBuckets, numSubGroupsPerBucket, numXThisWorker,
                numZ, numYThisWorker, inputType == FLOAT, partialsType == FLOAT,
                numWorkers);
            break;
          case OnTileMethod::GradWBlock:
            mulCycles = sparseDenseBlockMultiplyGradW(
                numBuckets, numBuckets, numSubGroupsPerBucket,
                numXGroupsThisWorker, numZ, xGrouping, yGrouping,
                numYThisWorker, inputType == FLOAT, partialsType == FLOAT,
                numWorkers);

            break;
          case OnTileMethod::GradWAMPBlock:
            // Each block is processed by all workers
            mulCycles = sparseDenseBlockMultiplyGradWAmp(
                numBuckets, numBuckets, numSubGroupsPerBucket,
                numXGroupsThisWorker, numZ, xGrouping, yGrouping,
                numYThisWorker, inputType == FLOAT, partialsType == FLOAT,
                numWorkers, numConvUnits);
            break;
          default:
            throw poputil::poplibs_error("Unhandled method when planning");
          }
          // Average over different values of Y. TODO: The Y provided aren't
          // statistically significant, they just assume a rectangle and
          // divide between workers so there is some accounting for overheads.
          mulCycles = gccs::ceildiv(mulCycles, numYThisWorker.size());
          maxMulCycles = std::max(maxMulCycles, mulCycles);
        }
        cycles += maxMulCycles * numZPartitions;
        return popsolver::DataType{cycles};
      });
  return mCycles;
}

static CostVariables addPropagationCost(
    popsolver::Model &m, const Target &target, const Type &inputType,
    const Options &options, const ExchangeEstimator &exchangeEstimator,
    const popsolver::Variable &mBytesPerBuffer, const PartitionVariables &p) {
  // Estimate temporary memory cost of a single iteration of the dynamically
  // executed exchange based on this plan.
  //
  // During the propagating exchange, we will need space for 2 buckets which
  // we will flip flop between to allow simulatenous forwarding and receiving
  // of buckets to/from other tiles.
  return CostVariables(m.zero(),
                       m.product({mBytesPerBuffer, m.addConstant(2u)}));
}

static std::tuple<CostVariables, CostVariables> addReductionCost(
    popsolver::Model &m, const Target &target, const Type &inputType,
    const Options &options, const ExchangeEstimator &exchangeEstimator,
    const popsolver::Variable &mPartialsPerTileToReduce,
    const popsolver::Variable &mReductionDepth,
    const std::vector<popsolver::Variable> &mReductionDepthCumulative,
    const std::vector<popsolver::Variable> &mTileLevelPartitions,
    popsolver::Variable mQTempBytesAfterCompute) {
  // This is not dependent upon the distribution of the sparsity
  // pattern as we are reducing the dense output. This occurs after all other
  // steps of exchange and compute are complete.
  //
  // The cost of reduction is determined by the factor by which we reduce.
  //
  // There is no on-tile reduction naturally as partials for the same result
  // are partitioned between tiles.
  const auto mBytesPerPartial =
      m.addConstant(target.getTypeSize(options.partialsType));
  std::vector<popsolver::Variable> mPartialsPerTile(2);
  popsolver::Variable mExchangeCycles, mExchangeTempBytes, mComputeCycles,
      mComputeTempBytes;
  const auto numWorkers = target.getNumWorkerContexts();
  const auto dataPathWidth = target.getDataPathWidth();
  for (int level = 1; level >= 0; --level) {
    if (static_cast<unsigned>(level) == 1) {
      mPartialsPerTile[level] = mPartialsPerTileToReduce;
    } else {
      // Now estimate compute portion of reduction exchange cost.
      const auto reducePartialsType = options.partialsType;
      const auto reduceOutputType =
          (level == 0) ? inputType : options.partialsType;
      bool floatPartials = reducePartialsType == FLOAT;
      bool floatOutput = reduceOutputType == FLOAT;
      const auto partialsVectorWidth =
          target.getVectorWidth(reducePartialsType);
      const auto outputVectorWidth = target.getVectorWidth(reduceOutputType);
      const auto mBytesPerOutput =
          m.addConstant(target.getTypeSize(reduceOutputType));

      mPartialsPerTile[level] =
          m.ceildiv(mPartialsPerTile[level + 1], mReductionDepth);

      const auto mNeedsReduction = m.min(
          {m.one(), m.sub(mReductionDepthCumulative[level + 1], m.one())});

      // The reduction's exchange cost will be given by each tile needing to
      // receive (reductionDepth - 1)/reductionDepth of the partials, and
      // send 1/reductionDepth of the partials. > 2 reductionFactor means we
      // cannot overlap send/receive of partials so cost is based on full
      // partials size. This is an all-to-all exchange.
      const auto mPartialsToExchangePerTile = mPartialsPerTile[level + 1];
      const auto mBytesToExchangePerTile = m.product(
          {mPartialsToExchangePerTile, mBytesPerPartial, mNeedsReduction});
      const auto mBytesToExchange =
          m.product({mBytesToExchangePerTile, mTileLevelPartitions[level + 1]});
      mExchangeCycles = exchangeEstimator(mBytesToExchange);
      mExchangeTempBytes =
          m.sum({mQTempBytesAfterCompute, mBytesToExchangePerTile});
      mComputeCycles = m.call<unsigned>(
          {mPartialsPerTile[level], mReductionDepth},
          [=](const std::vector<unsigned> &values) -> popsolver::DataType {
            const auto partialsPerTile = values[0];
            const auto reductionDepth = values[1];

            if (reductionDepth == 0) {
              return popsolver::DataType{0};
            }

            if (reductionDepth == 1) {
              if (floatOutput == floatPartials) {
                return popsolver::DataType{0};
              } else {
                return popsolver::DataType{
                    getCastCycleEstimate(partialsPerTile, partialsVectorWidth,
                                         outputVectorWidth, numWorkers)};
              }
            }

            return popsolver::DataType{getReduceCycleEstimate(
                partialsPerTile, reductionDepth, dataPathWidth, floatOutput,
                floatPartials, numWorkers)};
          });
      const auto mNeedsCast =
          m.addConstant(reducePartialsType != inputType ? 1u : 0u);
      const auto mNeedsCastOrReduction =
          m.min({m.one(), m.sum({mNeedsCast, mNeedsReduction})});

      mQTempBytesAfterCompute = m.product(
          {mNeedsCastOrReduction, mPartialsPerTile[level], mBytesPerOutput});
      mComputeTempBytes = m.sum({mExchangeTempBytes, mQTempBytesAfterCompute});
    }
  }
  CostVariables mExchangeCost(mExchangeCycles, mExchangeTempBytes);
  CostVariables mComputeCost(mComputeCycles, mComputeTempBytes);
  return std::make_tuple(mExchangeCost, mComputeCost);
}

static std::tuple<CostVariables, popsolver::Variable>
addTransposeBucketsCost(popsolver::Model &m, const Target &target,
                        const Type &inputType,
                        const popsolver::Variable &mGroupsPerBucket,
                        const Vector<popsolver::Variable> &mGrouping) {

  const auto mElemsPerGroup = m.product({mGrouping.x, mGrouping.y});
  const auto mGroupingSum = m.sum({mGrouping.x, mGrouping.y});
  const auto mNeedsTranspose = m.sub(
      m.one(), m.sub(mGroupingSum, m.min({mElemsPerGroup, mGroupingSum})));
  const auto mBytesPerInput = m.addConstant(target.getTypeSize(inputType));

  const auto numWorkers = target.getNumWorkerContexts();

  const auto calculateTransposeCycles =
      [=](const std::vector<unsigned> &values) {
        const auto numTransposes = values[0];
        const auto numSrcRows = values[1];
        const auto numSrcColumns = values[2];
        if (numSrcRows + numSrcColumns > numSrcRows * numSrcColumns) {
          return popsolver::DataType{0};
        }

        const auto cycles = getTransposeCycleEstimate(
            numTransposes, numSrcRows, numSrcColumns, inputType, numWorkers);

        return popsolver::DataType{cycles};
      };
  const auto mCycles =
      m.product({mNeedsTranspose,
                 m.call<unsigned>({mGroupsPerBucket, mGrouping.x, mGrouping.y},
                                  calculateTransposeCycles)});

  const auto mTransposedBytes = m.product(
      {mNeedsTranspose, mGroupsPerBucket, mElemsPerGroup, mBytesPerInput});
  return std::make_tuple(CostVariables(mCycles, mTransposedBytes),
                         mTransposedBytes);
}

static std::tuple<CostVariables, CostBreakdownVariables>
addEstimates(const Target &target, const Type &inputType,
             const Vector<std::size_t> &shape,
             const SparsityParams &sparsityParams, const double &nzRatio,
             const OnTileMethod &method,
             const ExchangeEstimator &exchangeEstimator,
             const PartitionToPNMapping &mapping, popsolver::Model &m,
             const PartitionVariables &p,
             const std::vector<Vector<popsolver::Variable>> &mGroups,
             const Vector<popsolver::Variable> &mGrouping,
             const popsolver::Variable &mRGroupsPerBucket,
             const popsolver::Variable &mRElemsPerGroup,
             const popsolver::Variable &mRMetaInfoElemsPerBucket,
             const bool transposeBuckets, const Options &options) {
  CostBreakdownVariables costBreakdown;

  const auto mBytesPerInput = m.addConstant(target.getTypeSize(inputType));
  const auto mBytesPerMetaInfoElem =
      m.addConstant(target.getTypeSize(deviceMetaInfoType));
  const auto &mRNonZeroBytesPerBucket =
      m.product({mRGroupsPerBucket, mRElemsPerGroup, mBytesPerInput});
  const auto &mRMetaInfoBytesPerBucket =
      m.product({mRMetaInfoElemsPerBucket, mBytesPerMetaInfoElem});
  const auto mRBytesPerBucket =
      m.sum({mRNonZeroBytesPerBucket, mRMetaInfoBytesPerBucket});

  CostVariables mTransposeBucketsCost(m.zero(), m.zero());
  popsolver::Variable mRTransposedBytes = m.zero();
  if (transposeBuckets) {
    std::tie(mTransposeBucketsCost, mRTransposedBytes) =
        addTransposeBucketsCost(m, target, inputType, mRGroupsPerBucket,
                                mGrouping);
    costBreakdown.emplace_back("Transpose buckets", mTransposeBucketsCost);
  }

  CostVariables mDistributionExchangeCost;
  popsolver::Variable mSTempBytesAfterExchange, mRTempBytesAfterExchange;
  std::tie(mDistributionExchangeCost, mSTempBytesAfterExchange,
           mRTempBytesAfterExchange) =
      addDistributionExchangeCostSparseDense(
          m, target, inputType, deviceMetaInfoType, options, exchangeEstimator,
          mapping, mGroups, mGrouping, mRBytesPerBucket, p);
  mDistributionExchangeCost.tempBytes =
      m.sum({mDistributionExchangeCost.tempBytes, mRTransposedBytes});
  costBreakdown.emplace_back("Pre-distribution + distribution exchange",
                             mDistributionExchangeCost);

  CostVariables mDistributionComputeCost;
  popsolver::Variable mQTempBytesAfterCompute;
  std::tie(mDistributionComputeCost, mQTempBytesAfterCompute) =
      addDistributionComputeCostSparseDense(
          m, target, inputType, nzRatio, options, method, mGroups.back(),
          mGrouping, p.cumulative.back(), mSTempBytesAfterExchange,
          mRTempBytesAfterExchange);
  mDistributionComputeCost.tempBytes =
      m.sum({mDistributionComputeCost.tempBytes, mRTransposedBytes});
  costBreakdown.emplace_back("Distribution compute", mDistributionComputeCost);

  auto mPropagationCost = addPropagationCost(
      m, target, inputType, options, exchangeEstimator, mRBytesPerBucket, p);
  mPropagationCost.tempBytes =
      m.sum({mPropagationCost.tempBytes, mSTempBytesAfterExchange,
             mQTempBytesAfterCompute, mRTransposedBytes});
  costBreakdown.emplace_back("Propagation", mPropagationCost);

  const popsolver::Variable mPartialsPerTileToReduce =
      m.product({mGroups.back().groups, mGroups.back().x, mGroups.back().z,
                 mGrouping.groups, mGrouping.x, mGrouping.z});
  popsolver::Variable mReductionDepth;
  std::vector<popsolver::Variable> mReductionDepthCumulative(2);

  mReductionDepth = p.partition[0].y;
  mReductionDepthCumulative[0] = p.cumulative[0].y;
  mReductionDepthCumulative[1] = p.cumulative[1].y;

  const auto &[mReductionExchangeCost, mReductionComputeCost] =
      addReductionCost(m, target, inputType, options, exchangeEstimator,
                       mPartialsPerTileToReduce, mReductionDepth,
                       mReductionDepthCumulative, p.tile,
                       mQTempBytesAfterCompute);
  costBreakdown.emplace_back("Exchange to reduce", mReductionExchangeCost);
  costBreakdown.emplace_back("Reduction or cast", mReductionComputeCost);

  CostVariables cost(
      m.sum({mTransposeBucketsCost.cycles, mDistributionExchangeCost.cycles,
             mDistributionComputeCost.cycles, mPropagationCost.cycles,
             mReductionExchangeCost.cycles, mReductionComputeCost.cycles}),
      m.max(
          {mTransposeBucketsCost.tempBytes, mDistributionExchangeCost.tempBytes,
           mDistributionComputeCost.tempBytes, mPropagationCost.tempBytes,
           mReductionExchangeCost.tempBytes, mReductionComputeCost.tempBytes}));
  costBreakdown.emplace_back("Total", cost);
  return std::make_tuple(cost, costBreakdown);
}

static std::tuple<CostVariables, CostBreakdownVariables> addEstimatesGradW(
    const Target &target, const Type &inputType,
    const Vector<std::size_t> &shape, const SparsityParams &sparsityParams,
    const double nzRatio, const OnTileMethod &method,
    const ExchangeEstimator &exchangeEstimator,
    const PartitionToPNMapping &mapping, const bool exchangeBuckets,
    popsolver::Model &m, const PartitionVariables &p,
    const std::vector<Vector<popsolver::Variable>> &mGroups,
    const Vector<popsolver::Variable> &mGrouping,
    const popsolver::Variable &mRGroupsPerBucket,
    const popsolver::Variable &mRElemsPerGroup,
    const popsolver::Variable &mRMetaInfoElemsPerBucket,
    const Options &options) {
  CostBreakdownVariables costBreakdown;

  // We pre-calculate certain variables that will be used for different exchange
  // costs below.
  const auto mBytesPerInput = m.addConstant(target.getTypeSize(inputType));
  const auto mBytesPerPartial =
      m.addConstant(target.getTypeSize(options.partialsType));
  const auto mBytesPerMetaInfoElem =
      m.addConstant(target.getTypeSize(UNSIGNED_SHORT));
  std::vector<popsolver::Variable> mQGradBytesPerTile(2), mSBytesPerTile(2);
  for (unsigned level = 0; level < 2; ++level) {
    mQGradBytesPerTile[level] =
        m.product({m.ceildiv(m.product({mGroups[level].groups, mGroups[level].x,
                                        mGroups[level].z}),
                             p.tile[level]),
                   mGrouping.groups, mGrouping.x, mGrouping.z, mBytesPerInput});
    mSBytesPerTile[level] =
        m.product({m.ceildiv(m.product({mGroups[level].groups, mGroups[level].y,
                                        mGroups[level].z}),
                             p.tile[level]),
                   mGrouping.groups, mGrouping.y, mGrouping.z, mBytesPerInput});
  }
  const auto mRGradPartialBytesPerTile =
      m.product({mRMetaInfoElemsPerBucket, mBytesPerMetaInfoElem});
  const auto mRMetaInfoBytesPerTile =
      m.product({mRGroupsPerBucket, mRElemsPerGroup, mBytesPerPartial});
  const auto mRGradBytesPerTile =
      m.sum({mRGradPartialBytesPerTile, mRMetaInfoBytesPerTile});

  popsolver::Variable mQGradTempBytes = m.zero(), mSTempBytes = m.zero();
  const auto mPreDistributionExchangeCost =
      addPreDistributionExchangeCostDenseDense(
          m, options, exchangeEstimator, mapping, mQGradBytesPerTile,
          mSBytesPerTile, mQGradTempBytes, mSTempBytes, p);
  costBreakdown.emplace_back("Pre-distribution exchange",
                             mPreDistributionExchangeCost);

  CostVariables mTransposeCost(m.zero(), m.zero());
  if (method == OnTileMethod::GradWAMPBlock) {
    // Estimate cycle cost for rearranging both activations and gradients
    // wrt output. Transpose and exchange and there need not be done for
    // each partition of z.
    const auto &[mQGradTransposeCycles, mQGradTransposeBytes] =
        rearrangeDenseCost(m, target, inputType, mGroups.back().x, mGrouping.x,
                           mGroups.back().z, mGrouping.z);

    const auto &[mSTransposeCycles, mSTransposeBytes] =
        rearrangeDenseCost(m, target, inputType, mGroups.back().y, mGrouping.y,
                           mGroups.back().z, mGrouping.z);
    mTransposeCost.cycles = m.sum({mQGradTransposeCycles, mSTransposeCycles});
    // We transpose all in one compute set so temp mem is sum
    mTransposeCost.tempBytes = m.sum(
        {mQGradTransposeBytes, mQGradTempBytes, mSTransposeBytes, mSTempBytes});
    mQGradTempBytes = mQGradTransposeBytes;
    mSTempBytes = mSTransposeBytes;
  }
  costBreakdown.emplace_back("Q-grad/S transpose", mTransposeCost);

  const auto mDistributionExchangeCycles =
      addDistributionExchangeCycleCostDenseDense(
          m, options, exchangeEstimator, mapping, exchangeBuckets,
          mRGradBytesPerTile, mQGradBytesPerTile.back(), mSBytesPerTile.back(),
          p);
  costBreakdown.emplace_back(
      "Distribution exchange",
      CostVariables(mDistributionExchangeCycles, m.zero()));

  const auto mDistributionComputeCycles =
      addDistributionComputeCycleCostDenseDense(
          m, target, inputType, nzRatio, options, method, mGroups.back(),
          mGrouping, p.cumulative.back(), mRGroupsPerBucket, mRElemsPerGroup);
  costBreakdown.emplace_back(
      "Distribution compute",
      CostVariables(mDistributionComputeCycles, m.zero()));

  const auto &[mDistributionAndPropagationMaxTempBytes, mRGradTempBytes] =
      addGradWExchangeAndComputeTempBytesCost(
          m, options, inputType, exchangeBuckets, mRGradPartialBytesPerTile,
          mRMetaInfoBytesPerTile, mQGradTempBytes, mSTempBytes, p);
  costBreakdown.emplace_back(
      "All exchange + compute",
      CostVariables(m.zero(), mDistributionAndPropagationMaxTempBytes));

  const auto mPartialsPerTileToReduce =
      m.product({mRGroupsPerBucket, mRElemsPerGroup});
  const popsolver::Variable mReductionDepth = m.one();
  const std::vector<popsolver::Variable> mReductionDepthCumulative(2, m.one());
  const auto &[mReductionExchangeCost, mReductionComputeCost] =
      addReductionCost(m, target, inputType, options, exchangeEstimator,
                       mPartialsPerTileToReduce, mReductionDepth,
                       mReductionDepthCumulative, p.tile, mRGradTempBytes);
  costBreakdown.emplace_back("Exchange to reduce", mReductionExchangeCost);
  costBreakdown.emplace_back("Reduction or cast", mReductionComputeCost);

  CostVariables cost(
      m.sum({mPreDistributionExchangeCost.cycles, mTransposeCost.cycles,
             mDistributionExchangeCycles, mDistributionComputeCycles,
             mReductionExchangeCost.cycles, mReductionComputeCost.cycles}),
      m.max({mPreDistributionExchangeCost.tempBytes, mTransposeCost.tempBytes,
             mDistributionAndPropagationMaxTempBytes,
             mReductionExchangeCost.tempBytes,
             mReductionComputeCost.tempBytes}));
  costBreakdown.emplace_back("Total", cost);
  return std::make_tuple(cost, costBreakdown);
}

// TODO: We could actually get this straight from the parameters. Until
// we've decided how blocks should be represented in FullyConnectedParams
// we'll calculate exactly what we want here.
static popsolver::Variable
addNumNonZeroGroups(popsolver::Model &m, const FullyConnectedParams &params,
                    const Type &inputType) {
  const auto &sparsityParams = params.getSparsityParams();
  const auto blockDimensions =
      getBlockDimensionsToUse(sparsityParams.blockDimensions, inputType);
  const auto outputBlocks =
      params.getOutputChannelsPerGroup() / blockDimensions.at(0);
  const auto inputBlocks =
      params.getInputChannelsPerGroup() / blockDimensions.at(1);
  const auto rGroups =
      params.getNumGroups() *
      unsigned(std::ceil(inputBlocks * outputBlocks * params.getNzRatio()));
  return m.addConstant(rGroups);
}

static popsolver::Variable addNumNonZeroGroupsPerBucket(
    popsolver::Model &m, const Target &target, const Type &inputType,
    const popsolver::Variable &mNonZeroGroups, unsigned nonZeroElemsPerGroup,
    const PartitionVariables &p, const Options &options) {
  // Find the number of groups per bucket when uniformly distributed.
  const auto mPerfectlyUniformGroupsPerBucket =
      m.ceildiv(mNonZeroGroups, p.tile.at(0));
  // Ensure the number of elements guarantees the size in bytes of the bucket
  // is a multiple of the exchange atom size.
  const unsigned bytesPerNonZeroElem = target.getTypeSize(inputType);
  const auto bytesPerGroup = nonZeroElemsPerGroup * bytesPerNonZeroElem;
  const unsigned exchangeAtomSize = target.getExchangeBytesPerCycle();
  const auto grainSizeInGroups =
      lcm(bytesPerGroup, exchangeAtomSize) / bytesPerGroup;
  return m.call<unsigned>(
      {mPerfectlyUniformGroupsPerBucket},
      [=](const std::vector<unsigned> &values) -> popsolver::DataType {
        // Number of groups when perfectly distributed is multiplied by some
        // factor given as an option to allow room for imbalance.
        const unsigned groups = std::round(
            values[0] * (1.0 + options.metaInfoBucketOversizeProportion));
        return popsolver::DataType{gccs::alignNext(groups, grainSizeInGroups)};
      });
}

// Given the meta-info is often shared between passes in some way, these
// are calculated and returned jointly.
static popsolver::Variable addMetaInfoElemsPerBucket(
    popsolver::Model &m, const Target &target, const Type &deviceMetaInfoType,
    const double &nzRatio, const OnTileMethod &method,
    const Vector<popsolver::Variable> &mGroupsPerTile, const Options &options) {
  const unsigned bytesPerMetaInfoElem = target.getTypeSize(deviceMetaInfoType);
  const unsigned exchangeAtomSize = target.getExchangeBytesPerCycle();
  const auto atomSizeInMetaInfoElems =
      lcm(bytesPerMetaInfoElem, exchangeAtomSize);

  // A chosen number of sub-groups per bucket just for memory planning.
  constexpr unsigned numSubgroupsPerBucket = 2U;

  auto calcFwdBucketSizeElemwise =
      [=](const std::vector<unsigned> &values) -> popsolver::DataType {
    const auto xGroups = values[0];
    const auto yGroups = values[1];
    unsigned xNonZeroGroups, yNonZeroGroups;
    std::tie(xNonZeroGroups, yNonZeroGroups) =
        getNumGroupsGivenUniformSparsityPattern(nzRatio, xGroups, yGroups);

    // Knowing that we use a CSR based format we can calculate the
    // number of elements of meta-info that would be required to
    // store this in a perfect world.
    const auto outputEntryElems =
        sizeof(MetaInfo<MetaInfoType>::OutputEntry) / sizeof(MetaInfoType);
    const auto subGroupElems =
        sizeof(MetaInfo<MetaInfoType>::SubGroupEntry) / sizeof(MetaInfoType);
    const auto workerEntryElems =
        sizeof(MetaInfo<MetaInfoType>::WorkerEntry) / sizeof(MetaInfoType);
    const auto numElemsPerfectlyUniform =
        xNonZeroGroups * (outputEntryElems + yNonZeroGroups);
    const auto gradWWorkerEntryElems =
        options.doGradWPass
            ? (1 + sizeof(MetaInfo<MetaInfoType>::GradWWorkerEntry) /
                       sizeof(MetaInfoType))
            : 0;

    const unsigned elems =
        (subGroupElems + target.getNumWorkerContexts() *
                             (workerEntryElems + gradWWorkerEntryElems)) *
            numSubgroupsPerBucket +
        std::ceil(numElemsPerfectlyUniform *
                  (1.0 + options.metaInfoBucketOversizeProportion));
    return popsolver::DataType{gccs::alignNext(elems, atomSizeInMetaInfoElems)};
  };

  const auto calcFwdBucketSizeAMPBlock =
      [=](const std::vector<unsigned> &values) -> popsolver::DataType {
    const auto xGroups = values[0];
    const auto yGroups = values[1];
    unsigned xNonZeroGroups, yNonZeroGroups;
    std::tie(xNonZeroGroups, yNonZeroGroups) =
        getNumGroupsGivenUniformSparsityPattern(nzRatio, xGroups, yGroups);

    const auto outputEntryElems =
        sizeof(BlockMetaInfo<MetaInfoType>::OutputEntry) / sizeof(MetaInfoType);
    const auto subGroupElems =
        sizeof(BlockMetaInfo<MetaInfoType>::SubGroupEntry) /
        sizeof(MetaInfoType);
    const auto gradWWorkerEntryElems =
        options.doGradWPass
            ? sizeof(BlockMetaInfo<MetaInfoType>::GradWWorkerEntry) /
                  sizeof(MetaInfoType)
            : 0;

    const auto numElemsPerfectlyUniform =
        xNonZeroGroups * (outputEntryElems + yNonZeroGroups);
    const unsigned elems =
        ((subGroupElems +
          target.getNumWorkerContexts() * gradWWorkerEntryElems) *
             numSubgroupsPerBucket +
         std::ceil(numElemsPerfectlyUniform *
                   (1.0 + options.metaInfoBucketOversizeProportion)));
    return popsolver::DataType{gccs::alignNext(elems, atomSizeInMetaInfoElems)};
  };

  switch (method) {
  case OnTileMethod::Forward: {
    return m.call<unsigned>({mGroupsPerTile.x, mGroupsPerTile.y},
                            calcFwdBucketSizeElemwise);
  }
  case OnTileMethod::GradA: {
    auto calcGradABucketSizeElemwise =
        [=](const std::vector<unsigned> &values) -> popsolver::DataType {
      const auto xGroups = values[0];
      const auto yGroups = values[1];
      unsigned xNonZeroGroups, yNonZeroGroups;
      std::tie(xNonZeroGroups, yNonZeroGroups) =
          getNumGroupsGivenUniformSparsityPattern(nzRatio, xGroups, yGroups);

      // Knowing that we use a CSR based format we can calculate the
      // number of elements of meta-info that would be required to
      // store this in a perfect world.
      const auto outputEntryElems =
          sizeof(MetaInfo<MetaInfoType>::OutputEntry) / sizeof(MetaInfoType);
      const auto subGroupElems =
          sizeof(MetaInfo<MetaInfoType>::SubGroupEntry) / sizeof(MetaInfoType);
      const auto workerEntryElems =
          sizeof(MetaInfo<MetaInfoType>::WorkerEntry) / sizeof(MetaInfoType);
      // yNonZeroGroups * 2 because we encode information to transpose
      // weights along with offsets for inputs if GradA method is selected
      // other wise the same bucket as forward is used
      constexpr unsigned elementsPerY = 2;
      const auto numElemsPerfectlyUniform =
          xNonZeroGroups * (outputEntryElems + yNonZeroGroups * elementsPerY);
      const unsigned elems =
          (subGroupElems + target.getNumWorkerContexts() * workerEntryElems) *
              numSubgroupsPerBucket +
          std::ceil(numElemsPerfectlyUniform *
                    (1.0 + options.metaInfoBucketOversizeProportion));
      return popsolver::DataType{
          gccs::alignNext(elems, atomSizeInMetaInfoElems)};
    };
    return m.call<unsigned>({mGroupsPerTile.x, mGroupsPerTile.y},
                            calcGradABucketSizeElemwise);
  }
  case OnTileMethod::Transpose: {
    // We actually use the same buckets as forward and for a joint plan
    // the split should just be the tranpose of the forward.
    return m.call<unsigned>({mGroupsPerTile.y, mGroupsPerTile.x},
                            calcFwdBucketSizeElemwise);
  }
  case OnTileMethod::ForwardAMPBlock: {
    return m.call<unsigned>({mGroupsPerTile.x, mGroupsPerTile.y},
                            calcFwdBucketSizeAMPBlock);
  }
  case OnTileMethod::TransposeAMPBlock: {
    return m.call<unsigned>({mGroupsPerTile.y, mGroupsPerTile.x},
                            calcFwdBucketSizeAMPBlock);
  }
  default:
    throw poputil::poplibs_error("Unhandled OnTileMethod");
  }
}

static void
applyPartitionPlanConstraint(popsolver::Model &m, const Options &options,
                             unsigned level,
                             const Vector<popsolver::Variable> &partition) {
  assert(level == 0);
  const auto &planConstraints = options.planConstraints;
  const auto &thisPartition = planConstraints.get_child_optional("partition");
  if (thisPartition) {
    const auto constrainVar = [&](const std::string &pathSuffix,
                                  const popsolver::Variable &var) {
      const auto constraint =
          thisPartition.get().get_optional<popsolver::DataType>(pathSuffix);
      if (constraint) {
        m.equal(var, *constraint);
      }
    };
    constrainVar("x", partition.x);
    constrainVar("y", partition.y);
    constrainVar("z", partition.z);
  }
}

// Add limits in the model to account for the range limit of meta-info
template <typename MetaInfoT>
static void addMetaInfoRangeLimits(popsolver::Model &m,
                                   const Vector<popsolver::Variable> &mGroups,
                                   const Vector<popsolver::Variable> &mGrouping,
                                   MetaInfoT, const Type &inputType,
                                   bool isBlockMetaInfoFormat,
                                   const Options &options) {
  // TODO: This should really live alongside the meta-info, but currently use
  // of the model prohibits this. We could add a wrapper for popsolver
  // variables that provides operator overloads such that we can just template
  // the calculation of the max value given the grouping etc.

  // TODO: This is not complete and only accounts for the offsets encoded in
  // meta-info for element-wise sparsity as these are the most likely to exceed
  // the range of the encoding type for the meta-info.
  //
  if (!isBlockMetaInfoFormat) {
    const auto mYOffsetFactor =
        m.addConstant(getYOffsetTypeScaleFactor(inputType == FLOAT));
    // Max offset Y in S on this tile is (Y - 1)
    const auto mMaxYOffset =
        m.sub(m.product({mGroups.y, mGrouping.y}), m.one());
    const auto mMaxYOffsetEncoded =
        m.product({mMaxYOffset, mGroups.z, mGrouping.z, mYOffsetFactor});
    const auto mMaxXOffsetEncoded =
        m.sub(m.product({mGroups.x, mGrouping.x}), m.one());
    const auto mMaxOffset = m.max({mMaxYOffsetEncoded, mMaxXOffsetEncoded});
    const auto mMaxEncodableValue =
        m.addConstant(std::numeric_limits<MetaInfoT>::max());
    m.lessOrEqual(mMaxOffset, mMaxEncodableValue);
  }
}

// For now just make this such that it never gets picked. In future
// will change this so that the planner will pick this variable
// based on the plan returned for the dense operation
static popsolver::Variable
addUseDenseVariable(popsolver::Model &m,
                    const poplibs_support::PlanConstraints &constraints) {
  const auto denseConstraint = constraints.get_optional<bool>("useDense");
  unsigned useDenseValue = 0;
  if (denseConstraint) {
    useDenseValue = static_cast<unsigned>(*denseConstraint);
  }
  return m.addConstant(useDenseValue);
}

static std::tuple<Plan, Cost, CostBreakdown>
createPlan(const PlanningObjective &objective, const Target &target,
           const Type &inputType, const FullyConnectedParams &params,
           const Method &method, const ExchangeAndMappingPlan &exchangePlan,
           const Cost &bestCost, const Options &options) {
  const auto tilesPerIPU = target.getTilesPerIPU();

  Vector<unsigned> size = {
      static_cast<unsigned>(params.getNumGroups()),              // groups
      static_cast<unsigned>(params.getOutputChannelsPerGroup()), // x
      static_cast<unsigned>(params.getInputChannelsPerGroup()),  // y
      static_cast<unsigned>(params.getBatchSize()),              // z
  };
  Vector<unsigned> groups =
      size.binaryOp(method.grouping, [&](const auto size, const auto grouping) {
        return gccs::ceildiv(size, grouping);
      });

  popsolver::Model m;
  unsigned level = 0;
  // Create partitions variables
  const PartitionVariables fwdPartition = [&] {
    std::vector<Vector<popsolver::Variable>> mPartitions(1);
    mPartitions[level] = Vector<popsolver::Variable>::generate(
        [&] { return m.addVariable(1, tilesPerIPU); });
    applyPartitionPlanConstraint(m, options, level, mPartitions[level]);

    auto partitionPrioGrp = m.addPriorityGroup();
    for (const auto &levelPartition : mPartitions) {
      for (const auto &var : levelPartition.asStdVector()) {
        m.setPriorityGroup(var, partitionPrioGrp);
      }
    }
    m.prioritiseOver(partitionPrioGrp, m.getDefaultPriorityGroup());
    return PartitionVariables(m, mPartitions);
  }();

  // Calculate grains, add constraints on partitions
  std::vector<Vector<popsolver::Variable>> mFwdGroups(level + 2);
  mFwdGroups[0] = groups.transform<popsolver::Variable>(
      [&](const auto groups) { return m.addConstant(groups); });
  m.lessOrEqual(fwdPartition.product[0], popsolver::DataType{tilesPerIPU});
  mFwdGroups[level + 1] = mFwdGroups[level].binaryOp(
      fwdPartition.partition[level],
      [&](const auto &groups, const auto &partition) {
        return m.ceildivConstrainDivisor(groups, partition);
      });

  // Partitions of Z must be of equal size on every tile.
  m.factorOf(mFwdGroups[level].z, fwdPartition.partition[level].z);

  // Our vertex doesn't handle groups at all.
  m.equal(mFwdGroups[level + 1].groups, popsolver::DataType{1});
  const auto mFwdGrouping = method.grouping.transform<popsolver::Variable>(
      [&](const auto grouping) { return m.addConstant(grouping); });

  const bool isBlockMetaInfoFormat =
      params.getSparsityParams().type == SparsityType::Block;
  addMetaInfoRangeLimits(m, mFwdGroups.back(), mFwdGrouping, MetaInfoType(),
                         inputType, isBlockMetaInfoFormat, options);

  // Calculate size of buckets.
  const auto mRGroups = addNumNonZeroGroups(m, params, inputType);
  const auto rElemsPerGroup =
      method.grouping.groups * method.grouping.x * method.grouping.y;
  const auto mRGroupsPerBucket = addNumNonZeroGroupsPerBucket(
      m, target, inputType, mRGroups, rElemsPerGroup, fwdPartition, options);
  const auto mRElemsPerGroup = m.addConstant(rElemsPerGroup);
  const auto mRFwdMetaInfoElemsPerBucket = addMetaInfoElemsPerBucket(
      m, target, deviceMetaInfoType, params.getNzRatio(), method.fwd,
      mFwdGroups.back(), options);

  CostVariables fwdCost;
  CostBreakdownVariables fwdCostBreakdown;
  const Vector<std::size_t> fwdShape = {
      params.getNumGroups(),
      params.getOutputChannelsPerGroup(),
      params.getInputChannelsPerGroup(),
      params.getBatchSize(),
  };
  const ExchangeEstimator exchangeEstimator(m, target);

  std::tie(fwdCost, fwdCostBreakdown) =
      addEstimates(target, inputType, fwdShape, params.getSparsityParams(),
                   params.getNzRatio(), method.fwd, exchangeEstimator,
                   exchangePlan.fwdMapping, m, fwdPartition, mFwdGroups,
                   mFwdGrouping, mRGroupsPerBucket, mRElemsPerGroup,
                   mRFwdMetaInfoElemsPerBucket, false, options);

  CostVariables gradACost(m.zero(), m.zero());
  CostBreakdownVariables gradACostBreakdown;
  popsolver::Variable mRGradAMetaInfoElemsPerBucket = m.zero();
  if (options.doGradAPass) {
    // TODO: Encapsulate this translation to GradA pass better.
    // This is just a swizzle applied to all vectors in 'planning
    // space'.
    const auto toGradA = [](const auto fwdV) {
      return decltype(fwdV){
          fwdV.groups,
          fwdV.y,
          fwdV.x,
          fwdV.z,
      };
    };

    const auto gradAShape = toGradA(fwdShape);
    const auto gradAPartition = [&] {
      auto gradA = fwdPartition;
      for (auto &p : gradA.partition) {
        p = toGradA(p);
      }
      for (auto &p : gradA.cumulative) {
        p = toGradA(p);
      }
      return gradA;
    }();
    const auto mGradAGroups = [&] {
      auto v = mFwdGroups;
      for (auto &g : v) {
        g = toGradA(g);
      }
      return v;
    }();
    const auto mGradAGrouping = toGradA(mFwdGrouping);

    if (!options.sharedBuckets) {
      addMetaInfoRangeLimits(m, mGradAGroups.back(), mGradAGrouping,
                             MetaInfoType(), inputType, isBlockMetaInfoFormat,
                             options);
    }

    mRGradAMetaInfoElemsPerBucket = addMetaInfoElemsPerBucket(
        m, target, deviceMetaInfoType, params.getNzRatio(), method.gradA,
        mGradAGroups.back(), options);

    std::tie(gradACost, gradACostBreakdown) =
        addEstimates(target, inputType, gradAShape, params.getSparsityParams(),
                     params.getNzRatio(), method.gradA, exchangeEstimator,
                     exchangePlan.gradAMapping, m, gradAPartition, mGradAGroups,
                     mGradAGrouping, mRGroupsPerBucket, mRElemsPerGroup,
                     mRGradAMetaInfoElemsPerBucket, true, options);
  }

  CostVariables gradWCost(m.zero(), m.zero());
  CostBreakdownVariables gradWCostBreakdown;
  if (options.doGradWPass) {
    std::tie(gradWCost, gradWCostBreakdown) = addEstimatesGradW(
        target, inputType, fwdShape, params.getSparsityParams(),
        params.getNzRatio(), method.gradW, exchangeEstimator,
        exchangePlan.gradWMapping, exchangePlan.gradWExchangeBuckets, m,
        fwdPartition, mFwdGroups, mFwdGrouping, mRGroupsPerBucket,
        mRElemsPerGroup, mRFwdMetaInfoElemsPerBucket, options);
  }

  const auto useDense = addUseDenseVariable(m, options.planConstraints);

  const CostVariables mCost(
      m.sum({fwdCost.cycles, gradACost.cycles, gradWCost.cycles}),
      m.max({fwdCost.tempBytes, gradACost.tempBytes, gradWCost.tempBytes}));
  CostBreakdownVariables mCostBreakdown;
  for (auto &entry : fwdCostBreakdown) {
    mCostBreakdown.emplace_back("Fwd: " + std::move(entry.first),
                                std::move(entry.second));
  }
  for (auto &entry : gradACostBreakdown) {
    mCostBreakdown.emplace_back("GradA: " + std::move(entry.first),
                                std::move(entry.second));
  }
  for (auto &entry : gradWCostBreakdown) {
    mCostBreakdown.emplace_back("GradW: " + std::move(entry.first),
                                std::move(entry.second));
  }

  popsolver::Solution solution;
  switch (objective.getType()) {
  case PlanningObjective::MINIMIZE_CYCLES:
    m.lessOrEqual(mCost.cycles, bestCost.cycles);
    m.lessOrEqual(mCost.tempBytes, objective.getTileTempMemoryBound());
    solution = m.minimize({mCost.cycles, mCost.tempBytes});
    break;
  case PlanningObjective::MINIMIZE_TILE_TEMP_MEMORY:
    m.lessOrEqual(mCost.tempBytes, bestCost.tempBytes);
    m.lessOrEqual(mCost.cycles, objective.getCyclesBound());
    solution = m.minimize({mCost.tempBytes, mCost.cycles});
    break;
  }

  if (!solution.validSolution()) {
    return {Plan(), highestCost, {}};
  }

  Plan plan;
  plan.partition = fwdPartition.partition[0].transform<unsigned>(
      [&](popsolver::Variable partitionVar) {
        return solution[partitionVar].getAs<unsigned>();
      });
  // For now these are hard-coded but we could plan for it further
  // down the road if memory or something else did not allow this.
  plan.initialDistributionPartitions = Vector<unsigned>(1);
  plan.initialDistributionPartitions.z = plan.partition.z;
  // We round up the number of nz values per bucket to a multiple of
  // 64-bits when used as partials in the GradW pass.
  assert(8 % target.getTypeSize(options.partialsType) == 0);
  const auto nzElemGrainSize =
      options.doGradWPass ? 8 / target.getTypeSize(options.partialsType) : 1;
  plan.nzElemsPerBucket = gccs::alignNext(
      solution[mRGroupsPerBucket].getAs<unsigned>() * rElemsPerGroup,
      nzElemGrainSize);
  plan.fwdMetaInfoElemsPerBucket =
      solution[mRFwdMetaInfoElemsPerBucket].getAs<unsigned>();
  plan.gradAMetaInfoElemsPerBucket =
      solution[mRGradAMetaInfoElemsPerBucket].getAs<unsigned>();
  plan.method = method;
  // We currently plan the gradW pass with dimensions ordered for the
  // forward pass but the implementation wants a different ordering so
  // do the switch here until such a time as planning/implementation agree
  // on dimension ordering for gradW pass.
  plan.exchangePlan = exchangePlan;
  const auto oldGradWMapping =
      exchangePlan.gradWMapping.getLinearisationOrder();
  plan.exchangePlan.gradWMapping =
      PartitionToPNMapping({oldGradWMapping.groups, oldGradWMapping.x,
                            oldGradWMapping.z, oldGradWMapping.y});

  plan.useDense = solution[useDense].getAs<unsigned>();

  Cost cost;
  cost.cycles = solution[mCost.cycles];
  cost.tempBytes = solution[mCost.tempBytes];

  CostBreakdown costBreakdown;
  costBreakdown.reserve(mCostBreakdown.size());
  for (const auto &entry : mCostBreakdown) {
    costBreakdown.emplace_back(
        std::move(entry.first),
        Cost(solution[entry.second.cycles], solution[entry.second.tempBytes]));
  }

  return std::make_tuple(std::move(plan), std::move(cost),
                         std::move(costBreakdown));
}

static std::vector<Method>
getCandidateMethods(const Target &target, const Type &inputType,
                    const FullyConnectedParams &params,
                    const Options &options) {
  std::vector<Method> methods;

  const auto &sparsityParams = params.getSparsityParams();
  const auto blockDimensions =
      getBlockDimensionsToUse(sparsityParams.blockDimensions, inputType);

  const unsigned xElemsPerBlock = blockDimensions.at(0);
  const unsigned yElemsPerBlock = blockDimensions.at(1);
  const unsigned elemsPerBlock = xElemsPerBlock * yElemsPerBlock;

  if (elemsPerBlock == 1) {
    // Element-wise methods
    Vector<unsigned> grouping = {
        1,                                // groups
        1,                                // x
        1,                                // y
        target.getVectorWidth(inputType), // z
    };
    methods.emplace_back(Method{
        grouping, OnTileMethod::Forward,
        // TODO: This could eventually be based on a memory/cycle tradeoff
        options.sharedBuckets ? OnTileMethod::Transpose : OnTileMethod::GradA,
        OnTileMethod::GradW});
  } else {
    // AMP-based block methods
    Vector<unsigned> grouping = {
        1,
        xElemsPerBlock,
        yElemsPerBlock,
        1,
    };
    methods.emplace_back(Method{grouping, OnTileMethod::ForwardAMPBlock,
                                OnTileMethod::TransposeAMPBlock,
                                OnTileMethod::GradWBlock});
    // Batch size restriction on AMP based GradW method

    const auto zGrouping = inputType == FLOAT ? 8U : 16U;
    const auto addGradWAmpMethod = (params.getBatchSize() % zGrouping == 0) &&
                                   (xElemsPerBlock % 4 == 0) &&
                                   (yElemsPerBlock % 4 == 0);
    if (addGradWAmpMethod) {
      // AMP-based block methods
      Vector<unsigned> groupingAmp = {
          1,
          xElemsPerBlock,
          yElemsPerBlock,
          zGrouping,
      };
      methods.emplace_back(Method{groupingAmp, OnTileMethod::ForwardAMPBlock,
                                  OnTileMethod::TransposeAMPBlock,
                                  OnTileMethod::GradWAMPBlock});
    }
  }
  return methods;
}

static std::vector<ExchangeAndMappingPlan>
getCandidateExchangePlans(const Options &options) {
  std::vector<ExchangeAndMappingPlan> candidates;

  const auto &planConstraints = options.planConstraints;
  const auto &exchangePlanConstraints =
      planConstraints.get_child_optional("exchange");

  std::vector<bool> validGradWExchangeBucketsOptions = {false, true};
  if (exchangePlanConstraints) {
    const auto &constraint = exchangePlanConstraints.get().get_optional<bool>(
        "gradWExchangeBuckets");
    if (constraint) {
      validGradWExchangeBucketsOptions = {*constraint};
    }
  }

  // Only leave one of the 2 possibilities if we aren't doing a
  // GradW pass anyway as gradWExchangeBuckets will have no effect
  // on a plan without a GradW pass.
  if (!options.doGradWPass && validGradWExchangeBucketsOptions.size() > 1) {
    validGradWExchangeBucketsOptions = {validGradWExchangeBucketsOptions[0]};
  }

  ExchangeAndMappingPlan basePlan;
  basePlan.fwdMapping = PartitionToPNMapping({0, 3, 1, 2});
  basePlan.gradAMapping = PartitionToPNMapping({0, 1, 3, 2});
  for (const auto &gradWExchangeBuckets : validGradWExchangeBucketsOptions) {
    basePlan.gradWExchangeBuckets = gradWExchangeBuckets;
    if (gradWExchangeBuckets) {
      // Note this is a conscious choice of re-use of the forward
      // pass mapping for the candidate where buckets are
      // exchanged.
      basePlan.gradWMapping = PartitionToPNMapping({0, 3, 1, 2});
    } else {
      basePlan.gradWMapping = PartitionToPNMapping({0, 1, 2, 3});
    }
    candidates.emplace_back(basePlan);
  }

  return candidates;
}

static std::tuple<Plan, Cost, CostBreakdown>
createPlan(const PlanningObjective &objective, const Target &target,
           const Type &inputType, const FullyConnectedParams &params,
           const Options &options) {
  const auto candidateMethods =
      getCandidateMethods(target, inputType, params, options);
  const auto candidateExchangePlans = getCandidateExchangePlans(options);
  assert(!candidateMethods.empty());
  assert(!candidateExchangePlans.empty());

  Plan best;
  Cost bestCost = highestCost;
  CostBreakdown bestCostBreakdown;
  for (const auto &candidateMethod : candidateMethods) {
    for (const auto &candidateExchangePlan : candidateExchangePlans) {
      Plan candidate;
      Cost candidateCost;
      CostBreakdown candidateCostBreakdown;

      std::tie(candidate, candidateCost, candidateCostBreakdown) =
          createPlan(objective, target, inputType, params, candidateMethod,
                     candidateExchangePlan, bestCost, options);

      if (candidateCost == highestCost) {
        continue;
      }

      if (objective.lowerCost(candidateCost, bestCost)) {
        best = std::move(candidate);
        bestCost = std::move(candidateCost);
        bestCostBreakdown = std::move(candidateCostBreakdown);
      }
    }
  }
  return std::make_tuple(best, bestCost, bestCostBreakdown);
}

static std::tuple<Plan, Cost> runPlanner(const Target &target,
                                         const Type &inputType,
                                         const FullyConnectedParams &params,
                                         const Options &options) {
  Plan plan;
  auto cost = highestCost;
  CostBreakdown costBreakdown;

  const unsigned availableTileMem =
      target.getBytesPerTile() * options.availableMemoryProportion;

  if (availableTileMem != 0) {
    auto objective = PlanningObjective::minimizeCycles();
    auto stepMemBound = availableTileMem;
    objective.setTileTempMemoryBound(popsolver::DataType{stepMemBound});
    logging::popsparse::debug(
        "Planning sparse-dense matrix multiply with a per-tile "
        "memory limit of {} bytes.",
        stepMemBound);

    do {
      std::tie(plan, cost, costBreakdown) =
          createPlan(objective, target, inputType, params, options);
      if (cost != highestCost) {
        break;
      }

      stepMemBound *= 2;
      logging::popsparse::warn(
          "Unable to meet memory target. Retrying with a per-tile "
          "memory limit of {} bytes.",
          stepMemBound);
      objective.setTileTempMemoryBound(popsolver::DataType{stepMemBound});
    } while (stepMemBound < target.getBytesPerTile() * 2);

    // If the above did not succeed, try again without any memory limit to
    // get a valid plan of some sort.
    if (cost == highestCost) {
      objective = PlanningObjective::minimizeCycles();
      logging::popsparse::warn(
          "Unable to meet memory target. Retrying with no per-tile "
          "memory limit.");
      std::tie(plan, cost, costBreakdown) =
          createPlan(objective, target, inputType, params, options);
    }
  } else {
    logging::popsparse::debug(
        "Planning sparse-dense matrix multiply with unlimited memory usage.");
  }

  logging::popsparse::debug("Found best plan: {}.", cost);
  if (logging::popsparse::shouldLog(logging::Level::Debug)) {
    logging::popsparse::debug("  Cost breakdown:");
    for (const auto &entry : costBreakdown) {
      logging::popsparse::debug("    {}: {}", entry.first, entry.second);
    }
  }
  logging::popsparse::debug("  for params:\n{}", params);
  logging::popsparse::debug("  and input type: {}", inputType);
  logging::popsparse::debug("  with options:\n{}", options);
  logging::popsparse::debug("{}", plan);

  return std::make_tuple(std::move(plan), std::move(cost));
}

std::ostream &operator<<(std::ostream &os, const Cost &c) {
  os << "Cost{";
  bool needComma = false;
  if (*c.cycles != 0) {
    os << "cycles=" << c.cycles;
    needComma = true;
  }
  if (*c.tempBytes != 0) {
    if (needComma) {
      os << ", ";
    }
    os << "memory=" << c.tempBytes;
  }
  os << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const OnTileMethod &m) {
  switch (m) {
  case OnTileMethod::Forward:
    os << "Forward";
    break;
  case OnTileMethod::GradA:
    os << "GradA";
    break;
  case OnTileMethod::GradW:
    os << "GradW";
    break;
  case OnTileMethod::Transpose:
    os << "Transpose";
    break;
  case OnTileMethod::ForwardAMPBlock:
    os << "ForwardAMPBlock";
    break;
  case OnTileMethod::TransposeAMPBlock:
    os << "TransposeAMPBlock";
    break;
  case OnTileMethod::GradWAMPBlock:
    os << "GradWAMPBlock";
    break;
  case OnTileMethod::GradWBlock:
    os << "GradWBlock";
    break;
  default:
    throw poputil::poplibs_error("Unrecognised on-tile method");
  }
  return os;
}

std::ostream &operator<<(std::ostream &os, const Method &m) {
  os << "{ grouping: " << m.grouping << ", forward pass: " << m.fwd
     << ", grad-a pass: " << m.gradA << ", grad-w pass: " << m.gradW << " }";
  return os;
}

std::ostream &operator<<(std::ostream &os, const ExchangeAndMappingPlan &p) {
  os << "{ forward pass: " << p.fwdMapping
     << ", grad-a pass: " << p.gradAMapping
     << ", grad-w pass: " << p.gradWMapping << ", grad-w pass exchanges "
     << (p.gradWExchangeBuckets ? "buckets" : "inputs") << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const Plan &p) {
  os << "Plan:\n  method: " << p.method << "\n  partition: " << p.partition
     << "\n  initial distribution partitions: "
     << p.initialDistributionPartitions
     << "\n  used tiles: " << product(p.partition.asStdVector())
     << "\n  exchange plan: " << p.exchangePlan
     << "\n  no. of non-zero elements per bucket: " << p.nzElemsPerBucket
     << "\n  no. of meta-info elements per bucket (forward): "
     << p.fwdMetaInfoElemsPerBucket
     << "\n  no. of meta-info elements per bucket (grad-a): "
     << p.gradAMetaInfoElemsPerBucket << "\n use dense operation:" << p.useDense
     << "\n";
  return os;
}

std::array<std::vector<std::size_t>, 3>
getPartitionStartIndices(const popsparse::dynamic::FullyConnectedParams &params,
                         const Plan &plan) {
  auto createSplit = [](unsigned size, unsigned partitionSize,
                        unsigned grainSize) {
    auto grains = gccs::ceildiv(size, grainSize);
    std::vector<std::size_t> split;
    const auto grainsPerPartition = gccs::ceildiv(grains, partitionSize);
    for (unsigned i = 0; i != partitionSize; ++i) {
      const auto tileBegin = i * grainsPerPartition * grainSize;
      split.push_back(tileBegin);
    }
    return split;
  };
  auto xSplits = createSplit(params.getOutputChannelsPerGroup(),
                             plan.partition.x, plan.method.grouping.x);
  auto ySplits = createSplit(params.getInputChannelsPerGroup(),
                             plan.partition.y, plan.method.grouping.y);
  auto zSplits = createSplit(params.getBatchSize(), plan.partition.z,
                             plan.method.grouping.z);

  return {xSplits, ySplits, zSplits};
}

std::tuple<Plan, Cost> getPlan(const Target &target, const Type &inputType,
                               const FullyConnectedParams &params,
                               const OptionFlags &optionFlags,
                               PlanningCache *cache) {
  // TODO: Verify some basic things about the input.
  const auto &options = parseOptionFlags(optionFlags);

  auto cacheImpl = cache ? cache->impl.get() : nullptr;
  PlanningCacheImpl::Key key(params, options);
  if (cacheImpl) {
    auto &plans = cacheImpl->plans;
    auto match = plans.find(key);
    if (match != plans.end()) {
      return match->second;
    }
  }

  auto planAndCost = runPlanner(target, inputType, params, options);
  if (cacheImpl) {
    cacheImpl->plans.emplace(key, planAndCost);
  }
  return planAndCost;
}

unsigned int getTotalMetaInfoElemsPerBuckets(const Plan &plan) {
  return plan.fwdMetaInfoElemsPerBucket +
         (plan.sharedBuckets() ? 0 : plan.gradAMetaInfoElemsPerBucket);
}

} // end namespace fullyconnected
} // end namespace popsparse
