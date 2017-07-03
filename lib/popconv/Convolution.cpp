#include "popconv/Convolution.hpp"
#include "ConvPlan.hpp"
#include <limits>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include "popconv/ConvUtil.hpp"
#include "popstd/Pad.hpp"
#include "popstd/Add.hpp"
#include "popstd/ActivationMapping.hpp"
#include "popreduce/Reduce.hpp"
#include "popstd/Regroup.hpp"
#include "popstd/VertexTemplates.hpp"
#include "util/gcd.hpp"
#include "PerformanceEstimation.hpp"
#include "VertexOptim.hpp"
#include "popstd/exceptions.hpp"
#include "popstd/Cast.hpp"
#include "popstd/Util.hpp"
#include "Winograd.hpp"
#include "popstd/Zero.hpp"
#include "popstd/Operations.hpp"
#include <unordered_set>
#include <boost/icl/interval_map.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popstd;

namespace popconv {

// Groups tensor from th standard [N][H][W][C] shape to internal shape
// [N][C1][H][W][C2]
//
// where C1 * C2 = C
static Tensor groupActivations(const Tensor &act) {
  auto chansPerGroup = detectChannelGrouping(act);
  assert(act.rank() == 4);
  assert(act.dim(3) % chansPerGroup == 0);
  const unsigned chanGroups = act.dim(3) / chansPerGroup;
  return act.reshape({act.dim(0), act.dim(1), act.dim(2),
                      chanGroups, chansPerGroup}).dimShuffle({0, 3, 1, 2, 4});
}

// Ungroups tensor from internal shape [N][C1][H][W][C2] to standard shape
// [N][H][W][C]
//
// where C1 * C2 = C
static Tensor ungroupActivations(const Tensor &act) {
  assert(act.rank() == 5);
  return act.dimShuffle({0, 2, 3, 1, 4}).reshape(
               {act.dim(0), act.dim(2), act.dim(3), act.dim(1) * act.dim(4)});
}

static std::pair<unsigned, unsigned>
detectWeightsChannelGrouping(const Tensor &w) {
  auto inChansPerGroup = detectChannelGrouping(w);
  const auto w1 =
      regroup(w, inChansPerGroup).reshape({w.dim(3) / inChansPerGroup,
                                           w.dim(0), w.dim(1),
                                           w.dim(2) * inChansPerGroup});
  auto outChansPerGroup = detectChannelGrouping(w1);
  assert(outChansPerGroup % inChansPerGroup == 0);
  outChansPerGroup /= inChansPerGroup;
  return {outChansPerGroup, inChansPerGroup};
}

// Groups tensor from standard convolution weight tensor shape [H][W][OC][IC] to
// internal shape [OC1][IC1][H][W][OC2][IC2]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
static Tensor groupWeights(const Tensor &weights, unsigned inChansPerGroup,
                           unsigned outChansPerGroup) {
  assert(weights.rank() == 4);
  assert(weights.dim(3) % inChansPerGroup == 0);
  assert(weights.dim(2) % outChansPerGroup == 0);
  const unsigned inChanGroups = weights.dim(3) / inChansPerGroup;
  const unsigned outChanGroups = weights.dim(2) / outChansPerGroup;

  return weights.reshape({weights.dim(0), weights.dim(1),
                          outChanGroups, outChansPerGroup,
                          inChanGroups, inChansPerGroup})
                .dimShuffle({2, 4, 0, 1, 3, 5});
}


static Tensor groupWeights(const Tensor &weights) {
  unsigned inChansPerGroup, outChansPerGroup;
  std::tie(outChansPerGroup, inChansPerGroup) =
      detectWeightsChannelGrouping(weights);
  return groupWeights(weights, inChansPerGroup, outChansPerGroup);
}

// Ungroups tensors from internal shape [OC1][IC1][H][W][OC2][IC2] to standard
// convolution weight tensor shape [H][W][OC][IC]
//
// where OC1 * OC2 = OC
// and   IC1 * IC2 = IC
static Tensor ungroupWeights(const Tensor &weights) {
  assert(weights.rank() == 6);
  return weights.dimShuffle({2, 3, 0, 4, 1, 5})
                .reshape({weights.dim(2), weights.dim(3),
                          weights.dim(0) * weights.dim(4),
                          weights.dim(1) * weights.dim(5)});
}

std::size_t ConvParams::getOutputSize(unsigned dim) const {
  auto paddedDilatedInputSize = getPaddedDilatedInputSize(dim);
  auto paddedDilatedKernelSize = getPaddedDilatedKernelSize(dim);
  return absdiff(paddedDilatedInputSize, paddedDilatedKernelSize) /
                 stride[dim] + 1;
}

std::size_t ConvParams::getOutputHeight() const {
  return getOutputSize(0);
}

std::size_t ConvParams::getOutputWidth() const {
  return getOutputSize(1);
}

std::vector<std::size_t> ConvParams::getOutputShape() const {
  return {inputShape[0], getOutputSize(0), getOutputSize(1),
          kernelShape[2]};
}

static void
applyTensorMapping(
    Graph &graph,
    const Tensor &t,
    const std::vector<
      std::vector<Interval<std::size_t>>
    > &mapping) {
  auto flattened = t.flatten();
  const auto numTiles = mapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &region : mapping[tile]) {
      graph.setTileMapping(flattened.slice(region.begin(), region.end()),
                           tile);
    }
  }
}

struct WeightGradAopTask {
  unsigned kernelY;
  unsigned kernelX;
  unsigned outZGroup;
  unsigned inZGroup;
  WeightGradAopTask(unsigned kernelY, unsigned kernelX,
                    unsigned outZGroup, unsigned inZGroup) :
    kernelY(kernelY), kernelX(kernelX),
    outZGroup(outZGroup), inZGroup(inZGroup) {}

};

static void
verifyStrideAndPaddingDimensions(const ConvParams &params) {
  if (params.stride.size() != 2) {
    throw popstd::poplib_error("Only 2D stride is valid");
  }
  if (params.inputPaddingLower.size() != 2) {
    throw popstd::poplib_error("Only 2D inputPaddingLower is valid");
  }
  if (params.inputPaddingUpper.size() != 2) {
    throw popstd::poplib_error("Only 2D inputPaddingUpper is valid");
  }
}

static void verifyInputShapes(const ConvParams &params,
                              const Tensor &in,
                              const Tensor &weights) {
  if (params.inputShape[0] != in.dim(0)) {
    throw popstd::poplib_error("Batchsize of input tensor does not match "
                               "convolution parameters");
  }
  if (params.inputShape[1] != in.dim(2)) {
    throw popstd::poplib_error("Height of input tensor does not match "
                               "convolution parameters");
  }
  if (params.inputShape[2] != in.dim(3)) {
    throw popstd::poplib_error("Width of input tensor does not match "
                               "convolution parameters");
  }
  if (params.inputShape[3] != in.dim(1) * in.dim(4)) {
    throw popstd::poplib_error("Number of channels of input tensor does "
                               "not match convolution parameters");
  }
  if (params.kernelShape[0] != weights.dim(2)) {
    throw popstd::poplib_error("Kernel height does not match convolution "
                               "parameters");
  }
  if (params.kernelShape[1] != weights.dim(3)) {
    throw popstd::poplib_error("Kernel width does not match convolution "
                               "parameters");
  }
  if (params.kernelShape[2] != weights.dim(0) * weights.dim(4)) {
    throw popstd::poplib_error("Kernel output channel size does not match "
                               "convolution parameters");
  }
  if (params.kernelShape[3] != weights.dim(1) * weights.dim(5)) {
    throw popstd::poplib_error("Kernel input channel size does not match "
                               "convolution parameters");
  }
}

static unsigned
getInChansPerGroup(const Plan &plan, unsigned numInChans) {
  return gcd(plan.inChansPerGroup, numInChans);
}

static unsigned
getWeightInChansPerGroup(const Plan &plan, unsigned numInChans) {
  return gcd(plan.inChansPerGroup, numInChans);
}

static unsigned
getWeightOutChansPerGroup(const Plan &plan, unsigned numOutChans) {
  return gcd(plan.partialChansPerGroup, numOutChans);
}

static unsigned
getOutChansPerGroup(const Plan &plan, unsigned numOutChans) {
  return gcd(plan.partialChansPerGroup, numOutChans);
}

poplar::Tensor
createBiases(poplar::Graph &graph, const Tensor &acts,
             const std::string &name) {
  const auto numOutChans = acts.dim(3);
  const auto dType = acts.elementType();
  auto biases = graph.addTensor(dType, {numOutChans}, name);
  mapBiases(graph, biases, acts);
  return biases;
}

static unsigned
linearizeTileIndices(unsigned batchGroup, unsigned numBatchGroups,
                     unsigned numTiles,
                     unsigned ky, unsigned izg,
                     unsigned ox, unsigned oy, unsigned ozg,
                     const Plan &plan,
                     bool isMultiIPU) {
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto tilesPerKernelYAxis = plan.tilesPerKernelYAxis;
  unsigned beginTile;
  if (numBatchGroups <= numTiles) {
    beginTile = (numTiles / numBatchGroups) * batchGroup;
  } else {
    const auto batchGroupsPerTile = (numBatchGroups + numTiles - 1) / numTiles;
    beginTile = batchGroup / batchGroupsPerTile;
  }
  // If this is a multi IPU system then choose an order that avoids splitting
  // partial sums over IPUs
  unsigned tile;
  switch (plan.linearizeTileOrder) {
  case Plan::LinearizeTileOrder::FC_WU:
    // For the fully connected weight update the in group and out group are
    // swapped compared to the forward pass.
    if (isMultiIPU)
      tile = beginTile +
        (ky + tilesPerKernelYAxis *
          (ozg + tilesPerZ *
            (ox + tilesPerX *
              (oy + tilesPerY * izg))));
    else
      tile = beginTile +
             (izg + tilesPerInZGroup *
               (ox + tilesPerX *
                 (oy + tilesPerY *
                   (ky + tilesPerKernelYAxis *
                     ozg))));
    break;
  case Plan::LinearizeTileOrder::FC_BWD_AS_CONV:
    // For the fully connected backward pass the width and the input channels
    // are swapped compared to the forward pass.
    if (isMultiIPU)
      tile = beginTile +
        (ky + tilesPerKernelYAxis *
          (ox + tilesPerX *
            (izg + tilesPerInZGroup *
              (oy + tilesPerY * ozg))));
    else
      tile = beginTile +
             (ozg + tilesPerZ *
               (izg + tilesPerInZGroup *
                 (oy + tilesPerY *
                   (ky + tilesPerKernelYAxis *
                     ox))));
    break;
  case Plan::LinearizeTileOrder::STANDARD:
    if (isMultiIPU)
      tile = beginTile +
        (ky + tilesPerKernelYAxis *
          (izg + tilesPerInZGroup *
            (ox + tilesPerX *
              (oy + tilesPerY * ozg))));
    // Use ozg as the innermost dimension to increase the chance that
    // tiles in a supertile both read the same activations. This reduces
    // exchange time when supertile send / receive is used.
    else
      tile = beginTile +
             (ozg + tilesPerZ *
               (ox + tilesPerX *
                 (oy + tilesPerY *
                   (ky + tilesPerKernelYAxis *
                     izg))));
    break;
  }
  assert(tile < numTiles);
  return tile;
}

static std::pair<unsigned,unsigned>
getOutZGroupRange(unsigned ozgIndex, unsigned partialNumChanGroups,
                  const Plan &plan) {
  const auto tilesPerZAxis = plan.tilesPerZAxis;
  const auto maxZGroupsPerTile = (partialNumChanGroups + tilesPerZAxis - 1) /
                                 tilesPerZAxis;
  const auto outZBegin =
      std::min(ozgIndex * maxZGroupsPerTile, partialNumChanGroups);
  const auto outZEnd =
      std::min((ozgIndex + 1) * maxZGroupsPerTile, partialNumChanGroups);
  return {outZBegin, outZEnd};
}

static unsigned
getFlattenedIndex(const std::vector<std::size_t> &shape,
                  const std::vector<std::size_t> &indices) {
  const auto rank = shape.size();
  assert(indices.size() == rank);
  unsigned index = 0;
  for (unsigned dim = 0; dim != rank; ++dim) {
    assert(indices[dim] < shape[dim]);
    index *= shape[dim];
    index += indices[dim];
  }
  return index;
}

static void
addFlattenedRegions(const std::vector<std::size_t> &shape,
                    const std::vector<std::size_t> &begin,
                    const std::vector<std::size_t> &end,
                    std::vector<Interval<std::size_t>> &regions) {
  const auto numDims = shape.size();
  assert(begin.size() == numDims);
  assert(end.size() == numDims);

  for (unsigned dim = 0; dim != numDims; ++dim) {
    if (begin[dim] == end[dim])
      return;
  }

  std::vector<std::size_t> indices = begin;
  bool done = false;
  while (!done) {
    unsigned regionBegin = getFlattenedIndex(shape, indices);
    unsigned regionEnd = regionBegin + (end.back() - begin.back());
    regions.emplace_back(regionBegin, regionEnd);
    done = true;
    for (unsigned dim = 0; dim != numDims - 1; ++dim) {
      if (indices[dim] + 1 == end[dim]) {
        indices[dim] = begin[dim];
      } else {
        ++indices[dim];
        done = false;
        break;
      }
    }
  }
}


struct ConvTileIndices {
  unsigned b;
  unsigned oy;
  unsigned ox;
  unsigned ozg;
  unsigned izg;
  unsigned ky;
};

struct ConvSlice {
  unsigned batchBegin, batchEnd;
  unsigned outYBegin, outYEnd;
  unsigned outXBegin, outXEnd;
  unsigned outZGroupBegin, outZGroupEnd;
  unsigned inZGroupBegin, inZGroupEnd;
  unsigned kernelYBegin, kernelYEnd;
};

static void
iterateTilePartition(const Graph &graph, const ConvParams &params,
                     const Plan &plan,
                     const std::function<
                       void(unsigned, const ConvTileIndices &,
                            const ConvSlice &)
                     > &f) {
  assert(plan.batchesPerGroup == 1);
  const unsigned numBatchGroups = params.getBatchSize();
  const auto isMultiIPU = graph.getDevice().getDeviceInfo().numIPUs > 1;
  const unsigned inNumChans = params.getInputDepth();
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(params.getOutputDepth() % partialChansPerGroup == 0);
  const auto partialNumChanGroups =
      params.getOutputDepth() / partialChansPerGroup;
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto tilesPerZ = plan.tilesPerZAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const unsigned outDimY = params.getOutputHeight();
  const unsigned outDimX = params.getOutputWidth();
  const unsigned numInZGroups = inNumChans / inChansPerGroup;
  const unsigned kernelHeight = params.kernelShape[0];
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  for (unsigned b = 0; b < numBatchGroups; ++b) {
    for (unsigned izg = 0; izg != tilesPerInZGroup; ++izg) {
      const auto inZGroupBegin = (izg * numInZGroups) / tilesPerInZGroup;
      const auto inZGroupEnd = ((izg + 1) * numInZGroups) / tilesPerInZGroup;
      for (unsigned ky = 0; ky != tilesPerKernelY; ++ky) {
        const auto kernelYBegin = (ky * kernelHeight) / tilesPerKernelY;
        const auto kernelYEnd = ((ky + 1) * kernelHeight) / tilesPerKernelY;
        for (unsigned ozg = 0; ozg != tilesPerZ; ++ozg) {
          unsigned outZGroupBegin, outZGroupEnd;
          std::tie(outZGroupBegin, outZGroupEnd) =
              getOutZGroupRange(ozg, partialNumChanGroups, plan);
          for (unsigned oy = 0; oy != tilesPerY; ++oy) {
            const auto outYBegin = (oy * outDimY) / tilesPerY;
            const auto outYEnd = ((oy + 1) * outDimY) / tilesPerY;
            for (unsigned ox = 0; ox != tilesPerX; ++ox) {
              const auto xAxisGrainSize = plan.xAxisGrainSize;
              const auto numXGrains = (outDimX + xAxisGrainSize - 1) /
                                      plan.xAxisGrainSize;
              const auto outXGrainBegin = (ox * numXGrains) / tilesPerX;
              const auto outXGrainEnd = ((ox + 1) * numXGrains) / tilesPerX;
              const auto outXBegin = outXGrainBegin * xAxisGrainSize;
              const auto outXEnd = std::min(outXGrainEnd * xAxisGrainSize,
                                            outDimX);
              const auto tile = linearizeTileIndices(b, numBatchGroups,
                                                     numTiles,
                                                     ky, izg,
                                                     ox, oy, ozg,
                                                     plan,
                                                     isMultiIPU);
              f(tile,
                {b, oy, ox, ozg, izg, ky},
                {b, b + 1,
                 outYBegin, outYEnd,
                 outXBegin, outXEnd,
                 outZGroupBegin, outZGroupEnd,
                 inZGroupBegin, inZGroupEnd,
                 kernelYBegin, kernelYEnd});
            }
          }
        }
      }
    }
  }
}

static std::vector<std::vector<Interval<std::size_t>>>
convertLinearMappingToRegionMapping(const std::vector<unsigned> &mapping) {
  assert(!mapping.empty());
  const auto numTiles = mapping.size() - 1;
  std::vector<std::vector<Interval<std::size_t>>>
      regionMapping(numTiles);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    if (mapping[tile] == mapping[tile + 1])
      continue;
    regionMapping[tile].emplace_back(mapping[tile], mapping[tile + 1]);
  }
  return regionMapping;
}

/// Extend a partial map to a total map in the range [lower, upper). The value
/// of keys not in the partial map are based on the value of the neighbouring
/// keys that are in the map. The partial map must contain at least one entry.
template <class K, class V> static void
extendPartialMap(boost::icl::interval_map<K, V> &map,
                 K lower, K upper) {
  assert(iterative_size(map) >= 0);
  boost::icl::interval_map<K, V> extendedMap;
  for (auto begin = map.begin(), it = begin, end = map.end(); it != end;
       ++it) {
    const auto &interval = it->first;
    auto next = std::next(it);
    auto extendedIntervalLower = it == begin ? lower : interval.lower();
    auto extendedIntervalUpper = next == end ? upper : next->first.lower();
    auto extendedInterval =
        boost::icl::interval<unsigned>::right_open(extendedIntervalLower,
                                                   extendedIntervalUpper);
    extendedMap.insert({extendedInterval, std::move(it->second)});
  }
  std::swap(map, extendedMap);
}

static bool
isHaloRegion(
    const std::set<unsigned> &prevTiles,
    const std::set<unsigned> &tiles,
    const std::set<unsigned> &nextTiles) {
  if (prevTiles.size() + nextTiles.size() != tiles.size())
    return false;
  return std::includes(tiles.begin(), tiles.end(),
                       prevTiles.begin(), prevTiles.end()) &&
         std::includes(tiles.begin(), tiles.end(),
                       nextTiles.begin(), nextTiles.end());
}

static void
optimizeHaloMapping(boost::icl::interval_map<
                      unsigned, std::set<unsigned>
                    > &map) {
  // Modify the map so that "halo" regions where the uses are the union of the
  // uses of the neighbouring regions are mapped as if they were only used by
  // one of the sets of tiles. This heuristic reduces exchange code for
  // convolutional layers since the halos tend to be small and mapping them
  // independently splits up the tensor tile mapping, increasing the amount of
  // exchange code required.
  boost::icl::interval_map<unsigned, std::set<unsigned>> optimizedMap;
  for (auto begin = map.begin(), it = begin, end = map.end(); it != end;
       ++it) {
    if (it != begin && std::next(it) != end &&
        isHaloRegion(std::prev(it)->second,
                     it->second,
                     std::next(it)->second)) {
      optimizedMap.insert({it->first, it == begin ? std::next(it)->second :
                                                    std::prev(it)->second});
    } else {
      optimizedMap.insert(*it);
    }
  }
  std::swap(map, optimizedMap);
}

static std::vector<std::vector<Interval<std::size_t>>>
calculateMappingBasedOnUsage(const Graph &graph,
                             const std::vector<std::size_t> &shape,
                             const boost::icl::interval_map<
                               unsigned, std::set<unsigned>
                             > &uses,
                             unsigned grainSize,
                             unsigned minElementsPerTile) {
  if (iterative_size(uses) == 0) {
    return convertLinearMappingToRegionMapping(
      computeTensorMapping(graph, shape, grainSize)
    );
  }
  boost::icl::interval_map<unsigned, std::set<unsigned>> grainToTiles;
  for (const auto &entry : uses) {
    const auto &interval = entry.first;
    unsigned grainLower = interval.lower() / grainSize;
    unsigned grainUpper = (interval.upper() - 1) / grainSize + 1;
    auto grainInterval =
        boost::icl::interval<unsigned>::right_open(grainLower, grainUpper);
    grainToTiles.insert({grainInterval, entry.second});
  }

  // Extend the grainUses map to total map.
  const auto numElements = std::accumulate(shape.begin(), shape.end(), 1UL,
                                           std::multiplies<std::size_t>());
  const unsigned numGrains = (numElements + grainSize - 1) / grainSize;
  extendPartialMap(grainToTiles, 0U, numGrains);

  optimizeHaloMapping(grainToTiles);

  // Build a map from sets of tiles to grains they use.
  std::map<std::set<unsigned>, std::vector<Interval<std::size_t>>>
      tilesToGrains;
  for (const auto &entry : grainToTiles) {
    tilesToGrains[entry.second].emplace_back(entry.first.lower(),
                                             entry.first.upper());
  }
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  std::vector<std::vector<Interval<std::size_t>>> mapping(numTiles);
  const auto minGrainsPerTile =
      (minElementsPerTile + grainSize - 1) / grainSize;
  for (const auto &entry : tilesToGrains) {
    const auto &tiles = entry.first;
    const auto &sharedGrains = entry.second;
    const auto perTileGrains =
        splitRegions(sharedGrains, 1, tiles.size(), minGrainsPerTile);
    unsigned i = 0;
    for (auto tile : tiles) {
      if (i == perTileGrains.size())
        break;
      mapping[tile].reserve(perTileGrains[i].size());
      for (const auto &interval : perTileGrains[i]) {
        const auto lower = interval.begin() * grainSize;
        const auto upper = std::min(interval.end() * grainSize, numElements);
        mapping[tile].emplace_back(lower, upper);
      }
      ++i;
    }
  }
  return mapping;
}

static boost::icl::discrete_interval<unsigned>
toIclInterval(const Interval<std::size_t> &interval) {
  return boost::icl::interval<unsigned>::right_open(interval.begin(),
                                                    interval.end());
}

static std::vector<std::vector<Interval<std::size_t>>>
calculateActivationMapping(Graph &graph, const ConvParams &params,
                           const Plan &plan) {
  // Build a map from activations to the set of tiles that access them.
  const auto numInChans = params.getInputDepth();
  assert(numInChans % plan.inChansPerGroup == 0);
  const auto numInChanGroups = numInChans / plan.inChansPerGroup;
  std::vector<std::size_t> actsShape = {
    params.getBatchSize(),
    numInChanGroups,
    params.getInputHeight(),
    params.getInputWidth(),
    plan.inChansPerGroup
  };
  boost::icl::interval_map<unsigned, std::set<unsigned>> actsToTiles;
  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile, const ConvTileIndices &,
                           const ConvSlice &slice) {
    auto inYRange =
        getInputRange(0, {slice.outYBegin, slice.outYEnd},
                      {slice.kernelYBegin, slice.kernelYEnd}, params);
    auto inXRange =
        getInputRange(1, {slice.outXBegin, slice.outXEnd}, params);
    std::vector<Interval<std::size_t>> intervals;
    addFlattenedRegions(actsShape,
                        {slice.batchBegin,
                         slice.inZGroupBegin,
                         inYRange.first,
                         inXRange.first,
                         0},
                        {slice.batchEnd,
                         slice.inZGroupEnd,
                         inYRange.second,
                         inXRange.second,
                         plan.inChansPerGroup},
                        intervals);
    for (const auto &interval : intervals) {
      actsToTiles += std::make_pair(toIclInterval(interval),
                                    std::set<unsigned>({tile}));
    }
  });
  // Limit the minimum number of activation bytes per tile to reduce the amount
  // of exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto actType = params.dType;
  const auto actTypeSize = actType == "float" ? 4 : 2;
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + actTypeSize - 1) / minBytesPerTile;
  const auto grainSize = plan.inChansPerGroup;
  return calculateMappingBasedOnUsage(graph, actsShape, actsToTiles,
                                      grainSize, minElementsPerTile);
}

/// Apply any pre-convolution transformations implied by the plan. The
/// plan and the parameters are updated to describe the convolution operation
/// performed on the transformed input. If the \a acts or \ weights pointers are
/// not null they are updated to be views of the original tensors with
/// dimensions that match the shape expected by the convolution operation.
static void
convolutionPreprocess(Graph &graph, ConvParams &params, Plan &plan,
                      Tensor *acts, Tensor *weights) {
  assert(plan.flattenXY || plan.batchesPerGroup == 1);
  if (plan.flattenXY) {
    const auto batchSize = params.getBatchSize();
    const auto batchesPerGroup = plan.batchesPerGroup;
    assert(batchSize % batchesPerGroup == 0);
    const auto numBatchGroups = batchSize / plan.batchesPerGroup;
    if (acts) {
      *acts = acts->dimShuffle({1, 0, 2, 3, 4})
                   .reshape({acts->dim(1),
                            numBatchGroups,
                            1,
                            batchesPerGroup * acts->dim(2) * acts->dim(3),
                            acts->dim(4)})
                   .dimShuffle({1, 0, 2, 3, 4});
    }
    plan.flattenXY = false;
    plan.batchesPerGroup = 1;
    params.inputShape[2] *= params.inputShape[1] * batchesPerGroup;
    params.inputShape[1] = 1;
    params.inputShape[0] /= batchesPerGroup;
  }
  const auto numInChans = params.getInputDepth();
  const auto convInChansPerGroup = plan.inChansPerGroup;
  const auto convNumChanGroups =
      (numInChans + convInChansPerGroup - 1) / convInChansPerGroup;
  const auto convNumChans = convInChansPerGroup * convNumChanGroups;
  if (convNumChans != numInChans) {
    // Zero pad the input / weights.
    if (acts) {
      auto inRegrouped = regroup(*acts, numInChans);
      auto inRegroupedPadded =
          pad(graph, inRegrouped, 0, convNumChans - numInChans, 4);
      *acts = regroup(inRegroupedPadded, plan.inChansPerGroup);
    }
    if (weights) {
      auto weightsRegrouped = regroup(*weights, 1, 5, numInChans);
      auto weightsRegroupedPadded =
          pad(graph, weightsRegrouped, 0, convNumChans - numInChans, 5);
      *weights = regroup(weightsRegroupedPadded, 1, 5, plan.inChansPerGroup);
    }
    params.inputShape[3] = convNumChans;
    params.kernelShape[3] = convNumChans;
  } else if (acts && acts->dim(4) != plan.inChansPerGroup) {
    *acts = regroup(*acts, plan.inChansPerGroup);
  }
  const auto outNumChans = params.getOutputDepth();
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  const auto partialNumChanGroups =
      (outNumChans + partialChansPerGroup - 1) / partialChansPerGroup;
  const auto partialNumChans = partialNumChanGroups * partialChansPerGroup;
  if (partialNumChans != outNumChans) {
    if (weights) {
      auto weightsRegrouped = regroup(*weights, 0, 4, outNumChans);
      // Zero pad the weights in the out channel axis.
      auto weightsRegroupedPadded =
          pad(graph, weightsRegrouped, 0, partialNumChans - outNumChans, 4);
      *weights = regroup(weightsRegroupedPadded, 0, 4, partialChansPerGroup);
    }
    params.kernelShape[2] = partialNumChans;
  }
}

/// Map the activations tensor such that the exchange required during the
/// convolution operation is minimized.
static void mapActivations(Graph &graph, ConvParams params,
                           Plan plan, Tensor acts) {
  // Depending on the plan the input may be transformed prior to the
  // convolution. Apply the same transformation here.
  convolutionPreprocess(graph, params, plan, &acts, nullptr);
  // Compute a mapping for the transformed activations tensor that minimizes
  // exchange.
  auto mapping = calculateActivationMapping(graph, params, plan);
  // Apply the mapping to the transformed activations tensor. This indirectly
  // maps the original (non-transformed) tensor.
  graph.setTileMapping(acts, mapping);
}

static Tensor
createInput(Graph &graph, const ConvParams &params,
            const std::string &name,
            const Plan &plan) {
  const auto inNumChans = params.getInputDepth();
  const auto inChansPerGroup = getInChansPerGroup(plan, inNumChans);
  assert(params.getInputDepth() % inChansPerGroup == 0);
  auto t = graph.addTensor(params.dType,
                           {params.inputShape[0],
                            params.inputShape[3] / inChansPerGroup,
                            params.inputShape[1], params.inputShape[2],
                            inChansPerGroup},
                           name);
  mapActivations(graph, params, plan, t);
  return t;
}

Tensor
createInput(Graph &graph, const ConvParams &params,
            const std::string &name,
            const ConvOptions &options) {
  verifyStrideAndPaddingDimensions(params);
  const auto plan = getPlan(graph, params, options);
  return ungroupActivations(createInput(graph, params, name, plan));
}

static std::vector<std::vector<Interval<std::size_t>>>
calculateWeightMapping(const Graph &graph,
                       const ConvParams &params,
                       const Plan &plan) {
  // Build a map from weights to the set of tiles that access them.
  const auto numInChans = params.getInputDepth();
  assert(numInChans % plan.inChansPerGroup == 0);
  const auto numInChanGroups = numInChans / plan.inChansPerGroup;
  const auto numOutChans = params.getOutputDepth();
  assert(numOutChans % plan.partialChansPerGroup == 0);
  const auto numOutChanGroups = numOutChans / plan.partialChansPerGroup;
  const auto kernelHeight = params.kernelShape[0];
  const auto kernelWidth = params.kernelShape[1];
  std::vector<std::size_t> weightsShape = {
    numOutChanGroups,
    numInChanGroups,
    kernelHeight,
    kernelWidth,
    plan.partialChansPerGroup,
    plan.inChansPerGroup
  };
  boost::icl::interval_map<unsigned, std::set<unsigned>> weightsToTiles;
  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile, const ConvTileIndices &,
                           const ConvSlice &slice) {
    std::vector<Interval<std::size_t>> intervals;
    addFlattenedRegions(weightsShape,
                        {slice.outZGroupBegin,
                         slice.inZGroupBegin,
                         slice.kernelYBegin,
                         0,
                         0,
                         0},
                        {slice.outZGroupEnd,
                         slice.inZGroupEnd,
                         slice.kernelYEnd,
                         params.kernelShape[1],
                         plan.partialChansPerGroup,
                         plan.inChansPerGroup},
                        intervals);
    for (const auto &interval : intervals) {
      weightsToTiles += std::make_pair(toIclInterval(interval),
                                       std::set<unsigned>({tile}));
    }
  });
  // Limit the minimum number of weight bytes per tile to reduce the
  // amount of exchange code. Increasing this constant reduces exchange code
  // size and increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto weightType = params.dType;
  const auto weightTypeSize = weightType == "float" ? 4 : 2;
  const auto minBytesPerTile = 256;
  const auto minElementsPerTile =
    (minBytesPerTile + weightTypeSize - 1) / minBytesPerTile;
  unsigned grainSize = plan.partialChansPerGroup * plan.inChansPerGroup;
  return calculateMappingBasedOnUsage(graph, weightsShape, weightsToTiles,
                                      grainSize, minElementsPerTile);
}

static void mapWeights(Graph &graph, Tensor weights,
                       ConvParams params, Plan plan) {
  // Depending on the plan the weights may be transformed prior to the
  // convolution. Apply the same transformation here.
  convolutionPreprocess(graph, params, plan, nullptr, &weights);
  // Compute a mapping for the transformed weights tensor that minimizes
  // exchange.
  auto weightsMapping = calculateWeightMapping(graph, params, plan);
  // Apply the mapping to the transformed weights tensor. This indirectly
  // maps the original (non-transformed) tensor.
  graph.setTileMapping(weights, weightsMapping);
}

void
mapWeights(Graph &graph, const Tensor &w, const ConvParams &params,
           const ConvOptions &options) {
  verifyStrideAndPaddingDimensions(params);
  const auto plan = getPlan(graph, params, options);
  mapWeights(graph, groupWeights(w), params, plan);
}

static Tensor
createWeights(Graph &graph,
              const ConvParams &params, const std::string &name,
              const Plan &plan) {
  const auto dType = params.dType;
  const auto inNumChans = params.inputShape[3];
  const auto outNumChans = params.kernelShape[2];
  const auto weightOutChansPerGroup =
      getWeightOutChansPerGroup(plan, outNumChans);
  assert(outNumChans % weightOutChansPerGroup == 0);
  const auto weightNumOutChanGroups = outNumChans / weightOutChansPerGroup;
  const auto weightInChansPerGroup = getWeightInChansPerGroup(plan, inNumChans);
  assert(inNumChans % weightInChansPerGroup == 0);
  const auto weightNumInChanGroups = inNumChans / weightInChansPerGroup;
  auto weights = graph.addTensor(dType, {weightNumOutChanGroups,
                                         weightNumInChanGroups,
                                         params.kernelShape[0],
                                         params.kernelShape[1],
                                         weightOutChansPerGroup,
                                         weightInChansPerGroup},
                                 name);
  mapWeights(graph, weights, params, plan);
  return ungroupWeights(weights);
}

Tensor
createWeights(Graph &graph,
              const ConvParams &params, const std::string &name,
              const ConvOptions &options) {
  verifyStrideAndPaddingDimensions(params);
  const auto plan = getPlan(graph, params, options);
  return createWeights(graph, params, name, plan);
}

static std::vector<std::vector<poplar::Interval<std::size_t>>>
computeBiasMapping(Graph &graph, const Tensor &out) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dType = out.elementType();
  const auto dTypeSize = dType == "float" ? 4 : 2;
  const auto numTiles = graph.getDevice().getDeviceInfo().getNumTiles();
  const unsigned numChans = out.dim(1) * out.dim(4);
  auto outRegrouped = out.dimShuffle({1, 4, 0, 2, 3})
                         .reshape({numChans, out.numElements() / numChans});
  const auto outRegroupedMapping = graph.getTileMapping(outRegrouped);
  // Build a map from the bias to the set of tiles that access it.
  boost::icl::interval_map<unsigned, std::set<unsigned>> biasToTiles;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &interval : outRegroupedMapping[tile]) {
      unsigned chanBegin = interval.begin() / outRegrouped.dim(1);
      unsigned chanEnd = (interval.end() + outRegrouped.dim(1) - 1) /
                         outRegrouped.dim(1);
      auto biasInterval =
          boost::icl::interval<unsigned>::right_open(chanBegin, chanEnd);
      biasToTiles += std::make_pair(biasInterval, std::set<unsigned>({tile}));
    }
  }
  const auto grainSize =
      dType == "float" ? deviceInfo.getFloatVectorWidth() :
                         deviceInfo.getHalfVectorWidth();
  // Limit the minimum number of bias bytes per tile to reduce the amount of
  // exchange code. Increasing this constant reduces exchange code size and
  // increases execution time due to imbalance. The current limit was
  // chosen experimentally.
  const auto minBytesPerTile = 8;
  const auto minElementsPerTile =
      (minBytesPerTile + dTypeSize - 1) / dTypeSize;
  return calculateMappingBasedOnUsage(graph, {numChans}, biasToTiles,
                                      grainSize, minElementsPerTile);
}

void mapBiases(poplar::Graph &graph, const poplar::Tensor &biases,
               const poplar::Tensor &out) {
  auto groupedOut = groupActivations(out);
  auto mapping = computeBiasMapping(graph, groupedOut);
  applyTensorMapping(graph, biases, mapping);
}

static void
createConvPartial1x1OutVertex(Graph &graph,
                              unsigned tile,
                              unsigned outXBegin, unsigned outXEnd,
                              unsigned outYBegin, unsigned outYEnd,
                              unsigned ozg,
                              unsigned kernelY,
                              unsigned inZGroupBegin, unsigned inZGroupEnd,
                              const ConvParams &params,
                              ComputeSet fwdCS,
                              const Tensor &in, const Tensor &weights,
                              const Tensor &out) {
  const auto inChansPerGroup = static_cast<unsigned>(in.dim(3));
  const auto outChansPerGroup = static_cast<unsigned>(out.dim(3));
  const auto dType = in.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex = deviceInfo.numWorkerContexts;
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(dType == "float");
  const auto convUnitCoeffLoadBytesPerCycle =
                deviceInfo.convUnitCoeffLoadBytesPerCycle;
  const auto outHeight = outYEnd - outYBegin;
  const auto outWidth = outXEnd - outXBegin;
  const auto partialType = out.elementType();
  unsigned inYBegin, inYEnd, inXBegin, inXEnd;
  std::tie(inYBegin, inYEnd) =
      getInputRange(0, {outYBegin, outYEnd}, kernelY, params);
  std::tie(inXBegin, inXEnd) =
      getInputRange(1, {outXBegin, outXEnd}, params);

  std::vector<std::vector<PartialRow>> workerPartition;
  workerPartition =
      partitionConvPartialByWorker(outHeight, outWidth,
                                   contextsPerVertex, params.inputDilation);

  std::vector<Tensor> inputEdges;
  std::vector<Tensor> outputEdges;

  unsigned numConvolutions = 0;
  for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
    for (unsigned i = 0; i != contextsPerVertex; ++i) {
      for (const auto &partialRow : workerPartition[i]) {
        const auto workerOutY = outYBegin + partialRow.rowNumber;
        const auto workerOutXBegin = outXBegin + partialRow.begin;
        const auto workerOutXEnd = outXBegin + partialRow.end;
        const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
        const auto workerInY = getInputIndex(0, workerOutY, kernelY, params);
        assert(workerInY != ~0U);
        unsigned workerInXBegin, workerInXEnd;
        std::tie(workerInXBegin, workerInXEnd) =
            getInputRange(1, {workerOutXBegin, workerOutXEnd}, params);
        const auto workerInWidth = workerInXEnd - workerInXBegin;
        assert(workerInWidth != 0);
        Tensor inWindow =
            in[izg][workerInY].slice(
              {workerInXBegin, 0},
              {workerInXEnd, inChansPerGroup}
            ).reshape({workerInWidth * inChansPerGroup});
        Tensor outWindow =
            out[ozg][workerOutY].slice(
              {workerOutXBegin, 0},
              {workerOutXEnd, outChansPerGroup}
            ).reshape({workerOutWidth * outChansPerGroup});
        inputEdges.push_back(inWindow);
        outputEdges.push_back(outWindow);
        graph.setTileMapping(outWindow, tile);
        ++numConvolutions;
      }
    }
  }
  const auto numEdges = 1 + 2 * numConvolutions;

  // Add the vertex.
  Tensor w =
      weights[ozg].slice(
  {inZGroupBegin, kernelY, 0, 0, 0},
  {inZGroupEnd, kernelY + 1, 1, outChansPerGroup, inChansPerGroup}
        ).flatten();
  auto v = graph.addVertex(
        fwdCS,
        templateVertex("popconv::ConvPartial1x1Out", dType, partialType,
                       useDeltaEdgesForConvPartials(numEdges) ?
                                                    "true" : "false"),
        {{"weights", w}}
        );
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["weightsPerConvUnit"], weightsPerConvUnit);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                        convUnitCoeffLoadBytesPerCycle);
  graph.setFieldSize(v["weightReuseCount"], contextsPerVertex);
  for (unsigned i = 0; i != contextsPerVertex; ++i) {
    graph.setInitialValue(
          v["weightReuseCount"][i],
        static_cast<std::uint32_t>(workerPartition[i].size())
        );
  }
  graph.connect(v["in"], inputEdges);
  graph.connect(v["out"], outputEdges);
  // Map the vertex and output.
  graph.setTileMapping(v, tile);
}

static unsigned getNumConvUnits(bool floatActivations,
                                bool floatPartial,
                                const poplar::DeviceInfo &deviceInfo) {
  if (floatActivations) {
    return deviceInfo.fp32InFp32OutConvUnitsPerTile;
  } else {
    return floatPartial ? deviceInfo.fp16InFp32OutConvUnitsPerTile :
                          deviceInfo.fp16InFp16OutConvUnitsPerTile;
  }
}

static void
createConvPartialnx1Vertex(Graph &graph,
                           unsigned tile,
                           unsigned outXBegin, unsigned outXEnd,
                           unsigned outYBegin, unsigned outYEnd,
                           unsigned outZGroupBegin, unsigned outZGroupEnd,
                           unsigned kernelYBegin, unsigned kernelYEnd,
                           unsigned inZGroupBegin, unsigned inZGroupEnd,
                           bool isInOut,
                           const ConvParams &params,
                           ComputeSet fwdCS,
                           const Tensor &in,
                           const Tensor &weights,
                           const Tensor &out,
                           const Tensor &zeros) {
  if (outXBegin == outXEnd ||
      outYBegin == outYEnd ||
      kernelYBegin == kernelYEnd ||
      inZGroupBegin == inZGroupEnd)
    return;
  const auto kernelSizeX = weights.dim(3);
  const auto inChansPerGroup = static_cast<unsigned>(in.dim(3));
  const auto outChansPerGroup = static_cast<unsigned>(out.dim(3));
  const auto dType = in.elementType();
  const bool floatActivations = dType == "float";
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto contextsPerVertex = deviceInfo.numWorkerContexts;
  const auto weightsPerConvUnit =
      deviceInfo.getWeightsPerConvUnit(floatActivations);
  assert(weightsPerConvUnit % inChansPerGroup == 0);
  const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
  const auto convUnitCoeffLoadBytesPerCycle
                      = deviceInfo.convUnitCoeffLoadBytesPerCycle;
  const auto partialType = out.elementType();
  const bool floatPartials = partialType == "float";
  const auto outChansPerPass = getNumConvUnits(floatActivations, floatPartials,
                                               deviceInfo);
  assert(outChansPerGroup % outChansPerPass == 0);
  const auto passesPerOutputGroup = outChansPerGroup / outChansPerPass;
  const auto outStrideX = passesPerOutputGroup * params.inputDilation.back();

  // It is possible that there is no calculation to perform that involves
  // the specified output slice and kernel weight slice. Instead of adding a
  // vertex to the graph upfront add it lazily when we first need it.
  unsigned numWeights = 0;
  unsigned numConvolutions = 0;
  std::vector<Tensor> inputEdges;
  std::vector<Tensor> outputEdges;
  std::vector<bool> zeroOut;
  std::vector<Tensor> weightEdges;
  std::vector<unsigned> weightReuseCount;

  for (unsigned wyBegin = kernelYBegin; wyBegin < kernelYEnd;
       wyBegin += convUnitWeightHeight) {
    const auto wyEnd = std::min(static_cast<unsigned>(kernelYEnd),
                                wyBegin + convUnitWeightHeight);
    unsigned convOutYBegin, convOutYEnd;
    std::tie(convOutYBegin, convOutYEnd) =
        getOutputRange(0, {outYBegin, outYEnd}, {wyBegin, wyEnd}, params);
    const auto convOutHeight = convOutYEnd - convOutYBegin;
    if (convOutHeight == 0)
      continue;
    for (unsigned wx = 0; wx != kernelSizeX; ++wx) {
      unsigned convOutXBegin, convOutXEnd;
      std::tie(convOutXBegin, convOutXEnd) =
          getOutputRange(1, {outXBegin, outXEnd}, wx, params);
      const auto convOutWidth = convOutXEnd - convOutXBegin;
      if (convOutWidth == 0)
        continue;
      // In a fractionally strided pass, if we are handling one row of the
      // kernel at a time, the partitioning of work across the workers can be
      // aware of the stride and only allocate work on the rows that get
      // affected.
      unsigned outputStrideY =
          convUnitWeightHeight == 1 ? params.inputDilation[0] : 1;
      std::vector<std::vector<PartialRow>> workerPartition =
          partitionConvPartialByWorker(convOutHeight, convOutWidth,
                                       contextsPerVertex,
                                       {outputStrideY,
                                        params.inputDilation.back()});
      assert(workerPartition.size() == contextsPerVertex);
      for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
        for (unsigned p = 0; p != passesPerOutputGroup; ++p) {
          const auto offsetInOutputGroup = p * outChansPerPass;
          for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
            for (unsigned wy = wyBegin; wy != wyBegin + convUnitWeightHeight;
                 ++wy) {
              Tensor w;
              if (wy < wyEnd) {
                w = weights[ozg][izg][wy][wx]
                        .slice(offsetInOutputGroup,
                               offsetInOutputGroup + outChansPerPass)
                        .flatten();
              } else {
                w = zeros.slice(0, inChansPerGroup * outChansPerPass);
              }
              weightEdges.push_back(w);
            }
            for (unsigned i = 0; i != contextsPerVertex; ++i) {
              weightReuseCount.push_back(
                static_cast<std::uint32_t>(workerPartition[i].size())
              );

              for (const auto &partialRow : workerPartition[i]) {
                const auto workerOutY = convOutYBegin + partialRow.rowNumber;
                unsigned workerOutXBegin, workerOutXEnd;
                std::tie(workerOutXBegin, workerOutXEnd) =
                    getOutputRange(1,
                                   {convOutXBegin + partialRow.begin,
                                    convOutXBegin + partialRow.end},
                                   wx,
                                   params);
                const auto workerOutWidth = workerOutXEnd - workerOutXBegin;
                unsigned workerInXBegin, workerInXEnd;
                std::tie(workerInXBegin, workerInXEnd) =
                    getInputRange(1, {workerOutXBegin, workerOutXEnd}, wx,
                                  params);
                const auto workerInWidth = workerInXEnd - workerInXBegin;
                for (unsigned wy = wyBegin;
                     wy != wyBegin + convUnitWeightHeight;
                     ++wy) {
                  const auto workerInY =
                      getInputIndex(0, workerOutY, wy, params);
                  Tensor inWindow;
                  if (workerInY == ~0U) {
                    inWindow = zeros.slice(0, workerInWidth * inChansPerGroup);
                  } else {
                    inWindow =
                        in[izg][workerInY].slice(
                          {workerInXBegin, 0},
                          {workerInXEnd, inChansPerGroup}
                        ).reshape({workerInWidth * inChansPerGroup});
                  }
                  inputEdges.push_back(inWindow);
                }
                Tensor outWindow =
                    out[ozg][workerOutY].slice(
                      {workerOutXBegin, 0},
                      {workerOutXEnd, outChansPerGroup}
                    ).reshape({workerOutWidth * outChansPerGroup})
                     .slice(offsetInOutputGroup,
                            (workerOutWidth - 1) * outChansPerGroup +
                            offsetInOutputGroup + outChansPerPass);
                // Note the output tensor is mapped in mapPartialSums.
                outputEdges.push_back(outWindow);
                // If this is an in/out vertex the partials are zeroed in
                // mapPartialSums. If this is an output vertex we zero the
                // partial sums on the first pass.
                if (!isInOut) {
                  zeroOut.push_back(izg == inZGroupBegin);
                }
                ++numConvolutions;
              }
            }
            ++numWeights;
          }
        }
      }
    }
  }
  if (numConvolutions == 0)
    return;

  const auto numEdges = numConvolutions * convUnitWeightHeight
                        + numConvolutions
                        + numWeights * convUnitWeightHeight;

  auto v = graph.addVertex(fwdCS,
                      templateVertex("popconv::ConvPartialnx1",
                                     dType, partialType,
                                     isInOut ? "true" : "false",
                                     useDeltaEdgesForConvPartials(numEdges) ?
                                                          "true" : "false"));
  graph.setInitialValue(v["inStride"], params.stride.back());
  graph.setInitialValue(v["outStride"], outStrideX);
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerPass);
  graph.setInitialValue(v["convUnitCoeffLoadBytesPerCycle"],
                          convUnitCoeffLoadBytesPerCycle);
  graph.setTileMapping(v, tile);
  graph.connect(v["in"], inputEdges);
  graph.connect(v["out"], outputEdges);
  graph.connect(v["weights"], weightEdges);
  graph.setInitialValue(v["zeroOut"], zeroOut);
  graph.setInitialValue(v["weightReuseCount"], weightReuseCount);
}

struct ConvOutputSlice {
  unsigned outXBegin;
  unsigned outXEnd;
  unsigned outY;
  unsigned outZGroup;
  ConvOutputSlice(unsigned outXBegin, unsigned outXEnd, unsigned outY,
                  unsigned outZGroup) :
    outXBegin(outXBegin), outXEnd(outXEnd),
    outY(outY), outZGroup(outZGroup) {}

};

static void
createConvPartialHorizontalMacVertex(
    Graph &graph,
    unsigned tile,
    const std::vector<ConvOutputSlice> &outRegions,
    unsigned kernelYBegin, unsigned kernelYEnd,
    unsigned inZGroupBegin, unsigned inZGroupEnd,
    const ConvParams &params,
    ComputeSet fwdCS,
    const Tensor &in,
    const Tensor &weights,
    const Tensor &out) {
  const auto kernelWidth = weights.dim(3);
  const auto dataPathWidth = graph.getDevice().getDeviceInfo().dataPathWidth;
  const auto dType = in.elementType();
  const auto partialType = out.elementType();
  const auto outChansPerGroup = out.dim(3);
  assert(outChansPerGroup == 1);
  (void)outChansPerGroup;
  std::vector<Tensor> inEdges;
  std::vector<Tensor> weightsEdges;
  std::vector<Tensor> outEdges;
  for (const auto &region : outRegions) {
    const auto ozg = region.outZGroup;
    const auto y = region.outY;
    const auto outXBegin = region.outXBegin;
    const auto outXEnd = region.outXEnd;
    for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
      for (unsigned ky = kernelYBegin; ky != kernelYEnd; ++ky) {
        for (unsigned kx = 0; kx != kernelWidth; ++kx) {
          unsigned inY = getInputIndex(0, y, ky, params);
          if (inY == ~0U)
            continue;
          auto inRange = getInputRange(1, {outXBegin, outXEnd}, kx, params);
          if (inRange.first == inRange.second)
            continue;
          auto outRange = getOutputRange(1, {outXBegin, outXEnd}, kx, params);
          Tensor inWindow =
              in[izg][inY].slice(inRange.first, inRange.second).flatten();
          Tensor weightsWindow = weights[ozg][izg][ky][kx].flatten();
          Tensor outWindow =
              out[ozg][y].slice(outRange.first, outRange.second).flatten();
          inEdges.emplace_back(std::move(inWindow));
          weightsEdges.emplace_back(std::move(weightsWindow));
          outEdges.emplace_back(std::move(outWindow));
        }
      }
    }
  }
  if (outEdges.empty())
    return;
  auto v = graph.addVertex(fwdCS,
                           templateVertex(
                             "popconv::ConvPartialHorizontalMac", dType,
                             partialType
                           ),
                           {{"in", inEdges},
                            {"weights", weightsEdges},
                            {"out", outEdges},
                           });
  graph.setInitialValue(v["inStride"], params.stride.back());
  graph.setInitialValue(v["outStride"], params.inputDilation.back());
  graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
  graph.setTileMapping(v, tile);
}

static void
mapPartialSums(Graph &graph, unsigned outXBegin, unsigned outXEnd,
               unsigned outYBegin, unsigned outYEnd,
               unsigned tileOutZGroupBegin, unsigned tileOutZGroupEnd,
               unsigned tile, bool zeroPartials, ComputeSet zeroCS,
               const Tensor &out) {
  Tensor flatOut = out.flatten();
  std::vector<Interval<std::size_t>> regionsToZero;
  for (unsigned ozg = tileOutZGroupBegin; ozg != tileOutZGroupEnd; ++ozg) {
    for (unsigned y = outYBegin; y != outYEnd; ++y) {
      const auto regionBegin = out.dim(3) *
                                (outXBegin + out.dim(2) *
                                 (y + out.dim(1) *
                                  ozg));
      const auto regionEnd = regionBegin + out.dim(3) * (outXEnd - outXBegin);
      graph.setTileMapping(flatOut.slice(regionBegin, regionEnd), tile);
      if (zeroPartials) {
        regionsToZero.emplace_back(regionBegin, regionEnd);
      }
    }
  }
  if (zeroPartials) {
    mergeAdjacentRegions(regionsToZero);
    zero(graph, out, regionsToZero, tile, zeroCS);
  }
}

static bool writtenRangeEqualsOutputRange(
    unsigned dim,
    std::pair<unsigned, unsigned> outRange,
    std::pair<unsigned, unsigned> kernelIndexRange,
    const ConvParams &params) {
  auto writtenYRange =
      getOutputRange(dim, outRange, kernelIndexRange, params);
  return writtenYRange == outRange;
}

static std::vector<std::vector<ConvOutputSlice>>
partitionConvOutputBetweenWorkers(const Graph &graph,
                                  unsigned outXBegin, unsigned outXEnd,
                                  unsigned outYBegin, unsigned outYEnd,
                                  unsigned outZGroupBegin,
                                  unsigned outZGroupEnd) {
  std::vector<std::vector<ConvOutputSlice>> perWorkerConvOutputSlices;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto outWidth = outXEnd - outXBegin;
  const auto outHeight = outYEnd - outYBegin;
  const auto numRows = outHeight * (outZGroupEnd - outZGroupBegin);
  const auto numWorkers = deviceInfo.numWorkerContexts;
  unsigned rowSplitFactor = numWorkers / gcd(numWorkers, numRows);
  unsigned numPartRows = numRows * rowSplitFactor;
  for (unsigned worker = 0; worker != numWorkers; ++worker) {
    const auto begin = (worker * numPartRows) / numWorkers;
    const auto end = ((worker + 1) * numPartRows) / numWorkers;
    perWorkerConvOutputSlices.emplace_back();
    for (unsigned partRow = begin; partRow != end; ++partRow) {
      auto row = partRow / rowSplitFactor;
      auto partInRow = partRow % rowSplitFactor;
      const auto ozg = outZGroupBegin + row / outHeight;
      const auto y = outYBegin + row % outHeight;
      const auto workerOutXBegin =
          outXBegin + (partInRow * outWidth) / rowSplitFactor;
      const auto workerOutXEnd =
          outXBegin + ((partInRow + 1) * outWidth) / rowSplitFactor;
      if (!perWorkerConvOutputSlices.back().empty() &&
          ozg == perWorkerConvOutputSlices.back().back().outZGroup &&
          y == perWorkerConvOutputSlices.back().back().outY) {
        perWorkerConvOutputSlices.back().back().outXEnd = workerOutXEnd;
      } else {
        perWorkerConvOutputSlices.back().emplace_back(workerOutXBegin,
                                                      workerOutXEnd, y, ozg);
      }
    }
  }
  return perWorkerConvOutputSlices;
}

static void
calcPartialConvOutput(Graph &graph,
                      const Plan &plan,
                      std::string dType,
                      unsigned tile,
                      unsigned outXBegin, unsigned outXEnd,
                      unsigned outYBegin, unsigned outYEnd,
                      unsigned outZGroupBegin, unsigned outZGroupEnd,
                      unsigned kernelYBegin, unsigned kernelYEnd,
                      unsigned inZGroupBegin, unsigned inZGroupEnd,
                      const ConvParams &params,
                      ComputeSet zeroCS,
                      ComputeSet fwdCS,
                      Tensor in, Tensor weights, Tensor out) {
  const auto tileKernelHeight = kernelYEnd - kernelYBegin;
  const auto kernelSizeX = weights.dim(3);
  const auto inChansPerGroup = plan.inChansPerGroup;
  const auto outChansPerGroup = plan.partialChansPerGroup;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  Tensor zeros;
  bool zeroPartialsBefore = true;
  bool useConvPartial1x1OutVertex = false;
  if (plan.useConvolutionInstructions) {
    const auto partialsType = out.elementType();
    const auto outChansPerPass = getNumConvUnits(dType == "float",
                                                 partialsType == "float",
                                                 deviceInfo);
    assert(outChansPerGroup % outChansPerPass == 0);
    const auto passesPerOutputGroup = outChansPerGroup / outChansPerPass;
    zeroPartialsBefore =
        kernelSizeX != 1 || tileKernelHeight != 1 ||
        (params.inputDilation[1] != 1 || params.inputDilation[0] != 1) ||
        !writtenRangeEqualsOutputRange(0, {outYBegin, outYEnd},
                                       {kernelYBegin, kernelYEnd}, params);
    useConvPartial1x1OutVertex = !zeroPartialsBefore &&
                                 passesPerOutputGroup == 1;
    const auto weightsPerConvUnit =
        deviceInfo.getWeightsPerConvUnit(dType == "float");
    assert(weightsPerConvUnit % inChansPerGroup == 0);
    const auto convUnitWeightHeight = weightsPerConvUnit / inChansPerGroup;
    if (!useConvPartial1x1OutVertex && convUnitWeightHeight != 1) {
      assert(plan.useConvolutionInstructions);
      const auto inputRange = getInputRange(1, {outXBegin, outXEnd}, params);
      const auto inputRangeSize = inputRange.second - inputRange.first;
      const auto zeroSize = std::max(inputRangeSize * inChansPerGroup,
                                     inChansPerGroup * outChansPerGroup);
      zeros = graph.addTensor(dType,
                              {zeroSize},
                              "zeros");
      if (zeroPartialsBefore) {
        // This isn't split across multiple workers since it can happen in
        // parallel with zeroing the partial sums.
        auto v = graph.addVertex(zeroCS, templateVertex("popstd::Zero", dType),
                                 {{"out", zeros}});
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setTileMapping(v, tile);
      } else {
        zero(graph, zeros, {{0, zeroSize}}, tile, zeroCS);
      }
      graph.setTileMapping(zeros, tile);
    }
  }
  if (useConvPartial1x1OutVertex) {
    assert(!zeroPartialsBefore);
    for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
      createConvPartial1x1OutVertex(graph, tile,
                                    outXBegin, outXEnd,
                                    outYBegin, outYEnd, ozg,
                                    kernelYBegin, inZGroupBegin, inZGroupEnd,
                                    params,
                                    fwdCS, in, weights, out);
    }
  } else {
    mapPartialSums(graph, outXBegin, outXEnd, outYBegin, outYEnd,
                   outZGroupBegin, outZGroupEnd, tile, zeroPartialsBefore,
                   zeroCS, out);
    if (plan.useConvolutionInstructions) {
      createConvPartialnx1Vertex(graph, tile, outXBegin, outXEnd,
                                 outYBegin, outYEnd,
                                 outZGroupBegin, outZGroupEnd,
                                 kernelYBegin, kernelYEnd,
                                 inZGroupBegin, inZGroupEnd,
                                 zeroPartialsBefore,
                                 params,
                                 fwdCS, in, weights, out,
                                 zeros);
    } else {
      assert(zeroPartialsBefore);
      auto perWorkerConvOutputSlices =
          partitionConvOutputBetweenWorkers(graph, outXBegin, outXEnd,
                                            outYBegin, outYEnd,
                                            outZGroupBegin, outZGroupEnd);
      for (const auto &workerConvOutputSlices : perWorkerConvOutputSlices) {
        createConvPartialHorizontalMacVertex(graph, tile,
                                             workerConvOutputSlices,
                                             kernelYBegin, kernelYEnd,
                                             inZGroupBegin, inZGroupEnd,
                                             params,
                                             fwdCS, in, weights, out);
      }
    }
  }
}

// Take an ordered list and return a list of ranges
// representing the contiguous regions in that list.
template <typename It>
static std::vector<std::pair<typename std::iterator_traits<It>::value_type,
                             typename std::iterator_traits<It>::value_type>>
getContiguousRegions(It begin,
                     It end)
{
  using T = typename std::iterator_traits<It>::value_type;
  std::vector<std::pair<T, T>> regions;
  unsigned curBegin = *begin;
  unsigned curEnd = curBegin + 1;
  auto it = begin + 1;
  while (it != end) {
    if (*it == curEnd) {
      ++curEnd;
    } else {
      regions.emplace_back(curBegin, curEnd);
      curBegin = *it;
      curEnd = curBegin + 1;
    }
    ++it;
  }
  regions.emplace_back(curBegin, curEnd);
  return regions;
}

static Program
calcPartialSums(Graph &graph,
                const Plan &plan,
                const ConvParams &params,
                std::string dType,
                Tensor in, Tensor weights, Tensor partials,
                const std::string &layerName) {
  ComputeSet zeroCS = graph.addComputeSet(layerName +"/Zero");
  ComputeSet convolveCS = graph.addComputeSet(layerName + "/Convolve");
  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile, const ConvTileIndices &indices,
                          const ConvSlice &slice) {
    if (slice.outZGroupBegin == slice.outZGroupEnd)
      return;
    assert(slice.batchEnd - slice.batchBegin == 1);
    unsigned b = slice.batchBegin;
    unsigned partialIndex =
        indices.izg * plan.tilesPerKernelYAxis + indices.ky;
    calcPartialConvOutput(graph, plan, dType, tile,
                          slice.outXBegin, slice.outXEnd,
                          slice.outYBegin, slice.outYEnd,
                          slice.outZGroupBegin, slice.outZGroupEnd,
                          slice.kernelYBegin, slice.kernelYEnd,
                          slice.inZGroupBegin,
                          slice.inZGroupEnd,
                          params,
                          zeroCS, convolveCS,
                          in[b], weights,
                          partials[partialIndex][b]);
  });
  Sequence prog;
  if (!graph.getComputeSet(zeroCS).empty()) {
    prog.add(Execute(zeroCS));
  }
  prog.add(Execute(convolveCS));
  return prog;
}

static Tensor
partialGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval<std::size_t>>> &
        tileGroupRegions,
    const Tensor &partials,
    unsigned outDepth,
    const std::string &resultType,
    ComputeSet cs) {
  const auto partialsDepth = partials.dim(0);
  assert(partialsDepth >= outDepth);
  auto outDims = partials.shape();
  outDims[0] = outDepth;
  Tensor out = graph.addTensor(resultType,
                               outDims,
                               "partialReduceOut");
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  const auto numTileGroups = tileGroupRegions.size();
  const auto grainSize =
      resultType == "float" ? deviceInfo.getFloatVectorWidth() :
                              deviceInfo.getHalfVectorWidth();
  for (unsigned i = 0; i != outDepth; ++i) {
    unsigned begin = (i * partialsDepth) / outDepth;
    unsigned end = ((i + 1) * partialsDepth) / outDepth;
    std::vector<std::vector<Interval<std::size_t>>>
        outSubMapping(numTiles);
    for (unsigned tileGroup = 0; tileGroup != numTileGroups; ++tileGroup) {
      const auto tilesInGroup = tileGroups[tileGroup].size();
      const auto tileBegin = (i * tilesInGroup) / outDepth;
      const auto tileEnd = ((i + 1) * tilesInGroup) / outDepth;
      const auto outSplitRegions =
          splitRegions(tileGroupRegions[tileGroup], grainSize,
                       tileEnd - tileBegin);
      for (unsigned j = 0; j != outSplitRegions.size(); ++j) {
        outSubMapping[tileGroups[tileGroup][j + tileBegin]] =
            outSplitRegions[j];
      }
    }
    applyTensorMapping(graph, out[i], outSubMapping);
    popreduce::reduce(graph, partials.slice(begin, end), out[i],
                      outSubMapping, cs);
  }
  return out;
}

static Tensor
groupedReduce(Graph &graph,
              const std::vector<std::vector<unsigned>> &tileGroups,
              const std::vector<
                std::vector<Interval<std::size_t>>
              > &tileGroupRegions,
              const Tensor &partials,
              const std::string &resultType,
              ComputeSet cs) {
  return partialGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
         1, resultType, cs).reshape(partials[0].shape());
}

/// Return the number of reduce stages to use for a reduction of the specified
/// reduction depth.
static unsigned getNumReduceStages(unsigned partialsDepth) {
  /// Using more reduce stages affects code size as follows.
  /// If the reduction depth is p then a single stage reduction requires each
  /// tile to receive p messages. If instead we break the reduction down into n
  /// stages then each stage involves a reduction of reduce p^(1/n) messages.
  /// The total number of messages is n*p^(1/n). For large p, increase n
  /// will reducing the total number of messages received which is turn likely
  /// to also reduce the exchange code size. The thresholds below have been
  /// chosen based on benchmarking.
  if (partialsDepth >= 125)
    return 3;
  if (partialsDepth >= 16)
    return 2;
  return 1;
}

/// Return a plan for how to split a reduction into multiple stages along with
/// an estimate of the cost of the plan. The first member of the pair is a
/// vector of the depth of each partials tensor in each intermediate stage.
/// If the vector is empty there are no intermediate stages and the reduction
/// is performed in a single step. The second member of the pair is an
/// estimated cost. The cost is an estimate of the average number of messages
/// required per tile.
static std::pair<std::vector<unsigned>, float>
getMultiStageReducePlanAndCost(unsigned partialsDepth, unsigned numStages) {
  if (numStages == 1) {
    return {{}, partialsDepth};
  }
  auto nextDepthRoundDown =
      static_cast<unsigned>(
        std::pow(static_cast<double>(partialsDepth),
                 (numStages - 1.0) / numStages)
      );
  std::vector<unsigned> roundDownPlan, roundUpPlan;
  float roundDownCost, roundUpCost;
  std::tie(roundDownPlan, roundDownCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundDown, numStages - 1);
  roundDownCost += static_cast<float>(partialsDepth) / nextDepthRoundDown;
  auto nextDepthRoundUp = nextDepthRoundDown + 1;
  std::tie(roundUpPlan, roundUpCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundUp, numStages - 1);
  roundUpCost += static_cast<float>(partialsDepth) / nextDepthRoundUp;
  if (roundDownCost < roundUpCost) {
    roundDownPlan.insert(roundDownPlan.begin(), nextDepthRoundDown);
    return {roundDownPlan, roundDownCost};
  }
  roundUpPlan.insert(roundUpPlan.begin(), nextDepthRoundUp);
  return {roundUpPlan, roundUpCost};
}

static std::vector<unsigned>
getMultiStageReducePlan(unsigned partialsDepth) {
  const auto numStages = getNumReduceStages(partialsDepth);
  return getMultiStageReducePlanAndCost(partialsDepth, numStages).first;
}

static Tensor
multiStageGroupedReduce(
    Graph &graph,
    const std::vector<std::vector<unsigned>> &tileGroups,
    const std::vector<std::vector<Interval<std::size_t>>> &
        tileGroupRegions,
    Tensor partials,
    const std::string &resultType,
    std::vector<ComputeSet> &computeSets,
    const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  auto plan = getMultiStageReducePlan(partialsDepth);
  for (unsigned i = computeSets.size(); i <= plan.size(); ++i) {
    computeSets.push_back(
      graph.addComputeSet(debugPrefix + "/Reduce" +
                             std::to_string(i))
    );
  }
  const auto partialsType = partials.elementType();
  for (unsigned i = 0; i != plan.size(); ++i) {
    partials = partialGroupedReduce(graph, tileGroups, tileGroupRegions,
                                    partials, plan[i], partialsType,
                                    computeSets[i]);
  }
  auto reduced = groupedReduce(graph, tileGroups, tileGroupRegions, partials,
                               resultType, computeSets[plan.size()]);
  return reduced;
}

static Tensor
multiStageGroupedReduce(Graph &graph,
                        Tensor partials,
                        const std::string &resultType,
                        std::vector<ComputeSet> &computeSets,
                        const std::string &debugPrefix) {
  const auto partialsDepth = partials.dim(0);
  // Build a map from the output to the set of tiles that contribute partial
  // sums.
  boost::icl::interval_map<unsigned, std::set<unsigned>> outputToTiles;
  for (unsigned i = 0; i != partialsDepth; ++i) {
    const auto tileMapping = graph.getTileMapping(partials[i]);
    for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
      for (const auto &interval : tileMapping[tile]) {
        outputToTiles += std::make_pair(toIclInterval(interval),
                                        std::set<unsigned>({tile}));
      }
    }
  }
  // Build a map from sets of tiles the outputs they contribute to.
  std::map<std::set<unsigned>, std::vector<Interval<std::size_t>>>
      tilesToOutputs;
  for (const auto &entry : outputToTiles) {
    tilesToOutputs[entry.second].emplace_back(entry.first.lower(),
                                              entry.first.upper());
  }
  std::vector<std::vector<unsigned>> tileGroups;
  std::vector<std::vector<Interval<std::size_t>>> tileGroupRegions;
  tileGroups.reserve(tilesToOutputs.size());
  tileGroupRegions.reserve(tilesToOutputs.size());
  for (const auto &entry : tilesToOutputs) {
    tileGroups.emplace_back(entry.first.begin(), entry.first.end());
    tileGroupRegions.push_back(std::move(entry.second));
  }
  return multiStageGroupedReduce(graph, tileGroups, tileGroupRegions, partials,
                                 resultType, computeSets, debugPrefix);
}

static std::pair<Program, Tensor>
convolutionByAmp(Graph &graph, const Plan &plan,
                 const ConvParams &params,
                 const Tensor &in, const Tensor &weights,
                 const std::string &debugPrefix) {
  verifyInputShapes(params, in, weights);
  const auto numBatchGroups = in.dim(0);
  Sequence prog;
  const auto dType = in.elementType();
  const auto outNumChans = weights.dim(0) * weights.dim(4);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  assert(outNumChans % partialChansPerGroup == 0);
  const auto partialNumChanGroups = outNumChans / partialChansPerGroup;
  const auto tilesPerInZGroup = plan.tilesPerInZGroupAxis;
  const auto tilesPerKernelY = plan.tilesPerKernelYAxis;

  const auto partialType = plan.getPartialType();

  // Calculate a set of partial sums of the convolutions.
  Tensor partials = graph.addTensor(partialType,
                                     {tilesPerInZGroup * tilesPerKernelY,
                                      numBatchGroups,
                                      partialNumChanGroups,
                                      params.getOutputHeight(),
                                      params.getOutputWidth(),
                                      partialChansPerGroup},
                                    "partials");
  prog.add(calcPartialSums(graph, plan, params, dType, in, weights, partials,
                           debugPrefix));

  std::vector<ComputeSet> reduceComputeSets;
  // For each element of the batch, we add the reduction vertices to same
  // compute sets so the batch will be executed in parallel.
  Tensor reduced;
  // Perform the reduction of partial sums.
  if (partials.dim(0) == 1) {
    if (dType != partialType) {
      reduced = graph.addTensor(dType, partials.shape(), "reduced");
      if (reduceComputeSets.empty()) {
        reduceComputeSets.push_back(graph.addComputeSet(debugPrefix +
                                                           "/Cast"));
      }
      applyTensorMapping(graph, reduced, graph.getTileMapping(partials));
      cast(graph, partials, reduced, reduceComputeSets[0]);
    } else {
      reduced = partials;
    }
    reduced = reduced[0];
  } else {
    reduced = multiStageGroupedReduce(graph, partials, dType, reduceComputeSets,
                                      debugPrefix);
  }
  for (const auto &cs : reduceComputeSets) {
    prog.add(Execute(cs));
  }
  return {prog, reduced};
}

template <typename T>
static std::string
getShapeAsString(const std::vector<T> &shape) {
  return shape.empty() ? std::string ()
    : std::accumulate (std::next(shape.begin()), shape.end (),
                       std::to_string(shape[0]),
                       [] (std::string a, unsigned b) {
                         return a + "x" + std::to_string(b);
                       });
}

static std::string
convSuffix(const ConvParams &params) {
  std::string s = "_";
  s += getShapeAsString(params.kernelShape);
  if (std::any_of(params.stride.begin(), params.stride.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_stride" + getShapeAsString(params.stride);
  }
  if (std::any_of(params.inputDilation.begin(), params.inputDilation.end(),
                  [](unsigned x) { return x != 1; })) {
    s += "_inDilation" + getShapeAsString(params.inputDilation);
  }
  return s;
}

Tensor
convolution(Graph &graph, const poplar::Tensor &in_,
            const poplar::Tensor &weights_,
            const ConvParams &params_,
            bool transposeAndFlipWeights, Sequence &prog,
            const std::string &debugPrefix, const ConvOptions &options) {
  auto in = groupActivations(in_);
  auto weights = weights_;
  auto params = params_;
  verifyStrideAndPaddingDimensions(params);
  const auto dType = in.elementType();
  const auto batchSize = in.dim(0);
  auto plan = getPlan(graph, params, options);
  if (transposeAndFlipWeights) {
    // Create transposed/flipped weights
    auto bwdWeights = createWeights(graph, params, "bwdWeights", options);
    weightsTransposeChansFlipXY(graph, weights, bwdWeights, prog, debugPrefix);
    weights = bwdWeights;
  }
  auto wInChansPerGroup =
      weights.dim(3) % plan.inChansPerGroup == 0 ? plan.inChansPerGroup : 1;
  auto wOutChansPerGroup = weights.dim(2) % plan.partialChansPerGroup == 0
                               ? plan.partialChansPerGroup
                               : 1;
  weights = groupWeights(weights, wInChansPerGroup, wOutChansPerGroup);
  const auto outputShape = getOutputShape(params);
  convolutionPreprocess(graph, params, plan, &in, &weights);
  Tensor activations;
  if (plan.useWinograd) {
    const auto wgOutputShape = getOutputShape(params);
    activations =
        graph.addTensor(dType, {wgOutputShape[0],
                                wgOutputShape[3] / plan.partialChansPerGroup,
                                wgOutputShape[1], wgOutputShape[2],
                                plan.partialChansPerGroup});
    ::mapActivations(graph, activations);
    prog.add(winogradConvolution(graph, params, in, weights, activations,
                                 plan.winogradPatchSize, plan.winogradPatchSize,
                                 plan.floatPartials ? "float" : "half",
                                 debugPrefix, options));
  } else {
    const auto layerName = debugPrefix + "/Conv" + convSuffix(params);
    Program convolveProg;
    std::tie(convolveProg, activations) =
      convolutionByAmp(graph, plan, params, in, weights, layerName);
    prog.add(convolveProg);
  }
  activations = activations.dimShuffle({0, 2, 3, 1, 4})
        .reshape({batchSize,
                  outputShape[1],
                  outputShape[2],
                  activations.dim(1),
                  activations.dim(4)})
        .dimShuffle({0, 3, 1, 2, 4});
  const auto outNumChans = outputShape[3];
  const auto partialNumChans = activations.dim(1) * activations.dim(4);
  if (partialNumChans != outNumChans) {
    // Truncate the activations in the channel axis.
    auto activationsRegrouped = regroup(activations, partialNumChans);
    auto activationsRegroupedTruncated =
        activationsRegrouped.slice(0, outNumChans, 4);
    const auto outChansPerGroup = getOutChansPerGroup(plan, outNumChans);
    assert(outNumChans % outChansPerGroup == 0);
    activations = regroup(activationsRegroupedTruncated, outChansPerGroup);
  }
  // Rearrange the activations so the tile mapping matches the tile mapping
  // returned by computeActivationsMapping().
  // TODO remove once the rest of the code has been updated to make no
  // assumptions about the tile mapping of activations.
  Tensor activationsRemapped = graph.addTensor(dType, activations.shape(),
                                               "activationsRemapped");
  ::mapActivations(graph, activationsRemapped);
  prog.add(Copy(activations, activationsRemapped));
  return ungroupActivations(activationsRemapped);
}

static std::uint64_t getNumberOfMACs(const ConvParams &params) {
  std::uint64_t numMACs = 0;
  auto outputShape = getOutputShape(params);
  for (unsigned y = 0; y < outputShape[1]; ++y) {
    unsigned inYBegin, inYEnd;
    std::tie(inYBegin, inYEnd) = getInputRange(0, y, params);
    const auto height = inYEnd - inYBegin;
    for (unsigned x = 0; x < outputShape[2]; ++x) {
      unsigned inXBegin, inXEnd;
      std::tie(inXBegin, inXEnd) = getInputRange(1, x, params);
      const auto width = inXEnd - inXBegin;
      numMACs += width * height * outputShape[3] * params.inputShape[3];
    }
  }
  const auto batchSize = outputShape[0];
  return batchSize * numMACs;
}

static uint64_t getFlops(const ConvParams &params) {
  verifyStrideAndPaddingDimensions(params);
  return (2 * getNumberOfMACs(params));
}

uint64_t getFwdFlops(const ConvParams &params) {
  return getFlops(params);
}

uint64_t getBwdFlops(const ConvParams &params) {
  return getFlops(params);
}

uint64_t getWuFlops(const ConvParams &params) {
  return getFlops(params);
}

static double getPerfectCycleCount(const Graph &graph,
                                   const ConvParams &params) {
  verifyStrideAndPaddingDimensions(params);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto numTiles = deviceInfo.getNumTiles();
  auto numMacs = getNumberOfMACs(params);
  if (params.dType == "float") {
    const auto floatVectorWidth = deviceInfo.getFloatVectorWidth();
    auto macCycles =
        static_cast<double>(numMacs) / (floatVectorWidth * numTiles);
    return macCycles;
  }
  assert(params.dType == "half");
  const auto convUnitsPerTile =
      std::max(std::max(deviceInfo.fp16InFp16OutConvUnitsPerTile,
                        deviceInfo.fp32InFp32OutConvUnitsPerTile),
               deviceInfo.fp16InFp32OutConvUnitsPerTile);
  const auto halfVectorWidth = deviceInfo.getHalfVectorWidth();
  auto macsPerCycle = convUnitsPerTile * halfVectorWidth;
  auto macCycles = static_cast<double>(numMacs) / (macsPerCycle * numTiles);
  return macCycles;
}

double getFwdPerfectCycleCount(const Graph &graph,
                               const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

double getBwdPerfectCycleCount(const Graph &graph,
                               const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

double getWuPerfectCycleCount(const Graph &graph, const ConvParams &params) {
  return getPerfectCycleCount(graph, params);
}

/**
 * Transpose the innermost pair of dimensions of the specified tensor, writing
 * the results to a new tensor.
 */
static Tensor weightsPartialTranspose(Graph &graph, Tensor in, ComputeSet cs) {
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto rank = in.rank();
  const auto numSrcRows = in.dim(rank - 2);
  const auto numSrcColumns = in.dim(rank - 1);
  const auto dType = in.elementType();
  auto outShape = in.shape();
  std::swap(outShape[rank - 2], outShape[rank - 1]);
  auto out = graph.addTensor(dType, outShape, "partialTranspose");
  auto inFlat = in.reshape({in.numElements() / (numSrcRows * numSrcColumns),
                            numSrcRows * numSrcColumns});
  auto outFlat = out.reshape(inFlat.shape());
  const auto transpositionMapping =
      graph.getTileMapping(inFlat.slice(0, 1, 1));
  const auto numTiles = transpositionMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerTranspositions =
        splitRegionsBetweenWorkers(deviceInfo, transpositionMapping[tile], 1);
    for (const auto &entry : perWorkerTranspositions) {
      const auto v =
          graph.addVertex(cs, templateVertex("popconv::Transpose2D", dType));
      graph.setInitialValue(v["numSrcColumns"],
                            static_cast<unsigned>(numSrcColumns));
      graph.setTileMapping(v, tile);
      unsigned i = 0;
      for (const auto &interval : entry) {
        for (auto transposition = interval.begin();
             transposition != interval.end(); ++transposition) {
          graph.connect(v["src"][i], inFlat[transposition]);
          graph.connect(v["dst"][i], outFlat[transposition]);
          graph.setTileMapping(outFlat[transposition], tile);
          ++i;
        }
      }
      graph.setFieldSize(v["src"], i);
      graph.setFieldSize(v["dst"], i);
    }
  }
  return out;
}

/** Copy the weights in 'weightsIn' into 'weightsOut' such that
 *  each element of the kernel is transposed w.r.t. the input and output
 *  channels and flip both the X and Y axis of the kernel field.
 */
void weightsTransposeChansFlipXY(Graph &graph,
                                 const Tensor &weightsInUnGrouped,
                                 const Tensor &weightsOutUnGrouped,
                                 Sequence &prog,
                                 const std::string &debugPrefix) {
  const auto weightsIn = groupWeights(weightsInUnGrouped);
  const auto weightsOut = groupWeights(weightsOutUnGrouped);
  // weightsIn = { O/G1, I/G2, KY, KX, G1, G2 }
  // weightsOut = { I/G3, O/G4, KY, KX, G3, G4 }

  const auto dType = weightsIn.elementType();
  const auto KY = weightsOut.dim(2);
  const auto KX = weightsOut.dim(3);
  const auto I = weightsOut.dim(0) * weightsOut.dim(4);
  const auto O = weightsOut.dim(1) * weightsOut.dim(5);
  const auto G1 = weightsIn.dim(4);
  const auto G2 = weightsIn.dim(5);
  const auto G3 = weightsOut.dim(4);
  const auto G4 = weightsOut.dim(5);

  // Express the rearrangement as a composition of two rearrangements such
  // that the first rearrangement avoids exchange and maximises the size of the
  // block that is rearranged in the second step. This reduces exchange code
  // since the second step involves fewer, larger messages.
  // G5 is the size of the innermost dimension after the partial transposition.
  // To avoid exchange it must divide G1. If G4 divides G1 then set G5 to G4 -
  // this results in the block size of G1 * gcd(G2, G3) elements in the
  // second step. Otherwise set G5 to G1 for a block size of gcd(G1, G4)
  // elements.
  const auto G5 = (G1 % G4 == 0) ? G4 : G1;
  Tensor partiallyTransposed;
  if (G5 == 1) {
    partiallyTransposed = weightsIn.reshape({O/G1, I/G2, KY, KX, G1, G2, 1});
  } else {
    auto cs = graph.addComputeSet(debugPrefix + "/WeightTranspose");
    partiallyTransposed =
        weightsPartialTranspose(
          graph,
          weightsIn.reshape({O/G1, I/G2, KY, KX, G1/G5, G5, G2}),
          cs
        );
    prog.add(Execute(cs));
  }

  auto wFlippedY = graph.addTensor(dType, {O/G1, I/G2, 0, KX, G1/G5, G2, G5});
  for (int wy = KY - 1; wy >= 0; --wy) {
     wFlippedY = concat(wFlippedY,
                        partiallyTransposed.slice(wy, wy + 1, 2), 2);
  }

  auto wFlippedYX= graph.addTensor(dType, {O/G1, I/G2, KY, 0, G1/G5, G2, G5});
  for (int wx = KX - 1; wx >= 0; --wx) {
     wFlippedYX = concat(wFlippedYX,
                         wFlippedY.slice(wx, wx + 1, 3), 3);
  }
  prog.add(Copy(wFlippedYX.dimShuffle({2, 3, 0, 4, 6, 1, 5})
                           .reshape({KY, KX, O/G4, G4, I/G3, G3})
                           .dimShuffle({4, 2, 0, 1, 5, 3}),
                weightsOut));
}

static void
createWeightGradAopVertex(Graph &graph, unsigned tile,
                          unsigned outXBegin, unsigned outXEnd,
                          unsigned outYBegin, unsigned outYEnd,
                          const WeightGradAopTask *taskBegin,
                          const WeightGradAopTask *taskEnd,
                          unsigned kernelSizeY, unsigned kernelSizeX,
                          const ConvParams &params,
                          ComputeSet cs,
                          const Tensor &acts, const Tensor &deltas,
                          const Tensor &weightDeltas) {
  const auto dType = acts.elementType();;
  const auto partialsType = weightDeltas.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto outChansPerGroup = static_cast<unsigned>(deltas.dim(3));
  const auto inChansPerGroup = static_cast<unsigned>(acts.dim(3));
  assert(weightDeltas.dim(4) == outChansPerGroup);
  assert(weightDeltas.dim(5) == inChansPerGroup);

  const auto numAopAccumulators = dType == "float" ?
                                  deviceInfo.fp32NumAopAccumulators :
                                  deviceInfo.fp16NumAopAccumulators;

  const auto numTasks = taskEnd - taskBegin;

  if (!numTasks)
    return;

  std::vector<Tensor> weightDeltasEdges(numTasks);
  std::vector<unsigned> weightReuseCount(numTasks);
  std::vector<Tensor> actsEdges;
  std::vector<Tensor> deltasEdges;
  unsigned numDeltasEdges = 0;
  for (auto it = taskBegin; it != taskEnd; ++it) {
    const auto &task = *it;
    const auto kernelX = task.kernelX;
    const auto kernelY = task.kernelY;
    const auto izg = task.inZGroup;
    const auto ozg = task.outZGroup;
    const auto weightIndex = it - taskBegin;

    weightDeltasEdges[weightIndex] = weightDeltas[ozg][izg]
                                                 [kernelY][kernelX].flatten();

    unsigned deltaXBegin, deltaXEnd;
    std::tie(deltaXBegin, deltaXEnd) =
        getOutputRange(1, {outXBegin, outXEnd}, kernelX, params);
    const auto actXBegin = getInputIndex(1, deltaXBegin, kernelX, params);
    const auto actXEnd = getInputIndex(1, deltaXEnd - 1, kernelX, params) + 1;
    unsigned deltaYBegin, deltaYEnd;
    std::tie(deltaYBegin, deltaYEnd) =
        getOutputRange(0, {outYBegin, outYEnd}, kernelY, params);

    weightReuseCount[weightIndex] = deltaYEnd - deltaYBegin;

    for (unsigned deltaY = deltaYBegin; deltaY != deltaYEnd;
         ++deltaY, ++numDeltasEdges) {
      const auto actY = getInputIndex(0, deltaY, kernelY, params);
      actsEdges.push_back(acts[izg][actY].slice(actXBegin, actXEnd).flatten());
      deltasEdges.push_back(deltas[ozg][deltaY]
                            .slice(deltaXBegin, deltaXEnd).flatten());
    }
  }

  const auto numEdges = 2 * numDeltasEdges + numTasks;

  auto v = graph.addVertex(
                cs,
                templateVertex("popconv::ConvWeightGradAop",
                               dType, partialsType,
                               useDeltaEdgesForWeightGradAop(numEdges) ?
                                                             "true" : "false"));
  graph.setTileMapping(v, tile);
  graph.setInitialValue(v["inChansPerGroup"], inChansPerGroup);
  graph.setInitialValue(v["outChansPerGroup"], outChansPerGroup);
  graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
  graph.setInitialValue(v["numAopAccumulators"], numAopAccumulators);
  graph.setInitialValue(v["weightReuseCount"], weightReuseCount);
  graph.connect(v["acts"], actsEdges);
  graph.connect(v["deltas"], deltasEdges);
  graph.connect(v["weightDeltas"], weightDeltasEdges);
}

static void
calcPartialWeightGradsAop(Graph &graph,
                          unsigned tile,
                          unsigned outXBegin, unsigned outXEnd,
                          unsigned outYBegin, unsigned outYEnd,
                          unsigned outZGroupBegin, unsigned outZGroupEnd,
                          unsigned kernelYBegin, unsigned kernelYEnd,
                          unsigned inZGroupBegin, unsigned inZGroupEnd,
                          unsigned kernelSizeY, unsigned kernelSizeX,
                          const ConvParams &params,
                          ComputeSet cs,
                          Tensor acts, Tensor deltas, Tensor weightDeltas) {
  std::vector<WeightGradAopTask> tasks;
  for (unsigned kernelY = kernelYBegin; kernelY != kernelYEnd; ++kernelY) {
    for (unsigned kernelX = 0; kernelX != kernelSizeX; ++kernelX) {
      auto xRange =
          getOutputRange(1, {outXBegin, outXEnd}, kernelX, params);
      if (xRange.first == xRange.second)
        continue;
      auto yRange =
          getOutputRange(0, {outYBegin, outYEnd}, kernelY, params);
      if (yRange.first == yRange.second)
        continue;
      for (unsigned ozg = outZGroupBegin; ozg != outZGroupEnd; ++ozg) {
        for (unsigned izg = inZGroupBegin; izg != inZGroupEnd; ++izg) {
          tasks.emplace_back(kernelY, kernelX, ozg, izg);
        }
      }
    }
  }
  if (tasks.empty())
    return;
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  auto numWorkers = deviceInfo.numWorkerContexts;
  const auto numTasks = tasks.size();
  const auto maxTasksPerVertex = (numTasks + numWorkers - 1) / numWorkers;
  const auto verticesToCreate =
      (numTasks + maxTasksPerVertex - 1) / maxTasksPerVertex;
  for (unsigned i = 0; i != verticesToCreate; ++i) {
    const auto taskBegin = (numTasks * i) / verticesToCreate;
    const auto taskEnd = (numTasks * (i + 1)) / verticesToCreate;
    assert(taskEnd - taskBegin > 0);
    createWeightGradAopVertex(graph, tile, outXBegin, outXEnd,
                              outYBegin, outYEnd, &tasks[0] + taskBegin,
                              &tasks[0] + taskEnd, kernelSizeY, kernelSizeX,
                              params, cs, acts,
                              deltas, weightDeltas);
  }
}

static void
addWeightDeltaPartialRegions(
    std::vector<Interval<std::size_t>> &regions,
    const std::vector<std::size_t> &partialDims,
    unsigned b, unsigned tileY, unsigned tileX,
    unsigned kernelYBegin, unsigned kernelYEnd,
    unsigned outZGroupBegin, unsigned outZGroupEnd,
    unsigned inZGroupBegin, unsigned inZGroupEnd) {
  addFlattenedRegions(partialDims,
                      {b,
                       tileY,
                       tileX,
                       outZGroupBegin,
                       inZGroupBegin,
                       kernelYBegin,
                       0,
                       0,
                       0},
                      {b + 1,
                       tileY + 1,
                       tileX + 1,
                       outZGroupEnd,
                       inZGroupEnd,
                       kernelYEnd,
                       partialDims[6],
                       partialDims[7],
                       partialDims[8]},
                      regions);
}

static Tensor
calculateWeightDeltasAop(Graph &graph, Plan plan,
                         const Plan &fwdPlan, Tensor zDeltas,
                         Tensor activations, ConvParams params,
                         Sequence &prog, const std::string &debugPrefix) {
  const auto numInChans = params.getInputDepth();
  const auto numOutChans = params.getOutputDepth();
  if (plan.flattenXY) {
    zDeltas = zDeltas.reshape({zDeltas.dim(0),
                               zDeltas.dim(1),
                               1,
                               zDeltas.dim(2) * zDeltas.dim(3),
                               zDeltas.dim(4)});
  }
  convolutionPreprocess(graph, params, plan, &activations, nullptr);
  const auto weightDeltaOutChans = params.getOutputDepth();
  if (weightDeltaOutChans != numOutChans) {
    auto zDeltasRegrouped = regroup(zDeltas, numOutChans);
    // Zero pad the zDeltas.
    auto zDeltasRegroupedPadded = pad(graph, zDeltasRegrouped, 0,
                                      weightDeltaOutChans - numOutChans, 4);
    zDeltas = regroup(zDeltasRegroupedPadded, plan.partialChansPerGroup);
  }
  const auto weightDeltaInChans = params.getInputDepth();
  const auto &partialsType = plan.getPartialType();
  const auto dType = zDeltas.elementType();
  const auto batchSize = activations.dim(0);
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto tilesPerX = plan.tilesPerXAxis;
  const auto tilesPerY = plan.tilesPerYAxis;
  const auto numTiles = deviceInfo.getNumTiles();
  const auto kernelSizeY = params.kernelShape[0];
  const auto kernelSizeX = params.kernelShape[1];
  assert(weightDeltaOutChans % plan.partialChansPerGroup == 0);
  const auto numPartialOutChanGroups =
      weightDeltaOutChans / plan.partialChansPerGroup;
  const auto numPartialInChanGroups =
      weightDeltaInChans / plan.inChansPerGroup;
  assert(weightDeltaInChans % plan.inChansPerGroup == 0);
  Tensor partials = graph.addTensor(partialsType, {batchSize,
                                                   tilesPerY, tilesPerX,
                                                   numPartialOutChanGroups,
                                                   numPartialInChanGroups,
                                                   kernelSizeY, kernelSizeX,
                                                   plan.partialChansPerGroup,
                                                   plan.inChansPerGroup},
                                    "partialWeightGrads");
  Tensor regroupedDeltas;
  if (zDeltas.dim(4) != plan.partialChansPerGroup) {
    regroupedDeltas = graph.addTensor(dType, {zDeltas.dim(0),
                                              numPartialOutChanGroups,
                                              zDeltas.dim(2),
                                              zDeltas.dim(3),
                                              plan.partialChansPerGroup},
                                              "zDeltas'");
    for (unsigned b = 0; b < batchSize; ++b) {
      auto regroupedDeltaMapping =
          computeActivationsMapping(graph, regroupedDeltas[b], b, batchSize);
      popstd::applyTensorMapping(graph, regroupedDeltas[b],
                                 regroupedDeltaMapping);
    }
    prog.add(Copy(regroup(zDeltas, plan.partialChansPerGroup),
                  regroupedDeltas));
  } else {
    regroupedDeltas = zDeltas;
  }
  std::vector<std::vector<Interval<std::size_t>>> partialsMapping(numTiles);
  ComputeSet weightGradCS = graph.addComputeSet(debugPrefix + "/WeightGrad");


  iterateTilePartition(graph, params, plan,
                       [&](unsigned tile, const ConvTileIndices &indices,
                           const ConvSlice &slice) {
    if (slice.outZGroupBegin == slice.outZGroupEnd)
      return;
    assert(slice.batchEnd - slice.batchBegin == 1);
    unsigned b = slice.batchBegin;
    addWeightDeltaPartialRegions(partialsMapping[tile],
                                 partials.shape(), b, indices.oy, indices.ox,
                                 slice.kernelYBegin, slice.kernelYEnd,
                                 slice.outZGroupBegin, slice.outZGroupEnd,
                                 slice.inZGroupBegin, slice.inZGroupEnd);
    calcPartialWeightGradsAop(graph, tile,
                              slice.outXBegin, slice.outXEnd,
                              slice.outYBegin, slice.outYEnd,
                              slice.outZGroupBegin, slice.outZGroupEnd,
                              slice.kernelYBegin, slice.kernelYEnd,
                              slice.inZGroupBegin, slice.inZGroupEnd,
                              kernelSizeY, kernelSizeX,
                              params,
                              weightGradCS,
                              activations[b],
                              regroupedDeltas[b],
                              partials[indices.b][indices.oy][indices.ox]);
  });
  mergeAdjacentRegions(partialsMapping);
  applyTensorMapping(graph, partials, partialsMapping);
  ComputeSet zeroCS = graph.addComputeSet(debugPrefix + "/Zero");
  zero(graph, partials, partialsMapping, zeroCS);
  prog.add(Execute(zeroCS));
  prog.add(Execute(weightGradCS));
  Tensor weightDeltas;
  auto numPartials = batchSize * tilesPerY * tilesPerX;
  if (numPartials == 1) {
    if (partialsType == dType) {
      weightDeltas = partials[0][0][0];
    } else {
      weightDeltas = popstd::cast(graph, partials[0][0][0], dType, prog,
                                  debugPrefix + "/cast");
    }
  } else {
    std::vector<ComputeSet> reduceComputeSets;
    auto toReduce = partials.reshape({numPartials, partials.dim(3),
                                      partials.dim(4), partials.dim(5),
                                      partials.dim(6), partials.dim(7),
                                      partials.dim(8)});
    weightDeltas = multiStageGroupedReduce(graph, toReduce, dType,
                                           reduceComputeSets, debugPrefix);
    for (const auto &cs : reduceComputeSets) {
      prog.add(Execute(cs));
    }
  }
  if (weightDeltaOutChans != numOutChans) {
    // Truncate the weight deltas in the out channel axis.
    auto weightDeltasRegrouped = regroup(weightDeltas, 0, 4,
                                         weightDeltaOutChans);
    weightDeltas = weightDeltasRegrouped.slice(0, numOutChans, 4);
  }
  const auto fwdWeightOutChansPerGroup =
      getWeightOutChansPerGroup(fwdPlan, numOutChans);
  weightDeltas = regroup(weightDeltas, 0, 4, fwdWeightOutChansPerGroup);
  if (weightDeltaInChans != numInChans) {
    // Truncate the weight deltas in the in channel axis.
    auto weightDeltasRegrouped = regroup(weightDeltas, 1, 5,
                                         weightDeltaInChans);
    weightDeltas = weightDeltasRegrouped.slice(0, numInChans, 5);
  }
  const auto fwdWeightInChansPerGroup =
      getWeightInChansPerGroup(fwdPlan, numInChans);
  weightDeltas = regroup(weightDeltas, 1, 5, fwdWeightInChansPerGroup);
  return weightDeltas;
}

static Tensor
roundUpDimension(Graph &graph, const Tensor &t, unsigned dim,
                 unsigned divisor) {
  const auto size = t.dim(dim);
  const auto roundedSize = ((size + divisor - 1) / divisor) * divisor;
  return pad(graph, t, 0, roundedSize - size, dim);
}

// Weight deltas can be computed by convolving the activations and the deltas.
// If the kernel is larger than 1x1 a direct computation of weight deltas
// requires a sliding deltas across activations in the main axis of
// accumulation. This sliding stops us using the AMP instruction because,
// as we slide, the vectors of elements we want to load in the inner loop will
// no longer be contiguous / aligned. We fix this by transforming the
// convolution into an equivalent convolution which doesn't require sliding in
// the main axis of accumulation. Given the activation and delta tensors for
// the original convolution (zero padded and reshaped to have a single channel
// dimension) transform them into tensors for the transformed convolution.
static void
convolutionWeightUpdateAmpPreProcess(
    Graph &graph,
    const Plan &plan,
    Tensor &activations,
    std::vector<unsigned> &activationsUpsampleFactor,
    std::vector<int> &activationsPaddingLower,
    std::vector<int> &activationsPaddingUpper,
    Tensor &deltas,
    std::vector<unsigned> &deltasUpsampleFactor,
    std::vector<int> &deltasPaddingLower,
    std::vector<int> &deltasPaddingUpper,
    std::vector<int> &weightDeltaTruncateLower,
    std::vector<int> &weightDeltaTruncateUpper,
    std::vector<unsigned> &weightDeltasStride,
    const std::vector<std::size_t> &kernelShape) {
  const auto dType = activations.elementType();
  assert(activationsUpsampleFactor.size() == 2);
  assert(activationsPaddingLower.size() == 2);
  assert(activationsPaddingUpper.size() == 2);
  assert(deltasUpsampleFactor.size() == 2);
  assert(deltasPaddingLower.size() == 2);
  assert(deltasPaddingUpper.size() == 2);
  assert(activationsUpsampleFactor == std::vector<unsigned>({1, 1}));
  assert(deltasPaddingLower == std::vector<int>({0, 0}));
  assert(deltasPaddingUpper == std::vector<int>({0, 0}));
  assert(plan.flattenXY ||
         (weightDeltaTruncateLower[0] == 0 &&
          weightDeltaTruncateUpper[0] == 0));
  // Eliminate the x axis of the kernel by taking the activations that are
  // multiplied by each column of the weights turning them into different input
  // channels.
  auto paddedActivations = pad(graph, activations, activationsPaddingLower[1],
                               activationsPaddingUpper[1], 2);
  activationsPaddingLower[1] = 0;
  activationsPaddingUpper[1] = 0;
  auto expandedActivations =
      graph.addTensor(dType, {paddedActivations.dim(0),
                              paddedActivations.dim(1),
                              deltas.dim(2),
                              0});
  std::vector<unsigned> paddedKernelShape(2);
  for (unsigned dim = 0; dim != 2; ++dim) {
    paddedKernelShape[dim] = (kernelShape[dim] - 1) * weightDeltasStride[dim]
                             + 1 + weightDeltaTruncateLower[dim] +
                             weightDeltaTruncateUpper[dim];
  }
  for (unsigned wx = 0; wx != kernelShape[1]; ++wx) {
    auto dilatedPaddedWeightX =
        static_cast<int>(wx * weightDeltasStride[1]) +
        weightDeltaTruncateLower[1];
    Tensor usedActivations;
    if (dilatedPaddedWeightX >= 0 &&
        static_cast<unsigned>(dilatedPaddedWeightX) < paddedKernelShape[1]) {
      usedActivations =
          paddedActivations.slice(dilatedPaddedWeightX,
                                  dilatedPaddedWeightX +
                                  (deltas.dim(2) - 1) *
                                  deltasUpsampleFactor[1] + 1,
                                  2);
      usedActivations =
          usedActivations.subSample(deltasUpsampleFactor[1], 2);
    } else {
      usedActivations =
          graph.addConstantTensor(dType, {paddedActivations.dim(0),
                                          paddedActivations.dim(1),
                                          deltas.dim(2),
                                          paddedActivations.dim(3)}, 0);
    }
    expandedActivations = concat(expandedActivations, usedActivations, 3);
  }
  deltasUpsampleFactor[1] = 1;
  weightDeltasStride[1] = 1;
  weightDeltaTruncateLower[1] = 0;
  weightDeltaTruncateUpper[1] = 0;
  if (plan.flattenXY) {
    // Eliminate the y axis of the kernel by taking the activations that are
    // multiplied by each row of the weights turning them into different input
    // channels.
    auto yPaddedActivations = pad(graph, expandedActivations,
                                  activationsPaddingLower[0],
                                  activationsPaddingUpper[0], 1);
    activationsPaddingLower[0] = 0;
    activationsPaddingUpper[0] = 0;
    auto yExpandedActivations =
        graph.addTensor(dType, {yPaddedActivations.dim(0),
                                deltas.dim(1),
                                deltas.dim(2),
                                0});
    for (unsigned wy = 0; wy != kernelShape[0]; ++wy) {
      auto dilatedPaddedWeightY =
          static_cast<int>(wy * weightDeltasStride[0]) +
          weightDeltaTruncateLower[0];
      Tensor usedActivations;
      if (dilatedPaddedWeightY >= 0 &&
          static_cast<unsigned>(dilatedPaddedWeightY) < paddedKernelShape[0]) {
        usedActivations =
            yPaddedActivations.slice(dilatedPaddedWeightY,
                                     dilatedPaddedWeightY +
                                     (deltas.dim(1) - 1) *
                                     deltasUpsampleFactor[0] + 1,
                                     1);
        usedActivations =
            usedActivations.subSample(deltasUpsampleFactor[0], 1);
      } else {
        usedActivations =
            graph.addConstantTensor(dType, {yPaddedActivations.dim(0),
                                            yPaddedActivations.dim(1),
                                            deltas.dim(2),
                                            yPaddedActivations.dim(3)}, 0);
      }
      yExpandedActivations = concat(yExpandedActivations, usedActivations,
                                    3);
    }
    expandedActivations = yExpandedActivations;
    deltasUpsampleFactor[0] = 1;
    weightDeltasStride[0] = 1;
    weightDeltaTruncateLower[0] = 0;
    weightDeltaTruncateUpper[0] = 0;
    // Flatten the x and y axes.
    expandedActivations =
        expandedActivations.reshape({expandedActivations.dim(0),
                                     1,
                                     expandedActivations.dim(1) *
                                     expandedActivations.dim(2),
                                     expandedActivations.dim(3)});
    deltas =
        deltas.reshape({deltas.dim(0),
                        1,
                        deltas.dim(1) * deltas.dim(2),
                        deltas.dim(3)});
  }
  // Rearrange the tensors so elements of the batch are treated as part of the
  // x-axis of the field.
  auto flattenedActivations =
      expandedActivations
          .dimShuffle({1, 2, 0, 3})
          .reshape({1,
                    expandedActivations.dim(1),
                    expandedActivations.dim(2) * expandedActivations.dim(0),
                    expandedActivations.dim(3)});
  auto flattenedDeltas =
      deltas.dimShuffle({1, 2, 0, 3})
             .reshape({1,
                       deltas.dim(1),
                       deltas.dim(2) * deltas.dim(0),
                       deltas.dim(3)});
  if (plan.ampWUMethod == Plan::ACTIVATIONS_AS_COEFFICENTS) {
    assert(activationsPaddingLower[1] == 0);
    assert(activationsPaddingUpper[1] == 0);
    std::swap(flattenedActivations, flattenedDeltas);
    std::swap(activationsUpsampleFactor, deltasUpsampleFactor);
    std::swap(activationsPaddingLower, deltasPaddingLower);
    std::swap(activationsPaddingUpper, deltasPaddingUpper);
  }
  activations = flattenedActivations;
  deltas = flattenedDeltas;
}

// convolutionWeightUpdateAmpPreProcess() translates a convolution into an
// equivalent convolution that we can use the AMP instruction to compute.
// Given the weight deltas for this transformed convolution (reshaped so neither
// channel dimension is split) transform them into weight deltas for the
// original convolution.
static void
convolutionWeightUpdateAmpPostProcess(const Plan &plan,
                                      Tensor &weightDeltas,
                                      unsigned kernelSizeY,
                                      unsigned kernelSizeX) {
  assert(weightDeltas.dim(1) == 1);
  if (plan.ampWUMethod == Plan::ACTIVATIONS_AS_COEFFICENTS) {
    weightDeltas = weightDeltas.dimShuffle({0, 1, 3, 2});
  }
  if (plan.flattenXY) {
    assert(weightDeltas.dim(0) == 1);
    weightDeltas =
        weightDeltas.reshape({weightDeltas.dim(2),
                              kernelSizeY,
                              kernelSizeX,
                              weightDeltas.dim(3) /
                              (kernelSizeY * kernelSizeX)})
                              .dimShuffle({1, 2, 0, 3});
  } else {
    weightDeltas =
        weightDeltas.reshape({weightDeltas.dim(0),
                              weightDeltas.dim(2),
                              kernelSizeX,
                              weightDeltas.dim(3) / kernelSizeX})
                              .dimShuffle({0, 2, 1, 3});
  }
}

static Tensor
calculateWeightDeltasAmp(Graph &graph, const Plan &plan,
                         const Plan &fwdPlan,
                         Tensor zDeltas,
                         Tensor activations,
                         const ConvParams &params,
                         Sequence &prog,
                         const std::string &debugPrefix) {
  // Shuffle dimensions of the activations and the deltas so there is a single
  // channel dimension.
  auto activationsView =
      activations.dimShuffle({0, 2, 3, 1, 4})
                 .reshape({activations.dim(0),
                           activations.dim(2),
                           activations.dim(3),
                           activations.dim(1) * activations.dim(4)});
  unsigned numInChans = activationsView.dim(3);
  auto deltasView =
      zDeltas.dimShuffle({0, 2, 3, 1, 4})
             .reshape({zDeltas.dim(0),
                       zDeltas.dim(2),
                       zDeltas.dim(3),
                       zDeltas.dim(1) * zDeltas.dim(4)});
  unsigned numOutChans = deltasView.dim(3);

  // Transform the weight update convolution into an equivalent convolution that
  // can be implemented using the AMP instruction.
  std::vector<unsigned> activationsUpsampleFactor = {1, 1};
  std::vector<int> activationsPaddingLower = params.inputPaddingLower;
  std::vector<int> activationsPaddingUpper = params.inputPaddingUpper;
  std::vector<unsigned> deltasUpsampleFactor = params.stride;
  std::vector<int> deltasPaddingLower = {0, 0};
  std::vector<int> deltasPaddingUpper = {0, 0};
  std::vector<int> weightDeltasTruncateLower = params.kernelPaddingLower;
  std::vector<int> weightDeltasTruncateUpper = params.kernelPaddingUpper;
  std::vector<unsigned> weightDeltasStride = params.kernelDilation;
  convolutionWeightUpdateAmpPreProcess(graph, plan, activationsView,
                                       activationsUpsampleFactor,
                                       activationsPaddingLower,
                                       activationsPaddingUpper, deltasView,
                                       deltasUpsampleFactor,
                                       deltasPaddingLower, deltasPaddingUpper,
                                       weightDeltasTruncateLower,
                                       weightDeltasTruncateUpper,
                                       weightDeltasStride,
                                       params.kernelShape);
  assert(weightDeltasTruncateLower == std::vector<int>({0, 0}));
  assert(weightDeltasTruncateUpper == std::vector<int>({0, 0}));

  const auto dType = activations.elementType();
  // Reshape so there is no batch dimension.
  assert(activationsView.dim(0) == 1);
  assert(deltasView.dim(0) == 1);
  activationsView = activationsView[0];
  deltasView = deltasView[0];

  assert(activationsView.dim(1) == deltasView.dim(1));

  // Pad the x-axis to a multiple of the input channels per group.
  assert(activationsPaddingLower[1] == 0);
  assert(activationsPaddingUpper[1] == 0);
  assert(activationsUpsampleFactor[1] == 1);
  assert(deltasPaddingLower[1] == 0);
  assert(deltasPaddingUpper[1] == 0);
  assert(deltasUpsampleFactor[1] == 1);
  const auto inChansPerGroup = plan.inChansPerGroup;
  activationsView = roundUpDimension(graph, activationsView, 1,
                                     inChansPerGroup);
  deltasView = roundUpDimension(graph, deltasView, 1, inChansPerGroup);
  // Pad the output channels to a multiple of the partial channels per group.
  const unsigned convOutChans = deltasView.dim(2);
  const auto partialChansPerGroup = plan.partialChansPerGroup;
  deltasView = roundUpDimension(graph, deltasView, 2, partialChansPerGroup);

  // Transpose the activations.
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  auto transformedParams = weightUpdateByAmpTransformParams(params,
                                                            deviceInfo,
                                                            plan);
  auto activationsTransposed =
      createInput(graph, transformedParams, "activationsTransposed", plan);
  prog.add(Copy(activationsView.reshape({activationsView.dim(0),
                                         activationsView.dim(1) /
                                         inChansPerGroup,
                                         inChansPerGroup,
                                         activationsView.dim(2)})
                           .dimShuffle({1, 0, 3, 2})
                           .reshape(activationsTransposed.shape()),
                activationsTransposed));
  // Transpose the deltas.
  auto deltasTransposed =
      groupWeights(createWeights(graph, transformedParams, "deltasTransposed",
                                 plan),
                   plan.inChansPerGroup,
                   plan.partialChansPerGroup);
  prog.add(Copy(deltasView.reshape({deltasView.dim(0),
                                    deltasView.dim(1) / inChansPerGroup,
                                    inChansPerGroup,
                                    deltasView.dim(2) / partialChansPerGroup,
                                    partialChansPerGroup})
                           .dimShuffle({3, 1, 0, 4, 2})
                           .reshape(deltasTransposed.shape()),
                deltasTransposed));

  // Perform the convolution.
  Tensor weightDeltasTransposed;
  Program convolveProg;
  std::tie(convolveProg, weightDeltasTransposed) =
      convolutionByAmp(graph, plan, transformedParams,
                       activationsTransposed,
                       deltasTransposed, debugPrefix);
  prog.add(convolveProg);

  // Shuffle dimensions so the output channel dimension is not split.
  auto weightDeltas =
      weightDeltasTransposed.dimShuffle({0, 2, 3, 1, 4})
                            .reshape({weightDeltasTransposed.dim(0),
                                      weightDeltasTransposed.dim(2),
                                      weightDeltasTransposed.dim(3),
                                      weightDeltasTransposed.dim(1) *
                                      weightDeltasTransposed.dim(4)
                                     });
  // Ignore output channels added for padding.
  weightDeltas = weightDeltas.slice(0, convOutChans, 3);
  // Reshape so there is no batch dimension.
  assert(weightDeltas.dim(0) == 1);
  weightDeltas = weightDeltas[0];

  // Make the input channel dimension the innermost dimension and add an
  // x-axis.
  weightDeltas =
      weightDeltas.dimShuffle({0, 2, 1})
                  .reshape({weightDeltas.dim(0),
                            1,
                            weightDeltas.dim(2),
                            weightDeltas.dim(1)});

  // Transform the weight deltas back into weight deltas for the original
  // weight update convolution.
  convolutionWeightUpdateAmpPostProcess(plan, weightDeltas,
                                        params.kernelShape[0],
                                        params.kernelShape[1]);
  // Split the input / output channel axes.
  const auto weightOutChansPerGroup =
      getWeightOutChansPerGroup(fwdPlan, numOutChans);
  assert(numOutChans % weightOutChansPerGroup == 0);
  const auto weightInChansPerGroup =
      getWeightInChansPerGroup(fwdPlan, numInChans);
  assert(numInChans % weightInChansPerGroup == 0);
  weightDeltas =
      weightDeltas.reshape({weightDeltas.dim(0),
                            weightDeltas.dim(1),
                            weightDeltas.dim(2) /
                            weightOutChansPerGroup,
                            weightOutChansPerGroup,
                            weightDeltas.dim(3) / weightInChansPerGroup,
                            weightInChansPerGroup})
                       .dimShuffle({2, 4, 0, 1, 3, 5});
  return weightDeltas;
}

static Tensor
calculateWeightDeltas(Graph &graph, const Plan &plan,
                      const Plan &fwdPlan, const Tensor &zDeltas,
                      const Tensor &activations,
                      const ConvParams &params,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  if (plan.useConvolutionInstructions) {
    return calculateWeightDeltasAmp(graph, plan, fwdPlan, zDeltas,
                                    activations, params, prog, debugPrefix);
  }
  return calculateWeightDeltasAop(graph, plan, fwdPlan, zDeltas, activations,
                                  params, prog, debugPrefix);
}

Tensor
calculateWeightDeltas(Graph &graph, const Tensor &zDeltasUnGrouped,
                      const Tensor &activationsUnGrouped,
                      const ConvParams &params,
                      Sequence &prog,
                      const std::string &debugPrefix,
                      const ConvOptions &options) {
  auto activations = groupActivations(activationsUnGrouped);
  auto zDeltas = groupActivations(zDeltasUnGrouped);
  const auto plan =
      getWeightUpdatePlan(graph, activations, zDeltas, params, options);
  const auto fwdPlan = getPlan(graph, params, options);
  auto w =  calculateWeightDeltas(graph, plan, fwdPlan, zDeltas, activations,
                                  params, prog, debugPrefix);
  return ungroupWeights(w);
}

void
convolutionWeightUpdate(Graph &graph,
                        const Tensor &zDeltas, const Tensor &weights,
                        const Tensor &activations,
                        const ConvParams &params,
                        float learningRate,
                        Sequence &prog,
                        const std::string &debugPrefix,
                        const ConvOptions &options) {
  verifyStrideAndPaddingDimensions(params);
  auto weightDeltas = calculateWeightDeltas(graph, zDeltas, activations, params,
                                            prog, debugPrefix, options);
  // Add the weight deltas to the weights.
  assert(weightDeltas.shape() == weights.shape());
  popstd::addTo(graph, weights, weightDeltas, -learningRate, prog,
                debugPrefix + "/UpdateWeights");
}

// Return a program to update the biases tensor with the gradients derived
// from the zDeltas tensor
void
convolutionBiasUpdate(Graph &graph, const Tensor &zDeltasUngrouped,
                      const Tensor &biases,
                      float learningRate,
                      Sequence &prog,
                      const std::string &debugPrefix) {
  auto zDeltas = groupActivations(zDeltasUngrouped);
  const auto layerName = debugPrefix + "/BiasUpdate";

  auto firstReduceCS = graph.addComputeSet(layerName + "/Reduce1");
  // The bias gradient is the sum of all the deltas.
  // The reduction of these deltas is done in three stages:
  //     The first stage reduces on each tile. It places the partial sum
  //     for each tile in the tensor 'tileReducedBiasDeltas[tile]'.
  //     The second stage reduces across tiles to a set of partial sums
  //     spread across the workers. It takes 'tileReducedBiasDeltas' as input
  //     and outputs to the 'biasPartials' 2-d tensor.
  //     The final stage reduces the 'biasPartials' 2-d tensor to get the
  //     final gradient for each bias, multiplies it by the learning rate and
  //     subtracts from the bias in the 'biases' tensor.
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  auto dType = zDeltas.elementType();
  auto numTiles = deviceInfo.getNumTiles();
  auto numBiases = biases.numElements();
  auto batchSize = zDeltas.dim(0);
  auto outNumChanGroups = zDeltas.dim(1);
  auto outDimY = zDeltas.dim(2), outDimX = zDeltas.dim(3);
  auto outChansPerGroup = zDeltas.dim(4);
  // Before the cross tile reduction. Reduce biases on each tile.
  auto zDeltasFlat = zDeltas.reshape({batchSize, outNumChanGroups,
                                      outDimY * outDimX, outChansPerGroup});

  // Calculate which bias groups have values to reduce on each tile
  std::vector<std::vector<unsigned>> deltaMappings;
  for (unsigned b = 0; b < batchSize; ++b)
    deltaMappings.push_back(computeActivationsMapping(graph, zDeltas[b], b,
                                                      batchSize));
  std::vector<std::vector<unsigned>> tileBiasGroups(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    std::unordered_set<unsigned> biasGroups;
    for (unsigned b = 0; b < batchSize; ++b) {
      auto begin = deltaMappings[b][tile];
      auto end = deltaMappings[b][tile + 1];
      auto M = outDimY * outDimX * outChansPerGroup;
      auto beginGroup = (begin / M);
      auto endGroup = ((end + M - 1) / M);
      for (unsigned biasGroup = beginGroup; biasGroup < endGroup; ++biasGroup) {
        biasGroups.insert(biasGroup);
      }
    }
    // Set tileBiasGroups[tile] to contain the indices of the bias groups to
    // be reduced on that tile.
    auto &vec = tileBiasGroups[tile];
    vec.insert(vec.end(), biasGroups.begin(), biasGroups.end());
  }

  // On each tile create vertices that reduce the on-tile deltas to a single
  // bias delta value for each bias on each tile stored in the
  // tensor tileReducedBiasDeltas[tile].
  std::vector<Tensor> tileReducedBiasDeltas;
  tileReducedBiasDeltas.reserve(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto tileNumBiasGroups = tileBiasGroups[tile].size();
    Tensor r = graph.addTensor(dType, {tileNumBiasGroups,
                                       outChansPerGroup},
                               "tileReducedBiasDeltas");
    tileReducedBiasDeltas.push_back(r);
    graph.setTileMapping(r, tile);
    for (unsigned i = 0; i < tileBiasGroups[tile].size(); ++i) {
      const auto biasGroup = tileBiasGroups[tile][i];
      auto v = graph.addVertex(firstReduceCS,
                               templateVertex("popconv::ConvBiasReduce1",
                                              dType));
      unsigned numRanges = 0;
      for (unsigned b = 0; b < batchSize; ++b) {
        auto begin = deltaMappings[b][tile];
        auto end = deltaMappings[b][tile + 1];
        auto M = outDimY * outDimX * outChansPerGroup;
        auto beginGroup = (begin / M);
        auto endGroup = ((end + M - 1) / M);
        if (beginGroup > biasGroup || endGroup <= biasGroup)
          continue;
        unsigned fieldBegin;
        if (biasGroup == beginGroup) {
          fieldBegin = (begin % M) / outChansPerGroup;
        } else {
          fieldBegin = 0;
        }
        unsigned fieldEnd;
        if (biasGroup == endGroup - 1) {
          fieldEnd = (end % M) / outChansPerGroup;
          if (fieldEnd == 0)
            fieldEnd = outDimX * outDimY;
        } else {
          fieldEnd = outDimX * outDimY;
        }
        auto in = zDeltasFlat[b][biasGroup].slice({fieldBegin, 0},
                                                  {fieldEnd, outChansPerGroup})
                                           .flatten();
        graph.connect(v["in"][numRanges++], in);
      }
      graph.setFieldSize(v["in"], numRanges);
      graph.connect(v["out"], r[i]);
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }

  /** The number of biases is often small. So the reduction of bias
   *  updates is done in two stages to balance compute.
   */
  auto numWorkers = deviceInfo.numWorkerContexts * deviceInfo.getNumTiles();
  unsigned workersPerBias, usedWorkers, maxBiasPerWorker;
  if (numWorkers > numBiases) {
    workersPerBias = numWorkers / numBiases;
    usedWorkers = workersPerBias * numBiases;
    maxBiasPerWorker = 1;
  } else {
    workersPerBias = 1;
    usedWorkers = numWorkers;
    maxBiasPerWorker = (numBiases + numWorkers - 1) / numWorkers;
  }
  auto biasPartials = graph.addTensor(dType, {usedWorkers, maxBiasPerWorker},
                                      "biasPartials");
  auto secondReduceCS = graph.addComputeSet(
                            layerName + "/Reduce2");
  for (unsigned worker = 0; worker  < usedWorkers; ++worker ) {
    auto tile = worker / deviceInfo.numWorkerContexts;
    graph.setTileMapping(biasPartials[worker].slice(0, maxBiasPerWorker), tile);
    unsigned biasBegin = (worker  * numBiases) / usedWorkers;
    unsigned biasEnd = ((worker  + workersPerBias) * numBiases) / usedWorkers;
    if (biasBegin == biasEnd)
      continue;
    unsigned numWorkerBiases = biasEnd - biasBegin;
    auto toReduce = graph.addTensor(dType, {0});
    std::vector<unsigned> numInputsPerBias;
    for (auto bias = biasBegin; bias != biasEnd; ++bias) {
      auto biasGroup = bias / outChansPerGroup;
      auto biasInGroup = bias % outChansPerGroup;
      auto biasDeltas = graph.addTensor(dType, {0});
      for (unsigned srcTile = 0; srcTile < numTiles; ++srcTile) {
        for (unsigned i = 0; i < tileBiasGroups[srcTile].size(); ++i) {
          if (biasGroup != tileBiasGroups[srcTile][i])
            continue;
          auto srcBias = tileReducedBiasDeltas[srcTile][i][biasInGroup];
          biasDeltas = append(biasDeltas, srcBias);
        }
      }
      const auto numDeltas = biasDeltas.numElements();
      auto deltaBegin =
          ((worker  % workersPerBias) * numDeltas) / workersPerBias;
      unsigned deltaEnd =
          (((worker  % workersPerBias) + 1) * numDeltas) / workersPerBias;
      toReduce = concat(toReduce, biasDeltas.slice(deltaBegin, deltaEnd));
      numInputsPerBias.push_back(deltaEnd - deltaBegin);
    }
    if (toReduce.numElements() == 0) {
      auto v = graph.addVertex(secondReduceCS,
                             templateVertex("popstd::Zero", dType));
      graph.connect(v["out"], biasPartials[worker].slice(0, maxBiasPerWorker));
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v, tile);
      continue;
    }
    auto v = graph.addVertex(secondReduceCS,
                             templateVertex("popconv::ConvBiasReduce2", dType));
    graph.connect(v["in"], toReduce);
    graph.connect(v["out"], biasPartials[worker].slice(0, numWorkerBiases));
    graph.setInitialValue(v["numInputsPerBias"], numInputsPerBias);
    graph.setTileMapping(v, tile);
  }
  auto updateBiasCS = graph.addComputeSet(layerName + "/FinalUpdate");
  const auto biasMapping = graph.getTileMapping(biases);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &interval : biasMapping[tile]) {
      for (unsigned bias = interval.begin(); bias != interval.end(); ++bias) {
        auto v = graph.addVertex(updateBiasCS,
                                 templateVertex("popconv::ConvBiasUpdate",
                                                dType));
        unsigned numPartials = 0;
        for (unsigned srcWorker = 0; srcWorker < usedWorkers; ++srcWorker) {
          unsigned biasBegin = (srcWorker * numBiases) / usedWorkers;
          unsigned biasEnd =
              ((srcWorker + workersPerBias) * numBiases) / usedWorkers;
          if (biasBegin > bias || biasEnd <= bias)
            continue;
          graph.connect(v["partials"][numPartials++],
                        biasPartials[srcWorker][bias - biasBegin]);
        }
        graph.setFieldSize(v["partials"], numPartials);
        graph.connect(v["bias"], biases[bias]);
        graph.setInitialValue(v["eta"], learningRate);
        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(firstReduceCS));
  prog.add(Execute(secondReduceCS));
  prog.add(Execute(updateBiasCS));
}

static void
addBias(Graph &graph, const Tensor &acts_, const Tensor &biases,
        ComputeSet cs) {
  const auto acts = groupActivations(acts_);
  const auto dType = acts.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto outChansPerGroup = acts.dim(4);
  const auto biasesByGroup =
      biases.reshape({biases.numElements() / outChansPerGroup,
                      outChansPerGroup});
  const auto firstInGroup = acts.dimShuffle({1, 0, 2, 3, 4})
                                .slice(0, 1, 4)
                                .reshape({acts.dim(1), acts.dim(0),
                                          acts.dim(2) * acts.dim(3)});
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerGroups =
        splitRegionsBetweenWorkers(deviceInfo, firstInGroupMapping[tile], 1);
    for (const auto &entry : perWorkerGroups) {
      auto v = graph.addVertex(cs,
                               templateVertex("popconv::AddBias",
                                              dType));
      graph.setTileMapping(v, tile);
      unsigned num = 0;
      for (const auto &interval : entry) {
        const auto begin = interval.begin();
        const auto end = interval.end();
        const auto last = end - 1;
        auto beginIndices = popstd::unflattenIndex(firstInGroup.shape(), begin);
        auto lastIndices = popstd::unflattenIndex(firstInGroup.shape(), last);
        for (unsigned g = beginIndices[0]; g != lastIndices[0] + 1; ++g) {
          unsigned batchBegin = g == beginIndices[0] ?
                                beginIndices[1] :
                                0;
          unsigned batchLast = g == lastIndices[0] ?
                               lastIndices[1] :
                               firstInGroup.dim(1) - 1;
          auto biasesWindow = biasesByGroup[g];
          for (unsigned b = batchBegin; b != batchLast + 1; ++b) {
            unsigned begin = g == beginIndices[0] && b == lastIndices[1] ?
                             beginIndices[2] :
                             0;
            unsigned last = g == lastIndices[0] && b == lastIndices[1] ?
                            lastIndices[2] :
                            firstInGroup.dim(2) - 1;
            auto actsWindow =
                acts[b][g].flatten().slice(begin * outChansPerGroup,
                                           (last + 1) * outChansPerGroup);
            graph.connect(v["acts"][num], actsWindow);
            graph.connect(v["biases"][num], biasesWindow);
            ++num;
          }
        }
      }
      graph.setFieldSize(v["acts"], num);
      graph.setFieldSize(v["biases"], num);
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    }
  }
}

void
addBias(Graph &graph, const Tensor &acts, const Tensor &biases,
        Sequence &prog, const std::string &debugPrefix) {
  ComputeSet cs = graph.addComputeSet(debugPrefix + "/addBias");
  addBias(graph, acts, biases, cs);
  prog.add(Execute(cs));
}

Tensor
fullyConnectedWeightTranspose(Graph &graph,
                              Tensor activations,
                              ConvParams params,
                              Sequence &prog, const std::string &debugPrefix,
                              const ConvOptions &options) {
  auto plan = getPlan(graph, params, options);
  auto fwdPlan = plan;
  std::swap(fwdPlan.xAxisGrainSize, fwdPlan.inChansPerGroup);
  std::swap(fwdPlan.tilesPerXAxis, fwdPlan.tilesPerInZGroupAxis);
  Tensor transposed = createInput(graph, params, "transposed", options);
  auto transposedUngroupedShape = transposed.shape();
  const auto fwdGroupSize =
      getInChansPerGroup(fwdPlan, static_cast<unsigned>(activations.dim(3)));
  const auto bwdGroupSize =
      getInChansPerGroup(plan, static_cast<unsigned>(activations.dim(2)));
  const auto dType = activations.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  activations = activations.reshape({activations.dim(0) * activations.dim(1),
                                     activations.dim(2) / bwdGroupSize,
                                     bwdGroupSize,
                                     activations.dim(3) / fwdGroupSize,
                                     fwdGroupSize})
                                    .dimShuffle({0, 1, 3, 2, 4});
  transposed = transposed.reshape({transposed.dim(0) * transposed.dim(1),
                                   transposed.dim(2) / fwdGroupSize,
                                   fwdGroupSize,
                                   transposed.dim(3) / bwdGroupSize,
                                   bwdGroupSize})
                         .dimShuffle({0, 1, 3, 2, 4});
  auto firstInBlock =
      activations.slice({0, 0, 0, 0, 0},
                        {activations.dim(0),
                         activations.dim(1),
                         activations.dim(2),
                         1,
                         1})
                 .reshape({activations.dim(0),
                           activations.dim(1),
                           activations.dim(2)});
  auto blockTileMapping =
      graph.getTileMapping(firstInBlock);
  auto transposeCS = graph.addComputeSet(debugPrefix + "/Transpose");
  for (unsigned tile = 0; tile != blockTileMapping.size(); ++tile) {

    const auto perWorkerGroups =
        splitRegionsBetweenWorkers(deviceInfo, blockTileMapping[tile], 1);
    for (const auto &entry : perWorkerGroups) {
      // Create a vertex.
      const auto v =
          graph.addVertex(transposeCS,
                          templateVertex("popconv::Transpose2D", dType));
      graph.setTileMapping(v, tile);
      graph.setInitialValue(v["numSrcColumns"],
                            static_cast<unsigned>(fwdGroupSize));
      unsigned index = 0;
      for (const auto interval : entry) {
        for (auto block = interval.begin(); block != interval.end(); ++block) {
          auto blockIndices = popstd::unflattenIndex(firstInBlock.shape(),
                                                     block);
          graph.connect(v["src"][index],
                        activations[blockIndices[0]]
                                   [blockIndices[1]]
                                   [blockIndices[2]].flatten());
          graph.connect(v["dst"][index++],
                        transposed[blockIndices[0]]
                                  [blockIndices[2]]
                                  [blockIndices[1]].flatten());
        }
      }
      graph.setFieldSize(v["dst"], index);
      graph.setFieldSize(v["src"], index);
    }
  }
  prog.add(Execute(transposeCS));
  return transposed.dimShuffle({0, 1, 3, 2, 4})
                   .reshape(transposedUngroupedShape);
}

void reportPlanInfo(std::ostream &out,
                    const poplar::Graph &graph,
                    const ConvParams &params, const ConvOptions &options) {
  verifyStrideAndPaddingDimensions(params);
  auto plan = getPlan(graph, params, options);
  out << plan;
}

void reportWeightUpdatePlanInfo(std::ostream &out,
                                const Graph &graph,
                                const Tensor &activationsUnGrouped,
                                const Tensor &zDeltasUnGrouped,
                                const ConvParams &params,
                                const ConvOptions &options) {
  auto activations = groupActivations(activationsUnGrouped);
  auto zDeltas = groupActivations(zDeltasUnGrouped);
  const auto plan =
      getWeightUpdatePlan(graph, activations, zDeltas, params, options);
  out << plan;
}


static Tensor
channelMul(Graph &graph, const Tensor &actsUngrouped, const Tensor &scale,
             Sequence &prog, const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/channelMul";
  auto cs = graph.addComputeSet(fnPrefix);

  auto actsScaledUngrouped = graph.clone(actsUngrouped);
  const auto acts = groupActivations(actsUngrouped);
  const auto actsScaled = groupActivations(actsScaledUngrouped);
  const auto dType = acts.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto outChansPerGroup = acts.dim(4);
  const auto scaleByGroup =
      scale.reshape({scale.numElements() / outChansPerGroup,
                      outChansPerGroup});
  const auto firstInGroup = acts.dimShuffle({1, 0, 2, 3, 4})
                                .slice(0, 1, 4)
                                .reshape({acts.dim(1), acts.dim(0),
                                          acts.dim(2) * acts.dim(3)});
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerGroups =
        splitRegionsBetweenWorkers(deviceInfo, firstInGroupMapping[tile], 1);
    for (const auto &entry : perWorkerGroups) {
      auto v = graph.addVertex(cs,
                               templateVertex("popconv::ChannelMul",
                                              dType));
      graph.setTileMapping(v, tile);
      unsigned num = 0;
      for (const auto &interval : entry) {
        const auto begin = interval.begin();
        const auto end = interval.end();
        const auto last = end - 1;
        auto beginIndices = popstd::unflattenIndex(firstInGroup.shape(), begin);
        auto lastIndices = popstd::unflattenIndex(firstInGroup.shape(), last);
        for (unsigned g = beginIndices[0]; g != lastIndices[0] + 1; ++g) {
          unsigned batchBegin = g == beginIndices[0] ?
                                beginIndices[1] :
                                0;
          unsigned batchLast = g == lastIndices[0] ?
                               lastIndices[1] :
                               firstInGroup.dim(1) - 1;
          auto scaleWindow = scaleByGroup[g];
          for (unsigned b = batchBegin; b != batchLast + 1; ++b) {
            unsigned begin = g == beginIndices[0] && b == lastIndices[1] ?
                             beginIndices[2] :
                             0;
            unsigned last = g == lastIndices[0] && b == lastIndices[1] ?
                            lastIndices[2] :
                            firstInGroup.dim(2) - 1;
            auto actsWindow =
                acts[b][g].flatten().slice(begin * outChansPerGroup,
                                           (last + 1) * outChansPerGroup);
            auto actsScaledWindow =
                actsScaled[b][g].flatten().slice(begin * outChansPerGroup,
                                                (last + 1) * outChansPerGroup);
            graph.connect(v["actsIn"][num], actsWindow);
            graph.connect(v["actsOut"][num], actsScaledWindow);
            graph.connect(v["scale"][num], scaleWindow);
            ++num;
          }
        }
      }
      graph.setFieldSize(v["actsIn"], num);
      graph.setFieldSize(v["actsOut"], num);
      graph.setFieldSize(v["scale"], num);
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    }
  }
  prog.add(Execute(cs));
  return actsScaledUngrouped;
}


static void
addToScaledChannel(Graph &graph, const Tensor &actsUngrouped,
                   const Tensor &addend, float scale, Sequence &prog,
                   const std::string debugPrefix) {
  const auto fnPrefix = debugPrefix + "/addToScaledChannel";
  auto cs = graph.addComputeSet(fnPrefix);
  const auto acts = groupActivations(actsUngrouped);
  const auto dType = acts.elementType();
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto outChansPerGroup = acts.dim(4);
  const auto addendByGroup =
      addend.reshape({addend.numElements() / outChansPerGroup,
                      outChansPerGroup});
  const auto firstInGroup = acts.dimShuffle({1, 0, 2, 3, 4})
                                .slice(0, 1, 4)
                                .reshape({acts.dim(1), acts.dim(0),
                                          acts.dim(2) * acts.dim(3)});
  const auto firstInGroupMapping = graph.getTileMapping(firstInGroup);
  const unsigned numTiles = firstInGroupMapping.size();
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto perWorkerGroups =
        splitRegionsBetweenWorkers(deviceInfo, firstInGroupMapping[tile], 1);
    for (const auto &entry : perWorkerGroups) {
      auto v = graph.addVertex(cs,
                               templateVertex("popconv::AddToScaledChannel",
                                              dType));
      graph.setTileMapping(v, tile);
      unsigned num = 0;
      for (const auto &interval : entry) {
        const auto begin = interval.begin();
        const auto end = interval.end();
        const auto last = end - 1;
        auto beginIndices = popstd::unflattenIndex(firstInGroup.shape(), begin);
        auto lastIndices = popstd::unflattenIndex(firstInGroup.shape(), last);
        for (unsigned g = beginIndices[0]; g != lastIndices[0] + 1; ++g) {
          unsigned batchBegin = g == beginIndices[0] ?
                                beginIndices[1] :
                                0;
          unsigned batchLast = g == lastIndices[0] ?
                               lastIndices[1] :
                               firstInGroup.dim(1) - 1;
          auto addendWindow = addendByGroup[g];
          for (unsigned b = batchBegin; b != batchLast + 1; ++b) {
            unsigned begin = g == beginIndices[0] && b == lastIndices[1] ?
                             beginIndices[2] :
                             0;
            unsigned last = g == lastIndices[0] && b == lastIndices[1] ?
                            lastIndices[2] :
                            firstInGroup.dim(2) - 1;
            auto actsWindow =
                acts[b][g].flatten().slice(begin * outChansPerGroup,
                                           (last + 1) * outChansPerGroup);
            graph.connect(v["acts"][num], actsWindow);
            graph.connect(v["addend"][num], addendWindow);
            ++num;
          }
        }
      }
      graph.setInitialValue(v["scale"], scale);
      graph.setFieldSize(v["acts"], num);
      graph.setFieldSize(v["addend"], num);
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
    }
  }
  prog.add(Execute(cs));
}

static Tensor
batchNormReduce(Graph &graph,
                const Tensor &actsUngrouped,
                float scale,
                bool doSquare,
                Sequence &prog,
                std::string partialsType,
                const std::string &debugPrefix) {
  auto redScaled = createBiases(graph, actsUngrouped, "ReducedScaled");

  auto acts = groupActivations(actsUngrouped);
  const auto layerName = debugPrefix;

  auto firstReduceCS = graph.addComputeSet(layerName + "/Reduce1");

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  auto dType = acts.elementType();
  auto numTiles = deviceInfo.getNumTiles();
  auto batchSize = acts.dim(0);
  auto outNumChanGroups = acts.dim(1);
  auto outDimY = acts.dim(2), outDimX = acts.dim(3);
  auto outChansPerGroup = acts.dim(4);
  auto numEstimates = outNumChanGroups * outChansPerGroup;

  // Before the cross tile reduction, compute running sum on each tile.
  auto actsFlat = acts.reshape({batchSize, outNumChanGroups,
                                outDimY * outDimX, outChansPerGroup});

  // Calculate which activation groups have values to reduce on each tile
  std::vector<std::vector<unsigned>> actsMappings;
  for (unsigned b = 0; b < batchSize; ++b)
    actsMappings.push_back(computeActivationsMapping(graph, acts[b], b,
                                                      batchSize));
  std::vector<std::vector<unsigned>> tileRunningEstimatesGroups(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    std::unordered_set<unsigned> runningEstimatesGroups;
    for (unsigned b = 0; b < batchSize; ++b) {
      auto begin = actsMappings[b][tile];
      auto end = actsMappings[b][tile + 1];
      auto M = outDimY * outDimX * outChansPerGroup;
      auto beginGroup = (begin / M);
      auto endGroup = ((end + M - 1) / M);
      for (unsigned reGroup = beginGroup; reGroup < endGroup; ++reGroup) {
        runningEstimatesGroups.insert(reGroup);
      }
    }
    // Set tileRunningEstimatesGroups[tile] to contain the indices of the
    // estimate groups to be reduced on that tile.
    auto &vec = tileRunningEstimatesGroups[tile];
    vec.insert(vec.end(), runningEstimatesGroups.begin(),
               runningEstimatesGroups.end());
  }

  // On each tile create vertices that use the on-tile activations to a single
  // sum value on each tile stored in the tensor tileRunningSum[tile]
  std::vector<Tensor> tileRunningSum;
  tileRunningSum.reserve(numTiles);
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto tileNumEstimatesGroups = tileRunningEstimatesGroups[tile].size();
    Tensor rSum = graph.addTensor(partialsType,
                                  {tileNumEstimatesGroups, outChansPerGroup},
                                  "tileRunningSum");

    tileRunningSum.push_back(rSum);
    graph.setTileMapping(rSum, tile);

    for (unsigned i = 0; i < tileRunningEstimatesGroups[tile].size(); ++i) {
      const auto estimatesGroup = tileRunningEstimatesGroups[tile][i];
      auto v = graph.addVertex(firstReduceCS,
                               templateVertex(doSquare ?
                                                "popconv::ConvBNReduceSquare" :
                                                "popconv::ConvBNReduce",
                                              dType, partialsType));
      unsigned numRanges = 0;
      for (unsigned b = 0; b < batchSize; ++b) {
        auto begin = actsMappings[b][tile];
        auto end = actsMappings[b][tile + 1];
        auto M = outDimY * outDimX * outChansPerGroup;
        auto beginGroup = (begin / M);
        auto endGroup = ((end + M - 1) / M);
        if (beginGroup > estimatesGroup || endGroup <= estimatesGroup)
          continue;
        unsigned fieldBegin;
        if (estimatesGroup == beginGroup) {
          fieldBegin = (begin % M) / outChansPerGroup;
        } else {
          fieldBegin = 0;
        }
        unsigned fieldEnd;
        if (estimatesGroup == endGroup - 1) {
          fieldEnd = (end % M) / outChansPerGroup;
          if (fieldEnd == 0)
            fieldEnd = outDimX * outDimY;
        } else {
          fieldEnd = outDimX * outDimY;
        }
        auto in =
          actsFlat[b][estimatesGroup].slice({fieldBegin, 0},
                                            {fieldEnd, outChansPerGroup})
                                      .flatten();
        graph.connect(v["in"][numRanges++], in);
      }
      graph.setFieldSize(v["in"], numRanges);
      graph.connect(v["sum"], rSum[i]);
      graph.setInitialValue(v["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }

  /** The number of channels is often small. So the reduction of
   *  updates is done in two stages to balance compute.
   */
  auto numWorkers = deviceInfo.numWorkerContexts * deviceInfo.getNumTiles();
  unsigned workersPerEstimates, usedWorkers, maxEstimatesPerWorker;
  if (numWorkers > numEstimates) {
    workersPerEstimates = numWorkers / numEstimates;
    usedWorkers = workersPerEstimates * numEstimates;
    maxEstimatesPerWorker = 1;
  } else {
    workersPerEstimates = 1;
    usedWorkers = numWorkers;
    maxEstimatesPerWorker = (numEstimates + numWorkers - 1) / numWorkers;
  }
  auto runningSum =
      graph.addTensor(partialsType, {usedWorkers, maxEstimatesPerWorker},
                      "runningSum");

  auto secondReduceCS = graph.addComputeSet(
                            layerName + "/Reduce2");
  for (unsigned worker = 0; worker  < usedWorkers; ++worker ) {
    auto tile = worker / deviceInfo.numWorkerContexts;
    graph.setTileMapping(runningSum[worker].slice(0, maxEstimatesPerWorker),
                         tile);

    unsigned estimatesBegin = (worker  * numEstimates) / usedWorkers;
    unsigned estimatesEnd = ((worker  + workersPerEstimates) * numEstimates) /
                            usedWorkers;
    if (estimatesBegin == estimatesEnd)
      continue;
    unsigned numWorkerEstimates = estimatesEnd - estimatesBegin;
    auto toReduceSum = graph.addTensor(partialsType, {0});
    std::vector<unsigned> numInputsPerEstimates;
    for (auto est = estimatesBegin; est != estimatesEnd; ++est) {
      auto estGroup = est / outChansPerGroup;
      auto estInGroup = est % outChansPerGroup;
      auto inpSum = graph.addTensor(partialsType, {0});
      for (unsigned srcTile = 0; srcTile < numTiles; ++srcTile) {
        for (unsigned i = 0; i < tileRunningEstimatesGroups[srcTile].size();
             ++i) {
          if (estGroup != tileRunningEstimatesGroups[srcTile][i])
            continue;
          auto srcSum = tileRunningSum[srcTile][i][estInGroup];
          inpSum = append(inpSum, srcSum);
        }
      }
      const auto numEsts = inpSum.numElements();
      auto estBegin =
          ((worker  % workersPerEstimates) * numEsts) / workersPerEstimates;
      unsigned estEnd =
          (((worker  % workersPerEstimates) + 1) * numEsts) /
          workersPerEstimates;
      toReduceSum = concat(toReduceSum, inpSum.slice(estBegin, estEnd));
      numInputsPerEstimates.push_back(estEnd - estBegin);
    }
    if (toReduceSum.numElements() == 0) {
      auto v1 = graph.addVertex(secondReduceCS,
                               templateVertex("popstd::Zero", partialsType));
      graph.connect(v1["out"],
                    runningSum[worker].slice(0, maxEstimatesPerWorker));
      graph.setInitialValue(v1["dataPathWidth"], deviceInfo.dataPathWidth);
      graph.setTileMapping(v1, tile);
      continue;
    }
    auto v = graph.addVertex(secondReduceCS,
                             templateVertex("popconv::ConvBiasReduce2",
                                            partialsType));
    graph.connect(v["in"], toReduceSum);
    graph.connect(v["out"], runningSum[worker].slice(0, numWorkerEstimates));
    graph.setInitialValue(v["numInputsPerBias"], numInputsPerEstimates);
    graph.setTileMapping(v, tile);
  }

  auto updateEstimatesCS = graph.addComputeSet(layerName + "/FinalUpdate");
  const auto estimatesMapping = graph.getTileMapping(redScaled);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    for (const auto &interval : estimatesMapping[tile]) {
      for (unsigned est = interval.begin(); est != interval.end(); ++est) {
        auto v = graph.addVertex(updateEstimatesCS,
                                 templateVertex("popconv::ConvBNReduceAndScale",
                                                partialsType, dType));
        unsigned numPartials = 0;
        for (unsigned srcWorker = 0; srcWorker < usedWorkers; ++srcWorker) {
          unsigned estBegin = (srcWorker * numEstimates) / usedWorkers;
          unsigned estEnd =
              ((srcWorker + workersPerEstimates) * numEstimates) / usedWorkers;
          if (estBegin > est || estEnd <= est)
            continue;

          graph.connect(v["sum"][numPartials++],
                        runningSum[srcWorker][est - estBegin]);
        }
        graph.connect(v["out"], redScaled[est]);
        graph.setFieldSize(v["sum"], numPartials);
        graph.setInitialValue(v["scale"], scale);
        graph.setTileMapping(v, tile);
      }
    }
  }
  prog.add(Execute(firstReduceCS));
  prog.add(Execute(secondReduceCS));
  prog.add(Execute(updateEstimatesCS));
  return redScaled;

}


std::pair<Tensor, Tensor>
batchNormEstimates(Graph &graph,
                   const Tensor &acts,
                   float eps,
                   Sequence &prog,
                   const std::string &partialsType,
                   const std::string &debugPrefix) {
  const auto fnPrefix = debugPrefix + "/BN/estimates";
  assert(acts.rank() == 4);

  // mean and standard deviation have the same mapping as biases
  const auto actsShape = acts.shape();
  const auto numElements = acts.numElements() / acts.dim(3);
  const float scale = 1.0 / numElements;

  auto mean = batchNormReduce(graph, acts, scale, false, prog, partialsType,
                              fnPrefix);

  auto power = batchNormReduce(graph, acts, scale, true, prog, partialsType,
                               fnPrefix);

  auto meanSquare = square(graph, mean, prog, fnPrefix);
  addTo(graph, power, meanSquare, -1.0, prog, fnPrefix);

  const auto varType = meanSquare.elementType();

  Tensor epsTensor;
  if (varType == "half") {
    epsTensor = graph.addConstantTensor<half>(varType, meanSquare.shape(), eps);
  } else {
    epsTensor = graph.addConstantTensor<float>(varType, meanSquare.shape(),
                                               eps);
  }

  addTo(graph, power, epsTensor, prog, fnPrefix);
  auto stdDev = sqrt(graph, power, prog, fnPrefix);
  return std::make_pair(mean, stdDev);
}

std::pair<Tensor, Tensor>
createBatchNormParams(Graph &graph, const Tensor &acts) {
  // map beta and gamma the same way as biases
  auto gamma = createBiases(graph, acts, "gamma");
  auto beta = createBiases(graph, acts, "beta");
  return std::make_pair(gamma, beta);
}

std::pair<Tensor, Tensor>
batchNormalise(Graph &graph,
               const Tensor &acts,
               const Tensor &gamma,
               const Tensor &beta,
               const Tensor &mean,
               const Tensor &stdDev,
               Sequence &prog,
               const std::string &debugPrefix) {
  assert(acts.rank() == 4);
  const auto fnPrefix = debugPrefix + "/BN/batchNormalise";
  const auto actsShape = acts.shape();
  const auto numElements = acts.numElements() / acts.dim(3);
  auto actsZeroMean = sub(graph, acts,
                          mean.broadcast(numElements, 0).reshape(actsShape),
                          prog, fnPrefix);

  auto actsWhitened = div(graph, actsZeroMean,
                          stdDev.broadcast(numElements, 0).reshape(actsShape),
                          prog, fnPrefix);

  auto actsOut = channelMul(graph, actsWhitened, gamma, prog, fnPrefix);

  addToScaledChannel(graph, actsOut, beta, 1.0, prog, fnPrefix);
  return std::make_pair(actsOut, actsWhitened);
}

std::pair<Tensor, Tensor>
batchNormDeltas(Graph &graph,
                const Tensor &actsWhitened,
                const Tensor &gradsIn,
                Sequence &prog,
                const std::string &partialsType,
                const std::string &debugPrefix) {

  const auto fnPrefix = debugPrefix + "/BN/deltas";

  const auto betaDelta =
      batchNormReduce(graph, gradsIn, 1.0, false, prog, partialsType, fnPrefix);

  const auto gradsInMultActs =
      mul(graph, gradsIn, actsWhitened, prog, fnPrefix);

  const auto gammaDelta =
      batchNormReduce(graph, gradsInMultActs, 1.0, false, prog, partialsType,
                      fnPrefix);
  return std::make_pair(gammaDelta, betaDelta);
}


Tensor batchNormGradients(Graph &graph,
                          const Tensor &actsWhitened,
                          const Tensor &gradsIn,
                          const Tensor &gammaDelta,
                          const Tensor &betaDelta,
                          const Tensor &stdDev,
                          const Tensor &gamma,
                          Sequence &prog,
                          const std::string &partialsType,
                          const std::string &debugPrefix) {
  assert(actsWhitened.rank() == 4);
  const auto fnPrefix = debugPrefix + "/BN/gradients";
  const auto actsShape = actsWhitened.shape();
  const auto numElements = actsWhitened.numElements() / actsWhitened.dim(3);
  const float rScale = 1.0 / numElements;

  auto gradient = graph.clone(gradsIn);
  prog.add(Copy(gradsIn, gradient));
  addTo(graph, gradient,
        channelMul(graph, actsWhitened, gammaDelta, prog, fnPrefix),
        -rScale, prog, fnPrefix);

  addToScaledChannel(graph, gradient, betaDelta, -rScale, prog, fnPrefix);

  return channelMul(graph, gradient,
                    div(graph, gamma, stdDev, prog, fnPrefix), prog, fnPrefix);
}

} // namespace conv
