#include "poputil/TileMapping.hpp"
#include "poplar/Program.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include "poplibs_support/gcd.hpp"
#include <boost/icl/interval_map.hpp>
#include <boost/integer/common_factor.hpp>
#include <unordered_map>

namespace poputil {

std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile,
                      unsigned grainSize) {
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto numElements = std::accumulate(shape.begin(), shape.end(), 1UL,
                                           std::multiplies<std::size_t>());
  std::vector<poplar::Interval> regions = {
    {0, numElements}
  };
  return splitRegions(regions, grainSize, numTiles, minElementsPerTile);
}

std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      const poplar::Tensor &t) {
  const auto dType = t.elementType();
  const auto &target = graph.getTarget();
  const auto typeSize = target.getTypeSize(dType);
  unsigned grainSize = target.getVectorWidth(dType);
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + typeSize - 1) / typeSize;
  return calcLinearTileMapping(graph, t.shape(), minElementsPerTile,
                               grainSize);
}

void
mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t,
                  unsigned minElementsPerTile ,
                  unsigned grainSize) {
  graph.setTileMapping(t, calcLinearTileMapping(graph, t.shape(),
                                                minElementsPerTile, grainSize));
}

void
mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t) {
  graph.setTileMapping(t, calcLinearTileMapping(graph, t));
}


unsigned
getTileImbalance(const poplar::Graph::TileToTensorMapping &mapping,
                  unsigned minElementsPerTile, unsigned grainSize) {
  unsigned maxElemsPerTile = 0;
  unsigned totalElems = 0;
  for (const auto &regions : mapping) {
    unsigned numElems = std::accumulate(regions.begin(), regions.end(), 0U,
                                        [](unsigned sum,
                                           const poplar::Interval &i) {
                                          return sum + i.size();
                                        });
    maxElemsPerTile = std::max(numElems, maxElemsPerTile);
    totalElems += numElems;
  }
  unsigned numTiles = mapping.size();
  auto balancedElemsPerTile = (totalElems + numTiles - 1) / numTiles;
  balancedElemsPerTile = std::max(balancedElemsPerTile, minElementsPerTile);
  balancedElemsPerTile = std::max(balancedElemsPerTile, grainSize);
  if (maxElemsPerTile < balancedElemsPerTile)
    return 0;
  return maxElemsPerTile - balancedElemsPerTile;
}

unsigned
getTileImbalance(const poplar::Graph &graph, const poplar::Tensor &t,
                 unsigned minElementsPerTile, unsigned grainSize) {
  return getTileImbalance(graph.getTileMapping(t), minElementsPerTile,
                          grainSize);
}

static void
rebalanceTensor(poplar::Graph &graph, const poplar::Tensor &t,
                const poplar::Graph::TileToTensorMapping &mapping,
                unsigned minElementsPerTile, unsigned grainSize,
                unsigned imbalanceThreshold) {
  auto imbalance = getTileImbalance(mapping);
  if (imbalance <= imbalanceThreshold)
    return;

  if (grainSize > minElementsPerTile)
    minElementsPerTile = grainSize;

  unsigned numTiles = mapping.size();
  std::vector<unsigned> numElemsPerTile(numTiles);
  std::vector<unsigned> targetElemsPerTile(numTiles);
  unsigned totalElems = 0;

  for (unsigned i = 0; i < numTiles; ++i) {
    const auto &regions = mapping[i];
    unsigned numElems = std::accumulate(regions.begin(), regions.end(), 0U,
                                        [](unsigned sum,
                                           const poplar::Interval &i) {
                                          return sum + i.size();
                                        });
    numElemsPerTile[i] = numElems;
    totalElems += numElems;
  }

  // If we cannot spread the tensor over all tiles then do not bother
  // rebalancing.
  // TODO: handle this case to balance over a smaller set of tiles
  if (totalElems / numTiles < minElementsPerTile)
    return;

  // Keep track of the tiles that have fewer than their required number of
  // elements.
  std::set<unsigned> lightTiles;

  auto numGrains = (totalElems + grainSize - 1) / grainSize;
  for (unsigned i = 0; i < numTiles; ++i) {
    auto beginGrain = ((i * numGrains) / numTiles);
    auto endGrain = (((i + 1) * numGrains) / numTiles);
    auto beginElem = beginGrain * grainSize;
    auto endElem = std::min(endGrain * grainSize, totalElems);
    targetElemsPerTile[i] = endElem - beginElem;
    if (targetElemsPerTile[i] > numElemsPerTile[i]) {
      lightTiles.insert(i);
    }
  }

  auto newMapping = mapping;
  for (unsigned i = 0; i < numTiles; ++i) {
    if (targetElemsPerTile[i] >= numElemsPerTile[i])
      continue;
    auto elemsToMove = numElemsPerTile[i] - targetElemsPerTile[i];
    for (auto it = lightTiles.begin(); elemsToMove != 0;) {
      auto dst = *it;
      auto space = targetElemsPerTile[dst] - numElemsPerTile[dst];
      auto N = std::min(elemsToMove, space);

      elemsToMove -= N;
      numElemsPerTile[i] -= N;
      numElemsPerTile[dst] += N;
      auto &srcRegions = newMapping[i];
      auto &dstRegions = newMapping[dst];
      for (auto regionIt = srcRegions.begin(); N != 0;) {
        auto R = regionIt->size();

        if (R <= N) {
          dstRegions.push_back(*regionIt);
          regionIt = srcRegions.erase(regionIt);
          N -= R;
        } else {
          auto a = regionIt->begin();
          auto b = regionIt->begin() + N;
          auto c = regionIt->begin() + R;
          dstRegions.push_back(poplar::Interval(a, b));
          *regionIt = poplar::Interval(b, c);
          N = 0;
          ++regionIt;
        }
      }
      if (numElemsPerTile[dst] == targetElemsPerTile[dst]) {
        auto next = std::next(it);
        lightTiles.erase(it);
        it = next;
      } else {
        ++it;
      }
    }
  }

  graph.setTileMapping(t, newMapping);
}

void
rebalanceTensor(poplar::Graph &graph, const poplar::Tensor &t,
                unsigned minElementsPerTile, unsigned grainSize,
                unsigned imbalanceThreshold) {
  rebalanceTensor(graph, t, graph.getTileMapping(t), minElementsPerTile,
                  grainSize, imbalanceThreshold);
}

void
mapOutputForElementWiseOp(
    poplar::Graph &graph,
    const std::vector<poplar::Tensor> &inputs,
    const poplar::Tensor &output,
    unsigned grainSize,
    unsigned minGrainsPerTile) {
  std::vector<unsigned> tilesOccupied(inputs.size());
  std::vector<unsigned> numRegions(inputs.size());
  std::vector<size_t> minTileElements(inputs.size(),
                                      std::numeric_limits<size_t>::max());
  std::vector<size_t> maxTileElements(inputs.size());
  std::vector<size_t> maxCommonGrainSize(inputs.size(), grainSize);
  std::vector<bool> parallelWriteable(inputs.size());

  // Gather info on distribution of inputs.
  for (unsigned i = 0; i < inputs.size(); ++i) {
    if (!inputs[i].isParallelWriteable())
      continue;
    parallelWriteable[i] = true;
    const auto mapping = graph.getTileMapping(inputs[i]);
    for (const auto &tile : mapping) {
      if (!tile.empty()) {
        tilesOccupied[i]++;
        numRegions[i] += tile.size();
        size_t tileElements = 0;
        for(const auto &interval : tile) {
          tileElements += interval.size();
          maxCommonGrainSize[i] =
            boost::integer::gcd(maxCommonGrainSize[i], interval.size());
        }
        minTileElements[i] = std::min(minTileElements[i], tileElements);
        maxTileElements[i] = std::max(maxTileElements[i], tileElements);
      }
    }
  }

  // If an input tensor has a suitable mapping then map the output to
  // match the most well distributed of these.
  int best = -1;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    // If not parallel writeable, either this has constant elements with
    // indeterminate mapping, or some elements alias others, and likely
    // the resulting tile mapping will not be well distributed.
    if (!parallelWriteable[i])
      continue;

    // If the input does not have a suitable grain size then skip it.
    if (maxCommonGrainSize[i] != grainSize)
      continue;

    // If the input does not have a suitable number of grains per-tile
    // then skip it.
    if (minTileElements[i] / grainSize < minGrainsPerTile)
      continue;

    // Select the tensor with the minimum maximum tile elements
    if (best < 0 || maxTileElements[i] < maxTileElements[best]) {
      best = i;
    } else if (maxTileElements[i] == maxTileElements[best]) {
      // If both have the same maximum, select the tensor which is spread onto
      // the most tiles, or if two tensors share the same number of tiles, then
      // select the one which has the fewest overall regions
      if ((tilesOccupied[i] > tilesOccupied[best]) ||
          (tilesOccupied[i] == tilesOccupied[best] &&
           numRegions[i] < numRegions[best])) {
        best = i;
      }
    }
  }

  // Set output's mapping either based on a suitable input tensor's mapping,
  // or a linear mapping with the given restrictions on grain size and no.
  // of grains per-tile.
  if (best >= 0) {
    graph.setTileMapping(output, graph.getTileMapping(inputs[best]));
  } else {
    poputil::mapTensorLinearly(graph, output,
                               minGrainsPerTile * grainSize, grainSize);
  }
}

// This value is set rather arbitrarily to match the default min elements
// per tile in the other mapping functions.
static const unsigned DEFAULT_IMBALANCE_THRESHOLD = 128;

void rebalanceTensor(poplar::Graph &graph, const poplar::Tensor &t) {
  const auto dType = t.elementType();
  const auto &target = graph.getTarget();
  const auto typeSize = target.getTypeSize(dType);
  unsigned grainSize = target.getVectorWidth(dType);
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile =
    (minBytesPerTile + typeSize - 1) / typeSize;
  rebalanceTensor(graph, t, grainSize, minElementsPerTile,
                  DEFAULT_IMBALANCE_THRESHOLD);
}

static boost::icl::interval<unsigned>::type
toIclInterval(const poplar::Interval &interval) {
  return boost::icl::interval<unsigned>::right_open(interval.begin(),
                                                    interval.end());
}

class TensorUseTrackerState {
public:
  using TileUsage = std::vector<boost::icl::interval_set<unsigned>>;
  std::map<poplar::VariableRef, TileUsage> usage;
  unsigned numTiles;
  TensorUseTrackerState(unsigned numTiles) : numTiles(numTiles) {}
  TileUsage &getUsage(poplar::VariableRef v) {
    auto m = usage.find(v);
    if (m != usage.end())
      return m->second;
    return usage.emplace(v, TileUsage(numTiles)).first->second;
  }
};

TensorUseTracker::TensorUseTracker(unsigned numTiles) {
  st = std::unique_ptr<TensorUseTrackerState>(
        new TensorUseTrackerState(numTiles)
       );
}

TensorUseTracker::~TensorUseTracker() {}

void TensorUseTracker::add(const poplar::Graph &graph,
                         unsigned tile, const poplar::Tensor &t) {
  const auto varRegions = t.getVarRegions();
  for (const auto &region : varRegions) {
    if (graph.isConstant(region.var))
      continue;
    auto &usage = st->getUsage(region.var);
    usage[tile].add(toIclInterval(region.interval));
  }
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


void TensorUseTracker::mapTensorsByUse(poplar::Graph &graph,
                                     unsigned grainSize,
                                     unsigned minElementsPerTile,
                                     bool optimizeHaloRegions) {
  const auto numTiles = graph.getTarget().getNumTiles();
  for (const auto &usageEntry : st->usage) {
    const auto t = graph.getVariable(usageEntry.first);
    const auto &usage = usageEntry.second;
    boost::icl::interval_map<unsigned, std::set<unsigned>> uses;
    for (unsigned tile = 0; tile < numTiles; ++tile) {
      std::set<unsigned> tileSet{tile};
      for (const auto &region : usage[tile]) {
        uses.add(std::make_pair(region, tileSet));
      }
    }
    if (iterative_size(uses) == 0) {
      auto mapping = poputil::calcLinearTileMapping(graph, t.shape(),
                                                    minElementsPerTile,
                                                    grainSize);
      graph.setTileMapping(t, mapping);
      continue;
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
    const auto numElements = t.numElements();
    const unsigned numGrains = (numElements + grainSize - 1) / grainSize;
    extendPartialMap(grainToTiles, 0U, numGrains);

    if (optimizeHaloRegions) {
      optimizeHaloMapping(grainToTiles);
    }

    // Build a map from sets of tiles to grains they use.
    std::map<std::set<unsigned>, std::vector<poplar::Interval>> tilesToGrains;
    for (const auto &entry : grainToTiles) {
      tilesToGrains[entry.second].emplace_back(entry.first.lower(),
                                               entry.first.upper());
    }
    std::vector<std::vector<poplar::Interval>> mapping(numTiles);
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
    graph.setTileMapping(t, mapping);
  }
}

bool TensorUseTracker::empty() const {
  return st->usage.empty();
}

poplar::Tensor
cloneToIpu(poplar::Graph& masterGraph, const poplar::Tensor &t,
           unsigned dstIpu, poplar::StringRef name,
           poplar::TensorCloneMethod method) {
  auto mapping = masterGraph.getTileMapping(t);
  const auto &target = masterGraph.getTarget();
  const auto tilesPerIPU = target.getTilesPerIPU();
  const auto numIPUs = target.getNumIPUs();
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    if (ipu == dstIpu)
      continue;
    for (unsigned i = 0; i != tilesPerIPU; ++i) {
      auto &oldTileIntervals = mapping[ipu * tilesPerIPU + i];
      if (oldTileIntervals.empty())
        continue;
      auto &newTileIntervals = mapping[dstIpu * tilesPerIPU + i];
      if (newTileIntervals.empty()) {
        newTileIntervals = std::move(oldTileIntervals);
      } else {
        newTileIntervals.insert(newTileIntervals.end(),
                                oldTileIntervals.begin(),
                                oldTileIntervals.end());
      }
      oldTileIntervals.clear();
    }
  }
  auto tLocal = masterGraph.clone(t, name, method);
  masterGraph.setTileMapping(tLocal, mapping);
  return tLocal;
}

poplar::Tensor copyToIpu(poplar::Graph& masterGraph, const poplar::Tensor &t,
                         poplar::program::Sequence &prog, unsigned dstIpu,
                         poplar::StringRef name,
                         poplar::TensorCloneMethod method) {
  auto tLocal = cloneToIpu(masterGraph, t, dstIpu, name, method);
  // Create source and destination tensor for the copy. These are different
  // from the source and cloned tensor only if the order and aliases are
  // preserved in the cloned tensor
  auto tLocalForCopy = tLocal;
  auto tForCopy = t;
  if (method == poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES) {
    // remove all aliased regions in the source and destination tensor
    auto tLocalFlat = tLocal.flatten();
    auto tFlat = t.flatten();
    auto tFlatRegions =
        masterGraph.getSortedContiguousRegions(tFlat,
                                               {{0, tFlat.numElements()}},
                                               true);
    tLocalForCopy = concat(tLocalFlat.slices(tFlatRegions));
    tForCopy = concat(tFlat.slices(tFlatRegions));
  }
  prog.add(poplar::program::Copy(tForCopy, tLocalForCopy));
  return tLocal;
}

bool dimIsSplitOverIPUs(const poplar::Graph &graph,
                        const poplar::Tensor &t,
                        unsigned dimension) {
  const auto &target = graph.getTarget();
  if (target.getNumIPUs() == 1) {
    return false;
  }

  const auto tilesPerIPU = target.getTilesPerIPU();
  const auto dimElems = t.dim(dimension);
  auto tShuf = t.dimRoll(dimension, t.rank() - 1);
  auto tMapping = graph.getTileMapping(tShuf);

  using IntervalMap = boost::icl::interval_map<std::size_t,
                                               unsigned,
                                               boost::icl::partial_enricher>;
  using Interval = boost::icl::interval<std::size_t>;

  IntervalMap intervalToIPU;
  for (unsigned tile = 0; tile < tMapping.size(); ++tile) {
    const auto ipu = tile / tilesPerIPU;
    for (const auto &i : tMapping[tile]) {
      intervalToIPU +=
        std::make_pair(Interval::right_open(i.begin(), i.end()), ipu);
    }
  }

  // Check each slice of the dimension is not split across multiple IPUs.
  for (const auto &entry : intervalToIPU) {
    const auto &region = entry.first;
    if ((region.lower() % dimElems) ||
        (region.upper() % dimElems)) {
      return true;
    }
  }
  return false;
}

unsigned detectInnermostGrouping(const poplar::Graph &graph,
                               const poplar::Tensor &t0) {
  if (t0.rank() == 0)
    throw poplibs_error("Cannot detect channel grouping of 0-rank tensor");

  if (t0.numElements() == 0)
    return 1;

  // Sample the first point in the inner dimension
  auto t = t0;
  while (t.rank() != 1)
    t = t[0];

  // Perform a binary search to find the largest contiguous slice in
  // the inner dimension.
  auto lower = 1U;
  auto upper = t.numElements();
  while (lower != upper) {
    // Find a mid-point such that lower < mid <= upper
    auto mid = upper - (upper - lower) / 2;
    if (t.slice(0, mid).isContiguous()) {
      lower = mid;
    } else {
      upper = mid - 1;
    }
  }

  // Find the largest contiguous region on a tile as an estimate of grouping
  const auto tileMapping = graph.getTileMapping(t);
  std::size_t maxRegionSize = 0;
  for (const auto &regions : tileMapping) {
    if (regions.empty())
      continue;
    const auto maxIt =
        std::max_element(regions.begin(), regions.end(),
                         [](const poplar::Interval &a,
                            const poplar::Interval &b) {
        return a.size() < b.size();
    });
    maxRegionSize = std::max(maxRegionSize, maxIt->size());
  }

  // Use the greatest common divisor between channel grouping detected on a tile
  // and contiguous regions of the tensor. Note that in the case when a group
  // is partially mapped to a tile, GCD doesn't  give the correct result.
  auto grouping = gcd(maxRegionSize, upper);

  // The channel grouping must divide the number of channels
  if (t.numElements() % grouping != 0)
    grouping = 1;
  return grouping;
}

std::vector<GroupingInfo>
detectDimGroupings(const poplar::Graph &graph, const poplar::Tensor &t) {
  std::vector<GroupingInfo> info;

  auto dims = t.rank();
  auto groupedT = t;
  unsigned totalGrouping = 1;
  while (true) {
    unsigned grouping = 1;
    unsigned groupedDim = 0;

    for (std::size_t d = 0; d < dims; ++d) {
      // Skip singular dimensions
      if (groupedT.dim(d) == 1)
        continue;
      // Detect grouping of this dim along with previous groupings
      auto permutation =
        groupedT.dimRoll(d, dims - 1).flatten(dims - 1, groupedT.rank());
      auto g = detectInnermostGrouping(graph, permutation);
      // Even though we may already have found some grouping, the new
      // grouping we find may not be a multiple of totalGrouping if
      // there is a grouping in a weirdly sized combination of dimensions
      // so bottom out at 1 so that the gcd below gives the desired result.
      auto thisGrouping = g % totalGrouping ? 1u : g / totalGrouping;
      thisGrouping = gcd<unsigned>(thisGrouping, groupedT.dim(d));
      if (thisGrouping > grouping) {
        groupedDim = d;
        grouping = thisGrouping;
      }
    }

    // No more groupings to be found, we're done.
    if (grouping == 1)
      break;

    info.emplace_back(groupedDim, grouping);
    totalGrouping *= grouping;
    assert((groupedT.dim(groupedDim) % grouping) == 0);
    // Roll the grouping to the back for the next round
    groupedT = groupedT.reshapePartial(groupedDim,
                                       groupedDim + 1,
                                       {groupedT.dim(groupedDim) / grouping,
                                       grouping})
                       .dimRoll(groupedDim + 1, dims);
  }

  return info;
}

} // end namespace popops
