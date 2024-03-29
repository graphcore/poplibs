// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "poputil/TileMapping.hpp"
#include "poplar/Program.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/logging.hpp"
#include "poputil/DebugInfo.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <boost/functional/hash.hpp>
#include <boost/icl/interval_map.hpp>
#include <boost/icl/interval_set.hpp>
#include <boost/integer/common_factor.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <set>
#include <tbb/parallel_for.h>

using namespace poplibs_support;

namespace poputil {

struct MappingSummary {
  unsigned first;
  unsigned last;
  unsigned used;
};
MappingSummary
getMappingSummary(const std::vector<std::vector<poplar::Interval>> &map) {
  MappingSummary summary = {static_cast<unsigned>(map.size()), 0, 0};
  for (unsigned i = 0; i < map.size(); i++) {
    if (map[i].size()) {
      summary.last = i;
      summary.first = std::min(summary.first, i);
      summary.used++;
    }
  }
  return summary;
}

static poplar::Graph::TileToTensorMapping
remapWithOffset(const poplar::Graph::TileToTensorMapping &mapping,
                std::size_t numTiles, std::size_t offset, bool ascendingOrder) {
  if (numTiles == 0) {
    throw poputil::poplibs_error("Cannot remap Tensor over zero tiles");
  }
  poplar::Graph::TileToTensorMapping newMapping(numTiles);
  std::size_t dstTile = ascendingOrder ? offset : numTiles - offset - 1;
  dstTile = dstTile % numTiles;
  auto increment = ascendingOrder ? 1 : -1;
  for (unsigned tile = 0; tile != mapping.size(); ++tile) {
    if (!mapping[tile].empty()) {
      newMapping[dstTile] = std::move(mapping[tile]);
    }
    if (dstTile == 0 && !ascendingOrder) {
      dstTile = numTiles - 1;
    } else if (dstTile == numTiles - 1 && ascendingOrder) {
      dstTile = 0;
    } else {
      dstTile += increment;
    }
  }
  return newMapping;
}

std::size_t chooseMappingOffset(std::size_t numTiles,
                                const std::vector<std::size_t> &shape,
                                std::size_t seed) {
  if (numTiles == 0) {
    throw poputil::poplibs_error("Cannot choose a offset from zero tiles");
  }
  boost::hash_range(seed, shape.begin(), shape.end());
  return seed % numTiles;
}

std::size_t chooseMappingOffset(std::size_t numTiles,
                                const std::vector<std::size_t> &shape) {
  return chooseMappingOffset(numTiles, shape, 0x9e3779b9UL);
}

std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Target &target,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile, unsigned grainSize,
                      unsigned offset, bool ascendingOrder) {
  const auto numTiles = target.getNumTiles();
  const auto numElements = std::accumulate(shape.begin(), shape.end(), 1UL,
                                           std::multiplies<std::size_t>());
  std::vector<poplar::Interval> regions = {{0, numElements}};
  auto mapping = splitRegions(regions, grainSize, numTiles, minElementsPerTile);

  if (offset != 0 || !ascendingOrder) {
    mapping = remapWithOffset(mapping, numTiles, offset, ascendingOrder);
  }
  if (logging::popops::shouldLog(logging::Level::Debug)) {
    const auto summary = getMappingSummary(mapping);
    logging::popops::debug("  CalcLinearMapping Offset:{} ascendingOrder:{}"
                           " Summary: Tiles:[{}, {}) Used:{} MeanPerTile:{}",
                           offset, ascendingOrder, summary.first, summary.last,
                           summary.used,
                           static_cast<float>(product(shape)) / summary.used);
  }
  return mapping;
}

std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph,
                      std::vector<std::size_t> shape,
                      unsigned minElementsPerTile, unsigned grainSize,
                      unsigned offset, bool ascendingOrder) {
  return calcLinearTileMapping(graph.getTarget(), shape, minElementsPerTile,
                               grainSize, offset, ascendingOrder);
}

std::vector<std::vector<poplar::Interval>> calcLinearTileMapping(
    const poplar::Target &target, const std::vector<std::size_t> shape,
    poplar::Type elementType, unsigned offset, bool ascendingOrder) {
  const auto typeSize = target.getTypeSize(elementType);
  unsigned grainSize = target.getVectorWidth(elementType);
  const auto minBytesPerTile = 128;
  const auto minElementsPerTile = (minBytesPerTile + typeSize - 1) / typeSize;
  return calcLinearTileMapping(target, shape, minElementsPerTile, grainSize,
                               offset, ascendingOrder);
}

std::vector<std::vector<poplar::Interval>>
calcLinearTileMapping(const poplar::Graph &graph, const poplar::Tensor &t,
                      unsigned offset, bool ascendingOrder) {
  return calcLinearTileMapping(graph.getTarget(), t.shape(), t.elementType(),
                               offset, ascendingOrder);
}

std::pair<poplar::Graph::TileToTensorMapping, unsigned>
calcLinearTileMappingAndNewOffset(const poplar::Graph &graph,
                                  const poplar::Tensor &t, unsigned offset) {
  auto mapping = poputil::calcLinearTileMapping(graph, t, 0, true);

  const auto tilePerIpu = graph.getTarget().getNumTiles();
  const auto tilesInMapping = mapping.size();
  assert(tilePerIpu >= tilesInMapping);
  assert(offset < tilePerIpu);

  mapping.resize(tilePerIpu);
  std::rotate(mapping.rbegin(), mapping.rbegin() + offset, mapping.rend());

  offset += tilesInMapping;
  offset %= tilePerIpu;

  return {mapping, offset};
}

void mapTensorLinearlyWithOffset(poplar::Graph &graph, const poplar::Tensor &t,
                                 unsigned minElementsPerTile,
                                 unsigned grainSize, unsigned offset,
                                 bool ascendingOrder) {
  logging::popops::debug(
      "LinearMapping minPerTile:{} grain:{} Tensor:{}({}):{}",
      minElementsPerTile, grainSize, t.shape(), t.elementType(),
      t.getDebugStr());
  logging::popops::debug("  Var:{}", t.getVarStr());

  graph.setTileMapping(t, calcLinearTileMapping(graph, t.shape(),
                                                minElementsPerTile, grainSize,
                                                offset, ascendingOrder));
}

void mapTensorLinearlyWithOffset(poplar::Graph &graph, const poplar::Tensor &t,
                                 unsigned offset, bool ascendingOrder) {
  logging::popops::debug(
      "LinearMapping minPerTile:Default grain:Default Tensor{}({}):{}",
      t.shape(), t.elementType(), t.getDebugStr());
  logging::popops::debug("  Var:{}", t.getVarStr());
  graph.setTileMapping(t,
                       calcLinearTileMapping(graph, t, offset, ascendingOrder));
}

void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t,
                       unsigned minElementsPerTile, unsigned grainSize) {
  mapTensorLinearlyWithOffset(graph, t, minElementsPerTile, grainSize, 0, true);
}

void mapTensorLinearly(poplar::Graph &graph, const poplar::Tensor &t) {
  mapTensorLinearlyWithOffset(graph, t, 0, true);
}

unsigned getTileImbalance(const poplar::Graph::TileToTensorMapping &mapping,
                          unsigned minElementsPerTile, unsigned grainSize) {
  unsigned maxElemsPerTile = 0;
  unsigned totalElems = 0;
  for (const auto &regions : mapping) {
    unsigned numElems = std::accumulate(
        regions.begin(), regions.end(), 0U,
        [](unsigned sum, const poplar::Interval &i) { return sum + i.size(); });
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

unsigned getTileImbalance(const poplar::Graph &graph, const poplar::Tensor &t_,
                          unsigned minElementsPerTile, unsigned grainSize) {
  auto t = t_.flatten();
  graph.reorderToSimplify(&t, {}, false);
  return getTileImbalance(graph.getTileMapping(t), minElementsPerTile,
                          grainSize);
}

poplar::Tensor cloneToIpu(poplar::Graph &masterGraph, const poplar::Tensor &t,
                          unsigned dstIpu,
                          const poplar::DebugContext &debugContext,
                          poplar::TensorCloneMethod method) {

  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, dstIpu, method));

  if (dstIpu >= masterGraph.getTarget().getNumIPUs()) {
    throw poputil::poplibs_error(
        "Destination IPU index (" + std::to_string(dstIpu) +
        ") is invalid. There are " +
        std::to_string(masterGraph.getTarget().getNumIPUs()) + " IPUs.");
  }

  auto tLocal = masterGraph.clone(t, {di}, method);
  auto tSimple = t.flatten();
  auto tLocalSimple = tLocal.flatten();
  masterGraph.reorderToSimplify(&tSimple, {&tLocalSimple});
  auto mapping = masterGraph.getTileMapping(tSimple);
  const auto &target = masterGraph.getTarget();
  const auto tilesPerIPU = target.getTilesPerIPU();
  const auto numIPUs = target.getNumIPUs();
  assert(mapping.size() >= target.getNumTiles());
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
  masterGraph.setTileMapping(tLocalSimple, mapping);
  di.addOutput(tLocal);
  return tLocal;
}

poplar::Tensor cloneToGraph(poplar::Graph &srcGraph, poplar::Graph &dstGraph,
                            const poplar::Tensor &t,
                            const poplar::DebugContext &debugContext,
                            poplar::TensorCloneMethod method) {
  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, method));

  if (srcGraph.getReplicationFactor() != dstGraph.getReplicationFactor()) {
    throw poputil::poplibs_error(
        "Cannot clone a tensor to a graph with a different replication factor. "
        "The source graph has replication factor " +
        std::to_string(srcGraph.getReplicationFactor()) +
        " and the destination graph has replication factor " +
        std::to_string(dstGraph.getReplicationFactor()) + ".");
  }

  auto tLocal = srcGraph.clone(t, {di}, method);
  auto tSimple = t.flatten();
  auto tLocalSimple = tLocal.flatten();
  srcGraph.reorderToSimplify(&tSimple, {&tLocalSimple});

  auto mapping = srcGraph.getTileMapping(tSimple);

  for (std::size_t tile = dstGraph.getTarget().getNumTiles();
       tile < mapping.size(); ++tile) {
    if (!mapping[tile].empty()) {
      throw poputil::poplibs_error(
          "Cannot clone a tensor to a graph with fewer tiles than required by "
          "the input tensor mapping. There are elements mapped to "
          "tile " +
          std::to_string(tile) +
          " in the source graph, but the destination graph only has " +
          std::to_string(dstGraph.getTarget().getNumTiles()) + " tiles.");
    }
  }

  mapping.resize(dstGraph.getTarget().getNumTiles());
  dstGraph.setTileMapping(tLocalSimple, mapping);

  di.addOutput(tLocal);
  return tLocal;
}

std::pair<poplar::Tensor, unsigned>
cloneAndExpandAliasing(poplar::Graph &graph, const poplar::Tensor &t,
                       unsigned offset,
                       const poplar::DebugContext &debugContext) {
  // If the source tensor doesn't contain aliasing, there is no special
  // remapping that needs to be done.
  if (!t.containsAliases()) {
    return {graph.clone(t, debugContext), offset};
  }

  const auto tFlat = t.flatten();

  std::vector<std::size_t> aliases;
  auto sequences = graph.getSortedContiguousRegions(
      tFlat, {{0, tFlat.numElements()}}, false, &aliases);

  // Flatten `sequences` into `intervals`.
  std::vector<poplar::Interval> intervals;
  for (const auto &sequence : sequences) {
    for (const auto &interval : sequence) {
      intervals.push_back(interval);
    }
  }
  // Also create a new interval vector that will undo the shuffling applied
  // using `intervals`. See the documentation of
  // `poputil::calculateUnshufflingIntervals`.
  const auto unshufflingIntervals = calculateUnshufflingIntervals(intervals);

  // Reorder `tFlat` such that contiguous regions are next to each other.
  auto tFlatReordered = poplar::concat(tFlat.slices(intervals));

  // Clone the source tensor. The `CREATE_NEW_ORDER` tensor clone method will
  // preserve the tile mapping of the source tensor and remove any aliasing
  // elements.
  auto dstFlatReordered =
      graph.clone(tFlatReordered, debugContext,
                  poplar::TensorCloneMethod::CREATE_NEW_ORDER);

  // Reorder `dstFlatReordered` back to the original ordering.
  auto dstFlat = poplar::concat(dstFlatReordered.slices(unshufflingIntervals));

  // Extract all aliasing intervals.
  std::vector<poplar::Interval> aliasingIntervals;
  for (std::size_t i = 0; i < intervals.size(); i++) {
    const auto &interval = intervals[i];
    if (interval.lower() != aliases[i]) { // Is the interval an alias?
      aliasingIntervals.push_back(interval);
    }
  }

  // If the source tensor didn't have aliasing intervals, this function would
  // have returned by now.
  assert(!aliasingIntervals.empty());

  // Linearly remap all tensor elements which correspond to aliasing intervals
  // in the source tensor.
  auto dstFlatAliasing = poplar::concat(dstFlat.slices(aliasingIntervals));
  const auto [mapping, newOffset] =
      calcLinearTileMappingAndNewOffset(graph, dstFlatAliasing, offset);
  graph.setTileMapping(dstFlatAliasing, mapping);

  return {dstFlat.reshape(t.shape()), newOffset};
}

poplar::Tensor createIpuCopy(poplar::Graph &masterGraph,
                             const poplar::Tensor &t, unsigned dstIpu,
                             poplar::Tensor &copySrc, poplar::Tensor &copyDst,
                             const poplar::DebugContext &debugContext,
                             poplar::TensorCloneMethod method) {
  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(t, dstIpu, copySrc, copyDst, method));

  auto tLocal = poputil::cloneToIpu(masterGraph, t, dstIpu, {di}, method);
  // Create source and destination tensor for the copy. These are different
  // from the source and cloned tensor only if the order and aliases are
  // preserved in the cloned tensor
  copyDst = tLocal;
  copySrc = t;
  if (method == poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES) {
    // remove all aliased regions in the source and destination tensor
    auto tLocalFlat = tLocal.flatten();
    auto tFlat = t.flatten();
    auto tFlatRegions = masterGraph.getSortedContiguousRegions(
        tFlat, {{0, tFlat.numElements()}}, true);
    copyDst = concat(tLocalFlat.slices(tFlatRegions));
    copySrc = concat(tFlat.slices(tFlatRegions));
  }
  di.addOutput(tLocal);
  return tLocal;
}

poplar::Tensor copyToIpu(poplar::Graph &graph, const poplar::Tensor &t,
                         poplar::program::Sequence &prog, unsigned dstIpu,
                         const poplar::DebugContext &debugContext,
                         poplar::TensorCloneMethod method) {
  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, dstIpu, method));

  poplar::Tensor tLocalForCopy, tForCopy;
  auto tLocal =
      createIpuCopy(graph, t, dstIpu, tForCopy, tLocalForCopy, {di}, method);
  prog.add(poplar::program::Copy(tForCopy, tLocalForCopy, false, {di}));
  di.addOutput(tLocal);
  return tLocal;
}

bool dimIsSplitOverTiles(const poplar::Graph &graph, const poplar::Tensor &t,
                         unsigned dimension) {
  const auto dimElems = t.dim(dimension);

  if (dimElems == 0) {
    return false;
  }

  // We split the tensor into one slice per element of the dimension to
  // check and then see if each slice's tile mapping matches.
  std::vector<poplar::Tensor> slices(dimElems);
  std::vector<poplar::Tensor *> slicePtrs;
  slicePtrs.reserve(dimElems - 1);
  slices[0] = t.slice(0, 1, dimension).flatten();
  for (std::size_t i = 1; i < dimElems; ++i) {
    slices[i] = t.slice(i, i + 1, dimension).flatten();
    slicePtrs.emplace_back(&slices[i]);
  }

  graph.reorderToSimplify(&slices[0], slicePtrs, false);

  const auto referenceMapping = graph.getTileMapping(slices[0]);
  for (std::size_t i = 1; i < dimElems; ++i) {
    if (graph.getTileMapping(slices[i]) != referenceMapping) {
      return true;
    }
  }

  return false;
}

bool dimIsSplitOverIPUs(const poplar::Graph &graph, const poplar::Tensor &t,
                        unsigned dimension) {
  const auto &target = graph.getTarget();
  if (target.getNumIPUs() == 1) {
    return false;
  }

  const auto tilesPerIPU = target.getTilesPerIPU();
  const auto dimElems = t.dim(dimension);
  const auto tShuf = t.dimRoll(dimension, t.rank() - 1);
  const auto tMapping = graph.getTileMapping(tShuf);

  using IntervalMap = boost::icl::interval_map<std::size_t, unsigned,
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
    if ((region.lower() % dimElems) || (region.upper() % dimElems)) {
      return true;
    }
  }
  return false;
}

poplar::Tensor
createBroadcastOperand(poplar::Graph &graph, const poplar::Tensor &fullTensor,
                       const poplar::Type &type, unsigned dim,
                       bool ditherMapping,
                       const poplar::DebugContext &debugContext) {

  POPUTIL_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(fullTensor, type, dim, ditherMapping));

  logging::popops::debug("createBroadcastOperand DebugStr:{}", debugContext);
  logging::popops::debug("  dither:{} dim:{} fullTensor:{}({}):{}",
                         ditherMapping, dim, fullTensor.shape(),
                         fullTensor.elementType(), fullTensor.getDebugStr());
  assert(dim < fullTensor.rank());
  const auto &target = graph.getTarget();
  auto t = fullTensor.dimRoll(dim, fullTensor.rank() - 1);
  const auto numDimElems = fullTensor.dim(dim);
  auto out = graph.addVariable(type, {numDimElems}, {di});

  const auto numTiles = target.getNumTiles();
  TensorUseTracker useTracker(numTiles);
  const auto mapping = graph.getTileMapping(t);

  tbb::parallel_for(std::size_t(0), mapping.size(), [&](std::size_t tile) {
    boost::icl::interval_set<std::size_t> outRegions;
    for (const auto &region : mapping[tile]) {
      if (region.begin() != region.end()) {
        auto rBegin = region.begin() % numDimElems;
        auto rEnd = region.end() % numDimElems;
        if (region.size() >= numDimElems) {
          // A single region with all elements in a tile
          outRegions +=
              boost::icl::interval<size_t>::right_open(0, numDimElems);
          break;
        } else {
          if (rBegin < rEnd) {
            outRegions +=
                boost::icl::interval<size_t>::right_open(rBegin, rEnd);
          } else {
            outRegions +=
                boost::icl::interval<size_t>::right_open(rBegin, numDimElems);
            outRegions += boost::icl::interval<size_t>::right_open(0, rEnd);
          }
        }
      }
      // All elements of the output are covered
      if (outRegions.size() == numDimElems) {
        break;
      }
    }
    // convert to poplar intervals
    std::vector<poplar::Interval> intervals;
    intervals.reserve(outRegions.iterative_size());
    for (const auto &it : outRegions) {
      intervals.emplace_back(it.lower(), it.upper());
    }
    if (!intervals.empty()) {
      useTracker.add(graph, tile, concat(out.slices(intervals)));
    }
  });

  if (useTracker.empty()) {
    logging::popops::debug("  Mapping linearly");
    mapTensorLinearly(graph, out);
  } else {
    const auto grainSize =
        target.getDataPathWidth() / (8 * target.getTypeSize(type));
    useTracker.mapTensorsByUse(
        graph, grainSize, 0, false,
        TensorUseTracker::MappingMethod::ConstrainMappingToUsedTiles);
  }

  // remap with dithering
  if (ditherMapping) {
    // Randomise the offset to the start tile for mapping of the variable
    const auto outMapping = graph.getTileMapping(out);
    const auto offset =
        chooseMappingOffset(outMapping.size(), fullTensor.shape());
    const auto newMapping =
        remapWithOffset(outMapping, outMapping.size(), offset, true);
    graph.setTileMapping(out, newMapping);
  }
  if (logging::popops::shouldLog(logging::Level::Debug)) {
    const auto summary = getMappingSummary(graph.getTileMapping(out));
    logging::popops::debug("  Tensor:{}({}):{}", out.shape(), out.elementType(),
                           out.getDebugStr());
    logging::popops::debug("  TensorVar:{}", out.getVarStr());

    logging::popops::debug("  Summary: Tiles:[{}, {}) Used:{} MeanPerTile:{}",
                           summary.first, summary.last, summary.used,
                           static_cast<float>(out.numElements()) /
                               summary.used);
  }

  di.addOutput(out);
  return out;
}

unsigned transformTileIndex(unsigned tile, unsigned numTiles, unsigned offset,
                            bool ascending) {
  if (ascending) {
    return (numTiles + tile - offset) % numTiles;
  } else {
    return (2 * numTiles - offset - tile) % numTiles;
  }
}

unsigned invTransformTileIndex(unsigned tile, unsigned numTiles,
                               unsigned offset, bool ascending) {
  if (ascending) {
    return (tile + offset) % numTiles;
  } else {
    return (2 * numTiles - offset - tile) % numTiles;
  }
}

} // namespace poputil
