// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popops/Rearrange.hpp"

#include "poplibs_support/Tracepoint.hpp"
#include <boost/icl/interval_map.hpp>
#include <boost/optional.hpp>
#include <limits>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplibs_support/logging.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/Util.hpp>
#include <poputil/VarStructure.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poputil;

static inline Tensor groupTensorAux(const Tensor &t, unsigned rank) {
  return t;
}
static inline Tensor ungroupTensorAux(const Tensor &t, unsigned) { return t; }

template <typename... G>
static inline Tensor groupTensorAux(const Tensor &t, unsigned rank,
                                    const GroupingInfo &g, G &&...gs) {
  return groupTensorAux(t.reshapePartial(g.first, g.first + 1,
                                         {t.dim(g.first) / g.second, g.second})
                            .dimRoll(g.first + 1, rank),
                        rank + 1, std::forward<G>(gs)...);
}

template <typename... G>
static inline Tensor ungroupTensorAux(const Tensor &t, unsigned rank,
                                      const GroupingInfo &g, G &&...gs) {
  return ungroupTensorAux(
      t.dimRoll(rank, g.first + 1).flatten(g.first, g.first + 2), rank,
      std::forward<G>(gs)...);
}

template <typename... G>
static inline Tensor groupTensor(const Tensor &t, G &&...gs) {
  return groupTensorAux(t, t.rank(), std::forward<G>(gs)...);
}

template <typename... G>
static inline Tensor ungroupTensor(const Tensor &t, G &&...gs) {
  return ungroupTensorAux(t, unsigned(t.rank() - sizeof...(gs)),
                          std::forward<G>(gs)...);
}

namespace popops {
namespace rearrange {

static std::vector<Type> getValidTransposeDataTypes() {
  return {HALF, FLOAT, UNSIGNED_SHORT, UNSIGNED_INT, SHORT, INT};
}

bool canUseFastTranspose(const poplar::Target &target, const poplar::Type &type,
                         unsigned numRows, unsigned numColumns,
                         unsigned numTranspositions) {

  bool is2ByteType = (type == HALF || type == UNSIGNED_SHORT || type == SHORT);
  if (!is2ByteType ||
      numTranspositions > std::numeric_limits<unsigned short>::max() ||
      numRows % 4 || numColumns % 4) {
    return false;
  }
  // Check machine limits
  if (numColumns == 4 && numRows == 4) {
    if ((numTranspositions >= 2) &&
        (numTranspositions - 2 > target.getRptCountMax()))
      return false;
  } else if (numColumns == 4) {
    if (((numRows >= 8) && (numRows / 4 - 2 > target.getRptCountMax())) ||
        (numRows / 4u * 3u - 1u > (1u << (target.getNumStrideBits() - 1u)))) {
      return false;
    }
  } else {
    if (((numColumns >= 8) && (numColumns / 4 - 2 > target.getRptCountMax())) ||
        (numColumns / 4u * 3u - 1u >
         (1u << (target.getNumStrideBits() - 1u))) ||
        (numRows / 4u >= (1u << (target.getNumStrideBits() - 1)))) {
      return false;
    }
  }
  return true;
}

static bool switchToWorkerTranspose(const unsigned numTileTranspositions,
                                    const unsigned rows, const unsigned cols) {
  return (numTileTranspositions == 1) ||
         ((rows == 4) && (cols == 4) && (numTileTranspositions <= 4));
}

void addTransposeVertices(
    Graph &graph, const ComputeSet &cs, const Type &dType, unsigned rows,
    unsigned cols, const poplar::Graph::TileToTensorMapping &mapping,
    std::function<std::pair<const poplar::Tensor, const poplar::Tensor>(size_t)>
        getInOut,
    const DebugContext &debugContext) {
  const auto &validTypes = getValidTransposeDataTypes();
  if (std::find(validTypes.begin(), validTypes.end(), dType) ==
      validTypes.end()) {
    throw poplibs_error("Transposition not supported for data type " +
                        dType.toString());
  }
  if (cols > std::numeric_limits<unsigned short>::max() ||
      rows > std::numeric_limits<unsigned short>::max()) {
    throw poplibs_error(
        "Number of source rows and columns exceed sizes "
        "supported by Transpose1DSingleWorker/Transpose2D vertex");
  }
  // Shorthand local function to accumulate total size of a vector of Intervals
  auto accumSize = [](const std::vector<Interval> &vi) {
    return std::accumulate(
        vi.begin(), vi.end(), 0,
        [](size_t acc, const Interval &i) { return acc + i.size(); });
  };
  const auto &target = graph.getTarget();
  for (unsigned tile = 0; tile != mapping.size(); ++tile) {
    // All the transpositions to do on this tile. This is a vector of intervals,
    // each one specifying a set of transpositions.
    const auto &tileTranspositions = mapping[tile];

    // How many transpositions in all for this tile?
    unsigned numTileTranspositions = accumSize(tileTranspositions);

    logging::popops::trace("addTransposeVertices: debugStr {}, tile {} "
                           "numTranspositions {}, rows {}, cols {}",
                           debugContext.getPathName(), tile,
                           numTileTranspositions, rows, cols);
    if (numTileTranspositions > 0) {

      // There are 3 types of vertices that we might use. Default is MultiVertex
      enum VertexType { Transpose1D, Transpose1DSingleWorker, Transpose2D };
      std::map<VertexType, std::string> vertexNames = {
          {Transpose1D, "popops::Transpose1D"},
          {Transpose1DSingleWorker, "popops::Transpose1DSingleWorker"},
          {Transpose2D, "popops::Transpose2D"},
      };
      VertexType vertexType = Transpose1D;
      // Will we end up splitting among workers (if not 1D MultiVertex)?
      bool splitToWorkers = false;
      // Can we really use the 1D MultiVertex to do them all?
      if (canUseFastTranspose(target, dType, rows, cols,
                              numTileTranspositions)) {
        // If we have to do a single matrix (of any size), it's faster to run
        // the 'plain' Transpose instead of Transpose1D.
        // Same is true if we have up to four 4x4 matrix
        if (switchToWorkerTranspose(numTileTranspositions, rows, cols)) {
          vertexType = Transpose1DSingleWorker;
        }
      } else {
        // Will need to partition to workers. vertexType will be chosen later
        splitToWorkers = true;
      }

      // Local function (as a lambda) to add a vertex for 'tile'.
      //     vType:          What kind of vertex to use
      //     tile:           where the vertex must be mapped
      //     transpositions: all transpositions this vertex has to do
      auto addOneVertex = [&](VertexType vType, unsigned tile,
                              std::vector<poplar::Interval> transpositions) {
        // Build inVec[], outVec[] to contain one element for each transposition
        std::vector<poplar::Tensor> inVec, outVec;
        for (const auto &interval : transpositions) {
          for (auto transposition = interval.begin();
               transposition != interval.end(); ++transposition) {
            poplar::Tensor in, out;
            std::tie(in, out) = getInOut(transposition);
            inVec.push_back(in);
            outVec.push_back(out);
            graph.setTileMapping(out, tile);
          }
        }
        std::string vertexName = vertexNames[vType];
        const auto v = graph.addVertex(cs, templateVertex(vertexName, dType));

        graph.setTileMapping(v, tile);
        if ((vType == Transpose1DSingleWorker) || (vType == Transpose1D)) {
          graph.connect(v["src"], concat(inVec));
          graph.connect(v["dst"], concat(outVec));
          graph.setInitialValue(v["numSrcColumnsD4"], cols / 4);
          graph.setInitialValue(v["numSrcRowsD4"], rows / 4);
          if (vType == Transpose1DSingleWorker) {
            graph.setInitialValue(v["numTranspositionsM1"], inVec.size() - 1);
          } else {
            // We will run one 1D MultiVertex vertex, starting the 6 workers.
            // The first 'workerCount' workers (1<=workerCount<=6) will
            // transpose 'numTranspositions' matrices and (6-workerCount)
            // workers transposing (numTranspositions-1) matrices.
            // Note that (6-workerCount) and/or (numTranspositions-1) might
            // be zero.
            // Note that this is NOT the same split as
            // splitRegionsBetweenWorkers() would do.
            unsigned numWorkerContexts = target.getNumWorkerContexts();
            unsigned workerCount = numWorkerContexts, numTranspositions = 1;
            if (numTileTranspositions <= numWorkerContexts) {
              workerCount = numTileTranspositions;
            } else {
              numTranspositions = numTileTranspositions / workerCount;
              unsigned rem = numTileTranspositions % workerCount;
              if (rem > 0) {
                workerCount = rem;
                numTranspositions += 1;
              }
            }
            graph.setInitialValue(v["numTranspositions"], numTranspositions);
            graph.setInitialValue(v["workerCount"], workerCount);
          }
        } else {
          graph.connect(v["src"], inVec);
          graph.connect(v["dst"], outVec);
          graph.setInitialValue(v["numSrcColumns"], cols);
          graph.setInitialValue(v["numSrcRows"], rows);
        }
      }; // addOneVertex()
      if (!splitToWorkers) {
        // A single vertex will do all the transpositions for this
        // tile
        addOneVertex(vertexType, tile, tileTranspositions);
      } else {
        // Need to split to multiple workers on this tile
        auto perWorkerTranspositions =
            splitRegionsBetweenWorkers(target, tileTranspositions, 1);
        for (const auto &transpositions : perWorkerTranspositions) {
          size_t size = accumSize(transpositions);
          vertexType = canUseFastTranspose(target, dType, rows, cols, size)
                           ? Transpose1DSingleWorker
                           : Transpose2D;
          addOneVertex(vertexType, tile, transpositions);
        } // for each worker
      }   // cannot use 1D MultiVertex variant
    }     // if (numTileTranspositions>0)
  }       // for each tile
}

Tensor partialTranspose(Graph &graph, const Tensor &in, const ComputeSet &cs,
                        const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(in, cs));

  const auto rank = in.rank();
  const auto numSrcRows = in.dim(rank - 2);
  const auto numSrcColumns = in.dim(rank - 1);

  // Get a view on the 'in' tensor that is just a 2D matrix where each
  // row is one of the matrices (the 'rightmost' 2 dimensions) to transpose
  // ('inFlat').
  // I.e. flatten all the leftmost N-2 dimension together and also the last 2
  // dimensions together.
  // Get an equivalent view for the 'out' tensor ('outFlat').
  const auto dType = in.elementType();
  auto outShape = in.shape();
  std::swap(outShape[rank - 2], outShape[rank - 1]);
  auto out = graph.addVariable(dType, outShape, {di, "partialTranspose"});
  auto inFlat = in.reshape({in.numElements() / (numSrcRows * numSrcColumns),
                            numSrcRows * numSrcColumns});
  auto outFlat = out.reshape(inFlat.shape());

  // Get the tile mapping for the first element of each 2D matrix to transpose
  // (i.e. the rows of 'inFlat').
  // This tile is where we will allocate the Transpose vertices and the
  // output transposed matrix.
  const auto transpositionMapping = graph.getTileMapping(inFlat.slice(0, 1, 1));

  addTransposeVertices(
      graph, cs, dType, numSrcRows, numSrcColumns, transpositionMapping,
      [&](size_t index) {
        return std::make_pair(inFlat[index], outFlat[index]);
      },
      debugContext);
  di.addOutput(out);
  return out;
}

unsigned getMinimumRegroupGrainSize(const Type &type) {
  bool is2ByteType = (type == HALF || type == UNSIGNED_SHORT || type == SHORT);
  bool is4ByteType = (type == FLOAT || type == UNSIGNED_INT || type == INT);
  if (is2ByteType) {
    return 4;
  } else if (is4ByteType) {
    return 2;
  }
  return 1;
}

// Returns an updated grouping based on original grouping and tile mapping
static std::tuple<GroupingInfo, GroupingInfo, Graph::TileToTensorMapping>
updateGroupingInternal(const Graph &graph, const Tensor &t,
                       const GroupingInfo &from, const GroupingInfo &to) {
  auto grouped = groupTensor(t, to, from);
  auto groupedFlat = grouped.flatten(0, grouped.rank() - 2).flatten(1, 3);
  const auto tMapping = graph.getTileMapping(groupedFlat);
  const auto numTiles = tMapping.size();
  const auto tilesPerIPU = graph.getTarget().getTilesPerIPU();
  const auto numWorkers = graph.getTarget().getNumWorkerContexts();
  const auto numIPUs = numTiles / tilesPerIPU;
  std::vector<std::size_t> elemsPerIpu(numIPUs);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto ipu = tile / tilesPerIPU;
    const auto &mapping = tMapping[tile];
    if (!mapping.empty()) {
      for (const auto &r : mapping) {
        elemsPerIpu[ipu] += r.size();
      }
    }
  }

  // Minimum number of elements in a group. Groups are split by a multiple of
  // this
  const unsigned minGroupsSize = getMinimumRegroupGrainSize(t.elementType());

  // find entry with max elements
  auto maxIt = std::max_element(std::begin(elemsPerIpu), std::end(elemsPerIpu));

  // A factor by which the number of transposes can increase by without
  // breaking the constraints set on group size
  auto additionalFactor = groupedFlat.dim(1) / (minGroupsSize * minGroupsSize);

  // This limits the number of transpositions allowed on the IPU
  const auto maxTranspositionsAllowedPerIpu = tilesPerIPU * numWorkers;

  // Estimate the number of transpositions on the IPU which has the maximum
  // number of elements mapped
  auto transpositionsOnIpuEstimate =
      (*maxIt + groupedFlat.dim(1) - 1) / groupedFlat.dim(1);

  // Sorted list of factors of a number
  auto sortedFactors = [](unsigned number) {
    std::vector<unsigned> result;
    for (unsigned x = 1; x * x <= number; ++x) {
      if (number % x == 0) {
        result.push_back(x);
        if (number != x * x) {
          result.push_back(number / x);
        }
      }
    }
    std::sort(result.begin(), result.end());
    return result;
  };

  bool allowIncrease =
      to.second % minGroupsSize == 0 && from.second % minGroupsSize == 0;

  // actual transpose factor used. Initialise with 1 which means no additional
  // factor is applied
  unsigned transposeFactor = 1;
  if (allowIncrease) {
    std::size_t bestCost = std::numeric_limits<std::size_t>::max();
    for (const auto x : sortedFactors(additionalFactor)) {
      if (transpositionsOnIpuEstimate * x > maxTranspositionsAllowedPerIpu) {
        break;
      }
      // Ideally we want to use cycles estimates, but we use a simplified,
      // inaccurate cost. Assume cost for the initial transposition to be
      // additionalFactor. Increasing transposeFactor decreases the size of the
      // transposition by that factor. As additionalFactor is an integer
      // multiple of x, we can divide it with no loss of information.
      auto cost = (additionalFactor / x) *
                  ceildiv(ceildiv(transpositionsOnIpuEstimate * x, tilesPerIPU),
                          numWorkers);
      if (cost < bestCost) {
        bestCost = cost;
        transposeFactor = x;
      }
    }
  }

  auto updatedFrom = from;
  auto updatedTo = to;

  if (transposeFactor != 1) {
    // TODO: T12893 Optimise split once the cost of using a 1D MultiVertex
    // is known.
    auto factorFrom = gcd(transposeFactor, from.second / minGroupsSize);
    transposeFactor /= factorFrom;
    auto factorTo = gcd(transposeFactor, to.second / minGroupsSize);
    updatedFrom.second /= factorFrom;
    updatedTo.second /= factorTo;
  }
  return std::make_tuple(updatedFrom, updatedTo, std::move(tMapping));
}

std::pair<GroupingInfo, GroupingInfo> updateGrouping(const Graph &graph,
                                                     const Tensor &t,
                                                     const GroupingInfo &from,
                                                     const GroupingInfo &to) {
  const auto result = updateGroupingInternal(graph, t, from, to);
  return std::make_pair(std::get<0>(result), std::get<1>(result));
}

Tensor regroupTensorInternal(Graph &graph, const Tensor &t,
                             std::vector<Copy> &copies,
                             const ComputeSet &transposeCS,
                             const GroupingInfo &from_, const GroupingInfo &to_,
                             const DebugNameAndId &dnai) {
  const std::string fnStr = "internal";
  logging::popops::debug("Regroup: debugstr={}", dnai.getPathName());
  logging::popops::debug("  t      shape={}", t.shape());
  logging::popops::debug("  from   grouping={{{},{}}}", from_.first,
                         from_.second);
  logging::popops::debug("  to     grouping={{{},{}}}", to_.first, to_.second);

  if (t.rank() <= 1) {
    return t;
  }

  const auto &validTypes = getValidTransposeDataTypes();
  if (std::find(validTypes.begin(), validTypes.end(), t.elementType()) ==
      validTypes.end()) {
    throw poplibs_error("Data type " + t.elementType().toString() +
                        " not supported by regroupTensor");
  }
  if (from_.first >= t.rank()) {
    throw poplibs_error("Grouping 'from' specifies dimension " +
                        std::to_string(from_.first) +
                        " which is invalid for tensor to regroup of rank " +
                        std::to_string(t.rank()));
  }
  if (to_.first >= t.rank()) {
    throw poplibs_error("Grouping 'to' specifies dimension " +
                        std::to_string(to_.first) +
                        " which is invalid for tensor to regroup of rank " +
                        std::to_string(t.rank()));
  }
  if (t.dim(from_.first) % from_.second) {
    throw poplibs_error(
        "Size of grouping 'from' " + std::to_string(from_.second) +
        " does not evenly divide dimension " + std::to_string(from_.first) +
        " with size " + std::to_string(t.dim(from_.first)) +
        " of the tensor to regroup");
  }
  if (t.dim(to_.first) % to_.second) {
    throw poplibs_error("Size of grouping 'to' " + std::to_string(to_.second) +
                        " does not evenly divide dimension " +
                        std::to_string(to_.first) + " with size " +
                        std::to_string(t.dim(to_.first)) +
                        " of the tensor to regroup");
  }
  GroupingInfo to, from;
  Graph::TileToTensorMapping tMapping;
  std::tie(from, to, tMapping) = updateGroupingInternal(graph, t, from_, to_);
  auto grouped = groupTensor(t, to, from);
  auto groupedFlat = grouped.flatten(0, grouped.rank() - 2).flatten(1, 3);

  if (!(from == from_ && to == to_)) {
    tMapping = graph.getTileMapping(groupedFlat);
  }

  // Explicitly copy to a single variable in order to force
  // regions to be contiguous. Performing a transpose alone
  // may leave multiple regions per-tile, one for each edge to a
  // transpose vertex.
  auto preRegroup =
      graph.addVariable(t.elementType(), grouped.shape(), {dnai, "preRegroup"});
  auto preRegroupTranspose = preRegroup.flatten(0, preRegroup.rank() - 2);
  auto preRegroupFlat =
      preRegroup.flatten(0, preRegroup.rank() - 2).flatten(1, 3);

  // Build a map giving which intervals are mapped to each
  // IPU. Track which tiles on each IPU have any elements
  // mapped.
  const auto numTiles = tMapping.size();
  const auto tilesPerIPU = graph.getTarget().getTilesPerIPU();
  const auto numIPUs = numTiles / tilesPerIPU;

  using IntervalMap = boost::icl::interval_map<std::size_t, unsigned,
                                               boost::icl::partial_enricher>;
  using Interval = boost::icl::interval<std::size_t>;
  IntervalMap intervalsToIPU;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto ipu = tile / tilesPerIPU;
    const auto &mapping = tMapping[tile];
    if (!mapping.empty()) {
      for (const auto &i : mapping) {
        intervalsToIPU.insert(
            std::make_pair(Interval::right_open(i.begin(), i.end()), ipu));
      }
    }
  }

  // Iterate each transposition, mapping this to an IPU based on the first
  // element in each.
  auto elemsPerTransposition = preRegroupFlat.dim(1);
  std::vector<std::vector<poplar::Interval>> ipuTranspositions(numIPUs);
  for (unsigned t = 0; t < preRegroupFlat.dim(0); ++t) {
    auto it = intervalsToIPU.find(Interval::right_open(
        t * elemsPerTransposition, t * elemsPerTransposition + 1));
    assert(it != intervalsToIPU.end());
    auto ipu = it->second;
    auto &ipuTs = ipuTranspositions[ipu];
    // Try and extend the previous region if possible
    if (!ipuTs.empty() && ipuTs.back().end() == t) {
      ipuTs.back() = poplar::Interval(ipuTs.back().begin(), t + 1);
    } else {
      ipuTs.emplace_back(t, t + 1);
    }
  }

  // Finally map slices of the new tensor to transpose mapped linearly
  // across the tiles on which the original tensor was mapped on the same
  // IPU the elements of the transposition were originally mapped to.
  //
  // FIXME: This currently allows external exchange to be incurred for a
  // given transposition. This should not be allowed as it is not expected
  // but for the timebeing the padding constants added to activations
  // are just mapped to tile 0 which can be a different IPU to the one
  // on which it should actually reside. T6427 is required for this as a
  // primary usage of these regrouping functions.
  for (unsigned ipu = 0; ipu < numIPUs; ++ipu) {
    const auto &transpositions = ipuTranspositions[ipu];
    auto numTranspositions = std::accumulate(
        transpositions.begin(), transpositions.end(), std::size_t(0),
        [](std::size_t t, const poplar::Interval &i) { return t + i.size(); });
    if (!numTranspositions)
      continue;

    // spread across all tiles
    auto numTiles = tilesPerIPU;

    // Map transpositions on this IPU evenly across the tiles on which
    // elements of the source tensor reside.
    auto transpositionsPerTile = (numTranspositions + numTiles - 1) / numTiles;
    auto interval = transpositions.begin();
    unsigned intervalOffset = 0;
    for (unsigned i = 0; i < numTiles; ++i) {
      auto remaining = std::min(transpositionsPerTile, numTranspositions);
      numTranspositions -= remaining;
      while (remaining > 0) {
        auto n = std::min(interval->size() - intervalOffset, remaining);
        auto slice =
            preRegroupFlat.slice(interval->begin() + intervalOffset,
                                 interval->begin() + intervalOffset + n, 0);
        graph.setTileMapping(slice, i);
        remaining -= n;
        intervalOffset += n;
        if (interval->begin() + intervalOffset == interval->end()) {
          ++interval;
          intervalOffset = 0;
        }
      }
    }
  }

  copies.emplace_back(grouped, preRegroup, false, DebugContext(dnai, fnStr));

  // Finally, transpose
  auto partiallyTransposed = popops::rearrange::partialTranspose(
      graph, preRegroup, transposeCS, {dnai, fnStr});

  auto output = ungroupTensor(partiallyTransposed, from, to);
  return output;
}

Tensor regroupTensor(Graph &graph, const Tensor &t, std::vector<Copy> &copies,
                     const ComputeSet &transposeCS, const GroupingInfo &from_,
                     const GroupingInfo &to_,
                     const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(t, copies, transposeCS, from_, to_));
  auto output =
      regroupTensorInternal(graph, t, copies, transposeCS, from_, to_, {di});

  di.addOutput(output);
  return output;
}

Tensor regroupTensor(Graph &graph, const Tensor &t_,
                     poplar::program::Sequence &prog,
                     const ComputeSet &transposeCS, const GroupingInfo &from_,
                     const GroupingInfo &to_,
                     const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(t_, transposeCS, from_, to_));
  std::vector<Copy> copies;
  auto t = regroupTensor(graph, t_, copies, transposeCS, from_, to_, {di});

  for (const auto &copy : copies) {
    prog.add(copy);
  }
  di.addOutput(t);
  return t;
}

Tensor regroupIfPossible(Graph &graph, const Tensor &t_,
                         poplar::program::Sequence &prog,
                         const GroupingInfo &to_,
                         const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t_, to_));

  const auto groupings = detectDimGroupings(graph, t_);
  logging::popops::debug("Regroup if possible {}", debugContext.getPathName());
  logging::popops::debug("  shape {}, groupings {}", t_.shape(), groupings);

  const auto invalid = (t_.dim(to_.first) % to_.second) || (t_.rank() <= 1) ||
                       (to_.first >= t_.rank());
  // If the innermost grouping is already the dimension for which regrouping
  // is requested, nothing needs to be done.
  const auto noRegroupingRequired =
      !groupings.empty() && std::get<0>(groupings.at(0)) == std::get<0>(to_);

  const auto grainSize = getMinimumRegroupGrainSize(t_.elementType());

  // Only check for grouping in the innermost dimension. If the innermost
  // dimension is the `to be regrouped` dimension, noRegroupingRequired = true
  // and no regrouping is performed. If the innermost grouping has the
  // requisite grain size and is not the `to be regrouped` dimension we use it
  // up as the from dimension.
  auto suitableGroupingFound =
      !groupings.empty() && std::get<1>(groupings.at(0)) % grainSize == 0;

  if (invalid || noRegroupingRequired || !suitableGroupingFound) {
    logging::popops::debug("  Regrouping not possible: invalid ? {}, "
                           "noRegroupingRequired {}, suitableGroupingFound {}",
                           invalid, noRegroupingRequired,
                           suitableGroupingFound);
    di.addOutput(t_);
    return t_;
  }

  std::vector<Copy> copies;
  auto transposeCS =
      graph.addComputeSet(debugContext.getPathName() + "/regroupTensor");
  auto t = regroupTensorInternal(graph, t_, copies, transposeCS,
                                 groupings.at(0), to_, {di});
  for (const auto &copy : copies) {
    prog.add(copy);
  }
  prog.add(Execute(transposeCS));
  di.addOutput(t);
  return t;
}

Tensor regroupIfBeneficial(Graph &graph, const Tensor &in_, const Tensor &ref,
                           std::vector<Copy> &preTranspose,
                           ComputeSet transposeCS,
                           const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(in_, ref, preTranspose, transposeCS));

  logging::popops::debug("Regroup if beneficial: debugstr={}",
                         debugContext.getPathName());
  logging::popops::debug("  input      shape={}", in_.shape());
  logging::popops::debug("  reference  shape={}", ref.shape());
  auto in = in_;

  if (in.rank() <= 1 || ref.rank() <= 1) {
    return in;
  }

  // If we can't transpose this data type, it's not beneficial to try
  const auto &transposableTypes = getValidTransposeDataTypes();
  if (std::find(transposableTypes.begin(), transposableTypes.end(),
                in.elementType()) == transposableTypes.end()) {
    return in;
  }

  if (in.shape() != ref.shape()) {
    throw poplibs_error("Input and reference tensors should be of "
                        "the same shape");
  }

  const auto inGrouping = detectDimGroupings(graph, in);
  const auto refGrouping = detectDimGroupings(graph, ref);
  logging::popops::debug("  input      groupings={}", inGrouping);
  logging::popops::debug("  reference  groupings={}", refGrouping);

  // TODO: T10360 Consider avoiding regrouping float inputs.
  auto grainSize = getMinimumRegroupGrainSize(in.elementType());
  if (!inGrouping.empty() && !refGrouping.empty() &&
      inGrouping[0].first != refGrouping[0].first &&
      (inGrouping[0].second % grainSize) == 0 &&
      (refGrouping[0].second % grainSize) == 0) {
    logging::popops::debug("  regrouped");
    in = regroupTensorInternal(graph, in, preTranspose, transposeCS,
                               inGrouping[0], refGrouping[0], {di});
  }
  di.addOutput(in);
  return in;
}

Tensor regroupIfBeneficial(Graph &graph, const Tensor &in_, const Tensor &ref,
                           Sequence &prog,
                           const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(in_, ref));

  std::vector<Copy> preTranspose;
  ComputeSet transposeCS = graph.addComputeSet({di, "Transpose"});

  auto in =
      regroupIfBeneficial(graph, in_, ref, preTranspose, transposeCS, {di});

  for (const auto &copy : preTranspose) {
    prog.add(copy);
  }
  prog.add(Execute(transposeCS, {di}));
  di.addOutput(in);
  return in;
}

Tensor regroupIfBeneficial(Graph &graph, const Tensor &in_,
                           std::size_t preferredGrouping_, Sequence &prog,
                           const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(in_, preferredGrouping_));
  logging::popops::debug("Regroup if beneficial (preferred): debugstr={}",
                         debugContext.getPathName());
  logging::popops::debug("  input        shape={}", in_.shape());
  logging::popops::debug("  preferred grouping={}", preferredGrouping_);
  auto in = in_;

  // If we can't transpose this data type, it's not beneficial to try
  const auto &transposableTypes = getValidTransposeDataTypes();
  if (std::find(transposableTypes.begin(), transposableTypes.end(),
                in.elementType()) == transposableTypes.end()) {
    return in;
  }

  if (in.dim(in.rank() - 1) % preferredGrouping_ != 0) {
    throw poplibs_error("Input tensor's innermost dimension is not "
                        "divisible by the given preferred grouping (" +
                        std::to_string(preferredGrouping_) + ")");
  }

  const auto inGrouping = detectDimGroupings(graph, in);
  logging::popops::debug("  input     grouping={}", inGrouping);
  const auto preferredGrouping =
      GroupingInfo{in.rank() - 1, preferredGrouping_};

  // TODO: T10360 Consider avoiding regrouping float inputs.
  auto grainSize = getMinimumRegroupGrainSize(in.elementType());
  logging::popops::trace(
      "beneficialRegroup decision: !empty {}, groupingChange "
      "{}, in matches grainsize {}, out matches grainsize {} "
      "(grainSize {})",
      !inGrouping.empty(),
      inGrouping.empty() ? 0 : inGrouping[0].first != preferredGrouping.first,
      inGrouping.empty() ? 0 : inGrouping[0].second % grainSize == 0,
      preferredGrouping.second % grainSize == 0, grainSize);
  if (!inGrouping.empty() && inGrouping[0].first != preferredGrouping.first &&
      inGrouping[0].second % grainSize == 0 &&
      preferredGrouping.second % grainSize == 0) {
    logging::popops::debug("  regrouped");
    ComputeSet transposeCS = graph.addComputeSet({di, "Transpose"});
    in = regroupTensor(graph, in, prog, transposeCS, inGrouping[0],
                       preferredGrouping, {di});
    prog.add(Execute(transposeCS, {di}));
  }

  return in;
}

} // end namespace rearrange
} // end namespace popops
