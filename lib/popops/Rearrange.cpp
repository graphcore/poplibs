// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popops/Rearrange.hpp"

#include <boost/icl/interval_map.hpp>
#include <boost/optional.hpp>

#include <poplibs_support/gcd.hpp>
#include <poplibs_support/logging.hpp>

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
                                    const GroupingInfo &g, G &&... gs) {
  return groupTensorAux(t.reshapePartial(g.first, g.first + 1,
                                         {t.dim(g.first) / g.second, g.second})
                            .dimRoll(g.first + 1, rank),
                        rank + 1, std::forward<G>(gs)...);
}

template <typename... G>
static inline Tensor ungroupTensorAux(const Tensor &t, unsigned rank,
                                      const GroupingInfo &g, G &&... gs) {
  return ungroupTensorAux(
      t.dimRoll(rank, g.first + 1).flatten(g.first, g.first + 2), rank,
      std::forward<G>(gs)...);
}

template <typename... G>
static inline Tensor groupTensor(const Tensor &t, G &&... gs) {
  return groupTensorAux(t, t.rank(), std::forward<G>(gs)...);
}

template <typename... G>
static inline Tensor ungroupTensor(const Tensor &t, G &&... gs) {
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
         (1u << (target.getNumStrideBits() - 1u)))) {
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
        getInOut) {
  const auto &validTypes = getValidTransposeDataTypes();
  if (std::find(validTypes.begin(), validTypes.end(), dType) ==
      validTypes.end()) {
    throw poplibs_error("Transposition not supported for data type " +
                        dType.toString());
  }
  if (cols > std::numeric_limits<unsigned short>::max() ||
      rows > std::numeric_limits<unsigned short>::max()) {
    throw poplibs_error("Number of source rows and columns exceed sizes "
                        "supported by Transpose/Transpose2d vertex");
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
    if (numTileTranspositions > 0) {

      // There are 3 types of vertices that we might use. Default is Supervisor
      enum VertexType { TransposeSupervisor, Transpose, Transpose2d };
      std::map<VertexType, std::string> vertexNames = {
          {TransposeSupervisor, "popops::TransposeSupervisor"},
          {Transpose, "popops::Transpose"},
          {Transpose2d, "popops::Transpose2d"},
      };
      VertexType vertexType = TransposeSupervisor;
      // Will we end up splitting among workers (if not supervisor)?
      bool splitToWorkers = false;
      // Can we really use the Supervisor Vertex to do them all?
      if (canUseFastTranspose(target, dType, rows, cols,
                              numTileTranspositions)) {
        // If we have to do a single matrix (of any size), it's faster to run
        // the 'plain' Transpose instead of TransposeSupervisor.
        // Same is true if we have up to four 4x4 matrix
        if (switchToWorkerTranspose(numTileTranspositions, rows, cols)) {
          vertexType = Transpose;
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
        if ((vType == Transpose) || (vType == TransposeSupervisor)) {
          graph.connect(v["src"], concat(inVec));
          graph.connect(v["dst"], concat(outVec));
          graph.setInitialValue(v["numSrcColumnsD4"], cols / 4);
          graph.setInitialValue(v["numSrcRowsD4"], rows / 4);
          if (vType == Transpose) {
            graph.setInitialValue(v["numTranspositionsM1"], inVec.size() - 1);
          } else {
            // We will run one supervisor vertex, starting the 6 workers.
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
                           ? Transpose
                           : Transpose2d;
          addOneVertex(vertexType, tile, transpositions);
        } // for each worker
      }   // cannot use Supervisor variant
    }     // if (numTileTranspositions>0)
  }       // for each tile
}

Tensor partialTranspose(Graph &graph, const Tensor &in, const ComputeSet &cs,
                        const std::string &debugPrefix) {
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
  auto out =
      graph.addVariable(dType, outShape, debugPrefix + "/partialTranspose");
  auto inFlat = in.reshape({in.numElements() / (numSrcRows * numSrcColumns),
                            numSrcRows * numSrcColumns});
  auto outFlat = out.reshape(inFlat.shape());

  // Get the tile mapping for the first element of each 2D matrix to transpose
  // (i.e. the rows of 'inFlat').
  // This tile is where we will allocate the Transpose vertices and the
  // output transposed matrix.
  const auto transpositionMapping = graph.getTileMapping(inFlat.slice(0, 1, 1));

  addTransposeVertices(graph, cs, dType, numSrcRows, numSrcColumns,
                       transpositionMapping, [&](size_t index) {
                         return std::make_pair(inFlat[index], outFlat[index]);
                       });
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
  const unsigned minGroupsSize = 4;

  // find entry with max elements
  auto maxIt = std::max_element(std::begin(elemsPerIpu), std::end(elemsPerIpu));

  // A factor by which the number of transposes can increase by without
  // breaking the constraints set on group size
  auto additionalFactor = groupedFlat.dim(1) / (minGroupsSize * minGroupsSize);

  auto isPrime = [](unsigned num) {
    for (unsigned i = 2; i <= num / 2; ++i) {
      if (num % i == 0) {
        return false;
      }
    }
    return true;
  };

  // This limits the number of transpositions allowed on the IPU
  const auto maxTranspositionsAllowedPerIpu = numTiles;

  // actual transpose factor used. Initialise with 1 which means no additional
  // factor is applied
  unsigned transposeFactor = 1;

  // Estimate the number of transpositions on the IPU which has the maximum
  // number of elements mapped
  auto transpositionsOnIpuEstimate =
      (*maxIt + groupedFlat.dim(1) - 1) / groupedFlat.dim(1);

  bool allowIncrease =
      to.second % minGroupsSize == 0 && from.second % minGroupsSize == 0;
  while (allowIncrease && additionalFactor != 1) {
    unsigned factor = 1;
    // TODO: T12892 This assumes that typical transposes are a multiple of very
    // small primes. Investigate other methods (e.g., dividing into prime
    // factors). A method that should give good results is to find the maximum
    // GCD across different values of transpositions (i.e.
    // maxTranspositionsAllowedPerIpu, maxTranspositionsAllowedPerIpu-1, ...)
    for (unsigned x = 2; x <= additionalFactor; ++x) {
      if (additionalFactor % x == 0 && isPrime(x)) {
        factor = x;
        break;
      }
    }
    if (transpositionsOnIpuEstimate * transposeFactor * factor >
        maxTranspositionsAllowedPerIpu) {
      break;
    }
    if (additionalFactor % factor != 0 || factor == 1) {
      throw poputil::poplibs_error("Invalid factor in regrouping");
    }
    transposeFactor *= factor;
    additionalFactor /= factor;
  }

  auto updatedFrom = from;
  auto updatedTo = to;

  if (transposeFactor != 1) {
    // TODO: T12893 Optimise split once the cost of using a supervisor vertex
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

Tensor regroupTensor(Graph &graph, const Tensor &t,
                     poplar::program::Sequence &copies,
                     const ComputeSet &transposeCS, const GroupingInfo &from_,
                     const GroupingInfo &to_, const std::string &debugPrefix) {
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
  auto preRegroup = graph.addVariable(t.elementType(), grouped.shape(),
                                      debugPrefix + "/preRegroup");
  auto preRegroupTranspose = preRegroup.flatten(0, preRegroup.rank() - 2);
  auto preRegroupFlat =
      preRegroup.flatten(0, preRegroup.rank() - 2).flatten(1, 3);

  // Build a map giving which intervals are mapped to each
  // IPU. Track which tiles on each IPU have any elements
  // mapped.
  const auto numTiles = tMapping.size();
  const auto tilesPerIPU = graph.getTarget().getTilesPerIPU();
  const auto numIPUs = numTiles / tilesPerIPU;

  std::vector<std::vector<unsigned>> mappedTilesByIPU(numIPUs);
  for (unsigned ipu = 0; ipu < numIPUs; ++ipu) {
    mappedTilesByIPU.reserve(tilesPerIPU);
  }
  using IntervalMap = boost::icl::interval_map<std::size_t, unsigned,
                                               boost::icl::partial_enricher>;
  using Interval = boost::icl::interval<std::size_t>;
  IntervalMap intervalsToIPU;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    auto ipu = tile / tilesPerIPU;
    const auto &mapping = tMapping[tile];
    if (!mapping.empty()) {
      mappedTilesByIPU[ipu].push_back(tile);
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
    const auto &mappedTiles = mappedTilesByIPU[ipu];
    const auto &transpositions = ipuTranspositions[ipu];
    auto numTiles = mappedTiles.size();
    auto numTranspositions = std::accumulate(
        transpositions.begin(), transpositions.end(), std::size_t(0),
        [](std::size_t t, const poplar::Interval &i) { return t + i.size(); });
    if (!numTranspositions)
      continue;

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
        graph.setTileMapping(slice, mappedTiles[i]);
        remaining -= n;
        intervalOffset += n;
        if (interval->begin() + intervalOffset == interval->end()) {
          ++interval;
          intervalOffset = 0;
        }
      }
    }
  }

  copies.add(program::Copy(grouped, preRegroup));

  // Finally, transpose
  auto partiallyTransposed = popops::rearrange::partialTranspose(
      graph, preRegroup, transposeCS, debugPrefix);

  return ungroupTensor(partiallyTransposed, from, to);
}

Tensor regroupIfBeneficial(Graph &graph, const Tensor &in_, const Tensor &ref,
                           Sequence &prog, const std::string &debugPrefix) {
  logging::debug("Regroup if beneficial: debugstr={}", debugPrefix);
  logging::debug("  input      shape={}", in_.shape());
  logging::debug("  reference  shape={}", ref.shape());
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
  logging::debug("  input      groupings={}", inGrouping);
  logging::debug("  reference  groupings={}", refGrouping);

  // TODO: T10360 Consider avoiding regrouping float inputs.
  auto grainSize = getMinimumRegroupGrainSize(in.elementType());
  if (!inGrouping.empty() && !refGrouping.empty() &&
      inGrouping[0].first != refGrouping[0].first &&
      (inGrouping[0].second % grainSize) == 0 &&
      (refGrouping[0].second % grainSize) == 0) {
    Sequence expandingCopies;
    ComputeSet transposeCS = graph.addComputeSet(debugPrefix + "/Transpose");
    in = regroupTensor(graph, in, expandingCopies, transposeCS, inGrouping[0],
                       refGrouping[0], debugPrefix);
    prog.add(expandingCopies);
    prog.add(Execute(transposeCS));
  }
  return in;
}

Tensor regroupIfBeneficial(Graph &graph, const Tensor &in_,
                           std::size_t preferredGrouping_, Sequence &prog,
                           const std::string &debugPrefix) {
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
  const auto preferredGrouping =
      GroupingInfo{in.rank() - 1, preferredGrouping_};

  // TODO: T10360 Consider avoiding regrouping float inputs.
  auto grainSize = getMinimumRegroupGrainSize(in.elementType());
  if (!inGrouping.empty() && inGrouping[0].first != preferredGrouping.first &&
      inGrouping[0].second % grainSize == 0 &&
      preferredGrouping.second % grainSize == 0) {
    ComputeSet transposeCS = graph.addComputeSet(debugPrefix + "/Transpose");
    in = regroupTensor(graph, in, prog, transposeCS, inGrouping[0],
                       preferredGrouping, debugPrefix);
    prog.add(Execute(transposeCS));
  }

  return in;
}

} // end namespace rearrange
} // end namespace popops
