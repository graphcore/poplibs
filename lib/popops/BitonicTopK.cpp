// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popops/Cast.hpp>
#include <popops/Encoding.hpp>
#include <popops/TopK.hpp>

#include <poputil/DebugInfo.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/VectorUtils.hpp>
#include <poplibs_support/gcd.hpp>
#include <poplibs_support/logging.hpp>
#include <poplibs_support/print.hpp>

#include <type_traits>
#include <unordered_map>

#include <boost/functional/hash.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poputil;

namespace poputil {

template <> poplar::ProfileValue toProfileValue(const popops::SortOrder &o) {
  switch (o) {
  case popops::SortOrder::NONE:
    return poplar::ProfileValue("NONE");
  case popops::SortOrder::ASCENDING:
    return poplar::ProfileValue("ASCENDING");
  case popops::SortOrder::DESCENDING:
    return poplar::ProfileValue("DESCENDING");
  default:
    return poplar::ProfileValue("<UNKNOWN>");
  }
}
} // namespace poputil

namespace popops {
namespace bitonic {

/** This returns the number of comparisons we will do at the given distance
 *  into a flat array with n elements.
 *
 *  e.g. n = 15 and distance = 4
 *
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *    +--------------->               +--------------->
 *        +--------------->               +--------------->
 *            +--------------->               +--------------->
 *                +--------------->
 *
 *  gives the number of comparisons as 7.
 */
template <typename T>
static inline T numComparisonsAtDistance(T n, T distance) {
  const auto q = floordiv(n, distance * 2);
  const auto r = n % (distance * 2);
  const auto comparisons = q * distance + r - std::min(r, distance);
  return comparisons;
}

namespace {

struct TensorKey {
  unsigned n;
  unsigned comparisons;
  unsigned distance;
  TensorKey(unsigned n, unsigned distance, unsigned nActive)
      : n(n), comparisons(numComparisonsAtDistance(nActive, distance)),
        distance(distance) {}
  inline bool operator<(const TensorKey &other) const {
    return std::tie(n, comparisons, distance) <
           std::tie(other.n, other.comparisons, other.distance);
  }
  inline bool operator==(const TensorKey &other) const {
    return std::tie(n, comparisons, distance) ==
           std::tie(other.n, other.comparisons, other.distance);
  }
};

struct TensorKeyHash {
  std::size_t operator()(const TensorKey &k) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, k.n);
    boost::hash_combine(seed, k.comparisons);
    boost::hash_combine(seed, k.distance);
    return seed;
  }
};

using TensorCache = std::unordered_map<TensorKey, Tensor, TensorKeyHash>;

} // end anonymous namespace

/** Allocates a tensor to be used when doing a compare and swap step
 *  with the given parameters. In particular this manages the correct
 *  mapping of elements to the tiles on which they will be processed
 *  and correct balancing of elements not processed across remaining
 *  tiles.
 *
 *  This is closely tied to the compareAndSwapAtDistance function.
 */
static Tensor allocate(Graph &graph, const Type &type, std::size_t n,
                       std::size_t nActive, std::size_t distance,
                       const DebugNameAndId &dnai) {
  const auto t = graph.addVariable(type, {n}, {dnai});
  const auto numTiles = graph.getTarget().getNumTiles();

  const auto comparisons = numComparisonsAtDistance(nActive, distance);
  // Allocation in each stage is done by first allocating to tiles elements
  // that are 'active' i.e. take part in a comparison at the given distance.
  const auto maxComparisonsPerTile = ceildiv(comparisons, numTiles);
  const auto activeTiles = ceildiv0(comparisons, maxComparisonsPerTile);

  // We know from the total number of elements n what the layout will be for
  // a tensor with distance equal to 1, which we use to determine how to
  // spread inactive elements over tiles.
  const auto maxActivePairs = floordiv(n, 2u);
  const auto maxAllocatedPairsPerTile = ceildiv(maxActivePairs, numTiles);
  const auto maxUsedTiles = ceildiv0(maxActivePairs, maxAllocatedPairsPerTile);

  // First allocate first comparisons * 2 elements to tiles according to
  // which will process those comparisons.
  for (unsigned tile = 0; tile < activeTiles; ++tile) {
    const auto begin = tile * maxComparisonsPerTile;
    const auto end = std::min((tile + 1) * maxComparisonsPerTile, comparisons);
    graph.setTileMapping(t.slice(begin * 2, end * 2), tile);
  }
  // Then allocate inactive elements to tiles, balancing total number of
  // elements allocated as much as possible.
  auto offset = comparisons * 2;
  auto inactiveRemaining = n - comparisons * 2;
  for (unsigned tile = 0; tile < maxUsedTiles; ++tile) {
    std::size_t activePairsThisTile = 0;
    if (tile < activeTiles) {
      const auto begin = tile * maxComparisonsPerTile;
      const auto end =
          std::min((tile + 1) * maxComparisonsPerTile, comparisons);
      activePairsThisTile = (end - begin);
    }
    const auto inactiveThisTile =
        std::min((maxAllocatedPairsPerTile - activePairsThisTile) * 2,
                 inactiveRemaining);
    graph.setTileMapping(t.slice(offset, offset + inactiveThisTile), tile);
    inactiveRemaining -= inactiveThisTile;
    offset += inactiveThisTile;
  }

  // If there is an odd element, we'll arbitrarily map it to the last tile.
  if (offset < n) {
    // We assume there is only ever 1 'odd' element
    assert(n - offset == 1);
    graph.setTileMapping(t.slice(offset, offset + 1),
                         std::max(maxUsedTiles, 1ul) - 1);
    offset += 1;
  }

  assert(offset == n);

  return t;
}

/** This gives the inverse of a permutation of elements of a tensor.
 */
static std::vector<Interval>
getInversePermutation(const std::vector<Interval> &is) {
  std::vector<Interval> matchingSlices;
  matchingSlices.reserve(is.size());
  std::size_t offset = 0;
  for (const auto &i : is) {
    matchingSlices.emplace_back(offset, offset + i.size());
    offset += i.size();
  }

  std::vector<std::size_t> sortedIndices(is.size());
  std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
  std::sort(sortedIndices.begin(), sortedIndices.end(),
            [&](const auto a, const auto b) {
              return is[a].begin() < is[b].begin();
            });

  std::vector<Interval> inverse;
  inverse.reserve(is.size());
  for (const auto &i : sortedIndices) {
    inverse.push_back(matchingSlices[i]);
  }
  return inverse;
}

/** Given an offset into a number of comparisons to do and the distance
 *  at which to compare, this gives the offset in elements for the lhs
 *  element taking part on that comparison in a flat array.
 *
 *  The calculation is essentially taking the number of multiples of
 *  the distance d * 2 to get number of elements plus any remainder.
 *
 *  e.g. for distance = 4 we have a flat array and the comparisons that
 *  will be done on that array
 *
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *    +--------------->               +--------------->
 *    0   +--------------->           4   +--------------->
 *        1   +--------------->           5
 *            2   +--------------->
 *                3
 *
 *  For comparison offset 3 we have no multiples of distance (4), plus
 *  the remainder 3, so the offset for the lhs element in that comparison
 *  is 3 as can be seen from the diagram. For comparison offset 5 we have
 *  1 multiple of 4, multiplied by 2 and add the remainder 1 equals 9 which
 *  again is evident from the diagram.
 */
template <typename T>
static inline T comparisonToElemOffset(T offset, T distance) {
  return roundDown(offset, distance) * 2 + offset % distance;
}

/** Given a tensor which was allocated with the given parameters, reorder
 *  it to give the canonical order. The canonical order is with all elements
 *  in the order they were given to the top-k operation.
 *
 *  We use the canonical order as a common ordering in which to transform
 *  tensors between different steps of the top-k operation.
 */
static Tensor toCanonicalOrder(const Graph &graph, const Tensor &t,
                               std::size_t distance, std::size_t nActive) {
  assert(t.rank() == 1);

  const auto n = t.dim(0);

  // Already in canonical order
  if (distance == 1 && nActive == n) {
    return t;
  }

  const auto numTiles = graph.getTarget().getNumTiles();

  const auto comparisons = numComparisonsAtDistance(nActive, distance);
  const auto maxComparisonsPerTile = ceildiv(comparisons, numTiles);
  const auto activeTiles = ceildiv0(comparisons, maxComparisonsPerTile);

  // This function just combines slices that are actually continuous.
  std::vector<Interval> slices;
  // Just reserve something, the exact number is hard to calculate up front.
  slices.reserve(activeTiles);
  const auto appendSlice = [&](const auto begin, const auto end) {
    if (begin == end) {
      return;
    }
    if (!slices.empty() && slices.back().end() == begin) {
      slices.back() = Interval(slices.back().begin(), end);
    } else {
      slices.emplace_back(begin, end);
    }
  };

  // We actually build the permutation of the canonically ordered tensor's
  // elements you would take to get the input tensor and then get the inverse
  // permutation and apply that to the tensor to retrieve the canonical form.
  for (unsigned tile = 0; tile < activeTiles; ++tile) {
    auto begin = tile * maxComparisonsPerTile;
    const auto end = std::min((tile + 1) * maxComparisonsPerTile, comparisons);

    if (const auto pre = std::min(distance - begin % distance, end - begin)) {
      const auto elemOffset = comparisonToElemOffset(begin, distance);
      appendSlice(elemOffset, elemOffset + pre);
      appendSlice(elemOffset + distance, elemOffset + distance + pre);
      begin += pre;
    }
    // Add multiples of 2 * distance chunks of elements.
    if (const auto multipleOfDistanceComparisons =
            roundDown(end - begin, distance)) {
      appendSlice(begin * 2, (begin + multipleOfDistanceComparisons) * 2);
      begin += multipleOfDistanceComparisons;
    }
    if (const auto post = end - begin) {
      const auto elemOffset = comparisonToElemOffset(begin, distance);
      appendSlice(elemOffset, elemOffset + post);
      appendSlice(elemOffset + distance, elemOffset + distance + post);
      begin += post;
    }
    assert(begin == end);
  }

  // This method is tied closely with the bitonic allocate method, any inactive
  // elements when comparing at a particular distance are always at the end of
  // the tensor so we can just tack them on.
  const auto remainingOffset = comparisonToElemOffset(comparisons, distance);
  appendSlice(remainingOffset, remainingOffset + n - comparisons * 2);

  return concat(t.slices(getInversePermutation(slices)));
}

/** Allocate a new tensor and rearrange the given input into it in preparation
 *  for a compare and swap step with the given parameters.
 *
 *  Assumes src is in canonical order.
 */
static Tensor rearrangeForStep(Graph &graph, Sequence &prog, const Tensor &src,
                               unsigned dstDistance, unsigned dstActive,
                               TensorCache &tensorCache,
                               const DebugNameAndId &dnai) {
  const auto n = src.dim(0);
  const TensorKey dstKey(n, dstDistance, dstActive);
  auto cacheIt = tensorCache.find(dstKey);
  if (cacheIt == tensorCache.end()) {
    const auto t =
        allocate(graph, src.elementType(), n, dstActive, dstDistance, dnai);
    cacheIt = tensorCache.emplace(dstKey, t).first;
  }
  const auto &dst = cacheIt->second;
  const auto &dstReordered =
      toCanonicalOrder(graph, dst, dstDistance, dstActive);
  prog.add(Copy(src, dstReordered));
  return dst;
}

/** WorklistBuilder is a one-time use class that is used per-tile to
 *  build worklists based on the available work on a tile.
 */
struct WorklistBuilder {
  const unsigned numWorkers;
  const unsigned maxComparisonsPerWorker;
  const bool initialOrder;
  const unsigned distanceToChangeOrder;
  unsigned globalComparisonOffset;

  using Worklists = std::vector<std::vector<std::size_t>>;
  Worklists lists;

  // The number of entries in the worklist for the current worker.
  unsigned numWorkerEntries = 0;
  // The number of remaining comparisons this worker can do based
  // on the maximum per worker.
  unsigned remainingWorkerComparisons = 0;
  // The offset into the data on this tile so far.
  unsigned tileOffset = 0;

  WorklistBuilder(unsigned numWorkers, unsigned maxComparisonsPerWorker,
                  bool initialOrder, unsigned distanceToChangeOrder,
                  unsigned globalComparisonOffset)
      : numWorkers(numWorkers),
        maxComparisonsPerWorker(maxComparisonsPerWorker),
        initialOrder(initialOrder),
        distanceToChangeOrder(distanceToChangeOrder),
        globalComparisonOffset(globalComparisonOffset) {
    lists.reserve(numWorkers);
  }

  // Work is limited by and assumed to be a multiple of the given distance
  // hence the parameters are the distance and a multiple, rather than a more
  // flexible total number of comparisons.
  void addWork(unsigned distance, unsigned repeats = 1) {
    const auto numComparisons = distance * repeats;
    auto remainingComparisons = numComparisons;

    unsigned comparisonOffset = 0;
    // NOTE: The encoding of the worklists is described alongside the c++
    // implementation of the vertex.
    while (remainingComparisons != 0) {
      // Start a new worker
      if (remainingWorkerComparisons == 0) {
        // Fill out the number of entries for the last worker if there
        // was one.
        if (numWorkerEntries != 0) {
          assert(!lists.empty());
          lists.back().front() = numWorkerEntries - 1;
          numWorkerEntries = 0;
        }

        lists.emplace_back();
        // Reserve a spot for number of entries in the worklist
        lists.back().emplace_back();
        const auto currWorkOffset =
            comparisonToElemOffset(comparisonOffset, distance);
        lists.back().emplace_back(tileOffset + currWorkOffset);

        // Knowing the starting order of comparison over the whole
        // input, we can work out for the slice of comparisons that
        // this worker does what order it should start comparing
        // in by using the starting offset for this worker into the
        // whole input to determine how many changes of order have come before.
        const bool workerInitialOrder =
            initialOrder ^
            ((globalComparisonOffset / distanceToChangeOrder) & 1u);
        // The worker keeps a count of comparisons left before it must
        // flip the order in which it compares and swaps. Because the
        // worker may start at any offset into the comparisons to be
        // done over the whole input, we provide a count to initialise
        // the counter with based on the offset into the whole input that
        // this worker starts its work. The worker will calculate
        // the remaining compare and swaps to do before flipping the order
        // by taking distanceToChangeOrder - initialCount.
        const auto initialCount =
            globalComparisonOffset % distanceToChangeOrder;
        assert(initialCount < (1u << 31));
        // We encode the initial direction and initial offset into
        // 32-bits in the worklist where the upper 31 bits are allocated
        // to the initial count and the lowest bit is allocated to the
        // initial order.
        const unsigned short initialLower =
            ((initialCount & ((1u << 15u) - 1)) << 1u) | workerInitialOrder;
        const unsigned short initialUpper =
            (initialCount >> 15u) & ((1u << 16) - 1);
        lists.back().emplace_back(initialLower);
        lists.back().emplace_back(initialUpper);

        remainingWorkerComparisons = maxComparisonsPerWorker;

        // Similar to initialCount, due to the fact that a worker can
        // start at any offset into the work to be done on this tile/
        // over the whole input, we provide an initial count for the
        // innermost loop that would ordinarily otherwise be set to
        // 'distance' for the first entry in the worklist.
        const auto firstInnerCount = std::min(
            distance - comparisonOffset % distance,
            std::min(remainingComparisons, remainingWorkerComparisons));
        lists.back().emplace_back(firstInnerCount);
      }

      const auto comparisonsThisEntry =
          std::min(remainingComparisons, remainingWorkerComparisons);

      lists.back().emplace_back(distance);
      lists.back().emplace_back(comparisonsThisEntry);

      comparisonOffset += comparisonsThisEntry;
      globalComparisonOffset += comparisonsThisEntry;
      remainingWorkerComparisons -= comparisonsThisEntry;
      remainingComparisons -= comparisonsThisEntry;
      ++numWorkerEntries;
    }
    tileOffset += numComparisons * 2;
  }

  // Invalidates the builder and returns the worklists that were built.
  Worklists finish() {
    if (numWorkerEntries != 0) {
      assert(!lists.empty());
      lists.back().front() = numWorkerEntries - 1;
      numWorkerEntries = 0;
    }
    return std::move(lists);
  }
};

/** The basic building block for the algorithm. This step
 *  performs a compare and swap at the given distance if the input
 *  tensors were flat array in canonical order.
 *
 *  e.g. nActive = 15, distance = 4, distanceToChangeOrder = 4
 *
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *    +--------------->               <---------------+
 *        +--------------->               <---------------+
 *            +--------------->               <---------------+
 *                +--------------->
 *
 *  e.g. nActive = 15, distance = 2, distanceToChangeOrder = 4
 *
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
 *  +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 *    +------->       +------->       <-------+       <-------+
 *        +------->       +------->       <-------+
 *
 *  In these examples each arrow indicates the 2 elements are compared,
 *  and if the element the arrow points to is less than the element it
 *  starts from, the elements are swapped.
 *
 *  We add some limitations to this step in order to appease the
 *  hardware. The main one is that the distance is consistent for the
 *  entire operation, as is the distance at which we flip the order of
 *  comparison.
 *
 *  The actual implementation maps comparisons to tiles by taking the
 *  flat array, and evenly splitting the work amongst tiles. Where the
 *  lhs and rhs of the comparison reside on different tiles, we gather
 *  the comparisons to be done in chunks of powers of 2. This limits
 *  the amount of exchange code required to rearrange for each step,
 *  and limits the overhead we add would add by having the vertex deal
 *  with different distances at which to compare to a number of different
 *  distances that is roughly logarithmic with respect to the number
 *  of elements per tile (in each compare and swap step).
 */
static std::string getVertexClass(const Tensor &keys,
                                  const std::optional<Tensor> &values,
                                  const bool valuesAreSecondaryKey) {
  if (values) {
    return templateVertex("popops::CompareAndSwapAtDistanceKeyVal",
                          keys.elementType(), values->elementType(),
                          valuesAreSecondaryKey);
  } else {
    return templateVertex("popops::CompareAndSwapAtDistance",
                          keys.elementType());
  }
}

static void
compareAndSwapAtDistance(Graph &graph, Sequence &prog, const Tensor &keys,
                         std::optional<Tensor> values,
                         bool valuesIsSecondaryKey, unsigned distance,
                         unsigned distanceToChangeOrder, bool initialOrder,
                         unsigned nActive, const DebugNameAndId &dnai) {
  assert(!values || keys.shape() == values->shape());

  // When values are only copied and not used for comparison or calculation
  // (e.g. when valuesIsSecondaryKey is not set), we can re-use code for one
  // value type for all value types with the same size per-element using a
  // reinterpret.
  if (!valuesIsSecondaryKey && values && values->elementType() != FLOAT) {
    values = values->reinterpret(FLOAT);
  }
  const auto vertexClass = getVertexClass(keys, values, valuesIsSecondaryKey);

  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto numWorkers = target.getNumWorkerContexts();

  const auto comparisons = numComparisonsAtDistance(nActive, distance);
  const auto maxComparisonsPerTile = ceildiv(comparisons, numTiles);
  const auto activeTiles = ceildiv0(comparisons, maxComparisonsPerTile);

  const auto cs = graph.addComputeSet({dnai, "CompareAndSwap"});
  for (unsigned tile = 0; tile < activeTiles; ++tile) {
    auto begin = tile * maxComparisonsPerTile;
    const auto end = std::min((tile + 1) * maxComparisonsPerTile, comparisons);
    const auto comparisonsThisTile = end - begin;

    const auto v = graph.addVertex(cs, vertexClass);
    graph.setTileMapping(v, tile);

    graph.connect(v["keys"], keys.slice(begin * 2, end * 2).flatten());
    if (values) {
      graph.connect(v["values"], values->slice(begin * 2, end * 2).flatten());
    }

    graph.setInitialValue(v["distanceToChangeOrder"], distanceToChangeOrder);

    // We just flatten all the comparisons to be done and evenly spread
    // them amongst workers for now without consideration of how the split
    // might introduce uneven overheads between workers.
    const auto comparisonsPerWorker = ceildiv(comparisonsThisTile, numWorkers);
    WorklistBuilder builder(numWorkers, comparisonsPerWorker, initialOrder,
                            distanceToChangeOrder, begin);

    if (const auto pre = std::min(distance - begin % distance, end - begin)) {
      builder.addWork(pre);
      begin += pre;
    }
    if (const auto multipleOfDistanceComparisons =
            floordiv(end - begin, distance)) {
      builder.addWork(distance, multipleOfDistanceComparisons);
      begin += multipleOfDistanceComparisons * distance;
    }
    if (const auto post = end - begin) {
      builder.addWork(post);
      begin += post;
    }
    assert(begin == end);

    const auto &worklists = builder.finish();

    const auto worklistsField = v["worklists"];
    graph.setFieldSize(worklistsField, worklists.size());
    for (unsigned i = 0; i < worklists.size(); ++i) {
      const auto t =
          graph.addConstant(UNSIGNED_SHORT, {worklists[i].size()},
                            worklists[i].data(), {dnai, "worklists"});
      graph.setTileMapping(t, tile);
      graph.connect(worklistsField[i], t);
    }
  }
  prog.add(Execute(cs));
}

/** Create an input for a top-k laid out in the most efficient
 *  way.
 */
Tensor createTopKInputImpl(Graph &graph, const Type &type,
                           const std::vector<std::size_t> &shape,
                           const DebugNameAndId &dnai) {
  if (shape.empty()) {
    throw poplibs_error("shape must have at least 1 dimension");
  }
  const auto numElems = product(shape);
  const auto n = shape.back();
  const auto b = numElems / n;
  auto t = allocate(graph, type, b * n, b * n, 1, dnai);
  return t.reshape({n, b}).transpose().reshape(shape);
}

/** Core implementation for all variants.
 *
 *  All the variants are implemented by application of our basic
 *  compareAndSwapAtDistance operation some number of times with the
 *  right parameters. A key limitation of compareAndSwapAtDistance is that
 *  it requires us to compare and swap at a consistent distance across the
 *  entire array, flipping the order imposed at another consistent interval.
 *
 *  ========= Sort =========
 *  I will not describe a bitonic sorting network for powers of 2 as this is
 *  well documented. For n a power of 2 the above limitation on
 *  compareAndSwapAtDistance is not a problem, as the size of partitions in
 *  each step (the distance at which we compare and swap) is always a
 *  consistent factor of 2 greater than the last step across the whole n
 *  elements to be sorted.
 *
 *  For n not a power of 2 we compose the sorting network of power-of-2
 *  sized sorting networks. e.g. if n=100, we compose the full sort with
 *  smaller sorting networks of size 64, 32, 4 (64 + 32 + 4 = 100). The
 *  result of these power of 2 sized networks are merged and re-sorted
 *  hierarchically.
 *
 *  With regards the limitation on compare and swap described above,
 *  the key insight is that when all the networks are of size a power of 2,
 *  sorting the sub-sequences into order always uses ascending powers of 2
 *  distances at which to compare and swap. Once the sub-sequences are in
 *  order merging them to form a larger non-power of 2 sized sequence is
 *  always a compare and swap at the next largest power of 2. We can therefore
 *  apply a series of compare and swap steps with increasing powers of 2
 *  distances to form our final fully sorted sequence and in each step the
 *  distance to compare and swap is always consistent across the whole
 *  array.
 *
 *  When merging the result of our power of 2 sized sorting networks,
 *  we must take care. In these cases we can do one of the following:
 *
 *   * Given an ascending/descending bitonic sequence s of size n and
 *     known inflection point p, we can compare and swap at distance
 *     p with descending order, and the result is guaranteed to be such
 *     that all in the set of elements left of p are greater than all
 *     in the set of elements right of p.
 *   * Conversely, given a descending/ascending bitonic sequence s of
 *     size n and known inflection point p, we can compare and swap at
 *     distance p with ascending order, and the result is guaranteed
 *     to be such that all in the set of elements left of p are less
 *     than all in the set of elements right of p.
 *
 *  In order to use the above this means each time we merge sorted power
 *  of 2 sequences, they must be sorted in a particular order. Different
 *  powers of 2 will take different numbers of compare and swap steps to
 *  end up in the correct order for merging. Since the smaller powers of
 *  2 must always take part in the same or fewer compare and swap operations
 *  than the larger powers of 2, we can carefully omit them from the
 *  right compare and swap operations on larger powers of 2 to ensure they
 *  still end up in the right order for merging and we get the correct
 *  result.
 *
 *  ========= Top-K =========
 *  In order to handle top-k, the general idea is to build sorted sequences
 *  of size k with alternating ascending/descending order, allowing us to
 *  merge 2 sequences of k with opposing ordering giving a result where one of
 *  the sequences has elements which are all greater than the elements in
 *  the other sequence. Each time we do this, we discard the smaller
 *  sequence and re-sort the sequences of k elements into ascending/
 *  descending order and repeat until we are left with k elements.
 *
 *  We make a small modification to this which is we actually first build
 *  sorted sequences of k' elements where k' is the next power of 2 greater
 *  then or equal to k. This again lets us keep our distance of compare and
 *  swap consistent in each step and doesn't further complicate the compare
 *  and swap operation. To get the final result if k is not a power of 2
 *  we keep sorting and discarding descreasing sized powers of 2 until we
 *  are left with just k elements.
 */
std::pair<Tensor, Tensor>
topKImpl(Graph &graph, Sequence &prog, const Tensor &t_,
         const std::optional<Tensor> &other_, const unsigned k,
         const bool largest, const bool sorted, const bool ascendingOrder,
         bool otherIsSecondaryKey, const DebugNameAndId &dnai) {

  if (other_ && (other_->shape() != t_.shape())) {
    throw poplibs_error("t.shape() (" + toString(t_.shape()) +
                        " != other.shape() (" + toString(other_->shape()) +
                        "). Other tensor in topKKeyValue must have matching "
                        "shape to tensor in which to find top-k.");
  }

  if (t_.rank() == 0) {
    throw poplibs_error("t must have at least one dimension");
  }

  const auto inputType = t_.elementType();
  if (inputType != HALF && inputType != FLOAT && inputType != UNSIGNED_INT &&
      inputType != INT) {
    throw poplibs_error("bitonic::topKImpl: Unsupported key type " +
                        inputType.toString());
  }
  if (other_ && other_->elementType() != HALF &&
      other_->elementType() != FLOAT && other_->elementType() != UNSIGNED_INT &&
      other_->elementType() != INT) {
    throw poplibs_error("bitonic::topKImpl: Unsupported value type " +
                        other_->elementType().toString());
  }
  if (!sorted && other_ && otherIsSecondaryKey) {
    logging::popops::warn(
        "bitonicTopKImpl: !sorted && other_ && otherIsSecondaryKey may cause "
        "suboptimal performance. Consider turning off otherIsSecondaryKey "
        "flag.");
  }

  const std::vector<std::size_t> outputShape = [&] {
    auto s = t_.shape();
    s.back() = k;
    return s;
  }();

  // We use {n, batch} as our canonical internal representation for
  // the shape.
  auto t =
      t_.rank() >= 2 ? t_.flatten(0, t_.rank() - 1) : t_.flatten().expand({0});
  t = t.transpose();
  std::optional<Tensor> other;
  if (other_) {
    other = other_->rank() >= 2 ? other_->flatten(0, other_->rank() - 1)
                                : other_->flatten().expand({0});
    other = other->transpose();
  }

  unsigned n = t.dim(0);
  const unsigned b = t.dim(1);

  logging::popops::debug("bitonicTopK(batchSize={}, n={}, k={}, sorted={}, "
                         "otherIsSecondaryKey={}, "
                         "haveOther={}, debugPath='{}')",
                         b, n, k, sorted, otherIsSecondaryKey,
                         (other ? "true" : "false"), dnai.getPathName());

  if (k > n) {
    throw poplibs_error(
        "k (" + std::to_string(k) +
        ") must be less than or equal to the number of input elements (" +
        std::to_string(n) + ")");
  }

  const auto logK = ceilLog2(k);
  const auto logN = ceilLog2(n);
  // We define k' as the next power of 2 greater or equal than k.
  const auto kDash = (1u << logK);
  const auto stepsToSortK = nthTriangular(logK);
  const auto totalSteps = nthTriangular(logN);

  logging::popops::debug(
      "bitonicTopKImpl: Calculated no. of steps: total={}, k={}", totalSteps,
      stepsToSortK);

  // Edge cases where the output is zero-sized.
  if (b * k == 0) {
    Tensor tResult =
        graph.addVariable(t.elementType(), outputShape, {dnai, "keys"});
    Tensor otherResult;
    if (other) {
      otherResult = graph.addVariable(other->elementType(), outputShape,
                                      {dnai, "values"});
    }
    return std::make_pair(std::move(tResult), std::move(otherResult));
  }

  t = t.flatten();
  if (other) {
    other = other->flatten();
  }

  // Handle some trivial cases where this is a no-op.
  //
  //  * We already have the top/bottom k elements and the result doesn't
  //    need to sorted.
  //  * n is 1, the output is sorted by default.
  if ((k == n && !sorted) || n == 1) {
    // We assume the result does not alias the input and is writeable so
    // ensure that is the case.
    t = poputil::duplicate(graph, t, prog, {dnai, "keys"});
    t = t.reshape({k, b}).transpose().reshape(outputShape);
    if (other) {
      other = poputil::duplicate(graph, *other, prog, {dnai, "values"});
      other = other->reshape({k, b}).transpose().reshape(outputShape);
    }
    return std::make_pair(std::move(t), other.value_or(Tensor{}));
  }

  if (inputType == HALF) {
    t = cast(graph, t, FLOAT, prog, dnai);
  }

  Type origValueType;
  if (other) {
    origValueType = other->elementType();
    if (other->elementType() == HALF) {
      other = cast(graph, *other, FLOAT, prog, dnai);
    }
  }

  // Because we always discard the upper of a pair of sequences of k'
  // elements, the merge order when merging sequences is based off
  // leaving the top/bottom k' elements in each pair of k' sequences
  // in the lower k' elements.
  const bool mergeKSequencesOrder = !largest;
  const bool oddKSequences = bool((n / kDash) & 1u);

  TensorCache tCache, otherCache;
  for (unsigned mergeStep = 0; mergeStep < logN; ++mergeStep) {
    const auto logMergeDistance = std::min(mergeStep, logK);
    const auto mergeDistance = 1u << logMergeDistance;
    const auto stepName = "Merge" + std::to_string(mergeStep);

    const bool mergeOrder = [&] {
      bool order;
      if (mergeStep >= logK) {
        // Use the merge order for merging sequences of k' elements if that
        // is what we are doing.
        order = mergeKSequencesOrder;
      } else {
        // If this is the last step, we need to sort in the desired order.
        if (mergeStep == logN - 1) {
          order = ascendingOrder;
          // If this is the last step of sorting k' sequences, make sure they
          // are sorted in the order needed for merging sequences of k'
          // elements.
        } else if (mergeStep + 1 == logK) {
          order = !mergeKSequencesOrder;
        } else {
          // Otherwise, we merge based on creating sorted sequences in the
          // order needed either to create the final sorted sequence or to
          // merge k' sequences correctly.
          order = logK != logN ? (mergeKSequencesOrder ^ oddKSequences)
                               : !ascendingOrder;
        }
      }
      return order;
    }();

    // As mentioned at the signature of this function, we must sometimes
    // omit the parts of n composed of smaller powers of 2 in order to
    // result in the correctly ordered sequence once merged with another
    // power of 2. This check was sort of experimentally worked out. In
    // the interests of time, I'm leaving it as it works but this should
    // be justified and there may be a simpler calculation.
    const auto nMod2d = n % (mergeDistance * 2);
    const auto nMod4d = n % (mergeDistance * 4);
    auto nThisStep = n;
    if (mergeStep + 1 < logK && (nMod2d == 0 || nMod2d == nMod4d)) {
      nThisStep -= nMod4d;
    }

    t = rearrangeForStep(graph, prog, t, mergeDistance * b, nThisStep * b,
                         tCache, {dnai, "keys" + stepName});
    if (other) {
      other = rearrangeForStep(graph, prog, *other, mergeDistance * b,
                               nThisStep * b, otherCache,
                               {dnai, "values" + stepName});
    }

    // While we are building sorted sequences of k' elements, the distance
    // at which we change direction is the same as the merge distance,
    // otherwise the direction should always be descending as we always
    // discard the higher k' elements.
    const auto changeDirDistance =
        mergeStep < logK ? mergeDistance : 1u << (logN - 1);
    compareAndSwapAtDistance(graph, prog, t, other, otherIsSecondaryKey,
                             mergeDistance * b, changeDirDistance * b,
                             mergeOrder, nThisStep * b, {dnai, stepName});

    t = toCanonicalOrder(graph, t, mergeDistance * b, nThisStep * b);
    if (other) {
      other = toCanonicalOrder(graph, *other, mergeDistance * b, nThisStep * b);
    }

    // If we're done building sequences of k' elements, and we still have
    // more than k' elements, discard the upper k' elements of each
    // pair of k' sized sequences as by this point we have imposed an
    // ordering between them.
    if (logMergeDistance >= logK && n > kDash) {
      const auto kDashMultiples = floordiv(n, 2 * kDash);
      const auto remainder = n - kDashMultiples * 2 * kDash;
      const auto t2d = t.reshape({n, b});
      const auto evenPart =
          t2d.slice(0, kDashMultiples * 2 * kDash)
              .reshapePartial(0, 1, {kDashMultiples, 2 * kDash});
      const auto oddPart = t2d.slice(kDashMultiples * 2 * kDash, n);
      t = concat(evenPart.slice(0, kDash, 1).flatten(0, 2),
                 oddPart.slice(0, std::min(remainder, kDash)))
              .flatten();
      if (other) {
        const auto t2d = other->reshape({n, b});
        const auto evenPart =
            t2d.slice(0, kDashMultiples * 2 * kDash)
                .reshapePartial(0, 1, {kDashMultiples, 2 * kDash});
        const auto oddPart = t2d.slice(kDashMultiples * 2 * kDash, n);
        other = concat(evenPart.slice(0, kDash, 1).flatten(0, 2),
                       oddPart.slice(0, std::min(remainder, kDash)))
                    .flatten();
      }
      n = nThisStep = kDashMultiples * kDash + std::min(remainder, kDash);
    }

    for (unsigned sortStep = 0; sortStep < logMergeDistance; ++sortStep) {
      const auto stepName =
          "Sort" + std::to_string(logMergeDistance - sortStep - 1);

      const auto sortDistance = 1u << (logMergeDistance - sortStep - 1);
      // Once we have built sorted sequences of k' elements, we keep
      // merging and re-sorting so the distance at which we change
      // directions once we start discarding is always k'/2.
      const auto changeDirDistance =
          std::min(mergeDistance, (1u << (logK - 1)));

      const bool sortOrder = [&] {
        bool order;
        if (mergeStep < logK) {
          // If we are still building sorted sequences of k' elements, the
          // sort order is the same as the merge order
          order = mergeOrder;
        } else {
          if (mergeStep == logN - 1) {
            // If this is the last step then sort into the final desired
            // order.
            order = ascendingOrder;
          } else {
            // Otherwise sort into the order needed to correctly merge
            // sequences of k' elements.
            order = !mergeKSequencesOrder;
          }
        }
        return order;
      }();

      t = rearrangeForStep(graph, prog, t, sortDistance * b, nThisStep * b,
                           tCache, {dnai, "keys" + stepName});
      if (other) {
        other = rearrangeForStep(graph, prog, *other, sortDistance * b,
                                 nThisStep * b, otherCache,
                                 {dnai, "values" + stepName});
      }

      compareAndSwapAtDistance(graph, prog, t, other, otherIsSecondaryKey,
                               sortDistance * b, changeDirDistance * b,
                               sortOrder, nThisStep * b, {dnai, stepName});
      t = toCanonicalOrder(graph, t, sortDistance * b, nThisStep * b);
      if (other) {
        other =
            toCanonicalOrder(graph, *other, sortDistance * b, nThisStep * b);
      }

      // If we have finished building sequences of k' elements and
      // there are more than k elements remaining, we can discard descending
      // powers of 2 during the re-sort until we are left with exactly k
      // elements.
      const auto lastKDashElements = ((n - 1) % kDash) + 1;
      // We either discard elements from the start or end of the last k'
      // elements depending on whether we want the top or bottom k elements
      // and depending on the sort order and offset into the array when it
      // was sorted.
      const bool fromStart =
          sortOrder ^ !largest ^ bool(((n - lastKDashElements) / kDash) & 1u);
      const auto nToDiscard =
          fromStart ? sortDistance
                    : ((lastKDashElements - 1) % sortDistance) + 1;
      if (logMergeDistance + 1 >= logK && lastKDashElements >= nToDiscard &&
          lastKDashElements - nToDiscard >= k) {
        std::vector<Interval> slices;
        if (fromStart) {
          slices.emplace_back(0, n - lastKDashElements);
          slices.emplace_back(n - lastKDashElements + nToDiscard, n);
        } else {
          slices.emplace_back(0, n - nToDiscard);
        }
        const auto t2d = t.reshape({n, b});
        t = concat(t2d.slices(slices)).flatten();
        if (other) {
          const auto t2d = other->reshape({n, b});
          other = concat(t2d.slices(slices)).flatten();
        }
        n -= nToDiscard;
        nThisStep = n;
      }
    }
  }

  if (inputType != t.elementType()) {
    t = cast(graph, t, inputType, prog, dnai);
  }

  if (other && origValueType != other->elementType()) {
    other = cast(graph, *other, origValueType, prog, dnai);
  }

  t = t.reshape({k, b}).transpose().reshape(outputShape);
  if (other) {
    other = other->reshape({k, b}).transpose().reshape(outputShape);
  }
  return std::make_pair(std::move(t), other.value_or(Tensor{}));
}

} // end namespace bitonic
} // end namespace popops
