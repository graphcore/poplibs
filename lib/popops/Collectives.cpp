#include "popops/Collectives.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Reduce.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <boost/optional/optional.hpp>

using namespace poplar;
using namespace poplar::program;

namespace popops {

namespace  {
  enum class CollectiveMethod {
    AUTO,
    // Send fragments clockwise around the ring. The number of fragments
    // is equal to the number of IPUs in the ring.
    CLOCKWISE_RING,
    // Send fragments anticlockwise around the ring. The number of fragments
    // is equal to the number of IPUs in the ring.
    ANTICLOCKWISE_RING,
    // Split the data into two halves and use the clockwise ring algorithm on
    // one half and the anticlockwise ring algorithm on the other in order
    // to fully utilize the links in both directions. The number of fragments
    // is equal to twice the number of IPUs in the ring.
    BIDIRECTIONAL_RING_PAIR,
    // Send half the fragments half way around the ring in the clockwise
    // direction and half the fragments half way around the ring in the
    // anticlockwise direction, meeting in the middle. The number of fragments
    // is equal to the number of IPUs in the ring. The disadvantage compared
    // to the BIDIRECTIONAL_RING_PAIR method is that the usage of available
    // bandwidth is not quite optimal, in particular the final step only uses
    // the links in one direction (assuming an even number of IPUs). The
    // advantage is the that it requires fewer steps and allows the use of
    // larger fragments.
    MEET_IN_MIDDLE_RING,
  };
}

static CollectiveMethod
parseCollectiveOptions(const poplar::OptionFlags &options) {
  CollectiveMethod method = CollectiveMethod::AUTO;
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec spec{
    { "method",
      OptionHandler::createWithEnum(
        method,
        {
          { "auto", CollectiveMethod::AUTO },
          { "clockwise_ring", CollectiveMethod::CLOCKWISE_RING },
          { "anticlockwise_ring", CollectiveMethod::ANTICLOCKWISE_RING },
          { "bidirectional_ring_pair",
            CollectiveMethod::BIDIRECTIONAL_RING_PAIR },
          { "meet_in_middle_ring", CollectiveMethod::MEET_IN_MIDDLE_RING }
        }
      )
    }
  };
  for (const auto &entry : options) {
    spec.parse(entry.first, entry.second);
  }
  return method;
}

static std::vector<Interval>
splitIntervals(std::vector<Interval> &toMatch,
               const std::vector<Interval> &reference) {
  // For each interval in to match make sure that it is split into multiple
  // intervals whenever there is a split between regions in the reference
  // intervals.
  std::vector<Interval> newToMatch;
  unsigned j = 0;
  unsigned sizeSoFar = 0;
  for (unsigned i = 0; i < toMatch.size(); ++i) {
    const auto referenceSize = reference[j].size();
    const auto thisSize = toMatch[i].size();
    if (thisSize + sizeSoFar <= referenceSize) {
      newToMatch.push_back(toMatch[i]);
      if (thisSize + sizeSoFar == referenceSize) {
        sizeSoFar = 0;
        ++j;
      } else {
        sizeSoFar += thisSize;
      }
    } else {
      // If this else is hit means interval needs splitting
      newToMatch.push_back(Interval(toMatch[i].begin(),
                                    toMatch[i].begin() + referenceSize));
      toMatch[i] = Interval(toMatch[i].begin() + referenceSize,
                            toMatch[i].end());
      sizeSoFar = 0;
      ++j;
      --i; // don't advance i as still need to finish remainder of this interval
    }
  }
  assert(j == reference.size());
  assert(newToMatch.size() == reference.size());
  return newToMatch;
}

static void
matchIntervals(std::vector<Interval> &toMatch,
               const std::vector<Interval> &reference) {
  // As getIpuMapping will have compressed the intervals by as much as possible
  // to get the intervals to match we only need to break up intervals that are
  // bigger that in the reference
  toMatch = splitIntervals(toMatch, reference);
}


// Compress intervals so that each interval on each ipu and at each index maps
// to an equivalent interval in the reference of the same size
static void
matchIntervals(std::vector<std::vector<Interval>> &toCompress,
               const std::vector<std::vector<Interval>> &reference) {
  assert(toCompress.size() == reference.size());
  std::vector<std::vector<Interval>> result(toCompress.size());
  for (unsigned ipu = 0; ipu < toCompress.size(); ++ipu) {
    matchIntervals(toCompress[ipu], reference[ipu]);
  }
}

// All the operations in the all reduce (splitIntoFragments and
// concat model parallel chunks) aim to preserve the order of the tensor
// on the ipu and only perform transforms of elements on different ipus.
// This means that when ever we get the elements of a tensor on an ipu (which
// uses this mapping) as long as we ensure that when creating the tensor
// the on ipu elements order is preserved then the final tensor's order will
// be preserved. This function returns the intervals of the tensor on each ipu
// ordered by the intervals
static std::vector<std::vector<Interval>>
getIpuMapping(const Graph &graph, const Tensor &t) {
  // find all intervals on each ipu
  const auto &tileMapping = graph.getTileMapping(t);
  std::vector<std::vector<Interval>> ipuMapping(graph.getTarget().getNumIPUs());
  for (unsigned tile = 0; tile < tileMapping.size(); ++tile) {
    const unsigned ipu = tile / graph.getTarget().getTilesPerIPU();
    for (const auto &interval : tileMapping[tile]) {
      ipuMapping[ipu].push_back(interval);
    }
  }

  // sort intervals
  for (unsigned ipu = 0; ipu < ipuMapping.size(); ++ipu) {
    std::sort(ipuMapping[ipu].begin(), ipuMapping[ipu].end(),
                          [&](Interval A, Interval B) {
      return A.begin() < B.begin();
    });
  }

  // compress intervals
  std::vector<std::vector<Interval>> result(ipuMapping.size());
  for (unsigned ipu = 0; ipu < ipuMapping.size(); ++ipu) {
    for (unsigned i = 0; i < ipuMapping[ipu].size(); ++i) {
      if (result[ipu].empty() ||
          result[ipu].back().end() != ipuMapping[ipu][i].begin()) {
        result[ipu].push_back(ipuMapping[ipu][i]);
      } else {
        result[ipu].back() = Interval(result[ipu].back().begin(),
                                      ipuMapping[ipu][i].end());
      }
    }
  }
  return result;
}

// Taking an ordered vector of intervals of the tensor that are on an ipu
// return the elements of the tensor on this ipu in a single tensor
static Tensor
getOnIpuTensor(const Tensor &t, Graph &graph,
               const std::vector<Interval> &ipuIntervals) {
  assert(t.rank() == 1);
  std::vector<Tensor> onIpuElements;
  for (const auto &inter : ipuIntervals) {
    onIpuElements.push_back(t.slice(inter.begin(), inter.end()));
  }
  if (onIpuElements.empty()) {
    return graph.addVariable(t.elementType(), {0});
  }
  return concat(onIpuElements);
}


// take a tensor and return a vector of tensors where each element
// is a slice of the original tensor that spans only one ipu
static std::vector<Tensor>
getPerIpuTensors(const Tensor &t, Graph &graph,
                 const unsigned rank, const unsigned ipusPerRank) {
  const unsigned startIpu = rank * ipusPerRank;
  const unsigned endIpu = startIpu + ipusPerRank;

  const auto ipuMapping = getIpuMapping(graph, t);

  std::vector<Tensor> result;
  for (unsigned ipu = startIpu; ipu < endIpu; ++ipu) {
    result.push_back(getOnIpuTensor(t, graph, ipuMapping[ipu]));
  }
  return result;
}

static std::vector<Tensor>
splitIntoFragments(const Tensor &t, unsigned numFragments) {
  assert(t.rank() == 1);
  unsigned numElements = t.dim(0);
  // Fragments indexed by ipu.
  std::vector<Tensor> fragments;
  fragments.reserve(numFragments);
  for (unsigned fragment = 0; fragment != numFragments; ++fragment) {
    auto elementBegin = (numElements * fragment) / numFragments;
    auto elementEnd = std::min(numElements,
                               (numElements * (fragment + 1)) / numFragments);
    fragments.push_back(t.slice(elementBegin, elementEnd));
  }
  return fragments;
}

static std::vector<Tensor>
splitRankIntoFragments(const Tensor &t, Graph &graph,
                       const unsigned numFragments,
                       const unsigned rank, const unsigned ipusPerRank) {
  assert(t.rank() == 1);
  if (ipusPerRank == 1) {
    return splitIntoFragments(t, numFragments);
  }
  const auto perIpuTensors = getPerIpuTensors(t, graph, rank, ipusPerRank);
  assert(perIpuTensors.size() == ipusPerRank);

  std::vector<std::vector<Tensor>> perIpuFragments(ipusPerRank);
  for (unsigned j = 0; j < perIpuTensors.size(); ++j) {
    perIpuFragments[j] = splitIntoFragments(perIpuTensors[j], numFragments);
  }
  std::vector<Tensor> fragments(numFragments);
  for (unsigned j = 0; j < numFragments; ++j) {
    std::vector<Tensor> fragmentsToConcat;
    for (unsigned k = 0; k < ipusPerRank; ++k) {
      fragmentsToConcat.push_back(std::move(perIpuFragments[k][j]));
    }
    if (fragmentsToConcat.empty()) {
      throw poputil::poplibs_error("No fragments to concat");
    }
    fragments[j] = concat(fragmentsToConcat);
  }
  return fragments;
}

// For a all reduce that has multiple ipus in each rank to use all the
// links must split each ipus part of the tensor into as many fragments
// as ranks. This function gets the tensor on each ipu, splits that into
// fragments, then concatenates it with the corresponding fragments
// from the other ipus in the rank
static std::vector<std::vector<Tensor>>
splitIntoFragments(const Tensor &t, unsigned numFragments, Graph &graph,
                   const unsigned ipusPerRank) {
  assert(t.rank() == 2);

  std::vector<std::vector<Tensor>> fragments(t.dim(0));
  for (unsigned i = 0; i < t.dim(0); ++i) {
    if (ipusPerRank == 1) {
      // can avoid having looking at ipu mapping and just call split into
      // fragments
      fragments[i] = splitIntoFragments(t[i], numFragments);
      continue;
    }
    // get per ipu tensors
    auto perIpuTensors =  getPerIpuTensors(t[i], graph, i, ipusPerRank);
    assert(perIpuTensors.size() == ipusPerRank);
    // now split each of the per ipu tensors into fragments
    std::vector<std::vector<Tensor>> perIpuFragments(ipusPerRank);
    for (unsigned j = 0; j < perIpuTensors.size(); ++j) {
      perIpuFragments[j] = splitIntoFragments(perIpuTensors[j], numFragments);
    }
    // now concat tensors each fragment has a part from every ipu in the rank
    for (unsigned j = 0; j < numFragments; ++j) {
      std::vector<Tensor> fragmentsToConcat;
      for (unsigned k = 0; k < ipusPerRank; ++k) {
        fragmentsToConcat.push_back(perIpuFragments[k][j]);
      }
      fragments[i].push_back(concat(fragmentsToConcat));
    }
  }
  return fragments;
}

static void checkTensorIpuMappings(const Graph &graph, const Tensor &toReduce) {
  if (toReduce.rank() != 2) {
    throw poputil::poplibs_error("Rank of tensor to reduce must be 2");
  }

  if (toReduce.dim(0) == 0) {
    return;
  }

  const unsigned ipusPerRank =
        graph.getTarget().getNumIPUs() / toReduce.dim(0);


  // every rank must have all the elements of the tensor to reduce
  // spread across the ipus in the rank exactly the same as every other
  // rank
  const auto firstMapping = getIpuMapping(graph, toReduce[0]);
  for (unsigned rank = 1U; rank < toReduce.dim(0); ++rank) {
    const auto ipuMapping = getIpuMapping(graph, toReduce[rank]);
    for (unsigned j = 0; j < ipusPerRank; ++j) {
      if (firstMapping[j] != ipuMapping[(rank * ipusPerRank) + j]) {
        throw poputil::poplibs_error("Different ranks must have tensor on "
                                      "same ipus with in the rank");
      }
    }
  }
}


static poplar::Tensor
cloneToRank(poplar::Graph &graph, const poplar::Tensor &t,
            unsigned dstRank, unsigned numRanks) {
  auto mapping = graph.getTileMapping(t);
  const auto &target = graph.getTarget();
  const auto tilesPerIPU = target.getTilesPerIPU();
  const auto numIPUs = target.getNumIPUs();
  assert(numIPUs % numRanks == 0);
  const auto ipusPerRank = numIPUs / numRanks;
  for (unsigned ipu = 0; ipu != numIPUs; ++ipu) {
    auto rank = ipu / ipusPerRank;
    if (rank == dstRank)
      continue;
    const auto ipuInRank = ipu % ipusPerRank;
    const auto dstIpu = ipuInRank + dstRank * ipusPerRank;
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
  auto tLocal = graph.clone(t);
  graph.setTileMapping(tLocal, mapping);
  return tLocal;
}

// Return the IPUs in clockwise direction around the ring starting at IPU 0.
static std::vector<unsigned> createRing(const unsigned n) {
  std::vector<unsigned> ring(n);
  unsigned i = 0;
  std::generate(ring.begin(), ring.begin() + ((n+1)/ 2),
                  [&] {i += 2; return i - 2;});
  if ((n & 1) != 0) {
    i -= 3;
  } else {
    --i;
  }
  std::generate(ring.begin() + ((n+1) / 2), ring.end(),
                  [&] {i -= 2; return i + 2;});
  return ring;
}

static std::vector<unsigned> arrangeInRing(const std::vector<Tensor> &data) {
  return createRing(data.size());
}

static std::vector<unsigned> arrangeInRing(const Tensor &toReduce) {
  return createRing(toReduce.dim(0));
}

static std::vector<unsigned> arrangeInRing(const std::vector<Chunk> &chunks) {
  return createRing(chunks.size());
}

// Create the chunks from the data returned from the scatter steps
static Chunks createChunks(Graph &graph, const Tensor &originalInput,
                           const std::vector<Tensor> &data,
                           const unsigned ipusPerRank,
                           const unsigned numPartials) {
  const auto ring = arrangeInRing(originalInput);
  Chunks chunks(graph.getTarget().getNumIPUs());

  chunks.originalInput = originalInput;
  for (unsigned i = 0; i != numPartials; ++i) {
    const auto ipuMapping = getIpuMapping(graph, data[i]);
    for (unsigned j = 0; j < ipusPerRank; ++j) {
      const auto ipu = (ring[i] * ipusPerRank) + j;
      if (ipusPerRank == 1) {
        chunks.chunks[ipu] = {data[i], i, j};
      } else {
        chunks.chunks[ipu] = {getOnIpuTensor(data[i], graph,
                                             ipuMapping[ipu]), i, j};
      }
    }
  }
  return chunks;
}

static void
opInPlace(Graph &graph, popops::Operation op, const Tensor &a,
          const Tensor &b, Sequence &prog, const std::string &debugPrefix) {
  using namespace popops::expr;
  switch (op) {
  case Operation::ADD: return addInPlace(graph, a, b, prog, debugPrefix);
  case Operation::MUL: return mulInPlace(graph, a, b, prog, debugPrefix);
  case Operation::MIN:
    return minInPlace(graph, a, b, prog, debugPrefix);
  case Operation::MAX:
    return maxInPlace(graph, a, b, prog, debugPrefix);
  case Operation::LOGICAL_AND:
    return logicalAndInPlace(graph, a, b, prog, debugPrefix);
  case Operation::LOGICAL_OR:
    return logicalOrInPlace(graph, a, b, prog, debugPrefix);
  case Operation::SQUARE_ADD:
    throw poputil::poplibs_error("Collective reduction using the SQUARE_ADD "
                                 "operation is not yet supported");
  }
}

static void
ringReduceScatterStep(Graph &graph,
                      const std::vector<std::vector<Tensor>> &fragments,
                      const std::vector<unsigned> &ring,
                      unsigned step,
                      std::vector<Tensor> &data,
                      std::vector<Tensor> &copySrcs,
                      std::vector<Tensor> &copyDsts,
                      std::vector<Tensor> &addOp0,
                      std::vector<Tensor> &addOp1,
                      bool clockwise) {
  const auto numFragments = fragments.size();
  const auto numSteps = numFragments - 1;
  assert(step < numSteps);
  std::vector<Tensor> nextData(numFragments);
  for (unsigned i = 0; i != numFragments; ++i) {
    unsigned fragmentNum;
    unsigned recvIndex;
    if (clockwise) {
      // At each step the IPU at index N in the ring receives data from the
      // previous IPU in the ring and reduces it with a local fragment and
      // sends the data it reduced in the previous step to the next IPU in the
      // ring. We want the IPU at index N to have the reduced result for the
      // N'th fragment after the final step and so working backwards in step
      // (numSteps - 1 - x) the IPU at index (N - x) % numFragments should
      // reduce the data it receives with the N'th fragment. This can be
      // rearranged to give the following expression for the fragment that the
      // IPU at index i should reduce at each step:
      fragmentNum = (i + numSteps - 1 - step) % numFragments;
      recvIndex = (i + numFragments - 1) % numFragments;
    } else {
      // At each step the IPU at index N in the ring receives data from the
      // next IPU in the ring and reduces it with local fragment and sends
      // the data it reduced in the previous step to the previous IPU in the
      // ring. In step (numSteps - 1 - x) the IPU at index (N + x) %
      // numFragments should reduce the data it receives with the N'th fragment.
      // Again this can be rearranged to give the following expression for the
      // fragment that the IPU at index i should reduce at each step:
      fragmentNum = (i - (numSteps - 1 - step)) % numFragments;
      recvIndex = (i + 1) % numFragments;
    }
    auto rank = ring[i];
    auto recvRank = ring[recvIndex];
    auto copySrc = step == 0 ? fragments[recvRank][fragmentNum] :
                              data[recvIndex];
    auto copyDst = cloneToRank(graph, copySrc, rank, ring.size());
    copySrcs.push_back(copySrc);
    copyDsts.push_back(copyDst);
    addOp0.push_back(copyDst);
    addOp1.push_back(fragments[rank][fragmentNum]);
    nextData[i] = copyDst;
  }
  data = nextData;
}

static void
meetInMiddleReduceScatterStep(Graph &graph,
                              const std::vector<std::vector<Tensor>> &fragments,
                              const std::vector<unsigned> &ring,
                              unsigned step,
                              std::vector<Tensor> &clockwiseData,
                              std::vector<Tensor> &anticlockwiseData,
                              std::vector<Tensor> &copySrcs,
                              std::vector<Tensor> &copyDsts,
                              std::vector<Tensor> &addOp0,
                              std::vector<Tensor> &addOp1) {
  // The slice that ends up on a single IPU is calculated as follows:
  // In the first step:
  // - The IPU (numIpus - 1) / 2 hops in the anticlockwise direction sends
  //   it's fragment clockwise.
  // - The IPU numIpus / 2 hops in the anticlockwise direction sends
  //   it's fragment clockwise.
  // In each subsequent step each IPU takes the fragment it received in
  // the previous step, adds it to the corresponding local fragment and
  // send the result to the next IPU along the ring until the data reaches
  // the final IPU. After numIPU / 2 steps all the data has reached the
  // final IPU.
  const auto numRanks = ring.size();
  assert(fragments.size() == numRanks);
  const auto numSteps = numRanks / 2;
  assert(numRanks % 2 == 0);
  assert(numRanks > 2);
  assert(step < numSteps);
  std::vector<Tensor> nextClockwiseData(numRanks);
  std::vector<Tensor> nextAnticlockwiseData(numRanks);
  for (bool clockwise : {true, false}) {
    if (clockwise && step == numSteps - 1)
      continue;
    for (unsigned i = 0; i != numRanks; ++i) {
      unsigned fragmentNum;
      unsigned recvIndex;
      if (clockwise) {
        // At each step the IPU at index N in the ring receives data from the
        // previous IPU in the ring and reduces it with a local fragment and
        // sends the data it reduced in the previous step to the next IPU in the
        // ring. We want the IPU at index N to have the reduced result for the
        // N'th fragments sent clockwise after the penultimate step and so
        // working backwards in step (numSteps - 2 - x) the IPU at index
        // (N - x) % numIpus should reduce the data it receives with the N'th
        // fragment. This can be rearranged to give the following expression for
        // the fragment that the IPU at index i should reduce at each step:
        fragmentNum = (i + numSteps - 2 - step) % numRanks;
        recvIndex = (i + numRanks - 1) % numRanks;
      } else {
        // At each step the IPU at index N in the ring receives data from the
        // next IPU in the ring and reduces it with local fragment and sends
        // the data it reduced in the previous step to the previous IPU in the
        // ring. In step (numSteps - 1 - x) the IPU at index (N + x) % numIpus
        // should reduce the data it receives with the N'th fragment. Again this
        // can be rearranged to give the following expression for the fragment
        // that the IPU at index i should reduce at each step:
        fragmentNum = (i - (numSteps - 1 - step)) % numRanks;
        recvIndex = (i + 1) % numRanks;
      }
      std::vector<Tensor> &data = clockwise ? clockwiseData : anticlockwiseData;
      auto rank = ring[i];
      auto recvRank = ring[recvIndex];
      auto copySrc = step == 0 ? fragments[recvRank][fragmentNum] :
                                 data[recvIndex];
      auto copyDst = cloneToRank(graph, copySrc, rank, ring.size());
      copySrcs.push_back(copySrc);
      copyDsts.push_back(copyDst);
      addOp0.push_back(copyDst);
      if (step == numSteps - 1) {
        assert(!clockwise);
        addOp1.push_back(clockwiseData[i]);
      } else {
        addOp1.push_back(fragments[rank][fragmentNum]);
      }
      std::vector<Tensor> &nextData = clockwise ? nextClockwiseData :
                                                  nextAnticlockwiseData;
      nextData[i] = copyDst;
    }
  }
  clockwiseData = nextClockwiseData;
  anticlockwiseData = nextAnticlockwiseData;
}

// Perform a collective reduce scatter operation.
static Chunks
unidirectionalRingReduceScatter(Graph &graph, const Tensor toReduce,
                                popops::Operation op, Sequence &prog,
                                bool clockwise,
                                const std::string &debugPrefix) {
  const auto numPartials = toReduce.dim(0);
  if (numPartials == 1) {
    Chunks result(1);
    result.chunks[0] = Chunk(poputil::duplicate(graph, toReduce[0], prog),
                             0, 0);
    result.originalInput = toReduce;
    return result;
  }
  auto ring = arrangeInRing(toReduce);
  const unsigned ipusPerRank = graph.getTarget().getNumIPUs() / ring.size();
  const auto numFragments = numPartials;
  auto fragments = splitIntoFragments(toReduce, numFragments,
                                      graph, ipusPerRank);
  // Temporary data indexed by the position in the ring and ipu in rank.
  std::vector<Tensor> data;
  const auto numSteps = ring.size() - 1;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    std::vector<Tensor> addOp0;
    std::vector<Tensor> addOp1;
    ringReduceScatterStep(graph, fragments, ring, step, data,
                          copySrcs, copyDsts, addOp0, addOp1, clockwise);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
    opInPlace(graph, op, concat(addOp0), concat(addOp1), prog,
              debugPrefix + "/Step" + std::to_string(step));
  }
  // indexed by partial rank then ipu with in rank
  return createChunks(graph, toReduce, data, ipusPerRank, numPartials);
}

static Chunks
ringMeetInMiddleReduceScatter(Graph &graph, const Tensor &toReduce,
                              popops::Operation op, Sequence &prog,
                              const std::string &debugPrefix) {
  const auto numPartials = toReduce.dim(0);
  if (numPartials == 1) {
    Chunks result(1);
    result.chunks[0] = Chunk(poputil::duplicate(graph, toReduce[0], prog),
                                                0, ~0U);
    return result;
  }
  if (numPartials == 2) {
    return unidirectionalRingReduceScatter(graph, toReduce,
                                           op, prog, true, debugPrefix);
  }
  auto ring = arrangeInRing(toReduce);
  const auto numFragments = numPartials;
  const unsigned ipusPerRank = graph.getTarget().getNumIPUs() / ring.size();
  auto fragments = splitIntoFragments(toReduce, numFragments,
                                      graph, ipusPerRank);

  std::vector<Tensor> clockwiseData;
  std::vector<Tensor> anticlockwiseData;
  const auto numSteps = numPartials / 2;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    std::vector<Tensor> addOp0;
    std::vector<Tensor> addOp1;
    meetInMiddleReduceScatterStep(graph, fragments, ring, step,
                                  clockwiseData, anticlockwiseData, copySrcs,
                                  copyDsts, addOp0, addOp1);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
    opInPlace(graph, op, concat(addOp0), concat(addOp1), prog,
              debugPrefix + "/Step" + std::to_string(step));
  }

  return createChunks(graph, toReduce, anticlockwiseData,
                      ipusPerRank, numPartials);
}

static Chunks
bidirectionalRingPairReduceScatter(Graph &graph, const Tensor &toReduce,
                                   popops::Operation op, Sequence &prog,
                                   const std::string &debugPrefix) {
  const auto numPartials = toReduce.dim(0);
  if (numPartials == 1) {
    Chunks result(1);
    result.chunks[0] = Chunk(poputil::duplicate(graph, toReduce[0], prog),
                                                0, ~0U);
    return result;
  }
  auto ring = arrangeInRing(toReduce);
  const unsigned ipusPerRank = graph.getTarget().getNumIPUs() / ring.size();
  const auto numFragments = numPartials * 2;
  auto fragments = splitIntoFragments(toReduce, numFragments,
                                      graph, ipusPerRank);
  // Split the fragments into two sets - even fragments are reduced
  // clockwise around the ring and odd fragments are reduced anticlockwise
  // around the ring.
  std::vector<std::vector<Tensor>> clockwiseFragments(numPartials);
  std::vector<std::vector<Tensor>> anticlockwiseFragments(numPartials);
  for (unsigned i = 0; i != numFragments; i +=2) {
    for (unsigned ipu = 0; ipu != numPartials; ++ipu) {
      clockwiseFragments[ipu].push_back(fragments[ipu][i]);
      anticlockwiseFragments[ipu].push_back(fragments[ipu][i + 1]);
    }
  }

  std::vector<Tensor> clockwiseData;
  std::vector<Tensor> anticlockwiseData;
  const auto numSteps = numPartials - 1;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    std::vector<Tensor> addOp0;
    std::vector<Tensor> addOp1;
    ringReduceScatterStep(graph, clockwiseFragments, ring, step,
                          clockwiseData, copySrcs, copyDsts, addOp0, addOp1,
                          true);
    ringReduceScatterStep(graph, anticlockwiseFragments, ring, step,
                          anticlockwiseData, copySrcs, copyDsts, addOp0, addOp1,
                          false);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
    opInPlace(graph, op, concat(addOp0), concat(addOp1), prog,
              debugPrefix + "/Step" + std::to_string(step));
  }
  std::vector<Tensor> data;
  for (unsigned i = 0; i != numPartials; ++i) {
    data.push_back(concat(clockwiseData[i], anticlockwiseData[i]));
  }
  return createChunks(graph, toReduce, data, ipusPerRank, numPartials);
}

static CollectiveMethod
pickReduceScatterMethod(Graph &graph, const Tensor &t,
                        popops::Operation op) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  if (t.dim(0) != numIpus || numIpus <= 2)
    return CollectiveMethod::CLOCKWISE_RING;
  const auto &target = graph.getTarget();
  unsigned bytesPerIpu = t.numElements() *
                         target.getTypeSize(t.elementType()) / numIpus;
  // Thresholds where the BIDIRECTIONAL_RING_PAIR method starts to beat the
  // MEET_IN_MIDDLE_RING method determined experimentally.
  if (bytesPerIpu < 1245184 ||
      (numIpus > 4 && bytesPerIpu < 4980736) ||
      (numIpus > 8 && bytesPerIpu < 39845888)) {
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  return CollectiveMethod::BIDIRECTIONAL_RING_PAIR;
}

static CollectiveMethod
pickAllGatherMethod(Graph &graph, const std::vector<Chunk> &toGather) {
  const auto numIpus = graph.getTarget().getNumIPUs();
  if (toGather.size() != numIpus || numIpus <= 2)
    return CollectiveMethod::CLOCKWISE_RING;
  const auto &target = graph.getTarget();
  const auto numBytes =
    std::accumulate(toGather.begin(), toGather.end(), 0,
                    [&](unsigned n, const Chunk &c) {
      return n + c.tensor.numElements() +
             target.getTypeSize(c.tensor.elementType());
    });
  unsigned bytesPerIpu = numBytes / numIpus;
  // Thresholds where the BIDIRECTIONAL_RING_PAIR method starts to beat the
  // MEET_IN_MIDDLE_RING method determined experimentally.
  if (bytesPerIpu < 622592 ||
      (numIpus > 4 && bytesPerIpu < 2490368) ||
      (numIpus > 8 && bytesPerIpu < 19922944)) {
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  return CollectiveMethod::BIDIRECTIONAL_RING_PAIR;
}

static Chunks
internalReduceScatter(Graph &graph, const Tensor &toReduce,
                      popops::Operation op,
                      Sequence &prog, const std::string &debugPrefix,
                      const poplar::OptionFlags &options) {
  if (toReduce.rank() != 2) {
    poputil::poplibs_error("Reduce scatter input tensor does not have rank 2");
  }
  checkTensorIpuMappings(graph, toReduce);
  CollectiveMethod method = parseCollectiveOptions(options);
  if (method == CollectiveMethod::AUTO) {
    method = pickReduceScatterMethod(graph, toReduce, op);
  }
  switch (method) {
  default: assert(0 && "Unexpected reduce method");
  case CollectiveMethod::CLOCKWISE_RING:
    return unidirectionalRingReduceScatter(graph, toReduce, op, prog, true,
                                           debugPrefix);
  case CollectiveMethod::ANTICLOCKWISE_RING:
    return unidirectionalRingReduceScatter(graph, toReduce, op, prog, false,
                                           debugPrefix);
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR:
    return bidirectionalRingPairReduceScatter(graph, toReduce, op, prog,
                                              debugPrefix);
  case CollectiveMethod::MEET_IN_MIDDLE_RING:
    return ringMeetInMiddleReduceScatter(graph, toReduce, op, prog,
                                         debugPrefix);
  }
}

static void
ringAllGatherStep(Graph &graph,
                  const std::vector<Chunk> &toGather,
                  const std::vector<unsigned> &ring,
                  unsigned step,
                  std::vector<Chunk> &data,
                  std::vector<std::vector<Tensor>> &resultChunks,
                  std::vector<Tensor> &copySrcs,
                  std::vector<Tensor> &copyDsts,
                  bool clockwise) {
  const auto numChunksToGather = toGather.size();
  std::vector<Chunk> nextData(numChunksToGather);
  for (unsigned i = 0; i != numChunksToGather; ++i) {
    auto rank = ring[i];
    unsigned recvIndex;
    if (clockwise) {
      recvIndex = (i + numChunksToGather - 1) % numChunksToGather;
    } else {
      recvIndex = (i + 1) % numChunksToGather;
    }
    auto &copySrc = step == 0 ? toGather[ring[recvIndex]] :
                                data[recvIndex];
    auto copyDst = cloneToRank(graph, copySrc.tensor, rank, ring.size());
    copySrcs.push_back(copySrc.tensor);
    copyDsts.push_back(copyDst);
    nextData[i] = { copyDst, copySrc.index, ~0U };
    resultChunks[rank][copySrc.index] = copyDst;
  }
  data = nextData;
}

static void
ringMeetInMiddleAllGatherStep(Graph &graph,
                              const std::vector<Chunk> &toGather,
                              const std::vector<unsigned> &ring,
                              unsigned step,
                              std::vector<Chunk> &clockwiseData,
                              std::vector<Chunk> &anticlockwiseData,
                              std::vector<std::vector<Tensor>> &resultChunks,
                              std::vector<Tensor> &copySrcs,
                              std::vector<Tensor> &copyDsts) {
  const auto numChunksToGather = toGather.size();
  const auto numSteps = numChunksToGather / 2;
  assert(step < numSteps);
  std::vector<Chunk> nextClockwiseData(numChunksToGather);
  std::vector<Chunk> nextAnticlockwiseData(numChunksToGather);
  for (unsigned i = 0; i != numChunksToGather; ++i) {
    for (bool clockwise : {true, false}) {
      if (clockwise && step == numSteps - 1)
        continue;
      auto rank = ring[i];
      unsigned recvIndex;
      if (clockwise) {
        recvIndex = (i + numChunksToGather - 1) % numChunksToGather;
      } else {
        recvIndex = (i + 1) % numChunksToGather;
      }
      auto &data = clockwise ? clockwiseData : anticlockwiseData;
      auto &copySrc = step == 0 ? toGather[ring[recvIndex]] :
                                  data[recvIndex];
      auto copyDst = cloneToRank(graph, copySrc.tensor, rank, ring.size());
      copySrcs.push_back(copySrc.tensor);
      copyDsts.push_back(copyDst);
      auto &nextData = clockwise ? nextClockwiseData : nextAnticlockwiseData;
      nextData[i] = { copyDst, copySrc.index, ~0U };
      resultChunks[rank][copySrc.index] = copyDst;
    }
  }
  clockwiseData = nextClockwiseData;
  anticlockwiseData = nextAnticlockwiseData;
}

// The chunks returned from the scatter step are per ipu and not ordered.
// The all gather steps need the equivalent chunks in one tensor so that
// they can merely clone them to the desired rank. This function concats
// all chunks with the same index
static std::vector<Chunk>
concatModelParallelChunks(std::vector<Chunk> toGather, Graph &graph) {

  const unsigned ringSize = (std::max_element(toGather.begin(), toGather.end(),
      [&] (const Chunk A, const Chunk B) {
        return A.index < B.index;
      }))->index + 1;

  assert(graph.getTarget().getNumIPUs() % ringSize == 0);
  const unsigned ipusPerRank = graph.getTarget().getNumIPUs() / ringSize;
  if (ipusPerRank == 1) {
    return toGather;
  }
  // ensure that chunks are ordered by offset to make sure that the
  // concatenation order is correct
  std::sort(toGather.begin(), toGather.end(), [&] (Chunk A, Chunk B) {
    return A.offset < B.offset;
  });

  const auto ring = createRing(ringSize);
  std::vector<Chunk> concatenated(ringSize);
  std::vector<std::vector<Tensor>> tensorPieces(ringSize);

  for (const auto &chunk : toGather) {
    tensorPieces[ring[chunk.index]].push_back(chunk.tensor);
    concatenated[ring[chunk.index]].index = chunk.index;
    // not used just setting to this for debug
    concatenated[ring[chunk.index]].offset = ~0U;
  }
  for (unsigned i = 0; i < ringSize; ++i) {
    concatenated[i].tensor = concat(tensorPieces[i]);
  }
  return concatenated;
}

static bool
isMappedRank(const Tensor &tensor, const Graph &graph, const unsigned rank,
              const unsigned ipusPerRank) {
  const auto &m = graph.getTileMapping(tensor);
  const auto tilesPerRank = ipusPerRank * graph.getTarget().getTilesPerIPU();
  for (unsigned tile = 0; tile < graph.getTarget().getNumTiles(); ++tile) {
    if (!m[tile].empty() && (tile / tilesPerRank) != rank) {
      return false;
    }
  }
  return true;
}

struct ipuIndexPair {
  unsigned ipu;
  unsigned index;
};

// Through out the all reduce the tensors created have had their order
// shuffled within the rank inorder to send corresponding fragments on
// different ipus with in the rank in the same step. This means that the
// returned tensors indexing won't match the orginal. To fix this must
// reorder the tensor before returning it. As splitting into fragments
// and concatting the modelParallel chunks will have maintained order
// on each ipu. Can use the ipu mappings of the original input tensor
// to work out the required ordering for the returned tensor, must be
// careful not to merge continuous intervals when getting ipu mapping
// as what might be continuous intervals on one input might not be in the
// result
static Tensor
reorderRank(Graph &graph, const std::vector<Tensor> &partials,
            const Tensor &originalInput,
            const unsigned ipusPerRank, const unsigned rank) {
  if (partials.empty()) {
    return Tensor();
  }
  if (ipusPerRank == 1) {
    // no need to reorder rank as already ordered
    return concat(partials);
  }
  // error checking
  assert(isMappedRank(originalInput, graph, rank, ipusPerRank));
  for (unsigned i = 0; i < partials.size(); ++i) {
    assert(isMappedRank(partials[i], graph, rank, ipusPerRank));
  }
  // get each ipus part of the tensor separately
  std::vector<std::vector<Tensor>> perIPUTensors(partials.size());
  for (unsigned i = 0; i < partials.size(); ++i) {
    const auto ipuMapping = getIpuMapping(graph, partials[i]);
    for (unsigned j = 0; j < ipusPerRank; ++j) {
      perIPUTensors[i].push_back(
          getOnIpuTensor(partials[i], graph,
                         ipuMapping[((rank * ipusPerRank) + j)]));
    }
  }
  std::vector<Tensor> unorderedPieces;
  for (unsigned j = 0; j < ipusPerRank; ++j) {
    for (unsigned i = 0; i < partials.size(); ++i) {
      unorderedPieces.push_back(perIPUTensors[i][j]);
    }
  }
  auto unorderedResult = concat(unorderedPieces);

  const auto orderedMapping = getIpuMapping(graph, originalInput);
  auto unorderedMapping = getIpuMapping(graph, unorderedResult);

  matchIntervals(unorderedMapping, orderedMapping);

  std::map<unsigned, ipuIndexPair> concatMap;

  for (unsigned ipu = 0; ipu < orderedMapping.size(); ++ipu) {
    for (unsigned index = 0; index < orderedMapping[ipu].size(); ++index) {
      const auto it = concatMap.insert({orderedMapping[ipu][index].begin(),
                                         {ipu, index}});
      if (!it.second) {
        throw poputil::poplibs_error("No element should be inserted twice");
      }
    }
  }

  std::vector<Tensor> orderedPieces;
  for (const auto &values : concatMap) {
    const auto &newInterval =
              unorderedMapping[values.second.ipu][values.second.index];
    orderedPieces.push_back(
                unorderedResult.slice(newInterval.begin(), newInterval.end()));
  }

  return concat(orderedPieces);
}


// convert the chunks produced from the all gather steps into the final
// one single tensor
static Tensor
convertChunksToTensor(Graph &graph,
                      const std::vector<std::vector<Tensor>> &resultChunks,
                      const Tensor &originalInput,
                      const unsigned ipusPerRank,
                      const unsigned numChunksToGather) {
  std::vector<Tensor> result;
  result.reserve(numChunksToGather);
  for (unsigned rank = 0; rank != numChunksToGather; ++rank) {
    if (originalInput.rank() != 2) {
      throw poputil::poplibs_error("Input tensor should be rank 2");
    }
    result.push_back(reorderRank(graph, resultChunks[rank],
                                 originalInput[rank],
                                 ipusPerRank, rank).expand({0}));
  }
  return concat(result);
}

static Tensor
unidirectionalRingAllGather(Graph &graph, const Chunks &toGatherChunks,
                            Sequence &prog, bool clockwise) {
  const auto &toGather = toGatherChunks.chunks;
  if (toGather.size() == 1) {
    return toGather[0].tensor;
  }
  const auto concatenatedChunks = concatModelParallelChunks(toGather, graph);
  const auto numChunksToGather = concatenatedChunks.size();
  const unsigned ipusPerRank = toGather.size() / numChunksToGather;
  auto ring = arrangeInRing(concatenatedChunks);
  std::vector<std::vector<Tensor>>
      resultChunks(numChunksToGather, std::vector<Tensor>(numChunksToGather));
  for (unsigned rank = 0 ; rank != numChunksToGather; ++rank) {
    resultChunks[rank][concatenatedChunks[rank].index] =
        poputil::duplicate(graph, concatenatedChunks[rank].tensor, prog);
  }
  const auto numSteps = ring.size() - 1;
  std::vector<Chunk> data;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    ringAllGatherStep(graph, concatenatedChunks, ring, step, data, resultChunks,
                      copySrcs, copyDsts, clockwise);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
  }
  return convertChunksToTensor(graph, resultChunks,
                               toGatherChunks.originalInput,
                               ipusPerRank, numChunksToGather);
}

static Tensor
bidirectionalRingPairAllGather(Graph &graph,
                               const Chunks &toGatherChunks,
                               Sequence &prog) {
  const auto &toGather = toGatherChunks.chunks;
  if (toGather.size() == 1) {
    return toGather[0].tensor;
  }
  const auto concatenatedChunks = concatModelParallelChunks(toGather, graph);
  const auto numChunksToGather = concatenatedChunks.size();
  auto ring = arrangeInRing(concatenatedChunks);
  // Split the each chunk into two parts - one part that is sent clockwise
  // around the ring and one part that is sent anticlockwise around the
  // ring.
  const unsigned ipusPerRank = graph.getTarget().getNumIPUs() / ring.size();
  std::vector<Chunk> clockwiseToGather;
  std::vector<Chunk> anticlockwiseToGather;
  std::vector<std::vector<Tensor>>
      clockwiseResultChunks(numChunksToGather,
                            std::vector<Tensor>(numChunksToGather));
  std::vector<std::vector<Tensor>>
      anticlockwiseResultChunks(numChunksToGather,
                                std::vector<Tensor>(numChunksToGather));
  for (unsigned rank = 0 ; rank != numChunksToGather; ++rank) {
    auto fragments =
       splitRankIntoFragments(concatenatedChunks[rank].tensor, graph,
                              2, rank, ipusPerRank);
    const auto index = concatenatedChunks[rank].index;
    clockwiseToGather.push_back({fragments.at(0), index, ~0U});
    anticlockwiseToGather.push_back({fragments.at(1), index, ~0U});
    clockwiseResultChunks[rank][index] =
        poputil::duplicate(graph, fragments.at(0), prog);
    anticlockwiseResultChunks[rank][index] =
        poputil::duplicate(graph, fragments.at(1), prog);
  }
  const auto numSteps = ring.size() - 1;
  std::vector<Chunk> clockwiseData, anticlockwiseData;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    ringAllGatherStep(graph, clockwiseToGather, ring, step, clockwiseData,
                      clockwiseResultChunks, copySrcs, copyDsts, true);
    ringAllGatherStep(graph, anticlockwiseToGather, ring, step,
                      anticlockwiseData, anticlockwiseResultChunks, copySrcs,
                      copyDsts, false);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
  }
  std::vector<Tensor> result;
  result.reserve(numChunksToGather);
  for (unsigned rank = 0; rank != numChunksToGather; ++rank) {
    std::vector<Tensor> resultChunks;
    for (unsigned chunk = 0 ; chunk != numChunksToGather; ++chunk) {
      resultChunks.push_back(clockwiseResultChunks[rank][chunk]);
      resultChunks.push_back(anticlockwiseResultChunks[rank][chunk]);
    }
    if (toGatherChunks.originalInput.rank() != 2) {
      throw poputil::poplibs_error("wrong rank");
    }
    result.push_back(reorderRank(graph, resultChunks,
                                 toGatherChunks.originalInput[rank],
                                 ipusPerRank, rank).expand({0}));
  }
  return concat(result);
}

static Tensor
ringMeetInMiddleAllGather(Graph &graph,
                          const Chunks &toGatherChunks,
                          Sequence &prog) {
  const auto &toGather = toGatherChunks.chunks;
  if (toGather.size() == 1) {
    return toGather[0].tensor;
  }
  const auto concatenatedChunks = concatModelParallelChunks(toGather, graph);
  const auto numChunksToGather = concatenatedChunks.size();
  const unsigned ipusPerRank = toGather.size() / numChunksToGather;
  auto ring = arrangeInRing(concatenatedChunks);
  std::vector<std::vector<Tensor>>
      resultChunks(numChunksToGather, std::vector<Tensor>(numChunksToGather));
  for (unsigned rank = 0; rank != numChunksToGather; ++rank) {
    resultChunks[rank][concatenatedChunks[rank].index] =
        poputil::duplicate(graph, concatenatedChunks[rank].tensor, prog);
  }
  const auto numSteps = ring.size() / 2;
  std::vector<Chunk> clockwiseData, anticlockwiseData;
  for (unsigned step = 0; step != numSteps; ++step) {
    std::vector<Tensor> copySrcs;
    std::vector<Tensor> copyDsts;
    ringMeetInMiddleAllGatherStep(graph, concatenatedChunks,
                                  ring, step, clockwiseData,
                                  anticlockwiseData, resultChunks, copySrcs,
                                  copyDsts);
    prog.add(Copy(concat(copySrcs), concat(copyDsts)));
  }
  return convertChunksToTensor(graph, resultChunks,
                               toGatherChunks.originalInput,
                               ipusPerRank, numChunksToGather);
}

static Tensor
internalAllGather(Graph &graph, const Chunks &toGather,
          Sequence &prog, const std::string &,
          const poplar::OptionFlags &options) {
  const auto numChunks = toGather.chunks.size();
  for (unsigned i = 0; i != numChunks; ++i) {
    if (toGather.chunks[i].tensor.rank() != 1) {
      poputil::poplibs_error("All gather input chunk " + std::to_string(i) +
                             " does not have rank 1");
    }
  }
  CollectiveMethod method = parseCollectiveOptions(options);
  if (method == CollectiveMethod::AUTO) {
    method = pickAllGatherMethod(graph, toGather.chunks);
  }
  switch (method) {
  default: assert(0 && "Unexpected reduce method");
  case CollectiveMethod::CLOCKWISE_RING:
    return unidirectionalRingAllGather(graph, toGather, prog, true);
  case CollectiveMethod::ANTICLOCKWISE_RING:
    return unidirectionalRingAllGather(graph, toGather, prog, false);
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR:
    return bidirectionalRingPairAllGather(graph, toGather, prog);
  case CollectiveMethod::MEET_IN_MIDDLE_RING:
    return ringMeetInMiddleAllGather(graph, toGather, prog);
  }
}

Chunks
reduceScatter(Graph &graph, const Tensor &toReduce, popops::Operation op,
              Sequence &prog, const std::string &debugPrefix,
              const poplar::OptionFlags &options) {
  if (toReduce.dim(0) != graph.getTarget().getNumIPUs()) {
    throw poputil::poplibs_error("Multi ipu ranks are not yet supported for "
                                 "reduceScatter");
  }
  return internalReduceScatter(graph, toReduce, op,
                               prog, debugPrefix, options);
}

Tensor
allGather(Graph &graph, const Chunks &toGather,
          Sequence &prog, const std::string &,
          const poplar::OptionFlags &options) {
  if (toGather.originalInput.dim(0) != graph.getTarget().getNumIPUs()) {
    throw poputil::poplibs_error("Multi ipu ranks are not yet supported for "
                                 "reduceScatter");
  }
  return internalAllGather(graph, toGather, prog, "", options);
}

poplar::Tensor
allReduce(poplar::Graph &graph, const poplar::Tensor &toReduce,
          popops::Operation op, poplar::program::Sequence &prog,
          const std::string &debugPrefix,
          const poplar::OptionFlags &options) {
  auto flattened = toReduce.flatten(1, toReduce.rank());
  auto scatteredResult = internalReduceScatter(graph, flattened, op,
                                               prog, debugPrefix,
                                               options);
  auto gatheredResult = internalAllGather(graph, scatteredResult,
                                          prog, debugPrefix,
                                          options);
  return gatheredResult.reshape(toReduce.shape());
}

poplar::Tensor
replicatedAllReduce(Graph &graph, Graph &parentGraph,
                    const poplar::Tensor &data,
                    popops::Operation op,
                    program::Sequence &prog,
                    const std::string &debugPrefix,
                    const poplar::OptionFlags &options) {
  auto result = graph.clone(data);
  auto dataReordered = data.flatten();
  auto resultReordered = result.flatten();
  graph.reorderToSimplify(&dataReordered, {&resultReordered});
  auto reduced =
      allReduce(parentGraph, parentGraph.getNonReplicatedTensor(dataReordered),
                op, prog, debugPrefix, options);
  prog.add(Copy(reduced, parentGraph.getNonReplicatedTensor(resultReordered)));
  return result;
}

} // End namespace popops
