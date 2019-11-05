#include "popops/Collectives.hpp"

#include "CollectivesProgram.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/OptionParsing.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/DynamicSlice.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Pad.hpp"
#include "popops/Reduce.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <boost/optional/optional.hpp>
#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace popops::expr;

namespace {

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

struct CollectiveOptions {
  CollectiveMethod method = CollectiveMethod::AUTO;
  bool useReplicatedImplementation = false;
};

} // End anonymous namespace.

namespace popops {

static std::vector<unsigned>
invertPermutation(const std::vector<unsigned> &permutation) {
  std::vector<unsigned> inverse(permutation.size());
  for (unsigned i = 0; i != permutation.size(); ++i) {
    inverse[permutation[i]] = i;
  }
  return inverse;
}

// Return the IPUs in clockwise direction around the ring starting at IPU 0.
static std::vector<unsigned> createRing(const unsigned n) {
  std::vector<unsigned> ring(n);
  unsigned i = 0;
  std::generate(ring.begin(), ring.begin() + ((n + 1) / 2), [&] {
    i += 2;
    return i - 2;
  });
  if ((n & 1) != 0) {
    i -= 3;
  } else {
    --i;
  }
  std::generate(ring.begin() + ((n + 1) / 2), ring.end(), [&] {
    i -= 2;
    return i + 2;
  });
  return ring;
}

namespace {

class RingTopology {
  // IPUs in clockwise direction around the ring starting at IPU 0.
  std::vector<unsigned> ringIndexToRank;
  std::vector<unsigned> rankToRingIndex;

public:
  RingTopology(unsigned n) {
    ringIndexToRank = createRing(n);
    rankToRingIndex = invertPermutation(ringIndexToRank);
  }

  /// Return the number of IPU that is the specified number of steps in the
  /// specified direction around the ring, starting at the specified base
  /// IPU.
  unsigned getRank(unsigned base, Direction direction, unsigned steps) const {
    auto numRanks = ringIndexToRank.size();
    auto index = rankToRingIndex[base];
    switch (direction) {
    case CLOCKWISE:
      index = (index + steps) % numRanks;
      break;
    case ANTICLOCKWISE:
      steps = steps % numRanks;
      index = (index + numRanks - steps) % numRanks;
      break;
    }
    return ringIndexToRank[index];
  }
  unsigned indexToRank(const unsigned index) const {
    return ringIndexToRank[index];
  }
  unsigned rankToIndex(const unsigned rank) const {
    return rankToRingIndex[rank];
  }
};

} // End anonymous namespace.

static void parseCollectiveOptions(const poplar::OptionFlags &optionFlags,
                                   CollectiveOptions &options) {
  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  const OptionSpec spec{
      {"method",
       OptionHandler::createWithEnum(
           options.method,
           {{"auto", CollectiveMethod::AUTO},
            {"clockwise_ring", CollectiveMethod::CLOCKWISE_RING},
            {"anticlockwise_ring", CollectiveMethod::ANTICLOCKWISE_RING},
            {"bidirectional_ring_pair",
             CollectiveMethod::BIDIRECTIONAL_RING_PAIR},
            {"meet_in_middle_ring", CollectiveMethod::MEET_IN_MIDDLE_RING}})},
      {"useReplicatedImplementation",
       OptionHandler::createWithBool(options.useReplicatedImplementation)}};
  for (const auto &entry : optionFlags) {
    spec.parse(entry.first, entry.second);
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
static std::vector<std::vector<Interval>> getIpuMapping(const Graph &graph,
                                                        const Tensor &t) {
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
              [&](Interval A, Interval B) { return A.begin() < B.begin(); });
  }

  // compress intervals
  std::vector<std::vector<Interval>> result(ipuMapping.size());
  for (unsigned ipu = 0; ipu < ipuMapping.size(); ++ipu) {
    for (unsigned i = 0; i < ipuMapping[ipu].size(); ++i) {
      if (result[ipu].empty() ||
          result[ipu].back().end() != ipuMapping[ipu][i].begin()) {
        result[ipu].push_back(ipuMapping[ipu][i]);
      } else {
        result[ipu].back() =
            Interval(result[ipu].back().begin(), ipuMapping[ipu][i].end());
      }
    }
  }
  return result;
}

static std::vector<std::size_t> getNumElementsPerIpu(const Graph &graph,
                                                     const Tensor &t) {
  const auto tilesPerIpu = graph.getTarget().getTilesPerIPU();
  const auto &tileMapping = graph.getTileMapping(t);
  std::vector<std::size_t> numElements(graph.getTarget().getNumIPUs());
  for (unsigned tile = 0; tile < tileMapping.size(); ++tile) {
    const unsigned ipu = tile / tilesPerIpu;
    for (const auto &interval : tileMapping[tile]) {
      numElements[ipu] += interval.size();
    }
  }
  return numElements;
}

static Tensor concatSlices(const Tensor &t, Graph &graph,
                           const std::vector<Interval> &intervals) {
  assert(t.rank() == 1);
  std::vector<Tensor> toConcat;
  for (const auto &interval : intervals) {
    toConcat.push_back(t.slice(interval.begin(), interval.end()));
  }
  if (toConcat.empty()) {
    return graph.addVariable(t.elementType(), {0});
  }
  return concat(toConcat);
}

// Take a tensor and return a vector of tensors where each element
// is a slice of the original tensor that spans only one ipu.
static std::vector<Tensor> getPerIpuTensors(const Tensor &t, Graph &graph) {
  const auto ipuMapping = getIpuMapping(graph, t);
  const auto numIpus = ipuMapping.size();
  std::vector<Tensor> result;
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    result.push_back(concatSlices(t, graph, ipuMapping[ipu]));
  }
  return result;
}

static CollectiveMethod pickAllGatherMethod(const Graph &graph,
                                            const Tensor &toGather) {
  const auto ipusPerRank = graph.getTarget().getNumIPUs();
  const auto numRanks = graph.getReplicationFactor();
  if (ipusPerRank > 1 || numRanks <= 2)
    return CollectiveMethod::CLOCKWISE_RING;
  const auto &target = graph.getTarget();
  const auto bytesPerIpu = toGather.numElements() *
                           target.getTypeSize(toGather.elementType()) /
                           ipusPerRank;
  // Thresholds where the BIDIRECTIONAL_RING_PAIR method starts to beat the
  // MEET_IN_MIDDLE_RING method determined experimentally.
  // TODO: T12970 Lots has changed since these thresholds were set - check if
  // they are still appropriate.
  if (bytesPerIpu < 622592 || (numRanks > 4 && bytesPerIpu < 2490368) ||
      (numRanks > 8 && bytesPerIpu < 19922944)) {
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  return CollectiveMethod::BIDIRECTIONAL_RING_PAIR;
}

static CollectiveMethod pickReduceScatterMethod(const Graph &graph,
                                                const Tensor &t,
                                                popops::Operation op) {
  const auto ipusPerRank = graph.getTarget().getNumIPUs();
  const auto numRanks = graph.getReplicationFactor();
  if (ipusPerRank > 1 || numRanks <= 2)
    return CollectiveMethod::CLOCKWISE_RING;
  const auto &target = graph.getTarget();
  unsigned bytesPerIpu =
      t.numElements() * target.getTypeSize(t.elementType()) / ipusPerRank;
  // Thresholds where the BIDIRECTIONAL_RING_PAIR method starts to beat the
  // MEET_IN_MIDDLE_RING method determined experimentally.
  // TODO: T12970 Lots has changed since these thresholds were set - check if
  // they are still appropriate.
  if (bytesPerIpu < 1245184 || (numRanks > 4 && bytesPerIpu < 4980736) ||
      (numRanks > 8 && bytesPerIpu < 39845888)) {
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  return CollectiveMethod::BIDIRECTIONAL_RING_PAIR;
}

// Split a tensor into the specified number of fragments such that the
// number of elements and the IPU mapping of each fragment is identical,
// adding padding if necessary to achieve this.
static Tensor replicatedSplitIntoFragments(const Tensor &t,
                                           unsigned numFragments,
                                           Graph &graph) {
  logging::debug("Split into fragments");
  std::vector<Tensor> perIpuFragments;
  for (auto &ipuTensor : getPerIpuTensors(t, graph)) {
    unsigned padding =
        (numFragments - ipuTensor.dim(0) % numFragments) % numFragments;
    auto padded = pad(graph, ipuTensor, {0}, {padding}, 0.0f,
                      padding::MappingMethod::EDGE);
    auto split = padded.reshape({numFragments, padded.dim(0) / numFragments});
    perIpuFragments.push_back(split);
  }
  return concat(perIpuFragments, 1);
}

static Tensor giveFragmentsIpuOrder(const Tensor &input,
                                    const RingTopology &ring) {

  std::vector<Tensor> result(input.dim(0));
  for (unsigned ringIndex = 0; ringIndex < input.dim(0); ++ringIndex) {
    const unsigned ipuIndex = ring.indexToRank(ringIndex);
    result[ringIndex] = input[ipuIndex].expand({0});
  }
  return concat(result, 0);
}

static void internalReplicatedSlice(SliceCopy &sliceCopy, Graph &graph,
                                    Tensor &fragments, const Tensor &dst,
                                    const Direction direction) {
  std::vector<Tensor> rearrange(fragments.dim(0));
  const unsigned n = rearrange.size();
  for (unsigned i = 0; i < n; ++i) {
    int shift = direction == Direction::CLOCKWISE ? 1 : -1;
    unsigned switchIndex = (i + n + shift) % n;
    rearrange[switchIndex] = fragments[i].expand({0});
  }
  auto dslice = dynamicSlice(graph, concat(rearrange),
                             sliceCopy.getSliceIndex().expand({0}), {0}, {1},
                             sliceCopy.getCopyProgram());
  // this copy is probably avoidable
  sliceCopy.getCopyProgram().add(Copy(dslice, dst));
}

static void replicatedRankSlice(SliceCopy &sliceCopy, Graph &graph,
                                const Tensor &fragments_, const Tensor &dst,
                                const RingTopology &ring,
                                const Direction direction) {
  logging::debug("Replicated rank slice");
  assert(fragments_.rank() == dst.rank() + 1);
  assert(fragments_[0].shape() == dst.shape());
  auto fragments = giveFragmentsIpuOrder(fragments_, ring);
  const auto topGraph = graph.getTopLevelGraph();
  if (topGraph.getReplicationFactor() == 1) {
    return internalReplicatedSlice(sliceCopy, graph, fragments, dst, direction);
  }
  assert(fragments_.dim(0) == sliceCopy.cases().size());
  assert(graph.getReplicationFactor() == topGraph.getReplicationFactor());
  unsigned n = sliceCopy.cases().size();
  for (unsigned i = 0; i < n; ++i) {
    int shift = direction == Direction::CLOCKWISE ? 1 : -1;
    unsigned switchIndex = (i + n + shift) % n;
    sliceCopy.cases()[switchIndex].setCopy(Copy(fragments[i], dst), direction);
  }
}

static void internalReplicatedUpdate(SliceCopy &sliceCopy, Graph &graph,
                                     Tensor &fragments, const Tensor &src,
                                     Direction) {
  assert(src.rank() == 1);
  dynamicUpdate(graph, fragments, src.expand({0}),
                sliceCopy.getSliceIndex().expand({0}), {0}, {1},
                sliceCopy.getCopyProgram());
}

static void replicatedRankUpdate(SliceCopy &sliceCopy, Graph &graph,
                                 const Tensor &src, const Tensor &fragments_,
                                 const RingTopology &ring,
                                 const Direction direction) {
  logging::debug("replicatedRankUpdate begin");
  assert(src.rank() + 1 == fragments_.rank());
  assert(src.shape() == fragments_[0].shape());
  auto fragments = giveFragmentsIpuOrder(fragments_, ring);
  if (graph.getTopLevelGraph().getReplicationFactor() == 1) {
    return internalReplicatedUpdate(sliceCopy, graph, fragments, src,
                                    direction);
  }
  assert(graph.getReplicationFactor() ==
         graph.getTopLevelGraph().getReplicationFactor());
  const unsigned n = sliceCopy.cases().size();
  for (unsigned i = 0; i < n; ++i) {
    sliceCopy.cases()[i].setCopy(Copy(src, fragments[i]), direction);
  }
}

static CrossReplicaCopy
crossReplicaCopy(Graph &graph, const Tensor &src, const Tensor &dst,
                 std::function<unsigned(unsigned)> mapping) {
  assert(src.shape() == dst.shape());
  std::map<unsigned, unsigned> replicaMap;
  unsigned replicationFactor = graph.getReplicationFactor();
  for (unsigned i = 0; i != replicationFactor; ++i) {
    replicaMap.emplace(i, mapping(i));
  }
  return CrossReplicaCopy(src, dst, replicaMap);
}

// Map a buffer so each element is mapped to the same IPU as the
// corresponding elements in the fragments.
static void mapBuffer(Graph &graph, const Tensor &buffer,
                      const Tensor &fragments) {
  assert(buffer.numElements() == fragments[0].numElements());
  // The IPU mapping of all fragments should be identical so we only need
  // to look at the first fragment.
  auto ipuMapping = getIpuMapping(graph, fragments[0]);
  const auto numIpus = ipuMapping.size();
  unsigned tilesPerIpu = graph.getTarget().getTilesPerIPU();
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    auto virtualGraph =
        graph.createVirtualGraph(ipu * tilesPerIpu, (ipu + 1) * tilesPerIpu);
    poputil::mapTensorLinearly(
        virtualGraph, concatSlices(buffer, virtualGraph, ipuMapping[ipu]));
  }
}

// using the replication index tensor create a new tensor with
// value = the position in a clockwise ring of this replica is
// This should probably come from the ring class so can support
// topoligies that aren't clockwise or anticlockwise
static Tensor initRingIndexTensor(Graph &graph, const Direction direction,
                                  const Tensor &repIndex, Sequence &prog,
                                  const std::string &debugPrefix,
                                  const unsigned repFactor,
                                  const int startOffset) {
  // start offset allows to initialise this at different positions
  // in the ring for clockwise and anticlockwise. Used by the meet
  // in the middle method. Will often be zero but when we can
  // turn on fusing compiler should optimise the add zero away
  return popops::map(
      graph,
      Rem(Add(Add(Mul(Const(repFactor - 1), Rem(_1, Const(2))),
                  Mul(Add(Mul(Rem(_1, Const(2)), Const(-2)), Const(1)),
                      Divide(_1, Const(2)))),
              Const((repFactor + startOffset) % repFactor)),
          Const(repFactor)),
      {repIndex}, prog, debugPrefix);
  // Until mapFusion is fixed this expression generates an error. When fixed
  // re-enable generate codelet

  // Expression above initialises replica id to clockwise ring index using
  // expression
  // ID = (repFactor * (replica % 2)) +
  //        (((replica % 2) * (-2)) + 1) * (replica/2)
  // return (ID + repFactor + startOffset) % startOffset
}

// the offset is so that the meet in the middle method can start at part
// way through the iterations. can be positive or negative so that the same
// number can be used to initialise the clockwise and anticlockwise ring
static CollectivesProgram unidirectionalRingReduceScatter(
    Graph &graph, const Tensor &toReduce, popops::Operation op,
    Direction direction, const std::string &debugPrefix,
    const unsigned numSteps, const int startOffset = 0) {
  logging::info("Unidirectional ring reduce scatter");

  const auto replicationFactor = graph.getReplicationFactor();
  const RingTopology ring(replicationFactor);
  auto numFragments = replicationFactor;

  auto fragments = replicatedSplitIntoFragments(toReduce, numFragments, graph);
  auto fragmentSize = fragments.dim(1);
  auto srcBuffer = graph.addVariable(toReduce.elementType(), {fragmentSize},
                                     debugPrefix + "/ScatterSrc");
  mapBuffer(graph, srcBuffer, fragments);
  auto dstBuffer = graph.clone(srcBuffer, debugPrefix + "/ScatterDst");
  auto repFactorTensor = graph.addReplicationIndexConstant();
  graph.setTileMapping(repFactorTensor, 0);

  // If the graph is not single image replicated we can't use different
  // branches of the switch so must use dynamic slice
  CollectivesProgram program = [&]() {
    if (graph.getTopLevelGraph().getReplicationFactor() == 1) {
      return CollectivesProgram(SliceCopy(DynamicSliceCopy()));
    } else {
      return CollectivesProgram(SliceCopy(SwitchSliceCopy(replicationFactor)));
    }
  }();

  program.repeatCounter = numSteps - 1;
  program.sliceFragments.setSliceIndex(
      initRingIndexTensor(graph, direction, repFactorTensor, program.initIndex,
                          debugPrefix, replicationFactor, startOffset));
  const unsigned incrementValue = direction == Direction::CLOCKWISE ? -1 : 1;
  // create program to change the slice index to it's next value.
  // called every iteration of the repeat
  popops::mapInPlace(graph,
                     Rem(Add(_1, Const(replicationFactor + incrementValue)),
                         Const(replicationFactor)),
                     {program.sliceFragments.getSliceIndex()},
                     program.incrementIndex);
  // create the cross replica copy the collective needs
  program.exchangeProg.setCopy(
      crossReplicaCopy(
          graph, srcBuffer, dstBuffer,
          [&](unsigned src) { return ring.getRank(src, direction, 1); }),
      direction);
  // create program that will do a dynamic slice with index bein the
  // slice index created ealier. The slice index is incremented by the
  // increment program on each iteration of the repeat
  replicatedRankSlice(program.sliceFragments, graph, fragments, srcBuffer, ring,
                      direction);
  // perform the reduction with the received data and the value sliced
  program.reduceProg =
      ReduceProg(srcBuffer, dstBuffer, op, debugPrefix + "/Reduce");
  program.undefTensor = concat({srcBuffer, dstBuffer});
  program.srcBuffer = std::move(srcBuffer);
  program.dstBuffer = std::move(dstBuffer);
  logging::info("Unidirectional ring reduce scatter end");
  return program;
}

static Tensor
bidirectionalRingPairReduceScatter(Graph &graph, const Tensor &toReduce,
                                   popops::Operation op, Sequence &prog,
                                   const std::string &debugPrefix) {
  // split to reduce in half and call the clockwsie and anticlockwise on
  // each. The bidirectionalSequence function will then interleave the
  // programs in the same repeat. Don't need to worry about ipu mapping when
  // splitting in half as this method won't be called unless one ipu per
  // replica
  logging::info("Bidirectional ring reduce scatter");
  if (graph.getReplicationFactor() == 1) {
    return toReduce;
  }

  auto fragments = replicatedSplitIntoFragments(
      toReduce, graph.getReplicationFactor(), graph);
  auto fragmentSize = fragments.dim(1);
  auto clockwiseFragments = fragments.slice(0, fragmentSize / 2, 1);
  auto anticlockwiseFragments =
      fragments.slice(fragmentSize / 2, fragmentSize, 1);
  auto clockwiseProg = unidirectionalRingReduceScatter(
      graph, clockwiseFragments.flatten(), op, Direction::CLOCKWISE,
      debugPrefix + "/clockwise", graph.getReplicationFactor());
  auto anticlockwiseProg = unidirectionalRingReduceScatter(
      graph, anticlockwiseFragments.flatten(), op, Direction::ANTICLOCKWISE,
      debugPrefix + "/anticlockwise", graph.getReplicationFactor());
  prog.add(bidirectionalSequence(clockwiseProg, anticlockwiseProg, graph));
  auto srcBuffer =
      concat(clockwiseProg.srcBuffer.get(), anticlockwiseProg.srcBuffer.get());
  return srcBuffer;
}

static Tensor ringMeetInMiddleReduceScatter(Graph &graph,
                                            const Tensor &toReduce,
                                            popops::Operation op,
                                            Sequence &prog,
                                            const std::string &debugPrefix) {
  logging::info("Meet in the middle reduce scatter");
  const auto replicationFactor = graph.getReplicationFactor();
  if (replicationFactor <= 2) {
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, CLOCKWISE, debugPrefix, replicationFactor);
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  RingTopology ring(replicationFactor);
  auto numFragments = replicationFactor;
  auto numSteps = 1 + numFragments / 2;
  const int clockwiseOffset = -1 * (numFragments - numSteps);
  const int anticlockwiseOffset = (numFragments - numSteps) + 1;
  auto fragments = replicatedSplitIntoFragments(toReduce, numFragments, graph);

  auto clockwiseProg = unidirectionalRingReduceScatter(
      graph, toReduce, op, Direction::CLOCKWISE, debugPrefix + "/clockwise",
      numSteps, clockwiseOffset);
  auto anticlockwiseProg = unidirectionalRingReduceScatter(
      graph, toReduce, op, Direction::ANTICLOCKWISE,
      debugPrefix + "/anticlockwise", numSteps - 1, anticlockwiseOffset);

  Sequence combineBuffers;
  opInPlace(graph, op, clockwiseProg.srcBuffer.get(),
            anticlockwiseProg.dstBuffer.get(), combineBuffers,
            debugPrefix + "/Reduce");
  prog.add(meetInMiddleReduceScatterSequence(clockwiseProg, anticlockwiseProg,
                                             graph, std::move(combineBuffers)));
  logging::info("Meet in the middle ring reduce scatter end");
  return clockwiseProg.srcBuffer.get();
}

static Tensor reduceScatter(Graph &graph, const Tensor &toReduce,
                            popops::Operation op, Sequence &prog,
                            const std::string &debugPrefix,
                            const CollectiveOptions &options) {
  CollectiveMethod method = options.method;
  if (method == CollectiveMethod::AUTO) {
    method = pickReduceScatterMethod(graph, toReduce, op);
  }
  switch (method) {
  default:
    assert(0 && "Unexpected reduce method");
  case CollectiveMethod::CLOCKWISE_RING: {
    logging::info("Reduce scatter collective method is clockwise ring");
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, CLOCKWISE, debugPrefix,
        graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  case CollectiveMethod::ANTICLOCKWISE_RING: {
    logging::info("reduce scatter collective method is anti-clockwise ring");
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, ANTICLOCKWISE, debugPrefix,
        graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR: {
    logging::info("Reduce scatter collective method is Bidirectional ring");
    return bidirectionalRingPairReduceScatter(graph, toReduce, op, prog,
                                              debugPrefix);
  }
  case CollectiveMethod::MEET_IN_MIDDLE_RING: {
    logging::info("Reduce scatter collective "
                  "method is Meet in the middle ring");
    return ringMeetInMiddleReduceScatter(graph, toReduce, op, prog,
                                         debugPrefix);
  }
  }
}

// Return the tile the last element of a tensor is mapped to.
static unsigned getTileOfLastElement(Graph &graph, const Tensor &t) {
  const auto numElements = t.numElements();
  assert(numElements > 0);
  auto last = t.flatten()[numElements - 1];
  auto tileMapping = graph.getTileMapping(last);
  for (unsigned tile = 0; tile != tileMapping.size(); ++tile) {
    if (!tileMapping[tile].empty())
      return tile;
  }
  POPLIB_UNREACHABLE();
}

// Add padding to the reference tensor so the number of elements on each
// IPU is equal to the number of elements of the fragment that is on that
// IPU times the number of fragments.
static Tensor padAllGatherResult(Graph &graph, const Tensor &fragment,
                                 unsigned numFragments, const Tensor &result) {
  auto fragmentElementsPerIpu = getNumElementsPerIpu(graph, fragment);
  auto referencePerIpu = getPerIpuTensors(result, graph);
  const auto numIpus = fragmentElementsPerIpu.size();
  assert(referencePerIpu.size() == numIpus);
  std::vector<Tensor> toConcat = {result};
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    const auto referenceElements = referencePerIpu[ipu].numElements();
    const auto fragmentElements = fragmentElementsPerIpu[ipu];
    assert(fragmentElements * numFragments >= referenceElements);
    unsigned paddingElements =
        fragmentElements * numFragments - referenceElements;
    if (paddingElements > 0) {
      auto padding = graph.addVariable(result.elementType(), {paddingElements},
                                       "AllGatherPadding");
      auto tile = getTileOfLastElement(graph, referencePerIpu[ipu]);
      graph.setTileMapping(padding, tile);
      toConcat.push_back(padding);
    }
  }
  return concat(toConcat);
}

static CollectivesProgram unidirectionalRingAllGather(
    Graph &graph, const Tensor &toGather, const Tensor &result,
    Direction direction, const std::string &debugPrefix,
    const unsigned numSteps, const int startOffset = 0) {
  logging::info("Unidirectional ring allGather");
  const auto replicationFactor = graph.getReplicationFactor();

  RingTopology ring(replicationFactor);
  auto numFragments = replicationFactor;
  auto srcBuffer = graph.clone(toGather, debugPrefix + "/GatherSrc");
  auto dstBuffer = graph.clone(toGather, debugPrefix + "/GatherDst");
  auto paddedResult = padAllGatherResult(graph, toGather, numFragments, result);
  auto fragments =
      replicatedSplitIntoFragments(paddedResult, numFragments, graph);
  assert(fragments.dim(1) == toGather.numElements());
  CollectivesProgram program = [&]() {
    if (graph.getTopLevelGraph().getReplicationFactor() == 1) {
      return CollectivesProgram(SliceCopy(DynamicSliceCopy()));
    } else {
      return CollectivesProgram(SliceCopy(SwitchSliceCopy(replicationFactor)));
    }
  }();

  program.repeatCounter = numSteps - 1;
  auto replicationIndex = graph.addReplicationIndexConstant();
  graph.setTileMapping(replicationIndex, 0);
  program.sliceFragments.setSliceIndex(
      initRingIndexTensor(graph, direction, replicationIndex, program.initIndex,
                          debugPrefix, replicationFactor, startOffset));
  program.firstGatherCopy.add(Copy(toGather, srcBuffer));
  const unsigned incrementValue = direction == Direction::CLOCKWISE ? -1 : 1;
  popops::mapInPlace(graph,
                     Rem(Add(_1, Const(replicationFactor + incrementValue)),
                         Const(replicationFactor)),
                     {program.sliceFragments.getSliceIndex()},
                     program.incrementIndex);
  program.exchangeProg.setCopy(
      crossReplicaCopy(
          graph, srcBuffer, dstBuffer,
          [&](unsigned src) { return ring.getRank(src, direction, 1); }),
      direction);
  program.allgatherCopy.add(Copy(dstBuffer, srcBuffer));
  replicatedRankUpdate(program.sliceFragments, graph, srcBuffer, fragments,
                       ring, direction);
  program.undefTensor = concat({paddedResult, srcBuffer, dstBuffer});
  return program;
}

static void bidirectionalRingPairAllGather(Graph &graph, const Tensor &toGather,
                                           const Tensor &result, Sequence &prog,
                                           const std::string &debugPrefix) {
  logging::info("Bidirectional ring allGather");
  const auto replicationFactor = graph.getReplicationFactor();

  auto numFragments = replicationFactor;
  auto fragmentSize = toGather.numElements();
  auto numSteps = replicationFactor;
  auto resultPadded = padAllGatherResult(graph, toGather, numFragments, result);
  auto fragments =
      replicatedSplitIntoFragments(resultPadded, numFragments, graph);
  auto clockwiseFragments = fragments.slice(0, fragmentSize / 2, 1);
  auto anticlockwiseFragments =
      fragments.slice(fragmentSize / 2, fragmentSize, 1);
  auto clockwiseProg = unidirectionalRingAllGather(
      graph, toGather.slice(0, fragmentSize / 2), clockwiseFragments.flatten(),
      Direction::CLOCKWISE, debugPrefix + "/clockwise", numSteps);
  auto anticlockwiseProg = unidirectionalRingAllGather(
      graph, toGather.slice(fragmentSize / 2, fragmentSize),
      anticlockwiseFragments.flatten(), Direction::ANTICLOCKWISE,
      debugPrefix + "/anticlockwise", numSteps);
  prog.add(bidirectionalSequence(clockwiseProg, anticlockwiseProg, graph));
}

static void ringMeetInMiddleAllGather(Graph &graph, const Tensor &toGather,
                                      const Tensor &result, Sequence &prog,
                                      const std::string &debugPrefix) {
  logging::info("Meet in the middle ring allGather");
  if (graph.getReplicationFactor() <= 2) {
    auto program = unidirectionalRingAllGather(
        graph, toGather, result, Direction::CLOCKWISE, debugPrefix,
        graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return;
  }
  auto numSteps = 1 + graph.getReplicationFactor() / 2;
  const int clockwiseOffset = 0;
  const int anticlockwiseOffset = 0;

  auto clockwiseProg = unidirectionalRingAllGather(
      graph, toGather, result, Direction::CLOCKWISE, debugPrefix + "/clockwise",
      numSteps, clockwiseOffset);
  auto anticlockwiseProg = unidirectionalRingAllGather(
      graph, toGather, result, Direction::ANTICLOCKWISE,
      debugPrefix + "/clockwise", numSteps - 1, anticlockwiseOffset);
  prog.add(
      meetInMiddleAllGatherSequence(clockwiseProg, anticlockwiseProg, graph));
}

// The IPU mapping of the result tensor determines how the gathered elements
// are interleaved. For each IPU the elements of the toGather tensor on that
// IPU are concatenated in order of their rank and written to the elements of
// the result tensor on that IPU. If the number of the gathered elements on an
// IPU is greater than the number of result elements on that IPU the excess
// gathered elements are ignored
static void allGather(Graph &graph, const Tensor &toGather,
                      const Tensor &result, Sequence &prog,
                      const std::string &debugPrefix,
                      const CollectiveOptions &options) {
  CollectiveMethod method = options.method;
  if (method == CollectiveMethod::AUTO) {
    method = pickAllGatherMethod(graph, toGather);
  }
  switch (method) {
  default:
    assert(0 && "Unexpected reduce method");
  case CollectiveMethod::CLOCKWISE_RING: {
    logging::info("All gather collective method is clockwise ring");
    auto program =
        unidirectionalRingAllGather(graph, toGather, result, CLOCKWISE,
                                    debugPrefix, graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return;
  }
  case CollectiveMethod::ANTICLOCKWISE_RING: {
    logging::info("All gather collective method is anti-clockwise ring");
    auto program =
        unidirectionalRingAllGather(graph, toGather, result, ANTICLOCKWISE,
                                    debugPrefix, graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return;
  }
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR: {
    logging::info("All gather collective method is Bidirectional ring");
    return bidirectionalRingPairAllGather(graph, toGather, result, prog,
                                          debugPrefix);
  }
  case CollectiveMethod::MEET_IN_MIDDLE_RING: {
    logging::info("All gather collective method is Meet in the middle ring");
    return ringMeetInMiddleAllGather(graph, toGather, result, prog,
                                     debugPrefix);
  }
  }
}

static void noCheckReplicatedAllReduce(Graph &graph, const poplar::Tensor &data,
                                       const poplar::Tensor &result,
                                       popops::Operation op,
                                       program::Sequence &prog,
                                       const std::string &debugPrefix,
                                       const poplar::OptionFlags &optionFlags) {
  auto topLevelGraph = graph.getTopLevelGraph();
  auto topLevelReplicationFactor = topLevelGraph.getReplicationFactor();
  CollectiveOptions options;
  options.useReplicatedImplementation = topLevelReplicationFactor > 1;
  parseCollectiveOptions(optionFlags, options);

  auto dataReordered = data.flatten();
  auto resultReordered = result.flatten();
  graph.reorderToSimplify(&dataReordered, {&resultReordered});
  if (options.useReplicatedImplementation) {
    logging::info("Using replicated version of allReduce");
    auto reduceScattered =
        reduceScatter(graph, dataReordered, op, prog, debugPrefix, options);
    allGather(graph, reduceScattered, resultReordered, prog, debugPrefix,
              options);
  } else {
    if (topLevelReplicationFactor > 1) {
      throw poputil::poplibs_error("Can't use non replicated collective "
                                   "implementation if the top level graph "
                                   "is replicated");
    }
    auto reduced = allReduce(
        topLevelGraph, topLevelGraph.getNonReplicatedTensor(dataReordered), op,
        prog, debugPrefix, optionFlags);
    prog.add(
        Copy(reduced, topLevelGraph.getNonReplicatedTensor(resultReordered)));
  }
}

void replicatedAllReduceWithOutput(Graph &graph, const poplar::Tensor &data,
                                   poplar::Tensor &result, popops::Operation op,
                                   program::Sequence &prog,
                                   const std::string &debugPrefix,
                                   const poplar::OptionFlags &optionFlags) {
  logging::info("Replicated all reduce begin ({}B)",
                data.numElements() *
                    graph.getTarget().getTypeSize(data.elementType()));
  if (data.shape() != result.shape()) {
    throw poputil::poplibs_error("Shape of input and output tensors "
                                 "are different");
  }
  if (data.elementType() != result.elementType()) {
    throw poputil::poplibs_error("result and input tensors must"
                                 " have same type");
  }
  const bool correctMapping =
      getIpuMapping(graph, data) == getIpuMapping(graph, result);
  if (!correctMapping) {
    logging::warn("Warning: the ipu mapping of result and input tensor "
                  "is different. This will introduce an extra copy");
  }
  const Tensor output = [&]() {
    if (correctMapping) {
      return result;
    } else {
      return graph.clone(data);
    }
  }();
  noCheckReplicatedAllReduce(graph, data, output, op, prog, debugPrefix,
                             optionFlags);
  if (!correctMapping) {
    prog.add(Copy(output, result));
  }
  logging::info("Replicated all reduce end");
}

Tensor replicatedAllReduce(Graph &graph, const poplar::Tensor &data,
                           popops::Operation op, program::Sequence &prog,
                           const std::string &debugPrefix,
                           const poplar::OptionFlags &optionFlags) {
  logging::info("Replicated all reduce begin ({}B)",
                data.numElements() *
                    graph.getTarget().getTypeSize(data.elementType()));
  auto result = graph.clone(data, debugPrefix + "/result");
  noCheckReplicatedAllReduce(graph, data, result, op, prog, debugPrefix,
                             optionFlags);
  logging::info("Replicated all reduce end");
  return result;
}

Tensor replicatedAllReduce(Graph &graph, Graph &parentGraph,
                           const poplar::Tensor &data, popops::Operation op,
                           program::Sequence &prog,
                           const std::string &debugPrefix,
                           const poplar::OptionFlags &optionFlags) {
  auto parentGraphReplicationFactor = parentGraph.getReplicationFactor();
  if (parentGraphReplicationFactor != 1) {
    throw poputil::poplibs_error("replicatedAllReduce() does not support "
                                 "replicated parent graphs");
  }
  return replicatedAllReduce(graph, data, op, prog, debugPrefix, optionFlags);
}

} // End namespace popops
