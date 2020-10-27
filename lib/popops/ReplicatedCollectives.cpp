// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/Collectives.hpp"

#include "CollectivesProgram.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/DynamicSlice.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Pad.hpp"
#include "popops/Reduce.hpp"
#include "poputil/OptionParsing.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/exceptions.hpp"
#include <boost/dll.hpp>
#include <boost/optional/optional.hpp>
#include <cassert>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;

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

enum class FragmentCopyMethod { SWITCH, DYNAMIC_SLICE };

} // End anonymous namespace.

namespace popops {

// Picks tile for mapping scalars based on an existing mapping
static unsigned getScalarTile(const Graph::TileToTensorMapping mapping) {
  auto it =
      std::find_if(mapping.begin(), mapping.end(),
                   [](const std::vector<Interval> &iv) { return !iv.empty(); });
  return it == mapping.end() ? 0 : std::distance(mapping.begin(), it);
}

static auto getNumILDs(Graph &graph) {
  const auto &target = graph.getTarget();
  const auto repNumIPUs = target.getNumIPUs();
  const auto totNumIPUs = repNumIPUs * graph.getReplicationFactor();
  return ceildiv(totNumIPUs, target.getIpuLinkDomainSize());
}

static auto replicasPerILD(Graph &graph) {
  return graph.getReplicationFactor() / getNumILDs(graph);
}

static std::vector<unsigned>
invertPermutation(const std::vector<unsigned> &permutation) {
  std::vector<unsigned> inverse(permutation.size());
  for (unsigned i = 0; i != permutation.size(); ++i) {
    inverse[permutation[i]] = i;
  }
  return inverse;
}

// Example patterns for 8 ipus
// Interleaved: 0, 7, 1, 6, 2, 5, 3, 4
// Flat: 0, 1, 2, 3, 4, 5, 6, 7
enum class RingPattern { Interleaved, Flat };

// Return the IPUs in clockwise direction around the ring starting at IPU 0.
static std::vector<unsigned> createRing(const unsigned n, RingPattern pattern) {
  std::vector<unsigned> ring(n);
  unsigned i = 0;
  switch (pattern) {
  case RingPattern::Interleaved:
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
    break;
  case RingPattern::Flat:
    std::generate(ring.begin(), ring.end(), [&] { return i++ % n; });
    break;
  default:
    POPLIB_UNREACHABLE();
    break;
  }
  return ring;
}

namespace {

class RingTopology {
  // IPUs in clockwise direction around the ring starting at IPU 0.
  std::vector<unsigned> ringIndexToRank;
  std::vector<unsigned> rankToRingIndex;
  RingPattern pattern;

public:
  RingTopology(unsigned n, unsigned ipusPerReplica,
               poplar::IpuLinkTopology topology) {
    if (topology == poplar::IpuLinkTopology::Mesh ||
        (topology == poplar::IpuLinkTopology::Torus && ipusPerReplica == 1)) {
      pattern = RingPattern::Interleaved;
    } else {
      assert(topology == poplar::IpuLinkTopology::Torus);
      pattern = RingPattern::Flat;
    }
    ringIndexToRank = createRing(n, pattern);
    rankToRingIndex = invertPermutation(ringIndexToRank);
  }

  /// Return the number of IPU that is the specified number of steps in the
  /// specified direction around the ring, starting at the specified base
  /// IPU.
  unsigned getRank(unsigned base, Direction direction, unsigned steps) const {
    auto numRanks = ringIndexToRank.size();
    auto ring = base / numRanks;
    auto index = rankToRingIndex[base % numRanks];
    switch (direction) {
    case CLOCKWISE:
      index = (index + steps) % numRanks;
      break;
    case ANTICLOCKWISE:
      steps = steps % numRanks;
      index = (index + numRanks - steps) % numRanks;
      break;
    }
    return ringIndexToRank[index] + ring * numRanks;
  }
  unsigned indexToRank(const unsigned index) const {
    return ringIndexToRank[index];
  }
  unsigned rankToIndex(const unsigned rank) const {
    return rankToRingIndex[rank];
  }

  // using the replication index tensor create a new tensor with
  // value = the position in a clockwise ring of this replica is
  Tensor initRingIndexTensor(Graph &graph, const Tensor &repIndex,
                             Sequence &prog, const std::string &debugPrefix,
                             const unsigned replicasPerRing,
                             const int startOffset) const {
    // determine the replica index within the local ring
    auto localRingRepIndex =
        popops::map(graph, popops::expr::_1 % replicasPerRing, {repIndex}, prog,
                    debugPrefix);

    // start offset allows to initialise this at different positions
    // in the ring for clockwise and anticlockwise. Used by the meet
    // in the middle method. Will often be zero.
    //
    // this expression initialises replica id to clockwise ring index
    const auto id = [&]() {
      const auto replica = popops::expr::_1;
      if (pattern == RingPattern::Interleaved) {
        const auto replicaMod2 = replica % 2;
        return ((replicasPerRing - 1) * replicaMod2) +
               ((replicaMod2 * -2 + 1) * (replica / 2));
      } else if (pattern == RingPattern::Flat) {
        return replica + 0;
      }
      POPLIB_UNREACHABLE();
    }();

    return popops::map(
        graph,
        (id + ((replicasPerRing + startOffset) % replicasPerRing)) %
            replicasPerRing,
        {localRingRepIndex}, prog, debugPrefix);
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
  toConcat.reserve(intervals.size());
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

static unsigned getIpusPerReplica(Graph &graph) {
  auto topLevelGraph = graph.getTopLevelGraph();
  unsigned numIpus = topLevelGraph.getTarget().getNumIPUs() *
                     topLevelGraph.getReplicationFactor();
  unsigned numReplicas = graph.getReplicationFactor();
  return numIpus / numReplicas;
}

static CollectiveMethod pickAllGatherMethod(Graph &graph,
                                            std::size_t numBytes) {
  const auto ipusPerRank = getIpusPerReplica(graph);
  const auto numRanks = replicasPerILD(graph);
  const auto &target = graph.getTarget();
  if (target.getIpuLinkTopology() == IpuLinkTopology::Torus &&
      ipusPerRank > 1) { // Note we don't utilize the loopback cable for 1 ipu
                         // per replica (T25224)
    // TODO: T26094 Investigate when to use BIDIRECTIONAL_RING_PAIR instead
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  if (ipusPerRank > 1 || numRanks <= 2)
    return CollectiveMethod::CLOCKWISE_RING;
  const auto bytesPerIpu = numBytes / ipusPerRank;
  // Thresholds where the BIDIRECTIONAL_RING_PAIR method starts to beat the
  // MEET_IN_MIDDLE_RING method determined experimentally.
  // TODO: T12970 Lots has changed since these thresholds were set - check if
  // they are still appropriate.
  if (bytesPerIpu < 622592 || (numRanks > 4 && bytesPerIpu < 2490368) ||
      (numRanks > 8 && bytesPerIpu < 19922944) || numRanks > 16) {
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  return CollectiveMethod::BIDIRECTIONAL_RING_PAIR;
}

static CollectiveMethod pickAllGatherMethod(Graph &graph,
                                            const Tensor &toGather) {
  const auto &target = graph.getTarget();
  const auto numBytes =
      toGather.numElements() * target.getTypeSize(toGather.elementType());
  return pickAllGatherMethod(graph, numBytes);
}

static CollectiveMethod pickReduceScatterMethod(Graph &graph,
                                                std::size_t numBytes) {
  const auto ipusPerRank = getIpusPerReplica(graph);
  const auto numRanks = replicasPerILD(graph);
  const auto &target = graph.getTarget();
  if (target.getIpuLinkTopology() == IpuLinkTopology::Torus &&
      ipusPerRank > 1) {
    // Note we don't utilize the loopback cable for 1 ipu per replica (T25224)
    // TODO: T26094 Investigate when to use BIDIRECTIONAL_RING_PAIR instead
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  if (ipusPerRank > 1 || numRanks <= 2)
    return CollectiveMethod::CLOCKWISE_RING;
  unsigned bytesPerIpu = numBytes / ipusPerRank;
  // Thresholds where the BIDIRECTIONAL_RING_PAIR method starts to beat the
  // MEET_IN_MIDDLE_RING method determined experimentally.
  // TODO: T12970 Lots has changed since these thresholds were set - check if
  // they are still appropriate.
  if (bytesPerIpu < 1245184 || (numRanks > 4 && bytesPerIpu < 4980736) ||
      (numRanks > 8 && bytesPerIpu < 39845888) || numRanks > 16) {
    return CollectiveMethod::MEET_IN_MIDDLE_RING;
  }
  return CollectiveMethod::BIDIRECTIONAL_RING_PAIR;
}

static CollectiveMethod pickReduceScatterMethod(Graph &graph, const Tensor &t) {
  const auto &target = graph.getTarget();
  const auto numBytes = t.numElements() * target.getTypeSize(t.elementType());
  return pickReduceScatterMethod(graph, numBytes);
}

// Split a tensor into the specified number of fragments such that the
// number of elements and the IPU mapping of each fragment is identical,
// adding padding if necessary to achieve this.
static Tensor replicatedSplitIntoFragments(const Tensor &t,
                                           unsigned numFragments,
                                           Graph &graph) {
  logging::popops::debug("Split into fragments");
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

static Tensor giveFragmentsRankOrder(const Tensor &input,
                                     const RingTopology &ring) {
  std::vector<Tensor> result;
  result.reserve(input.dim(0));
  for (unsigned replica = 0; replica < input.dim(0); ++replica) {
    const unsigned ringIndex = ring.rankToIndex(replica);
    result.push_back(input[ringIndex].expand({0}));
  }
  return concat(result, 0);
}

static Tensor giveFragmentsRingOrder(const Tensor &input,
                                     const RingTopology &ring) {
  std::vector<Tensor> result;
  result.reserve(input.dim(0));
  for (unsigned ringIndex = 0; ringIndex < input.dim(0); ++ringIndex) {
    const unsigned replica = ring.indexToRank(ringIndex);
    result.push_back(input[replica].expand({0}));
  }
  return concat(result, 0);
}

static void internalReplicatedSlice(Graph &graph, const Tensor &fragmentsByRing,
                                    const Tensor &sliceIndex, const Tensor &dst,
                                    Sequence &prog) {
  auto dslice = dynamicSlice(graph, fragmentsByRing, sliceIndex.expand({0}),
                             {0}, {1}, prog);
  // this copy is probably avoidable
  prog.add(Copy(dslice, dst));
}

static void replicatedRankSlice(Graph &graph, const Tensor &fragmentsByRing,
                                const Tensor &sliceIndex, const Tensor &dst,
                                Sequence &prog,
                                FragmentCopyMethod fragmentCopyMethod) {
  logging::popops::debug("Replicated rank slice");
  assert(fragmentsByRing.rank() == dst.rank() + 1);
  assert(fragmentsByRing[0].shape() == dst.shape());
  if (fragmentCopyMethod == FragmentCopyMethod::DYNAMIC_SLICE) {
    return internalReplicatedSlice(graph, fragmentsByRing, sliceIndex, dst,
                                   prog);
  }
  assert(fragmentCopyMethod == FragmentCopyMethod::SWITCH);
  unsigned n = fragmentsByRing.dim(0);
  auto swtch = Switch::switchWithUnreachableDefault(sliceIndex);
  for (unsigned i = 0; i < n; ++i) {
    swtch.add(i, Copy(fragmentsByRing[i], dst));
  }
  prog.add(swtch);
}

static void internalReplicatedUpdate(Graph &graph, const Tensor &fragments,
                                     const Tensor &sliceIndex,
                                     const Tensor &src, Sequence &prog) {
  assert(src.rank() == 1);
  dynamicUpdate(graph, fragments, src.expand({0}), sliceIndex.expand({0}), {0},
                {1}, prog);
}

static void replicatedRankUpdate(Graph &graph, const Tensor &src,
                                 const Tensor &fragments,
                                 const Tensor &sliceIndex, Sequence &prog,
                                 FragmentCopyMethod fragmentCopyMethod) {
  logging::popops::debug("replicatedRankUpdate begin");
  assert(src.rank() + 1 == fragments.rank());
  assert(src.shape() == fragments[0].shape());
  if (fragmentCopyMethod == FragmentCopyMethod::DYNAMIC_SLICE) {
    return internalReplicatedUpdate(graph, fragments, sliceIndex, src, prog);
  }
  assert(fragmentCopyMethod == FragmentCopyMethod::SWITCH);
  assert(graph.getReplicationFactor() ==
         graph.getTopLevelGraph().getReplicationFactor());
  unsigned n = fragments.dim(0);
  auto swtch = Switch::switchWithUnreachableDefault(sliceIndex);
  for (unsigned i = 0; i < n; ++i) {
    swtch.add(i, Copy(src, fragments[i]));
  }
  prog.add(swtch);
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
  auto mapping = graph.getTileMapping(fragments);
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    auto virtualGraph =
        graph.createVirtualGraph(ipu * tilesPerIpu, (ipu + 1) * tilesPerIpu);
    // Spread the buffer across the tiles the fragments are on.
    std::vector<unsigned> usedTiles;
    for (unsigned tileInIpu = 0; tileInIpu != tilesPerIpu; ++tileInIpu) {
      if (mapping[ipu * tilesPerIpu + tileInIpu].empty())
        continue;
      usedTiles.push_back(tileInIpu);
    }
    virtualGraph = virtualGraph.createVirtualGraph(usedTiles);
    poputil::mapTensorLinearly(
        virtualGraph, concatSlices(buffer, virtualGraph, ipuMapping[ipu]));
  }
}

std::vector<size_t> orderIpusBySize(const std::vector<Tensor> &tensorPerIpu) {
  std::vector<size_t> ipuIndexOrderedBySize(tensorPerIpu.size());
  std::iota(ipuIndexOrderedBySize.begin(), ipuIndexOrderedBySize.end(), 0);
  std::stable_sort(ipuIndexOrderedBySize.begin(), ipuIndexOrderedBySize.end(),
                   [&](size_t lhs, size_t rhs) {
                     return tensorPerIpu[lhs].numElements() <
                            tensorPerIpu[rhs].numElements();
                   });
  return ipuIndexOrderedBySize;
}

Graph getReplicatedGraphWithAllIpus(Graph &graph) {
  auto topLevelGraph = graph.getTopLevelGraph();
  if (topLevelGraph.getReplicationFactor() == graph.getReplicationFactor())
    return topLevelGraph;
  return topLevelGraph.createReplicatedGraph(
      graph.getReplicationFactor() / topLevelGraph.getReplicationFactor());
}

/*
 For CrossReplicaCopies which would involve through routing on every IPU on a
 given vertical (e.g. ipu 0, 2, 4, 6 ..) with a loopback cable, there is the
 potential for deadlock. This function splits these CrossReplicaCopies into
 multiple CrossReplicaCopies which don't involve through routing. For example,
 replication factor = 2, ipus per replica = 4; each copy is routed through
 another ipu (ipu 0 (replica 0) -> ipu 0 (replica 1) is via ipu 2).

 e.g. the left hand side rail of the ladder would be broken up into two
 CrossReplicaCopies:
    CrossReplicaCopy 0: ipu 0 (replica 0) -> ipu 0 (replica 1)
    CrossReplicaCopy 1: ipu 2 (replica 0) -> ipu 2 (replica 1) ...
 Because the ring configuration of replica size > 2 is two rings, for each even
 and odd side of the ladder, we combine a even with an odd CrossReplicaCopy to
 happen in parallel as there should be no deadlock introduced.

 Each entry is a separate exchange; src/dst Tensor pair
*/
std::vector<std::pair<Tensor, Tensor>>
splitExchangesToAvoidDeadlock(Graph &graph, const Tensor &src,
                              const Tensor &dst) {
  if (graph.getTarget().getIpuLinkTopology() != IpuLinkTopology::Torus ||
      getIpusPerReplica(graph) <= 2) {
    // We only have problems with torus configuration and replica sizes which
    // introduce through routing to the neighbouring replica
    return {std::make_pair(src, dst)};
  }
  logging::popops::debug("splitExchangesToAvoidDeadlock");

  auto replicatedGraphWithAllIpus = getReplicatedGraphWithAllIpus(graph);
  if (replicatedGraphWithAllIpus.getTarget().getNumIPUs() % 2 != 0) {
    throw poputil::poplibs_error("Odd sized replicas are unsupported");
  }
  const auto srcTensorPerIpu =
      getPerIpuTensors(src, replicatedGraphWithAllIpus);
  const auto dstTensorPerIpu =
      getPerIpuTensors(dst, replicatedGraphWithAllIpus);

  std::vector<Tensor> evenRailSrcTensors;
  std::vector<Tensor> oddRailSrcTensors;
  std::vector<Tensor> evenRailDstTensors;
  std::vector<Tensor> oddRailDstTensors;

  for (unsigned i = 0; i < srcTensorPerIpu.size(); i++) {
    if (i % 2 == 0) {
      evenRailSrcTensors.push_back(srcTensorPerIpu[i]);
      evenRailDstTensors.push_back(dstTensorPerIpu[i]);
    } else {
      oddRailSrcTensors.push_back(srcTensorPerIpu[i]);
      oddRailDstTensors.push_back(dstTensorPerIpu[i]);
    }
  }

  // We order each ipu on each rail by tensor sizes, so that when we combine the
  // cross replica copies from each rail, we are pairing them so they are more
  // balanced
  // e.g. Given the following cross replica copies:
  // - ipu 0 (replica 0) -> ipu 0 (replica 1) : LARGE
  // - ipu 2 (replica 0) -> ipu 2 (replica 1) : SMALL
  // - ipu 1 (replica 0) -> ipu 1 (replica 1) : SMALL
  // - ipu 3 (replica 0) -> ipu 3 (replica 1) : LARGE
  // Then, we want to pair the large ipu 0 & ipu 3 cross replica copies, and
  // then ipu 2 & ipu 1 cross replica copies to give more balanced exchanges.
  const auto evenRailIdxOrdered = orderIpusBySize(evenRailSrcTensors);
  const auto oddRailIdxOrdered = orderIpusBySize(oddRailSrcTensors);

  // The max number of exchanges depends on whichever rail has the most cross
  // replica copies as each will need to be separated into a separate exchange
  const auto numExchangesRequired =
      std::max(evenRailSrcTensors.size(), oddRailSrcTensors.size());
  std::vector<std::pair<Tensor, Tensor>> exchanges;
  for (unsigned i = 0; i < numExchangesRequired; i++) {
    if (i < evenRailSrcTensors.size() && i < oddRailSrcTensors.size()) {
      // We can combine even and odd exchanges at the same time as they are not
      // overlapping, we only have issues with deadlocks when it creates a loop
      // vertically on the ladder
      exchanges.push_back(
          std::make_pair(concat(evenRailSrcTensors[evenRailIdxOrdered[i]],
                                oddRailSrcTensors[oddRailIdxOrdered[i]]),
                         concat(evenRailDstTensors[evenRailIdxOrdered[i]],
                                oddRailDstTensors[oddRailIdxOrdered[i]])));
    } else if (i < evenRailSrcTensors.size()) {
      exchanges.push_back(
          std::make_pair(evenRailSrcTensors[evenRailIdxOrdered[i]],
                         evenRailDstTensors[evenRailIdxOrdered[i]]));
    } else if (i < oddRailSrcTensors.size()) {
      exchanges.push_back(
          std::make_pair(oddRailSrcTensors[oddRailIdxOrdered[i]],
                         oddRailDstTensors[oddRailIdxOrdered[i]]));
    } else {
      POPLIB_UNREACHABLE();
    }
  }
  if (exchanges.size() != 1) {
    logging::popops::debug(
        "split into {} exchanges to avoid a potential deadlock",
        exchanges.size());
  }
  assert(exchanges.size() && "Exchanges cannot be empty");
  return exchanges;
}

static std::size_t
getMaxPerTileElements(Graph::TileToTensorMapping::const_iterator begin,
                      Graph::TileToTensorMapping::const_iterator end) {
  std::size_t maxElements = 0;
  for (auto it = begin; it != end; ++it) {
    auto elements =
        std::accumulate(it->begin(), it->end(), 0UL,
                        [](std::size_t count, const Interval &interval) {
                          return count + interval.size();
                        });
    if (elements > maxElements) {
      maxElements = elements;
    }
  }
  return maxElements;
}

static std::size_t getMaxPerTileElements(Graph &graph, const Tensor &t) {
  auto mapping = graph.getTileMapping(t);
  return getMaxPerTileElements(mapping.begin(), mapping.end());
}

static std::optional<Tensor>
createSliceableTensorIfBenefical(Graph &graph, const Tensor &slice,
                                 unsigned numSlices, const std::string &name) {
  // Copying the input to a sliceable tensor avoids a switch which reduces the
  // amount of control code required. The disadvantage is that it increases
  // the amount of live tensor data by introducing another copy of the input.
  // Try to estimate which approach takes the least memory.
  const auto sliceableTensorMaxBytesPerTile =
      getMaxPerTileElements(graph, slice) * numSlices;
  // Currently each case requires 3 instructions - the jump table entry,
  // a call to the exchange code and a jump to the end.
  const auto bytesPerCase = 3 * 4;
  // Switch code is always live (and so must be summed across all collective
  // calls) but the second copy of the input is only live for a short time. To
  // compare the two we assume the collectives are called this many times in the
  // program.
  const auto expectedNumCollectiveCalls = 8;
  const auto switchCodePerTile =
      numSlices * bytesPerCase * expectedNumCollectiveCalls;
  if (switchCodePerTile < sliceableTensorMaxBytesPerTile)
    return {};
  return createSliceableTensorFromSlice(graph, slice.expand({0}), {0},
                                        {numSlices}, name);
}

static Tensor
copyToSliceableTensorIfBeneficial(Graph &graph, const Tensor &fragments,
                                  const Tensor &slice, const std::string &name,
                                  Sequence &prog,
                                  FragmentCopyMethod &fragmentCopyMethod) {
  auto rearranged =
      createSliceableTensorIfBenefical(graph, slice, fragments.dim(0), name);
  if (!rearranged)
    return fragments;
  logging::popops::debug("Copy input to sliceable tensor");
  prog.add(Copy(fragments, *rearranged));
  fragmentCopyMethod = FragmentCopyMethod::DYNAMIC_SLICE;
  return *rearranged;
}

static Tensor
copyFromSliceableTensorIfBeneficial(Graph &graph, const Tensor &fragments,
                                    const Tensor &slice,
                                    const std::string &name, Sequence &prog,
                                    FragmentCopyMethod &fragmentCopyMethod) {
  auto rearranged =
      createSliceableTensorIfBenefical(graph, slice, fragments.dim(0), name);
  if (!rearranged)
    return fragments;
  logging::popops::debug("Copy output from sliceable tensor");
  prog.add(Copy(*rearranged, fragments));
  fragmentCopyMethod = FragmentCopyMethod::DYNAMIC_SLICE;
  return *rearranged;
}

// the offset is so that the meet in the middle method can start at part
// way through the iterations. can be positive or negative so that the same
// number can be used to initialise the clockwise and anticlockwise ring
static CollectivesProgram unidirectionalRingReduceScatter(
    Graph &graph, const Tensor &toReduce, popops::Operation op,
    Direction direction, const std::string &debugPrefix,
    const unsigned numSteps, const int startOffset = 0) {
  logging::popops::debug("Unidirectional ring reduce scatter");
  CollectivesProgram program;
  auto fragmentCopyMethod = graph.getTopLevelGraph().getReplicationFactor() > 1
                                ? FragmentCopyMethod::SWITCH
                                : FragmentCopyMethod::DYNAMIC_SLICE;

  const auto replicasPerRing = replicasPerILD(graph);
  const unsigned ipusPerReplica = getIpusPerReplica(graph);
  const RingTopology ring(replicasPerRing, ipusPerReplica,
                          graph.getTarget().getIpuLinkTopology());
  auto numFragments = replicasPerRing;

  auto fragmentsByReplica =
      replicatedSplitIntoFragments(toReduce, numFragments, graph);
  auto fragmentsByRing = giveFragmentsRingOrder(fragmentsByReplica, ring);
  auto fragmentSize = fragmentsByRing.dim(1);
  auto srcBuffer = graph.addVariable(toReduce.elementType(), {fragmentSize},
                                     debugPrefix + "/ScatterSrc");
  mapBuffer(graph, srcBuffer, fragmentsByRing);
  fragmentsByRing = copyToSliceableTensorIfBeneficial(
      graph, fragmentsByRing, srcBuffer, debugPrefix + "/InputRearranged",
      program.rearrangePre, fragmentCopyMethod);
  auto dstBuffer = graph.clone(srcBuffer, debugPrefix + "/ScatterDst");
  auto repFactorTensor = graph.addReplicationIndexConstant();

  // Map index tensor to IPU involved in this collective program
  graph.setTileMapping(repFactorTensor,
                       getScalarTile(graph.getTileMapping(toReduce)));

  program.repeatCounter = numSteps - 1;
  const unsigned incrementValue = direction == Direction::CLOCKWISE ? -1 : 1;
  auto sliceIndex = ring.initRingIndexTensor(
      graph, repFactorTensor, program.initIndex, debugPrefix, replicasPerRing,
      startOffset + incrementValue);
  // create program to change the slice index to it's next value.
  // called every iteration of the repeat
  popops::mapInPlace(graph,
                     (popops::expr::_1 + (replicasPerRing + incrementValue)) %
                         replicasPerRing,
                     {sliceIndex}, program.incrementIndex);
  // create the cross replica copy the collective needs
  const auto copies =
      splitExchangesToAvoidDeadlock(graph, srcBuffer, dstBuffer);
  program.exchangeProg.resize(copies.size());
  for (unsigned i = 0; i < copies.size(); i++) {
    program.exchangeProg[i].setCopy(
        crossReplicaCopy(
            graph, copies[i].first, copies[i].second,
            [&](unsigned src) { return ring.getRank(src, direction, 1); }),
        direction);
  }
  // Create program that will do a dynamic slice with index being the
  // slice index created earlier. The slice index is incremented by the
  // increment program on each iteration of the repeat.
  replicatedRankSlice(graph, fragmentsByRing, sliceIndex, srcBuffer,
                      program.sliceFragments, fragmentCopyMethod);
  // perform the reduction with the received data and the value sliced
  program.reduceProg =
      ReduceProg(srcBuffer, dstBuffer, op, debugPrefix + "/Reduce");
  program.undefTensor = concat({srcBuffer, dstBuffer});
  program.srcBuffer = std::move(srcBuffer);
  program.dstBuffer = std::move(dstBuffer);
  logging::popops::debug("Unidirectional ring reduce scatter end");
  return program;
}

static Tensor
bidirectionalRingPairReduceScatter(Graph &graph, const Tensor &toReduce,
                                   popops::Operation op, Sequence &prog,
                                   const std::string &debugPrefix) {
  const auto replicasPerRing = replicasPerILD(graph);

  // split to reduce in half and call the clockwise and anticlockwise on
  // each. The bidirectionalSequence function will then interleave the
  // programs in the same repeat. Don't need to worry about ipu mapping when
  // splitting in half as this method won't be called unless one ipu per
  // replica
  logging::popops::debug("Bidirectional ring reduce scatter");
  if (replicasPerRing == 1) {
    return toReduce;
  }

  auto fragments =
      replicatedSplitIntoFragments(toReduce, replicasPerRing, graph);
  auto fragmentSize = fragments.dim(1);
  auto clockwiseFragments = fragments.slice(0, fragmentSize / 2, 1);
  auto anticlockwiseFragments =
      fragments.slice(fragmentSize / 2, fragmentSize, 1);
  auto clockwiseProg = unidirectionalRingReduceScatter(
      graph, clockwiseFragments.flatten(), op, Direction::CLOCKWISE,
      debugPrefix + "/clockwise", replicasPerRing);
  auto anticlockwiseProg = unidirectionalRingReduceScatter(
      graph, anticlockwiseFragments.flatten(), op, Direction::ANTICLOCKWISE,
      debugPrefix + "/anticlockwise", replicasPerRing);
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
  logging::popops::debug("Meet in the middle reduce scatter");
  const auto replicasPerRing = replicasPerILD(graph);
  if (replicasPerRing <= 2) {
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, CLOCKWISE, debugPrefix, replicasPerRing);
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  auto numFragments = replicasPerRing;
  auto numSteps = 1 + numFragments / 2;
  const int clockwiseOffset = -1 * (numFragments - numSteps);
  const int anticlockwiseOffset = (numFragments - numSteps) + 1;

  auto clockwiseProg = unidirectionalRingReduceScatter(
      graph, toReduce, op, Direction::CLOCKWISE, debugPrefix + "/clockwise",
      numSteps, clockwiseOffset);
  auto anticlockwiseProg = unidirectionalRingReduceScatter(
      graph, toReduce, op, Direction::ANTICLOCKWISE,
      debugPrefix + "/anticlockwise", numSteps - 1, anticlockwiseOffset);

  unsigned topLevelControlTile =
      replicasPerRing == graph.getTopLevelGraph().getReplicationFactor()
          ? getScalarTile(graph.getTopLevelGraph().getTileMapping(toReduce))
          : 0;

  Sequence combineBuffers;
  opInPlace(graph, op, clockwiseProg.srcBuffer.get(),
            anticlockwiseProg.dstBuffer.get(), combineBuffers,
            debugPrefix + "/Reduce");
  prog.add(meetInMiddleReduceScatterSequence(clockwiseProg, anticlockwiseProg,
                                             graph, std::move(combineBuffers),
                                             topLevelControlTile));
  logging::popops::debug("Meet in the middle ring reduce scatter end");
  return clockwiseProg.srcBuffer.get();
}

static Tensor internalReduceScatter(Graph &graph, const Tensor &toReduce,
                                    popops::Operation op, Sequence &prog,
                                    const std::string &debugPrefix,
                                    const CollectiveOptions &options) {
  CollectiveMethod method = options.method;
  if (method == CollectiveMethod::AUTO) {
    method = pickReduceScatterMethod(graph, toReduce);
  }
  switch (method) {
  default:
    assert(0 && "Unexpected reduce method");
  case CollectiveMethod::CLOCKWISE_RING: {
    logging::popops::debug(
        "Reduce scatter collective method is clockwise ring");
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, CLOCKWISE, debugPrefix, replicasPerILD(graph));
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  case CollectiveMethod::ANTICLOCKWISE_RING: {
    logging::popops::debug(
        "reduce scatter collective method is anti-clockwise ring");
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, ANTICLOCKWISE, debugPrefix, replicasPerILD(graph));
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR: {
    logging::popops::debug(
        "Reduce scatter collective method is Bidirectional ring");
    return bidirectionalRingPairReduceScatter(graph, toReduce, op, prog,
                                              debugPrefix);
  }
  case CollectiveMethod::MEET_IN_MIDDLE_RING: {
    logging::popops::debug("Reduce scatter collective "
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
  std::vector<Tensor> toConcat = {result.flatten()};
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

static CollectivesProgram unidirectionalRingAllGatherImpl(
    Graph &graph, const Tensor &toGather, const Tensor &fragmentsByRing,
    const RingTopology &ring, Direction direction,
    const std::string &debugPrefix, const unsigned numSteps,
    const int startOffset, FragmentCopyMethod fragmentCopyMethod) {
  CollectivesProgram program;

  const auto replicasPerRing = replicasPerILD(graph);
  auto srcBuffer = graph.clone(toGather, debugPrefix + "/GatherSrc");
  auto dstBuffer = graph.clone(toGather, debugPrefix + "/GatherDst");
  assert(fragmentsByRing.dim(1) == toGather.numElements());

  program.repeatCounter = numSteps - 1;
  auto replicationIndex = graph.addReplicationIndexConstant();
  graph.setTileMapping(replicationIndex,
                       getScalarTile(graph.getTileMapping(toGather)));
  auto sliceIndex =
      ring.initRingIndexTensor(graph, replicationIndex, program.initIndex,
                               debugPrefix, replicasPerRing, startOffset);
  program.firstGatherCopy.add(Copy(toGather, srcBuffer));
  const unsigned incrementValue = direction == Direction::CLOCKWISE ? -1 : 1;
  popops::mapInPlace(graph,
                     (popops::expr::_1 + (replicasPerRing + incrementValue)) %
                         replicasPerRing,
                     {sliceIndex}, program.incrementIndex);
  const auto copies =
      splitExchangesToAvoidDeadlock(graph, srcBuffer, dstBuffer);
  program.exchangeProg.resize(copies.size());
  for (unsigned i = 0; i < copies.size(); i++) {
    program.exchangeProg[i].setCopy(
        crossReplicaCopy(
            graph, copies[i].first, copies[i].second,
            [&](unsigned src) { return ring.getRank(src, direction, 1); }),
        direction);
  }
  program.allgatherCopy.add(Copy(dstBuffer, srcBuffer));
  replicatedRankUpdate(graph, srcBuffer, fragmentsByRing, sliceIndex,
                       program.sliceFragments, fragmentCopyMethod);
  program.undefTensor = concat(
      {fragmentsByRing.flatten(), srcBuffer.flatten(), dstBuffer.flatten()});
  return program;
}

static std::pair<CollectivesProgram, Tensor> unidirectionalRingAllGather(
    Graph &graph, const Tensor &toGather, const std::optional<Tensor> &dst,
    Direction direction, const std::string &debugPrefix) {
  logging::popops::debug("Unidirectional ring allGather");
  const auto replicasPerRing = replicasPerILD(graph);
  const unsigned ipusPerReplica = getIpusPerReplica(graph);
  RingTopology ring(replicasPerRing, ipusPerReplica,
                    graph.getTarget().getIpuLinkTopology());
  auto numFragments = replicasPerRing;
  auto fragmentCopyMethod = graph.getTopLevelGraph().getReplicationFactor() > 1
                                ? FragmentCopyMethod::SWITCH
                                : FragmentCopyMethod::DYNAMIC_SLICE;
  Tensor result, fragmentsByRing;
  Sequence resultRearrange;
  if (dst) {
    result = *dst;
    auto paddedResult =
        padAllGatherResult(graph, toGather, numFragments, result);
    auto fragmentsByReplica =
        replicatedSplitIntoFragments(paddedResult, numFragments, graph);
    fragmentsByRing = giveFragmentsRingOrder(fragmentsByReplica, ring);
    fragmentsByRing = copyFromSliceableTensorIfBeneficial(
        graph, fragmentsByRing, toGather, debugPrefix + "/SliceableOutput",
        resultRearrange, fragmentCopyMethod);
  } else {
    // Since we are free to choose the layout we can use dynamic slice without
    // the penalty of a rearranging copy.
    fragmentsByRing =
        createSliceableTensorFromSlice(graph, toGather.expand({0}), {0},
                                       {numFragments}, debugPrefix + "/Output");
    result = giveFragmentsRankOrder(fragmentsByRing, ring);
    fragmentCopyMethod = FragmentCopyMethod::DYNAMIC_SLICE;
  }
  auto prog = unidirectionalRingAllGatherImpl(
      graph, toGather, fragmentsByRing, ring, direction, debugPrefix,
      numFragments, 0, fragmentCopyMethod);
  prog.rearrangePost.add(resultRearrange);
  return {prog, result};
}

static Tensor bidirectionalRingPairAllGather(Graph &graph,
                                             const Tensor &toGather,
                                             const std::optional<Tensor> &dst,
                                             Sequence &prog,
                                             const std::string &debugPrefix) {
  logging::popops::debug("Bidirectional ring allGather");
  const auto replicasPerRing = replicasPerILD(graph);

  auto numFragments = replicasPerRing;
  auto fragmentSize = toGather.numElements();
  CollectivesProgram clockwiseProg, anticlockwiseProg;
  std::optional<Tensor> clockwiseDst, anticlockwiseDst;
  if (dst) {
    auto resultPadded = padAllGatherResult(graph, toGather, numFragments, *dst);
    auto fragments =
        replicatedSplitIntoFragments(resultPadded, numFragments, graph);
    clockwiseDst = fragments.slice(0, fragmentSize / 2, 1).flatten();
    anticlockwiseDst =
        fragments.slice(fragmentSize / 2, fragmentSize, 1).flatten();
  }
  Tensor clockwiseResult, anticlockwiseResult;
  std::tie(clockwiseProg, clockwiseResult) = unidirectionalRingAllGather(
      graph, toGather.flatten().slice(0, fragmentSize / 2), clockwiseDst,
      Direction::CLOCKWISE, debugPrefix + "/clockwise");
  std::tie(anticlockwiseProg, anticlockwiseResult) =
      unidirectionalRingAllGather(
          graph, toGather.flatten().slice(fragmentSize / 2, fragmentSize),
          anticlockwiseDst, Direction::ANTICLOCKWISE,
          debugPrefix + "/anticlockwise");
  prog.add(bidirectionalSequence(clockwiseProg, anticlockwiseProg, graph));
  if (dst)
    return *dst;
  auto resultShape = toGather.expand({0}).broadcast(numFragments, 0).shape();
  return concat(clockwiseResult, anticlockwiseResult, 1).reshape(resultShape);
}

static Tensor ringMeetInMiddleAllGather(Graph &graph, const Tensor &toGather,
                                        const std::optional<Tensor> &dst,
                                        Sequence &prog,
                                        const std::string &debugPrefix) {
  const auto replicasPerRing = replicasPerILD(graph);
  logging::popops::debug("Meet in the middle ring allGather");
  if (replicasPerRing <= 2) {
    CollectivesProgram program;
    Tensor result;
    std::tie(program, result) = unidirectionalRingAllGather(
        graph, toGather, dst, Direction::CLOCKWISE, debugPrefix);
    prog.add(unidirectionalSequence(program, graph));
    return result;
  }
  const unsigned ipusPerReplica = getIpusPerReplica(graph);
  RingTopology ring(replicasPerRing, ipusPerReplica,
                    graph.getTarget().getIpuLinkTopology());
  const auto numFragments = replicasPerRing;
  auto numSteps = 1 + replicasPerRing / 2;
  const int clockwiseOffset = 0;
  const int anticlockwiseOffset = 0;

  unsigned topLevelControlTile =
      replicasPerRing == graph.getTopLevelGraph().getReplicationFactor()
          ? getScalarTile(graph.getTopLevelGraph().getTileMapping(toGather))
          : 0;
  Tensor result;
  Sequence resultRearrange;
  auto fragmentCopyMethod = graph.getTopLevelGraph().getReplicationFactor() > 1
                                ? FragmentCopyMethod::SWITCH
                                : FragmentCopyMethod::DYNAMIC_SLICE;

  if (dst) {
    result = *dst;
    result = padAllGatherResult(graph, toGather, numFragments, result);
    result = replicatedSplitIntoFragments(result, numFragments, graph);
    result = giveFragmentsRingOrder(result, ring);
    result = copyFromSliceableTensorIfBeneficial(
        graph, result, toGather, debugPrefix + "/SliceableOutput",
        resultRearrange, fragmentCopyMethod);
  } else {
    // Since we are free to choose the layout we can use dynamic slice without
    // the penalty of a rearranging copy.
    result =
        createSliceableTensorFromSlice(graph, toGather.expand({0}), {0},
                                       {numFragments}, debugPrefix + "/Output");
    fragmentCopyMethod = FragmentCopyMethod::DYNAMIC_SLICE;
  }

  auto clockwiseProg = unidirectionalRingAllGatherImpl(
      graph, toGather, result, ring, Direction::CLOCKWISE,
      debugPrefix + "/clockwise", numSteps, clockwiseOffset,
      fragmentCopyMethod);
  auto anticlockwiseProg = unidirectionalRingAllGatherImpl(
      graph, toGather, result, ring, Direction::ANTICLOCKWISE,
      debugPrefix + "/anticlockwise", numSteps - 1, anticlockwiseOffset,
      fragmentCopyMethod);
  prog.add(meetInMiddleAllGatherSequence(clockwiseProg, anticlockwiseProg,
                                         graph, topLevelControlTile));
  prog.add(resultRearrange);
  if (dst)
    return *dst;
  return giveFragmentsRankOrder(result, ring);
}

// The IPU mapping of the result tensor determines how the gathered elements
// are interleaved. For each IPU the elements of the toGather tensor on that
// IPU are concatenated in order of their rank and written to the elements of
// the result tensor on that IPU. If the number of the gathered elements on an
// IPU is greater than the number of result elements on that IPU the excess
// gathered elements are ignored
static Tensor allGather(Graph &graph, const Tensor &toGather,
                        const std::optional<Tensor> &dst, Sequence &prog,
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
    logging::popops::debug("All gather collective method is clockwise ring");
    Tensor result;
    CollectivesProgram program;
    std::tie(program, result) = unidirectionalRingAllGather(
        graph, toGather, dst, CLOCKWISE, debugPrefix);
    prog.add(unidirectionalSequence(program, graph));
    return result;
  }
  case CollectiveMethod::ANTICLOCKWISE_RING: {
    logging::popops::debug(
        "All gather collective method is anti-clockwise ring");
    Tensor result;
    CollectivesProgram program;
    std::tie(program, result) = unidirectionalRingAllGather(
        graph, toGather, dst, ANTICLOCKWISE, debugPrefix);
    prog.add(unidirectionalSequence(program, graph));
    return result;
  }
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR: {
    logging::popops::debug(
        "All gather collective method is Bidirectional ring");
    return bidirectionalRingPairAllGather(graph, toGather, dst, prog,
                                          debugPrefix);
  }
  case CollectiveMethod::MEET_IN_MIDDLE_RING: {
    logging::popops::debug(
        "All gather collective method is Meet in the middle ring");
    return ringMeetInMiddleAllGather(graph, toGather, dst, prog, debugPrefix);
  }
  }
}

poplar::Tensor replicatedReduceScatter(Graph &graph, const Tensor &toReduce,
                                       popops::Operation op, Sequence &prog,
                                       const poplar::DebugContext &debugContext,
                                       const OptionFlags &optionFlags) {
  const auto debugPrefix = debugContext.getPathName();
  logging::popops::info("replicatedReduceScatter data={}, op={}, name={}",
                        toReduce.shape(), op, debugPrefix);
  logging::popops::debug("Replicated reduce scatter begin ({}B)",
                         toReduce.numElements() * graph.getTarget().getTypeSize(
                                                      toReduce.elementType()));
  if (toReduce.rank() != 1) {
    throw poputil::poplibs_error("Input tensor to replicatedReduceScatter "
                                 "must have rank 1, but had rank " +
                                 std::to_string(toReduce.rank()));
  }

  CollectiveOptions options;
  parseCollectiveOptions(optionFlags, options);

  auto output =
      internalReduceScatter(graph, toReduce, op, prog, debugPrefix, options);
  logging::popops::debug("Replicated reduce scatter end");
  return output;
}

static Tensor
noCheckReplicatedAllGather(Graph &graph, const Tensor &toGather,
                           const std::optional<Tensor> &dst, Sequence &prog,
                           const std::string &debugPrefix,
                           const poplar::OptionFlags &optionFlags) {
  CollectiveOptions options;
  parseCollectiveOptions(optionFlags, options);

  return allGather(graph, toGather, dst, prog, debugPrefix, options);
}

poplar::Tensor replicatedAllGather(Graph &graph, const Tensor &toGather,
                                   Sequence &prog,
                                   const poplar::DebugContext &debugContext,
                                   const poplar::OptionFlags &optionFlags) {
  const auto debugPrefix = debugContext.getPathName();
  logging::popops::info("replicatedAllGather data={}, name={}",
                        toGather.shape(), debugPrefix);
  logging::popops::debug("Replicated all gather begin ({}B)",
                         toGather.numElements() * graph.getTarget().getTypeSize(
                                                      toGather.elementType()));

  if (graph.getTopLevelGraph().getReplicationFactor() > 1 &&
      graph.getReplicationFactor() !=
          graph.getTopLevelGraph().getReplicationFactor()) {
    throw poputil::poplibs_error(
        "replicatedAllGather doesn't support a mix of single image and "
        "non-single image replication within the same graph.");
  }

  auto output = noCheckReplicatedAllGather(graph, toGather, std::nullopt, prog,
                                           debugPrefix, optionFlags);

  logging::popops::debug("Replicated all gather end");
  return output;
}

static std::size_t
getNumElements(Graph::TileToTensorMapping::const_iterator begin,
               Graph::TileToTensorMapping::const_iterator end) {
  return std::accumulate(
      begin, end, 0UL,
      [](std::size_t count, const std::vector<Interval> &intervals) {
        return std::accumulate(intervals.begin(), intervals.end(), count,
                               [](std::size_t count, const Interval &interval) {
                                 return count + interval.size();
                               });
      });
}

// Permute the data tensor to improve the efficiency of an all-reduce using
// the specified number of fragments and apply the same permutation to the
// result tensor.
static void allReduceReorder(Graph &graph, poplar::Tensor *data,
                             poplar::Tensor *result, unsigned numFragments) {
  logging::popops::debug(
      "Reorder data tensor for an all-reduce with {} fragments", numFragments);
  auto dataReordered = data->flatten();
  auto resultReordered = result->flatten();
  graph.reorderToSimplify(&dataReordered, {&resultReordered});
  auto tileMapping = graph.getTileMapping(dataReordered);
  auto numIpus = graph.getTarget().getNumIPUs();
  auto tilesPerIpu = graph.getTarget().getTilesPerIPU();
  auto typeSize = graph.getTarget().getTypeSize(data->elementType());
  auto exchangeBytesPerCycle = graph.getTarget().getExchangeBytesPerCycle();
  std::vector<Interval> reorderedIntervals;
  for (unsigned ipu = 0; ipu != numIpus; ++ipu) {
    auto tileBegin = ipu * tilesPerIpu;
    auto tileEnd = (ipu + 1) * tilesPerIpu;

    // Permute data so when we later split it into fragments each fragment is
    // well balanced across tiles. We do this by concatenating elements from
    // each tile in round robin fashion. Instead of concatenating indivdual
    // elements (which would increase code size due to copies of many small
    // regions) we concatenate slices. The largest slice size we can use without
    // increasing the maximum number of elements a single tile contributes to a
    // fragment is given by the maximum number of elements on any tile divided
    // by the number of fragments.
    auto maxElements = getMaxPerTileElements(tileMapping.begin() + tileBegin,
                                             tileMapping.begin() + tileEnd);
    auto tileBytesPerFragment = ceildiv(maxElements * typeSize, numFragments);
    // Round the number of elements up and enforce a minimum size to avoid
    // fragments being spread too thinly (which would increases memory a lot
    // for a small reduction in cycles).
    // This number was chosen arbitrarily so may not be optimial
    const auto minTileBytesPerFragment = 128UL;
    tileBytesPerFragment =
        std::max(tileBytesPerFragment, minTileBytesPerFragment);
    auto grainSize = lcm<unsigned>(exchangeBytesPerCycle, typeSize);
    auto tileElementsPerFragment =
        roundUp(tileBytesPerFragment, grainSize) / typeSize;
    struct MappingOffset {
      std::size_t interval = 0;
      std::size_t offsetInInterval = 0;
    };
    auto numElements = getNumElements(tileMapping.begin() + tileBegin,
                                      tileMapping.begin() + tileEnd);
    auto elementsPerFragment = ceildiv(numElements, numFragments);
    std::vector<MappingOffset> offsets(tilesPerIpu);
    unsigned tile = tileBegin;
    auto ipuElementsRemaining = numElements;
    // Construct intervals a fragment at a time by cycling through the tiles in
    // a round robin fashion.
    for (unsigned fragment = 0; fragment != numFragments; ++fragment) {
      auto fragmentElementsRemaining =
          std::min(elementsPerFragment, ipuElementsRemaining);
      while (fragmentElementsRemaining > 0) {
        auto tileElementsRemaining =
            std::min(tileElementsPerFragment, fragmentElementsRemaining);
        auto &offset = offsets[tile - tileBegin];
        while (tileElementsRemaining > 0) {
          if (offset.interval == tileMapping[tile].size())
            break;
          const auto &interval = tileMapping[tile][offset.interval];
          auto sliceBegin = interval.begin() + offset.offsetInInterval;
          auto sliceEnd =
              std::min(sliceBegin + tileElementsRemaining, interval.end());
          auto sliceSize = sliceEnd - sliceBegin;
          logging::popops::trace(
              "Add interval [{},{}) on tile {} to fragment {}", sliceBegin,
              sliceEnd, tile, fragment);
          reorderedIntervals.emplace_back(sliceBegin, sliceEnd);
          if (sliceEnd == interval.end()) {
            ++offset.interval;
            offset.offsetInInterval = 0;
          } else {
            offset.offsetInInterval += sliceSize;
          }
          tileElementsRemaining -= sliceSize;
          fragmentElementsRemaining -= sliceSize;
          ipuElementsRemaining -= sliceSize;
        }
        ++tile;
        if (tile == tileEnd)
          tile = tileBegin;
      }
    }
  }
  *data = concatSlices(dataReordered, graph, reorderedIntervals);
  *result = concatSlices(resultReordered, graph, reorderedIntervals);
}

static unsigned getReduceScatterNumFragments(Graph &graph,
                                             const CollectiveOptions &options,
                                             const poplar::Tensor &data) {
  CollectiveMethod method = options.method;
  if (method == CollectiveMethod::AUTO) {
    method = pickReduceScatterMethod(graph, data);
  }
  switch (method) {
  default:
    POPLIB_UNREACHABLE();
  case CollectiveMethod::CLOCKWISE_RING:
  case CollectiveMethod::ANTICLOCKWISE_RING:
  case CollectiveMethod::MEET_IN_MIDDLE_RING:
    return replicasPerILD(graph);
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR:
    return replicasPerILD(graph) * 2;
  }
}

static unsigned getAllGatherNumFragments(Graph &graph,
                                         const CollectiveOptions &options,
                                         unsigned numBytes) {
  CollectiveMethod method = options.method;
  if (method == CollectiveMethod::AUTO) {
    method = pickAllGatherMethod(graph, numBytes);
  }
  switch (method) {
  default:
    POPLIB_UNREACHABLE();
  case CollectiveMethod::CLOCKWISE_RING:
  case CollectiveMethod::ANTICLOCKWISE_RING:
  case CollectiveMethod::MEET_IN_MIDDLE_RING:
    return replicasPerILD(graph);
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR:
    return replicasPerILD(graph) * 2;
  }
}

static unsigned getAllReduceNumFragments(Graph &graph,
                                         const CollectiveOptions &options,
                                         const poplar::Tensor &data) {
  auto typeSize = graph.getTarget().getTypeSize(data.elementType());
  auto numElements = data.numElements();
  const auto numRanks = replicasPerILD(graph);
  auto numElementsAfterReduceScatter = ceildiv(numElements, numRanks);
  return std::max(
      getReduceScatterNumFragments(graph, options, data),
      getAllGatherNumFragments(graph, options,
                               numElementsAfterReduceScatter * typeSize));
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
  allReduceReorder(graph, &dataReordered, &resultReordered,
                   getAllReduceNumFragments(graph, options, data));
  if (options.useReplicatedImplementation) {
    logging::popops::debug("Using replicated version of allReduce");
    auto reduceScattered = internalReduceScatter(graph, dataReordered, op, prog,
                                                 debugPrefix, options);
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
                                   const poplar::DebugContext &debugContext,
                                   const poplar::OptionFlags &optionFlags) {
  const auto debugPrefix = debugContext.getPathName();
  logging::popops::info(
      "replicatedAllReduceWithOutput data={}, result={}, op={}, name={}",
      data.shape(), result.shape(), op, debugPrefix);

  logging::popops::debug("Replicated all reduce begin ({}B)",
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
    logging::popops::warn("Warning: the ipu mapping of result and input tensor "
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
  logging::popops::debug("Replicated all reduce end");
}

void replicatedAllReduceInPlace(poplar::Graph &graph, poplar::Tensor &data,
                                popops::Operation op,
                                poplar::program::Sequence &prog,
                                const poplar::DebugContext &debugContext,
                                const poplar::OptionFlags &options) {
  const auto debugPrefix = debugContext.getPathName();
  return replicatedAllReduceWithOutput(graph, data, data, op, prog, debugPrefix,
                                       options);
}

Tensor replicatedAllReduce(Graph &graph, const poplar::Tensor &data,
                           popops::Operation op, program::Sequence &prog,
                           const poplar::DebugContext &debugContext,
                           const poplar::OptionFlags &optionFlags) {
  const auto debugPrefix = debugContext.getPathName();

  logging::popops::info("replicatedAllReduce data={}, op={}, name={}",
                        data.shape(), op, debugPrefix);

  logging::popops::debug("Replicated all reduce begin ({}B)",
                         data.numElements() *
                             graph.getTarget().getTypeSize(data.elementType()));
  auto result = graph.clone(data, debugPrefix + "/result");
  noCheckReplicatedAllReduce(graph, data, result, op, prog, debugPrefix,
                             optionFlags);
  logging::popops::debug("Replicated all reduce end");
  return result;
}

Tensor replicatedAllReduce(Graph &graph, Graph &parentGraph,
                           const poplar::Tensor &data, popops::Operation op,
                           program::Sequence &prog,
                           const poplar::DebugContext &debugContext,
                           const poplar::OptionFlags &optionFlags) {
  const auto debugPrefix = debugContext.getPathName();
  auto parentGraphReplicationFactor = parentGraph.getReplicationFactor();
  if (parentGraphReplicationFactor != 1) {
    throw poputil::poplibs_error("replicatedAllReduce() does not support "
                                 "replicated parent graphs");
  }
  return replicatedAllReduce(graph, data, op, prog, debugPrefix, optionFlags);
}

static std::vector<std::map<unsigned, unsigned>>
createCommunicationMap(unsigned replicationFactor) {
  std::vector<std::map<unsigned, unsigned>> communicationMap;

  // We only have replicationFactor-1 communication steps.
  for (unsigned step = 0; step < replicationFactor - 1; ++step) {
    // Add the map for this step of the iteration.
    communicationMap.push_back({});
    std::map<unsigned, unsigned> &theMap = communicationMap.back();

    for (unsigned replica = 0; replica < replicationFactor; ++replica) {

      // The replica we are sending data to.
      unsigned destReplica = replica + step + 1;

      // Wrap around.
      if (destReplica >= replicationFactor) {
        destReplica -= replicationFactor;
      }

      // Mapped as dest:source
      theMap.insert({replica, destReplica});
    }
  }

  return communicationMap;
}

Tensor allToAllPersonalizedExchange(Graph &graph, const poplar::Tensor &input,
                                    program::Sequence &sequence,
                                    const poplar::DebugContext &debugContext,
                                    const poplar::OptionFlags &options) {
  const auto debugPrefix = debugContext.getPathName();
  // Options are currently not supported.
  (void)options;
  using namespace popops::expr;
  if (graph.getTopLevelGraph().getReplicationFactor() !=
      graph.getReplicationFactor()) {
    throw poputil::poplibs_error(
        "allToAllPersonalizedExchange only supports single image replication");
  }

  if (input.shape()[0] != graph.getReplicationFactor()) {
    throw poputil::poplibs_error(
        "allToAllPersonalizedExchange expects the size of the first dimension"
        "to be of replicationFactor size");
  }

  // Get the replication factor from the graph.
  unsigned replicationFactor = graph.getReplicationFactor();

  // Clone the output and source target.
  Tensor output = poputil::duplicate(graph, input, sequence);

  // Slice up the input and output tensor into replica number of slices.
  std::vector<Interval> sliceIntervals;
  sliceIntervals.reserve(replicationFactor);
  for (unsigned replica = 0; replica < replicationFactor; ++replica) {
    sliceIntervals.push_back({replica, replica + 1});
  }

  // We need to have a consistent communication pattern between the IPUs so each
  // one can know (or work out) which IPU it has just received data from and so
  // can know where that should go. We do this in a clockwise fasion moving the
  // destination IPU each iteration but keeping the source the same. Take:
  // [IPU0] [IPU1]
  // [IPU2] [IPU3]
  // In this case over three iterations we communicate like so:
  // Iteration 1: IPU0->IPU1, IPU1->IPU2, IPU2->IPU3, IPU3->IPU0
  // Iteration 2: IPU0->IPU2, IPU1->IPU3, IPU2->IPU0, IPU3->IPU1
  // Iteration 3: IPU0->IPU3, IPU1->IPU0, IPU2->IPU1, IPU3->IPU2
  const std::vector<std::map<unsigned, unsigned>> communicationMap =
      createCommunicationMap(replicationFactor);

  // Slice the input.
  std::vector<Tensor> slicedInput = input.slices(sliceIntervals, 0);

  // Slice the output.
  std::vector<Tensor> slicedOutput = output.slices(sliceIntervals, 0);

  // Add the replication constant to the graph.
  Tensor replicationFactorTensor = graph.addReplicationIndexConstant();
  graph.setTileMapping(replicationFactorTensor,
                       getScalarTile(graph.getTileMapping(input)));

  // The index into the tensor we are sending this iteration.
  Tensor sendIndex =
      poputil::duplicate(graph, replicationFactorTensor, sequence);

  // The index into the tensor we are recieving this iteration.
  Tensor recvIndex =
      poputil::duplicate(graph, replicationFactorTensor, sequence);

  // The index into the tensor we are sending this iteration.
  Tensor zeroConstant =
      graph.addConstant(UNSIGNED_INT, {}, 0, debugPrefix + "/ConstantZero");
  graph.setTileMapping(zeroConstant,
                       getScalarTile(graph.getTileMapping(input)));

  Tensor stepIndex =
      graph.addVariable(UNSIGNED_INT, {}, VariableMappingMethod::LINEAR,
                        debugPrefix + "/StepCount");
  sequence.add(Copy(zeroConstant, stepIndex));

  // The temporary memory buffer used on each replica to store the incoming
  // value before moving it to the correct location.
  Tensor tempSendBuffer = graph.clone(slicedInput[0]);
  Tensor tempReceiveBuffer = graph.clone(slicedOutput[0]);

  // Perform the actual exchange.
  // 1. Use a switch statement to extract from the input the slice we want to
  // send this iteration. (see communicationMap comment)
  // 2. CrossReplicaCopy the input to a temporary target buffer.
  // 3. Use a switch statement to copy that to the correct location in the
  // output.
  // 4. Repeat 1-4 for numReplicas - 1.

  Sequence loop_body;
  // Increment the send index, and clamp to range 0 to
  // replicationFactor-1.
  popops::mapInPlace(graph, Rem(Add(_1, Const(1u)), Const(replicationFactor)),
                     {sendIndex}, loop_body, debugPrefix);

  // Before sending, extract the element to be sent by copying to tempBuffer
  // in a switch.
  Switch inputExtractionSwitch(sendIndex);
  for (unsigned i = 0; i < replicationFactor; ++i) {
    inputExtractionSwitch.add(i, Copy(slicedInput[i], tempSendBuffer));
  }

  // After recieving, copy from the tempBuffer into the correct location using
  // the switch.
  Switch outputExtractionSwitch(recvIndex);
  for (unsigned i = 0; i < replicationFactor; ++i) {
    outputExtractionSwitch.add(i, Copy(tempReceiveBuffer, slicedOutput[i]));
  }

  // We calculate the IPU we are recieving from by decrementing the index
  // starting from
  popops::mapInPlace(
      graph,
      Rem(Add(_1, Const(replicationFactor - 1u)), Const(replicationFactor)),
      {recvIndex}, loop_body, debugPrefix);

  // Cross replica switch.
  Switch crossReplicaSwitch(stepIndex);
  for (unsigned step = 0; step < replicationFactor - 1; ++step) {
    crossReplicaSwitch.add(step,
                           CrossReplicaCopy(tempSendBuffer, tempReceiveBuffer,
                                            communicationMap[step]));
  }

  loop_body.add(inputExtractionSwitch);
  loop_body.add(crossReplicaSwitch);
  loop_body.add(outputExtractionSwitch);

  popops::addInPlace(graph, stepIndex, 1u, loop_body, debugPrefix);

  sequence.add(Repeat(replicationFactor - 1, loop_body));

  return output;
}

} // End namespace popops
