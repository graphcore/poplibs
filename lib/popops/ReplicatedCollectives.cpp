// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popops/Collectives.hpp"
#include "popops/CollectivesInterface.hpp"

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

} // End anonymous namespace.

namespace popops {

class ReplicatedCollectives : public ReplicatedCollectivesInterface {

public:
  virtual ~ReplicatedCollectives() {}

  std::string version() override { return "default"; }

  poplar::Tensor
  replicatedAllReduce(poplar::Graph &graph, const poplar::Tensor &data,
                      popops::Operation op, poplar::program::Sequence &prog,
                      const std::string &debugPrefix,
                      const poplar::OptionFlags &options) override;

  void replicatedAllReduceWithOutput(
      poplar::Graph &graph, const poplar::Tensor &data, poplar::Tensor &output,
      popops::Operation op, poplar::program::Sequence &prog,
      const std::string &debugPrefix,
      const poplar::OptionFlags &options) override;

  poplar::Tensor
  replicatedAllReduce(poplar::Graph &graph, poplar::Graph &parentGraph,
                      const poplar::Tensor &data, popops::Operation op,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix,
                      const poplar::OptionFlags &options) override;

  poplar::Tensor
  replicatedReduceScatter(poplar::Graph &graph, const poplar::Tensor &data,
                          popops::Operation op, poplar::program::Sequence &prog,
                          const std::string &debugPrefix,
                          const poplar::OptionFlags &optionFlags) override;

  poplar::Tensor
  replicatedAllGather(poplar::Graph &graph, const poplar::Tensor &data,
                      poplar::program::Sequence &prog,
                      const std::string &debugPrefix,
                      const poplar::OptionFlags &optionFlags) override;
};

// boost::dll::import returns boost::shared_ptr but since we cannot expose
// third party libraries in public headers we convert and expose a
// std::shared_ptr wrapping the boost::shared_ptr.
template <class T> std::shared_ptr<T> toStd(boost::shared_ptr<T> &p) {
  return std::shared_ptr<T>(p.get(), [p](...) mutable { p.reset(); });
}

// Picks tile for mapping scalars based on an existing mapping
static unsigned getScalarTile(const Graph::TileToTensorMapping mapping) {
  auto it =
      std::find_if(mapping.begin(), mapping.end(),
                   [](const std::vector<Interval> &iv) { return !iv.empty(); });
  return it == mapping.end() ? 0 : std::distance(mapping.begin(), it);
}

std::shared_ptr<ReplicatedCollectivesInterface>
    ReplicatedCollectivesInterface::defaultImpl{new ReplicatedCollectives()};

static const auto impl = []() {
  using namespace std::string_literals;
  if (std::getenv("GCL_NUM_IO_TILES")) {
    try {
      const char *ct_path = std::getenv("GCL_LIBRARY_PATH");
      const std::string libpath =
          (!ct_path || ct_path[0] == '\0')
              ? std::string("libgcl_ct.so")
              : std::string(ct_path) + std::string("/libgcl_ct.so");
      auto lib = boost::dll::import<ReplicatedCollectivesInterface>(
          libpath, "ReplicatedCollectives",
          boost::dll::load_mode::append_decorations);
      const auto version = lib ? lib->version() : "null";
      if (version != "replicatedGcl") {
        throw poputil::poplibs_error("Invalid gcl version: '" + version + "'");
      }
      return toStd(lib);
    } catch (const boost::system::system_error &err) {
      throw poputil::poplibs_error("Could not load/use gcl: "s + err.what());
    }
  }
  return ReplicatedCollectivesInterface::defaultImpl;
}();

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

  // using the replication index tensor create a new tensor with
  // value = the position in a clockwise ring of this replica is
  Tensor initRingIndexTensor(Graph &graph, const Direction direction,
                             const Tensor &repIndex, Sequence &prog,
                             const std::string &debugPrefix,
                             const unsigned repFactor,
                             const int startOffset) const {
    // start offset allows to initialise this at different positions
    // in the ring for clockwise and anticlockwise. Used by the meet
    // in the middle method. Will often be zero.
    //
    // this expression initialises replica id to clockwise ring index
    const auto id = [&]() {
      const auto replica = popops::expr::_1;
      if (pattern == RingPattern::Interleaved) {
        const auto replicaMod2 = replica % 2;
        return ((repFactor - 1) * replicaMod2) +
               ((replicaMod2 * -2 + 1) * (replica / 2));
      } else if (pattern == RingPattern::Flat) {
        return replica + 0;
      }
      POPLIB_UNREACHABLE();
    }();

    return popops::map(
        graph, (id + ((repFactor + startOffset) % repFactor)) % repFactor,
        {repIndex}, prog, debugPrefix);
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

static CollectiveMethod pickAllGatherMethod(const Graph &graph,
                                            std::size_t numBytes) {
  const auto ipusPerRank = graph.getTarget().getNumIPUs();
  const auto numRanks = graph.getReplicationFactor();
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

static CollectiveMethod pickAllGatherMethod(const Graph &graph,
                                            const Tensor &toGather) {
  const auto &target = graph.getTarget();
  const auto numBytes =
      toGather.numElements() * target.getTypeSize(toGather.elementType());
  return pickAllGatherMethod(graph, numBytes);
}

static CollectiveMethod pickReduceScatterMethod(const Graph &graph,
                                                std::size_t numBytes) {
  const auto ipusPerRank = graph.getTarget().getNumIPUs();
  const auto numRanks = graph.getReplicationFactor();
  const auto &target = graph.getTarget();
  if (target.getIpuLinkTopology() == IpuLinkTopology::Torus &&
      ipusPerRank > 1) { // Note we don't utilize the loopback cable for 1 ipu
                         // per replica (T25224)
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

static CollectiveMethod pickReduceScatterMethod(const Graph &graph,
                                                const Tensor &t) {
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

static void internalReplicatedSlice(Graph &graph, const Tensor &fragments,
                                    const Tensor &sliceIndex, const Tensor &dst,
                                    const Direction direction, Sequence &prog) {
  std::vector<Tensor> rearrange(fragments.dim(0));
  const unsigned n = rearrange.size();
  for (unsigned i = 0; i < n; ++i) {
    int shift = direction == Direction::CLOCKWISE ? 1 : -1;
    unsigned switchIndex = (i + n + shift) % n;
    rearrange[switchIndex] = fragments[i].expand({0});
  }
  auto dslice = dynamicSlice(graph, concat(rearrange), sliceIndex.expand({0}),
                             {0}, {1}, prog);
  // this copy is probably avoidable
  prog.add(Copy(dslice, dst));
}

static void replicatedRankSlice(Graph &graph, const Tensor &fragments_,
                                const Tensor &sliceIndex, const Tensor &dst,
                                const RingTopology &ring,
                                const Direction direction, Sequence &prog) {
  logging::debug("Replicated rank slice");
  assert(fragments_.rank() == dst.rank() + 1);
  assert(fragments_[0].shape() == dst.shape());
  auto fragments = giveFragmentsIpuOrder(fragments_, ring);
  const auto topGraph = graph.getTopLevelGraph();
  if (topGraph.getReplicationFactor() == 1) {
    return internalReplicatedSlice(graph, fragments, sliceIndex, dst, direction,
                                   prog);
  }
  assert(graph.getReplicationFactor() == topGraph.getReplicationFactor());
  unsigned n = fragments_.dim(0);
  auto swtch = Switch::switchWithUnreachableDefault(sliceIndex);
  for (unsigned i = 0; i < n; ++i) {
    int shift = direction == Direction::CLOCKWISE ? 1 : -1;
    unsigned switchIndex = (i + n + shift) % n;
    swtch.add(switchIndex, Copy(fragments[i], dst));
  }
  prog.add(swtch);
}

static void internalReplicatedUpdate(Graph &graph, const Tensor &fragments,
                                     const Tensor &sliceIndex,
                                     const Tensor &src, Direction,
                                     Sequence &prog) {
  assert(src.rank() == 1);
  dynamicUpdate(graph, fragments, src.expand({0}), sliceIndex.expand({0}), {0},
                {1}, prog);
}

static void replicatedRankUpdate(Graph &graph, const Tensor &src,
                                 const Tensor &fragments_,
                                 const Tensor &sliceIndex,
                                 const RingTopology &ring,
                                 const Direction direction, Sequence &prog) {
  logging::debug("replicatedRankUpdate begin");
  assert(src.rank() + 1 == fragments_.rank());
  assert(src.shape() == fragments_[0].shape());
  auto fragments = giveFragmentsIpuOrder(fragments_, ring);
  if (graph.getTopLevelGraph().getReplicationFactor() == 1) {
    return internalReplicatedUpdate(graph, fragments, sliceIndex, src,
                                    direction, prog);
  }
  assert(graph.getReplicationFactor() ==
         graph.getTopLevelGraph().getReplicationFactor());
  unsigned n = fragments_.dim(0);
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
  if (graph.getTarget().getIpuLinkTopology() != IpuLinkTopology::Torus) {
    // We only have problems with torus configuration
    return {std::make_pair(src, dst)};
  }
  const auto numIpusPerReplica = graph.getTarget().getNumIPUs();
  const auto numExchangesRequired = numIpusPerReplica >> 1;
  const auto srcTensorPerIpu = getPerIpuTensors(src, graph);
  const auto dstTensorPerIpu = getPerIpuTensors(dst, graph);

  std::vector<Tensor> evenRailSrcTensors;
  std::vector<Tensor> oddRailSrcTensors;
  for (unsigned i = 0; i < srcTensorPerIpu.size(); i++) {
    if (i % 2 == 0) {
      evenRailSrcTensors.push_back(srcTensorPerIpu[i]);
    } else {
      oddRailSrcTensors.push_back(srcTensorPerIpu[i]);
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
  const auto evenRailSrcIdxOrdered = orderIpusBySize(evenRailSrcTensors);
  const auto oddRailSrcIdxOrdered = orderIpusBySize(oddRailSrcTensors);

  std::vector<std::pair<Tensor, Tensor>> exchanges;
  for (unsigned i = 0; i < numExchangesRequired; i++) {
    // We can combine even and odd exchanges at the same time as they are not
    // overlapping, we only have issues with deadlocks when it creates a loop
    // vertically on the ladder
    exchanges.push_back(std::make_pair(
        concat(srcTensorPerIpu[2 * evenRailSrcIdxOrdered[i]],
               srcTensorPerIpu[2 * oddRailSrcIdxOrdered[i] + 1]),
        concat(dstTensorPerIpu[2 * evenRailSrcIdxOrdered[i]],
               dstTensorPerIpu[2 * oddRailSrcIdxOrdered[i] + 1])));
  }
  return exchanges;
}

// the offset is so that the meet in the middle method can start at part
// way through the iterations. can be positive or negative so that the same
// number can be used to initialise the clockwise and anticlockwise ring
static CollectivesProgram unidirectionalRingReduceScatter(
    Graph &graph, const Tensor &toReduce, popops::Operation op,
    Direction direction, const std::string &debugPrefix,
    const unsigned numSteps, const int startOffset = 0) {
  logging::debug("Unidirectional ring reduce scatter");

  const auto replicationFactor = graph.getReplicationFactor();
  const unsigned ipusPerReplica = graph.getTarget().getNumIPUs();
  const RingTopology ring(replicationFactor, ipusPerReplica,
                          graph.getTarget().getIpuLinkTopology());
  auto numFragments = replicationFactor;

  auto fragments = replicatedSplitIntoFragments(toReduce, numFragments, graph);
  auto fragmentSize = fragments.dim(1);
  auto srcBuffer = graph.addVariable(toReduce.elementType(), {fragmentSize},
                                     debugPrefix + "/ScatterSrc");
  mapBuffer(graph, srcBuffer, fragments);
  auto dstBuffer = graph.clone(srcBuffer, debugPrefix + "/ScatterDst");
  auto repFactorTensor = graph.addReplicationIndexConstant();

  // Map index tensor to IPU involved in this collective program
  graph.setTileMapping(repFactorTensor,
                       getScalarTile(graph.getTileMapping(toReduce)));

  CollectivesProgram program;

  program.repeatCounter = numSteps - 1;
  auto sliceIndex = ring.initRingIndexTensor(graph, direction, repFactorTensor,
                                             program.initIndex, debugPrefix,
                                             replicationFactor, startOffset);
  const unsigned incrementValue = direction == Direction::CLOCKWISE ? -1 : 1;
  // create program to change the slice index to it's next value.
  // called every iteration of the repeat
  popops::mapInPlace(graph,
                     (popops::expr::_1 + (replicationFactor + incrementValue)) %
                         replicationFactor,
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
  replicatedRankSlice(graph, fragments, sliceIndex, srcBuffer, ring, direction,
                      program.sliceFragments);
  // perform the reduction with the received data and the value sliced
  program.reduceProg =
      ReduceProg(srcBuffer, dstBuffer, op, debugPrefix + "/Reduce");
  program.undefTensor = concat({srcBuffer, dstBuffer});
  program.srcBuffer = std::move(srcBuffer);
  program.dstBuffer = std::move(dstBuffer);
  logging::debug("Unidirectional ring reduce scatter end");
  return program;
}

static Tensor
bidirectionalRingPairReduceScatter(Graph &graph, const Tensor &toReduce,
                                   popops::Operation op, Sequence &prog,
                                   const std::string &debugPrefix) {
  // split to reduce in half and call the clockwise and anticlockwise on
  // each. The bidirectionalSequence function will then interleave the
  // programs in the same repeat. Don't need to worry about ipu mapping when
  // splitting in half as this method won't be called unless one ipu per
  // replica
  logging::debug("Bidirectional ring reduce scatter");
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
  logging::debug("Meet in the middle reduce scatter");
  const auto replicationFactor = graph.getReplicationFactor();
  if (replicationFactor <= 2) {
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, CLOCKWISE, debugPrefix, replicationFactor);
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  const unsigned ipusPerReplica = graph.getTarget().getNumIPUs();
  RingTopology ring(replicationFactor, ipusPerReplica,
                    graph.getTarget().getIpuLinkTopology());
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

  unsigned topLevelControlTile =
      graph.getReplicationFactor() ==
              graph.getTopLevelGraph().getReplicationFactor()
          ? getScalarTile(graph.getTopLevelGraph().getTileMapping(toReduce))
          : 0;

  Sequence combineBuffers;
  opInPlace(graph, op, clockwiseProg.srcBuffer.get(),
            anticlockwiseProg.dstBuffer.get(), combineBuffers,
            debugPrefix + "/Reduce");
  prog.add(meetInMiddleReduceScatterSequence(clockwiseProg, anticlockwiseProg,
                                             graph, std::move(combineBuffers),
                                             topLevelControlTile));
  logging::debug("Meet in the middle ring reduce scatter end");
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
    logging::debug("Reduce scatter collective method is clockwise ring");
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, CLOCKWISE, debugPrefix,
        graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  case CollectiveMethod::ANTICLOCKWISE_RING: {
    logging::debug("reduce scatter collective method is anti-clockwise ring");
    auto program = unidirectionalRingReduceScatter(
        graph, toReduce, op, ANTICLOCKWISE, debugPrefix,
        graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return program.srcBuffer.get();
  }
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR: {
    logging::debug("Reduce scatter collective method is Bidirectional ring");
    return bidirectionalRingPairReduceScatter(graph, toReduce, op, prog,
                                              debugPrefix);
  }
  case CollectiveMethod::MEET_IN_MIDDLE_RING: {
    logging::debug("Reduce scatter collective "
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
  logging::debug("Unidirectional ring allGather");
  const auto replicationFactor = graph.getReplicationFactor();
  const unsigned ipusPerReplica = graph.getTarget().getNumIPUs();
  RingTopology ring(replicationFactor, ipusPerReplica,
                    graph.getTarget().getIpuLinkTopology());
  auto numFragments = replicationFactor;
  auto srcBuffer = graph.clone(toGather, debugPrefix + "/GatherSrc");
  auto dstBuffer = graph.clone(toGather, debugPrefix + "/GatherDst");
  auto paddedResult = padAllGatherResult(graph, toGather, numFragments, result);
  auto fragments =
      replicatedSplitIntoFragments(paddedResult, numFragments, graph);
  assert(fragments.dim(1) == toGather.numElements());
  CollectivesProgram program;

  program.repeatCounter = numSteps - 1;
  auto replicationIndex = graph.addReplicationIndexConstant();
  graph.setTileMapping(replicationIndex,
                       getScalarTile(graph.getTileMapping(toGather)));
  auto sliceIndex = ring.initRingIndexTensor(graph, direction, replicationIndex,
                                             program.initIndex, debugPrefix,
                                             replicationFactor, startOffset);
  program.firstGatherCopy.add(Copy(toGather, srcBuffer));
  const unsigned incrementValue = direction == Direction::CLOCKWISE ? -1 : 1;
  popops::mapInPlace(graph,
                     (popops::expr::_1 + (replicationFactor + incrementValue)) %
                         replicationFactor,
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
  replicatedRankUpdate(graph, srcBuffer, fragments, sliceIndex, ring, direction,
                       program.sliceFragments);
  program.undefTensor = concat({paddedResult, srcBuffer, dstBuffer});
  return program;
}

static void bidirectionalRingPairAllGather(Graph &graph, const Tensor &toGather,
                                           const Tensor &result, Sequence &prog,
                                           const std::string &debugPrefix) {
  logging::debug("Bidirectional ring allGather");
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
  logging::debug("Meet in the middle ring allGather");
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

  unsigned topLevelControlTile =
      graph.getReplicationFactor() ==
              graph.getTopLevelGraph().getReplicationFactor()
          ? getScalarTile(graph.getTopLevelGraph().getTileMapping(toGather))
          : 0;

  auto clockwiseProg = unidirectionalRingAllGather(
      graph, toGather, result, Direction::CLOCKWISE, debugPrefix + "/clockwise",
      numSteps, clockwiseOffset);
  auto anticlockwiseProg = unidirectionalRingAllGather(
      graph, toGather, result, Direction::ANTICLOCKWISE,
      debugPrefix + "/clockwise", numSteps - 1, anticlockwiseOffset);
  prog.add(meetInMiddleAllGatherSequence(clockwiseProg, anticlockwiseProg,
                                         graph, topLevelControlTile));
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
    logging::debug("All gather collective method is clockwise ring");
    auto program =
        unidirectionalRingAllGather(graph, toGather, result, CLOCKWISE,
                                    debugPrefix, graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return;
  }
  case CollectiveMethod::ANTICLOCKWISE_RING: {
    logging::debug("All gather collective method is anti-clockwise ring");
    auto program =
        unidirectionalRingAllGather(graph, toGather, result, ANTICLOCKWISE,
                                    debugPrefix, graph.getReplicationFactor());
    prog.add(unidirectionalSequence(program, graph));
    return;
  }
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR: {
    logging::debug("All gather collective method is Bidirectional ring");
    return bidirectionalRingPairAllGather(graph, toGather, result, prog,
                                          debugPrefix);
  }
  case CollectiveMethod::MEET_IN_MIDDLE_RING: {
    logging::debug("All gather collective method is Meet in the middle ring");
    return ringMeetInMiddleAllGather(graph, toGather, result, prog,
                                     debugPrefix);
  }
  }
}

poplar::Tensor replicatedReduceScatter(Graph &graph, const Tensor &toReduce,
                                       popops::Operation op, Sequence &prog,
                                       const std::string &debugPrefix,
                                       const OptionFlags &optionFlags) {

  return impl->replicatedReduceScatter(graph, toReduce, op, prog, debugPrefix,
                                       optionFlags);
}

poplar::Tensor ReplicatedCollectives::replicatedReduceScatter(
    Graph &graph, const Tensor &toReduce, popops::Operation op, Sequence &prog,
    const std::string &debugPrefix, const OptionFlags &optionFlags) {
  logging::info("replicatedReduceScatter data={}, op={}, name={}",
                toReduce.shape(), op, debugPrefix);
  logging::debug("Replicated reduce scatter begin ({}B)",
                 toReduce.numElements() *
                     graph.getTarget().getTypeSize(toReduce.elementType()));
  if (toReduce.rank() != 1) {
    throw poputil::poplibs_error("Input tensor to replicatedReduceScatter "
                                 "must have rank 1, but had rank " +
                                 std::to_string(toReduce.rank()));
  }

  CollectiveOptions options;
  parseCollectiveOptions(optionFlags, options);

  auto output =
      internalReduceScatter(graph, toReduce, op, prog, debugPrefix, options);
  logging::debug("Replicated reduce scatter end");
  return output;
}

static void noCheckReplicatedAllGather(Graph &graph, const Tensor &toGather,
                                       const Tensor &result, Sequence &prog,
                                       const std::string &debugPrefix,
                                       const poplar::OptionFlags &optionFlags) {
  CollectiveOptions options;
  parseCollectiveOptions(optionFlags, options);

  allGather(graph, toGather, result, prog, debugPrefix, options);
}

poplar::Tensor replicatedAllGather(Graph &graph, const Tensor &toGather,
                                   Sequence &prog,
                                   const std::string &debugPrefix,
                                   const poplar::OptionFlags &optionFlags) {
  return impl->replicatedAllGather(graph, toGather, prog, debugPrefix,
                                   optionFlags);
}

poplar::Tensor ReplicatedCollectives::replicatedAllGather(
    Graph &graph, const Tensor &toGather, Sequence &prog,
    const std::string &debugPrefix, const poplar::OptionFlags &optionFlags) {
  logging::info("replicatedAllGather data={}, name={}", toGather.shape(),
                debugPrefix);
  logging::debug("Replicated all gather begin ({}B)",
                 toGather.numElements() *
                     graph.getTarget().getTypeSize(toGather.elementType()));

  if (graph.getTopLevelGraph().getReplicationFactor() > 1 &&
      graph.getReplicationFactor() !=
          graph.getTopLevelGraph().getReplicationFactor()) {
    throw poputil::poplibs_error(
        "replicatedAllGather doesn't support a mix of single image and "
        "non-single image replication within the same graph.");
  }

  // Create a new view on the input.
  const Tensor input = toGather.flatten();

  // Create the output tensor.
  std::vector<poplar::Tensor> outputs(graph.getReplicationFactor());
  std::for_each(outputs.begin(), outputs.end(),
                [&](poplar::Tensor &out) { out = graph.clone(input); });
  poplar::Tensor output = poplar::concat(outputs);

  noCheckReplicatedAllGather(graph, input, output, prog, debugPrefix,
                             optionFlags);

  // Reshape the output to be in the shape of [numReplcias][InputShape].
  std::vector<std::size_t> newShape = toGather.shape();
  newShape.insert(newShape.begin(), graph.getReplicationFactor());

  output = output.reshape(newShape);

  logging::debug("Replicated all gather end");
  return output;
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
  logging::debug("Reorder data tensor for an all-reduce with {} fragments",
                 numFragments);
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
          logging::trace("Add interval [{},{}) on tile {} to fragment {}",
                         sliceBegin, sliceEnd, tile, fragment);
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
    return graph.getReplicationFactor();
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR:
    return graph.getReplicationFactor() * 2;
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
    return graph.getReplicationFactor();
  case CollectiveMethod::BIDIRECTIONAL_RING_PAIR:
    return graph.getReplicationFactor() * 2;
  }
}

static unsigned getAllReduceNumFragments(Graph &graph,
                                         const CollectiveOptions &options,
                                         const poplar::Tensor &data) {
  auto typeSize = graph.getTarget().getTypeSize(data.elementType());
  auto numElements = data.numElements();
  auto replicationFactor = graph.getReplicationFactor();
  auto numElementsAfterReduceScatter = ceildiv(numElements, replicationFactor);
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
    logging::debug("Using replicated version of allReduce");
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
                                   const std::string &debugPrefix,
                                   const poplar::OptionFlags &optionFlags) {
  impl->replicatedAllReduceWithOutput(graph, data, result, op, prog,
                                      debugPrefix, optionFlags);
}

void ReplicatedCollectives::replicatedAllReduceWithOutput(
    Graph &graph, const poplar::Tensor &data, poplar::Tensor &result,
    popops::Operation op, program::Sequence &prog,
    const std::string &debugPrefix, const poplar::OptionFlags &optionFlags) {
  logging::info(
      "replicatedAllReduceWithOutput data={}, result={}, op={}, name={}",
      data.shape(), result.shape(), op, debugPrefix);

  logging::debug("Replicated all reduce begin ({}B)",
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
  logging::debug("Replicated all reduce end");
}

void replicatedAllReduceInPlace(poplar::Graph &graph, poplar::Tensor &data,
                                popops::Operation op,
                                poplar::program::Sequence &prog,
                                const std::string &debugPrefix,
                                const poplar::OptionFlags &options) {
  return replicatedAllReduceWithOutput(graph, data, data, op, prog, debugPrefix,
                                       options);
}

Tensor replicatedAllReduce(Graph &graph, const poplar::Tensor &data,
                           popops::Operation op, program::Sequence &prog,
                           const std::string &debugPrefix,
                           const poplar::OptionFlags &optionFlags) {
  return impl->replicatedAllReduce(graph, data, op, prog, debugPrefix,
                                   optionFlags);
}

Tensor ReplicatedCollectives::replicatedAllReduce(
    Graph &graph, const poplar::Tensor &data, popops::Operation op,
    program::Sequence &prog, const std::string &debugPrefix,
    const poplar::OptionFlags &optionFlags) {
  logging::info("replicatedAllReduce data={}, op={}, name={}", data.shape(), op,
                debugPrefix);

  logging::debug("Replicated all reduce begin ({}B)",
                 data.numElements() *
                     graph.getTarget().getTypeSize(data.elementType()));
  auto result = graph.clone(data, debugPrefix + "/result");
  noCheckReplicatedAllReduce(graph, data, result, op, prog, debugPrefix,
                             optionFlags);
  logging::debug("Replicated all reduce end");
  return result;
}

Tensor replicatedAllReduce(Graph &graph, Graph &parentGraph,
                           const poplar::Tensor &data, popops::Operation op,
                           program::Sequence &prog,
                           const std::string &debugPrefix,
                           const poplar::OptionFlags &optionFlags) {
  return impl->replicatedAllReduce(graph, parentGraph, data, op, prog,
                                   debugPrefix, optionFlags);
}

Tensor ReplicatedCollectives::replicatedAllReduce(
    Graph &graph, Graph &parentGraph, const poplar::Tensor &data,
    popops::Operation op, program::Sequence &prog,
    const std::string &debugPrefix, const poplar::OptionFlags &optionFlags) {
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
                                    const std::string &debugPrefix,
                                    const poplar::OptionFlags &options) {
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
