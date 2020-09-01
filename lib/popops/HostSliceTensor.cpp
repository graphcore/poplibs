// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <poplar/Graph.hpp>
#include <poplibs_support/Algorithm.hpp>
#include <poplibs_support/logging.hpp>
#include <popops/HostSliceTensor.hpp>
#include <poputil/exceptions.hpp>
#include <unordered_map>

using namespace poplibs_support;

namespace popops {

namespace {

const unsigned numXBs = 32;
const unsigned writePacketSizeInBytes = 256;
const unsigned readPacketSizeInBytes = 1024;

struct XBs {
  struct XBValue {
    unsigned value;
    XBValue(unsigned value) : value(value) {}
    unsigned getDistance() const {
      // so that the map can be ordered by xbs closest to the west edge
      // provide function to get distance
      const auto halfWay = numXBs / 2;
      if (value < halfWay) {
        return 2 * value;
      }
      // after half way the distance beomes smaller as value gets larger
      // +1 is so that the values are unique from the values in the first half
      return (2 * ((numXBs - 1) - value)) + 1;
    }
  };
  struct XBByDistance {
    bool operator()(const XBValue &A, const XBValue &B) const {
      return A.getDistance() < B.getDistance();
    }
  };

  using MapType =
      boost::container::flat_map<XBValue, std::vector<unsigned>, XBByDistance>;
  MapType xbToTiles;

  void findNextIterators(MapType::const_iterator &xbIt,
                         unsigned &tileCounter) const {
    assert(xbIt != xbToTiles.end());
    while (true) {
      ++xbIt;
      if (xbIt == xbToTiles.end()) {
        xbIt = xbToTiles.begin();
        ++tileCounter;
      }
      if (tileCounter < xbIt->second.size()) {
        return;
      }
    }
  }
};

struct PacketsAndIndices {
  std::vector<poplar::Tensor> packets;
  std::vector<unsigned> indices;
};

} // namespace

static std::vector<std::vector<size_t>>
getPerIpuShapes(const std::vector<size_t> &shape, const unsigned numIpus) {
  if (numIpus == 1) {
    return {shape};
  }
  throw poputil::poplibs_error(
      "Create host sliceable tensor doesn;t support multiple IPUs yet");
}

static poplar::Graph getIpuGraph(poplar::Graph &graph,
                                 const poplar::Target &target,
                                 const unsigned ipu) {
  const auto tilesPerIpu = target.getTilesPerIPU();
  return graph.createVirtualGraph(ipu * tilesPerIpu, (ipu + 1) * tilesPerIpu);
}

static XBs::XBValue getXB(const poplar::Graph &graph, const unsigned tile) {
  const auto physTile = graph.convertVirtualTileToPhysicalTile(tile);
  const unsigned validBits = physTile % 64;
  const unsigned column = validBits >> 2;
  const unsigned side = validBits % 2;
  return XBs::XBValue((2 * column) + side);
}

static XBs findAvailableXBs(const poplar::Graph &graph) {
  XBs result;
  for (unsigned tile = 0; tile < graph.getTarget().getNumTiles(); ++tile) {
    const auto xb = getXB(graph, tile);
    result.xbToTiles[xb].emplace_back(tile);
  }
  // for odd xb values the larger tile ids are closer to the spine, for even the
  // opposite is true above tiles are inserted in order with smaller tile ids at
  // the start of the vector To select the closer tiles first sort odd xbs
  for (auto &entry : result.xbToTiles) {
    if (entry.first.value % 2 == 1) {
      std::sort(entry.second.begin(), entry.second.end(),
                std::greater<unsigned>());
    }
  }

  return result;
}

static PacketsAndIndices splitIntoPackets(poplar::Tensor &t,
                                          const poplar::Graph &graph,
                                          const bool isRead) {
  const unsigned packetSizeInBytes =
      isRead ? readPacketSizeInBytes : writePacketSizeInBytes;
  const unsigned packetSize =
      packetSizeInBytes / graph.getTarget().getTypeSize(t.elementType());
  PacketsAndIndices result;
  result.packets.reserve(t.numElements() / packetSize);
  result.indices.reserve(t.numElements() / packetSize);
  std::unordered_map<unsigned, unsigned> sizeBreakDown;
  for (unsigned i = 0; i < t.dim(0); ++i) {
    // the outer dimension is for the offset tensor so can't be in same packet
    // as elements from a different out dimension
    const auto row = t[i];
    assert(row.rank() == 1);
    for (unsigned p = 0; p < ceildiv(row.dim(0), packetSize); ++p) {
      const auto start = p * packetSize;
      const auto end = std::min((p + 1) * packetSize, (unsigned)row.dim(0));
      result.indices.push_back(i);
      result.packets.emplace_back(row.slice(start, end));
      if (logging::popops::shouldLog(logging::Level::Trace)) {
        sizeBreakDown.emplace(end - start, 0);
        ++sizeBreakDown[end - start];
      }
    }
  }
  for (const auto entry : sizeBreakDown) {
    logging::popops::trace("{} packets of size {}", entry.second, entry.first);
  }
  return result;
}

static std::vector<unsigned> numPacketsPerTile(const XBs &xbs,
                                               const poplar::Target &target,
                                               const unsigned numPackets) {
  // assign tiles a roughly even amount of packets each but if some tiles have
  // extra try and fill a different xb context for each one
  std::vector<unsigned> result(target.getNumTiles(),
                               numPackets / target.getNumTiles());
  const unsigned extra = numPackets % target.getNumTiles();
  auto xbIt = xbs.xbToTiles.begin();
  unsigned tileCounter = 0;
  for (unsigned i = 0; i < extra; ++i) {
    const auto tile = xbIt->second[tileCounter];
    ++result[tile];
    xbs.findNextIterators(xbIt, tileCounter);
  }
  return result;
}

static void assignTileMappings(const PacketsAndIndices &packets,
                               const poplar::Tensor &indices, const XBs &xbs,
                               poplar::Graph &graph) {
  assert(packets.packets.size() % indices.dim(0) == 0);
  assert(indices.rank() == 1);
  assert(packets.packets.size() == packets.indices.size());
  const auto packetsPerTile =
      numPacketsPerTile(xbs, graph.getTarget(), packets.packets.size());
  const auto numTiles = graph.getTarget().getNumTiles();
  auto packetCounter = 0U;
  unsigned lastIndex = ~0U;
  for (unsigned tile = 0; tile < numTiles; ++tile) {
    for (unsigned p = packetCounter; p < packetCounter + packetsPerTile[tile];
         ++p) {
      graph.setTileMapping(packets.packets[p], tile);
      if (packets.indices[p] != lastIndex) {
        lastIndex = packets.indices[p];
        // Set the tile mapping of the indices tensor to be the first tile
        // which has packets that will use that index
        graph.setTileMapping(indices[lastIndex], tile);
      }
    }
    packetCounter += packetsPerTile[tile];
  }
}

static void createPerIpuTensors(std::vector<poplar::Tensor> &toConcat,
                                std::vector<poplar::Tensor> &indicesToConcat,
                                poplar::Graph &graph, const poplar::Type &type,
                                const std::vector<size_t> &shape,
                                const bool isRead,
                                const std::string &debugPrefix) {
  assert(graph.getTarget().getNumIPUs() == 1U);
  toConcat.emplace_back(
      graph.addVariable(type, shape, debugPrefix + "/HostSliceAble"));
  indicesToConcat.emplace_back(graph.addVariable(
      poplar::UNSIGNED_INT, {shape[0]}, debugPrefix + "/HostSliceIndices"));
  const auto xbs = findAvailableXBs(graph);
  auto &t = toConcat.back();
  auto &indices = indicesToConcat.back();
  const auto packets = splitIntoPackets(t, graph, isRead);
  assignTileMappings(packets, indices, xbs, graph);
}

IndicesAndTensor createHostSliceableTensor(poplar::Graph &graph,
                                           const poplar::Type &type,
                                           const std::vector<size_t> &shape,
                                           const bool isRead,
                                           const std::string &debugPrefix) {
  logging::popops::info("createHostSliceableTensor begin");
  if (shape.size() != 2U) {
    throw poputil::poplibs_error(
        "Host sliceable tensors must have rank of 2 not " +
        std::to_string(shape.size()));
  }
  const auto &target = graph.getTarget();
  const auto numIpus = target.getNumIPUs();
  std::vector<poplar::Tensor> toConcat;
  toConcat.reserve(target.getNumTiles());
  std::vector<poplar::Tensor> indicesToConcat;
  indicesToConcat.reserve(target.getNumTiles());
  const auto perIpuShapes = getPerIpuShapes(shape, numIpus);
  for (unsigned i = 0; i < numIpus; ++i) {
    auto ipuGraph = getIpuGraph(graph, target, i);
    createPerIpuTensors(toConcat, indicesToConcat, ipuGraph, type,
                        perIpuShapes[i], isRead, debugPrefix);
  }
  const auto result = concat(toConcat);
  const auto indices = concat(indicesToConcat);
  assert(result.shape() == shape);
  logging::popops::info("createHostSliceableTensor end");
  return {indices, result};
}

poplar::Tensor createHostTransferableTensor(poplar::Graph &graph,
                                            const poplar::Type &type,
                                            const std::vector<size_t> &shape,
                                            bool isRead,
                                            const std::string &debugPrefix) {
  size_t flattenedSize = std::accumulate(shape.begin(), shape.end(), 1U,
                                         std::multiplies<size_t>());
  auto resultPair = createHostSliceableTensor(graph, type, {1, flattenedSize},
                                              isRead, debugPrefix);

  return resultPair.tensor.reshape(shape);
}

} // namespace popops
