#include "popstd/Regroup.hpp"
#include "util/gcd.hpp"
#include "popstd/VertexTemplates.hpp"
#include "popstd/Util.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/Interval.hpp"
#include <cassert>
#include <numeric>
#include <algorithm>

using namespace poplar;
using namespace poplar::program;

namespace popstd {

/** Return the sub-tensor acquired by indexing 't' at position 'offset' in
 * dimension 'dim'. The other output dimensions will match the size of the
 * corresponding input dimensions.
 *
 * \param graph       The poplar graph
 * \param t           The source tensor
 * \param offset      The offset in \a's \a dim dimension. This tensor must be
 *                    rank 0
 * \param dim         The dimension to slice
 * \param numElements The size of the output Tensor in the sliced dimension
 * \param prog        The program to be updated
 * \param debugPrefix The prefix prepended to debugging info
 * \returns           The specified subtensor
 */
static Tensor dynamicSlice(Graph &graph,
                    const Tensor &t,
                    const Tensor &offset,
                    unsigned dim,
                    unsigned numOutElements,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix)
{
  const unsigned numInElements = t.dim(dim);
  assert(dim < t.rank());
  assert(numOutElements <= t.dim(dim));
  assert(offset.rank() == 0); // Index must be a rank-0 tensor
  // Get a 2d view of the source tensor, with the dim we're slicing at dim0
  // and the other dimensions collapsed into dim1
  Tensor t2d = t.dimRoll(dim).reshape({numInElements,
                                       t.numElements() / numInElements});
  Tensor s = graph.clone(t.slice(0, numOutElements, dim),
                         "sliced_" + std::to_string(dim));
  Tensor s2d = s.dimRoll(dim).reshape({numOutElements,
                                       s.numElements() / numOutElements});

  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto grainSize = deviceInfo.getVectorWidth(t.elementType());
  const auto numTiles = deviceInfo.getNumTiles();

  // map all output slices following the mapping of the first input slice
  const auto mapping = graph.getTileMapping(t2d[0]);
  auto cs = graph.addComputeSet(debugPrefix + "/slice");
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(t2d[0], mapping[tile]);
    auto vertexSeqs =
      splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                 grainSize, 2 * grainSize);
    for (const auto &sequences : vertexSeqs) {
      // vector of sequences per vertex

      std::vector<Tensor> in, out;
      for (const auto &regions : sequences) {
        for (const auto &region : regions) {
          for (unsigned slice = 0; slice != numInElements; ++slice) {
            in.emplace_back(t2d[slice].slice(region));
          }
          for (unsigned slice = 0; slice != numOutElements; ++slice) {
            Tensor outRegion = s2d[slice].slice(region);
            out.emplace_back(std::move(outRegion));
          }
        }
      }
      auto v = graph.addVertex(cs,
                               templateVertex("popstd::DynamicSelect",
                                              t.elementType()),
                               {{"offset", offset},
                                {"in", in},
                                {"out", out}
                               });
      graph.setInitialValue(v["numInElements"], numInElements);
      graph.setInitialValue(v["numOutElements"], numOutElements);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
  prog.add(Execute(cs));

  return s;
}

Tensor dynamicSlice(Graph &graph,
                    const Tensor &t,
                    const Tensor &offset,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix = "")
{
  auto tRank = t.rank();
  if (offset.rank() != 1 || offset.numElements() != dims.size()
      || dims.size() != sizes.size())
    throw graph_connection_error(
      "dynamicSlice offset (" + std::to_string(offset.numElements()) +
      "), dims (" + std::to_string(dims.size()) +
      ") and sizes " + std::to_string(sizes.size()) +
      ") must all be the same size");
  for (unsigned i = 0; i != dims.size(); ++i) {
    if (dims[i] >= tRank)
      throw graph_connection_error(
        "dynamicSlice: invalid dimension " + std::to_string(dims[i]));
    if (sizes[i] == 0)
      // Should this be allowed?
      throw graph_connection_error(
        "dynamicSlice: requested empty dimension");
    if (sizes[i] > t.dim(dims[i]))
      throw graph_connection_error(
        "dynamicSlice: requested output dimension bigger than input");
  }
  // process variable offsets in order of decreasing size
  Tensor out = t;
  std::vector<size_t> idxOrder(dims.size());
  std::iota(idxOrder.begin(), idxOrder.end(), 0);
  std::sort(idxOrder.begin(), idxOrder.end(),
            [&](size_t a, size_t b) {
              return t.dim(dims[a]) > t.dim(dims[b]);});

  for (auto i : idxOrder) {
    out = dynamicSlice(graph, out,
                       offset[i].reshape({}),
                       dims[i],
                       sizes[i],
                       prog,
                       debugPrefix + "dynamicSlice_d" +
                       std::to_string(dims[i]));
  }

  return out;
}

} // end namespace popstd
