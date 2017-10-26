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

/** Create vertices with matching elements in t2d and s2d
 * \param vName     The base name of vertices to create
 * \param graph     The graph to update
 * \param cs        The compute set to update
 * \param offset    The offset within t2d corresponding to the first element in
 *                  s2d. A single element for all tiles, or one element per tile
 * \param t2d       A 2d base tensor
 * \param s2d       A 2d sub tensor
 **/
static void generateVertices(std::string vertexName,
                             Graph &graph,
                             ComputeSet &cs,
                             const Tensor &offset,
                             const Tensor &t2d,   // 2d base Tensor [sliceD][]
                             const Tensor &s2d)   // 2d sub Tensor [sizeD][]
{
  assert(t2d.rank() == 2);
  assert(s2d.rank() == 2);
  assert(t2d.dim(1) == s2d.dim(1));
  const auto &deviceInfo = graph.getDevice().getDeviceInfo();
  const auto dataPathWidth = deviceInfo.dataPathWidth;
  const auto grainSize = deviceInfo.getVectorWidth(t2d.elementType());
  const auto numTiles = deviceInfo.getNumTiles();
  const unsigned numBaseElements = t2d.dim(0);
  const unsigned numSubElements = s2d.dim(0);
  assert(numSubElements <= numBaseElements);

  // offset can be specified for each tile
  assert((offset.rank() == 0 && offset.numElements() == 1) ||
         (offset.rank() == 1 && offset.numElements() == numTiles));

  // map vertices following the mapping of t's first slice
  const auto mapping = graph.getTileMapping(t2d[0]);
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(t2d[0], mapping[tile]);
    if (tileContiguousRegions.size() == 0)
      // do nothing on this tile
      continue;

    auto &tileOffset = offset.numElements() == 1 ? offset : offset[tile];
    if (tileContiguousRegions.size() == 1) {
      unsigned regionSize = 0;
      std::vector<Tensor> baseSlices, subSlices; // [slice]
      for (auto &regions : tileContiguousRegions) {
        for (const auto &region : regions) {
          regionSize += region.size();
          for (unsigned slice = 0; slice != numBaseElements; ++slice)
            baseSlices.emplace_back(t2d[slice].slice(region));
          for (unsigned slice = 0; slice != numSubElements; ++slice)
            subSlices.emplace_back(s2d[slice].slice(region));
        }
      }

      Tensor tileBase = concat(baseSlices);
      Tensor tileSub = concat(subSlices);

      if (tileBase.isContiguous()) {
        auto numWorkers = deviceInfo.numWorkerContexts;
        auto elementsPerWorker = (regionSize + numWorkers - 1)
                                 / numWorkers;
        auto v = graph.addVertex(cs,
                                templateVertex(vertexName + "2d",
                                                t2d.elementType()),
                                 {{"offset", tileOffset},
                                  {"baseT", tileBase},
                                  {"subT", tileSub}
                                 });
        graph.setInitialValue(v["numBaseElements"], numBaseElements);
        graph.setInitialValue(v["numSubElements"], numSubElements);
        graph.setInitialValue(v["regionSize"], regionSize);
        graph.setInitialValue(v["elementsPerWorker"], elementsPerWorker);
        graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
        graph.setInitialValue(v["numWorkers"], numWorkers);
        graph.setTileMapping(v, tile);
        continue;
      }
    }

    auto vertexSeqs =
      splitRegionsBetweenWorkers(deviceInfo, tileContiguousRegions,
                                 grainSize, 2 * grainSize);
    for (const auto &sequences : vertexSeqs) {
      // vector of sequences per vertex
      std::vector<Tensor> base, sub;
      for (const auto &regions : sequences) {
        for (const auto &region : regions) {
          for (unsigned slice = 0; slice != numBaseElements; ++slice) {
            base.emplace_back(t2d[slice].slice(region));
          }
          for (unsigned slice = 0; slice != numSubElements; ++slice) {
            Tensor subRegion = s2d[slice].slice(region);
            sub.emplace_back(std::move(subRegion));
          }
        }
      }
      auto v = graph.addVertex(cs,
                               templateVertex(vertexName, t2d.elementType()),
                               {{"offset", tileOffset},
                                {"baseT", base},
                                {"subT", sub}
                               });
      graph.setInitialValue(v["numBaseElements"], numBaseElements);
      graph.setInitialValue(v["numSubElements"], numSubElements);
      graph.setInitialValue(v["dataPathWidth"], dataPathWidth);
      graph.setTileMapping(v, tile);
    }
  }
}

/** Return the sub-tensor acquired by indexing 't' at position 'offset' in
 * dimension 'dim'. The other output dimensions will match the size of the
 * corresponding input dimensions.
 *
 * \param graph           The poplar graph
 * \param t               The source tensor
 * \param offset          The offset in \a's \a dim dimension. This tensor must
 *                        have a single element, or an element per tile
 * \param dim             The dimension to slice
 * \param numOutIndices   The size of the output Tensor in the sliced dimension
 * \param prog            The program to be updated
 * \param debugPrefix     The prefix prepended to debugging info
 * \returns               The specified subtensor
 */
static Tensor slice(Graph &graph,
                    const Tensor &t,
                    const Tensor &offset,
                    unsigned dim,
                    unsigned numOutIndices,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix)
{
  const unsigned numInIndices = t.dim(dim);
  assert(dim < t.rank());
  assert(numOutIndices <= t.dim(dim));
  // Get a 2d view of the source tensor, with the dim we're slicing at dim0
  // and the other dimensions collapsed into dim1
  Tensor t2d = t.dimRoll(dim).reshape({numInIndices,
                                       t.numElements() / numInIndices});
  Tensor s = graph.clone(t.slice(0, numOutIndices, dim),
                         debugPrefix + "/sliced_" + std::to_string(dim));
  Tensor s2d = s.dimRoll(dim).reshape({numOutIndices,
                                       s.numElements() / numOutIndices});
  auto cs = graph.addComputeSet(debugPrefix + "/slice");

  generateVertices("popstd::DynamicSelect",
                   graph, cs, offset, t2d, s2d);
  prog.add(Execute(cs));

  return s;
}

/** Update the sub-tensor at 'offset; within \a t's dimension 'dim' with the
 *  contents of 's'
 *
 *  \param graph        The poplar graph
 *  \param t            The base tensor
 *  \param s            The subtensor to insert. Its dimensions must match t's,
 *                      except in dimension \a dim
 *  \param offset       The offset in \a t's \a dim dimension. This tensor must
 *                      have either a single element, or an element per tile
 *  \param dim          The dimension in which to insert
 *  \param prog         The program to be updated
 *  \param debugPrefix  The prefix prepended to debugging info
 **/
static void update(Graph &graph,
                   const Tensor &t,
                   const Tensor &s,
                   const Tensor &offset,
                   unsigned dim,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix)
{
  const unsigned numTElements = t.dim(dim);
  const unsigned numSElements = s.dim(dim);
  assert(t.rank() == s.rank());
  for (unsigned d = 0; d != t.rank(); ++d) {
    if (d != dim)
      assert (s.dim(d) == t.dim(d));
    else
      assert (s.dim(d) <= t.dim(d));
  }
  assert(dim < t.rank());
  assert(numSElements <= numTElements);
  // Get a 2d view of the source tensor, with the dim we're updating at dim0
  // and the other dimensions collapsed into dim1
  Tensor t2d = t.dimRoll(dim).reshape({numTElements,
                                       t.numElements() / numTElements});
  Tensor s2d = s.dimRoll(dim).reshape({numSElements,
                                       s.numElements() / numSElements});
  auto cs = graph.addComputeSet(debugPrefix + "/update");

  generateVertices("popstd::DynamicUpdateSlice",
                   graph, cs, offset, t2d, s2d);
  prog.add(Execute(cs));

}

// If we are slicing up a tensor with the given `shape` in the dimensions
// `dims`, and the slice size in each dimension is `sizes`, then what is
// the best order to do the slices? The returned vector contains
// indexes into `dims` (and `sizes`).
static std::vector<size_t> bestSliceOrder(const std::vector<std::size_t> &shape,
                                          const std::vector<std::size_t> &dims,
                                          const std::vector<std::size_t> &sizes)
{

  assert(dims.size() == sizes.size());
  assert(dims.size() == shape.size());

  // Process the dimensions in an order that slices out the most elements
  // first. That dimension is the one that reduces the size of the tensor
  // to the lowest percentage of its former size. Since each slice only
  // reduces the tensor's size in one dimension, that percentage is equal to
  //
  //    sizes[a] / shape[dims[a]]
  //
  // so if we sort on  (sizes[a] / shape[dims[a]] < sizes[b] / shape[dims[b]])
  // then we should end up slicing in an optimal order.

  // Initialise with default order (0, 1, 2...)
  std::vector<size_t> idxOrder(dims.size());
  std::iota(idxOrder.begin(), idxOrder.end(), 0);

  // Sort the most slicey dimension first. Assumes no integer overflows.
  std::sort(idxOrder.begin(), idxOrder.end(),
            [&](size_t a, size_t b) {
              return sizes[b] * shape[dims[a]] > sizes[a] * shape[dims[b]];
            });

  return idxOrder;
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
  if (offset.rank() > 2 || offset.dim(0) != dims.size()
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
  Tensor out = t;

  auto idxOrder = bestSliceOrder(t.shape(), dims, sizes);

  for (auto i : idxOrder) {
    out = slice(graph, out,
                offset[i],
                dims[i],
                sizes[i],
                prog,
                debugPrefix + "/dynamicSlice_d" + std::to_string(dims[i]));
  }

  return out;
}

void dynamicUpdate(Graph &graph,
                   const Tensor &t,
                   const Tensor &s,
                   const Tensor &offset,
                   const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix = "")
{
  // We insert into a single dimension at a time. When more than one dimension
  // is to be inserted this entails slicing off the outer dimensions until there
  // is a single dynamic dimension. That Tensor is updated with s. Then
  // the dimension traversal is reversed, updating one into one extra dimension
  // each time.
  if (offset.rank() == 0)
    return;

  auto idxOrder = bestSliceOrder(t.shape(), dims, sizes);

  std::vector<Tensor> reducedT;
  reducedT.emplace_back(t); // reducedT[0] = t
  // slice off the larger dimensions one at a time
  for (unsigned i = 0; i != idxOrder.size() - 1; ++i) {
    auto dim = idxOrder[i];
    reducedT.emplace_back(slice(graph, reducedT[i],
                                offset[dim],
                                dims[dim],
                                sizes[dim],
                                prog,
                                debugPrefix + "/dynamicUpdateS_d" +
                                std::to_string(dims[i])));
  }
  // copy s into the reduced t, iterating back to full dimensions
  reducedT.emplace_back(s);
  for (unsigned ii = idxOrder.size(); ii != 0; --ii) {
    auto i = ii - 1;
    auto dsIdx = idxOrder[i]; // index into dims[] and sizes[]
    update(graph, reducedT[i], reducedT[i + 1], offset[dsIdx],
           dims[dsIdx], prog,
           debugPrefix + "/dynamicUpdateU_d" + std::to_string(dims[dsIdx]));
  }
}

} // end namespace popstd
