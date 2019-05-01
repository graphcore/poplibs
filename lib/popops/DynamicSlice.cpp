#include "popops/DynamicSlice.hpp"

#include "poplibs_support/gcd.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/Util.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/Interval.hpp"
#include "poplar/Program.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include <cassert>
#include <numeric>
#include <algorithm>
using namespace poplar;
using namespace poplar::program;
using namespace poputil;

namespace popops {

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
                             Tensor t2d,   // 2d base Tensor [sliceD][]
                             Tensor s2d)   // 2d sub Tensor [sizeD][]
{
  assert(t2d.rank() == 2);
  assert(s2d.rank() == 2);
  assert(t2d.dim(1) == s2d.dim(1));
  const auto &target = graph.getTarget();
  const auto grainSize = target.getVectorWidth(t2d.elementType());
  const auto numTiles = target.getNumTiles();
  const unsigned numBaseElements = t2d.dim(0);
  const unsigned numSubElements = s2d.dim(0);
  unsigned elemSize = s2d.dim(1);
  assert(numSubElements <= numBaseElements);

  // Offset must be a scalar. It will be replicated over tiles
  // by the small graph  replication optimisation during lowering.
  assert(offset.rank() == 0 && offset.numElements() == 1);
  // Reorder every element to minimize the number of contiguous regions when
  // copying.
  std::vector<Tensor *> toRearrange;
  std::vector<Tensor> s2dElems(numSubElements), t2dElems(numBaseElements);

  for (unsigned i = 0; i != numSubElements; ++i) {
    s2dElems[i] = s2d[i];
    if (i != 0)
      toRearrange.push_back(&s2dElems[i]);
  }
  for (unsigned i = 0; i != numBaseElements; ++i) {
    t2dElems[i] = t2d[i];
    toRearrange.push_back(&t2dElems[i]);
  }
  graph.reorderToSimplify(&s2dElems[0], toRearrange);

  // Reordering may cause the element size to change if there were repeated
  // elements in s2d.
  elemSize = s2dElems[0].numElements();
  s2d = concat(s2dElems).reshape({numSubElements, elemSize});
  t2d = concat(t2dElems).reshape({numBaseElements, elemSize});

  // instantiate vertices following the mapping of t's first slice
  const auto mapping = graph.getTileMapping(t2d[0]);
  for (unsigned tile = 0; tile != numTiles; ++tile) {

    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(t2d[0], mapping[tile]);
    if (tileContiguousRegions.size() == 0)
      // do nothing on this tile
      continue;

    assert(offset.numElements() == 1);
    if (tileContiguousRegions.size() == 1) {
      unsigned regionSize = 0;
      std::vector<Tensor> baseSlices, subSlices; // [slice]
      auto &regions = tileContiguousRegions[0];
      for (const auto &region : regions) {
        regionSize += region.size();
        baseSlices.emplace_back(t2d.transpose().slice(region));
        subSlices.emplace_back(s2d.transpose().slice(region));
      }

      Tensor tileBase = concat(baseSlices).transpose().flatten();
      Tensor tileSub = concat(subSlices).transpose().flatten();

      if (tileBase.isContiguous()) {
        auto v = graph.addVertex(cs,
                                 templateVertex(vertexName + "Supervisor",
                                                t2d.elementType()),
                                 {{"offset", offset},
                                  {"baseT", tileBase},
                                  {"subT", tileSub}
                                 });
        graph.setInitialValue(v["numBaseElements"], numBaseElements);
        graph.setInitialValue(v["numSubElements"], numSubElements);
        graph.setInitialValue(v["regionSize"], regionSize);
        graph.setTileMapping(v, tile);
        continue;
      }
    }

    auto vertexSeqs =
      splitRegionsBetweenWorkers(target, tileContiguousRegions,
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
                               templateVertex(vertexName + "2d",
                                t2d.elementType()),
                               {{"offset", offset},
                                {"baseT", base},
                                {"subT", sub}
                               });
      graph.setInitialValue(v["numBaseElements"], numBaseElements);
      graph.setInitialValue(v["numSubElements"], numSubElements);
      graph.setInitialValue(v["numRegions"], base.size()/numBaseElements);
      graph.setTileMapping(v, tile);
    }
  } // end loop over tiles
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
 * \param prog            Pointer to program to be updated. If the program
 *                        pointer is nullptr, vertices are not generated
 * \param debugPrefix     The prefix prepended to debugging info
 * \returns               The specified subtensor
 */
static Tensor slice(Graph &graph,
                    const Tensor &t,
                    const Tensor &offset,
                    unsigned dim,
                    unsigned numOutIndices,
                    poplar::program::Sequence *prog,
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

  rebalanceTensor(graph, s);
  if (prog != nullptr) {
    Tensor s2d = s.dimRoll(dim).reshape({numOutIndices,
                                         s.numElements() / numOutIndices});
    auto cs = graph.addComputeSet(debugPrefix + "/slice");

    generateVertices("popops::DynamicSlice", graph, cs, offset, t2d, s2d);
    prog->add(Execute(cs));
  }
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

  generateVertices("popops::DynamicUpdateSlice",
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
  assert(dims.size() <= shape.size());

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

static void ValidateParams(std::string name,
                           const Tensor &t,
                           const Tensor &offset,
                           const std::vector<std::size_t> &dims,
                           const std::vector<std::size_t> &sizes,
                           bool checkOffset = true
                           ) {
  auto tRank = t.rank();
  std::string exceptionStr;
  if (checkOffset) {
    auto offsetElems = offset.rank() == 0 ? 0 : offset.dim(0);
   if  (offset.rank() > 2 || offsetElems != dims.size())
     exceptionStr = name + " offset (" + std::to_string(offsetElems) + ") ";
  }
  if (dims.size() != sizes.size()) {
    exceptionStr +=  "dims (" + std::to_string(dims.size()) +
                      ") and sizes " + std::to_string(sizes.size()) + ") ";
  }
  if (!exceptionStr.empty()) {
    exceptionStr +=  ": must be the same size";
    throw graph_connection_error(exceptionStr);
  }
  std::vector<bool> dimUsed(dims.size());
  for (unsigned i = 0; i != dims.size(); ++i) {
    if (dims[i] >= tRank)
      throw graph_connection_error(
        name + ": invalid dimension " + std::to_string(dims[i]));
    if (sizes[i] > t.dim(dims[i]))
      throw graph_connection_error(
        name + ": requested slice dimension bigger than buffer");
    if (dimUsed[dims[i]])
      throw graph_connection_error(
        name + ": dimension " + std::to_string(dims[i])
        + " specified multiple times");
    dimUsed[dims[i]] = true;
  }
}

// Create and map a tensor so that dynamic slicing of it will not require
// exchange
// The underlying layout will be [U/N][S0]..[Sn][N] where
// N is the number of contiguous unsliced elements per tile
// U is the product of the unsliced dimensions
// S0-Sn are the sliced dimensions, sorted to optimise the number of copies
// This distibutes the input/output slice across U/N tiles.
// If U/N << numTiles an outer stage can be added to convert part of an
// S dimension to an extra U dimensions
poplar::Tensor
createSliceableTensor(poplar::Graph &graph,
                      const poplar::Type &type,
                      const std::vector<size_t> &shape,
                      const std::vector<size_t> &dims,
                      const std::vector<std::size_t> &sizes,
                      const std::string &debugPrefix)
{
  const auto numTiles = graph.getTarget().getNumTiles();
  const unsigned numDims = shape.size();
  const unsigned numSlicedDims = dims.size();
  if (numSlicedDims == 0) {
    // no slicing specified
    auto t = graph.addVariable(type, shape);
    mapTensorLinearly(graph, t);
    return t;
  }

  std::vector<size_t> internalShape;
  // vector recording the permutation from [internal] to external dimension
  std::vector<unsigned> externalDims(shape.size());
  std::vector<bool> slicedDim(shape.size(), false);
  size_t sliceNumElements = 1; // number of elements in an output slice
  size_t nonSliceNumElements = 1;
  for (auto d : dims) {
    if (d >= shape.size())
      throw poputil::poplibs_error(
          "createSliceableTensor called to slice dimension " +
          std::to_string(d) + " but the target has rank " +
          std::to_string(shape.size()));
    if (slicedDim[d])
      throw poputil::poplibs_error(
          "createSliceableTensor called with repeated dims entry");
    slicedDim[d] = true;
    sliceNumElements *= shape[d];
  }

  // Unsliced dimensions on the outside, sliced on the inside - this makes
  // the unsliced dimensions contiguous on the tiles.
  std::vector<size_t> unslicedExternalDims;
  for (auto d = 0u; d != numDims; ++d) {
    if (!slicedDim[d]) {
      nonSliceNumElements *= shape[d];
      externalDims[d] = unslicedExternalDims.size();
      unslicedExternalDims.emplace_back(shape[d]);
    }
  }
  internalShape.emplace_back(nonSliceNumElements);// single outer dim
  // The number of elements that would be returned by each tile if we're
  // balanced across all tiles
  size_t balancedOutPerTile = (nonSliceNumElements + numTiles - 1) / numTiles;

  // Order the sliced indices to optimise multi-dimensional slicing.
  auto idxOrder = bestSliceOrder(shape, dims, sizes);
  for (auto i = 0u; i != dims.size(); ++i) {
    auto d = dims.at(idxOrder[i]);
    if (slicedDim[d]) {
      externalDims[d] = unslicedExternalDims.size() + internalShape.size() - 1;
      internalShape.emplace_back(shape[d]);
    }
  }
  // add an innermost dimension holding the number of consecutive elements
  // output by each tile
  auto numOutPerTile = gcd(internalShape.front(), balancedOutPerTile);
  internalShape.front() /= numOutPerTile;
  internalShape.emplace_back(numOutPerTile);

  assert(shape.size() + 1 ==
         internalShape.size() - 1 + unslicedExternalDims.size());
  Tensor core = graph.addVariable(type, internalShape,
                                  debugPrefix + "/sliceable");

  auto grainSize = sliceNumElements * balancedOutPerTile;
  mapTensorLinearly(graph, core, 0, grainSize);

  // roll the unsliced dimensions together
  auto reordered = core.dimRoll(internalShape.size()-1, 1);
  // reshape them to the original external sizes
  reordered = reordered.reshapePartial(0, 2, unslicedExternalDims);
  // shuffle to external dimension order
  reordered = reordered.dimShuffle(externalDims);
  return reordered;
}

static
Tensor dynamicSlice(Graph &graph,
                    const Tensor &t,
                    const Tensor &offset,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence *prog,
                    const std::string &debugPrefix)
{
  bool checkOffset = prog != nullptr;
  ValidateParams("dynamicSlice", t, offset, dims, sizes, checkOffset);
  for (unsigned i = 0; i != dims.size(); ++i) {
    if (sizes[i] == 0) {
      // Since one of the slice sizes is zero, the resulting tensor has no
      // elements. We can return a static slice of the original tensor
      // of the correct size. The offset for each slice can be 0 because
      // it won't have any elements anyway. Tensorflow tests for 0-sized slices.
      Tensor emptyT = t;
      for (unsigned d = 0; d < dims.size(); ++d)
        emptyT = emptyT.slice(0, sizes[d], dims[d]);
      return emptyT;
    }
  }
  Tensor out = t;

  auto idxOrder = bestSliceOrder(t.shape(), dims, sizes);

  for (auto i : idxOrder) {
    // don't care about offset if vertices are not mapped as we are only
    // interested in the mapping
    out = slice(graph, out,
                checkOffset ? offset[i] : offset,
                dims[i],
                sizes[i],
                prog,
                debugPrefix + "/dynamicSlice_d" + std::to_string(dims[i]));
  }

  return out;
}

Tensor dynamicSlice(Graph &graph,
                    const Tensor &t,
                    const Tensor &offset,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix) {
  return
    dynamicSlice(graph, t, offset, dims, sizes, &prog, debugPrefix);
}

Graph::TileToTensorMapping
getSliceMapping(poplar::Graph &graph,
                const poplar::Tensor &t,
                const std::vector<std::size_t> &dims,
                const std::vector<std::size_t> &sizes) {
  // give a dummy offset tensor as it is not used
  Tensor offset;
  auto sliceT =
    dynamicSlice(graph, t, offset, dims, sizes, nullptr, "");
  return graph.getTileMapping(sliceT);
}

void dynamicUpdate(Graph &graph,
                   const Tensor &t,
                   const Tensor &s,
                   const Tensor &offset,
                   const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix)
{
  ValidateParams("dynamicUpdate", t, offset, dims, sizes);
  // empty sizes or dimensions are full update (TF does this)
  if (dims.size() == 0) {
    prog.add(Copy(s, t));
    return;
  }
  // If any of sizes is 0 then this is a nop. Tensorflow tests do this.
  for (auto& sz : sizes)
    if (sz == 0)
      return;

  // We insert into a single dimension at a time. When more than one dimension
  // is to be inserted this entails slicing off the outer dimensions until there
  // is a single dynamic dimension. That Tensor is updated with s. Then
  // the dimension traversal is reversed, updating one into one extra dimension
  // each time.

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
                                &prog,
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

} // end namespace popops
