#include "popops/DynamicSlice.hpp"

#include "poplibs_support/gcd.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/Util.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/Interval.hpp"
#include "poplar/Program.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/exceptions.hpp"
#include "popops/ElementWise.hpp"
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
  constexpr unsigned slicedDim = 0;
  constexpr unsigned unslicedDim = 1;
  assert(t2d.rank() == 2);
  assert(s2d.rank() == 2);
  assert(t2d.dim(unslicedDim) == s2d.dim(unslicedDim));
  const auto &target = graph.getTarget();
  const auto grainSize = target.getVectorWidth(t2d.elementType());
  const auto numTiles = target.getNumTiles();
  const unsigned numBaseElements = t2d.dim(slicedDim);
  const unsigned numSubElements = s2d.dim(slicedDim);
  assert(numSubElements <= numBaseElements);

  // Offset must be a scalar. It will be replicated over tiles
  // by the small graph  replication optimisation during lowering.
  assert(offset.rank() == 0 && offset.numElements() == 1);
  // Build vertices assuming all sliced dimensions have the same mapping as
  // the first one.
  auto mapping = graph.getTileMapping(t2d[0]);
  auto numVarRegions = t2d[0].getVarRegions().size();
  unsigned numUsedTiles = 0;
  for (const auto &e : mapping) {
    if (e.size() != 0)
      ++numUsedTiles;
  }
  // If there are multiple regions on a tile try reordering to simplify vertex
  // state. Reordering can be expensive when there are many elements so don't
  // reorder if it is unnecessary
  if (numVarRegions > numUsedTiles)
  {
    // Reorder to minimize the number of contiguous regions.
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
    unsigned elemSize = s2dElems[0].numElements();
    s2d = concat(s2dElems).reshape({numSubElements, elemSize});
    t2d = concat(t2dElems).reshape({numBaseElements, elemSize});
    mapping = graph.getTileMapping(t2d[0]);
  }

  // instantiate vertices following the mapping of t's first slice
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

static void generateMultiSliceVertices(
    const std::string &vertexNameUntemplated,
    bool isUpdate,
    Graph &graph,
    ComputeSet &cs,
    const Tensor &offsets,
    const Tensor &base,
    const Tensor &slices,
    std::string &debugName) {
  constexpr unsigned slicedDim = 0; // in base
  constexpr unsigned unslicedDim = 1; // in slices
  assert(offsets.rank() == 2);
  assert(base.rank() == 2);
  assert(slices.rank() == base.rank() + 1);
  assert(base.dim(unslicedDim) == slices.dim(1 + unslicedDim));
  assert(offsets.dim(0) == slices.dim(0));
  // only single-dim slicing supported by these vertices
  assert(offsets.dim(1) == 1);
  auto offsets1d = offsets.squeeze({1});
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto type = base.elementType();
  const unsigned vectorWidth = target.getDataPathWidth() /
                               ((type == HALF) ? 16 : 32);
  const unsigned numBaseElements = base.dim(slicedDim);
#ifndef DEBUG
  const unsigned numSubElements = slices.dim(1 + slicedDim);
  assert(numSubElements == 1);
#endif
  auto vertexName = templateVertex(vertexNameUntemplated, base.elementType());

  // Build vertices assuming all sliced dimensions have the same mapping as
  // the first one and the non-sliced dimension is contiguous. If this is
  // not honoured gathering internal exchange/copies will be generated
  auto mapping = graph.getTileMapping(base[0]);

  // instantiate vertices following the mapping of t's first slice
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
      graph.getSortedContiguousRegions(base[0], mapping[tile]);
    if (tileContiguousRegions.size() == 0)
      // do nothing on this tile
      continue;
    // separate vertices for each
    unsigned regionSize = 0;
    std::vector<Tensor> baseSlices, subSlices;
    for (const auto &tcr : tileContiguousRegions) {
      for (const auto &region : tcr) {
        regionSize += region.size();
        baseSlices.emplace_back(base.transpose().slice(region));
        subSlices.emplace_back(slices.dimRoll(2, 1).slice(region, 1));
      }
    }
    // When tcr.size() == 1 and the tensors are correctly layed out no gather
    // will be required for these edges
    // If multiple elements of the slice are on the same tile numBaseElements
    // and regionSize will differ
    Tensor tileBase = concat(baseSlices).transpose().flatten();
    Tensor tileSub = concat(subSlices).dimRoll(2, 1).flatten();

    auto numParallelWorkers = isUpdate ? 1 : target.getNumWorkerContexts();

    auto copiesPerOffset = (regionSize + vectorWidth - 1) / vectorWidth;
    // min 4 copies per thread to avoid excessive vertex state
    auto offsetsPerThread =
        std::max((offsets1d.numElements() + numParallelWorkers - 1
                 ) / numParallelWorkers,
        4ul / copiesPerOffset);
    offsetsPerThread = std::min(offsetsPerThread,
                                graph.getMaxFieldDim(vertexName, "offsets", 0));
    for (unsigned o = 0; o != offsets1d.numElements();) {
      auto firstOffset = o;
      o = std::min(o + offsetsPerThread, offsets1d.numElements());
      Tensor workerOffsets = offsets1d.slice({firstOffset, o});
      Tensor workerSlices = tileSub.slice({firstOffset, o});
      auto v = graph.addVertex(cs,
                               vertexName,
                               {{"offsets", workerOffsets},
                                {"baseT", tileBase.flatten()},
                                {"subT", workerSlices.flatten()}
                               });
      graph.setInitialValue(v["numBaseElements"], numBaseElements);
      graph.setInitialValue(v["regionSize"], regionSize);
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
                           const std::vector<std::size_t> &shape,
                           const Tensor &offset,
                           const std::vector<std::size_t> &dims,
                           const std::vector<std::size_t> &sizes,
                           bool checkOffset = true,
                           bool checkSizes = true
                           ) {
  auto tRank = shape.size();
  std::string exceptionStr;
  if (checkOffset) {
    auto offsetElems = offset.rank() == 0 ? 0 : offset.dim(0);
   if  (offset.rank() > 2 || offsetElems != dims.size())
     exceptionStr = name + " offset (" + std::to_string(offsetElems) + ") ";
  }
  if (checkSizes && dims.size() != sizes.size()) {
    exceptionStr +=  "dims (" + std::to_string(dims.size()) +
                      ") and sizes " + std::to_string(sizes.size()) + ") ";
  }
  if (!exceptionStr.empty()) {
    exceptionStr +=  ": must be the same size";
    throw graph_connection_error(exceptionStr);
  }
  std::vector<bool> dimUsed(tRank);
  for (unsigned i = 0; i != dims.size(); ++i) {
    if (dims[i] >= tRank)
      throw graph_connection_error(
        name + ": invalid dimension " + std::to_string(dims[i]));
    if (checkSizes && sizes[i] > shape[dims[i]])
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
static Tensor
createSliceableTensorGivenOrder(poplar::Graph &graph,
                                const poplar::Type &type,
                                const std::vector<std::size_t> &shape,
                                const std::vector<std::size_t> &dims,
                                const std::vector<std::size_t> &idxOrder,
                                std::size_t minGrainSize,
                                const std::string &debugPrefix)
{
  ValidateParams("createSliceableTensor", shape, {}, dims, {}, false, false);
  const auto numTiles = graph.getTarget().getNumTiles();
  const unsigned numDims = shape.size();
  const unsigned numSlicedDims = dims.size();
  bool noOutputElements = *std::min_element(shape.cbegin(), shape.cend()) == 0;
  if (numSlicedDims == 0 || noOutputElements) {
    // no slicing specified
    auto t = graph.addVariable(type, shape);
    mapTensorLinearly(graph, t);
    return t;
  }

  std::vector<size_t> internalShape;
  // vector recording the permutation from [internal] to external dimension
  std::vector<unsigned> externalPermutation(shape.size());
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
      externalPermutation[d] = unslicedExternalDims.size();
      unslicedExternalDims.emplace_back(shape[d]);
    }
  }
  internalShape.emplace_back(nonSliceNumElements);// single outer dim

  // The number of elements that would be returned by each tile if we're
  // balanced across all tiles
  size_t balancedOutPerTile = (nonSliceNumElements + numTiles - 1) / numTiles;

  // Order the sliced indices to optimise multi-dimensional slicing.
  for (auto i = 0u; i != dims.size(); ++i) {
    auto d = dims.at(idxOrder[i]);
    if (slicedDim[d]) {
      externalPermutation[d] = unslicedExternalDims.size() +
                               internalShape.size() - 1;
      internalShape.emplace_back(shape[d]);
    }
  }
  // add an innermost dimension holding the number of consecutive elements
  // output by each tile
  auto numOutPerTile = std::max(balancedOutPerTile, minGrainSize);
  numOutPerTile = gcd(internalShape.front(), numOutPerTile);
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
  reordered = reordered.dimShuffle(externalPermutation);
  return reordered;
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
Tensor
createSliceableTensor(poplar::Graph &graph,
                      const poplar::Type &type,
                      const std::vector<std::size_t> &shape,
                      const std::vector<std::size_t> &dims,
                      const std::vector<std::size_t> &sizes,
                      std::size_t minGrainSize,
                      const std::string &debugPrefix)
{
  auto idxOrder = bestSliceOrder(shape, dims, sizes);
  return createSliceableTensorGivenOrder(graph, type, shape, dims, idxOrder,
                                         minGrainSize, debugPrefix);
}

Tensor
createUpdateTensor(Graph &graph,
                   const Tensor &t,
                   const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   const std::size_t numUpdates,
                   const std::string &debugPrefix) {
  ValidateParams("createUpdateTensor", t.shape(), {}, dims, sizes, false);

  Tensor s;
  std::string name = debugPrefix + "/update";
  if (numUpdates == 1) {
    s = t;
    // When updating a single slice map the update tensor with the mapping
    // of the first slice of the base tensor
    for (unsigned i = 0; i != dims.size(); ++i) {
      s = s.slice(0, sizes[i], dims[i]);
      name = name + "_d" + std::to_string(dims[i]);
    }
    auto mapping = graph.getTileMapping(s);
    s = graph.clone(s, name);
    graph.setTileMapping(s, mapping);
    return s.expand({0});
  }

  // The update tensor has an an outer dimension of the number of slices to be
  // updated, with the remaining dimensions taken from t reduced to the sliced
  // size
  auto uShape = t.shape();
  uShape.insert(uShape.begin(), numUpdates);
  // uDims holds dims shifted due to the new outer numUpdates dimension
  auto uDims = dims;
  for (unsigned i = 0; i != dims.size(); ++i)
    ++uDims[i];
  // update/slicing order is based on the tensor shape before any update is
  // performed. full-sized dimensions do not affect the order.
  auto idxOrder = bestSliceOrder(uShape, dims, sizes);

  // shrink the dimensions to the size of the update
  for (unsigned i = 0; i != dims.size(); ++i) {
    uShape[uDims[i]] = sizes[i];
  }
  s = createSliceableTensorGivenOrder(graph, t.elementType(), uShape, dims,
                                      idxOrder, 0, debugPrefix);
  return s;
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
  ValidateParams("dynamicSlice", t.shape(), offset, dims, sizes, checkOffset);
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
  ValidateParams("dynamicUpdate", t.shape(), offset, dims, sizes);
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

// create a sequence that runs \a loopProgram the number of times stored in
// \a i. \a i is incremented after each call
static poplar::program::Sequence
countedLoop(poplar::Graph &graph,
            unsigned count,
            poplar::Tensor &i,
            poplar::program::Program &loopProgram,
            const std::string &debugPrefix) {
  poplar::program::Sequence result;
  auto one =
      graph.addConstant(poplar::UNSIGNED_INT, {}, 1, debugPrefix + "/const_1");
  graph.setTileMapping(one, 0);

  poplar::program::Sequence loopProgramInc;
  loopProgramInc.add(loopProgram);
  addInPlace(graph, i.reshape({}), one, loopProgramInc,
             debugPrefix + "/i/increment");

  result.add(poplar::program::Repeat(count, loopProgramInc));

  return result;
}

Tensor multiSlice(Graph &graph,
                  const Tensor &t,
                  const Tensor &offset,
                  const std::vector<std::size_t> &dims,
                  const std::vector<std::size_t> &sizes,
                  Sequence &prog,
                  const std::string &debugPrefix) {
  // small number of slices are instantiated individually
  // large number of slices are sliced by a specialisation or in a loop
  std::string dName = debugPrefix + "/multiSlice";
  // Check the offsets have been specified with a multi-slice dimension
  if (offset.rank() != 2)
    throw poputil::poplibs_error(
        "multiSlice expects offset.rank() == 2 but it is" +
        std::to_string(offset.rank()));
  if (offset.dim(1) != dims.size())
    throw poputil::poplibs_error(
        "multiSlice expects offset.dim(1) == dims.size(); offset.dim(1)==" +
        std::to_string(offset.dim(1)) + ", dims.size()== " +
        std::to_string(dims.size()));
  ValidateParams("multiSlice", t.shape(), offset[0], dims, sizes);
  // We always map the output in the same way to avoid surprising changes when
  // the number of slices changes
  auto sMulti = createUpdateTensor(graph, t, dims, sizes, offset.dim(0),
                                   dName);
  // When there are only a few slices the looping code can be larger than
  // instantiating multiple vertices
  constexpr unsigned inliningThreshold = 3;
  if (offset.dim(0) <= inliningThreshold) {
    for (unsigned slice = 0; slice != offset.dim(0); ++slice) {
      auto s = dynamicSlice(graph, t, offset[slice], dims, sizes, prog,
                            dName + "/" + std::to_string(slice));
      prog.add(Copy(s, sMulti[slice]));
    }
    return sMulti;
  }
  // When there are many offsets of single slices there is a fast vertex.
  // For now only 2d based tensors are supported.
  if (t.rank() == 2 && dims.size() == 1 &&
      offset.rank() == 2 && offset.dim(1) == 1 && offset.dim(0) > 6) {
    auto cs = graph.addComputeSet(dName);
    generateMultiSliceVertices("popops::MultiSlice", false, graph, cs, offset,
                               t, sMulti, dName) ;
    prog.add(Execute(cs));
    return sMulti;
  }

  // looping case
  Sequence body;
  auto sIdx = graph.addVariable(UNSIGNED_INT, {1}, dName + "/sIdx");
  auto zero = graph.addConstant(UNSIGNED_INT, {1}, 0, dName + "/zero");
  graph.setTileMapping(sIdx, 0);
  graph.setTileMapping(zero, 0);
  prog.add(Copy(zero, sIdx));
  auto tIdx = dynamicSlice(graph, offset, sIdx, {0}, {1},
                           body, dName + "/sliceIndex").squeeze({0});

  auto sI = dynamicSlice(graph, t, tIdx, dims, sizes, body,
                         dName + "/slice").expand({0});
  dynamicUpdate(graph, sMulti, sI, sIdx, {0}, {1}, body, dName + "/update");
  prog.add(countedLoop(graph, offset.dim(0), sIdx, body, dName + "/loop"));
  return sMulti;
}

// This is derived from multiSlice with \a s input rather than generated,
// the tensors swapped, etc
void multiUpdate(Graph &graph,
                  const Tensor &t,
                  const Tensor &sMulti,
                  const Tensor &offset,
                  const std::vector<std::size_t> &dims,
                  const std::vector<std::size_t> &sizes,
                  Sequence &prog,
                  const std::string &debugPrefix) {
  // small number of slices are updated individually
  // large number of slices are updated by a specialisation or in a loop
  std::string dName = debugPrefix + "/multiSlice";
  // Check the offsets have been specified with a multi-slice dimension
  if (offset.rank() != 2)
    throw poputil::poplibs_error(
        "multiUpdate expects offset.rank() == 2 but it is" +
        std::to_string(offset.rank()));
  if (offset.dim(1) != dims.size())
    throw poputil::poplibs_error(
        "multiUpdate expects offset.dim(1) == dims.size(); offset.dim(1)==" +
        std::to_string(offset.dim(1)) + ", dims.size()== " +
        std::to_string(dims.size()));
  ValidateParams("multiUpdate", t.shape(), offset[0], dims, sizes);
  // When there are only a few slices the looping code can be larger than
  // instantiating multiple vertices
  constexpr unsigned inliningThreshold = 3;
  if (offset.dim(0) <= inliningThreshold) {
    for (unsigned slice = 0; slice != offset.dim(0); ++slice) {
      dynamicUpdate(graph, t, sMulti[slice], offset[slice], dims, sizes, prog,
                    dName + "/" + std::to_string(slice));
    }
    return;
  }
  // When there are many offsets of single slices there is a fast vertex.
  // For now only 2d based tensors are supported.
  if (t.rank() == 2 && dims.size() == 1 &&
      offset.rank() == 2 && offset.dim(1) == 1 && offset.dim(0) > 6) {
    auto cs = graph.addComputeSet(dName);
    generateMultiSliceVertices("popops::MultiUpdate", true, graph, cs, offset,
                               t, sMulti, dName) ;
    prog.add(Execute(cs));
    return;
  }
  // looping case
  Sequence body;
  auto sIdx = graph.addVariable(UNSIGNED_INT, {1}, dName + "/sIdx");
  auto zero = graph.addConstant(UNSIGNED_INT, {1}, 0, dName + "/zero");
  graph.setTileMapping(sIdx, 0);
  graph.setTileMapping(zero, 0);
  prog.add(Copy(zero, sIdx));
  auto tIdx = dynamicSlice(graph, offset, sIdx, {0}, {1},
                           body, dName + "/sliceIndex").squeeze({0});

  auto sI = dynamicSlice(graph, sMulti, sIdx, dims, sizes, body,
                         dName + "/slice").squeeze({0});
  dynamicUpdate(graph, t, sI, tIdx, {0}, {1}, body, dName + "/update");
  prog.add(countedLoop(graph, offset.dim(0), sIdx, body, dName + "/loop"));
}

} // end namespace popops
