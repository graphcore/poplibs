// Copyright (c) Graphcore Ltd, All rights reserved.
#include "popops/DynamicSlice.hpp"
#include "DynamicSliceInternal.hpp"
#include "poplar/Interval.hpp"
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/Algorithms.hpp"
#include "poplibs_support/ContiguousRegionsByTile.hpp"
#include "poplibs_support/PlanConstraints.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Encoding.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "popops/Zero.hpp"
#include "popsolver/Model.hpp"
#include "poputil/TileMapping.hpp"
#include "poputil/Util.hpp"
#include "poputil/VarStructure.hpp"
#include "poputil/VertexTemplates.hpp"
#include "poputil/exceptions.hpp"

#include <boost/optional.hpp>

#include <algorithm>
#include <boost/range/adaptor/reversed.hpp>
#include <cassert>
#include <numeric>
#include <type_traits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;
using namespace poplibs;

namespace popops {

namespace {

constexpr std::size_t minIndicesPerTile = 32;

struct SliceOptions {
  SliceOptions() = default;

  PlanConstraints planConstraints;
  // Specify whether a plan is to be used for an update.
  bool usedForUpdate = true;
  // TODO: T12930 Add option to specify whether a plan is to be used for a
  // lookup.

  // The target maximum temporary memory usage for the operation. This
  // may not be satisfiable.
  double availableMemoryProportion = 0.6;
};

struct ValidateSlicePlanConstraintsOption {
  void operator()(const boost::property_tree::ptree &t) const {
    if (t.empty() && !t.data().empty()) {
      throw poplar::invalid_option("Plan constraints must be an object");
    }

    for (const auto &child : t) {
      if (child.first != "lookupSplit" && child.first != "slicedDimSplit" &&
          child.first != "unslicedDimSplit" &&
          child.first != "unslicedGrainSize") {
        throw poplibs_error("Unrecognised constraint " + child.first);
      }

      validatePlanConstraintsUnsigned(child.first, child.second);
    }
  }
};

} // unnamed namespace

std::ostream &operator<<(std::ostream &o, const SlicePlanInternal &p) {
  o << "SlicePlan:\n";
  o << "  Partition:\n";
  o << "    lookupSplit=" << p.partition.lookupSplit << "\n";
  o << "    slicedDimSplit=" << p.partition.slicedDimSplit << "\n";
  o << "    unslicedDimSplit=" << p.partition.unslicedDimSplit << "\n";
  o << "    unslicedGrainSize=" << p.partition.unslicedGrainSize << "\n";
  return o;
}

SlicePlan::SlicePlan() : internal(std::make_unique<SlicePlanInternal>()) {}
SlicePlan::~SlicePlan() = default;
SlicePlan::SlicePlan(const SlicePlan &other) {
  internal = other.internal->clone();
}
SlicePlan::SlicePlan(SlicePlan &&other) = default;
SlicePlan &SlicePlan::operator=(const SlicePlan &other) {
  internal = other.internal->clone();
  return *this;
}
SlicePlan &SlicePlan::operator=(SlicePlan &&other) = default;

SlicePlan::SlicePlan(std::unique_ptr<SlicePlanInternal> internal)
    : internal(std::move(internal)) {}

std::ostream &operator<<(std::ostream &o, const SlicePlan &p) {
  if (!p.internal->isNull) {
    o << *p.internal;
  } else {
    o << "SlicePlan: Introspect\n";
  }
  return o;
}

static SliceOptions parseSliceOptions(const OptionFlags &optionFlags) {
  SliceOptions options;

  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  using poplibs_support::makePlanConstraintsOptionHandler;

  const auto makeSlicePlanConstraintsOptionHandler =
      &makePlanConstraintsOptionHandler<ValidateSlicePlanConstraintsOption>;

  /*
   * Any changes to spec must be reflected in the documentation comment in
   * the header.
   */
  const OptionSpec spec{
      {"planConstraints",
       makeSlicePlanConstraintsOptionHandler(options.planConstraints)},
      {"usedForUpdate", OptionHandler::createWithBool(options.usedForUpdate)},
      {"availableMemoryProportion",
       OptionHandler::createWithDouble(options.availableMemoryProportion)}};

  for (const auto &entry : optionFlags) {
    spec.parse(entry.first, entry.second);
  }

  return options;
}

// This is specifically for embedding layer shaped operations currently.
// Given an index into a set of indices into partitions of different
// dimensions of the operation, return the tile on which this portion
// of the operation will be calculated.
static unsigned linearizeSliceIndices(const std::size_t slicedPartition,
                                      const std::size_t unslicedPartition,
                                      const std::size_t indexIdx,
                                      const std::size_t slicedIdx,
                                      const std::size_t unslicedIdx) {
  // indices
  unsigned tile = indexIdx;

  // sliced dimensions
  tile = tile * slicedPartition + slicedIdx;

  // unsliced dimensions
  tile = tile * unslicedPartition + unslicedIdx;

  return tile;
}

/** Create vertices with matching elements in t2d and s2d
 * \param vName     The base name of vertices to create
 * \param graph     The graph to update
 * \param cs        The compute set to update
 * \param offset    The offset within t2d corresponding to the first element in
 *                  s2d. A single element for all tiles, or one element per tile
 * \param t2d       A 2d base tensor
 * \param s2d       A 2d sub tensor
 **/
static void generateVertices(std::string vertexName, Graph &graph,
                             Sequence &prog, const Tensor &offset,
                             Tensor t2d, // 2d base Tensor [sliceD][]
                             Tensor s2d, // 2d sub Tensor [sizeD][]
                             const std::string &debugName) {
  auto cs = graph.addComputeSet(debugName);

  constexpr unsigned slicedDim = 0;
#ifndef NDEBUG
  constexpr unsigned unslicedDim = 1;
#endif
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
  if (numVarRegions > numUsedTiles) {
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
        auto v = graph.addVertex(
            cs, templateVertex(vertexName + "1d", t2d.elementType()),
            {{"offset", offset}, {"baseT", tileBase}, {"subT", tileSub}});

        // the assembly relies on underflow of baseIdx with numBaseElements,
        // therefore the maximum value each can be is 2^31 - 1. we can't check
        // baseIdx at compile time but we can the size of numBaseElements at
        // the very least. both are checked at runtime in the C++ codelet.
        assert(numBaseElements < (1u << 31u));
        graph.setInitialValue(v["numBaseElements"], numBaseElements);
        graph.setInitialValue(v["numSubElements"], numSubElements);
        graph.setInitialValue(v["regionSize"], regionSize);
        graph.setTileMapping(v, tile);
        continue;
      }
    }

    auto vertexSeqs = splitRegionsBetweenWorkers(target, tileContiguousRegions,
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
      auto v = graph.addVertex(
          cs, templateVertex(vertexName + "2d", t2d.elementType()),
          {{"offset", offset}, {"baseT", base}, {"subT", sub}});
      graph.setInitialValue(v["numBaseElements"], numBaseElements);
      graph.setInitialValue(v["numSubElements"], numSubElements);
      graph.setInitialValue(v["numRegions"], base.size() / numBaseElements);
      graph.setTileMapping(v, tile);
    }
  } // end loop over tiles

  prog.add(Execute(cs));
}

// Generate vertices on a specified tile to perform a multi-slice
// where indices are potentially split between workers depending on the
// operation.
static void generateMultiSliceVerticesOnTile(
    Graph &graph, const ComputeSet &cs, unsigned tile, const Tensor &base,
    const Tensor &offset, const Tensor &slices, const Tensor *scale,
    const std::string &vertexName, bool isUpdate, unsigned baseSlicedDim,
    boost::optional<unsigned> baseOffset, const std::string &debugPrefix) {
  assert(base.rank() == 2);
  assert(offset.rank() == 1);
  assert(slices.rank() == base.rank() + 1);
  assert(offset.dim(0) == slices.dim(0));
  assert(baseSlicedDim < base.rank());
  // Only support slicing single elements from the sliced dimension currently.
  assert(slices.dim(1 + baseSlicedDim) == 1);

  const auto dType = base.elementType();
  const auto &target = graph.getTarget();
  const auto atomsPerWord =
      target.getAtomicStoreGranularity() / target.getTypeSize(dType);
  const unsigned vectorWidth = target.getVectorWidth(dType);
  const auto numParallelWorkers = isUpdate ? 1 : target.getNumWorkerContexts();
  const auto regionSize = base.dim(baseSlicedDim ^ 1);
  auto copiesPerOffset =
      (base.dim(baseSlicedDim) + vectorWidth - 1) / vectorWidth;

  // min 4 copies per thread to avoid excessive vertex state
  auto offsetsPerThread = std::max(
      (offset.numElements() + numParallelWorkers - 1) / numParallelWorkers,
      4ul / copiesPerOffset);

  // ensure that words are not split between workers
  // (the Cpu target may have zero atomsPerWord)
  if (atomsPerWord) {
    if (auto numSubwordElements = offsetsPerThread % atomsPerWord) {
      offsetsPerThread += atomsPerWord - numSubwordElements;
    }
  }

  // TODO: T12931 Consider splitting the sliced dimension between the workers.
  // All workers would have to check every offset but time to copy/update
  // entries would be distributed; this would not be effective if many
  // offsets were in the same split of the sliced dimension.
  offsetsPerThread = std::min(offsetsPerThread,
                              graph.getMaxFieldDim(vertexName, "offsets", 0));
  for (unsigned o = 0; o != offset.numElements();) {
    auto firstOffset = o;
    o = std::min(o + offsetsPerThread, offset.numElements());
    Tensor workerOffsets = offset.slice({firstOffset, o});
    Tensor workerSlices = slices.slice({firstOffset, o});
    auto v = graph.addVertex(cs, vertexName,
                             {{"offsets", workerOffsets},
                              {"baseT", base.flatten()},
                              {"subT", workerSlices.flatten()}});
    if (scale != nullptr) {
      graph.connect(v["scale"], *scale);
    }

    graph.setInitialValue(v["baseOffset"], baseOffset ? *baseOffset : 0u);
    graph.setInitialValue(v["numBaseElements"], base.dim(baseSlicedDim));
    graph.setInitialValue(v["regionSize"], regionSize);
    graph.setTileMapping(v, tile);
  }
}

static void generateMultiSliceVertices(
    const std::string &vertexNameUntemplated, bool isUpdate, bool isUpdateAdd,
    Graph &graph, Sequence &prog, const Tensor &offsets, Tensor base,
    Tensor slices, const Tensor *scale, unsigned baseSlicedDim,
    boost::optional<unsigned> baseOffset, const OptionFlags &optionFlags,
    const std::string &debugName) {

  const auto options = parseSliceOptions(optionFlags);

  auto cs = graph.addComputeSet(debugName);

  // un-/slicedDim are in base, must add one in slices
  constexpr unsigned slicedDim = 0;
#ifndef NDEBUG
  constexpr unsigned unslicedDim = 1;
#endif
  assert(offsets.rank() == 2);
  assert(base.rank() == 2);
  assert(slices.rank() == base.rank() + 1);
  assert(offsets.dim(0) == slices.dim(0));
  // only single-dim slicing supported by these vertices
  assert(offsets.dim(1) == 1);
  if (baseSlicedDim != slicedDim) {
    // This function is coded to slice the innermost dimension. If the outermost
    // is being sliced swap the tensor dimesions.
    base = base.transpose();
    slices = slices.dimRoll(2, 1);
  }
  assert(base.dim(unslicedDim) == slices.dim(1 + unslicedDim));
  assert(isUpdate || scale == nullptr); // no scale on slice

  auto offsets1d = offsets.squeeze({1});
  const auto &target = graph.getTarget();
  const auto numTiles = target.getNumTiles();
  const auto type = base.elementType();

  // Build vertices assuming all sliced dimensions have the same mapping as
  // the first one and the non-sliced dimension is contiguous. If this is
  // not honoured gathering internal exchange/copies will be generated
  auto baseSlice0 = base.slice(0, 1, slicedDim);
  auto mappingSlice0 = graph.getTileMapping(baseSlice0);

  // Check the spread of the base tensor over tiles against an available
  // memory proportion to determine if we will use excessive temporary memory
  // for the base tensor and result or not.
  //
  boost::optional<Tensor> originalBase;
  {
    const auto maxUnslicedElemsPerTile = [&] {
      std::size_t maxElems = 0;
      for (unsigned tile = 0; tile < mappingSlice0.size(); ++tile) {
        const auto elemsThisTile = std::accumulate(
            mappingSlice0[tile].begin(), mappingSlice0[tile].end(),
            std::size_t(0),
            [](std::size_t t, const Interval &i) { return t + i.size(); });
        maxElems = std::max(maxElems, elemsThisTile);
      }
      return maxElems;
    }();

    const auto balancedUnslicedElemsPerTile = ceildiv(base.dim(1), numTiles);

    // If we are already as well balanced as we can be then we can't do
    // anything about this without a planned multi-stage or even a serialized
    // slice which we won't try for the timebeing.
    if (maxUnslicedElemsPerTile > balancedUnslicedElemsPerTile) {
      const auto bytesPerElem = target.getTypeSize(type);
      const auto maxBaseBytesPerTile =
          maxUnslicedElemsPerTile * base.dim(slicedDim) * bytesPerElem;
      const unsigned availableBytesPerTile = std::ceil(
          target.getBytesPerTile() * options.availableMemoryProportion);

      // We first check if having to rearrange the base slice would cause us to
      // exceed our temporary memory limit to avoid introspecting again if we
      // don't need to.
      if (maxBaseBytesPerTile > availableBytesPerTile) {
        // Do a cheap but imprecise approximation of whether or not all the
        // slices of the base tensor have the same tile mapping as the first by
        // checking just one other slice, chosen to be the last heuristically.
        //
        // If the mapping of all slices does not match we will have to
        // rearrange and hence we will know that our temporary memory budget
        // will be exceeded by rearranging this base tensor (we already checked
        // the max size on a tile above).
        auto n = base.dim(slicedDim);
        auto baseSliceN = base.slice(n - 1, n, slicedDim);
        auto mappingSliceN = graph.getTileMapping(baseSliceN);

        if (mappingSlice0 != mappingSliceN) {
          // Rearrange the base tensor to be better spread
          originalBase = base;
          base = createSliceableTensor(graph, type, base.shape(), {slicedDim},
                                       {1}, 0, debugName + "/baseRearranged");
          prog.add(Copy(*originalBase, base));
          baseSlice0 = base.slice(0, 1, slicedDim);
          mappingSlice0 = graph.getTileMapping(baseSlice0);
        }
      }
    }
  }

  // instantiate vertices following the mapping of t's first slice
  std::vector<unsigned> multiUpdateSubwordTiles;
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(baseSlice0, mappingSlice0[tile]);
    if (tileContiguousRegions.size() == 0) {
      // do nothing on this tile
      continue;
    }

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
    // When tcr.size() == 1 and the tensors are correctly laid-out, no gather
    // will be required for these edges.
    // If multiple elements of the slice are on the same tile numBaseElements
    // and regionSize will differ.

    Tensor tileBase = concat(baseSlices, slicedDim).transpose();
    Tensor tileSub = concat(subSlices, 1 + slicedDim).dimRoll(2, 1);

    std::string vertexName;
    if (isUpdateAdd) {
      bool padTo32Bits = false; // TODO: T12932 Control this via a plan field.
      if (!padTo32Bits) {
        // We have different specialisations for half data depending on the need
        // for subword writes.
        //
        // Note gcd is used here for e.g. CPU where the atomic write size is 1.
        const unsigned bytesPerAtom =
            lcm(target.getAtomicStoreGranularity(), target.getTypeSize(type));
        const unsigned elemsPerAtom = bytesPerAtom / target.getTypeSize(type);
        bool needSubwordWrites = regionSize % elemsPerAtom != 0;

        if (needSubwordWrites)
          multiUpdateSubwordTiles.emplace_back(tile);
        vertexName = templateVertex(vertexNameUntemplated, base.elementType(),
                                    needSubwordWrites);
      } else {
        // For halves we process 32-bit at a time and therefore pad the tensors
        // in the case where region size is odd.
        if (target.getTypeSize(type) == 2 && regionSize % 2 != 0) {
          const auto padWithSelf = [&](const StringRef name, const Tensor &t) {
            logging::debug("Padding {} in {} to avoid sub-word writes.", name,
                           debugName);

            // As we want to pad the last dimension, we might as well do that
            // with ourselves. so slice that dimension out, clone it (to avoid
            // aliasing) and then interleave it back with the original.
            const auto lastDim = t.rank() - 1;
            const auto first = t.slice(0, 1, lastDim);
            const auto firstCloned = graph.clone(first, debugName + "/padding");

            // This handles odd grain sizes, which are not expected to be used.
            // TODO: T12998 A WriteUndef may be needed here (see T11457).
            prog.add(Copy(first, firstCloned));
            return concat({t, firstCloned}, lastDim);
          };

          tileBase = padWithSelf("baseT", tileBase);
          tileSub = padWithSelf("subT", tileSub);
          ++regionSize;
          vertexName =
              templateVertex(vertexNameUntemplated, base.elementType(), false);
        }
      }
    } else {
      vertexName = templateVertex(vertexNameUntemplated, base.elementType());
    }

    generateMultiSliceVerticesOnTile(graph, cs, tile, tileBase, offsets1d,
                                     tileSub, scale, vertexName, isUpdate, 0u,
                                     baseOffset, debugName);
  }

  if (!multiUpdateSubwordTiles.empty()) {
    logging::debug("UpdateAdd in {} with odd regionSize on tile(s) {}",
                   debugName, multiUpdateSubwordTiles);
  }

  prog.add(Execute(cs));

  // If this is an update and we rearranged the input, copy back to the original
  if (originalBase && isUpdate) {
    prog.add(Copy(base, *originalBase));
  }
}

static void
generatePlannedMultiUpdateAdd(const std::string &vertexNameUntemplated,
                              const SlicePlanInternal &plan, Graph &graph,
                              Sequence &seq, const Tensor &offsets, Tensor base,
                              Tensor slices, const Tensor scale,
                              unsigned baseSlicedDim, std::string &debugName) {

  // When a two-stage update is perform we use 32bit partials
  const auto twoStagePartialType = FLOAT;

  const auto csU = graph.addComputeSet(debugName + "/Update");

  // record of tiles handling misalignment
  std::vector<unsigned> multiUpdateSubwordTiles;

  // un-/slicedDim are in base, must add one in slices
  constexpr unsigned slicedDim = 0;
  constexpr unsigned unslicedDim = 1;
  assert(offsets.rank() == 2);
  assert(base.rank() == 2);
  assert(slices.rank() == base.rank() + 1);
  assert(offsets.dim(0) == slices.dim(0));
  // only single-dim slicing supported by these vertices
  assert(offsets.dim(1) == 1);
  assert(baseSlicedDim == slicedDim);
  assert(base.dim(unslicedDim) == slices.dim(1 + unslicedDim));

  const auto offsets1d = offsets.squeeze({1});
  const auto &target = graph.getTarget();
  const auto type = base.elementType();

#ifndef NDEBUG
  const unsigned numSubElements = slices.dim(1 + slicedDim);
  assert(numSubElements == 1);
#endif

  // Build vertices assuming that the base tensor is laid out according to the
  // plan. We loop around the three partitions:
  // lookupSplit: the updates are split into groups. Each group is updated
  //   into a zero tensor, the groups are reduce/added together, then the
  //   total of the updates is applied to the base tensor. When there is no
  //   split the inner processing updates the base tensor directly in a single
  //   stage. Note that the reductions and final add are dense operations
  // slicedDimSplits: the base tensor's sliced dimension is split, each
  //   split sees all updates and only adds those whose indices are within its
  //   subrange
  // unslicedDimSplits: the embedding dimension is split, each split
  //   only sees the appropriate elements
  const auto &p = plan.partition;
  const unsigned slicedSplit = p.slicedDimSplit;
  const unsigned unslicedSplit = p.unslicedDimSplit;

  // Updates are divided into `lookupSplit` groups.
  // When the plan is for many index splits and there are few in this instance
  // some of the splits can be empty. In this case we generate no partials or
  // vertices on tiles in those splits.
  const auto subSlicedDim = 0;
  const auto endSubIndex = slices.dim(subSlicedDim);
  const auto subIndicesPerSplit = ceildiv(endSubIndex, p.lookupSplit);
  const auto nonEmptyLookupSplits = ceildiv(endSubIndex, subIndicesPerSplit);
  assert(nonEmptyLookupSplits <= p.lookupSplit);

  const unsigned numUsedTiles =
      slicedSplit * unslicedSplit * nonEmptyLookupSplits;
  bool multipleStages = nonEmptyLookupSplits > 1;

  // Each subSplit holds a subset of the subIndices. When numSubSplits>1 dense
  // updates are made into zeroed partials then reduced into the base.

  const auto unslicedSize = base.dim(unslicedDim);
  const auto endBaseIndex = base.dim(slicedDim);
  const auto baseIndicesPerSplit = ceildiv(endBaseIndex, slicedSplit);
  const auto elemsPerUSplit = ceildiv(unslicedSize, unslicedSplit);

  logging::debug("PlannedMUAdd: activeTiles={}, split {}/{}/{}, shapes {} {}",
                 numUsedTiles, nonEmptyLookupSplits, slicedSplit, unslicedSplit,
                 base.shape(), slices.shape());

  // First stage: update each lookupSplit into a temporary dense buffer. When
  // lookupSplit is 1 (which is typical for large base index sizes) this is the
  // only stage and the update goes directly into the base Tensor.
  Type stage0OutputType;
  Tensor slicesInput, stage0Output;
  // Scaling is applied in the update when there's a single stage, but in a
  // later add when there is an lookupSplit
  Tensor stage0Scale, stage1Scale;
  if (!multipleStages) {
    slicesInput = slices;
    stage0Output = base.expand({0}); // insert lookupSplit dimension
    stage0Scale = scale;
    stage0OutputType = base.elementType();
  } else {
    // Separate accumulation for each lookupSplit into temporary partial buffers
    // with temporary input and accumulation buffers if the base/slice tensors
    // have type half.
    stage0OutputType = twoStagePartialType;
    stage0Scale =
        graph.addConstant(stage0OutputType, {}, 1., debugName + "/one");
    graph.setTileMapping(stage0Scale, 0);
    stage1Scale =
        cast(graph, scale, stage0OutputType, seq, debugName + "/CastScale");

    // lookupSplit copies of the base tensor
    auto wantedShape = base.shape();
    wantedShape.insert(wantedShape.begin(), nonEmptyLookupSplits);

    // TODO: T12933 Consider cast after broadcasting to first stage updateAdd
    // vertices to save time spent exchanging the larger data type. This may be
    // a tradeoff with temporary memory usage in order to keep a broadcasted
    // half and float copy of the slices during the cast.
    slicesInput = slices.elementType() == twoStagePartialType
                      ? slices
                      : popops::cast(graph, slices, twoStagePartialType, seq,
                                     debugName + "/CastSlices");
    stage0Output = createPartitionableTensor(
        graph, twoStagePartialType, wantedShape,
        {nonEmptyLookupSplits, p.slicedDimSplit, p.unslicedDimSplit},
        debugName + "/gathered");

    // stage0Output is zeroed before stage0 executes; the zero program
    // is added after we've added the stage0 vertices and mapped the output
    // but is sequenced before `csU`.
  }

  for (unsigned lookupSplitIdx = 0; lookupSplitIdx != nonEmptyLookupSplits;
       ++lookupSplitIdx) {
    const unsigned beginSubIdx =
        std::min((lookupSplitIdx + 0) * subIndicesPerSplit, endSubIndex);
    const unsigned endSubIdx =
        std::min((lookupSplitIdx + 1) * subIndicesPerSplit, endSubIndex);

    const auto indices = offsets.slice(beginSubIdx, endSubIdx, 0).flatten();

    auto thisBase = stage0Output[lookupSplitIdx];
    for (unsigned s = 0; s != slicedSplit; ++s) {
      // indices in the index dimension
      const unsigned beginBaseIdx =
          std::min((s + 0) * baseIndicesPerSplit, endBaseIndex);
      const unsigned endBaseIdx =
          std::min((s + 1) * baseIndicesPerSplit, endBaseIndex);

      const boost::optional<unsigned> baseOffset(slicedSplit > 1, beginBaseIdx);

      // Update vertex is invoked on all slicedIdx tiles; the vertex decides
      // whether to make an update based on the range of base indices present
      auto numBaseIndices = endBaseIdx - beginBaseIdx;
      if (numBaseIndices == 0) {
        continue;
      }

      for (unsigned u = 0; u != unslicedSplit; ++u) {
        // indices in the embedding dimension;
        const unsigned beginOffset =
            std::min((u + 0) * elemsPerUSplit, unslicedSize);
        const unsigned endOffset =
            std::min((u + 1) * elemsPerUSplit, unslicedSize);
        auto numOffsets = endOffset - beginOffset;
        if (numOffsets == 0) {
          continue;
        }
        const auto tile = linearizeSliceIndices(slicedSplit, unslicedSplit,
                                                lookupSplitIdx, s, u);
        // We have different specialisations for half data depending on the need
        // for subword writes
        //
        // TODO: T12934 Pad if not a multiple of grain size to ensure uniform
        // execution time of update on each tile given an uneven split.
        bool needSubwordWrites =
            target.getTypeSize(type) == 2 && numOffsets % 2 != 0;

        if (needSubwordWrites) {
          multiUpdateSubwordTiles.emplace_back(tile);
        }

        const auto vertexName = templateVertex(
            vertexNameUntemplated, stage0OutputType, needSubwordWrites);

        logging::trace("generatePlannedMultiUpdateAdd: "
                       "Offsets {}/{} ({}); "
                       "BaseIdx {}/{} ({}), "
                       "SubIdx {}/{} ({}) "
                       "for indices {},{},{} "
                       "on tile {}",
                       beginOffset, endOffset, unslicedDim, beginBaseIdx,
                       endBaseIdx, baseSlicedDim, beginSubIdx, endSubIdx,
                       subSlicedDim, lookupSplitIdx, s, u, tile);

        const Tensor tileBase =
            thisBase.slice(beginBaseIdx, endBaseIdx, baseSlicedDim)
                .slice(beginOffset, endOffset, unslicedDim);
        const Tensor tileSlice =
            slicesInput.slice(beginSubIdx, endSubIdx, subSlicedDim)
                .slice(beginOffset, endOffset, 1 + unslicedDim);

        if (p.lookupSplit > 1) {
          // base tensor was distributed across `p.lookupSplit` groups
          // so we must copy our input
          graph.setTileMapping(tileBase, tile);
        } else {
          // Check that this vertex is mapped to the tile where the data lives
          if (graph.getTileMapping(tileBase)[tile].empty() ||
              graph.getTileMapping(tileBase)[tile].begin()->size() !=
                  numBaseIndices * numOffsets) {
            throw poputil::poplibs_error(
                __func__ +
                std::string(": Base tensor mapping wrong for tile ") +
                std::to_string(tile));
          }
        }
        generateMultiSliceVerticesOnTile(
            graph, csU, tile, tileBase, indices, tileSlice, &stage0Scale,
            vertexName, true, baseSlicedDim, baseOffset, debugName);
      }
    }
  }

  if (!multiUpdateSubwordTiles.empty()) {
    logging::debug("UpdateAdd in {} with odd regionSize on tile(s) {}",
                   debugName, multiUpdateSubwordTiles);
  }

  if (multipleStages) {
    // Reduce dense partials
    zero(graph, stage0Output, seq);
    seq.add(Execute(csU));

    const auto cumulativeUpdate =
        graph.clone(twoStagePartialType, base, debugName + "/sumUpdates");
    reduceWithOutput(graph, stage0Output, cumulativeUpdate, {0},
                     {Operation::ADD}, seq, debugName + "/Reduce");

    // Add the sum of the partials to the base tensor
    bool baseCastRequired = base.elementType() != twoStagePartialType;
    const Tensor addDst = [&] {
      if (baseCastRequired) {
        return cast(graph, base, twoStagePartialType, seq, "/castBase");
      } else {
        return base;
      }
    }();
    scaledAddTo(graph, addDst, cumulativeUpdate, stage1Scale, seq,
                debugName + "/Add");

    // cast the final result back into base; when !castBase the addTo was
    // directly into base anyway
    if (baseCastRequired) {
      seq.add(cast(graph, addDst, base, debugName + "/castBack"));
    }
  } else {
    seq.add(Execute(csU));
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
static Tensor slice(Graph &graph, const Tensor &t, const Tensor &offset,
                    unsigned dim, unsigned numOutIndices,
                    poplar::program::Sequence *prog,
                    const std::string &debugPrefix) {
  const unsigned numInIndices = t.dim(dim);
  assert(dim < t.rank());
  assert(numOutIndices <= t.dim(dim));
  // Get a 2d view of the source tensor, with the dim we're slicing at dim0
  // and the other dimensions collapsed into dim1
  Tensor t2d =
      t.dimRoll(dim).reshape({numInIndices, t.numElements() / numInIndices});
  Tensor s = graph.clone(t.slice(0, numOutIndices, dim),
                         debugPrefix + "/sliced_" + std::to_string(dim));

  rebalanceTensor(graph, s);
  if (prog != nullptr) {
    Tensor s2d = s.dimRoll(dim).reshape(
        {numOutIndices, s.numElements() / numOutIndices});

    generateVertices("popops::DynamicSlice", graph, *prog, offset, t2d, s2d,
                     debugPrefix + "/slice");
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
static void update(Graph &graph, const Tensor &t, const Tensor &s,
                   const Tensor &offset, unsigned dim,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix) {
  const unsigned numTElements = t.dim(dim);
  const unsigned numSElements = s.dim(dim);
  assert(t.rank() == s.rank());
  for (unsigned d = 0; d != t.rank(); ++d) {
    if (d != dim)
      assert(s.dim(d) == t.dim(d));
    else
      assert(s.dim(d) <= t.dim(d));
  }
  assert(dim < t.rank());
  assert(numSElements <= numTElements);
  // Get a 2d view of the source tensor, with the dim we're updating at dim0
  // and the other dimensions collapsed into dim1
  Tensor t2d =
      t.dimRoll(dim).reshape({numTElements, t.numElements() / numTElements});
  Tensor s2d =
      s.dimRoll(dim).reshape({numSElements, s.numElements() / numSElements});

  generateVertices("popops::DynamicUpdateSlice", graph, prog, offset, t2d, s2d,
                   debugPrefix + "/update");
}

// If we are slicing up a tensor with the given `shape` in the dimensions
// `dims`, and the slice size in each dimension is `sizes`, then what is
// the best order to do the slices? The returned vector contains
// indexes into `dims` (and `sizes`).
static std::vector<size_t>
bestSliceOrder(const std::vector<std::size_t> &shape,
               const std::vector<std::size_t> &dims,
               const std::vector<std::size_t> &sizes) {

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

  // Sort the most sliceable dimension first. Assumes no integer overflows.
  std::sort(idxOrder.begin(), idxOrder.end(), [&](size_t a, size_t b) {
    return sizes[b] * shape[dims[a]] > sizes[a] * shape[dims[b]];
  });

  return idxOrder;
}

static void validatePlanForGivenParameters(
    const SlicePlanInternal &p, const OptionFlags &options,
    const std::vector<std::size_t> &shape, const std::vector<std::size_t> &dims,
    const std::vector<std::size_t> &sizes, const std::string &callerDebugStr) {
  if (p.slicedDims != dims) {
    std::stringstream ss;
    ss << callerDebugStr
       << ": Dimensions sliced when building the given plan (";
    printContainer(p.slicedDims, ss);
    ss << ") differ from dimensions given for this operation (";
    printContainer(dims, ss);
    ss << ")";
    throw poplibs_error(ss.str());
  }
  if (p.slicedDimSizes != sizes) {
    std::stringstream ss;
    ss << callerDebugStr
       << ": Sizes of slices in each dimension when building the given plan (";
    printContainer(p.slicedDimSizes, ss);
    ss << ") differ from the sizes of slices in each dimension given for this "
          "operation (";
    printContainer(sizes, ss);
    ss << ")";
    throw poplibs_error(ss.str());
  }
  if (p.rank != shape.size()) {
    std::stringstream ss;
    ss << callerDebugStr
       << ": Rank of the shape used when building the given plan (" << p.rank
       << ") differs from the rank of the given tensor (" << shape.size()
       << ")";
    throw poplibs_error(ss.str());
  }
}

static void validateParams(std::string name, const SlicePlan &plan,
                           const OptionFlags &options,
                           const std::vector<std::size_t> &shape,
                           const Tensor &offset,
                           const std::vector<std::size_t> &dims,
                           const std::vector<std::size_t> &sizesOrSlices,
                           bool checkOffset = true, bool checkSizes = true,
                           bool sizesAreSlices = false) {
  if (!plan.getImpl().isNull) {
    validatePlanForGivenParameters(plan.getImpl(), options, shape, dims,
                                   sizesOrSlices, name);
  }
  auto tRank = shape.size();
  std::string exceptionStr;
  std::string sizesStr = sizesAreSlices ? "numSlices" : "sizes";
  if (checkOffset) {
    auto offsetElems = offset.rank() == 0 ? 0 : offset.dim(0);
    if (offset.rank() > 2 || offsetElems != dims.size())
      exceptionStr = name + " offset (" + std::to_string(offsetElems) + ") ";
  }
  if (checkSizes && dims.size() != sizesOrSlices.size()) {
    exceptionStr += "dims (" + std::to_string(dims.size()) + ") and " +
                    sizesStr + " (" + std::to_string(sizesOrSlices.size()) +
                    ") ";
  }
  if (!exceptionStr.empty()) {
    exceptionStr += ": must be the same size";
    throw graph_connection_error(exceptionStr);
  }
  std::vector<bool> dimUsed(tRank);
  for (unsigned i = 0; i != dims.size(); ++i) {
    if (dims[i] >= tRank)
      throw graph_connection_error(name + ": invalid dimension " +
                                   std::to_string(dims[i]));
    if (checkSizes && !sizesAreSlices && sizesOrSlices[i] > shape[dims[i]])
      throw graph_connection_error(
          name + ": requested slice dimension bigger than buffer");
    if (dimUsed[dims[i]])
      throw graph_connection_error(name + ": dimension " +
                                   std::to_string(dims[i]) +
                                   " specified multiple times");
    dimUsed[dims[i]] = true;
  }
}

// Create and map a tensor so that dynamic slicing of it will not require
// exchange.
// The underlying variables will be [U/N][S0]..[Sn][N] where
// N is the number of contiguous unsliced elements per tile
// U is the product of the unsliced dimensions
// This distributes the input/output slice across U/N tiles.
// S0-Sn are the sliced dimensions, sorted to optimise the number of copies
// Typically two variables are used; the second variable for the final
// tile, which may have a different N.
// If U/N << numTiles an outer stage can be added to convert part of an
// S dimension to an extra U dimensions
static Tensor createSliceableTensorGivenOrder(
    poplar::Graph &graph, const poplar::Type &type,
    const std::vector<std::size_t> &shape, const std::vector<std::size_t> &dims,
    const std::vector<std::size_t> &idxOrder, std::size_t minGrainSize,
    const std::string &debugPrefix) {
  bool noOutputElements = std::any_of(shape.begin(), shape.end(),
                                      [](std::size_t n) { return n == 0; });
  if (dims.size() == 0 || noOutputElements) {
    // no slicing specified
    auto t = graph.addVariable(type, shape);
    mapTensorLinearly(graph, t);
    return t;
  }

  std::vector<bool> dimIsSliced(shape.size(), false);
  std::vector<unsigned> inversePermutation(shape.size());
  std::vector<std::size_t> createShape;
  createShape.reserve(dims.size() + 1);
  for (const auto i : idxOrder) {
    const auto d = dims[i];
    if (d >= shape.size()) {
      throw poputil::poplibs_error(
          "createSliceableTensor called to slice dimension " +
          std::to_string(d) + " but the target has rank " +
          std::to_string(shape.size()));
    }
    if (dimIsSliced[d]) {
      throw poputil::poplibs_error(
          "createSliceableTensor called with repeated dims entry");
    }
    dimIsSliced[d] = true;
    inversePermutation[d] = createShape.size();
    createShape.push_back(shape[dims[i]]);
  }
  std::size_t numUnslicedElems = 1;
  std::vector<std::size_t> unslicedShape;
  unslicedShape.reserve(shape.size() - dims.size());
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (!dimIsSliced[d]) {
      inversePermutation[d] = createShape.size() + unslicedShape.size();
      unslicedShape.push_back(shape[d]);
      numUnslicedElems *= shape[d];
    }
  }
  createShape.push_back(numUnslicedElems);

  // Calculate how we should divide the unsliced dimension.
  //
  // T10013 - We don't necessarily have to map this to minimize the
  // number of tiles used - i.e. we could have multiple tiles with
  // fewer than unslicedElemsPerSplit elements mapped to them.
  const auto numTiles = graph.getTarget().getNumTiles();
  const auto unslicedElemsPerSplit =
      std::max(ceildiv(numUnslicedElems, numTiles), minGrainSize);
  const auto unslicedSplit = ceildiv(numUnslicedElems, unslicedElemsPerSplit);
  std::vector<std::size_t> dimSplits(createShape.size(), 1);
  dimSplits.back() = unslicedSplit;

  auto t = createPartitionableTensor(graph, type, createShape, dimSplits,
                                     debugPrefix + "/sliceable");

  // Distribute over tiles starting from 0.
  unsigned tile = 0;
  iterateTensorPartitions(
      t, dimSplits, [&](const std::vector<std::size_t> &, const Tensor &s) {
        graph.setTileMapping(s, tile++);
      });

  t = t.reshapePartial(t.rank() - 1, t.rank(), unslicedShape)
          .dimShuffle(inversePermutation);

  logging::debug("createSliceableTensor {}, minGrainSize {}, dims {}, "
                 "used tiles {}, "
                 "{} tiles with {} elems, "
                 "{} tiles with {} elems",
                 t.shape(), minGrainSize, dims, unslicedSplit,
                 // Tiles with ceildiv(numElems, numSplits) elements
                 numUnslicedElems / unslicedElemsPerSplit,
                 unslicedElemsPerSplit,
                 // Any remainder
                 numUnslicedElems % unslicedElemsPerSplit ? 1 : 0,
                 numUnslicedElems % unslicedElemsPerSplit);
  return t;
}

static Tensor createSliceableTensor(Graph &graph, const Type &type,
                                    const std::vector<std::size_t> &shape,
                                    const std::size_t slicedDim,
                                    const SlicePlanInternal &plan,
                                    const OptionFlags &options,
                                    const std::string &debugName) {
  std::vector<bool> dimIsSliced(shape.size());
  dimIsSliced[slicedDim] = true;

  assert(!shape.empty());
  std::vector<unsigned> unslicedDims;
  std::vector<std::size_t> unslicedShape;
  unslicedDims.reserve(shape.size() - 1);
  unslicedShape.reserve(unslicedDims.size());
  for (unsigned d = 0; d < shape.size(); ++d) {
    if (!dimIsSliced[d]) {
      unslicedDims.push_back(d);
      unslicedShape.push_back(shape[d]);
    }
  }

  // Product of unsliced dimensions
  const auto totalUnslicedElems = std::accumulate(
      unslicedDims.begin(), unslicedDims.end(), std::size_t(1),
      [&](std::size_t total, unsigned d) { return total * shape[d]; });

  std::vector<std::size_t> createShape = {shape[slicedDim], totalUnslicedElems};
  std::vector<std::size_t> createSplits = {plan.partition.slicedDimSplit,
                                           plan.partition.unslicedDimSplit};

  auto t = createPartitionableTensor(graph, type, createShape, createSplits,
                                     debugName);

  // If there is an indices split we will broadcast each
  // contiguous chunk of the tensor between tiles while
  // respecting grain size.
  iterateTensorPartitions(
      t, createSplits,
      [&](const std::vector<std::size_t> &i, const Tensor &tSlice) {
        const auto extraSplit = plan.partition.lookupSplit;
        const auto grainSize = plan.partition.unslicedGrainSize;

        // We flatten all but the grain size from the plan and distribute this
        // between tiles that use these elements.
        const auto flattenedSlice = tSlice.flatten();
        const auto sliceNumElems = flattenedSlice.numElements();

        const auto sliceNumGrains = ceildiv(sliceNumElems, grainSize);
        const auto grainsPerSplit = ceildiv(sliceNumGrains, extraSplit);
        const auto elemsPerSplit = grainsPerSplit * grainSize;

        assert(i.size() == 2);
        const std::size_t slicedIdx = i.front();
        const std::size_t unslicedIdx = i.back();

        for (std::size_t indexIdx = 0; indexIdx < plan.partition.lookupSplit;
             ++indexIdx) {
          unsigned tile = linearizeSliceIndices(
              plan.partition.slicedDimSplit, plan.partition.unslicedDimSplit,
              indexIdx, slicedIdx, unslicedIdx);
          const auto begin = std::min(sliceNumElems, indexIdx * elemsPerSplit);
          const auto end =
              std::min(sliceNumElems, (indexIdx + 1) * elemsPerSplit);
          graph.setTileMapping(flattenedSlice.slice(begin, end), tile);
        }
      });

  std::vector<unsigned> inversePermutation(shape.size());
  inversePermutation[slicedDim] = 0;

  // Unsliced dimensions (starting from dims.size() which is always 1).
  for (std::size_t i = 0; i < unslicedDims.size(); ++i) {
    inversePermutation[unslicedDims[i]] = 1 + i;
  }

  t = t.reshapePartial(1, 2, unslicedShape).dimShuffle(inversePermutation);

  return t;
}

// Create and map a tensor so that dynamic slicing of it will not require
// exchange
// The underlying layout will be [U/N][S0]..[Sn][N] where
// N is the number of contiguous unsliced elements per tile
// U is the product of the unsliced dimensions
// S0-Sn are the sliced dimensions, sorted to optimise the number of copies
// This distributes the input/output slice across U/N tiles.
// If U/N << numTiles an outer stage can be added to convert part of an
// S dimension to an extra U dimensions
Tensor createSliceableTensor(Graph &graph, const Type &type,
                             const std::vector<std::size_t> &shape,
                             const std::vector<std::size_t> &dims,
                             const std::vector<std::size_t> &sizes,
                             std::size_t minGrainSize,
                             const std::string &debugPrefix) {
  logging::info("createSliceableTensor/NoPlan for {} / {} / {}", shape, dims,
                sizes);
  validateParams("createSliceableTensor", {}, {}, shape, {}, dims, sizes, false,
                 true);
  const auto idxOrder = bestSliceOrder(shape, dims, sizes);
  std::string tName = debugPrefix + "/sliceable";
  std::string sep = "";
  for (const auto &d : shape) {
    tName += sep + std::to_string(d);
    sep = "x";
  }
  return createSliceableTensorGivenOrder(graph, type, shape, dims, idxOrder,
                                         minGrainSize, tName);
}

Tensor createSliceableTensor(Graph &graph, const Type &type,
                             const std::vector<std::size_t> &shape,
                             const std::vector<std::size_t> &dims,
                             const std::vector<std::size_t> &sizes,
                             const SlicePlan &plan, const OptionFlags &options,
                             const std::string &debugName) {
  logging::info("createSliceableTensor for {} / {} / {}; nullplan? {}", shape,
                dims, sizes, plan.getImpl().isNull);
  if (plan.getImpl().isNull) {
    return createSliceableTensor(graph, type, shape, dims, sizes, 0, debugName);
  }
  validateParams("createSliceableTensor", {}, {}, shape, {}, dims, sizes, false,
                 true);
  // We don't plan anything which slices more than one dimension for now or
  // more than a single slice.
  assert(dims.size() == 1);
  assert(sizes.size() == 1 && sizes[0] == 1);
  return createSliceableTensor(graph, type, shape, dims[0], plan.getImpl(),
                               options, debugName);
}

static Tensor createSliceTensor(Graph &graph, const poplar::Type &type,
                                const std::vector<std::size_t> &inputShape,
                                const std::vector<std::size_t> &dims,
                                const std::vector<std::size_t> &sizes,
                                const std::size_t numUpdates,
                                const std::string &debugPrefix) {
  auto uShape = inputShape;
  // update/slicing order is based on the tensor shape before any update is
  // performed. full-sized dimensions do not affect the order.
  auto idxOrder = bestSliceOrder(uShape, dims, sizes);

  // shrink the dimensions to the size of the update
  for (unsigned i = 0; i != dims.size(); ++i) {
    uShape[dims[i]] = sizes[i];
  }
  // The update tensor has an an outer dimension of the number of slices to be
  // updated, with the remaining dimensions taken from t reduced to the sliced
  // size
  uShape.insert(uShape.begin(), numUpdates);
  // uDims holds dims shifted due to the new outer numUpdates dimension
  std::vector<std::size_t> uDims(dims.size() + 1);
  std::vector<std::size_t> uIdxOrder(idxOrder.size() + 1);
  uDims[0] = 0;
  for (unsigned i = 0; i != dims.size(); ++i)
    uDims[1 + i] = 1 + dims[i];
  // adjust uIdxOrder for the new outer numUpdates dimension
  for (unsigned i = 0; i != idxOrder.size(); ++i)
    uIdxOrder[i] = 1 + idxOrder[i];
  uIdxOrder[idxOrder.size()] = 0;

  // For an update tensor only the outermost dimensions is "sliceable"
  return createSliceableTensorGivenOrder(
      graph, type, uShape, uDims, uIdxOrder, 0,
      debugPrefix + "/slices" + std::to_string(numUpdates));
}

static Tensor createSliceTensor(Graph &graph, const Type &type,
                                const std::vector<std::size_t> &shape,
                                const std::size_t slicedDim,
                                const std::size_t numIndices,
                                const SlicePlanInternal &plan,
                                const OptionFlags &options,
                                const std::string &debugName) {
  std::vector<bool> dimIsSliced(shape.size());
  dimIsSliced[slicedDim] = true;

  assert(!shape.empty());
  std::vector<unsigned> unslicedDims;
  std::vector<std::size_t> unslicedShape;
  unslicedDims.reserve(shape.size() - 1);
  unslicedShape.reserve(unslicedDims.size());
  for (unsigned d = 0; d < shape.size(); ++d) {
    if (!dimIsSliced[d]) {
      unslicedDims.push_back(d);
      unslicedShape.push_back(shape[d]);
    }
  }

  // Product of unsliced dimensions
  const auto totalUnslicedElems = std::accumulate(
      unslicedDims.begin(), unslicedDims.end(), std::size_t(1),
      [&](std::size_t total, unsigned d) { return total * shape[d]; });

  std::vector<std::size_t> createShape = {numIndices, 1, totalUnslicedElems};
  std::vector<std::size_t> createSplits = {plan.partition.lookupSplit, 1,
                                           plan.partition.unslicedDimSplit};

  auto t = createPartitionableTensor(graph, type, createShape, createSplits,
                                     debugName);

  const auto iElemsPerPartition =
      ceildiv(numIndices, plan.partition.lookupSplit);
  const auto iElemsPerPartitionStage1 =
      ceildiv(iElemsPerPartition, plan.partition.slicedDimSplit);
  const auto iSplitStage1 =
      ceildiv(iElemsPerPartition, iElemsPerPartitionStage1);

  iterateTensorPartitions(
      t, createSplits,
      [&](const std::vector<std::size_t> &i, const Tensor &tSlice) {
        std::size_t indexIdx = i[0];
        const auto unslicedIdx = i.back();
        // If there is a split of the sliced dimension there is
        // also a second split of the indices in the second stage
        // of slicing which affects where the final output ends up
        // so we account for this here.
        for (std::size_t s = 0; s < iSplitStage1; ++s) {
          const auto slicedIdx = s;
          const auto sBegin =
              std::min(tSlice.dim(0), s * iElemsPerPartitionStage1);
          const auto sEnd =
              std::min(tSlice.dim(0), (s + 1) * iElemsPerPartitionStage1);
          unsigned tile = linearizeSliceIndices(
              plan.partition.slicedDimSplit, plan.partition.unslicedDimSplit,
              indexIdx, slicedIdx, unslicedIdx);
          graph.setTileMapping(tSlice.slice(sBegin, sEnd, 0), tile);
        }
      });

  std::vector<unsigned> inversePermutation(shape.size() + 1);
  inversePermutation[0] = 0;

  // Sliced dimensions (starting from 1)
  inversePermutation[1 + slicedDim] = 1;

  // Unsliced dimensions (starting from 1 + dims.size(), which is always 1)
  for (std::size_t i = 0; i < unslicedDims.size(); ++i) {
    inversePermutation[1 + unslicedDims[i]] = 2 + i;
  }
  t = t.reshapePartial(2, 3, unslicedShape).dimShuffle(inversePermutation);
  return t;
}

Tensor createSliceTensor(Graph &graph, const Type &type,
                         const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &dims,
                         const std::vector<std::size_t> &sizes,
                         const std::size_t numIndices, const SlicePlan &plan,
                         const OptionFlags &options,
                         const std::string &debugName) {
  validateParams("createSliceTensor", plan, options, shape, {}, dims, sizes,
                 false);
  const auto &p = plan.getImpl();
  if (p.isNull) {
    return createSliceTensor(graph, type, shape, dims, sizes, numIndices,
                             debugName);
  } else {
    // We don't plan anything which slices more than one dimension for now or
    // more than a single slice.
    assert(dims.size() == 1);
    assert(sizes.size() == 1 && sizes[0] == 1);
    return createSliceTensor(graph, type, shape, dims[0], numIndices, p,
                             options, debugName);
  }
}

Tensor createSliceTensor(Graph &graph, const Tensor &t,
                         const std::vector<std::size_t> &dims,
                         const std::vector<std::size_t> &sizes,
                         const std::size_t numIndices,
                         const std::string &debugPrefix) {
  validateParams("createSliceTensor", {}, {}, t.shape(), {}, dims, sizes,
                 false);
  // Special case for 1 index, we just clone the input tensor's first slice.
  if (numIndices == 1) {
    std::string name = debugPrefix + "/slice";
    Tensor s = t;
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
  return createSliceTensor(graph, t.elementType(), t.shape(), dims, sizes,
                           numIndices, debugPrefix);
}

poplar::Tensor createIndicesTensor(Graph &graph,
                                   const std::vector<std::size_t> &dims,
                                   const std::size_t numIndices,
                                   const SlicePlan & /* plan */,
                                   const OptionFlags & /* options */,
                                   const std::string &debugPrefix) {
  logging::info("createIndicesTensor for {} / {}", numIndices, dims);
  const auto indices =
      graph.addVariable(UNSIGNED_INT, {numIndices, dims.size()}, debugPrefix);
  mapTensorLinearly(graph, indices, minIndicesPerTile, 1);
  return indices;
}

template <typename T>
std::vector<std::vector<T>> flattenInnermostRegions(
    const std::vector<std::vector<std::vector<T>>> &regions) {

  std::vector<std::vector<T>> result(regions.size());
  for (std::size_t i = 0; i < regions.size(); ++i) {
    result[i] = regions[i][0];
    for (std::size_t j = 1; j < regions[i].size(); ++j) {
      std::copy(regions[i][j].begin(), regions[i][j].end(),
                std::back_inserter(result[i]));
    }
  }
  return result;
}

Tensor createSliceableTensorFromSlice(Graph &graph, const Tensor &s,
                                      const std::vector<std::size_t> &dims,
                                      const std::vector<std::size_t> &numSlices,
                                      const std::string &debugPrefix) {
  validateParams("createSliceableTensorFromSlice", {}, {}, s.shape(), {}, dims,
                 numSlices, false, true, true);
  std::vector<std::size_t> sizes(dims.size());
  for (std::size_t i = 0; i < dims.size(); ++i) {
    sizes[i] = s.dim(dims[i]);
  }

  // The final shape of the returned sliceable tensor.
  auto sliceableShape = s.shape();
  for (unsigned i = 0; i < dims.size(); ++i) {
    sliceableShape[dims[i]] *= numSlices[i];
  }

  const auto idxOrder = bestSliceOrder(sliceableShape, dims, sizes);

  // Create a variable with sliced dimensions factored out
  // as the outermost dimensions.
  auto createShape = s.shape();
  for (const auto idx : boost::adaptors::reverse(idxOrder)) {
    createShape.insert(createShape.begin(), numSlices[idx]);
  }

  auto t =
      graph.addVariable(s.elementType(), createShape, debugPrefix).flatten();

  const auto totalNumSlices =
      std::accumulate(numSlices.begin(), numSlices.end(), std::size_t(1),
                      std::multiplies<std::size_t>());
  // We build up the memory regions of the sliceable tensor
  // based on the given slice such that each slice/update operation
  // operates on contiguous memory and produces contiguous memory.
  const auto sBroadcast = s.expand({0}).broadcast(totalNumSlices, 0);
  const auto mapping = graph.getTileMapping(sBroadcast);
  const auto contiguousRegionsByTile =
      getSortedContiguousRegionsByTile(graph, sBroadcast, mapping);

  std::size_t offset = 0;
  for (unsigned tile = 0; tile < contiguousRegionsByTile.size(); ++tile) {
    const auto numElems =
        intervalSequenceNumElements(contiguousRegionsByTile[tile]);
    graph.setTileMapping(t.slice(offset, offset + numElems), tile);
    offset += numElems;
  }

  const auto mappingOrderedContiguously =
      flattenInnermostRegions(contiguousRegionsByTile);
  const auto inverseMapping = getInverseMapping(mappingOrderedContiguously);

  std::vector<Tensor> toConcat;
  toConcat.reserve(inverseMapping.size());
  for (const auto &i : inverseMapping) {
    toConcat.push_back(t.slice(i.begin(), i.end()));
  }

  t = concat(toConcat).reshape(createShape);
  auto referenceMapping = graph.getTileMapping(s);

  for (std::size_t i = 0; i < dims.size(); ++i) {
    const auto dim = dims.size() - i + dims[idxOrder[i]];
    t = t.dimRoll(0, dim - 1).flatten(dim - 1, dim + 1);
  }
  assert(t.shape() == sliceableShape);

  return t;
}

static Tensor dynamicSlice(Graph &graph, const Tensor &t, const Tensor &offset,
                           const std::vector<std::size_t> &dims,
                           const std::vector<std::size_t> &sizes,
                           poplar::program::Sequence *prog,
                           const std::string &debugPrefix) {
  logging::info("dynamicSlice t={}, offset={}, dims={}, sizes={}, name={}",
                t.shape(), offset.shape(), dims, sizes, debugPrefix);

  bool checkOffset = prog != nullptr;
  validateParams("dynamicSlice", {}, {}, t.shape(), offset, dims, sizes,
                 checkOffset);

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
    out =
        slice(graph, out, checkOffset ? offset[i] : offset, dims[i], sizes[i],
              prog, debugPrefix + "/dynamicSlice_d" + std::to_string(dims[i]));
  }

  return out;
}

Tensor dynamicSlice(Graph &graph, const Tensor &t, const Tensor &offset,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence &prog,
                    const std::string &debugPrefix) {
  return dynamicSlice(graph, t, offset, dims, sizes, &prog, debugPrefix);
}

Graph::TileToTensorMapping
getSliceMapping(poplar::Graph &graph, const poplar::Tensor &t,
                const std::vector<std::size_t> &dims,
                const std::vector<std::size_t> &sizes) {
  // give a dummy offset tensor as it is not used
  Tensor offset;
  auto sliceT = dynamicSlice(graph, t, offset, dims, sizes, nullptr, "");
  return graph.getTileMapping(sliceT);
}

void dynamicUpdate(Graph &graph, const Tensor &t, const Tensor &s,
                   const Tensor &offset, const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   poplar::program::Sequence &prog,
                   const std::string &debugPrefix) {
  logging::info(
      "dynamicUpdate t={}, s={}, offset={}, dims={}, sizes={}, name={}",
      t.shape(), s.shape(), offset.shape(), dims, sizes, debugPrefix);

  validateParams("dynamicUpdate", {}, {}, t.shape(), offset, dims, sizes);

  // empty sizes or dimensions are full update (TF does this)
  if (dims.size() == 0) {
    prog.add(Copy(s, t));
    return;
  }
  // If any of sizes is 0 then this is a nop. Tensorflow tests do this.
  for (auto &sz : sizes)
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
    reducedT.emplace_back(
        slice(graph, reducedT[i], offset[dim], dims[dim], sizes[dim], &prog,
              debugPrefix + "/dynamicUpdateS_d" + std::to_string(dims[i])));
  }
  // copy s into the reduced t, iterating back to full dimensions
  reducedT.emplace_back(s);
  for (unsigned ii = idxOrder.size(); ii != 0; --ii) {
    auto i = ii - 1;
    auto dsIdx = idxOrder[i]; // index into dims[] and sizes[]
    update(graph, reducedT[i], reducedT[i + 1], offset[dsIdx], dims[dsIdx],
           prog,
           debugPrefix + "/dynamicUpdateU_d" + std::to_string(dims[dsIdx]));
  }
}

// create a sequence that runs \a loopProgram the number of times stored in
// \a i. \a i is incremented after each call
static poplar::program::Sequence
countedLoop(poplar::Graph &graph, unsigned count, poplar::Tensor &i,
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

// Implementation of multiSlice with a non-null plan
static void multiSlicePlanned(Graph &graph, const Tensor &t,
                              const Tensor &offset, const Tensor &slice,
                              const std::vector<std::size_t> &dims,
                              const std::vector<std::size_t> &sizes,
                              Sequence &prog, const SlicePlanInternal &p,
                              const OptionFlags &options,
                              const std::string &debugPrefix) {
  assert(!p.isNull);
  assert(offset.rank() == 2);
  assert(offset.dim(1) == 1);
  assert(dims.size() == 1);
  assert(sizes.size() == dims.size());
  assert(t.rank() == 2);
  assert(slice.rank() == t.rank() + 1);

  const auto slicedDim = dims[0];
  const auto unslicedDim = slicedDim ^ 1;

  const auto iSplit = p.partition.lookupSplit;
  const auto sSplit = p.partition.slicedDimSplit;
  const auto hSplit = p.partition.unslicedDimSplit;

  const auto iTotalElems = offset.dim(0);
  const auto iElemsPerPartition = ceildiv(offset.dim(0), iSplit);
  const auto sTotalElems = t.dim(slicedDim);
  const auto sElemsPerPartition = ceildiv(sTotalElems, sSplit);
  const auto hTotalElems = t.dim(unslicedDim);
  const auto hElemsPerPartition = ceildiv(hTotalElems, hSplit);

  // If this is multi-stage create a new tensor laid out appropriately
  // for stage 0 to output to. Otherwise we output directly to the
  // given output tensor.
  const Tensor stage0Slice = [&] {
    if (sSplit > 1) {
      auto shape = t.shape();
      shape[dims[0]] = sizes[0];
      shape.insert(shape.begin(), iTotalElems);
      shape.insert(shape.begin(), sSplit);
      std::vector<std::size_t> nPartitions(shape.size(), 1);
      nPartitions[0] = sSplit;
      nPartitions[1] = iSplit;
      nPartitions.back() = hSplit;
      return createPartitionableTensor(graph, t.elementType(), shape,
                                       nPartitions,
                                       debugPrefix + "/stage0Output");
    }
    return slice.expand({0});
  }();

  const std::string vertexClass =
      templateVertex("popops::MultiSlice", t.elementType());
  const auto cs1 = graph.addComputeSet(debugPrefix + "/stage0");
  for (std::size_t i = 0; i < iSplit; ++i) {
    const auto iBegin = std::min(iTotalElems, i * iElemsPerPartition);
    const auto iEnd = std::min(iTotalElems, (i + 1) * iElemsPerPartition);
    if (iEnd - iBegin == 0) {
      break;
    }
    const Tensor iSplitByI = offset.slice(iBegin, iEnd, 0);
    const Tensor sSplitByI = stage0Slice.slice(iBegin, iEnd, 1);
    for (std::size_t s = 0; s < sSplit; ++s) {
      const auto sBegin = std::min(sTotalElems, s * sElemsPerPartition);
      const auto sEnd = std::min(sTotalElems, (s + 1) * sElemsPerPartition);
      if (sEnd - sBegin == 0) {
        break;
      }
      const Tensor tSplitByS = t.slice(sBegin, sEnd, slicedDim);
      const Tensor sSplitByS = sSplitByI.slice(s, s + 1, 0).squeeze({0});
      boost::optional<unsigned> baseOffset;
      if (sSplit > 1) {
        baseOffset = sBegin;
      }

      for (std::size_t h = 0; h < hSplit; ++h) {
        unsigned tile = linearizeSliceIndices(
            p.partition.slicedDimSplit, p.partition.unslicedDimSplit, i, s, h);
        const auto hBegin = std::min(hTotalElems, h * hElemsPerPartition);
        const auto hEnd = std::min(hTotalElems, (h + 1) * hElemsPerPartition);
        if (hEnd - hBegin == 0) {
          break;
        }
        const Tensor indices = iSplitByI.squeeze({1});
        const Tensor input = tSplitByS.slice(hBegin, hEnd, unslicedDim);
        const Tensor output = sSplitByS.slice(hBegin, hEnd, 1 + unslicedDim);
        graph.setTileMapping(output, tile);
        generateMultiSliceVerticesOnTile(graph, cs1, tile, input, indices,
                                         output, nullptr, vertexClass, false,
                                         slicedDim, baseOffset, debugPrefix);
      }
    }
  }
  prog.add(Execute(cs1));

  // Reduce remaining partials in a second compute set.
  if (sSplit > 1) {
    const auto cs2 = graph.addComputeSet(debugPrefix + "/Stage1");

    // Split work by indices for the second stage else we end up with
    // all the outputs on a subset of tiles quite naturally as a result of
    // the first split.

    // Calculate how much we can split indices by for the second stage.
    const auto iElemsPerPartitionStage1 = ceildiv(iElemsPerPartition, sSplit);
    const auto iSplitStage1 =
        ceildiv(iElemsPerPartition, iElemsPerPartitionStage1);

    const Tensor transformedOffset = [&] {
      // Take our original 1D indices which index:
      //
      //   {sTotalElems}
      //
      // and transform them to into 2D indices in each partition of shape:
      //
      //   {sSplit, ceildiv(iElemsPerPartition, iSplitStage1)}
      //
      //   or more simply:
      //
      //   {sSplit, iElemsPerPartitionStage1}
      //
      // We do this by flattening the 2D indices to 1D.
      //
      // To achieve this we divide the original indices by sElemsPerPartition
      // to get the correct index into the outer-dimension. Then get the offset
      // into the inner-most dimension [0:iElemsPerPartitionStage1) for this
      // index. Then flatten our 2D index to 1D by multiplying the outer-most
      // dimension's index by the number of elements in the inner-most dimension
      // in this partition.
      //
      const Tensor innerIdx = [&] {
        Tensor t = graph.clone(offset.slice(0, iElemsPerPartitionStage1));
        iota(graph, t.squeeze({1}), 0u, prog, debugPrefix);
        t = t.broadcast(iSplitStage1, 0)
                .slice(0, iElemsPerPartition)
                .broadcast(iSplit, 0)
                .slice(0, iTotalElems);
        return t;
      }();

      const Tensor innerElems = [&] {
        const auto nFirst =
            roundDown(iElemsPerPartition, iElemsPerPartitionStage1);
        const auto nLast = iElemsPerPartition % iElemsPerPartitionStage1;
        const auto firstMultiplier = iElemsPerPartitionStage1;
        const auto lastMultiplier =
            iElemsPerPartition % iElemsPerPartitionStage1;
        const auto first =
            graph.addConstant(UNSIGNED_INT, {1, 1}, firstMultiplier);
        const auto last =
            graph.addConstant(UNSIGNED_INT, {1, 1}, lastMultiplier);
        graph.setTileMapping(first, 0);
        graph.setTileMapping(last, 0);
        return concat(first.broadcast(nFirst, 0), last.broadcast(nLast, 0))
            .broadcast(iSplit, 0)
            .slice(0, iTotalElems);
      }();

      using namespace expr;
      return map(graph, _2 + ((_1 / sElemsPerPartition) * _3),
                 {offset, innerIdx, innerElems}, prog,
                 debugPrefix + "/adjustedIndicesStage1");
    }();

    for (std::size_t i = 0; i < iSplit; ++i) {
      const auto iBegin = std::min(iTotalElems, i * iElemsPerPartition);
      const auto iEnd = std::min(iTotalElems, (i + 1) * iElemsPerPartition);
      if (iEnd - iBegin == 0) {
        break;
      }
      const Tensor iSplitByI = transformedOffset.slice(iBegin, iEnd, 0);
      const Tensor tSplitByI = stage0Slice.slice(iBegin, iEnd, 1);
      const Tensor sSplitByI = slice.slice(iBegin, iEnd, 0);
      for (std::size_t s = 0; s < iSplitStage1; ++s) {
        const auto sBegin =
            std::min(iEnd - iBegin, s * iElemsPerPartitionStage1);
        const auto sEnd =
            std::min(iEnd - iBegin, (s + 1) * iElemsPerPartitionStage1);
        if (sEnd - sBegin == 0) {
          break;
        }
        const Tensor iSplitByS = iSplitByI.slice(sBegin, sEnd, 0);
        const Tensor tSplitByS = tSplitByI.slice(sBegin, sEnd, 1)
                                     .flatten(2, tSplitByI.rank())
                                     .flatten(0, 2);
        const Tensor sSplitByS =
            sSplitByI.slice(sBegin, sEnd, 0).flatten(2, sSplitByI.rank());
        for (std::size_t h = 0; h < hSplit; ++h) {
          const auto hBegin = std::min(hTotalElems, h * hElemsPerPartition);
          const auto hEnd = std::min(hTotalElems, (h + 1) * hElemsPerPartition);
          if (hEnd - hBegin == 0) {
            break;
          }
          unsigned tile =
              linearizeSliceIndices(p.partition.slicedDimSplit,
                                    p.partition.unslicedDimSplit, i, s, h);
          const Tensor indices = iSplitByS.squeeze({1});
          const Tensor input = tSplitByS.slice(hBegin, hEnd, 1);
          const Tensor output = sSplitByS.slice(hBegin, hEnd, 2);
          generateMultiSliceVerticesOnTile(graph, cs2, tile, input, indices,
                                           output, nullptr, vertexClass, false,
                                           slicedDim, 0, debugPrefix);
        }
      }
    }
    prog.add(Execute(cs2));
  }
}

Tensor multiSlice(Graph &graph, const Tensor &t, const Tensor &offset,
                  const std::vector<std::size_t> &dims,
                  const std::vector<std::size_t> &sizes, Sequence &prog,
                  const SlicePlan &plan, const OptionFlags &options,
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
        std::to_string(offset.dim(1)) +
        ", dims.size()== " + std::to_string(dims.size()));
  validateParams("multiSlice", plan, options, t.shape(), offset[0], dims,
                 sizes);

  // We always map the output in the same way to avoid surprising changes when
  // the number of slices changes
  Tensor sMulti;
  if (plan.getImpl().isNull) {
    sMulti = createSliceTensor(graph, t, dims, sizes, offset.dim(0), dName);
  } else {
    sMulti = createSliceTensor(graph, t.elementType(), t.shape(), dims, sizes,
                               offset.dim(0), plan, options, dName);
  }

  logging::info("multiSlice {} -> {}, name={}, nullplan?={}", t.shape(),
                sMulti.shape(), debugPrefix, plan.getImpl().isNull);

  if (!plan.getImpl().isNull) {
    multiSlicePlanned(graph, t, offset, sMulti, dims, sizes, prog,
                      plan.getImpl(), options, dName);
    return sMulti;
  }

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
  // For now only 1d slices of 2d base tensors are supported.
  if (t.rank() == 2 && dims.size() == 1 && sMulti.rank() == 3 &&
      offset.rank() == 2 && offset.dim(1) == 1 && offset.dim(0) > 6) {
    generateMultiSliceVertices("popops::MultiSlice", false, false, graph, prog,
                               offset, t, sMulti, nullptr, dims[0], boost::none,
                               options, dName);
    return sMulti;
  }

  // looping case
  Sequence body;
  auto sIdx = graph.addVariable(UNSIGNED_INT, {1}, dName + "/sIdx");
  auto zero = graph.addConstant(UNSIGNED_INT, {1}, 0, dName + "/zero");
  graph.setTileMapping(sIdx, 0);
  graph.setTileMapping(zero, 0);
  prog.add(Copy(zero, sIdx));
  auto tIdx =
      dynamicSlice(graph, offset, sIdx, {0}, {1}, body, dName + "/sliceIndex")
          .squeeze({0});

  auto sI = dynamicSlice(graph, t, tIdx, dims, sizes, body, dName + "/slice")
                .expand({0});
  dynamicUpdate(graph, sMulti, sI, sIdx, {0}, {1}, body, dName + "/update");
  prog.add(countedLoop(graph, offset.dim(0), sIdx, body, dName + "/loop"));
  return sMulti;
}

// This is derived from multiSlice with \a s input rather than generated,
// the tensors swapped, etc
void multiUpdate(Graph &graph, const Tensor &t, const Tensor &sMulti,
                 const Tensor &offset, const std::vector<std::size_t> &dims,
                 const std::vector<std::size_t> &sizes, Sequence &prog,
                 const SlicePlan &plan, const OptionFlags &options,
                 const std::string &debugPrefix) {
  logging::info("multiUpdate {} into {}, name={}", sMulti.shape(), t.shape(),
                debugPrefix);
  // small number of slices are updated individually
  // large number of slices are updated by a specialisation or in a loop
  std::string dName = debugPrefix + "/multiUpdate";

  // Check the offsets have been specified with a multi-slice dimension
  if (offset.rank() != 2)
    throw poputil::poplibs_error(
        "multiUpdate expects offset.rank() == 2 but it is" +
        std::to_string(offset.rank()));
  if (offset.dim(1) != dims.size())
    throw poputil::poplibs_error(
        "multiUpdate expects offset.dim(1) == dims.size(); offset.dim(1)==" +
        std::to_string(offset.dim(1)) +
        ", dims.size()== " + std::to_string(dims.size()));
  if (!plan.getImpl().isNull) {
    throw poputil::poplibs_error(
        "multiUpdate does not currently handle non-default SlicePlans");
  }
  validateParams("multiUpdate", plan, options, t.shape(), offset[0], dims,
                 sizes);

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
  // For now only 1d slices of 2d base tensors are supported.
  if (t.rank() == 2 && dims.size() == 1 && sMulti.rank() == 3 &&
      offset.rank() == 2 && offset.dim(1) == 1 && offset.dim(0) > 6) {
    generateMultiSliceVertices("popops::MultiUpdate", true, false, graph, prog,
                               offset, t, sMulti, nullptr, dims[0], boost::none,
                               options, dName);
    return;
  }
  // looping case
  Sequence body;
  auto sIdx = graph.addVariable(UNSIGNED_INT, {1}, dName + "/sIdx");
  auto zero = graph.addConstant(UNSIGNED_INT, {1}, 0, dName + "/zero");
  graph.setTileMapping(sIdx, 0);
  graph.setTileMapping(zero, 0);
  prog.add(Copy(zero, sIdx));
  auto tIdx =
      dynamicSlice(graph, offset, sIdx, {0}, {1}, body, dName + "/sliceIndex")
          .squeeze({0});

  auto sI =
      dynamicSlice(graph, sMulti, sIdx, dims, sizes, body, dName + "/slice")
          .squeeze({0});
  dynamicUpdate(graph, t, sI, tIdx, {0}, {1}, body, dName + "/update");
  prog.add(countedLoop(graph, offset.dim(0), sIdx, body, dName + "/loop"));
}

// This is derived from multiUpdate, but s is added to t rather than replacing
// it
// Currently only a single dimension may be sliced
void multiUpdateAdd(Graph &graph, const Tensor &t, const Tensor &sMulti,
                    const Tensor &offset, const Tensor &scale,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes, Sequence &prog,
                    const SlicePlan &plan, const OptionFlags &options,
                    const std::string &debugPrefix) {
  logging::info("multiUpdateAdd {} into {}, name={}, nullplan={}",
                sMulti.shape(), t.shape(), debugPrefix, plan.getImpl().isNull);
  std::string dName = debugPrefix + "/multiUpdateAdd";
  // Check the offsets have been specified with a multi-slice dimension
  if (offset.rank() != 2)
    throw poputil::poplibs_error(
        "multiUpdateAdd expects offset.rank() == 2 but it is" +
        std::to_string(offset.rank()));
  if (offset.dim(1) != dims.size())
    throw poputil::poplibs_error(
        "multiUpdateAdd expects offset.dim(1) == dims.size(); offset.dim(1)==" +
        std::to_string(offset.dim(1)) +
        ", dims.size()== " + std::to_string(dims.size()));
  validateParams("multiUpdateAdd", plan, options, t.shape(), offset[0], dims,
                 sizes);

  if (t.rank() != 2 || dims.size() != 1 || offset.rank() != 2 ||
      offset.dim(1) != 1)
    throw poputil::poplibs_error(
        "multiUpdateAdd requires t to have 2 dimensions and dims to specify "
        "1 dimension");
  if (t.elementType() != sMulti.elementType() ||
      t.elementType() != scale.elementType())
    throw poputil::poplibs_error(
        "multiUpdateAdd expects t, sMulti and scale to have the same type");
  if (scale.rank() != 0)
    throw poputil::poplibs_error("multiUpdateAdd scale must be a scaler");
  if (plan.getImpl().isNull) {
    generateMultiSliceVertices("popops::MultiUpdateAdd", true, true, graph,
                               prog, offset, t, sMulti, &scale, dims[0],
                               boost::none, options, dName);
  } else {
    generatePlannedMultiUpdateAdd("popops::MultiUpdateAdd", plan.getImpl(),
                                  graph, prog, offset, t, sMulti, scale,
                                  dims[0], dName);
  }
}

namespace embedding {

static void applyPlanConstraints(popsolver::Model &m,
                                 const PlanConstraints &planConstraints,
                                 const popsolver::Variable mSlicedDimSplit,
                                 const popsolver::Variable mUnslicedDimSplit,
                                 const popsolver::Variable mLookupSplit) {
  const auto constrainVar = [&](const char *name, popsolver::Variable var) {
    if (auto constraint = planConstraints.get_optional<unsigned>(name)) {
      m.equal(var, *constraint);
    }
  };

  // unslicedGrainSize is constrained at the beginning of model construction
  // as that number is used for calculating other values in the model.
  constrainVar("slicedDimSplit", mSlicedDimSplit);
  constrainVar("unslicedDimSplit", mUnslicedDimSplit);
  constrainVar("lookupSplit", mLookupSplit);
}

// Plan an embedding layer for slicing/updating.
// This planner aims to minimise the persistent tile memory while keeping
// temporary memory below a bound.
SlicePlan plan(const Graph &graph, const Type &dataType,
               const std::size_t numEntries,
               const std::size_t outputSize, // embedding size
               const std::vector<std::size_t> &numLookups,
               const OptionFlags &optionFlags) {
  const auto options = parseSliceOptions(optionFlags);

  logging::debug("DynamicSlicePlan for type {}, numEntries {}, outputSize {},"
                 " numLookups {}",
                 dataType, numEntries, outputSize, numLookups);
  const auto &target = graph.getTarget();
  const auto dataElementSize = target.getTypeSize(dataType);

  // Plan based on the max supplied number of indices
  unsigned plannedNumIndices =
      numLookups.empty()
          ? 1
          : *std::max_element(numLookups.cbegin(), numLookups.cend());
  SlicePlanInternal p;

  // Choose the grainsize in unsliced dimension to avoid subword writes
  const std::size_t minGrainSizeBytes = target.getAtomicStoreGranularity();

  // The embedding dimension can be split (embeddingSplit),
  // the entries can be split (dictSplit),
  // the indices can be split (lookupSplit)
  popsolver::Model m;
  // Indices are int32 so 4bytes each
  const auto mBytesPerIndex = m.addConstant(target.getTypeSize(UNSIGNED_INT));
  const auto mBytesPerFloat = m.addConstant(target.getTypeSize(FLOAT));

  // The grainsize can be constrained externally so bytesPerGrain must be
  // derived from it
  const auto unslicedGrainSize =
      options.planConstraints.get_optional<unsigned>("unslicedGrainSize")
          .value_or(minGrainSizeBytes /
                    gcd(minGrainSizeBytes, dataElementSize));
  const auto mUnslicedGrainSize =
      m.addConstant(unslicedGrainSize, "unslicedGrainSize");
  const auto bytesPerGrain = unslicedGrainSize * dataElementSize;
  const auto mBytesPerGrain = m.addConstant(bytesPerGrain);

  const auto mOutputSize = m.addConstant(outputSize, "outputSize");
  const auto mNumUnslicedGrains = // per row
      m.ceildiv(mOutputSize, mUnslicedGrainSize, "numUnslicedGrains");

  // split the embedding between \a mEmbeddingSplit tiles
  const auto mEmbeddingSplit =
      m.addVariable(1, std::numeric_limits<unsigned>::max(), "embeddingSplit");
  m.lessOrEqual(mEmbeddingSplit, mNumUnslicedGrains);
  m.ceildivConstrainDivisor(mNumUnslicedGrains, mEmbeddingSplit);

  // The entries are split across \a entriesSplit groups of tiles,
  // each of which will select a candidate in the first stage of a lookup.
  // A second stage is then required to select between theses candidates. This
  // means that temporary memory is required after the first pass.
  // Splits leaving less than 2 entries per tile will have more unmeasured
  // overhead than is saved in base memory so are prohibited.
  const auto mDictSplit =
      m.addVariable(1, ceildiv(numEntries, 2u), "entriesSplit");
  // mDictIsSplit=0 when mDictSplit==1, else 1
  const auto mDictIsSplit =
      m.sub(m.addConstant(1), m.floordiv(m.addConstant(1), mDictSplit));

  // When there are many lookups we can split the lookups between multiple
  // groups of tiles each performing the same lookup on a subset of indices.
  // This requires the embedding to be broadcast for lookups, and the updates
  // to be serialised or reduced on update
  // When there is an indices split a temporary embedding buffer is required in
  // both passes
  const auto mLookupSplit = m.addVariable(1, plannedNumIndices, "lookupSplit");
  // mLookupsAreSplit=0 when mLookupSplit==1 split, else 1
  const auto mLookupsAreSplit =
      m.sub(m.addConstant(1), m.floordiv(m.addConstant(1), mLookupSplit));
  const auto mNumTiles = m.addConstant(target.getNumTiles(), "numTiles");
  const auto mNumEntries = m.addConstant(numEntries);
  const auto mNumIndices = m.addConstant(plannedNumIndices);

  // When `mLookupSplit` != 1 the dictionary is distributed across the different
  // lookup instantiations and broadcast before use
  const auto mDictEntriesPerTile = m.ceildivConstrainDivisor(
      mNumEntries, m.product({mDictSplit, mLookupSplit}));

  const auto mBaseGrainsPerRow = m.ceildiv(mNumUnslicedGrains, mEmbeddingSplit);
  const auto mIndicesPerLGroup = m.ceildiv(mNumIndices, mLookupSplit);
  const auto mUsedTiles =
      m.product({mEmbeddingSplit, mDictSplit, mLookupSplit}, "totalSplit");
  m.lessOrEqual(mUsedTiles, mNumTiles);

  // The memory required by the base (embedding) tensor.
  // Note we budget assuming each group will have 1/mDictSplit
  // of the embedding plus a full copy in temporary memory.
  const auto mBaseGrains = m.product({mBaseGrainsPerRow, mDictEntriesPerTile});
  const auto mSlicesGrains = m.product({mBaseGrainsPerRow, mIndicesPerLGroup});
  const auto mOutputGrains =
      m.product({mBaseGrainsPerRow, m.ceildiv(mIndicesPerLGroup, mDictSplit)});
  const auto mBaseBytes = m.product({mBaseGrains, mBytesPerGrain});
  const auto mIndicesBytes = m.product({mIndicesPerLGroup, mBytesPerIndex});
  const auto mOutputBytes = m.product({mOutputGrains, mBytesPerGrain});

  // The base tensor must be broadcast across the `mLookupSplit` groups as it
  // is distributed to balance memory.
  // The indices must be received from a set of tiles, so a number of setmux
  // instructions are required.
  auto mIndicesPerTile = m.max(
      {m.ceildiv(mNumIndices, mNumTiles), m.addConstant(minIndicesPerTile)});
  auto mIndicesExchangeInstrs0 = m.ceildiv(mIndicesPerLGroup, mIndicesPerTile);

  // 0 and 1 indicate which stage in the forward pass this exchange is
  // attributed to
  auto mEmbeddingExchangeInstrs0 = m.product({mLookupsAreSplit, mLookupSplit});
  // When there is a dictSplit the data will be exchanged between groups of
  // `mDictSplit` tiles
  auto mOutputToInputExchangeInstrs1 = m.product({mDictIsSplit, mDictSplit});
  // The indices are copied implicitly and are re-broadcast for the second stage
  auto &mIndicesExchangeInstrs1 = mIndicesExchangeInstrs0;
  auto mExchangeCodeBytes = m.product(
      {m.addConstant(4u),
       m.sum({mEmbeddingExchangeInstrs0, mIndicesExchangeInstrs0,
              mOutputToInputExchangeInstrs1, mIndicesExchangeInstrs1})});

  auto mUpdateTmpBytes = m.addConstant(0);
  if (options.usedForUpdate) {
    // When no index split there are no temporaries beyond those used in a
    // lookup, the vertices work directly on the base, slices and indices
    // tensors.
    // When `mLookupsAreSplit` the indices and updates are rearranged onto the
    // tile, the updates are cast to FLOAT and then accumulated
    // with a FLOAT copy of the base tensor.
    const auto mPreCastUpdateBytes = // copy of the slices for a tile
        m.product({mSlicesGrains, mBytesPerGrain});
    const auto mCastUpdateBytes =
        m.product({mSlicesGrains, mUnslicedGrainSize, mBytesPerFloat});
    const auto mPartialBytes = m.product(
        {mBaseGrains, mLookupSplit, mUnslicedGrainSize, mBytesPerFloat});
    const auto mRearrangedIndices =
        m.product({mIndicesPerLGroup, mBytesPerIndex});

    const auto mMaxTmp =
        m.max({// pre-cast and float-cast updates
               m.sum({mPreCastUpdateBytes, mCastUpdateBytes}),
               // float-updates, indices and partial
               m.sum({mRearrangedIndices, mCastUpdateBytes, mPartialBytes}),
               // reduction (also the actual update will have the base upcast to
               // the same size as the partials, so the same footprint)
               m.sum({mPartialBytes, mPartialBytes})});
    mUpdateTmpBytes = m.product({mLookupsAreSplit, mMaxTmp});

    // indices are as for the forward pass;
    // plus the rearrangement will be an all-all exchange
    mExchangeCodeBytes =
        m.sum({mExchangeCodeBytes,
               m.product({mIndicesExchangeInstrs0, m.addConstant(4)}),
               m.product({mLookupsAreSplit, mLookupSplit, m.addConstant(4)})});
  }

  // When `mLookupsAreSplit` the base tensor must be reconstituted
  const auto mTmpTileDictBytes =
      m.product({mLookupsAreSplit, mLookupSplit, mBaseBytes});

  // When splitting the dictionary a rearrangement is required between the two
  // stages
  const auto mTmpRearrangeGrains =
      m.product({mDictIsSplit, mBaseGrainsPerRow, mIndicesPerLGroup});
  const auto mTmpRearrangeBytes =
      m.product({mTmpRearrangeGrains, mBytesPerGrain});

  const auto mPeakTmpBytes = m.max(
      {// copy of the required part of the embedding matrix
       m.sum({mTmpTileDictBytes, mTmpRearrangeBytes}),
       // output of first stage + rearrangement version of the second
       m.sum({mTmpRearrangeBytes, mTmpRearrangeBytes}),
       !options.usedForUpdate ? m.addConstant(0)
                              : m.sum({mTmpRearrangeBytes, mUpdateTmpBytes})});

  if (false) {
    // No hard constaint on temp memory at the moment
    const auto maxGrainsPerTile = target.getBytesPerTile() / bytesPerGrain;
    const auto mMaxAllowedTmpBytes =
        m.addConstant(0.6 * maxGrainsPerTile * bytesPerGrain);
    m.lessOrEqual(mPeakTmpBytes, mMaxAllowedTmpBytes);
  }

  // Minimise total memory footprint, prioritising persistent memory
  // indices are persistent if they are required for the update pass
  //
  // TODO: T12935 Consider hard limit on temporary bytes specified via options
  // to the plan.
  auto goal = m.sum({mBaseBytes, mOutputBytes, mExchangeCodeBytes});
  if (options.usedForUpdate) {
    goal = m.sum({goal, mIndicesBytes});
  }
  goal = m.product({goal, m.addConstant(10)});
  goal = m.sum({goal, mPeakTmpBytes});

  applyPlanConstraints(m, options.planConstraints, mDictSplit, mEmbeddingSplit,
                       mLookupSplit);
  popsolver::Solution s = m.minimize({goal});

  // We must have a valid solution.
  if (!s.validSolution()) {
    logging::warn(
        "Slice planner could not find a valid solution, opting for no plan");
    return std::make_unique<SlicePlanInternal>();
  }

  p.partition.lookupSplit = s[mLookupSplit];
  p.partition.slicedDimSplit = s[mDictSplit];
  p.partition.unslicedDimSplit = s[mEmbeddingSplit];
  p.partition.unslicedGrainSize = s[mUnslicedGrainSize];
  p.rank = 2;
  p.slicedDims = {0};
  p.slicedDimSizes = {1};
  p.isNull = false;

  logging::debug("Embedding {}", p);
  logging::debug("UsedTiles {}", s[mUsedTiles]);
  logging::debug("mNumUnslicedGrains {}, mBaseGrainsPerRow {}",
                 s[mNumUnslicedGrains], s[mBaseGrainsPerRow]);
  logging::debug("Memory estimates(bytes): base {}, output {}, indices {},"
                 " indicesSrcs {}, exch {}"
                 " DictTemp {}, ReTemp {}, UpdateReduction {}, goal {}",
                 s[mBaseBytes], s[mOutputBytes], s[mIndicesBytes],
                 s[mIndicesExchangeInstrs0], s[mExchangeCodeBytes],
                 s[mTmpTileDictBytes], s[mTmpRearrangeBytes],
                 s[mUpdateTmpBytes], s[goal]);
  logging::debug("mDictSplit {}, mEmbeddingSplit {}, lookupSplit {}",
                 s[mDictSplit], s[mEmbeddingSplit], s[mLookupSplit]);

  return std::make_unique<SlicePlanInternal>(std::move(p));
}

} // end namespace embedding

} // end namespace popops
