// Copyright (c) 2017 Graphcore Ltd. All rights reserved.
#include "popops/DynamicSlice.hpp"

#include "CastModelling.hpp"
#include "DynamicSliceInternal.hpp"
#include "ExchangeEstimator.hpp"
#include "FillModelling.hpp"
#include "ScaledAddModelling.hpp"
#include "reduction/Modelling.hpp"

#include "poplar/Interval.hpp"
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/Algorithms.hpp"
#include "poplibs_support/Compiler.hpp"
#include "poplibs_support/ContiguousRegionsByTile.hpp"
#include "poplibs_support/PlanConstraints.hpp"
#include "poplibs_support/TileHierarchy.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/gcd.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Encoding.hpp"
#include "popops/Loop.hpp"
#include "popops/OperationDefUtil.hpp"
#include "popops/PerformanceEstimation.hpp"
#include "popops/Reduce.hpp"
#include "popops/ScaledAdd.hpp"
#include "popops/Zero.hpp"
#include "popsolver/Model.hpp"
#include "poputil/DebugInfo.hpp"
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
#include <optional>
#include <type_traits>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;
using namespace poplibs;
using namespace popops::internal;
using namespace popops::modelling;

// Temporary environment variable that can be used to globally force the
// planning target (memory/cycles) for a planned multiSlice/Update.
static const char forceUsePlanningTargetVar[] =
    "POPLIBS_SLICE_PLAN_FORCE_TARGET";

namespace poputil {
template <> poplar::ProfileValue toProfileValue(const popops::SlicePlan &p) {
  poplar::ProfileValue::Map v;
  if (p.internal) {
    v.insert(
        {"lookupSplit", toProfileValue(p.internal->partition.lookupSplit)});
    v.insert({"slicedDimSplit",
              toProfileValue(p.internal->partition.slicedDimSplit)});
    v.insert({"unslicedDimSplit",
              toProfileValue(p.internal->partition.unslicedDimSplit)});
    v.insert({"unslicedGrainSize",
              toProfileValue(p.internal->partition.unslicedGrainSize)});
  }
  return v;
}
} // namespace poputil

namespace popops {

namespace {

constexpr std::size_t minIndicesPerTile = 32;

enum class PlanMinimisationTarget {
  /// Minimise a weighted combination of estimated operand & temporary memory
  /// usage.
  MEMORY,
  /// Minimise total estimated cycles.
  CYCLES
};

static std::map<std::string, PlanMinimisationTarget> planMinimisationTargetMap{
    {"memory", PlanMinimisationTarget::MEMORY},
    {"cycles", PlanMinimisationTarget::CYCLES},
};

inline std::ostream &operator<<(std::ostream &os,
                                const PlanMinimisationTarget &t) {
  switch (t) {
  case PlanMinimisationTarget::MEMORY:
    os << "memory";
    break;
  case PlanMinimisationTarget::CYCLES:
    os << "cycles";
    break;
  default:
    throw poplibs_error("Unknown PlanMinimisationTarget");
  }
  return os;
}

enum class IndicesDistribution {
  // Indices equally likely to take on any possible index in
  // the valid range.
  UNIFORM,
  // Indices only take on a single value.
  ONE_POINT
};

inline std::ostream &operator<<(std::ostream &os,
                                const IndicesDistribution &d) {
  switch (d) {
  case IndicesDistribution::UNIFORM:
    os << "uniform";
    break;
  case IndicesDistribution::ONE_POINT:
    os << "onePoint";
    break;
  default:
    throw poplibs_error("Unknown IndicesDistribution");
  }
  return os;
}

static std::map<std::string, IndicesDistribution> indicesDistributionMap{
    {"uniform", IndicesDistribution::UNIFORM},
    {"onePoint", IndicesDistribution::ONE_POINT}};

struct SliceOptions {
  SliceOptions() = default;

  PlanConstraints planConstraints;
  // Specify whether a plan is to be used for a slice.
  bool usedForSlice = true;

  // Specify whether a plan is to be used for an update.
  bool usedForUpdate = true;

  // Specify the update operation to perform.
  std::optional<Operation> opForUpdate = Operation::ADD;

  // TODO: T12930 Add option to specify whether a plan is to be used for a
  // lookup.

  // The target maximum temporary memory usage for the operation. This
  // may not be satisfiable.
  std::optional<double> availableMemoryProportion;

  // For use when planning, the distribution of indices to assume when
  // estimating cycles.
  IndicesDistribution indicesDistribution = IndicesDistribution::UNIFORM;

  // Controls the target for optimisation when planning.
  PlanMinimisationTarget planMinimisationTarget =
      PlanMinimisationTarget::MEMORY;

  // Indices are sorted in increasing order
  bool indicesAreSorted = false;

  bool alwaysIncludeBaseRearrangementCost = true;
};

std::ostream &operator<<(std::ostream &os, const SliceOptions &o) {
  os << "{usedForSlice=" << (o.usedForSlice ? "true" : "false")
     << ", usedForUpdate=" << (o.usedForUpdate ? "true" : "false");
  if (o.opForUpdate == std::nullopt) {
    os << " (op=none)";
  } else {
    os << "(op=" << *o.opForUpdate << ")";
  }
  os << ", availableMemoryProportion=";
  if (o.availableMemoryProportion) {
    os << *o.availableMemoryProportion;
  } else {
    os << "none";
  }
  os << ", indicesDistribution=" << o.indicesDistribution
     << ", planMinimisationTarget=" << o.planMinimisationTarget
     << ", indicesAreSorted=" << (o.indicesAreSorted ? "true" : "false")
     << ", internal.alwaysIncludeBaseRearrangementCost="
     << (o.alwaysIncludeBaseRearrangementCost ? "true" : "false") << "}";
  return os;
}

struct ValidateSlicePlanConstraintsOption {
  void operator()(const boost::property_tree::ptree &t) const {
    if (t.empty() && !t.data().empty()) {
      throw poplar::invalid_option("Plan constraints must be an object");
    }

    for (const auto &child : t) {
      if (child.first != "lookupSplit" && child.first != "slicedDimSplit" &&
          child.first != "unslicedDimSplit" &&
          child.first != "unslicedGrainSize" &&
          child.first != "useOrderingInfo") {
        throw poplibs_error("Unrecognised constraint " + child.first);
      }

      if (child.first == "useOrderingInfo") {
        validatePlanConstraintsBoolean(child.first, child.second);
      } else {
        validatePlanConstraintsUnsigned(child.first, child.second);
      }
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
  o << "  useIndicesOrderingInfo="
    << (p.useIndicesOrderingInfo ? " true" : "false");
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

bool operator<(const SlicePlan &a, const SlicePlan &b) noexcept {
  return *a.internal < *b.internal;
}

bool operator==(const SlicePlan &a, const SlicePlan &b) noexcept {
  return *a.internal == *b.internal;
}

bool operator!=(const SlicePlan &a, const SlicePlan &b) noexcept {
  return !(a == b);
}

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

static SliceOptions parseSliceOptions(const OptionFlags &optionFlags_) {
  auto optionFlags = optionFlags_;
  SliceOptions options;

  using poplibs::OptionHandler;
  using poplibs::OptionSpec;
  using poplibs_support::makePlanConstraintsOptionHandler;

  const auto makeSlicePlanConstraintsOptionHandler =
      &makePlanConstraintsOptionHandler<ValidateSlicePlanConstraintsOption>;

  // TODO: T45888 - Remove this environment variable and the option it
  // forces.
  if (const char *forcedPlanningTarget =
          std::getenv(forceUsePlanningTargetVar)) {
    optionFlags.set("planMinimisationTarget", forcedPlanningTarget);
  }

  /*
   * Any changes to spec must be reflected in the documentation comment in
   * the header.
   */
  const OptionSpec spec{
      {"planConstraints",
       makeSlicePlanConstraintsOptionHandler(options.planConstraints)},
      {"usedForSlice", OptionHandler::createWithBool(options.usedForSlice)},
      {"usedForUpdate", OptionHandler::createWithBool(options.usedForUpdate)},
      {"operationForUpdate",
       createOptionalEnumHandler(
           options.opForUpdate,
           {{"add", Operation::ADD}, {"max", Operation::MAX}})},
      {"availableMemoryProportion",
       createOptionalDoubleHandler(options.availableMemoryProportion, 0.)},
      {"indicesDistribution",
       OptionHandler::createWithEnum(options.indicesDistribution,
                                     indicesDistributionMap)},
      // TODO: T45888 - Remove this option
      {"planMinimisationTarget",
       OptionHandler::createWithEnum(
           options.planMinimisationTarget,
           {{"memory", PlanMinimisationTarget::MEMORY},
            {"cycles", PlanMinimisationTarget::CYCLES}})},
      {"indicesAreSorted",
       OptionHandler::createWithBool(options.indicesAreSorted)},
      {"internal.alwaysIncludeBaseRearrangementCost",
       OptionHandler::createWithBool(
           options.alwaysIncludeBaseRearrangementCost)}};

  for (const auto &entry : optionFlags) {
    spec.parse(entry.first, entry.second);
  }

  return options;
}

template <typename T>
static T valueOr(const std::optional<T> &value, const T &otherwise) {
  return value ? *value : otherwise;
}

static Tensor createSliceTensor(Graph &graph, const Type &type,
                                const std::vector<std::size_t> &shape,
                                const std::size_t slicedDim,
                                const std::size_t numIndices,
                                const SlicePlanInternal &plan,
                                const OptionFlags &options,
                                const DebugNameAndId &dnai);

// This is specifically for embedding layer shaped operations currently.
// Given an index into a set of indices into partitions of different
// dimensions of the operation, return the tile on which this portion
// of the operation will be calculated.
static unsigned linearizeSliceIndices(const std::size_t indexPartition,
                                      const std::size_t slicedPartition,
                                      const std::size_t unslicedPartition,
                                      const std::size_t indexIdx,
                                      const std::size_t slicedIdx,
                                      const std::size_t unslicedIdx) {
  unsigned tile = 0;

  // indices
  tile = tile * indexPartition + indexIdx;

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
                             const DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai});

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

  std::vector<Tensor> t2dSlices(t2d.dim(0)), s2dSlices(s2d.dim(0));
  std::vector<Tensor *> slicePtrs;
  slicePtrs.reserve(t2d.dim(0) - 1 + s2d.dim(0));
  t2dSlices[0] = t2d.slice(0, 1).flatten();
  for (std::size_t s = 1; s < t2d.dim(0); ++s) {
    t2dSlices[s] = t2d.slice(s, s + 1).flatten();
    slicePtrs.push_back(&t2dSlices[s]);
  }
  for (std::size_t s = 0; s < s2d.dim(0); ++s) {
    s2dSlices[s] = s2d.slice(s, s + 1).flatten();
    slicePtrs.push_back(&s2dSlices[s]);
  }
  graph.reorderToSimplify(&t2dSlices[0], slicePtrs, false);

  const auto sliceShape = t2d.slice(0, 1).shape();
  for (std::size_t s = 0; s < t2dSlices.size(); ++s) {
    t2dSlices[s] = t2dSlices[s].reshape(sliceShape);
  }
  for (std::size_t s = 0; s < s2dSlices.size(); ++s) {
    s2dSlices[s] = s2dSlices[s].reshape(sliceShape);
  }
  t2d = concat(t2dSlices);
  s2d = concat(s2dSlices);

  auto mapping = graph.getTileMapping(t2d[0]);

  // instantiate vertices following the mapping of t's first slice
  for (unsigned tile = 0; tile != numTiles; ++tile) {
    const auto tileContiguousRegions =
        graph.getSortedContiguousRegions(t2d[0], mapping[tile]);
    if (tileContiguousRegions.size() == 0)
      // do nothing on this tile
      continue;

    assert(offset.numElements() == 1);
    if (tileContiguousRegions.size() == 1) {
      const auto &regions = tileContiguousRegions[0];
      const Tensor tileBase = concat(t2d.slices(regions, 1), 1);
      const Tensor tileSub = concat(s2d.slices(regions, 1), 1);

      if (tileBase.isContiguous()) {
        auto v = graph.addVertex(
            cs, templateVertex(vertexName + "1D", t2d.elementType()),
            {{"offset", offset},
             {"baseT", tileBase.flatten()},
             {"subT", tileSub.flatten()}});

        // the assembly relies on underflow of baseIdx with numBaseElements,
        // therefore the maximum value each can be is 2^31 - 1. we can't check
        // baseIdx at compile time but we can the size of numBaseElements at
        // the very least. both are checked at runtime in the C++ codelet.
        assert(numBaseElements < (1u << 31u));
        graph.setInitialValue(v["numBaseElements"], numBaseElements);
        graph.setInitialValue(v["numSubElements"], numSubElements);
        graph.setInitialValue(v["regionSize"], tileBase.dim(1));
        graph.setTileMapping(v, tile);
        continue;
      }
    }

    const auto templatedVertexName =
        templateVertex(vertexName + "2D", t2d.elementType());

    // Get the minimum of the maximum field sizes for base and sub edges
    const auto maxBaseTRegionSize =
        graph.getMaxFieldDim(templatedVertexName, "baseT", 1);
    const auto maxSubTRegionSize =
        graph.getMaxFieldDim(templatedVertexName, "subT", 1);
    const auto maxRegionSize = std::min(maxBaseTRegionSize, maxSubTRegionSize);

    // Limit the vector size the vertex deals with to meet edge size
    // constraints. This could create more vertices than the number
    // of workers.
    auto vertexSeqs = splitRegionsBetweenWorkers(
        target, tileContiguousRegions, grainSize, 2 * grainSize, maxRegionSize);
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

      auto v =
          graph.addVertex(cs, templatedVertexName,
                          {{"offset", offset}, {"baseT", base}, {"subT", sub}});
      graph.setInitialValue(v["numBaseElements"], numBaseElements);
      graph.setInitialValue(v["numSubElements"], numSubElements);
      graph.setInitialValue(v["numRegions"], base.size() / numBaseElements);
      graph.setTileMapping(v, tile);
    }
  } // end loop over tiles

  prog.add(Execute(cs, {dnai}));
}

static inline unsigned
getMultiUpdateOpMaxElemsPerWorker(unsigned numWorkerContexts,
                                  unsigned numElems) {
  return ceildiv(numElems, numWorkerContexts);
}

// Generate vertices on a specified tile to perform a multi-slice
// where indices are potentially split between workers depending on the
// operation.
static void generateMultiSliceVerticesOnTile(
    Graph &graph, const ComputeSet &cs, unsigned tile, const Tensor &base,
    const Tensor &offset, const Tensor &slices,
    const boost::optional<Tensor> &scale, const std::string &vertexName,
    bool isUpdate, unsigned baseSlicedDim, boost::optional<unsigned> baseOffset,
    boost::optional<Operation> op, bool indicesAreSorted,
    const DebugNameAndId &dnai) {
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
    if (scale) {
      graph.connect(v["scale"], scale.get());
    }
    if (isUpdate) {
      // Divide work for multi-update with an operation
      assert(numParallelWorkers == 1);
      const auto tileElements = base.dim(baseSlicedDim);
      const auto maxElementsPerWorker = getMultiUpdateOpMaxElemsPerWorker(
          target.getNumWorkerContexts(), tileElements);
      graph.setInitialValue(v["maxElementsPerWorker"], maxElementsPerWorker);
    }

    graph.setInitialValue(v["indicesAreSorted"], indicesAreSorted && isUpdate);
    graph.setInitialValue(v["baseOffset"], baseOffset ? *baseOffset : 0u);
    graph.setInitialValue(v["numBaseElements"], base.dim(baseSlicedDim));
    graph.setInitialValue(v["regionSize"], regionSize);
    graph.setTileMapping(v, tile);
  }
}

static void generateMultiSliceVertices(
    const std::string &vertexNameUntemplated, bool isUpdate,
    boost::optional<Operation> op, Graph &graph, Sequence &prog,
    const Tensor &offsets, Tensor base, Tensor slices,
    const boost::optional<Tensor> &scale, unsigned baseSlicedDim,
    boost::optional<unsigned> baseOffset, const OptionFlags &optionFlags,
    const DebugNameAndId &dnai) {
  const auto options = parseSliceOptions(optionFlags);

  auto cs = graph.addComputeSet({dnai});

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
  assert(isUpdate || scale == boost::none); // no scale on slice

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
      const unsigned availableBytesPerTile =
          std::ceil(target.getBytesPerTile() *
                    valueOr(options.availableMemoryProportion, 0.6));

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
                                       {1}, 0, {dnai, "baseRearranged"});
          prog.add(Copy(*originalBase, base, false, {dnai}));
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
    if (op != boost::none) {
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
        if (scale == boost::none) {
          vertexName = templateVertex(vertexNameUntemplated, base.elementType(),
                                      needSubwordWrites, *op);
        } else {
          vertexName =
              templateVertex(vertexNameUntemplated, base.elementType(),
                             scale->elementType(), needSubwordWrites, *op);
        }
      } else {
        // For halves we process 32-bit at a time and therefore pad the tensors
        // in the case where region size is odd.
        if (target.getTypeSize(type) == 2 && regionSize % 2 != 0) {
          const auto padWithSelf = [&](const StringRef name, const Tensor &t) {
            logging::popops::debug("Padding {} in {} to avoid sub-word writes.",
                                   name, dnai.getPathName());

            // As we want to pad the last dimension, we might as well do that
            // with ourselves. so slice that dimension out, clone it (to avoid
            // aliasing) and then interleave it back with the original.
            const auto lastDim = t.rank() - 1;
            const auto first = t.slice(0, 1, lastDim);
            const auto firstCloned = graph.clone(first, {dnai, "padding"});

            // This handles odd grain sizes, which are not expected to be used.
            // TODO: T12998 A WriteUndef may be needed here (see T11457).
            prog.add(Copy(first, firstCloned, false, {dnai}));
            return concat({t, firstCloned}, lastDim);
          };

          tileBase = padWithSelf("baseT", tileBase);
          tileSub = padWithSelf("subT", tileSub);
          ++regionSize;
          if (scale == boost::none) {
            vertexName = templateVertex(vertexNameUntemplated,
                                        base.elementType(), false, *op);
          } else {
            vertexName =
                templateVertex(vertexNameUntemplated, base.elementType(),
                               scale->elementType(), false, *op);
          }
        }
      }
    } else {
      vertexName = templateVertex(vertexNameUntemplated, base.elementType());
    }

    generateMultiSliceVerticesOnTile(graph, cs, tile, tileBase, offsets1d,
                                     tileSub, scale, vertexName, isUpdate, 0u,
                                     baseOffset, op, false, {dnai});
  }

  if (!multiUpdateSubwordTiles.empty()) {
    logging::popops::debug("UpdateOp in {} with odd regionSize on tile(s) {}",
                           dnai.getPathName(), multiUpdateSubwordTiles);
  }

  prog.add(Execute(cs, {dnai}));

  // If this is an update and we rearranged the input, copy back to the original
  if (originalBase && isUpdate) {
    prog.add(Copy(base, *originalBase, false, {dnai}));
  }
}

static void
checkOrderingInfoConsistencyWithOptions(const bool useIndicesOrderingInfo,
                                        const bool indicesAreSorted) {
  if (useIndicesOrderingInfo && !indicesAreSorted) {
    throw poplibs_error("Cannot use indices ordering constraint when option "
                        "flags does not indictate indices are sorted");
  }
}

static void generatePlannedMultiUpdateOp(
    const std::string &vertexNameUntemplated, const SlicePlanInternal &plan,
    Graph &graph, Sequence &seq, const Tensor &offsets, Tensor base,
    Tensor slices, boost::optional<Tensor> scale, unsigned baseSlicedDim,
    boost::optional<Operation> op, const OptionFlags &optionFlags,
    const DebugNameAndId &dnai) {

  // When a two-stage update is perform we use 32bit partials
  const auto twoStagePartialType = FLOAT;
  const auto options = parseSliceOptions(optionFlags);

  if (!plan.isNull) {
    checkOrderingInfoConsistencyWithOptions(plan.useIndicesOrderingInfo,
                                            options.indicesAreSorted);
  }

  const auto csU = graph.addComputeSet({dnai, "Update"});

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

  logging::popops::debug(
      "PlannedMUOp: activeTiles={}, split {}/{}/{}, shapes {} {}", numUsedTiles,
      nonEmptyLookupSplits, slicedSplit, unslicedSplit, base.shape(),
      slices.shape());

  // There are two situations in which we choose to rearrange the slices
  // into this multi-update:
  //
  // * If the slices will be cast to a higher precision
  // we will send/receive less data over exchange if we use the lower precision
  // with fewer bytes per-element.
  // * If the slices will be multi-cast to tiles on which the multi-update
  // takes place, the cost of rearranging via exchange in receive pointers
  // will be multiplied by the factor the slices are broadcast by. The factor
  // chosen below as a threshold to rearrange before the broadcast is rather
  // arbitrary so may not be optimal.
  //
  // This is a concern for the slices because these are the ones that are
  // likely to not have the layout we expected when we planned the operation.
  //
  constexpr static unsigned slicesBroadcastDestRearrangeThreshold = 4;
  if ((multipleStages && type != twoStagePartialType) ||
      slicedSplit >= slicesBroadcastDestRearrangeThreshold) {
    const auto slicesRearranged = createSliceTensor(
        graph, type, base.shape(), slicedDim, offsets1d.numElements(), plan,
        optionFlags, {dnai, "slicesRearranged"});
    seq.add(Copy(slices, slicesRearranged, false, {dnai}));
    slices = slicesRearranged;
    logging::popops::trace(
        "PlannedMUOp: Adding copy to rearrange slices into "
        "multiUpdateOp to reduce copy vertex state/exchange code");
  }

  // First stage: update each lookupSplit into a temporary dense buffer. When
  // lookupSplit is 1 (which is typical for large base index sizes) this is the
  // only stage and the update goes directly into the base Tensor.
  Type stage0OutputType;
  Tensor slicesInput, stage0Output;
  // Scaling is applied in the update when there's a single stage, but in a
  // later add when there is an lookupSplit
  Tensor stage1Scale;
  boost::optional<Tensor> stage0Scale = boost::none;
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
    if (scale != boost::none) {
      stage0Scale = graph.addConstant(stage0OutputType, {}, 1., {dnai, "one"});
      graph.setTileMapping(*stage0Scale, 0);
      stage1Scale =
          cast(graph, *scale, stage0OutputType, seq, {dnai, "CastScale"});
    }

    // lookupSplit copies of the base tensor
    auto wantedShape = base.shape();
    wantedShape.insert(wantedShape.begin(), nonEmptyLookupSplits);

    // TODO: T12933 Consider cast after broadcasting to first stage updateOp
    // vertices to save time spent exchanging the larger data type. This may be
    // a tradeoff with temporary memory usage in order to keep a broadcasted
    // half and float copy of the slices during the cast.
    slicesInput = slices.elementType() == twoStagePartialType
                      ? slices
                      : popops::cast(graph, slices, twoStagePartialType, seq,
                                     {dnai, "CastSlices"});
    stage0Output = createPartitionableTensor(
        graph, twoStagePartialType, wantedShape,
        {nonEmptyLookupSplits, p.slicedDimSplit, p.unslicedDimSplit},
        {dnai, "gathered"});

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
        const auto tile = linearizeSliceIndices(
            p.lookupSplit, slicedSplit, unslicedSplit, lookupSplitIdx, s, u);
        // We have different specialisations for half data depending on the need
        // for subword writes
        //
        // TODO: T12934 Pad if not a multiple of grain size to ensure uniform
        // execution time of update on each tile given an uneven split.
        bool needSubwordWrites =
            target.getTypeSize(stage0OutputType) == 2 && numOffsets % 2 != 0;

        if (needSubwordWrites) {
          multiUpdateSubwordTiles.emplace_back(tile);
        }

        std::string vertexName;
        if (op == boost::none) {
          vertexName = templateVertex(vertexNameUntemplated, stage0OutputType);
        } else {
          vertexName =
              stage0Scale == boost::none
                  ? templateVertex(vertexNameUntemplated, stage0OutputType,
                                   needSubwordWrites, *op)
                  : templateVertex(vertexNameUntemplated, stage0OutputType,
                                   stage0Scale->elementType(),
                                   needSubwordWrites, *op);
        }

        logging::popops::trace("generatePlannedMultiUpdateOp: "
                               "Offsets {}/{} ({}); "
                               "BaseIdx {}/{} ({}), "
                               "SubIdx {}/{} ({}) "
                               "for indices {},{},{} "
                               "on tile {}",
                               beginOffset, endOffset, unslicedDim,
                               beginBaseIdx, endBaseIdx, baseSlicedDim,
                               beginSubIdx, endSubIdx, subSlicedDim,
                               lookupSplitIdx, s, u, tile);

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
        generateMultiSliceVerticesOnTile(graph, csU, tile, tileBase, indices,
                                         tileSlice, stage0Scale, vertexName,
                                         true, baseSlicedDim, baseOffset, op,
                                         plan.useIndicesOrderingInfo, {dnai});
      }
    }
  }

  if (!multiUpdateSubwordTiles.empty()) {
    logging::popops::debug("UpdateOp in {} with odd regionSize on tile(s) {}",
                           dnai.getPathName(), multiUpdateSubwordTiles);
  }

  if (multipleStages) {
    // Reduce dense partials
    zero(graph, stage0Output, seq, {dnai, "zeroPartials"});
    seq.add(Execute(csU, {dnai}));

    const auto cumulativeUpdate =
        graph.clone(twoStagePartialType, base, {dnai, "opUpdates"});

    // Given we know that partials for a set of columns on each tile are always
    // contiguous in the same way, we can use our knowledge to reorder the
    // columns and make the reduction library's job easier. This could go
    // away once T15113 is done.
    std::vector<Tensor> stage0OutputReordered, cumulativeUpdateReordered;
    iterateTensorPartitions(
        stage0Output, {1, slicedSplit, unslicedSplit},
        [&](const std::vector<std::size_t> &, const Tensor &s) {
          stage0OutputReordered.emplace_back(s.flatten(1, 3));
        });
    iterateTensorPartitions(
        cumulativeUpdate, {slicedSplit, unslicedSplit},
        [&](const std::vector<std::size_t> &, const Tensor &s) {
          cumulativeUpdateReordered.emplace_back(s.flatten());
        });

    reduceWithOutput(graph, concat(stage0OutputReordered, 1u),
                     concat(cumulativeUpdateReordered), {0}, {*op}, seq,
                     {dnai, "Reduce"});

    // Add the sum of the partials to the base tensor
    bool baseCastRequired = base.elementType() != twoStagePartialType;
    const Tensor addDst = [&] {
      if (baseCastRequired) {
        return cast(graph, base, twoStagePartialType, seq, {dnai, "castBase"});
      } else {
        return base;
      }
    }();

    if (*op == Operation::ADD) {
      scaledAddTo(graph, addDst, cumulativeUpdate, stage1Scale, seq,
                  {dnai, "Add"});
    } else if (*op == Operation::MAX) {
      maxInPlace(graph, addDst, cumulativeUpdate, seq, {dnai, "Max"});
    } else {
      const std::string opName = asString(*op);
      throw poplibs_error("Unsupported multiUpdate operation" + opName);
    }

    // cast the final result back into base; when !castBase the addTo was
    // directly into base anyway
    if (baseCastRequired) {
      seq.add(cast(graph, addDst, base, {dnai, "castBack"}));
    }
  } else {
    seq.add(Execute(csU, {dnai}));
  }
}

/** Copy the sub-tensor acquired by indexing 't' at position 'offset' in
 * dimension 'dim' to 's'. The other output dimensions will match the size of
 * the corresponding input dimensions.
 *
 * \param graph           The poplar graph
 * \param s               The destination tensor
 * \param t               The source tensor
 * \param offset          The offset in \a's \a dim dimension. This tensor must
 *                        have a single element, or an element per tile
 * \param dim             The dimension to slice
 * \param numOutIndices   The size of the output Tensor in the sliced dimension
 * \param prog            Program to be updated.
 * \param dnai            The debug reference
 */
static void sliceWithOutput(Graph &graph, const Tensor &s, const Tensor &t,
                            const Tensor &offset, unsigned dim,
                            unsigned numOutIndices, Sequence &prog,
                            const DebugNameAndId &dnai) {
  const unsigned numInIndices = t.dim(dim);
  assert(dim < t.rank());
  assert(numOutIndices <= t.dim(dim));
  // Get a 2d view of the source tensor, with the dim we're slicing at dim0
  // and the other dimensions collapsed into dim1
  Tensor t2d =
      t.dimRoll(dim).reshape({numInIndices, t.numElements() / numInIndices});

  Tensor s2d =
      s.dimRoll(dim).reshape({numOutIndices, s.numElements() / numOutIndices});

  generateVertices("popops::DynamicSlice", graph, prog, offset, t2d, s2d,
                   {dnai, "slice"});
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
 * \param prog            Optional program to be updated. If no program given
 *                        then vertices are not generated
 * \param dnai            The debug reference
 * \returns               The specified subtensor
 */
static Tensor slice(Graph &graph, const Tensor &t,
                    const boost::optional<Tensor> &offset, unsigned dim,
                    unsigned numOutIndices, boost::optional<Sequence &> prog,
                    const DebugNameAndId &dnai) {
  assert(dim < t.rank());
  assert(numOutIndices <= t.dim(dim));

  Tensor s = graph.clone(t.slice(0, numOutIndices, dim),
                         {dnai, std::string("sliced_") + std::to_string(dim)});

  if (prog && offset) {
    sliceWithOutput(graph, s, t, offset.get(), dim, numOutIndices, prog.get(),
                    dnai);
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
 *  \param dnai            The debug reference
 **/
static void update(Graph &graph, const Tensor &t, const Tensor &s,
                   const Tensor &offset, unsigned dim,
                   poplar::program::Sequence &prog,
                   const DebugNameAndId &dnai) {
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
                   {dnai, "update"});
}

/// If we are slicing up a tensor with the given `shape` in the dimensions
/// `dims`, and the slice size in each dimension is `sizes`, this determines
/// what is the best order to do the slices that reduces the elements in the
/// resulting slice fastest. The returned vector contains indexes into
/// `dims` (and `sizes`).
/// Example 1:
///  shape = {10, 20, 30}; dims = {1,2}; sizes = {1, 1}.
///  return order is {1, 0}, which means slice along dim 2 then dim 1.
/// Example 2:
///  shape = {10, 20, 30}; dims = {1,2}; sizes = {2, 5}.
///  return order is {0, 1}, i.e. slice along dim 1 then dim 2.
static std::vector<size_t>
bestSliceOrder(const std::vector<std::size_t> &shape,
               const std::vector<std::size_t> &dims,
               const std::vector<std::size_t> &sizes) {
  assert(dims.size() <= shape.size());
  assert(dims.size() == sizes.size());

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
    const SlicePlanInternal &p, const std::vector<std::size_t> &shape,
    const std::vector<std::size_t> &dims, const std::vector<std::size_t> &sizes,
    const std::string &callerDebugStr) {
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
                           const boost::optional<Tensor> &offset,
                           const std::vector<std::size_t> &dims,
                           const std::vector<std::size_t> &sizesOrSlices,
                           bool checkSizes = true,
                           bool sizesAreSlices = false) {
  if (!plan.getImpl().isNull) {
    validatePlanForGivenParameters(plan.getImpl(), shape, dims, sizesOrSlices,
                                   name);
  }
  auto tRank = shape.size();
  std::string exceptionStr;
  std::string sizesStr = sizesAreSlices ? "numSlices" : "sizes";
  if (offset) {
    auto offsetElems = offset.get().rank() == 0 ? 0 : offset.get().dim(0);
    if (offset.get().rank() > 2 || offsetElems != dims.size())
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

static void validateParamsWithOutput(
    std::string name, const SlicePlan &plan, const OptionFlags &options,
    const std::vector<std::size_t> &outputShape,
    const std::vector<std::size_t> &shape,
    const boost::optional<Tensor> &offset, const std::vector<std::size_t> &dims,
    const std::vector<std::size_t> &sizesOrSlices) {

  // Check the output rank matches.
  if (outputShape.size() != shape.size()) {
    throw graph_connection_error(
        fmt::format("{} output rank (shape=[{}]) does not match the input rank "
                    "(shape=[{}])",
                    name, outputShape, shape));
  }

  // Check the output non-sliced dimensions matches the input.
  std::unordered_set<std::size_t> dims_set(dims.begin(), dims.end());
  for (std::size_t dim = 0u; dim < shape.size(); ++dim) {
    if (!dims_set.count(dim) && shape[dim] != outputShape[dim]) {
      throw graph_connection_error(
          fmt::format("{} output non-sliced dimension {} (shape=[{}]) does not "
                      "match the input dimension (shape=[{}])",
                      name, dim, outputShape, shape));
    }
  }

  // Check the output sliced dimensions matches the slice sizes.
  for (std::size_t dim = 0u; dim < dims.size(); ++dim) {
    if (outputShape[dims[dim]] != sizesOrSlices[dim]) {
      throw graph_connection_error(fmt::format(
          "{} output dimension {} (shape=[{}]) does not match the slice count "
          "{} (sizes=[{}])",
          name, dims[dim], outputShape, sizesOrSlices[dim], sizesOrSlices));
    }
  }

  validateParams(name, plan, options, shape, offset, dims, sizesOrSlices);
}

// Create and map a tensor so that dynamic slicing of it will not require
// exchange.
// The underlying variables will be [U/N][S0]..[Sn][N] where
// N is the number of contiguous unsliced elements per tile and
// U is the product of the unsliced dimensions.
// This distributes the input/output slice across U/N tiles.
// S0-Sn are the sliced dimensions, sorted to optimise the number of copies.
// Typically two variables are used; the second variable for the final
// tile, which may have a different N.
// If U/N << numTiles an outer stage can be added to convert part of an
// S dimension to an extra U dimensions
//
// \param shape    Shape of the tensor to create and return.
// \param dims     Dimensions to slice in.
// \param idxOrder Order in which to slice the dimensions dim.
// \returns tensor requiring no exchange on slicing.
// Example:
//   shape = {10, 20, 30, 40}; dims = {1, 2}; idxOrder = {1, 0}
//   Then createShape = {30, 20, 400}, and nPartitions = {1, 1, 400}
static Tensor createSliceableTensorGivenOrder(
    poplar::Graph &graph, const poplar::Type &type,
    const std::vector<std::size_t> &shape, const std::vector<std::size_t> &dims,
    const std::vector<std::size_t> &idxOrder, std::size_t minGrainSize,
    const DebugNameAndId &dnai) {
  // Return a linearly mapped tensor if no slice dim is specified.
  // Or return an EmptyTensor if shape has a zero dimension.
  bool noOutputElements = std::any_of(shape.begin(), shape.end(),
                                      [](std::size_t n) { return n == 0; });
  if (dims.size() == 0 || noOutputElements) {
    auto t = graph.addVariable(type, shape, {dnai});
    mapTensorLinearly(graph, t);
    return t;
  }

  std::vector<bool> dimIsSliced(shape.size(), false);
  std::vector<unsigned> inversePermutation(shape.size());
  std::vector<std::size_t> createShape;
  createShape.reserve(dims.size() + 1);

  // First, put together createShape for the sliced dimensions.
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
    createShape.push_back(shape[d]);
  }

  // Second, add a last dimension to createShape which contains
  // sum of all elements in unsliced dimensions.
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
  auto unslicedElemsPerSplit =
      std::max(ceildiv(numUnslicedElems, numTiles), minGrainSize);

  if (createShape.size() == 2 && minGrainSize > 1) {
    // If we're slicing individaul elements coerce the slices to be atomic write
    // aligned if that matches the specified minGrainSize
    const auto &target = graph.getTarget();
    const auto bytesPerElement = target.getTypeSize(type);
    const auto granularity = target.getAtomicStoreGranularity();

    if (granularity > bytesPerElement &&
        minGrainSize * bytesPerElement >= granularity) {
      const auto elemPerAtom = granularity / bytesPerElement;
      const auto unrounded = unslicedElemsPerSplit;
      auto remainder = unslicedElemsPerSplit % elemPerAtom;
      if (remainder != 0) {
        unslicedElemsPerSplit += elemPerAtom - remainder;
        logging::popops::trace("createSliceableTensor rounded {} to {}",
                               unrounded, unslicedElemsPerSplit);
      }
    }
  }

  const auto tilesUsed = ceildiv(numUnslicedElems, unslicedElemsPerSplit);
  std::vector<std::size_t> nPartitions(createShape.size(), 1);
  nPartitions.back() = tilesUsed;

  auto t = createPartitionableTensor(graph, type, createShape, nPartitions,
                                     {dnai, "sliceable"});

  // Distribute over tiles starting from 0.
  unsigned tile = 0;
  iterateTensorPartitions(
      t, nPartitions, [&](const std::vector<std::size_t> &, const Tensor &s) {
        graph.setTileMapping(s, tile++);
      });
  t = t.reshapePartial(t.rank() - 1, t.rank(), unslicedShape)
          .dimShuffle(inversePermutation);

  logging::popops::debug("createSliceableTensor {}, minGrainSize {}, dims {}, "
                         "used tiles {}, "
                         "elem size {}, "
                         "{} tiles with {} elems, "
                         "{} tiles with {} elems",
                         t.shape(), minGrainSize, dims, tilesUsed,
                         graph.getTarget().getTypeSize(type),
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
                                    const DebugNameAndId &dnai) {
  assert(!shape.empty() && "cannot createSliceableTensor from empty shape");
  assert(slicedDim < shape.size() && "slice dim cannot exceed rank of tensor");
  std::vector<bool> dimIsSliced(shape.size(), false);
  dimIsSliced[slicedDim] = true;

  std::vector<unsigned> unslicedDims;
  std::vector<std::size_t> unslicedShape;
  unslicedDims.reserve(shape.size() - 1);
  unslicedShape.reserve(shape.size() - 1);

  // Get and store unslicedDims, unslicedShape info.
  for (unsigned d = 0; d < shape.size(); ++d) {
    if (!dimIsSliced[d]) {
      unslicedDims.push_back(d);
      unslicedShape.push_back(shape[d]);
    }
  }

  // Product of size of each unsliced dimension.
  const auto totalUnslicedElems =
      std::accumulate(unslicedShape.begin(), unslicedShape.end(),
                      std::size_t(1), std::multiplies<std::size_t>());

  std::vector<std::size_t> createShape = {shape[slicedDim], totalUnslicedElems};
  std::vector<std::size_t> createSplits = {plan.partition.slicedDimSplit,
                                           plan.partition.unslicedDimSplit};

  // Get 't' such that slice of 't' corresponding to each partition
  // in 'createSplits' is a single contiguous region.
  auto t =
      createPartitionableTensor(graph, type, createShape, createSplits, {dnai});

  // If there is an indices split we will broadcast each
  // contiguous chunk of the tensor between tiles while
  // respecting grain size.
  const auto grainSize = plan.partition.unslicedGrainSize;
  const auto extraSplit = plan.partition.lookupSplit;
  // Use an extra grain size when spreading elements in
  // a partition of sliced and unsliced dimensions among
  // partitions of the lookup dimension to try and
  // take advantage of double-width exchange when possible.
  const auto &target = graph.getTarget();
  const auto bytesPerElem = target.getTypeSize(type);
  const std::size_t exchangeBusShareAtomBytes =
      target.getExchangeBytesPerCycle() * target.getTilesPerSharedExchangeBus();
  assert(exchangeBusShareAtomBytes % bytesPerElem == 0);
  const auto elemsPerExchangeBusShareAtom =
      exchangeBusShareAtomBytes / bytesPerElem;
  const auto extraGrainSize = lcm(grainSize, elemsPerExchangeBusShareAtom);
  iterateTensorPartitions(
      t, createSplits,
      [&](const std::vector<std::size_t> &i, const Tensor &tSlice) {
        // We flatten all but the grain size from the plan and distribute this
        // between tiles that use these elements.
        const auto flattenedSlice = tSlice.flatten();
        const auto sliceNumElems = flattenedSlice.numElements();

        const auto sliceNumGrains = ceildiv(sliceNumElems, extraGrainSize);
        const auto grainsPerSplit = ceildiv(sliceNumGrains, extraSplit);
        const auto elemsPerSplit = grainsPerSplit * extraGrainSize;

        assert(i.size() == 2);
        const std::size_t slicedIdx = i.front();
        const std::size_t unslicedIdx = i.back();

        for (std::size_t indexIdx = 0; indexIdx < plan.partition.lookupSplit;
             ++indexIdx) {
          unsigned tile = linearizeSliceIndices(
              plan.partition.lookupSplit, plan.partition.slicedDimSplit,
              plan.partition.unslicedDimSplit, indexIdx, slicedIdx,
              unslicedIdx);
          const auto begin = std::min(sliceNumElems, indexIdx * elemsPerSplit);
          const auto end =
              std::min(sliceNumElems, (indexIdx + 1) * elemsPerSplit);
          graph.setTileMapping(flattenedSlice.slice(begin, end), tile);
        }
      });

  std::vector<unsigned> inversePermutation(shape.size());
  inversePermutation[slicedDim] = 0;

  // Unsliced dimensions (starting from dims.size() which is always 1).
  for (std::size_t d = 0; d < unslicedDims.size(); ++d) {
    inversePermutation[unslicedDims[d]] = 1 + d;
  }
  // Give expected shape and order of dimensions to the returned tensor.
  return t.reshapePartial(1, 2, unslicedShape).dimShuffle(inversePermutation);
}

// Create and map a tensor so that dynamic slicing of it will not require
// exchange.
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
                             const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(type, shape, dims, sizes, minGrainSize));

  logging::popops::info("createSliceableTensor/NoPlan for {} / {} / {}", shape,
                        dims, sizes);
  validateParams("createSliceableTensor", {}, {}, shape, boost::none, dims,
                 sizes, true);
  const auto idxOrder = bestSliceOrder(shape, dims, sizes);
  std::string tName = "sliceable";
  std::string sep = "";
  for (const auto &d : shape) {
    tName += sep + std::to_string(d);
    sep = "x";
  }
  auto output = createSliceableTensorGivenOrder(
      graph, type, shape, dims, idxOrder, minGrainSize, {di, tName});
  di.addOutput(output);
  return output;
}

Tensor createSliceableTensor(Graph &graph, const Type &type,
                             const std::vector<std::size_t> &shape,
                             const std::vector<std::size_t> &dims,
                             const std::vector<std::size_t> &sizes,
                             const SlicePlan &plan, const OptionFlags &options,
                             const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(type, shape, dims, sizes, plan, options));

  logging::popops::info("createSliceableTensor for {} / {} / {}; nullplan? {}",
                        shape, dims, sizes, plan.getImpl().isNull);
  if (plan.getImpl().isNull) {
    return createSliceableTensor(graph, type, shape, dims, sizes, 0, {di});
  }
  validateParams("createSliceableTensor", {}, {}, shape, boost::none, dims,
                 sizes, true);
  // For now we don't plan anything which slices more than one dimension or
  // more than a single slice.
  assert(dims.size() == 1);
  assert(sizes.size() == 1 && sizes[0] == 1);
  auto output = createSliceableTensor(graph, type, shape, dims[0],
                                      plan.getImpl(), options, {di});
  di.addOutput(output);
  return output;
}

static Tensor createSliceTensor(Graph &graph, const poplar::Type &type,
                                const std::vector<std::size_t> &inputShape,
                                const std::vector<std::size_t> &dims,
                                const std::vector<std::size_t> &sizes,
                                const std::size_t numUpdates,
                                const DebugNameAndId &dnai) {
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
      {dnai, std::string("slices") + std::to_string(numUpdates)});
}

static Tensor createSliceTensor(Graph &graph, const Type &type,
                                const std::vector<std::size_t> &shape,
                                const std::size_t slicedDim,
                                const std::size_t numIndices,
                                const SlicePlanInternal &plan,
                                const OptionFlags &options,
                                const DebugNameAndId &dnai) {
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

  auto t =
      createPartitionableTensor(graph, type, createShape, createSplits, {dnai});

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
              plan.partition.lookupSplit, plan.partition.slicedDimSplit,
              plan.partition.unslicedDimSplit, indexIdx, slicedIdx,
              unslicedIdx);
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
                         const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(type, shape, dims, sizes, numIndices, plan, options));
  validateParams("createSliceTensor", plan, options, shape, {}, dims, sizes,
                 false);
  const auto &p = plan.getImpl();
  Tensor output;
  if (p.isNull) {
    output =
        createSliceTensor(graph, type, shape, dims, sizes, numIndices, {di});
  } else {
    // We don't plan anything which slices more than one dimension for now or
    // more than a single slice.
    assert(dims.size() == 1);
    assert(sizes.size() == 1 && sizes[0] == 1);
    output = createSliceTensor(graph, type, shape, dims[0], numIndices, p,
                               options, {di});
  }
  di.addOutput(output);
  return output;
}

Tensor createSliceTensor(Graph &graph, const Tensor &t,
                         const std::vector<std::size_t> &dims,
                         const std::vector<std::size_t> &sizes,
                         const std::size_t numIndices,
                         const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(t, dims, sizes, numIndices));

  validateParams("createSliceTensor", {}, {}, t.shape(), boost::none, dims,
                 sizes);
  // Special case for 1 index, we just clone the input tensor's first slice.
  if (numIndices == 1) {
    std::string name = "slice";
    Tensor s = t;
    // When updating a single slice map the update tensor with the mapping
    // of the first slice of the base tensor
    for (unsigned i = 0; i != dims.size(); ++i) {
      s = s.slice(0, sizes[i], dims[i]);
      name = name + "_d" + std::to_string(dims[i]);
    }
    auto mapping = graph.getTileMapping(s);
    s = graph.clone(s, {di, name});
    graph.setTileMapping(s, mapping);
    return s.expand({0});
  }
  auto output = createSliceTensor(graph, t.elementType(), t.shape(), dims,
                                  sizes, numIndices, {di});
  di.addOutput(output);
  return output;
}

poplar::Tensor createIndicesTensor(Graph &graph,
                                   const std::vector<std::size_t> &dims,
                                   const std::size_t numIndices,
                                   const SlicePlan & /* plan */,
                                   const OptionFlags & /* options */,
                                   const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(dims, numIndices));

  logging::popops::info("createIndicesTensor for {} / {}", numIndices, dims);
  const auto indices =
      graph.addVariable(UNSIGNED_INT, {numIndices, dims.size()}, {di});
  mapTensorLinearly(graph, indices, minIndicesPerTile, 1);
  di.addOutput(indices);
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

Tensor
createSliceableTensorFromSlice(Graph &graph, const Tensor &s,
                               const std::vector<std::size_t> &dims,
                               const std::vector<std::size_t> &numSlices,
                               const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(s, dims, numSlices));

  validateParams("createSliceableTensorFromSlice", {}, {}, s.shape(),
                 boost::none, dims, numSlices, true, true);

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

  const auto totalNumSlices =
      std::accumulate(numSlices.begin(), numSlices.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  auto t = graph.cloneN(
      s, totalNumSlices, {di},
      TensorCloneMethod::GATHER_AND_PRESERVE_TILE_ORDER_AND_ALIASES,
      TensorCloneDuplicationMethod::DUPLICATE_BY_TILE_CONTIGUOUS_REGION);

  t = t.reshape(createShape);

  for (std::size_t i = 0; i < dims.size(); ++i) {
    const auto dim = dims.size() - i + dims[idxOrder[i]];
    t = t.dimRoll(0, dim - 1).flatten(dim - 1, dim + 1);
  }
  assert(t.shape() == sliceableShape);
  di.addOutput(t);
  return t;
}

static void dynamicSliceWithOutputImpl(Graph &graph, const Tensor &out,
                                       const Tensor &t, const Tensor &offset,
                                       const std::vector<std::size_t> &dims,
                                       const std::vector<std::size_t> &sizes,
                                       Sequence &prog,
                                       const DebugNameAndId &dnai) {
  logging::popops::info(
      "dynamicSlice out={}, t={}, offset={}, dims={}, sizes={}, name={}",
      out.shape(), t.shape(), offset.shape(), dims, sizes, dnai.getPathName());

  validateParamsWithOutput("dynamicSlice", {}, {}, out.shape(), t.shape(),
                           offset, dims, sizes);

  for (unsigned i = 0; i != dims.size(); ++i) {
    if (sizes[i] == 0) {
      // Since one of the slice sizes is zero, the resulting tensor has no
      // elements. We can return immediately.
      return;
    }
  }
  Tensor temp = t;

  auto idxOrder = bestSliceOrder(t.shape(), dims, sizes);

  // Extract the last slice. This is used to slice into the output tensor
  auto last = idxOrder.back();
  idxOrder.pop_back();

  for (auto i : idxOrder) {
    // don't care about offset if vertices are not mapped as we are only
    // interested in the mapping
    temp =
        slice(graph, temp, offset[i], dims[i], sizes[i], prog,
              {dnai, std::string("dynamicSlice_d") + std::to_string(dims[i])});
  }

  // Apply the final slice into the output tensor.
  sliceWithOutput(
      graph, out, temp, offset[last], dims[last], sizes[last], prog,
      {dnai, std::string("dynamicSlice_d") + std::to_string(dims[last])});
}

static Tensor dynamicSliceImpl(Graph &graph, const Tensor &t,
                               const boost::optional<Tensor> &offset,
                               const std::vector<std::size_t> &dims,
                               const std::vector<std::size_t> &sizes,
                               boost::optional<Sequence &> prog,
                               const DebugNameAndId &dnai) {
  bool checkOffset = offset != boost::none;
  if (checkOffset) {
    logging::popops::info(
        "dynamicSlice t={}, offset={}, dims={}, sizes={}, name={}", t.shape(),
        offset.get().shape(), dims, sizes, dnai.getPathName());
  } else {
    logging::popops::info("dynamicSlice t={}, dims={}, sizes={}, name={}",
                          t.shape(), dims, sizes, dnai.getPathName());
  }

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
    out = slice(
        graph, out, checkOffset ? offset.get()[i] : offset, dims[i], sizes[i],
        prog, {dnai, std::string("dynamicSlice_d") + std::to_string(dims[i])});
  }

  return out;
}

Tensor dynamicSlice(Graph &graph, const Tensor &t, const Tensor &offset,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes,
                    poplar::program::Sequence &prog,
                    const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(t, offset, dims, sizes));
  auto output = dynamicSliceImpl(graph, t, offset, dims, sizes, prog, {di});
  di.addOutput(output);
  return output;
}

void dynamicSliceWithOutput(poplar::Graph &graph, const poplar::Tensor &output,
                            const poplar::Tensor &t,
                            const poplar::Tensor &offset,
                            const std::vector<std::size_t> &dims,
                            const std::vector<std::size_t> &sizes,
                            poplar::program::Sequence &prog,
                            const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(output, t, offset, dims, sizes));
  dynamicSliceWithOutputImpl(graph, output, t, offset, dims, sizes, prog, {di});
}

Graph::TileToTensorMapping
getSliceMapping(poplar::Graph &graph, const poplar::Tensor &t,
                const std::vector<std::size_t> &dims,
                const std::vector<std::size_t> &sizes) {
  auto sliceT =
      dynamicSliceImpl(graph, t, boost::none, dims, sizes, boost::none, "");
  return graph.getTileMapping(sliceT);
}

void dynamicUpdate(Graph &graph, const Tensor &t, const Tensor &s,
                   const Tensor &offset, const std::vector<std::size_t> &dims,
                   const std::vector<std::size_t> &sizes,
                   poplar::program::Sequence &prog,
                   const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(t, s, offset, dims, sizes));
  logging::popops::info(
      "dynamicUpdate t={}, s={}, offset={}, dims={}, sizes={}, name={}",
      t.shape(), s.shape(), offset.shape(), dims, sizes,
      debugContext.getPathName());

  validateParams("dynamicUpdate", {}, {}, t.shape(), offset, dims, sizes);

  // empty sizes or dimensions are full update (TF does this)
  if (dims.size() == 0) {
    prog.add(Copy(s, t, false, {di}));
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
  reducedT.reserve(idxOrder.size() + 1);
  reducedT.emplace_back(t); // reducedT[0] = t
  // slice off the larger dimensions one at a time
  for (unsigned i = 0; i != idxOrder.size() - 1; ++i) {
    auto dim = idxOrder[i];
    reducedT.emplace_back(
        slice(graph, reducedT[i], offset[dim], dims[dim], sizes[dim], prog,
              {di, std::string("dynamicUpdateS_d") + std::to_string(dims[i])}));
  }
  // copy s into the reduced t, iterating back to full dimensions
  reducedT.emplace_back(s);
  for (unsigned ii = idxOrder.size(); ii != 0; --ii) {
    auto i = ii - 1;
    auto dsIdx = idxOrder[i]; // index into dims[] and sizes[]
    update(graph, reducedT[i], reducedT[i + 1], offset[dsIdx], dims[dsIdx],
           prog,
           {di, std::string("dynamicUpdateU_d") + std::to_string(dims[dsIdx])});
  }
}

// Implementation of multiSlice with a non-null plan
static void multiSlicePlanned(Graph &graph, const Tensor &t,
                              const Tensor &offset, const Tensor &slice,
                              const std::vector<std::size_t> &dims,
                              const std::vector<std::size_t> &sizes,
                              Sequence &prog, const SlicePlanInternal &p,
                              const OptionFlags &optionFlags,
                              const DebugNameAndId &dnai) {
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
  const auto options = parseSliceOptions(optionFlags);

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
                                       nPartitions, {dnai, "stage0Output"});
    }
    return slice.expand({0});
  }();

  const std::string vertexClass =
      templateVertex("popops::MultiSlice", t.elementType());
  const auto cs1 = graph.addComputeSet({dnai, "stage0"});
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
            p.partition.lookupSplit, p.partition.slicedDimSplit,
            p.partition.unslicedDimSplit, i, s, h);
        const auto hBegin = std::min(hTotalElems, h * hElemsPerPartition);
        const auto hEnd = std::min(hTotalElems, (h + 1) * hElemsPerPartition);
        if (hEnd - hBegin == 0) {
          break;
        }
        const Tensor indices = iSplitByI.squeeze({1});
        const Tensor input = tSplitByS.slice(hBegin, hEnd, unslicedDim);
        const Tensor output = sSplitByS.slice(hBegin, hEnd, 1 + unslicedDim);
        graph.setTileMapping(output, tile);
        generateMultiSliceVerticesOnTile(
            graph, cs1, tile, input, indices, output, boost::none, vertexClass,
            false, slicedDim, baseOffset, boost::none, options.indicesAreSorted,
            {dnai});
      }
    }
  }
  prog.add(Execute(cs1, {dnai}));

  // Reduce remaining partials in a second compute set.
  if (sSplit > 1) {
    const auto cs2 = graph.addComputeSet({dnai, "Stage1"});

    // A split of the sliced dimension in the first stage produces
    // iTotalElems * sSplit results, only iTotalElems of which are
    // the desired final results.
    //
    // We perform a second slice with the indices modified in order
    // to pick the correct results. We spread this second slice over
    // tiles by splitting iTotalElems further by sSplit (as the original
    // sliced dimension no longer exists to spread over tiles).
    //
    // Each partition in the second stage has some number
    // ceildiv(iTotalElems, iSplit) of slices that we further split
    // using sSplit.
    //
    // This means each tile has a piece of the output of stage 0
    // like:
    //
    //   {sSplit, iElemsThisPartition, hElemsThisPartition}
    //
    // Where we need to select the correct index from the
    // outer-most dimension.
    //
    // We do this by transforming the indices each partition in this
    // second stage processes as follows:
    //
    // for (i in range(0, iElemsThisPartition)) {
    //   // Offset into iElemsPerPartition to pick out
    //   stage1Indices[i] = i;
    //   sPartitionWithCorrectAnswer = stage0Indices[i] / sElemsPerPartition;
    //   stage1Indices[i] += sPartitionWithCorrectAnswer * iElemsThisPartition.
    // }
    //
    const auto iElemsPerPartitionStage1 = ceildiv(iElemsPerPartition, sSplit);
    const auto iSplitStage1 =
        ceildiv(iElemsPerPartition, iElemsPerPartitionStage1);

    const Tensor transformedOffset = [&] {
      // innerIdx represents the offset into the indices in the partition
      // on each tile and hence is just an ascending sequence of integers.
      const Tensor innerIdx = [&] {
        Tensor t =
            graph.clone(offset.slice(0, iElemsPerPartitionStage1), {dnai});
        iota(graph, t.squeeze({1}), 0u, prog, {dnai});
        t = t.broadcast(iSplitStage1, 0)
                .slice(0, iElemsPerPartition)
                .broadcast(iSplit, 0)
                .slice(0, iTotalElems);
        return t;
      }();

      // innerElems represents the number of indices this partition processes.
      // There are at most 3 different numbers of indices processed by a
      // partition, e.g. the following example:
      //
      // +                                                                   +
      // |                                                                   |
      // +-------------------------------------------------------------------+
      // |                           iTotalElems=9                           |
      // +                                                                   +
      //
      //
      //
      // +                                                                   +
      // |                                       +                           |
      // +-------------------------------------------------------------------+
      // |                   5                   +              4            |
      // +                                                                   +
      //                                       iSplit=2
      //
      //
      //
      // +                                                                   +
      // |                      +                +                     +     |
      // +-------------------------------------------------------------------+
      // |           3          +        2       +           3         +  1  |
      // +                                                                   +
      //                    sSplit=2                               sSplit=2

      const Tensor innerElems = [&] {
        const auto ceil0 = ceildiv(iTotalElems, iSplit);
        const auto rem0 = iTotalElems % ceil0;
        const auto ceil0And1 = ceildiv(ceil0, sSplit);
        const auto ceil0AndRem1 = ceil0 % ceil0And1;
        const auto rem0And1 = rem0 % ceil0And1;

        const auto nCeil0And1 = roundDown(ceil0, ceil0And1);
        const auto nCeil0AndRem1 = ceil0 - nCeil0And1;
        const auto nCeil0 = floordiv(iTotalElems, ceil0);
        const auto nRem0AndCeil1 = roundDown(rem0, ceil0And1);
        const auto nRem0And1 = rem0 - nRem0AndCeil1;

        const auto tCeil0And1 =
            graph.addConstant(UNSIGNED_INT, {1}, ceil0And1, {dnai});
        const auto tCeil0AndRem1 =
            graph.addConstant(UNSIGNED_INT, {1}, ceil0AndRem1, {dnai});
        const auto tRem0And1 =
            graph.addConstant(UNSIGNED_INT, {1}, rem0And1, {dnai});
        graph.setTileMapping(tCeil0And1, 0);
        graph.setTileMapping(tCeil0AndRem1, 0);
        graph.setTileMapping(tRem0And1, 0);

        return concat(
                   // Evenly split part
                   concat(tCeil0And1.broadcast(nCeil0And1, 0),
                          tCeil0AndRem1.broadcast(nCeil0AndRem1, 0), 0)
                       .broadcast(nCeil0, 0),
                   // Remainder
                   concat(tCeil0And1.broadcast(nRem0AndCeil1, 0),
                          tRem0And1.broadcast(nRem0And1, 0), 0))
            .expand({1});
      }();

      using namespace expr;
      return map(graph, _2 + ((_1 / sElemsPerPartition) * _3),
                 {offset, innerIdx, innerElems}, prog,
                 {dnai, "adjustedIndicesStage1"});
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
          unsigned tile = linearizeSliceIndices(
              p.partition.lookupSplit, p.partition.slicedDimSplit,
              p.partition.unslicedDimSplit, i, s, h);
          const Tensor indices = iSplitByS.squeeze({1});
          const Tensor input = tSplitByS.slice(hBegin, hEnd, 1);
          const Tensor output = sSplitByS.slice(hBegin, hEnd, 2);
          generateMultiSliceVerticesOnTile(
              graph, cs2, tile, input, indices, output, boost::none,
              vertexClass, false, slicedDim, 0, boost::none, false, {dnai});
        }
      }
    }
    prog.add(Execute(cs2, {dnai}));
  }
}

Tensor multiSlice(Graph &graph, const Tensor &t, const Tensor &offset,
                  const std::vector<std::size_t> &dims,
                  const std::vector<std::size_t> &sizes, Sequence &prog,
                  const SlicePlan &plan, const OptionFlags &options,
                  const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(t, offset, dims, sizes, plan, options));

  // small number of slices are instantiated individually
  // large number of slices are sliced by a specialisation or in a loop
  std::string dName = "multiSlice";

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
    sMulti =
        createSliceTensor(graph, t, dims, sizes, offset.dim(0), {di, dName});
  } else {
    sMulti = createSliceTensor(graph, t.elementType(), t.shape(), dims, sizes,
                               offset.dim(0), plan, options, {di, dName});
  }

  logging::popops::info("multiSlice {} -> {}, name={}, nullplan?={}", t.shape(),
                        sMulti.shape(), debugContext.getPathName(),
                        plan.getImpl().isNull);

  if (!plan.getImpl().isNull) {
    multiSlicePlanned(graph, t, offset, sMulti, dims, sizes, prog,
                      plan.getImpl(), options, {di, dName});
    di.addOutput(sMulti);
    return sMulti;
  }

  // When there are only a few slices the looping code can be larger than
  // instantiating multiple vertices
  constexpr unsigned inliningThreshold = 3;
  if (offset.dim(0) <= inliningThreshold) {
    for (unsigned slice = 0; slice != offset.dim(0); ++slice) {
      auto s = dynamicSlice(graph, t, offset[slice], dims, sizes, prog,
                            {di, dName + "/" + std::to_string(slice)});
      prog.add(Copy(s, sMulti[slice], false, {di}));
    }
    di.addOutput(sMulti);
    return sMulti;
  }

  // When there are many offsets of single slices there is a fast vertex.
  // For now only 1d slices of 2d base tensors are supported.
  if (t.rank() == 2 && dims.size() == 1 && sMulti.rank() == 3 &&
      offset.rank() == 2 && offset.dim(1) == 1 && offset.dim(0) > 6) {
    generateMultiSliceVertices("popops::MultiSlice", false, boost::none, graph,
                               prog, offset, t, sMulti, boost::none, dims[0],
                               boost::none, options, {di, dName});
    di.addOutput(sMulti);
    return sMulti;
  }

  // looping case

  prog.add(popops::countedLoop(
      graph, offset.dim(0),
      [&](poplar::Tensor sIdx) {
        Sequence body({}, {di});
        auto tIdx = dynamicSlice(graph, offset, sIdx, {0}, {1}, body,
                                 {di, dName + "/sliceIndex"})
                        .squeeze({0});

        auto sI = dynamicSlice(graph, t, tIdx, dims, sizes, body,
                               {di, dName + "/slice"})
                      .expand({0});
        dynamicUpdate(graph, sMulti, sI, sIdx, {0}, {1}, body,
                      {di, dName + "/update"});
        return body;
      },
      {di, dName + "/loop"}));

  di.addOutput(sMulti);
  return sMulti;
}

static void multiUpdateOp(Graph &graph, const Tensor &t, const Tensor &sMulti,
                          const Tensor &offset, boost::optional<Tensor> scale,
                          const std::vector<std::size_t> &dims,
                          const std::vector<std::size_t> &sizes, Sequence &prog,
                          const SlicePlan &plan, boost::optional<Operation> op,
                          const OptionFlags &options,
                          const DebugNameAndId &dnai) {
  const std::string opName = op == boost::none ? "" : asString(*op);
  const std::string multiUpdateName = "multiUpdate" + opName;
  logging::popops::info(multiUpdateName + " {} into {}, name={}, nullplan={}",
                        sMulti.shape(), t.shape(), dnai.getPathName(),
                        plan.getImpl().isNull);
  // Check the offsets have been specified with a multi-slice dimension
  if (offset.rank() != 2)
    throw poputil::poplibs_error(multiUpdateName +
                                 " expects offset.rank() == 2 but it is" +
                                 std::to_string(offset.rank()));
  if (offset.dim(1) != dims.size())
    throw poputil::poplibs_error(
        multiUpdateName +
        " expects offset.dim(1) == dims.size(); offset.dim(1)==" +
        std::to_string(offset.dim(1)) +
        ", dims.size()== " + std::to_string(dims.size()));
  validateParams(multiUpdateName, plan, options, t.shape(), offset[0], dims,
                 sizes);

  if (t.rank() != 2 || dims.size() != 1 || offset.rank() != 2 ||
      offset.dim(1) != 1)
    throw poputil::poplibs_error(
        multiUpdateName +
        " requires t to have 2 dimensions and dims to specify "
        "1 dimension");
  if (t.elementType() != sMulti.elementType())
    throw poputil::poplibs_error(multiUpdateName +
                                 " expects t, sMulti to have the same type");
  if (scale != boost::none) {
    if (scale->rank() != 0)
      throw poputil::poplibs_error(multiUpdateName + " scale must be a scaler");
  }
  std::string vertexName;
  if (op == boost::none) {
    vertexName = "popops::MultiUpdate";
  } else {
    vertexName = scale == boost::none ? "popops::MultiUpdateOp"
                                      : "popops::ScaledMultiUpdateOp";
  }
  if (plan.getImpl().isNull) {
    generateMultiSliceVertices(vertexName, true, op, graph, prog, offset, t,
                               sMulti, scale, dims[0], boost::none, options,
                               {dnai, multiUpdateName});
  } else {
    generatePlannedMultiUpdateOp(vertexName, plan.getImpl(), graph, prog,
                                 offset, t, sMulti, scale, dims[0], op, options,
                                 {dnai, multiUpdateName});
  }
}

// This is derived from multiSlice with \a s input rather than generated,
// the tensors swapped, etc
void multiUpdate(Graph &graph, const Tensor &t, const Tensor &sMulti,
                 const Tensor &offset, const std::vector<std::size_t> &dims,
                 const std::vector<std::size_t> &sizes, Sequence &prog,
                 const SlicePlan &plan, const OptionFlags &options,
                 const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(t, sMulti, offset, dims, sizes, plan, options));

  logging::popops::info("multiUpdate {} into {}, name={}", sMulti.shape(),
                        t.shape(), debugContext.getPathName());
  // small number of slices are updated individually
  // large number of slices are updated by a specialisation or in a loop
  std::string dName = "multiUpdate";

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

  // when planned we just use the multi-update op path with op set to
  // boost::none
  if (!plan.getImpl().isNull) {
    // there can't be a look-up split with a planned multi-update
    if (plan.getImpl().partition.lookupSplit != 1) {
      throw poplibs_error("Planned multiUpdate may not have been given the "
                          "the correct options (operationForUpdate possibly "
                          "set incorrectly");
    }
    multiUpdateOp(graph, t, sMulti, offset, boost::none, dims, sizes, prog,
                  plan, boost::none, options, {di});
    return;
  }

  validateParams("multiUpdate", plan, options, t.shape(), offset[0], dims,
                 sizes);

  // When there are only a few slices the looping code can be larger than
  // instantiating multiple vertices
  constexpr unsigned inliningThreshold = 3;
  if (offset.dim(0) <= inliningThreshold) {
    for (unsigned slice = 0; slice != offset.dim(0); ++slice) {
      dynamicUpdate(graph, t, sMulti[slice], offset[slice], dims, sizes, prog,
                    {di, dName + "/" + std::to_string(slice)});
    }
    return;
  }
  // When there are many offsets of single slices there is a fast vertex.
  // For now only 1d slices of 2d base tensors are supported.
  if (t.rank() == 2 && dims.size() == 1 && sMulti.rank() == 3 &&
      offset.rank() == 2 && offset.dim(1) == 1 && offset.dim(0) > 6) {
    generateMultiSliceVertices("popops::MultiUpdate", true, boost::none, graph,
                               prog, offset, t, sMulti, boost::none, dims[0],
                               boost::none, options, dName);
    return;
  }
  // looping case
  prog.add(countedLoop(graph, offset.dim(0),
                       [&](poplar::Tensor sIdx) {
                         Sequence body({}, {di});
                         auto tIdx =
                             dynamicSlice(graph, offset, sIdx, {0}, {1}, body,
                                          {di, dName + "/sliceIndex"})
                                 .squeeze({0});

                         auto sI =
                             dynamicSlice(graph, sMulti, sIdx, dims, sizes,
                                          body, {di, dName + "/slice"})
                                 .squeeze({0});
                         dynamicUpdate(graph, t, sI, tIdx, {0}, {1}, body,
                                       {di, dName + "/update"});
                         return body;
                       },
                       {di, dName + "/loop"}));
}

// This is derived from multiUpdate, but s is added to t rather than replacing
// it
// Currently only a single dimension may be sliced
void multiUpdateAdd(Graph &graph, const Tensor &t, const Tensor &sMulti,
                    const Tensor &offset, const Tensor &scale,
                    const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes, Sequence &prog,
                    const SlicePlan &plan, const OptionFlags &options,
                    const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(t, sMulti, offset, scale, dims, sizes, plan, options));

  if (t.elementType() != HALF && scale.elementType() != t.elementType()) {
    throw poplibs_error("Scale type can be different from data type only for "
                        "multiUpdateAdd of type half");
  }

  multiUpdateOp(graph, t, sMulti, offset, scale, dims, sizes, prog, plan,
                Operation::ADD, options, {di});
}

// This is derived from multiUpdate, but a max of s is and t is done rather than
// replacing it. Currently only a single dimension may be sliced
void multiUpdateMax(Graph &graph, const Tensor &t, const Tensor &sMulti,
                    const Tensor &offset, const std::vector<std::size_t> &dims,
                    const std::vector<std::size_t> &sizes, Sequence &prog,
                    const SlicePlan &plan, const OptionFlags &options,
                    const poplar::DebugContext &debugContext) {
  POPOPS_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(t, sMulti, offset, dims, sizes, plan, options));
  multiUpdateOp(graph, t, sMulti, offset, boost::none, dims, sizes, prog, plan,
                Operation::MAX, options, {di});
}

namespace embedding {
static void applyPlanConstraints(popsolver::Model &m,
                                 const PlanConstraints &planConstraints,
                                 const popsolver::Variable mSlicedDimSplit,
                                 const popsolver::Variable mUnslicedDimSplit,
                                 const popsolver::Variable mLookupSplit) {
  const auto constrainVar = [&](const char *name, popsolver::Variable var) {
    if (auto constraint = planConstraints.get_optional<unsigned>(name)) {
      m.equal(var, popsolver::DataType{*constraint});
    }
  };

  // unslicedGrainSize is constrained at the beginning of model construction
  // as that number is used for calculating other values in the model.
  constrainVar("slicedDimSplit", mSlicedDimSplit);
  constrainVar("unslicedDimSplit", mUnslicedDimSplit);
  constrainVar("lookupSplit", mLookupSplit);
}

static sliceInternal::Partition<std::size_t> fromSolution(
    const popsolver::Solution &s,
    const sliceInternal::Partition<popsolver::Variable> &partitionVars) {
  sliceInternal::Partition<std::size_t> partition;
  partition.lookupSplit = *s[partitionVars.lookupSplit];
  partition.slicedDimSplit = *s[partitionVars.slicedDimSplit];
  partition.unslicedDimSplit = *s[partitionVars.unslicedDimSplit];
  partition.unslicedGrainSize = *s[partitionVars.unslicedGrainSize];
  return partition;
}

template <typename T> struct EmbeddingEstimates {
  EmbeddingEstimates() = default;
  EmbeddingEstimates(const T &init)
      : baseStorageBytesPerTile(init), outputStorageBytesPerTile(init),
        indicesStorageBytesPerTile(init), exchangeCodeBytes(init),
        sliceTempBytes(init), updateTempBytes(init), peakTempBytes(init),
        sliceFirstStageExchangeCycles(init), sliceFirstStageComputeCycles(init),
        sliceSecondStageExchangeCycles(init),
        sliceSecondStageComputeCycles(init), sliceTotalCycles(init),
        updateRearrangeBaseCycles(init), updateCastSlicesCycles(init),
        updateZeroPartialsCycles(init), updateFirstStageExchangeCycles(init),
        updateFirstStageComputeCycles(init), updateReduceExchangeCycles(init),
        updateReduceComputeCycles(init), updateCastBasePreCycles(init),
        updateFinalElemwiseCycles(init), updateCastBasePostCycles(init),
        updateTotalCycles(init), totalCycles(init) {}

  // Memory (all per-tile)
  T baseStorageBytesPerTile;
  T outputStorageBytesPerTile;
  T indicesStorageBytesPerTile;
  T exchangeCodeBytes;

  T sliceTempBytes;
  T updateTempBytes;
  T peakTempBytes;

  // Cycles (worst tile) only valid when usedForSlice is set
  T sliceFirstStageExchangeCycles;
  T sliceFirstStageComputeCycles;
  T sliceSecondStageExchangeCycles;
  T sliceSecondStageComputeCycles;
  T sliceTotalCycles;

  // mUpdate* only valid when usedForUpdate is set.
  T updateRearrangeBaseCycles;
  T updateCastSlicesCycles;
  T updateZeroPartialsCycles;
  T updateFirstStageExchangeCycles;
  T updateFirstStageComputeCycles;
  T updateReduceExchangeCycles;
  T updateReduceComputeCycles;
  T updateCastBasePreCycles;
  T updateFinalElemwiseCycles;
  T updateCastBasePostCycles;
  T updateTotalCycles;

  T totalCycles;
};

static std::tuple<sliceInternal::Partition<popsolver::Variable>,
                  EmbeddingEstimates<popsolver::Variable>>
constructModel(popsolver::Model &m, const Target &target, const Type &dataType,
               const std::size_t numEntries, const std::size_t outputSize,
               const std::vector<std::size_t> &numLookups, bool useOrderingInfo,
               const SliceOptions &options) {
  const auto dataElementSize = target.getTypeSize(dataType);
  const auto numWorkerContexts = target.getNumWorkerContexts();
  // TODO: Get the correct op + scale parameters in here for
  // multiUpdateMax or other possible future update rules.
  const auto operation = options.opForUpdate;

  if (!options.usedForSlice && !options.usedForUpdate) {
    throw poplibs_error("Slice plan must be for either slice or update, or "
                        "both");
  }

  // Plan based on the max supplied number of indices
  unsigned plannedNumIndices =
      numLookups.empty()
          ? 1
          : *std::max_element(numLookups.cbegin(), numLookups.cend());

  // Choose the grainsize in unsliced dimension to avoid subword writes
  const std::size_t minGrainSizeBytes = target.getAtomicStoreGranularity();

  // The embedding dimension can be split (embeddingSplit),
  // the entries can be split (dictSplit),
  // the indices can be split (lookupSplit)
  sliceInternal::Partition<popsolver::Variable> partition;
  EmbeddingEstimates<popsolver::Variable> e(m.zero());

  // The number of repeated indices per dictionary entry is a function only of
  // the distribution
  const double offsetsPerDictEntry =
      options.indicesDistribution == IndicesDistribution::UNIFORM
          ? static_cast<double>(plannedNumIndices) / numEntries
          : plannedNumIndices;
  const auto hierarchy = getTileHierarchy(target);
  const auto perLevelExchangeBytesPerCycle =
      getPerLevelExchangeBytesPerCycle(target);
  ExchangeEstimator exchangeEstimator(m, target, hierarchy,
                                      perLevelExchangeBytesPerCycle);
  const auto ipuLevel = hierarchy.size() - 1;

  // Indices are int32 so 4bytes each
  const auto mBytesPerIndex = m.addConstant(target.getTypeSize(UNSIGNED_INT));
  const auto mBytesPerFloat = m.addConstant(target.getTypeSize(FLOAT));

  // The grainsize can be constrained externally so bytesPerGrain must be
  // derived from it
  const auto unslicedGrainSize =
      options.planConstraints.get_optional<unsigned>("unslicedGrainSize")
          .value_or(ceildiv(lcm(minGrainSizeBytes, dataElementSize),
                            dataElementSize));
  const auto bytesPerGrain = unslicedGrainSize * dataElementSize;

  partition.unslicedGrainSize =
      m.addConstant(unslicedGrainSize, "unslicedGrainSize");
  const auto mBytesPerGrain = m.addConstant(bytesPerGrain);
  const auto mOutputSize = m.addConstant(outputSize, "outputSize");

  const auto mNumUnslicedGrains = // per row
      m.ceildiv(mOutputSize, partition.unslicedGrainSize, "numUnslicedGrains");

  // split the embedding between \a partition.unslicedDimSplit tiles
  partition.unslicedDimSplit =
      m.addVariable(1, std::numeric_limits<unsigned>::max(), "embeddingSplit");
  m.lessOrEqual(partition.unslicedDimSplit, mNumUnslicedGrains);
  m.ceildivConstrainDivisor(mNumUnslicedGrains, partition.unslicedDimSplit);

  // The entries are split across \a mDictSplit groups of tiles,
  // each of which will select a candidate in the first stage of a lookup.
  // A second stage is then required to select between theses candidates. This
  // means that temporary memory is required after the first pass.
  // Splits leaving less than 2 entries per tile will have more unmeasured
  // overhead than is saved in base memory so are prohibited.
  partition.slicedDimSplit =
      m.addVariable(1, ceildiv(numEntries, 2u), "entriesSplit");
  const auto mDictIsSplit = m.reifiedLess(m.one(), partition.slicedDimSplit);

  // When there are many lookups we can split the lookups between multiple
  // groups of tiles each performing the same lookup on a subset of indices.
  // This requires the embedding to be broadcast for lookups, and the updates
  // to be serialised or reduced on update
  // When there is an indices split a temporary embedding buffer is required
  // in both passes

  // We can't split lookup for a multiUpdate without an operation because it
  // would entail a reduction and we can't guarantee a reduction would be valid
  // as the data may not be reduce-able.
  const auto maxLookupSplits =
      options.usedForUpdate && options.opForUpdate == std::nullopt
          ? 1
          : plannedNumIndices;
  partition.lookupSplit = m.addVariable(1, maxLookupSplits, "lookupSplit");
  const auto mLookupsAreSplit = m.reifiedLess(m.one(), partition.lookupSplit);
  const auto mNumTiles = m.addConstant(target.getNumTiles(), "numTiles");
  const auto mNumEntries = m.addConstant(numEntries);
  const auto mNumIndices = m.addConstant(plannedNumIndices);

  // Max number of each dimension of the embedding processed on each
  // tile during forward pass (slice)
  const auto mUnslicedGrainsPerTile =
      m.ceildivConstrainDivisor(mNumUnslicedGrains, partition.unslicedDimSplit);
  const auto mDictEntriesPerTile =
      m.ceildivConstrainDivisor(mNumEntries, partition.slicedDimSplit);
  const auto mLookupsPerTile =
      m.ceildivConstrainDivisor(mNumIndices, partition.lookupSplit);

  const auto mUsedTiles =
      m.product({partition.unslicedDimSplit, partition.slicedDimSplit,
                 partition.lookupSplit},
                "totalSplit");
  m.lessOrEqual(mUsedTiles, mNumTiles);

  const auto mBaseGrainsPerTile =
      m.product({mUnslicedGrainsPerTile, mDictEntriesPerTile});

  const auto mUnslicedElemsPerTile =
      m.product({mUnslicedGrainsPerTile, partition.unslicedGrainSize});

  // Calculate persistent bytes for storage per-tile.
  // We also spread base tensor grains over tiles that will use them when
  // allocating i.e. over lookupSplit tiles.
  const auto mBaseGrainsStoragePerTile =
      m.ceildiv(mBaseGrainsPerTile, partition.lookupSplit);
  const auto mBaseElemsStoragePerTile =
      m.product({mBaseGrainsStoragePerTile, partition.unslicedGrainSize});
  e.baseStorageBytesPerTile =
      m.product({mBaseGrainsStoragePerTile, mBytesPerGrain});

  const auto mBaseBytesPerTile =
      m.product({mBaseGrainsPerTile, mBytesPerGrain});
  auto mBaseBytesPerTileDiffWithPerfectlyDistributed = m.zero();
  if (options.alwaysIncludeBaseRearrangementCost) {
    const auto mBaseBytesStoragePerTilePerfectlyDistributed = m.product(
        {m.ceildiv(m.product({mNumEntries, mNumUnslicedGrains}), mNumTiles),
         mBytesPerGrain});
    mBaseBytesPerTileDiffWithPerfectlyDistributed =
        m.sub(mBaseBytesPerTile, mBaseBytesStoragePerTilePerfectlyDistributed);
  }

  // We allocate indices linearly with a minimum no. per-tile.
  const auto mMinIndicesPerTile =
      m.min({mNumIndices, m.addConstant(minIndicesPerTile)});
  const auto mIndicesPerTile =
      m.max({mMinIndicesPerTile, m.ceildiv(mNumIndices, mNumTiles)});
  e.indicesStorageBytesPerTile = m.product({mIndicesPerTile, mBytesPerIndex});

  // We allocate output based on forward pass (slice) usage.
  // The first stage results in partition.slicedDimSplit partials spread over
  // tiles. Partials per-tile are mLookupsPerTile * mUnslicedGrainsPerTile.
  const auto mSecondStageLookupsPerTile =
      m.ceildiv(mLookupsPerTile, partition.slicedDimSplit);
  // The second stage results in mLookupsPerTile spread over
  // partition.slicedDimSplit tiles.
  const auto mOutputGrainsPerTile =
      m.product({mSecondStageLookupsPerTile, mUnslicedGrainsPerTile});
  const auto mOutputElemsPerTile =
      m.product({mOutputGrainsPerTile, partition.unslicedGrainSize});
  e.outputStorageBytesPerTile =
      m.product({mOutputGrainsPerTile, mBytesPerGrain});

  // The base tensor must be broadcast across the `partition.lookupSplit` groups
  // as it is distributed to balance memory. The indices must be received from a
  // set of tiles, so a number of setmux instructions are required.
  //
  // 0 and 1 indicate which stage in the forward pass this exchange is
  // attributed to.
  const auto mIndicesExchangeInstrs0 =
      m.ceildiv(mLookupsPerTile, mIndicesPerTile);

  if (options.usedForSlice) {
    // mBaseBytesPerTile gives the size of data taken as input to MultiSlice
    // vertices in the first stage. If there is a lookup split then this data
    // must be broadcast.
    //
    // If there is no lookup split and the base tensor is allocated according to
    // the partition in the plan, we should not need to exchange anything.
    // However, we add a cost anyway here based on the difference in memory
    // usage with a perfectly spread base tensor over tiles.
    // This biases the planner towards utilising more tiles, particularly
    // when the base tensor is large but there is not much work to do (T46394).
    // We justify this as being like assuming that if the base tensor is not
    // well spread over tiles, we would not allocate it out according to the
    // partition in the plan, and would therefore need to copy it to the tiles
    // for each partition before the slice/update.
    // This is controlled by the internal option
    // "internal.alwaysIncludeBaseRearrangementCost".
    const auto mBaseTempBytesPerTile =
        m.max({m.product({mLookupsAreSplit, mBaseBytesPerTile}),
               mBaseBytesPerTileDiffWithPerfectlyDistributed});
    // mBaseGrainsPerTile gives the number of grains taken as input
    // to MultiSlice vertices in the first stage. If there is a lookup
    // split then this data must be broadcast.
    e.sliceFirstStageExchangeCycles = exchangeEstimator(
        mBaseTempBytesPerTile, ipuLevel, "slice.0.exchange.cycles");
    const MultiSliceTargetParameters targetParams{target, dataType};
    e.sliceFirstStageComputeCycles = m.call<unsigned>(
        {
            mUnslicedElemsPerTile,
            mLookupsPerTile,
            mNumEntries,
            mDictEntriesPerTile,
        },
        [numWorkerContexts, targetParams,
         options](const std::vector<unsigned> &values) {
          const auto elemsPerSlice = values[0];
          const auto numOffsets = values[1];
          const auto numDictEntries = values[2];
          const auto maxDictEntriesPerTile = values[3];
          const double proportionIndicesInRange =
              double(maxDictEntriesPerTile) / double(numDictEntries);
          const auto maxOffsetsPerWorker =
              ceildiv(numOffsets, numWorkerContexts);
          unsigned offsetsInRangePerWorker =
              options.indicesDistribution == IndicesDistribution::ONE_POINT
                  ? maxOffsetsPerWorker
                  :
                  // Samples are assumed to be IID
                  // with uniform distribution and a random mapping to tiles.
                  std::ceil(maxOffsetsPerWorker * proportionIndicesInRange);
          const auto cycles = getMultiSliceCycleEstimate(
              targetParams, elemsPerSlice, maxOffsetsPerWorker,
              offsetsInRangePerWorker, 0, false, false);
          return popsolver::DataType{cycles * numWorkerContexts};
        },
        "slice.0.compute.cycles");

    // For the second stage, we exchange the result of the first slice
    // all to all between groups of tiles.
    const auto mSecondStageInputBytesPerTile =
        m.product({partition.slicedDimSplit, mSecondStageLookupsPerTile,
                   mUnslicedGrainsPerTile, mBytesPerGrain});
    e.sliceSecondStageExchangeCycles = exchangeEstimator(
        mSecondStageInputBytesPerTile, ipuLevel, "slice.1.exchange.cycles");
    e.sliceSecondStageComputeCycles = m.call<unsigned>(
        {
            mUnslicedGrainsPerTile,
            mSecondStageLookupsPerTile,
            mNumEntries,
            mDictEntriesPerTile,
        },
        [numWorkerContexts, targetParams,
         options](const std::vector<unsigned> &values) {
          const auto elemsPerSlice = values[0];
          const auto numOffsets = values[1];
          const auto numDictEntries = values[2];
          const auto maxDictEntriesPerTile = values[3];
          const double proportionIndicesInRange =
              double(maxDictEntriesPerTile) / double(numDictEntries);
          const auto maxOffsetsPerWorker =
              ceildiv(numOffsets, numWorkerContexts);
          unsigned offsetsInRangePerWorker =
              options.indicesDistribution == IndicesDistribution::ONE_POINT
                  ? maxOffsetsPerWorker
                  :
                  // Samples are assumed to be IID
                  // with uniform distribution and a random mapping to tiles.
                  std::ceil(maxOffsetsPerWorker * proportionIndicesInRange);

          const auto cycles = getMultiSliceCycleEstimate(
              targetParams, elemsPerSlice, maxOffsetsPerWorker,
              offsetsInRangePerWorker,
              0, // Sorting information is not used
              false, false);
          return popsolver::DataType{cycles * numWorkerContexts};
        },
        "slice.1.compute.cycles");

    e.sliceSecondStageExchangeCycles =
        m.product({e.sliceSecondStageExchangeCycles, mDictIsSplit});
    e.sliceSecondStageComputeCycles =
        m.product({e.sliceSecondStageComputeCycles, mDictIsSplit});
    e.sliceTotalCycles = m.sum(
        {e.sliceFirstStageExchangeCycles, e.sliceFirstStageComputeCycles,
         e.sliceSecondStageExchangeCycles, e.sliceSecondStageComputeCycles});
    e.totalCycles = e.sliceTotalCycles;

    const auto mEmbeddingExchangeInstrs0 =
        m.product({mLookupsAreSplit, partition.lookupSplit});
    // When there is a dictSplit the data will be exchanged between groups of
    // `partition.slicedDimSplit` tiles
    const auto mOutputToInputExchangeInstrs1 =
        m.product({mDictIsSplit, partition.slicedDimSplit});

    // The indices are copied implicitly and are re-broadcast for the second
    // stage
    const auto &mIndicesExchangeInstrs1 = mIndicesExchangeInstrs0;
    e.exchangeCodeBytes = m.product(
        {m.addConstant(4u),
         m.sum({mEmbeddingExchangeInstrs0, mIndicesExchangeInstrs0,
                mOutputToInputExchangeInstrs1, mIndicesExchangeInstrs1})});

    // When splitting the dictionary a the output of the first stage will be
    // rearranged for the second stage.
    const auto mSlicesFirstStageOutputTempBytes =
        m.product({mDictIsSplit, mLookupsPerTile, mUnslicedGrainsPerTile,
                   mBytesPerGrain});
    const auto mSlicesSecondStageInputTempBytes = m.product(
        {mDictIsSplit, partition.slicedDimSplit, mSecondStageLookupsPerTile,
         mUnslicedGrainsPerTile, mBytesPerGrain});

    const auto mIndicesFirstStageTempBytes =
        m.product({mLookupsPerTile, mBytesPerIndex});
    const auto mIndicesSecondStageTempBytes =
        m.product({mSecondStageLookupsPerTile, mBytesPerIndex});

    e.sliceTempBytes =
        m.max({// Potentially multi-cast copy of base tensor/indices, and
               // temporary bytes for output of first stage
               m.sum({mBaseTempBytesPerTile, mSlicesFirstStageOutputTempBytes,
                      mIndicesFirstStageTempBytes}),
               // Temporary bytes for output of first stage, rearranged version
               // as input to the second stage, and multi-cast indices.
               m.sum({mSlicesFirstStageOutputTempBytes,
                      mSlicesSecondStageInputTempBytes,
                      mIndicesSecondStageTempBytes})});
  }

  if (options.usedForUpdate) {
    // See comment for base exchange cost in slice estimates.
    // We bias the planner with a cost to rearrange the input tensor.
    const auto mBaseTempBytesPerTile =
        mBaseBytesPerTileDiffWithPerfectlyDistributed;

    // When no index split there are no temporaries beyond those used in a
    // lookup, the vertices work directly on the base, slices and indices
    // tensors.
    // When `mLookupsAreSplit` the indices and updates are rearranged onto the
    // tile, the updates are cast to FLOAT and then accumulated
    // with a FLOAT copy of the base tensor.

    // For now we force float partial type for the update.
    const auto mFloatData = m.addConstant(dataType == FLOAT ? 1u : 0u);
    const auto mNotFloatData = m.booleanNot(mFloatData);
    const auto mOpIsAdd = m.addConstant(
        operation != std::nullopt && *operation == Operation::ADD ? 1u : 0u);
    const auto mNeedsCast =
        m.booleanAnd(m.booleanAnd(mNotFloatData, mLookupsAreSplit), mOpIsAdd);
    const auto mUpdatesCastTempBytesPerTile =
        m.product({mNeedsCast, mOutputElemsPerTile, mBytesPerFloat});
    // Temp bytes needed if the updates are multi-cast to tiles.
    const auto mUpdatesTempBytesPerTile =
        m.product({mDictIsSplit, mLookupsPerTile, mUnslicedGrainsPerTile,
                   partition.unslicedGrainSize, mBytesPerFloat});
    // Temp bytes needed if the updates need to be cast to a higher precision
    // if they do not also need to be multi-cast - i.e. if not multi-cast
    // we will directly use the casted updates and they will stay live,
    // otherwise we will use the multi-cast updates and the casted updates
    // will die.
    const auto mUpdatesCastTempBytesPerTileAfterMulticast =
        m.product({m.booleanNot(mDictIsSplit), mUpdatesCastTempBytesPerTile});

    const auto mPartialElemsPerTile =
        m.product({mDictEntriesPerTile, mUnslicedGrainsPerTile,
                   partition.unslicedGrainSize});

    // Multipled by 2 because we would need to copy from source base tensor
    // *and back*.
    e.updateRearrangeBaseCycles =
        m.product({exchangeEstimator(mBaseTempBytesPerTile, ipuLevel,
                                     "update.0.rearrangeBase.cycles"),
                   m.addConstant(2u)});

    e.updateCastSlicesCycles =
        modelContiguousCast(target, dataType, FLOAT, m, mOutputElemsPerTile,
                            "update.0.castSlices")
            .cycles;
    e.updateCastSlicesCycles =
        m.product({mNeedsCast, e.updateCastSlicesCycles});
    e.updateZeroPartialsCycles =
        modelContiguousFill(target, FLOAT, m, mPartialElemsPerTile,
                            "update.0.zeroPartials")
            .cycles;
    e.updateZeroPartialsCycles =
        m.product({mLookupsAreSplit, e.updateZeroPartialsCycles});
    // Account for exchange of updates to tiles here, where the updates
    // will be broadcast by the number of splits of the sliced dimension.
    e.updateFirstStageExchangeCycles = exchangeEstimator(
        mUpdatesTempBytesPerTile, ipuLevel, "update.0.exchange.cycles");

    {
      e.updateFirstStageComputeCycles = m.call<unsigned>(
          {mUnslicedElemsPerTile, mLookupsPerTile, mNumEntries,
           mDictEntriesPerTile, mNeedsCast},
          [&target, &options, operation, offsetsPerDictEntry, useOrderingInfo,
           &dataType](const std::vector<unsigned> &values) {
            const auto elemsPerSlice = values[0];
            const auto numOffsets = values[1];
            const auto numDictEntries = values[2];
            const auto maxDictEntriesPerTile = values[3];
            const auto needsCast = values[4];

            const MultiUpdateOpTargetParameters targetMultiUpdateOpParams{
                target, needsCast ? FLOAT : dataType};

            const MultiSliceTargetParameters targetMultiUpdateParams{
                target, needsCast ? FLOAT : dataType};

            const auto maxDictEntriesPerWorker =
                getMultiUpdateOpMaxElemsPerWorker(
                    targetMultiUpdateOpParams.numWorkerContexts,
                    maxDictEntriesPerTile);
            const unsigned maxOffsetsPerDictEntry =
                std::ceil(offsetsPerDictEntry);
            const double maxProportionOfIndexableRangePerWorker =
                double(maxDictEntriesPerWorker) / double(numDictEntries);

            unsigned numOffsetsInRangePerWorker;
            if (options.indicesDistribution == IndicesDistribution::ONE_POINT) {
              numOffsetsInRangePerWorker = numOffsets;
            } else {
              // If indices are sorted we know the worst case in the average
              if (options.indicesAreSorted) {
                numOffsetsInRangePerWorker = std::min(
                    static_cast<unsigned>(std::ceil(maxDictEntriesPerWorker *
                                                    offsetsPerDictEntry)),
                    numOffsets);
              } else {
                // If indices are not sorted the samples are assumed to be IID
                // with uniform distribution and a random mapping to tiles.
                numOffsetsInRangePerWorker = std::ceil(
                    numOffsets * maxProportionOfIndexableRangePerWorker);
              }
            }

            const bool isScaled = true;
            // cycle estimates for update without an operation has same cycles
            // as a slice.
            const auto cycles =
                operation == std::nullopt
                    ? getMultiSliceCycleEstimate(
                          targetMultiUpdateParams, elemsPerSlice, numOffsets,
                          numOffsetsInRangePerWorker, maxOffsetsPerDictEntry,
                          true, useOrderingInfo)
                    : getMultiUpdateOpCycleEstimate(
                          targetMultiUpdateOpParams,
                          /* subWordWritesRequired */ false, elemsPerSlice,
                          numOffsets, numOffsetsInRangePerWorker,
                          maxOffsetsPerDictEntry, *operation, isScaled,

                          false, useOrderingInfo);
            return popsolver::DataType{cycles};
          },
          "update.0.compute.cycles");
    }

    const auto mPartialsBytesPerTile =
        m.product({mLookupsAreSplit, mPartialElemsPerTile, mBytesPerFloat});
    auto mReduceEstimateCyles = m.zero();

    // We can't do a reduction or cast if there is no operation in the
    // multi-update. Take conditional paths only because there is no
    // reduction defined for no update operation.
    if (operation != std::nullopt) {
      const auto mUpdateReduceEstimates = modelBalancedIntertileReduction(
          target, hierarchy, FLOAT, FLOAT, *operation, /* isUpdate */ false, m,
          exchangeEstimator, mPartialElemsPerTile, partition.lookupSplit,
          "update.1.reduce");
      e.updateReduceExchangeCycles =
          mUpdateReduceEstimates.cyclesBreakdown.exchange;
      e.updateReduceComputeCycles =
          mUpdateReduceEstimates.cyclesBreakdown.compute;
      mReduceEstimateCyles = mUpdateReduceEstimates.cycles;
      if (*operation == Operation::ADD) {
        e.updateCastBasePreCycles =
            modelContiguousCast(target, dataType, FLOAT, m,
                                mBaseElemsStoragePerTile, "update.1.castBase")
                .cycles;
        e.updateCastBasePreCycles =
            m.product({mNeedsCast, e.updateCastBasePreCycles});
        // NOTE: Optimistically assuming fast path - this is not forced
        // but a runtime check opportunistically selects the fast path if
        // inputs are in different memory elements.
        e.updateFinalElemwiseCycles =
            modelContiguousScaledAdd(
                target, FLOAT, FLOAT, /* scaleIsConstant */ false,
                /* usesMemoryConstraints */ true, m, mBaseElemsStoragePerTile,
                "update.1.updateBase")
                .cycles;
        e.updateFinalElemwiseCycles =
            m.product({mLookupsAreSplit, e.updateFinalElemwiseCycles});
        e.updateCastBasePostCycles =
            modelContiguousCast(target, FLOAT, dataType, m,
                                mBaseElemsStoragePerTile,
                                "update.1.castBaseBack")
                .cycles;
        e.updateCastBasePostCycles =
            m.product({mNeedsCast, e.updateCastBasePostCycles});
      } else if (*operation == Operation::MAX) {
        // estimate a maxInPlace.
        // TODO: use modelled estimates that are general for other operations
        // (T45159)
        e.updateFinalElemwiseCycles =
            m.ceildiv(mBaseElemsStoragePerTile,
                      m.addConstant(target.getVectorWidth(dataType)));
        e.updateFinalElemwiseCycles =
            m.product({mLookupsAreSplit, e.updateFinalElemwiseCycles});
      }
    }
    e.updateTotalCycles =
        m.sum({e.updateRearrangeBaseCycles, e.updateCastSlicesCycles,
               e.updateZeroPartialsCycles, e.updateFirstStageExchangeCycles,
               e.updateFirstStageComputeCycles, mReduceEstimateCyles,
               e.updateCastBasePreCycles, e.updateFinalElemwiseCycles,
               e.updateCastBasePostCycles});

    e.totalCycles = m.sum({e.totalCycles, e.updateTotalCycles});

    const auto mIndicesTempBytesPerTile =
        m.product({mLookupsPerTile, mBytesPerIndex});

    e.updateTempBytes = m.sum(
        {mBaseTempBytesPerTile,
         m.max({// If we need a cast version of the updates, this will take
                // temporary memory.
                mUpdatesCastTempBytesPerTile,
                // If we have split the dictionary, we will need to multi-cast
                // the updates.
                m.sum({mUpdatesCastTempBytesPerTile, mUpdatesTempBytesPerTile}),
                // During the update, we have partials, casted/multi-cast
                // updates, and multi-cast indices temporarily.
                m.sum({mUpdatesCastTempBytesPerTileAfterMulticast,
                       mUpdatesTempBytesPerTile, mPartialsBytesPerTile,
                       mIndicesTempBytesPerTile}),
                // If we need a reduction we will have
                // reduction (also the actual update will have the base upcast
                // to the same size as the partials, so the same footprint)
                m.product({mLookupsAreSplit, mPartialsBytesPerTile,
                           m.addConstant(2u)})})});

    // Indices are as for the forward pass;
    // plus the rearrangement will be an all-all exchange
    e.exchangeCodeBytes =
        m.sum({e.exchangeCodeBytes,
               m.product({mIndicesExchangeInstrs0, m.addConstant(4)}),
               m.product({mLookupsAreSplit, partition.lookupSplit,
                          m.addConstant(4)})});
  }

  e.peakTempBytes = m.max({e.sliceTempBytes, e.updateTempBytes});
  return std::make_tuple(partition, e);
}

static void
applyConstraints(popsolver::Model &m, const Target &target,
                 const sliceInternal::Partition<popsolver::Variable> &partition,
                 const EmbeddingEstimates<popsolver::Variable> &estimates,
                 const SliceOptions &options, bool limitTempMemory) {
  applyPlanConstraints(m, options.planConstraints, partition.slicedDimSplit,
                       partition.unslicedDimSplit, partition.lookupSplit);
  if (options.availableMemoryProportion && limitTempMemory) {
    const auto mMaxAllowedTempBytes = m.addConstant(
        options.availableMemoryProportion.value() * target.getBytesPerTile());
    m.lessOrEqual(estimates.peakTempBytes, mMaxAllowedTempBytes);
  }
}

static popsolver::Solution
minimize(popsolver::Model &m, const Target &target,
         const EmbeddingEstimates<popsolver::Variable> &estimates,
         const PlanMinimisationTarget &minimisationTarget) {
  popsolver::Variable goal;
  switch (minimisationTarget) {
  case PlanMinimisationTarget::MEMORY: {
    // Minimise total memory footprint, prioritising persistent memory
    // indices are persistent if they are required for the update pass
    goal = m.sum(
        {estimates.baseStorageBytesPerTile, estimates.outputStorageBytesPerTile,
         estimates.indicesStorageBytesPerTile, estimates.exchangeCodeBytes});
    goal = m.product({goal, m.addConstant(10)});
    goal = m.sum({goal, estimates.peakTempBytes});
    break;
  }
  case PlanMinimisationTarget::CYCLES: {
    goal = estimates.totalCycles;
    break;
  }
  }

  return m.minimize({goal});
}

// Estimates for a solution and plan
struct PlanAndEstimates {
  EmbeddingEstimates<std::size_t> e;
  SlicePlanInternal plan;
};

static PlanAndEstimates choosePlan(const Target &target, const Type &dataType,
                                   const std::size_t numEntries,
                                   const std::size_t outputSize,
                                   const std::vector<std::size_t> &numLookups,
                                   const SliceOptions &options) {
  std::uint64_t bestPlanCyclesCost = std::numeric_limits<std::size_t>::max();
  std::vector<bool> sortedIndicesCandidates;

  auto useOrderingInfoConstraint =
      options.planConstraints.get_optional<bool>("useOrderingInfo");
  if (useOrderingInfoConstraint) {
    checkOrderingInfoConsistencyWithOptions(*useOrderingInfoConstraint,
                                            options.indicesAreSorted);
    sortedIndicesCandidates.push_back(*useOrderingInfoConstraint);
  } else {
    sortedIndicesCandidates.push_back(false);
    if (options.indicesAreSorted) {
      sortedIndicesCandidates.push_back(true);
    }
  }

  PlanAndEstimates best{};

  for (auto useOrderingInfo : sortedIndicesCandidates) {
    popsolver::Model m;
    auto [partition, estimates] =
        constructModel(m, target, dataType, numEntries, outputSize, numLookups,
                       useOrderingInfo, options);
    applyConstraints(m, target, partition, estimates, options, true);
    auto s = minimize(m, target, estimates, options.planMinimisationTarget);
    if (s.validSolution()) {
      auto totalCycles = *s[estimates.totalCycles];
      SlicePlanInternal p;
      p.partition = fromSolution(s, partition);
      p.useIndicesOrderingInfo = useOrderingInfo;
      p.rank = 2;
      p.slicedDims = {0};
      p.slicedDimSizes = {1};
      p.isNull = false;
      if (totalCycles < bestPlanCyclesCost && s.validSolution()) {
        bestPlanCyclesCost = totalCycles;
        best.plan = p;

        best.e.baseStorageBytesPerTile = *s[estimates.baseStorageBytesPerTile];
        best.e.outputStorageBytesPerTile =
            *s[estimates.outputStorageBytesPerTile];
        best.e.indicesStorageBytesPerTile =
            *s[estimates.indicesStorageBytesPerTile];
        best.e.exchangeCodeBytes = *s[estimates.exchangeCodeBytes];
        best.e.sliceTempBytes = *s[estimates.sliceTempBytes];
        best.e.updateTempBytes = *s[estimates.updateTempBytes];
        best.e.peakTempBytes = *s[estimates.peakTempBytes];

        best.e.sliceFirstStageExchangeCycles =
            *s[estimates.sliceFirstStageExchangeCycles];
        best.e.sliceFirstStageComputeCycles =
            *s[estimates.sliceFirstStageComputeCycles];
        best.e.sliceSecondStageExchangeCycles =
            *s[estimates.sliceSecondStageExchangeCycles];
        best.e.sliceSecondStageComputeCycles =
            *s[estimates.sliceSecondStageComputeCycles];
        best.e.sliceTotalCycles = *s[estimates.sliceTotalCycles];
        best.e.updateRearrangeBaseCycles =
            *s[estimates.updateRearrangeBaseCycles];
        best.e.updateCastSlicesCycles = *s[estimates.updateCastSlicesCycles];
        best.e.updateZeroPartialsCycles =
            *s[estimates.updateZeroPartialsCycles];
        best.e.updateFirstStageExchangeCycles =
            *s[estimates.updateFirstStageExchangeCycles];
        best.e.updateFirstStageComputeCycles =
            *s[estimates.updateFirstStageComputeCycles];
        best.e.updateReduceExchangeCycles =
            *s[estimates.updateReduceExchangeCycles];
        best.e.updateReduceComputeCycles =
            *s[estimates.updateReduceComputeCycles];
        best.e.updateCastBasePreCycles = *s[estimates.updateCastBasePreCycles];
        best.e.updateFinalElemwiseCycles =
            *s[estimates.updateFinalElemwiseCycles];
        best.e.updateCastBasePostCycles =
            *s[estimates.updateCastBasePostCycles];
        best.e.updateTotalCycles = *s[estimates.updateTotalCycles];
        best.e.totalCycles = *s[estimates.totalCycles];
      }
    }
  }
  return best;
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

  logging::popops::debug(
      "DynamicSlicePlan for type {}, numEntries {}, outputSize {},"
      " numLookups {},\n  options={}",
      dataType, numEntries, outputSize, numLookups, options);
  const auto &target = graph.getTarget();

  auto best =
      choosePlan(target, dataType, numEntries, outputSize, numLookups, options);

  if (best.plan.isNull && options.availableMemoryProportion) {
    // Warn and try again without the available memory proportion if
    // no solution could be found.
    logging::popops::warn("embedding::plan could not find a valid solution "
                          "with availableMemoryProportion={}, trying again "
                          "with unlimited temporary memory",
                          *options.availableMemoryProportion);
    best = choosePlan(target, dataType, numEntries, outputSize, numLookups,
                      options);
  }

  // If we don't have a valid plan, warn and return a null plan.
  if (best.plan.isNull) {
    logging::popops::warn(
        "Slice planner could not find a valid solution, opting for no plan");
    return std::make_unique<SlicePlanInternal>();
  }

  logging::popops::debug("Embedding plan {}", best.plan);

  logging::popops::debug(
      "Tile memory estimates (bytes on worst tile):\n"
      "  base storage {},\n"
      "  output storage {},\n"
      "  indices storage {},\n"
      "  exchange code {}, \n"
      "  slice peak temporary memory {},\n"
      "  update peak temporary memory {}, \n"
      "  peak temporary memory {}\n",
      best.e.baseStorageBytesPerTile, best.e.outputStorageBytesPerTile,
      best.e.indicesStorageBytesPerTile, best.e.exchangeCodeBytes,
      best.e.sliceTempBytes, best.e.updateTempBytes, best.e.peakTempBytes);

  logging::popops::debug(
      "Cycle estimates (worst tile):\n"
      "  slice first stage exchange {},\n"
      "  slice first stage compute {},\n"
      "  slice second stage exchange {},\n"
      "  slice second stage compute {},\n"
      "  slice total {},\n"
      "  update rearrange base cycles {},\n"
      "  update cast slices cycles {},\n"
      "  update zero partials cycles {},\n"
      "  update first stage exchange {},\n"
      "  update first stage compute {},\n"
      "  update reduce exchange {},\n"
      "  update reduce compute {},\n"
      "  update cast base pre cycles {},\n"
      "  update final elementwise cycles {},\n"
      "  update cast base post cycles {},\n"
      "  update total {},\n"
      "  total {}",
      best.e.sliceFirstStageExchangeCycles, best.e.sliceFirstStageComputeCycles,
      best.e.sliceSecondStageExchangeCycles,
      best.e.sliceSecondStageComputeCycles, best.e.sliceTotalCycles,
      best.e.updateRearrangeBaseCycles, best.e.updateCastSlicesCycles,
      best.e.updateZeroPartialsCycles, best.e.updateFirstStageExchangeCycles,
      best.e.updateFirstStageComputeCycles, best.e.updateReduceExchangeCycles,
      best.e.updateReduceComputeCycles, best.e.updateCastBasePreCycles,
      best.e.updateFinalElemwiseCycles, best.e.updateCastBasePostCycles,
      best.e.updateTotalCycles, best.e.totalCycles);

  return std::make_unique<SlicePlanInternal>(std::move(best.plan));
}

} // end namespace embedding

} // end namespace popops
