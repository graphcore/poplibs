// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/FullyConnected.hpp"

#include "FullyConnectedOnTile.hpp"
#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPlan.hpp"
#include "FullyConnectedTensorMetaData.hpp"
#include "FullyConnectedUtils.hpp"
#include "PlanningCacheImpl.hpp"
#include "TensorMetaDataBase.hpp"

// FIXME: poplin internal includes
#include "ConvOptions.hpp"
#include "ConvReduce.hpp"
#include "MatMulInternal.hpp"

#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Pad.hpp>
#include <popops/Reduce.hpp>

#include "poputil/TensorMetaData.hpp"
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/TileHierarchy.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/logging.hpp"

#include <boost/optional.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace poplibs_support;
using namespace poputil;

namespace popsparse {

using namespace fullyconnected;

namespace {

// Needed strictly in this file. Other operations are pass-agnostic and expect
// any transformation of parameters to have been done before-hand
struct SinglePassPlan {
  fullyconnected::Vector<unsigned> grouping;
  fullyconnected::Vector<unsigned> partition;
  fullyconnected::Vector<unsigned> initialDistributionBucketPartition;
  unsigned nzElemsPerBucket;
  unsigned metaInfoElemsPerBucket;
  fullyconnected::PartitionToPNMappingOrder mappingOrder;
  fullyconnected::OnTileMethod method;
  // Some things need mapping back to forward pass such as tile mapping
  // of operations and sub-group id calculation.
  std::vector<std::size_t> dimShuffleToFwd;
};

} // end anonymous namespace

template <typename T>
static std::vector<T> shuffleVector(const std::vector<T> &v,
                                    const std::vector<std::size_t> &shuffle) {
  assert(v.size() == shuffle.size());
  std::vector<T> result;
  result.reserve(v.size());
  for (const auto i : shuffle) {
    result.emplace_back(v[i]);
  }
  return result;
}

template <typename T>
static std::vector<T> inversePermutation(const std::vector<T> &shuffle) {
  std::vector<T> result(shuffle.size());
  for (std::size_t i = 0; i < shuffle.size(); ++i) {
    result[shuffle[i]] = i;
  }
  return result;
}

// Starting offset for propagation phase, result given in terms of
// forward pass, and is the same for all passes hence 'canonical'
static std::vector<unsigned>
getPropagationStartingOffsetCanonical(const SinglePassPlan &plan) {
  const auto partition =
      shuffleVector(plan.partition.asStdVector(), plan.dimShuffleToFwd);
  auto indices =
      shuffleVector((plan.partition / plan.initialDistributionBucketPartition -
                     Vector<unsigned>(1))
                        .asStdVector(),
                    plan.dimShuffleToFwd);

  // A check that our starting index does not 'slice' outer dimensions
  // leaving parts of the volume uncovered or covered multiple times by
  // our pattern of iteration.
#ifndef NDEBUG
  bool innerDimsAreCompletelyCovered = true;
  for (int i = indices.size() - 1; i > 0; --i) {
    assert(indices[i] == 0 || innerDimsAreCompletelyCovered);
    innerDimsAreCompletelyCovered &= indices[i] == partition[i] - 1;
  }
#endif

  indices.back()++;
  for (int i = indices.size() - 1; i > 0; --i) {
    if (indices[i] == partition[i]) {
      ++indices[i - 1];
      indices[i] = 0;
    }
  }
  return indices;
}

// As above but the starting offset is given in terms of the pass
// in question.
static Vector<unsigned>
getPropagationStartingOffset(const SinglePassPlan &plan) {
  const auto fwdOffset = getPropagationStartingOffsetCanonical(plan);
  const auto indices =
      shuffleVector(fwdOffset, inversePermutation(plan.dimShuffleToFwd));
  return indices;
}

// Unflatten a flattened index, but allowing indices that extend beyond
// the bounds of the given shape. In this case the outer index
// can be greater than the bounds of the shape in the outer dimension.
template <typename T>
static std::vector<T> unflattenIndexBoundless(const std::vector<T> &shape,
                                              std::size_t index) {
  std::vector<T> coord(shape.size());
  for (std::size_t i = shape.size() - 1; i > 0; --i) {
    coord[i] = index % shape[i];
    index /= shape[i];
  }
  // Outer-most index is not bounded
  coord[0] = index;

  return coord;
}

// Flatten an index where the outer index may not reside within the shape.
template <class T>
static std::size_t flattenIndexBoundless(const std::vector<T> &shape,
                                         const std::vector<T> &indices) {
  auto rank = shape.size();
  assert(indices.size() == rank);
  std::size_t index = 0;
  for (unsigned i = 0; i != rank; ++i) {
    index = index * shape[i] + indices[i];
  }
  return index;
}

// Get all the information needed to iterate in the control program.
// This determines the optimisation whereby we may not need to
// iterate all dims, as well as the order of iteration and starting
// index values.
static std::tuple<std::vector<std::size_t>, std::vector<unsigned>>
getPropagationIteratedDimsAndStartingIndices(const SinglePassPlan &plan) {
  auto indices = getPropagationStartingOffsetCanonical(plan);

  // Get the starting index into the volume formed by the loop bounds.
  // This is always ordered the same as the forward pass as this is
  // the way we allocate information to buckets hence should give the
  // fewest total iterations in general.
  std::vector<std::size_t> dimsToIterate, dimsToIterateFwd;

  // Multi-dimensional bounds of the loops we iterate.
  std::vector<unsigned> loopBounds;
  for (std::size_t dimIdx = 0; dimIdx < plan.partition.size(); ++dimIdx) {
    const auto dim = plan.dimShuffleToFwd.at(dimIdx);
    const auto partition = plan.partition.asStdVector().at(dim);
    if (partition > 1) {
      // We assume 'groups' does not have a partition for now.
      assert(dim >= 1);
      assert(dimIdx >= 1);
      dimsToIterate.emplace_back(dim - 1);
      dimsToIterateFwd.emplace_back(dimIdx - 1);
      loopBounds.emplace_back(partition);
    }
  }

  // Flatten the index into the shape given by the partitions
  // and then unflatten into the shape given by the partitions
  // of dimensions we need to iterate.
  const auto partitionFwd =
      shuffleVector(plan.partition.asStdVector(), plan.dimShuffleToFwd);
  const auto partitionIndexFlat = flattenIndexBoundless(partitionFwd, indices);
  const auto usedIndices =
      unflattenIndexBoundless(loopBounds, partitionIndexFlat);

  // We have indices for all used loop bounds, ordered for the forward pass.
  // We should return a full set of indices ordered for this pass.
  indices.clear();
  indices.resize(partitionFwd.size());
  assert(usedIndices.size() == dimsToIterate.size());
  assert(usedIndices.size() == dimsToIterateFwd.size());
  for (std::size_t i = 0; i < usedIndices.size(); ++i) {
    indices[dimsToIterateFwd[i] + 1] = usedIndices[i];
  }
  indices = shuffleVector(indices, inversePermutation(plan.dimShuffleToFwd));
  // No 'groups' dimension
  indices.erase(indices.begin());

  return std::make_tuple(std::move(dimsToIterate), std::move(indices));
}

constexpr static std::size_t numDirections = 3;
constexpr static std::size_t numBuffers = 2;

static Tensor getPassEndIndices(const Tensor &t, const SinglePassPlan &plan) {
  // First numDirections elements give end indices in terms of forward pass so
  // cut these out and shuffle them for this pass.
  std::vector<unsigned> shuffleIndices(numDirections);
  for (std::size_t dimIdx = 0; dimIdx < plan.dimShuffleToFwd.size(); ++dimIdx) {
    // Check groups are not shuffled between passes and ignore them
    assert(dimIdx > 0 || plan.dimShuffleToFwd.at(dimIdx) == 0);
    if (dimIdx > 0) {
      shuffleIndices[dimIdx - 1] = plan.dimShuffleToFwd[dimIdx] - 1;
    }
  }

  std::vector<Tensor> toConcat;
  for (const auto &idx : shuffleIndices) {
    toConcat.emplace_back(t.slice(idx, idx + 1));
  }

  return concat(toConcat);
}

static std::size_t getOuterLoopDim(const SinglePassPlan &plan) {
  // Outer loop is always the X dimension from the forward pass.
  // Find this from the plan in terms of the current pass.
  auto it =
      std::find(plan.dimShuffleToFwd.begin(), plan.dimShuffleToFwd.end(), 1);
  const std::size_t dim = std::distance(plan.dimShuffleToFwd.begin(), it);
  // We never expect this to be dim 0 (groups)
  assert(dim > 0);
  return dim - 1;
}

static Tensor getOuterLoopIterationsToSkip(const Tensor &t) {
  return t.slice(numDirections, t.numElements());
}

namespace {

// Handles building of the program to execute the fully connected layer
// implementation.
class ProgBuilder {
public:
  std::vector<Sequence> pre;
  std::vector<Sequence> distributionExchange;
  Sequence preDistribution;
  ComputeSet distributionCS;
  Sequence firstPropagationExchange;
  // [direction][buffer index]
  std::array<Program, numDirections> propagationExchanges;
  Program propagationCompute;
  std::vector<std::vector<ComputeSet>> reductionCSs;
  std::vector<Sequence> post;

  ProgBuilder(Graph &graph, const std::vector<unsigned> &hierarchy,
              const std::string &debugPrefix)
      : pre(hierarchy.size() + 1), distributionExchange(hierarchy.size() + 1),
        distributionCS(graph.addComputeSet(
            debugPrefix + "/ComputePartialsInitialDistribution")),
        reductionCSs(hierarchy.size()), post(hierarchy.size() + 1) {}

  void addToSequence(Graph &graph, Sequence &seq,
                     const SinglePassPlan &plan = {},
                     const boost::optional<Tensor> &overflowInfo = boost::none,
                     const std::string &debugPrefix = "") {
    for (unsigned level = 0; level < pre.size(); ++level) {
      seq.add(pre[level]);
      seq.add(distributionExchange[level]);
    }
    seq.add(preDistribution);
    seq.add(Execute(distributionCS));
    const bool needDynamic =
        plan.partition.x * plan.partition.y * plan.partition.z > 1;
    if (needDynamic && overflowInfo) {
      std::vector<Sequence> progs;
      progs.emplace_back();

      // Shuffle controlInfo to match this pass.
      auto endIndices = getPassEndIndices(overflowInfo.get(), plan);
      endIndices = popops::cast(graph, endIndices, UNSIGNED_INT, progs.back(),
                                debugPrefix + "/endIndices");
      const auto outerLoopDim = getOuterLoopDim(plan);
      const auto outerLoopIterationsToSkip =
          getOuterLoopIterationsToSkip(overflowInfo.get());

      progs.back().add(firstPropagationExchange);

      std::vector<std::size_t> dimsToIterate;
      std::vector<unsigned> startingIndices;
      std::tie(dimsToIterate, startingIndices) =
          getPropagationIteratedDimsAndStartingIndices(plan);

      const auto indices = graph.clone(endIndices, debugPrefix + "/indices");
      const auto zero = graph.addConstant(UNSIGNED_INT, {1}, 0);
      const auto one = graph.addConstant(UNSIGNED_INT, {1}, 1);
      const auto initialIndices =
          graph.addConstant(UNSIGNED_INT, endIndices.shape(),
                            ArrayRef<unsigned>(startingIndices));
      graph.setTileMapping(zero, 0);
      graph.setTileMapping(one, 0);
      graph.setTileMapping(initialIndices, 0);
      progs.back().add(Copy(initialIndices, indices));

      const std::array<std::string, numDirections> dimNames = {"X", "Y", "Z"};

      if (dimsToIterate.size() >= 1) {
        for (auto dimIt = dimsToIterate.rbegin();
             dimIt != std::prev(dimsToIterate.rend()); ++dimIt) {
          progs.emplace_back();

          progs.back().add(Copy(zero, indices[*dimIt]));
          popops::addInPlace(graph, indices[*std::next(dimIt)], one,
                             progs.back(),
                             debugPrefix + "/adjust" + dimNames[*dimIt] +
                                 "StartingIndicesToLoopBounds");

          auto prog = std::move(progs.back());
          progs.pop_back();
          Sequence condBody;
          const auto doesNotFitInLoopBound = popops::gteq(
              graph, indices[*dimIt], endIndices[*dimIt], progs.back(),
              debugPrefix + "/is" + dimNames[*dimIt] +
                  "StartingIndexOutsideLoopBounds");
          progs.back().add(
              If(doesNotFitInLoopBound, std::move(prog), Sequence()));
        }
      }
      for (std::size_t i = 0; i < dimsToIterate.size(); ++i) {
        progs.emplace_back();
      }
      progs.back().add(propagationCompute);
      for (auto dimIt = dimsToIterate.rbegin(); dimIt != dimsToIterate.rend();
           ++dimIt) {
        // If this is the outer-most loop, we have extra information that
        // allows us to potentially skip the body for this iteration.
        if (*dimIt == outerLoopDim) {
          auto prog = std::move(progs.back());
          progs.pop_back();
          progs.emplace_back();

          const auto doIteration = graph.addVariable(
              UNSIGNED_INT, {},
              debugPrefix + "/do" + dimNames[*dimIt] + "Iteration");
          graph.setTileMapping(doIteration, 0);
          const auto cs = graph.addComputeSet(debugPrefix + "/shouldDo" +
                                              dimNames[*dimIt] + "Iteration");
          const auto v =
              graph.addVertex(cs,
                              templateVertex("popsparse::BitIsSet",
                                             UNSIGNED_SHORT, UNSIGNED_INT),
                              {{"bits", outerLoopIterationsToSkip},
                               {"index", indices[*dimIt]},
                               {"out", doIteration}});
          graph.setTileMapping(v, 0);
          progs.back().add(Execute(cs));
          progs.back().add(
              Switch(doIteration, {{0, Sequence()}}, std::move(prog)));
        }
        progs.back().add(propagationExchanges.at(*dimIt));
        popops::addInPlace(graph, indices[*dimIt], one, progs.back(),
                           debugPrefix + "/increment" + dimNames[*dimIt] +
                               "Index");
        auto prog = std::move(progs.back());
        progs.pop_back();
        Sequence condBody;
        const auto ltDim =
            popops::lt(graph, indices[*dimIt], endIndices[*dimIt], condBody,
                       debugPrefix + "/isDim" + dimNames[*dimIt] + "Finished");
        progs.back().add(
            RepeatWhileTrue(std::move(condBody), ltDim, std::move(prog)));
        // Outer-most loop doesn't need to zero its index
        if (std::next(dimIt) != dimsToIterate.rend()) {
          progs.back().add(Copy(zero, indices[*dimIt]));
        }
      }

      assert(progs.size() == 1);
      seq.add(std::move(progs.back()));
    }
    for (int level = post.size() - 1; level >= 0; --level) {
      if (static_cast<unsigned>(level) < reductionCSs.size()) {
        for (const auto &cs : reductionCSs[level]) {
          seq.add(Execute(cs));
        }
      }
      seq.add(post[level]);
    }
  }
};

template <typename T> struct MetaInfoAndValues {
  T metaInfo;
  T values;
};

} // end anonymous namespace

static SinglePassPlan getFwdPlan(const Plan &plan) {
  SinglePassPlan p;
  p.grouping = plan.method.grouping;
  p.partition = plan.partition;
  p.initialDistributionBucketPartition =
      plan.initialDistributionBucketPartition;
  p.mappingOrder = plan.mappingOrder;
  p.nzElemsPerBucket = plan.nzElemsPerBucket;
  p.metaInfoElemsPerBucket = plan.fwdMetaInfoElemsPerBucket;
  p.method = plan.method.fwd;
  p.dimShuffleToFwd = {0, 1, 2, 3};
  return p;
}

static SinglePassPlan getGradAPlan(const Plan &plan) {
  SinglePassPlan p = getFwdPlan(plan);
  std::swap(p.grouping.x, p.grouping.y);
  std::swap(p.partition.x, p.partition.y);
  std::swap(p.initialDistributionBucketPartition.x,
            p.initialDistributionBucketPartition.y);
  p.metaInfoElemsPerBucket = plan.gradAMetaInfoElemsPerBucket;
  p.method = plan.method.gradA;
  p.dimShuffleToFwd = {0, 2, 1, 3};
  return p;
}

static SinglePassPlan getGradWPlan(const Plan &plan) {
  SinglePassPlan p = getFwdPlan(plan);
  // GradW doesn't use this for now.
  p.initialDistributionBucketPartition = p.partition;
  std::swap(p.grouping.y, p.grouping.z);
  std::swap(p.partition.y, p.partition.z);
  std::swap(p.initialDistributionBucketPartition.y,
            p.initialDistributionBucketPartition.z);
  p.metaInfoElemsPerBucket = plan.fwdMetaInfoElemsPerBucket;
  p.method = plan.method.gradW;
  p.dimShuffleToFwd = {0, 1, 3, 2};
  return p;
}

template <typename T, typename F>
static inline void iterateMultiDimSpace(const std::vector<T> &bounds,
                                        const F &f) {
  std::size_t carryDim;
  std::vector<T> i(bounds.size());
  do {
    f(i);
    carryDim = 0;
    while (carryDim < i.size()) {
      ++i[carryDim];
      if (i[carryDim] < bounds[carryDim]) {
        break;
      }
      i[carryDim] = 0;
      ++carryDim;
    }
  } while (carryDim < i.size());
}

static Tensor weightBucketsByPartition(const Tensor &weightBuckets,
                                       const Vector<unsigned> &partition) {
  const auto totalPartitions = product(partition.asStdVector());
  auto shape = partition.asStdVector<std::size_t>();
  assert(weightBuckets.dim(0) % totalPartitions == 0);
  const auto bucketsPerPartition = weightBuckets.dim(0) / totalPartitions;
  shape.insert(shape.end(), bucketsPerPartition);
  return weightBuckets.reshapePartial(0, 1, shape);
}

template <typename F>
static void iteratePartitions(const Vector<unsigned> &partition_, const F &f) {
  const auto &partition = partition_.asStdVector();
  iterateMultiDimSpace(partition,
                       [&](const auto &i) { f(Vector<unsigned>(i)); });
}

template <typename F>
static void iteratePartitions(const Vector<unsigned> &shape_,
                              const Vector<unsigned> &partition_,
                              const Vector<unsigned> &grouping_, const F &f) {
  const auto ceildivPred = [](const auto a, const auto b) {
    return ceildiv(a, b);
  };
  const auto minPred = [](const auto a, const auto b) {
    return std::min(a, b);
  };
  const auto &partition = partition_.asStdVector();
  const auto groupedShape = shape_ / grouping_;
  const auto partitionShape = groupedShape.binaryOp(partition_, ceildivPred);
  iterateMultiDimSpace(partition, [&](const auto &i_) {
    const Vector<unsigned> i(i_);
    // FIXME: Begin/end aren't actually correct because they are indices of
    // grouped shape this doesn't currently affect anything.
    const auto begin = (partitionShape * i).binaryOp(groupedShape, minPred);
    const auto end = (partitionShape + begin).binaryOp(groupedShape, minPred);
    f(i, begin, end);
  });
}

static Tensor stitchNextLevelOutputs(const Vector<unsigned> &partition_,
                                     std::vector<Tensor> ts) {
  const auto &partition = partition_.asStdVector();
  // Essentially a multi-stage reduction in-place where number of
  // stages is number of dimensions, factor to reduce by is
  // number of partitions in that dimension, and operation is
  // a concat in that dimension.
  auto numOutputs = product(partition);
  for (std::size_t d = 0; d < partition.size(); ++d) {
    auto reductionFactor = partition[d];
    numOutputs /= reductionFactor;
    for (std::size_t p = 1; p < reductionFactor; ++p) {
      for (std::size_t o = 0; o < numOutputs; ++o) {
        ts[o] = concat(ts[o], ts[p * numOutputs + o], d);
      }
    }
    assert(d != partition.size() - 1 || numOutputs == 1);
  }
  return std::move(ts[0]);
}

static unsigned getPartitionTile(const std::vector<unsigned> &hierarchy,
                                 const SinglePassPlan &plan,
                                 const std::vector<Vector<unsigned>> &indices) {
  assert(indices.size() == hierarchy.size());

  // We only support single-IPU currently.
  assert(hierarchy.size() == 1);

  unsigned tile = 0;
  for (unsigned level = 0; level < hierarchy.size(); ++level) {
    const auto &levelIndex = indices[level];
    const auto &levelPartition = plan.partition;
    const auto &levelIndexFwd =
        shuffleVector(levelIndex.asStdVector(), plan.dimShuffleToFwd);
    const auto &levelPartitionFwd =
        shuffleVector(levelPartition.asStdVector(), plan.dimShuffleToFwd);
    const auto levelPNId = getPNIdForPartition(
        plan.mappingOrder, levelPartitionFwd, levelIndexFwd);
    tile = tile * hierarchy[level] + levelPNId;
  }
  return tile;
}

static bool needsReduction(const Vector<unsigned> &partition) {
  return partition.y > 1;
}

static unsigned getSubGroupId(const SinglePassPlan &plan,
                              const std::vector<Vector<unsigned>> &indices) {
  // This is only valid at the tile-level.
  assert(indices.size() == 1);
  // Sub-group ID is given by X and Y partition indices of the forward pass.
  auto cumulativePartitions = Vector<unsigned>::generate([] { return 1u; });
  auto cumulativeIndex = Vector<unsigned>::generate([] { return 0u; });
  for (unsigned level = 0; level < indices.size(); ++level) {
    const auto &levelIndex = indices[level];
    const auto &levelPartition = plan.partition;
    cumulativePartitions *= levelPartition;
    cumulativeIndex *= levelPartition;
    cumulativeIndex += levelIndex;
  }
  const Vector<unsigned> cumulativePartitionFwd(
      shuffleVector(cumulativePartitions.asStdVector(), plan.dimShuffleToFwd));
  const Vector<unsigned> cumulativeIndexFwd(
      shuffleVector(cumulativeIndex.asStdVector(), plan.dimShuffleToFwd));
  return calculateSubGroupId(cumulativePartitionFwd.x, cumulativePartitionFwd.y,
                             cumulativeIndexFwd.x, cumulativeIndexFwd.y);
}

static Tensor groupActs(const Tensor &t, const Vector<unsigned> &grouping) {
  // The concept of grouping is only applied to the dense tensors
  // currently. Handling of anything but element-wise sparsity is
  // left as future work.
  const std::vector<std::size_t> actGrouping = {grouping.groups, grouping.y,
                                                grouping.z};

  assert(t.rank() == actGrouping.size());
  bool canBeGrouped = true;
  for (std::size_t d = 0; d < t.rank() && canBeGrouped; ++d) {
    canBeGrouped &= (t.dim(d) % actGrouping[d] == 0);
  }
  if (!canBeGrouped) {
    // TODO: Allocate and zero padding if it is needed...?
    // There is a question here about when exactly the padding is done.
    // Do we do it on-tile once all broadcasting of dense input is
    // complete?
    throw poputil::poplibs_error(
        "Padding of input to meet grouping not yet handled");
  }
  return factorDims(t, actGrouping);
}

// Tile-level
static void getNextLevelInputs(Graph &graph, const Vector<unsigned> &shape,
                               const std::vector<Vector<unsigned>> &indices,
                               const SinglePassPlan &plan,
                               const std::vector<unsigned> &hierarchy,
                               const unsigned level, const Tensor &acts,
                               Tensor &nextLevelInputs) {
  nextLevelInputs = acts;
}

// IPU-level (and above)
template <typename NextLevelInput>
static void getNextLevelInputs(Graph &graph, const Vector<unsigned> &shape,
                               const std::vector<Vector<unsigned>> &indices,
                               const SinglePassPlan &plan,
                               const std::vector<unsigned> &hierarchy,
                               const unsigned level, const Tensor &acts,
                               std::vector<NextLevelInput> &nextLevelInputs) {
  const auto &partition = plan.partition;
  const auto &grouping = plan.grouping;
  const auto totalPartitions = product(partition.asStdVector());
  nextLevelInputs.resize(totalPartitions);
  iteratePartitions(
      shape, partition, grouping,
      [&](const auto &i, const auto &begin, const auto &end) {
        const auto subShape = end - begin;
        const std::vector<std::size_t> actBegin = {begin.groups, begin.y,
                                                   begin.z};
        const std::vector<std::size_t> actEnd = {end.groups, end.y, end.z};
        const auto subActs = acts.slice(actBegin, actEnd);
        auto subIndices = indices;
        subIndices.emplace_back(i);
        getNextLevelInputs(graph, subShape, subIndices, plan, hierarchy,
                           level + 1, subActs,
                           nextLevelInputs[flattenIndex(partition.asStdVector(),
                                                        i.asStdVector())]);
      });
}

static void copyPartitions(Graph &graph, Sequence &prog,
                           const std::vector<Tensor> &src,
                           const std::vector<Tensor> &dst,
                           bool padSrc = false) {
  assert(src.size() == dst.size());
  // Just flatten to a single source/dest tensor!
  std::vector<Tensor> flatSrc, flatDst;
  for (std::size_t i = 0; i < src.size(); ++i) {
    auto s = src[i];
    auto d = dst[i];
    // TODO: For GradW the source can have a smaller shape
    // in its outer-most non-singular dimension. In this case
    // we can allow this and simply trim the destination
    // tensor to match the source.
    if (s.shape() != d.shape()) {
      std::vector<std::ptrdiff_t> paddingLower(s.rank(), 0);
      std::vector<std::ptrdiff_t> paddingUpper(s.rank());
      for (std::size_t dim = 0; dim < s.rank(); ++dim) {
        assert(s.dim(dim) <= d.dim(dim));
        assert(padSrc || s.dim(dim) == d.dim(dim));
        paddingUpper[dim] = -std::ptrdiff_t(d.dim(dim) - s.dim(dim));
      }
      // Truncate destination if necessary:
      d = popops::pad(graph, d, paddingLower, paddingUpper);
    }
    flatSrc.emplace_back(s.flatten());
    flatDst.emplace_back(d.flatten());
  }
  prog.add(Copy(concat(flatSrc), concat(flatDst)));
}

static void allocatePerPartitionInputs(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Type &inputType, const std::vector<unsigned> &hierarchy,
    const unsigned level, Tensor &input, const std::string &debugName) {
  const auto &grouping = plan.grouping;
  const auto inputMemOrdering = getOnTileActsOrdering(plan.method);
  const std::vector<std::size_t> shapeInternal = {
      shape.groups * grouping.groups, shape.y * grouping.y,
      shape.z * grouping.z};
  std::vector<std::size_t> shapeAllocation(inputMemOrdering.size());
  for (std::size_t i = 0; i < inputMemOrdering.size(); ++i) {
    shapeAllocation[i] = shapeInternal[inputMemOrdering[i]];
  }
  input = graph.addVariable(inputType, shapeAllocation, debugName)
              .dimShuffle(inversePermutation(inputMemOrdering));
  const std::vector<std::size_t> inputGrouping = {grouping.groups, grouping.y,
                                                  grouping.z};
  input = factorDims(input, inputGrouping);
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  graph.setTileMapping(input, tile);
}

template <typename NextLevelInput>
static void allocatePerPartitionInputs(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Type &inputType, const std::vector<unsigned> &hierarchy,
    const unsigned level, std::vector<NextLevelInput> &perPartitionInputs,
    const std::string &debugName) {
  const auto &partition = plan.partition;
  const auto totalPartitions = product(partition.asStdVector());
  perPartitionInputs.resize(totalPartitions);
  iteratePartitions(
      shape, plan.partition, plan.grouping,
      [&](const auto &i, const auto &begin, const auto &end) {
        const auto subShape = end - begin;
        auto subIndices = indices;
        subIndices.emplace_back(i);
        const auto partitionIndexFlat =
            flattenIndex(partition.asStdVector(), i.asStdVector());
        allocatePerPartitionInputs(
            graph, subShape, subIndices, plan, inputType, hierarchy, level + 1,
            perPartitionInputs[partitionIndexFlat], debugName);
      });
}

// Tile-level
static void getNextLevelWeights(Graph &graph, const Vector<unsigned> &shape,
                                const std::vector<Vector<unsigned>> &indices,
                                const SinglePassPlan &plan,
                                const std::vector<unsigned> &hierarchy,
                                const unsigned level, const Tensor &weights,
                                Tensor &nextLevelWeights) {
  nextLevelWeights = weights;
}

// IPU-level (and above)
template <typename NextLevelInput>
static void getNextLevelWeights(Graph &graph, const Vector<unsigned> &shape,
                                const std::vector<Vector<unsigned>> &indices,
                                const SinglePassPlan &plan,
                                const std::vector<unsigned> &hierarchy,
                                const unsigned level, const Tensor &weights,
                                std::vector<NextLevelInput> &nextLevelWeights) {
  const auto &partition = plan.partition;
  const auto &grouping = plan.grouping;
  const auto totalPartitions = product(partition.asStdVector());
  nextLevelWeights.resize(totalPartitions);
  iteratePartitions(
      shape, partition, grouping,
      [&](const auto &i, const auto &begin, const auto &end) {
        const auto subShape = end - begin;
        const std::vector<std::size_t> weightBegin = {begin.groups, begin.x,
                                                      begin.y};
        const std::vector<std::size_t> weightEnd = {end.groups, end.x, end.y};
        const auto subWeights = weights.slice(weightBegin, weightEnd);
        auto subIndices = indices;
        subIndices.emplace_back(i);
        getNextLevelWeights(graph, subShape, subIndices, plan, hierarchy,
                            level + 1, subWeights,
                            nextLevelWeights[flattenIndex(
                                partition.asStdVector(), i.asStdVector())]);
      });
}

// Tile-level
static void getNextLevelDistributionBuckets(Graph &graph,
                                            const SinglePassPlan &plan,
                                            const Options &options,
                                            const Tensor &buckets,
                                            Tensor &nextLevelBuckets) {
  nextLevelBuckets = buckets;
}

// IPU-level
template <typename NextLevelBuckets>
static void getNextLevelDistributionBuckets(
    Graph &graph, const SinglePassPlan &plan, const Options &options,
    const Tensor &buckets, std::vector<NextLevelBuckets> &nextLevelBuckets) {
  const auto &partition = plan.partition;
  const auto &bucketPartition = plan.initialDistributionBucketPartition;
  const auto totalPartitions = product(partition.asStdVector());
  nextLevelBuckets.resize(totalPartitions);
  const auto &bucketsByPartition =
      weightBucketsByPartition(buckets, bucketPartition);
  iteratePartitions(partition, [&](const auto &i) {
    const auto bucketsPerPartition = partition / bucketPartition;
    const auto bucketPartitionStart = i / bucketsPerPartition;
    const auto bucketPartitionEnd = bucketPartitionStart + Vector<unsigned>(1);

    std::vector<std::size_t> squeezeDims(i.size());
    std::iota(squeezeDims.begin(), squeezeDims.end(), 0);
    const auto subBuckets =
        bucketsByPartition
            .slice(bucketPartitionStart.template asStdVector<std::size_t>(),
                   bucketPartitionEnd.template asStdVector<std::size_t>())
            .squeeze(squeezeDims);
    assert(subBuckets.rank() == 2);
    const auto partitionIndexFlat =
        flattenIndex(partition.asStdVector(), i.asStdVector());
    getNextLevelDistributionBuckets(graph, plan, options, subBuckets,
                                    nextLevelBuckets[partitionIndexFlat]);
  });
}

static void getBucketsByPartition(const SinglePassPlan &plan,
                                  const Tensor &bucket,
                                  Tensor &nextLevelBucket) {
  nextLevelBucket = bucket;
}

template <typename NextLevelBuckets>
static void
getBucketsByPartition(const SinglePassPlan &plan, const Tensor &buckets,
                      std::vector<NextLevelBuckets> &nextLevelBuckets) {
  nextLevelBuckets.resize(product(plan.partition.asStdVector()));
  const auto &bucketsByPartition =
      weightBucketsByPartition(buckets, plan.partition);
  iteratePartitions(plan.partition, [&](const auto &i) {
    const auto iPlusOne = i + Vector<unsigned>(1);
    std::vector<std::size_t> squeezeDims(i.size());
    std::iota(squeezeDims.begin(), squeezeDims.end(), 0);
    const auto sBuckets =
        bucketsByPartition
            .slice(i.template asStdVector<std::size_t>(),
                   iPlusOne.template asStdVector<std::size_t>())
            .squeeze(squeezeDims);
    assert(sBuckets.rank() == 2);
    const auto partitionIndexFlat =
        flattenIndex(plan.partition.asStdVector(), i.asStdVector());
    getBucketsByPartition(plan, sBuckets, nextLevelBuckets[partitionIndexFlat]);
  });
}

// Tile-level
static void createPartialsDense(Graph &graph, const Vector<unsigned> &shape,
                                const std::vector<Vector<unsigned>> &indices,
                                const SinglePassPlan &plan,
                                const Options &options,
                                const std::vector<unsigned> &hierarchy,
                                unsigned level, Tensor &partials,
                                const std::string &debugName) {
  const auto &grouping = plan.grouping;
  const std::vector<std::size_t> partialsShape = {
      shape.groups * grouping.groups, shape.x * grouping.x,
      shape.z * grouping.z};
  const auto partialsMemOrdering = getOnTilePartialsOrdering(plan.method);
  std::vector<std::size_t> partialsShapeAllocation(partialsShape.size());
  for (std::size_t i = 0; i < partialsMemOrdering.size(); ++i) {
    partialsShapeAllocation[i] = partialsShape[partialsMemOrdering[i]];
  }

  partials =
      graph
          .addVariable(options.partialsType, partialsShapeAllocation, debugName)
          .dimShuffle(inversePermutation(partialsMemOrdering));

  const std::vector<std::size_t> partialsGrouping = {
      plan.grouping.groups, plan.grouping.x, plan.grouping.z};
  partials = factorDims(partials, partialsGrouping);
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  graph.setTileMapping(partials, tile);
}

// IPU-level (and above)
template <typename NextLevelPartials>
static void
createPartialsDense(Graph &graph, const Vector<unsigned> &shape,
                    const std::vector<Vector<unsigned>> &indices,
                    const SinglePassPlan &plan, const Options &options,
                    const std::vector<unsigned> &hierarchy, unsigned level,
                    std::vector<NextLevelPartials> &partials,
                    const std::string &debugName) {
  const auto &partition = plan.partition;
  const auto &grouping = plan.grouping;
  const auto totalPartitions = product(partition.asStdVector());
  partials.resize(totalPartitions);
  iteratePartitions(shape, partition, grouping,
                    [&](const auto &i, const auto &begin, const auto &end) {
                      const auto partitionIndexFlat = flattenIndex(
                          partition.asStdVector(), i.asStdVector());
                      auto subIndices = indices;
                      subIndices.emplace_back(i);
                      const auto subShape = end - begin;
                      createPartialsDense(
                          graph, subShape, subIndices, plan, options, hierarchy,
                          level + 1, partials[partitionIndexFlat], debugName);
                    });
}

// Tile-level
static void createPartialsSparse(Graph &graph, const Vector<unsigned> &shape,
                                 const std::vector<Vector<unsigned>> &indices,
                                 const SinglePassPlan &plan,
                                 const Options &options,
                                 const std::vector<unsigned> &hierarchy,
                                 unsigned level, Tensor &partials,
                                 const std::string &debugName) {
  // We include partitions in the shape of partials. This makes it easier
  // to get back to the output shape.
  partials = graph.addVariable(
      options.partialsType, {1, 1, 1, 1, 1, plan.nzElemsPerBucket}, debugName);
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  graph.setTileMapping(partials, tile);
}

// IPU-level (and above)
template <typename NextLevelPartials>
static void
createPartialsSparse(Graph &graph, const Vector<unsigned> &shape,
                     const std::vector<Vector<unsigned>> &indices,
                     const SinglePassPlan &plan, const Options &options,
                     const std::vector<unsigned> &hierarchy, unsigned level,
                     std::vector<NextLevelPartials> &partials,
                     const std::string &debugName) {
  const auto &partition = plan.partition;
  const auto &grouping = plan.grouping;
  const auto totalPartitions = product(partition.asStdVector());
  partials.resize(totalPartitions);
  iteratePartitions(shape, partition, grouping,
                    [&](const auto &i, const auto &begin, const auto &end) {
                      const auto partitionIndexFlat = flattenIndex(
                          partition.asStdVector(), i.asStdVector());
                      auto subIndices = indices;
                      subIndices.emplace_back(i);
                      const auto subShape = end - begin;
                      createPartialsSparse(
                          graph, subShape, subIndices, plan, options, hierarchy,
                          level + 1, partials[partitionIndexFlat], debugName);
                    });
}

static void getSubGroupIds(Graph &graph,
                           const std::vector<Vector<unsigned>> &indices,
                           const SinglePassPlan &plan, const Options &options,
                           const std::vector<unsigned> &hierarchy, unsigned &id,
                           const std::string &debugName) {
  id = getSubGroupId(plan, indices);
}

static void getSubGroupIds(Graph &graph,
                           const std::vector<Vector<unsigned>> &indices,
                           const SinglePassPlan &plan, const Options &options,
                           const std::vector<unsigned> &hierarchy, Tensor &id,
                           const std::string &debugName) {
  const auto val = getSubGroupId(plan, indices);
  id = graph.addConstant(UNSIGNED_SHORT, {1}, val, debugName);
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  graph.setTileMapping(id, tile);
}

template <typename NextLevelIDs>
static void
getSubGroupIds(Graph &graph, const std::vector<Vector<unsigned>> &indices,
               const SinglePassPlan &plan, const Options &options,
               const std::vector<unsigned> &hierarchy,
               std::vector<NextLevelIDs> &ids, const std::string &debugName) {
  ids.resize(product(plan.partition.asStdVector()));
  iteratePartitions(plan.partition, [&](const auto &i) {
    const auto partitionIndexFlat =
        flattenIndex(plan.partition.asStdVector(), i.asStdVector());
    auto subIndices = indices;
    subIndices.emplace_back(i);
    getSubGroupIds(graph, subIndices, plan, options, hierarchy,
                   ids[partitionIndexFlat], debugName);
  });
}

static void compute(Graph &graph, const ComputeSet &cs,
                    const Vector<unsigned> &shape,
                    const std::vector<Vector<unsigned>> &indices,
                    const SinglePassPlan &plan, const Options &options,
                    const std::vector<unsigned> &hierarchy, unsigned level,
                    bool zeroPartials, const Tensor &acts,
                    const Tensor &weights, const Tensor &metaInfo,
                    const Tensor &partials,
                    const boost::variant<unsigned, Tensor> &subGroupId,
                    const std::string &debugPrefix) {
  const std::string levelPrefix = debugPrefix + "/l" + std::to_string(level);
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  const std::array<std::size_t, 2> blockDimensions = {plan.grouping.x,
                                                      plan.grouping.y};
  onTileImpl(graph, cs, tile, plan.method, zeroPartials, subGroupId,
             shape.asStdVector<std::size_t>(), metaInfo, weights, acts,
             partials, blockDimensions, levelPrefix);
}

template <typename NextLevelInput, typename NextLevelWeights,
          typename NextLevelMetaInfo, typename NextLevelPartials,
          typename NextLevelSubGroupIds>
static void
compute(Graph &graph, const ComputeSet &cs, const Vector<unsigned> &shape,
        const std::vector<Vector<unsigned>> &indices,
        const SinglePassPlan &plan, const Options &options,
        const std::vector<unsigned> &hierarchy, unsigned level,
        bool zeroPartials, const std::vector<NextLevelInput> &nextLevelInputs,
        const std::vector<NextLevelWeights> &nextLevelWeights,
        const std::vector<NextLevelMetaInfo> &nextLevelMetaInfo,
        const std::vector<NextLevelPartials> &partials,
        const std::vector<NextLevelSubGroupIds> &subGroupIds,
        const std::string &debugPrefix) {
  const auto &partition = plan.partition;
  const auto &grouping = plan.grouping;
  iteratePartitions(
      shape, partition, grouping,
      [&](const auto &i, const auto &begin, const auto &end) {
        auto subIndices = indices;
        subIndices.emplace_back(i);
        const auto subShape = end - begin;
        const auto partitionIndexFlat =
            flattenIndex(partition.asStdVector(), i.asStdVector());
        compute(graph, cs, subShape, subIndices, plan, options, hierarchy,
                level + 1, zeroPartials, nextLevelInputs[partitionIndexFlat],
                nextLevelWeights[partitionIndexFlat],
                nextLevelMetaInfo[partitionIndexFlat],
                partials[partitionIndexFlat], subGroupIds[partitionIndexFlat],
                debugPrefix);
      });
}

static Tensor finalReduction(Graph &graph, ProgBuilder &progBuilder,
                             const Vector<unsigned> &shape,
                             const Type &resultType,
                             const std::vector<Vector<unsigned>> &indices,
                             const SinglePassPlan &plan, const Options &options,
                             const std::vector<unsigned> &hierarchy,
                             unsigned level, const Tensor &partials,
                             bool forGradW, const std::string &debugPrefix) {
  const std::string levelPrefix = debugPrefix + "/l" + std::to_string(level);
  const Type &levelResultType = options.partialsType;

  Tensor result = partials;
  if (result.elementType() != levelResultType) {
    result =
        popops::cast(graph, result, levelResultType, progBuilder.post.at(level),
                     levelPrefix + "/castResult");
  }
  return result;
}

template <typename NextLevelPartials>
static Tensor
finalReduction(Graph &graph, ProgBuilder &progBuilder,
               const Vector<unsigned> &shape, const Type &resultType,
               const std::vector<Vector<unsigned>> &indices,
               const SinglePassPlan &plan, const Options &options,
               const std::vector<unsigned> &hierarchy, unsigned level,
               const std::vector<NextLevelPartials> &partials, bool forGradW,
               const std::string &debugPrefix) {
  const std::string levelPrefix = debugPrefix + "/l" + std::to_string(level);
  const Type levelResultType = level == 0 ? resultType : options.partialsType;

  const auto &partition = plan.partition;
  const auto &grouping = plan.grouping;
  const auto totalPartitions = product(partition.asStdVector());
  std::vector<Tensor> nextLevelOutputs(totalPartitions);
  iteratePartitions(shape, partition, grouping,
                    [&](const auto &i, const auto &begin, const auto &end) {
                      const auto partitionIndexFlat = flattenIndex(
                          partition.asStdVector(), i.asStdVector());
                      auto sIndices = indices;
                      sIndices.emplace_back(i);
                      const auto sShape = end - begin;
                      auto output = finalReduction(
                          graph, progBuilder, sShape, resultType, sIndices,
                          plan, options, hierarchy, level + 1,
                          partials[partitionIndexFlat], forGradW, debugPrefix);
                      if (!forGradW) {
                        output = output.expand({2});
                      }
                      nextLevelOutputs[partitionIndexFlat] = output;
                    });

  Tensor result = stitchNextLevelOutputs(partition, nextLevelOutputs);

  if (!forGradW) {
    constexpr std::size_t partialsDim = 2;
    if (needsReduction(partition)) {
      // TODO: Make ConvReduce just take bool args or its own
      // options structure rather than having to grab this unrelated
      // convolution options type.
      poplin::ConvOptions convOpts{};
      convOpts.enableMultiStageReduce = false;

      const std::vector<std::size_t> partialsGrouping = {
          grouping.groups, grouping.x, grouping.z};

      // Ensure partials are ordered in tensor as they are in memory.
      std::vector<unsigned> partialsMemOrder =
          getOnTilePartialsOrdering(plan.method);
      auto partialsGroupingMemOrder = partialsGrouping;
      for (std::size_t i = 0; i < partialsMemOrder.size(); ++i) {
        partialsGroupingMemOrder[i] = partialsGrouping[partialsMemOrder[i]];
        partialsMemOrder[i]++;
      }
      std::vector<unsigned> partialsCurrentOrder(partialsMemOrder.size());
      std::iota(partialsCurrentOrder.begin(), partialsCurrentOrder.end(), 1);
      // Roll dimension to be reduced to outer dimension, and remove grouping
      // from the tensor shape, then shuffle dimensions into their ordering
      // in memory.
      const auto partialsInMemOrder =
          unfactorDims(result.dimRoll(partialsDim, 0), partialsMemOrder.size(),
                       1)
              .dimShufflePartial(partialsCurrentOrder, partialsMemOrder);

      // Conv reductions take a grouping in the inner-most dimension so apply
      // the grouping we have in the innermost dimension in memory.
      const auto partialsInMemOrderWithGrouping =
          factorDims(partialsInMemOrder, {partialsGroupingMemOrder.back()},
                     partialsInMemOrder.rank() - 1);
      result = poplin::multiStageGroupedReduce(
          graph, partialsInMemOrderWithGrouping, resultType,
          progBuilder.reductionCSs.at(level), convOpts, levelPrefix);
      // Remove grouping of inner-most dimension as used by conv reductions.
      result = unfactorDims(result, 1, result.rank() - 2);
      // Reorder back to internal dimension ordering and restore groupings.
      result = result.expand({0})
                   .dimShufflePartial(partialsMemOrder, partialsCurrentOrder)
                   .squeeze({0});
      result = factorDims(result, partialsGrouping);
    } else {
      result = result.squeeze({2});
    }
  }

  if (result.elementType() != levelResultType) {
    result = popops::cast(graph, result, resultType, progBuilder.post.at(level),
                          levelPrefix + "/castResult");
  }

  return result;
}

static void writeUndefPartitions(Sequence &prog,
                                 const std::vector<Tensor> &ts) {
  std::vector<Tensor> flattenedTs;
  for (const auto &t : ts) {
    flattenedTs.emplace_back(t.flatten());
  }
  prog.add(WriteUndef(concat(flattenedTs)));
}

static void mapBuckets(Graph &graph,
                       const std::vector<Vector<unsigned>> &indices,
                       const SinglePassPlan &plan,
                       const std::vector<unsigned> &hierarchy,
                       const Tensor &buckets) {
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  graph.setTileMapping(buckets, tile);
}

template <typename NextLevelBuckets>
static void
mapBuckets(Graph &graph, const std::vector<Vector<unsigned>> &indices,
           const SinglePassPlan &plan, const std::vector<unsigned> &hierarchy,
           const NextLevelBuckets &buckets) {
  iteratePartitions(plan.partition, [&](const auto &i) {
    auto sIndices = indices;
    sIndices.emplace_back(i);
    const auto partitionIndexFlat =
        flattenIndex(plan.partition.asStdVector(), i.asStdVector());
    mapBuckets(graph, sIndices, plan, hierarchy, buckets[partitionIndexFlat]);
  });
}

static Tensor createBuckets(Graph &graph, const Type &type,
                            const SinglePassPlan &plan,
                            const std::size_t elemsPerBucket,
                            const std::vector<unsigned> &hierarchy,
                            const std::string &debugName) {
  const std::size_t numBuckets = product(plan.partition.asStdVector());
  const auto buckets =
      graph.addVariable(type, {numBuckets, elemsPerBucket}, debugName);
  std::vector<Tensor> bucketsByPartition;
  getBucketsByPartition(plan, buckets, bucketsByPartition);
  mapBuckets(graph, {}, plan, hierarchy, bucketsByPartition);
  return buckets;
}

static void createPropagationBuffers(
    Graph &graph, const Type &inputType, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Options &options, const std::vector<unsigned> &hierarchy,
    unsigned level, const Tensor &reference, const Tensor &allocationReference,
    Tensor &buffer, const std::string &debugName) {
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  buffer = graph.clone(allocationReference, debugName);
  graph.setTileMapping(buffer, tile);
}

template <typename NextLevelBuffers>
static void createPropagationBuffers(
    Graph &graph, const Type &inputType, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Options &options, const std::vector<unsigned> &hierarchy,
    unsigned level, const std::vector<NextLevelBuffers> &reference,
    const Tensor &, std::vector<NextLevelBuffers> &buffers,
    const std::string &debugPrefix) {
  buffers.resize(product(plan.partition.asStdVector()));
  // Note this is done under the assumption that the first partition is
  // always the max shape possible based on our partitioning algorithm.
  const auto allocReference = reference.at(0);
  iteratePartitions(
      shape, plan.partition, plan.grouping,
      [&](const auto &i, const auto &begin, const auto &end) {
        const auto sShape = end - begin;
        auto sIndices = indices;
        sIndices.emplace_back(i);
        const auto partitionIndexFlat =
            flattenIndex(plan.partition.asStdVector(), i.asStdVector());
#ifndef NDEBUG
        for (std::size_t d = 0; d < allocReference.rank(); ++d) {
          assert(reference[partitionIndexFlat].dim(d) <= allocReference.dim(d));
        }
#endif
        createPropagationBuffers(graph, inputType, sShape, sIndices, plan,
                                 options, hierarchy, level + 1,
                                 reference[partitionIndexFlat], allocReference,
                                 buffers[partitionIndexFlat], debugPrefix);
      });
}

static void getPropagationExchangeSources(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Options &options, const std::vector<unsigned> &hierarchy,
    const unsigned level, const Vector<unsigned> &offset, const Tensor &buckets,
    const Tensor &prevBuckets, Tensor &nextLevelSources,
    const std::string &debugPrefix) {
  nextLevelSources = prevBuckets;
}

template <typename NextLevelBuckets, typename NextLevelExchangeSources>
static void getPropagationExchangeSources(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Options &options, const std::vector<unsigned> &hierarchy,
    const unsigned level, const Vector<unsigned> &offset,
    const std::vector<NextLevelBuckets> &buckets,
    const std::vector<NextLevelBuckets> &prevBuckets,
    std::vector<NextLevelExchangeSources> &nextLevelSources,
    const std::string &debugPrefix) {
  nextLevelSources.resize(product(plan.partition.asStdVector()));
  iteratePartitions(
      shape, plan.partition, plan.grouping,
      [&](const auto &i, const auto &begin, const auto &end) {
        const auto sShape = end - begin;
        auto sIndices = indices;
        sIndices.emplace_back(i);

        // We find and pass down 'previous buckets/buffers' for the sources of
        // exchanges at each level though currently the only level that handles
        // this is the tile level.
        const auto prevI = (i + (plan.partition - offset)) % plan.partition;
        // Pick out current (i) and previous (prevI) buckets that will get
        // passed on to the next level.
        const auto idxFlat =
            flattenIndex(plan.partition.asStdVector(), i.asStdVector());
        const auto prevIdxFlat =
            flattenIndex(plan.partition.asStdVector(), prevI.asStdVector());
        getPropagationExchangeSources(graph, sShape, sIndices, plan, options,
                                      hierarchy, level + 1, offset,
                                      buckets[idxFlat], buckets[prevIdxFlat],
                                      nextLevelSources[idxFlat], debugPrefix);
      });
}

static unsigned int getTotalMetaInfoElemsPerBuckets(const Plan &plan) {
  return plan.fwdMetaInfoElemsPerBucket +
         (plan.sharedBuckets() ? 0 : plan.gradAMetaInfoElemsPerBucket);
}

static void addBufferIncrementProg(Graph &graph, const Tensor &t, Sequence &seq,
                                   const std::string &debugPrefix) {
  auto cs = graph.addComputeSet(debugPrefix + "/bufIncrement");
  auto v = graph.addVertex(
      cs, templateVertex("popsparse::BufferIndexUpdate", t.elementType()));
  graph.connect(v["index"], t);
  graph.setTileMapping(v, 0);
  seq.add(Execute(cs));
}

static void addPropagationExchanges(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const SinglePassPlan &plan, const Options &options,
    const std::vector<unsigned> &hierarchy,
    const MetaInfoAndValues<std::vector<Tensor>> &buckets,
    const MetaInfoAndValues<std::array<std::vector<Tensor>, 2>> &buffers,
    const Tensor &bufferIdx, const std::string &debugPrefix) {
  MetaInfoAndValues<std::vector<Tensor>> homeSources;
  const auto homeOffset = getPropagationStartingOffset(plan);
  getPropagationExchangeSources(graph, shape, {}, plan, options, hierarchy, 0,
                                homeOffset, buckets.metaInfo, {},
                                homeSources.metaInfo, debugPrefix);
  getPropagationExchangeSources(graph, shape, {}, plan, options, hierarchy, 0,
                                homeOffset, buckets.values, {},
                                homeSources.values, debugPrefix);
  // We also set buffer index before exchanging.
  constexpr std::size_t initialBufferIdx = 0;
  const auto bufferIdxInitialVal =
      graph.addConstant(UNSIGNED_INT, {1}, initialBufferIdx);
  graph.setTileMapping(bufferIdxInitialVal, 0);
  progBuilder.firstPropagationExchange.add(
      Copy(bufferIdxInitialVal, bufferIdx));
  copyPartitions(graph, progBuilder.firstPropagationExchange,
                 homeSources.metaInfo, buffers.metaInfo.at(initialBufferIdx));
  copyPartitions(graph, progBuilder.firstPropagationExchange,
                 homeSources.values, buffers.values.at(initialBufferIdx));
  const auto getDir = [](std::size_t dirIdx) -> Vector<unsigned> {
    std::vector<unsigned> v(4, 0);
    v[1 + dirIdx] = 1;
    return v;
  };
  // We have 3 directions, X, Y, and Z, in which to propagate.
  for (std::size_t dirIdx = 0; dirIdx < numDirections; ++dirIdx) {
    const auto offset = getDir(dirIdx);
    std::array<Sequence, numBuffers> exchangeProgs;
    for (std::size_t destBuffer = 0; destBuffer < numBuffers; ++destBuffer) {
      MetaInfoAndValues<std::vector<Tensor>> bufferSources;
      const auto sourceBuffer = (destBuffer + (numBuffers - 1)) % numBuffers;
      getPropagationExchangeSources(graph, shape, {}, plan, options, hierarchy,
                                    0, offset,
                                    buffers.metaInfo.at(sourceBuffer), {},
                                    bufferSources.metaInfo, debugPrefix);
      getPropagationExchangeSources(graph, shape, {}, plan, options, hierarchy,
                                    0, offset, buffers.values.at(sourceBuffer),
                                    {}, bufferSources.values, debugPrefix);
      copyPartitions(graph, exchangeProgs.at(destBuffer),
                     bufferSources.metaInfo, buffers.metaInfo.at(destBuffer));
      copyPartitions(graph, exchangeProgs.at(destBuffer), bufferSources.values,
                     buffers.values.at(destBuffer));
    }
    // We also toggle buffer index before exchanging
    Sequence prog;
    // Toggle buffer index
    addBufferIncrementProg(graph, bufferIdx, prog,
                           debugPrefix + "/toggleBuffer");
    prog.add(Switch(bufferIdx, {{0, exchangeProgs.at(false)}},
                    exchangeProgs.at(true)));
    progBuilder.propagationExchanges.at(dirIdx) = prog;
  }
}

static void addPropagationExchangesGradW(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const SinglePassPlan &plan, const Options &options,
    const std::vector<unsigned> &hierarchy, const std::vector<Tensor> &inputs,
    const std::vector<Tensor> &weights, const std::vector<Tensor> &subGroupIds,
    const std::array<std::vector<Tensor>, numBuffers> &inputBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &weightBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &subGroupIdBuffers,
    const Tensor &bufferIdx, const std::string &debugPrefix) {
  std::vector<Tensor> inputHomeSources, weightHomeSources,
      subGroupIdHomeSources;
  // This is opposite direction to Fwd/GradA
  const auto homeOffset =
      (plan.partition - getPropagationStartingOffset(plan)) % plan.partition;
  constexpr std::size_t initialBufferIdx = 0b000;
  const auto bufferIdxInitialVal =
      graph.addConstant(UNSIGNED_INT, {1}, initialBufferIdx);
  graph.setTileMapping(bufferIdxInitialVal, 0);
  progBuilder.firstPropagationExchange.add(
      Copy(bufferIdxInitialVal, bufferIdx));
  getPropagationExchangeSources(graph, shape, {}, plan, options, hierarchy, 0,
                                homeOffset, inputs, {}, inputHomeSources,
                                debugPrefix);
  copyPartitions(graph, progBuilder.firstPropagationExchange, inputHomeSources,
                 inputBuffers.at(initialBufferIdx), true);
  getPropagationExchangeSources(graph, shape, {}, plan, options, hierarchy, 0,
                                homeOffset, weights, {}, weightHomeSources,
                                debugPrefix);
  copyPartitions(graph, progBuilder.firstPropagationExchange, weightHomeSources,
                 weightBuffers.at(initialBufferIdx), true);
  getPropagationExchangeSources(graph, shape, {}, plan, options, hierarchy, 0,
                                homeOffset, subGroupIds, {},
                                subGroupIdHomeSources, debugPrefix);
  copyPartitions(graph, progBuilder.firstPropagationExchange,
                 subGroupIdHomeSources, subGroupIdBuffers.at(initialBufferIdx));

  // This is opposite direction to Fwd/GradA
  const auto getDir = [&](std::size_t dirIdx) -> Vector<unsigned> {
    std::vector<unsigned> v(4, 0);
    const auto partition = plan.partition.asStdVector();
    v[1 + dirIdx] = (partition[1 + dirIdx] - 1) % partition[1 + dirIdx];
    return v;
  };
  // We have 3 directions, X, Y, and Z, in which to propagate.
  for (std::size_t dirIdx = 0; dirIdx < numDirections; ++dirIdx) {
    const auto offset = getDir(dirIdx);
    Switch exchangeSwitch(bufferIdx);
    const bool moveInput = offset.y + offset.z > 0;
    const bool moveWeights = offset.x + offset.y > 0;
    const bool moveSubGroupIds = offset.x + offset.z > 0;
    for (std::size_t inputDestBuffer = 0; inputDestBuffer < numBuffers;
         ++inputDestBuffer) {
      const auto inputSourceBuffer =
          (inputDestBuffer + (numBuffers - 1)) % numBuffers;
      for (std::size_t weightDestBuffer = 0; weightDestBuffer < numBuffers;
           ++weightDestBuffer) {
        const auto weightSourceBuffer =
            (weightDestBuffer + (numBuffers - 1)) % numBuffers;
        for (std::size_t subGroupIdDestBuffer = 0;
             subGroupIdDestBuffer < numBuffers; ++subGroupIdDestBuffer) {
          const auto subGroupIdSourceBuffer =
              (subGroupIdDestBuffer + (numBuffers - 1)) % numBuffers;
          Sequence exchangeProg;
          // These are conditional as if motion of operands is only in
          // a direction that the operand is duplicated across we do
          // not need to move it.
          if (moveInput) {
            std::vector<Tensor> sources;
            getPropagationExchangeSources(
                graph, shape, {}, plan, options, hierarchy, 0, offset,
                inputBuffers.at(inputSourceBuffer), {}, sources, debugPrefix);
            copyPartitions(graph, exchangeProg, sources,
                           inputBuffers.at(inputDestBuffer));
          }
          if (moveWeights) {
            std::vector<Tensor> sources;
            getPropagationExchangeSources(
                graph, shape, {}, plan, options, hierarchy, 0, offset,
                weightBuffers.at(weightSourceBuffer), {}, sources, debugPrefix);
            copyPartitions(graph, exchangeProg, sources,
                           weightBuffers.at(weightDestBuffer));
          }
          if (moveSubGroupIds) {
            std::vector<Tensor> sources;
            getPropagationExchangeSources(
                graph, shape, {}, plan, options, hierarchy, 0, offset,
                subGroupIdBuffers.at(subGroupIdSourceBuffer), {}, sources,
                debugPrefix);
            copyPartitions(graph, exchangeProg, sources,
                           subGroupIdBuffers.at(subGroupIdDestBuffer));
          }
          exchangeSwitch.add(inputDestBuffer * numBuffers * numBuffers +
                                 weightDestBuffer * numBuffers +
                                 subGroupIdDestBuffer,
                             std::move(exchangeProg));
        }
      }
    }
    Sequence prog;
    // Calculate buffer index.
    const auto bitsToFlip = graph.addConstant(
        UNSIGNED_INT, {1},
        (unsigned(moveInput) << 2u | unsigned(moveWeights) << 1u |
         unsigned(moveSubGroupIds) << 0u));
    graph.setTileMapping(bitsToFlip, 0);
    popops::bitwiseXorInPlace(graph, bufferIdx, bitsToFlip, prog,
                              debugPrefix + "/toggleBuffers");
    prog.add(std::move(exchangeSwitch));
    progBuilder.propagationExchanges.at(dirIdx) = std::move(prog);
  }
}

// Handles implementation of sparse * dense = dense matmul as one
// pass of a fully connected layer. This means parameters are modified
// prior to this function and it just handles implementation of a
// sparse * dense matmul without knowing details of the pass except
// what is given as parameters.
static Tensor fullyConnectedImpl(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Options &options, const std::vector<unsigned> &hierarchy,
    unsigned level, Tensor metaInfoBuckets, Tensor nzValueBuckets, Tensor acts,
    const std::string &debugPrefix) {
  // Only supporting single-IPU currently.
  assert(hierarchy.size() == 1);

  const std::string levelPrefix = debugPrefix + "/l" + std::to_string(level);
  const auto &inputType = acts.elementType();

  // At the top level for now, before doing anything else, enforce and
  // introduce grouping into the given tensors.
  if (level == 0) {
    acts = groupActs(acts, plan.grouping);
  }

  const Type resultType = inputType;

  // Gather/create all operands required by partition index
  std::vector<Tensor> nextLevelInputs;
  MetaInfoAndValues<std::vector<Tensor>> nextLevelDistributionBuckets;
  std::vector<Tensor> partials;
  MetaInfoAndValues<std::vector<Tensor>> nextLevelPropagationBuckets;
  std::vector<unsigned> subGroupIds;
  getNextLevelInputs(graph, shape, {}, plan, hierarchy, 0, acts,
                     nextLevelInputs);
  getNextLevelDistributionBuckets(graph, plan, options, metaInfoBuckets,
                                  nextLevelDistributionBuckets.metaInfo);
  getNextLevelDistributionBuckets(graph, plan, options, nzValueBuckets,
                                  nextLevelDistributionBuckets.values);
  createPartialsDense(graph, shape, {}, plan, options, hierarchy, 0, partials,
                      debugPrefix + "/partials");
  getSubGroupIds(graph, {}, plan, options, hierarchy, subGroupIds,
                 debugPrefix + "/subGroupIds");
  // For now the buckets to propagate are just those on each tile without
  // broadcast hence these are just home locations.
  getBucketsByPartition(plan, metaInfoBuckets,
                        nextLevelPropagationBuckets.metaInfo);
  getBucketsByPartition(plan, nzValueBuckets,
                        nextLevelPropagationBuckets.values);
  const Tensor bufferIdx =
      graph.addVariable(UNSIGNED_INT, {}, debugPrefix + "/bufferIdx");
  graph.setTileMapping(bufferIdx, 0);

  // Pre-arrange the inputs for the next level and hold onto them to
  // avoid exchanging multiple times in propagation phases (if they
  // are present).
  {
    std::vector<Tensor> perPartitionInputs;
    allocatePerPartitionInputs(graph, shape, {}, plan, inputType, hierarchy, 0,
                               perPartitionInputs,
                               debugPrefix + "/partitionedInputs");
    copyPartitions(graph, progBuilder.preDistribution, nextLevelInputs,
                   perPartitionInputs);
    std::swap(nextLevelInputs, perPartitionInputs);
  }

  writeUndefPartitions(progBuilder.preDistribution, partials);
  compute(graph, progBuilder.distributionCS, shape, {}, plan, options,
          hierarchy, 0, true, nextLevelInputs,
          nextLevelDistributionBuckets.values,
          nextLevelDistributionBuckets.metaInfo, partials, subGroupIds,
          debugPrefix);

  // We need a set of buffers on each tile for use during dynamic exchange
  // and compute steps (propagation phase).
  MetaInfoAndValues<std::array<std::vector<Tensor>, numBuffers>>
      propagationBuffers;
  for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
    const auto metaInfoBuffer = createBuckets(
        graph, UNSIGNED_SHORT, plan, plan.metaInfoElemsPerBucket, hierarchy,
        debugPrefix + "/metaInfoPropagationBuffer" + std::to_string(buffer));
    const auto nzValueBuffer = createBuckets(
        graph, inputType, plan, plan.nzElemsPerBucket, hierarchy,
        debugPrefix + "/nzValuesPropagationBuffer" + std::to_string(buffer));
    getBucketsByPartition(plan, metaInfoBuffer,
                          propagationBuffers.metaInfo[buffer]);
    getBucketsByPartition(plan, nzValueBuffer,
                          propagationBuffers.values[buffer]);
    // WriteUndef buffers as they are written to/read from during dynamic
    // control flow
    progBuilder.firstPropagationExchange.add(WriteUndef(metaInfoBuffer));
    progBuilder.firstPropagationExchange.add(WriteUndef(nzValueBuffer));
  }
  addPropagationExchanges(graph, progBuilder, shape, plan, options, hierarchy,
                          nextLevelPropagationBuckets, propagationBuffers,
                          bufferIdx, debugPrefix);
  std::array<ComputeSet, numBuffers> propagationCS;
  for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
    propagationCS.at(buffer) =
        graph.addComputeSet(debugPrefix + "/ComputePartialsPropagateBuffer" +
                            std::to_string(buffer));
    compute(graph, propagationCS.at(buffer), shape, {}, plan, options,
            hierarchy, 0, false, nextLevelInputs,
            propagationBuffers.values.at(buffer),
            propagationBuffers.metaInfo.at(buffer), partials, subGroupIds,
            debugPrefix);
  }
  progBuilder.propagationCompute =
      Switch(bufferIdx, {{0, Execute(propagationCS.at(false))}},
             Execute(propagationCS.at(true)));
  const auto output =
      finalReduction(graph, progBuilder, shape, resultType, {}, plan, options,
                     hierarchy, 0, partials, false, debugPrefix);

  return unfactorDims(output, 3);
}

static Tensor fullyConnectedSparseGradWImpl(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Options &options, const std::vector<unsigned> &hierarchy,
    unsigned level, Tensor metaInfoBuckets, Tensor weights, Tensor acts,
    const std::string &debugPrefix) {
  // Only supporting single-IPU currently.
  assert(hierarchy.size() == 1);

  const std::string levelPrefix = debugPrefix + "/l" + std::to_string(level);
  const auto &inputType = acts.elementType();

  // At the top level for now, before doing anything else, enforce and
  // introduce grouping into the given tensors.
  if (level == 0) {
    assert(plan.grouping.groups * plan.grouping.x * plan.grouping.z == 1);
    // Accumulated dimension is always inner-most
    std::vector<std::size_t> weightGrouping = {
        plan.grouping.groups, plan.grouping.x, plan.grouping.y};
    assert(weights.rank() == weightGrouping.size());
    bool weightsCanBeGrouped = true;
    for (std::size_t d = 0; d < weights.rank(); ++d) {
      weightsCanBeGrouped &= (weights.dim(d) % weightGrouping[d] == 0);
    }
    if (!weightsCanBeGrouped) {
      throw poputil::poplibs_error(
          "Padding of weights to meet grouping not yet handled");
    }
    weights = factorDims(weights, weightGrouping);
    acts = groupActs(acts, plan.grouping);
  }

  const Type resultType = inputType;

  std::vector<Tensor> nextLevelInputs;
  std::vector<Tensor> nextLevelWeights;
  std::vector<Tensor> nextLevelMetaInfoBuckets;
  getNextLevelInputs(graph, shape, {}, plan, hierarchy, 0, acts,
                     nextLevelInputs);
  getNextLevelWeights(graph, shape, {}, plan, hierarchy, 0, weights,
                      nextLevelWeights);
  getBucketsByPartition(plan, metaInfoBuckets, nextLevelMetaInfoBuckets);
  std::vector<Tensor> partials;
  createPartialsSparse(graph, shape, {}, plan, options, hierarchy, 0, partials,
                       debugPrefix + "/partials");
  std::vector<Tensor> subGroupIds;
  getSubGroupIds(graph, {}, plan, options, hierarchy, subGroupIds,
                 debugPrefix + "/subGroupIds");

  writeUndefPartitions(progBuilder.preDistribution, partials);
  compute(graph, progBuilder.distributionCS, shape, {}, plan, options,
          hierarchy, 0, true, nextLevelInputs, nextLevelWeights,
          nextLevelMetaInfoBuckets, partials, subGroupIds, debugPrefix);

  std::array<std::vector<Tensor>, numBuffers> inputPropagationBuffers,
      weightPropagationBuffers, subGroupIdPropagationBuffers;
  for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
    createPropagationBuffers(
        graph, inputType, shape, {}, plan, options, hierarchy, 0,
        nextLevelInputs, {}, inputPropagationBuffers.at(buffer),
        debugPrefix + "/inputPropagationBuffer" + std::to_string(buffer));
    createPropagationBuffers(
        graph, inputType, shape, {}, plan, options, hierarchy, 0,
        nextLevelWeights, {}, weightPropagationBuffers.at(buffer),
        debugPrefix + "/weightPropagationBuffer" + std::to_string(buffer));
    createPropagationBuffers(
        graph, inputType, shape, {}, plan, options, hierarchy, 0, subGroupIds,
        {}, subGroupIdPropagationBuffers.at(buffer),
        debugPrefix + "/subGroupIdPropagationBuffer" + std::to_string(buffer));
    writeUndefPartitions(progBuilder.firstPropagationExchange,
                         inputPropagationBuffers.at(buffer));
    writeUndefPartitions(progBuilder.firstPropagationExchange,
                         weightPropagationBuffers.at(buffer));
    writeUndefPartitions(progBuilder.firstPropagationExchange,
                         subGroupIdPropagationBuffers.at(buffer));
  }
  const auto bufferIdx =
      graph.addVariable(UNSIGNED_INT, {}, debugPrefix + "/bufferIdx");
  graph.setTileMapping(bufferIdx, 0);
  addPropagationExchangesGradW(
      graph, progBuilder, shape, plan, options, hierarchy, nextLevelInputs,
      nextLevelWeights, subGroupIds, inputPropagationBuffers,
      weightPropagationBuffers, subGroupIdPropagationBuffers, bufferIdx,
      debugPrefix);

  auto computeSwitch = Switch(bufferIdx);
  for (std::size_t inputBuffer = 0; inputBuffer < numBuffers; ++inputBuffer) {
    for (std::size_t weightBuffer = 0; weightBuffer < numBuffers;
         ++weightBuffer) {
      for (std::size_t subGroupIdBuffer = 0; subGroupIdBuffer < numBuffers;
           ++subGroupIdBuffer) {
        const auto cs = graph.addComputeSet(
            debugPrefix + "/ComputePartialsPropagateBufferIn" +
            std::to_string(inputBuffer) + "Weights" +
            std::to_string(weightBuffer) + "SubGroupId" +
            std::to_string(subGroupIdBuffer));
        compute(graph, cs, shape, {}, plan, options, hierarchy, 0, false,
                inputPropagationBuffers.at(inputBuffer),
                weightPropagationBuffers.at(weightBuffer),
                nextLevelMetaInfoBuckets, partials,
                subGroupIdPropagationBuffers.at(subGroupIdBuffer), debugPrefix);
        computeSwitch.add(inputBuffer * numBuffers * numBuffers +
                              weightBuffer * numBuffers + subGroupIdBuffer,
                          Execute(cs));
      }
    }
  }
  progBuilder.propagationCompute = computeSwitch;
  const auto output =
      finalReduction(graph, progBuilder, shape, resultType, {}, plan, options,
                     hierarchy, 0, partials, true, debugPrefix);
  // Output includes partitions and buckets per partition in its shape so
  // flatten these away for return.
  // TODO: We could just keep buckets by partition in the shape for
  // implementation functions and this would be more consistent.
  return output.flatten(0, 5);
}

static void iterateInputTensorUsage(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const std::vector<unsigned> &hierarchy, const unsigned level,
    const Tensor &acts, TensorUseTracker &usage) {
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  usage.add(graph, tile, acts);
}

template <typename NextLevelActs>
static void iterateInputTensorUsage(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const std::vector<unsigned> &hierarchy, const unsigned level,
    const NextLevelActs &acts, TensorUseTracker &usage) {
  iteratePartitions(
      shape, plan.partition, plan.grouping,
      [&](const auto &i, const auto &begin, const auto &end) {
        const auto sShape = end - begin;
        auto sIndices = indices;
        sIndices.emplace_back(i);
        const auto partitionIndexFlat =
            flattenIndex(plan.partition.asStdVector(), i.asStdVector());
        iterateInputTensorUsage(graph, sShape, sIndices, plan, hierarchy,
                                level + 1, acts[partitionIndexFlat], usage);
      });
}

static void mapInput(Graph &graph, const std::vector<unsigned> &hierarchy,
                     const Vector<unsigned> &shape, const SinglePassPlan &plan,
                     Tensor input) {
  TensorUseTracker usage(graph.getTarget().getNumTiles());

  input = groupActs(input, plan.grouping);

  std::vector<Tensor> nextLevelInputs;
  getNextLevelInputs(graph, shape, {}, plan, hierarchy, 0, input,
                     nextLevelInputs);
  iterateInputTensorUsage(graph, shape, {}, plan, hierarchy, 0, nextLevelInputs,
                          usage);

  const std::vector<std::size_t> actGrouping = {
      plan.grouping.groups, plan.grouping.y, plan.grouping.z};
  const unsigned grainSize = product(actGrouping);
  // Reduce exchange code especially for smaller fc layers by mapping inputs
  // to fewer tiles when there aren't enough elements to go around.
  const unsigned minBytesPerTile = 128;
  const unsigned minElemsPerTile = ceildiv(
      minBytesPerTile, graph.getTarget().getTypeSize(input.elementType()));
  usage.mapTensorsByUse(graph, grainSize, minElemsPerTile, true);
}

namespace dynamic {

// There may be limitations to this interface because the conv planner doesn't
// really know what the inner loop does. The planner assumes that dynamic
// slice is used but has no information on what layout the operation which
// runs on each split is neither its layout requirements.
std::tuple<unsigned, unsigned, unsigned>
fullyConnectedDenseGradWSerialSplits(const Graph &graph, const Type &inputType,
                                     const FullyConnectedParams &fcParams,
                                     const poplar::OptionFlags &options_,
                                     PlanningCache *cache) {
  auto options = fullyconnected::parseOptionFlags(options_);
  OptionFlags matMulOptions;
  matMulOptions.set("fullyConnectedPass", "NONE");
  matMulOptions.set("availableMemoryProportion",
                    std::to_string(options.availableMemoryProportion));
  return poplin::groupedMatMulOutputSerialSplits(
      graph, inputType, options.partialsType,
      {fcParams.getNumGroups(), fcParams.getInputChannelsPerGroup(),
       fcParams.getBatchSize()},
      {fcParams.getNumGroups(), fcParams.getBatchSize(),
       fcParams.getOutputChannelsPerGroup()},
      matMulOptions);
}

SparseTensor createFullyConnectedWeights(Graph &graph, const Type &inputType,
                                         const FullyConnectedParams &params,
                                         const std::string &debugName,
                                         const OptionFlags &optionFlags,
                                         PlanningCache *cache) {
  const auto &options = parseOptionFlags(optionFlags);
  logging::debug(
      "popsparse::createFullyConnectedWeights: '{}' params={}, options={}",
      debugName, params, options);

  Plan plan;
  Cost cost;
  std::tie(plan, cost) =
      getPlan(graph.getTarget(), inputType, params, optionFlags, cache);
  const auto &target = graph.getTarget();
  const auto hierarchy = poplibs::getTileHierarchy(target);

  const auto fwdPlan = getFwdPlan(plan);

  // Create variables. This handles forward pass only but in future there should
  // be multiple variables for any pass-specific meta-info.
  const auto fwdMetaInfoBuckets = createBuckets(
      graph, UNSIGNED_SHORT, fwdPlan, plan.fwdMetaInfoElemsPerBucket, hierarchy,
      debugName + "/metaInfoForward");
  const auto gradAMetaInfoBuckets =
      plan.sharedBuckets()
          ? Tensor()
          : createBuckets(graph, UNSIGNED_SHORT, fwdPlan,
                          plan.gradAMetaInfoElemsPerBucket, hierarchy,
                          debugName + "/metaInfoGradA");
  const auto nzValueBuckets =
      createBuckets(graph, inputType, fwdPlan, plan.nzElemsPerBucket, hierarchy,
                    debugName + "/nzValues");
  const auto weightBuckets =
      SparseTensor(plan.sharedBuckets()
                       ? fwdMetaInfoBuckets
                       : concat(fwdMetaInfoBuckets, gradAMetaInfoBuckets, 1),
                   nzValueBuckets);
  const auto overflowInfoElems = getNumOverflowInfoElems(
      target.getTypeSize(UNSIGNED_SHORT), plan.partition.x, plan.partition.y,
      plan.partition.z);
  const auto overflowInfo = graph.addVariable(
      UNSIGNED_SHORT, {overflowInfoElems}, debugName + "/overflowInfo");
  graph.setTileMapping(overflowInfo, 0);

  const auto packed =
      packWeights(weightBuckets, getTotalMetaInfoElemsPerBuckets(plan),
                  plan.nzElemsPerBucket, overflowInfo);

  // Attach meta-data to the sparse tensor.
  std::unique_ptr<TensorMetaDataBase> opMetaData =
      std::make_unique<FullyConnectedTensorMetaData>(params, options);
  return SparseTensor(packed.getMetaInfoTensor(), packed.getNzValuesTensor(),
                      std::move(opMetaData));
}

Tensor createFullyConnectedInput(Graph &graph, const Type &inputType,
                                 const FullyConnectedParams &params,
                                 const std::string &debugName,
                                 const OptionFlags &optionFlags,
                                 PlanningCache *cache) {
  const auto &options = parseOptionFlags(optionFlags);
  logging::debug(
      "popsparse::createFullyConnectedInput: '{}' params={}, options={}",
      debugName, params, options);

  Plan plan;
  Cost cost;
  std::tie(plan, cost) =
      getPlan(graph.getTarget(), inputType, params, optionFlags, cache);
  const auto &target = graph.getTarget();
  const auto hierarchy = poplibs::getTileHierarchy(target);

  const auto fwdPlan = getFwdPlan(plan);

  const auto inputMemOrdering = getOnTileActsOrdering(fwdPlan.method);
  const std::vector<std::size_t> inputShapeInternal = {
      params.getNumGroups(), params.getInputChannelsPerGroup(),
      params.getBatchSize()};
  std::vector<std::size_t> inputShapeAllocation(inputMemOrdering.size());
  for (std::size_t i = 0; i < inputMemOrdering.size(); ++i) {
    inputShapeAllocation[i] = inputShapeInternal[inputMemOrdering[i]];
  }

  const auto input =
      graph.addVariable(inputType, inputShapeAllocation, debugName)
          .dimShuffle(inversePermutation(inputMemOrdering));

  const Vector<unsigned> shape = {
      static_cast<unsigned>(params.getNumGroups()),
      static_cast<unsigned>(params.getOutputChannelsPerGroup()),
      static_cast<unsigned>(params.getInputChannelsPerGroup()),
      static_cast<unsigned>(params.getBatchSize())};
  mapInput(graph, hierarchy, shape, fwdPlan, input);

  return inputInternalToExternalShape(input, params.getNumGroups());
}

static void validateSparseOperandMetaData(const SparseTensor &weights,
                                          const FullyConnectedParams &params,
                                          const Options &options) {
  const auto opMetaData = weights.getOpMetaData();

  // For FullyConnected interface, this validation is optional dependent on
  // whether or not the meta-data was given.
  // TODO: Make this mandatory
  if (opMetaData.getData() == nullptr) {
    return;
  }

  const auto *fcMetaData =
      dynamic_cast<const FullyConnectedTensorMetaData *>(opMetaData.getData());
  if (!fcMetaData) {
    throw poplibs_error(
        "Op meta-data is present on sparse tensor but tensor was "
        " not created through createFullyConnectedWeights");
  }

  if (fcMetaData->planningKey != PlanningCacheImpl::Key(params, options)) {
    throw poplibs_error(
        "Given sparse tensor was not created for this operation");
  }
}

Tensor fullyConnectedFwd(Graph &graph, const SparseTensor &weights,
                         const Tensor &activations,
                         const FullyConnectedParams &params, Sequence &prog,
                         const std::string &debugPrefix,
                         const OptionFlags &optionFlags, PlanningCache *cache) {
  // TODO: Parameter validation - shapes/sizes match given params etc.
  const auto &target = graph.getTarget();
  const auto &inputType = activations.elementType();
  const auto &options = parseOptionFlags(optionFlags);
  logging::debug("popsparse::fullyConnectedFwd: '{}' params={}, options={}",
                 debugPrefix, params, options);
  validateSparseOperandMetaData(weights, params, options);
  Plan plan;
  Cost cost;
  std::tie(plan, cost) = getPlan(target, inputType, params, optionFlags, cache);

  const auto hierarchy = poplibs::getTileHierarchy(target);

  const Vector<unsigned> shape = {
      static_cast<unsigned>(params.getNumGroups()),
      static_cast<unsigned>(params.getOutputChannelsPerGroup()),
      static_cast<unsigned>(params.getInputChannelsPerGroup()),
      static_cast<unsigned>(params.getBatchSize())};
  const auto input = inputExternalToInternalShape(activations, shape.groups);

  const auto overflowInfoElems = getNumOverflowInfoElems(
      target.getTypeSize(UNSIGNED_SHORT), plan.partition.x, plan.partition.y,
      plan.partition.z);
  SparseTensor weightBuckets;
  Tensor overflowInfo;
  std::tie(weightBuckets, overflowInfo) = unpackWeights(
      weights, overflowInfoElems, getTotalMetaInfoElemsPerBuckets(plan),
      plan.nzElemsPerBucket);

  // Get meta-info required for forward pass.
  weightBuckets = weightsInternalSliceBuckets(weightBuckets, 0u,
                                              plan.fwdMetaInfoElemsPerBucket);

  const auto fwdPlan = getFwdPlan(plan);
  ProgBuilder progBuilder(graph, hierarchy, debugPrefix);
  std::vector<Vector<unsigned>> indices;
  const auto &outputActivations = fullyConnectedImpl(
      graph, progBuilder, shape, indices, fwdPlan, options, hierarchy,
      0u /* level */, weightBuckets.getMetaInfoTensor(),
      weightBuckets.getNzValuesTensor(), input, debugPrefix);
  progBuilder.addToSequence(graph, prog, fwdPlan, overflowInfo, debugPrefix);

  return inputInternalToExternalShape(outputActivations, shape.groups);
}

Tensor fullyConnectedGradA(Graph &graph, const SparseTensor &weights,
                           const Tensor &activations,
                           const FullyConnectedParams &params, Sequence &prog,
                           const std::string &debugPrefix,
                           const OptionFlags &optionFlags,
                           PlanningCache *cache) {
  const auto &target = graph.getTarget();
  const auto &inputType = activations.elementType();
  const auto &options = parseOptionFlags(optionFlags);
  logging::debug("popsparse::fullyConnectedGradA: '{}' params={}, options={}",
                 debugPrefix, params, options);
  validateSparseOperandMetaData(weights, params, options);
  Plan plan;
  Cost cost;
  std::tie(plan, cost) = getPlan(target, inputType, params, optionFlags, cache);

  const auto hierarchy = poplibs::getTileHierarchy(target);

  const Vector<unsigned> shape = {
      static_cast<unsigned>(params.getNumGroups()),
      static_cast<unsigned>(params.getInputChannelsPerGroup()),
      static_cast<unsigned>(params.getOutputChannelsPerGroup()),
      static_cast<unsigned>(params.getBatchSize())};

  const auto input = inputExternalToInternalShape(activations, shape.groups);

  const auto overflowInfoElems = getNumOverflowInfoElems(
      target.getTypeSize(UNSIGNED_SHORT), plan.partition.x, plan.partition.y,
      plan.partition.z);
  SparseTensor weightBuckets;
  Tensor overflowInfo;
  std::tie(weightBuckets, overflowInfo) = unpackWeights(
      weights, overflowInfoElems, getTotalMetaInfoElemsPerBuckets(plan),
      plan.nzElemsPerBucket);

  // Get meta-info required for backward pass.
  weightBuckets = weightsInternalSliceBuckets(
      weightBuckets, plan.sharedBuckets() ? 0 : plan.fwdMetaInfoElemsPerBucket,
      plan.gradAMetaInfoElemsPerBucket);

  const auto gradAPlan = getGradAPlan(plan);

  // We need to shuffle buckets around to the order expected by the grad-a pass.
  weightBuckets =
      SparseTensor(weightBucketsByPartition(weightBuckets.getMetaInfoTensor(),
                                            plan.partition),
                   weightBucketsByPartition(weightBuckets.getNzValuesTensor(),
                                            plan.partition));
  const std::vector<unsigned> shuffleSrc = {0, 1, 2, 3};
  const auto shuffleDest = vectorConvert<unsigned>(gradAPlan.dimShuffleToFwd);
  weightBuckets = SparseTensor(weightBuckets.getMetaInfoTensor()
                                   .dimShufflePartial(shuffleSrc, shuffleDest)
                                   .flatten(0, 5),
                               weightBuckets.getNzValuesTensor()
                                   .dimShufflePartial(shuffleSrc, shuffleDest)
                                   .flatten(0, 5));

  ProgBuilder progBuilder(graph, hierarchy, debugPrefix);
  std::vector<Vector<unsigned>> indices;
  const auto &inputGradients = fullyConnectedImpl(
      graph, progBuilder, shape, indices, gradAPlan, options, hierarchy,
      0u /* level */, weightBuckets.getMetaInfoTensor(),
      weightBuckets.getNzValuesTensor(), input, debugPrefix);
  progBuilder.addToSequence(graph, prog, gradAPlan, overflowInfo, debugPrefix);

  return inputInternalToExternalShape(inputGradients, shape.groups);
}

Tensor fullyConnectedSparseGradW(Graph &graph, const Tensor sparsityMetaInfo,
                                 const Tensor &gradA, const Tensor &activations,
                                 const FullyConnectedParams &params,
                                 Sequence &prog, const std::string &debugPrefix,
                                 const OptionFlags &optionFlags,
                                 PlanningCache *cache) {
  // TODO: Should this take meta-data for sparse tensor for validation?
  // Validation is the only purpose this serves right now.

  const auto &target = graph.getTarget();
  const auto &inputType = activations.elementType();
  const auto &options = parseOptionFlags(optionFlags);
  logging::debug(
      "popsparse::fullyConnectedSparseGradW: '{}' params={}, options={}",
      debugPrefix, params, options);
  Plan plan;
  Cost cost;
  std::tie(plan, cost) = getPlan(target, inputType, params, optionFlags, cache);

  const auto hierarchy = poplibs::getTileHierarchy(target);

  const Vector<unsigned> shape = {
      static_cast<unsigned>(params.getNumGroups()),
      static_cast<unsigned>(params.getOutputChannelsPerGroup()),
      static_cast<unsigned>(params.getBatchSize()),
      static_cast<unsigned>(params.getInputChannelsPerGroup()),
  };

  // Both input and weights in form {G, C, N} where N is forward pass
  // batch size and therefore the dimension we will accumulate over in
  // the GradW pass. C here is input channels for input and output
  // channels for output
  const auto input =
      inputExternalToInternalShape(activations, shape.groups).dimRoll(1, 2);
  const auto outputGrad = inputExternalToInternalShape(gradA, shape.groups);

  const auto overflowInfoElems = getNumOverflowInfoElems(
      target.getTypeSize(UNSIGNED_SHORT), plan.partition.x, plan.partition.y,
      plan.partition.z);
  Tensor metaInfoBuckets;
  Tensor overflowInfo;
  std::tie(metaInfoBuckets, overflowInfo) =
      unpackWeights(sparsityMetaInfo, overflowInfoElems,
                    getTotalMetaInfoElemsPerBuckets(plan));

  // Get meta-info required for grad-w pass.
  metaInfoBuckets = weightsInternalSliceBuckets(metaInfoBuckets, 0,
                                                plan.fwdMetaInfoElemsPerBucket);

  const auto gradWPlan = getGradWPlan(plan);

  // We need to shuffle buckets around to the order expected by the grad-w pass.
  metaInfoBuckets = weightBucketsByPartition(metaInfoBuckets, plan.partition);
  metaInfoBuckets =
      metaInfoBuckets
          .dimShufflePartial({0, 1, 2, 3},
                             vectorConvert<unsigned>(gradWPlan.dimShuffleToFwd))
          .flatten(0, 5);

  ProgBuilder progBuilder(graph, hierarchy, debugPrefix);
  std::vector<Vector<unsigned>> indices;
  auto weightGradientBuckets = fullyConnectedSparseGradWImpl(
      graph, progBuilder, shape, indices, gradWPlan, options, hierarchy,
      0u /* level */, metaInfoBuckets, outputGrad, input, debugPrefix);
  progBuilder.addToSequence(graph, prog, gradWPlan, overflowInfo, debugPrefix);

  // Rearrange resulting weight gradient buckets into order expected for
  // the forward pass.
  weightGradientBuckets =
      weightBucketsByPartition(weightGradientBuckets, gradWPlan.partition);
  weightGradientBuckets =
      weightGradientBuckets
          .dimShufflePartial(vectorConvert<unsigned>(gradWPlan.dimShuffleToFwd),
                             {0, 1, 2, 3})
          .flatten(0, 5);

  return weightsInternalToExternalShape(weightGradientBuckets,
                                        plan.nzElemsPerBucket);
}

} // end namespace dynamic
} // end namespace popsparse
