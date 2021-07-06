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
#include <popops/Rearrange.hpp>
#include <popops/Reduce.hpp>

#include "poputil/TensorMetaData.hpp"
#include <poplin/MatMul.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/VarStructure.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include "poplibs_support/Algorithm.hpp"
#include "poplibs_support/TileHierarchy.hpp"
#include "poplibs_support/Tracepoint.hpp"
#include "poplibs_support/VectorUtils.hpp"
#include "poplibs_support/logging.hpp"

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
  // What portion of the problem is covered by the initial distribution phase.
  fullyconnected::Vector<unsigned> initialDistributionPartitions;
  // What portion of the problem is covered by each step in the
  // propagation phase.
  fullyconnected::Vector<unsigned> propagationPartitions;
  unsigned nzElemsPerBucket;
  unsigned metaInfoElemsPerBucket;
  fullyconnected::PartitionToPNMapping mapping;
  // Only really useful for GradW pass. If set for GradW pass we exchange
  // buckets (with partials) instead of inputs.
  bool exchangeBuckets;
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

// Check if the given volume in the given space would be contiguous
// when the space is flattened to 1 dimension.
//
// e.g. given a multi-dimensional space with bound {8, 5, 6}:
// When the space is flattened to 1 dimension (with size {240}, do the
// following volumes map to a contiguous region of 1-dimensional space?
//
// {4, 5, 6} - yes, this is maps contiguously to 1-dimensional space with
//             size {120}
// {4, 1, 6} - no, this maps into 4 noncontiguous volumes in 1-dimensional
//             space with size {6} each.
// {8, 5, 1} - no, this maps into 40 noncontiguous volumes in 1-dimensional
//             space with size {1} each.
//
[[maybe_unused]] static bool
volumeIsContiguousInFlattenedSpace(const std::vector<unsigned> &space,
                                   const std::vector<unsigned> &shape) {
  assert(space.size() == shape.size());
  int dim = shape.size() - 1;
  while (dim >= 0 && shape[dim] == space[dim])
    dim--;
  const auto notEqualOne = [](const auto n) { return n != 1; };
  if (dim > 0 &&
      std::any_of(shape.begin(), shape.begin() + dim - 1, notEqualOne)) {
    return false;
  }
  return true;
}

static unsigned getPropagationStartingOffsetFlat(const SinglePassPlan &plan) {
  const auto partition =
      shuffleVector(plan.partition.asStdVector(), plan.dimShuffleToFwd);
  auto initialDistributionPartitions = shuffleVector(
      plan.initialDistributionPartitions.asStdVector(), plan.dimShuffleToFwd);
  assert(volumeIsContiguousInFlattenedSpace(partition,
                                            initialDistributionPartitions));
  return product(initialDistributionPartitions);
}

// Starting offset for propagation phase, result given in terms of
// forward pass, and is the same for all passes hence 'canonical'
static std::vector<unsigned>
getPropagationStartingOffsetCanonical(const SinglePassPlan &plan) {
  const auto partition =
      shuffleVector(plan.partition.asStdVector(), plan.dimShuffleToFwd);
  const auto indexFlat = getPropagationStartingOffsetFlat(plan);
  auto indices = unflattenIndexBoundless(partition, indexFlat);
  // We ignore groups so just get rid of them for now.
  indices.at(1) +=
      indices.front() * std::accumulate(std::next(partition.begin()),
                                        partition.end(), unsigned(1),
                                        std::multiplies<unsigned>());
  indices.front() = 0;
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
  std::vector<std::size_t> dimsToIterate;

  // Multi-dimensional bounds of the loops we iterate.
  for (std::size_t dimIdx = 0; dimIdx < plan.partition.size(); ++dimIdx) {
    const auto dim = plan.dimShuffleToFwd.at(dimIdx);
    const auto partition = plan.partition.asStdVector().at(dim);
    const auto increment = plan.propagationPartitions.asStdVector().at(dim);
    if (partition > increment) {
      // We assume 'groups' does not have a partition for now.
      assert(dim >= 1);
      assert(dimIdx >= 1);
      dimsToIterate.emplace_back(dim - 1);
    }
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

static std::vector<unsigned> getLoopIncrements(const SinglePassPlan &plan) {
  // Increments are just the partitions covered per step of the propagation
  // phase, only we need to cut groups out of the vector.
  const auto &propagationPartitions = plan.propagationPartitions.asStdVector();
  const std::vector<unsigned> increments(propagationPartitions.begin() + 1,
                                         propagationPartitions.end());
  return increments;
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

// Put single-tile vertices and indices for managing dynamic control
// flow on a tile with less stuff already assigned to it.
static unsigned
getTileForDynamicControlFlowManagement(const Graph &graph,
                                       const SinglePassPlan &plan) {
  const auto numUsedTiles = product(plan.partition.asStdVector());
  const auto numTiles = graph.getTarget().getNumTiles();
  return std::min(numTiles - 1, numUsedTiles);
}

namespace {

// Handles building of the program to execute the fully connected layer
// implementation.
class ProgBuilder {
public:
  std::vector<Sequence> pre;
  std::vector<Sequence> distributionExchange;
  Sequence preDistribution;
  Sequence distributionCompute;
  Sequence prePropagation;
  // [direction]
  std::array<Program, numDirections> propagationExchanges;
  Sequence propagationCompute;
  bool propagationMustReachStartingOffset = false;
  Sequence postPropagation;
  std::vector<std::vector<ComputeSet>> reductionCSs;
  std::vector<Sequence> post;

  ProgBuilder(Graph &graph, const std::vector<unsigned> &hierarchy,
              const poplar::DebugNameAndId &dnai)
      : pre(hierarchy.size() + 1), distributionExchange(hierarchy.size() + 1),
        reductionCSs(hierarchy.size()), post(hierarchy.size() + 1) {}

  void setPropagationMustReachStartingOffset() {
    propagationMustReachStartingOffset = true;
  }

  void addToSequence(Graph &graph, Sequence &seq, const SinglePassPlan &plan,
                     const Tensor &overflowInfo,
                     const poplar::DebugNameAndId &dnai) {
    for (unsigned level = 0; level < pre.size(); ++level) {
      seq.add(pre[level]);
      seq.add(distributionExchange[level]);
    }
    seq.add(preDistribution);
    seq.add(distributionCompute);

    std::vector<std::size_t> dimsToIterate;
    std::vector<unsigned> startingIndices;
    std::tie(dimsToIterate, startingIndices) =
        getPropagationIteratedDimsAndStartingIndices(plan);
    auto staticLoopBounds = plan.partition.asStdVector();
    staticLoopBounds.erase(staticLoopBounds.begin());
    const bool needPropagationPhase =
        flattenIndexBoundless(staticLoopBounds, startingIndices) <
        product(staticLoopBounds);
    if (needPropagationPhase) {
      std::vector<Sequence> progs;
      progs.emplace_back();

      // Shuffle controlInfo to match this pass.
      auto endIndices = getPassEndIndices(overflowInfo, plan);
      endIndices = popops::cast(graph, endIndices, UNSIGNED_INT, progs.back(),
                                {dnai, "endIndices"});
      const auto outerLoopDim = getOuterLoopDim(plan);
      const auto outerLoopIterationsToSkip =
          getOuterLoopIterationsToSkip(overflowInfo);
      const auto increments = getLoopIncrements(plan);

      using namespace popops;

      const auto controlFlowTile =
          getTileForDynamicControlFlowManagement(graph, plan);
      const auto zero = graph.addConstant(UNSIGNED_INT, {}, 0, {dnai});
      const auto one = graph.addConstant(UNSIGNED_INT, {}, 1, {dnai});
      graph.setTileMapping(zero, controlFlowTile);
      graph.setTileMapping(one, controlFlowTile);

      // The following code constructs a control program that looks roughly
      // like the following C++-like pseudocode:
      //
      // totalRemaining = product(endIndices) - startingIndexFlat;
      // if (totalRemaining) {
      //   prePropagation();
      //
      //   while (indices[x] < endIndices[x] && totalRemaining > 0) {
      //     if (doXIteration[indices[x]]) {
      //       while (indices[y] < endIndices[y] && totalRemaining > 0) {
      //         while (indices[z] < endIndices[z] && totalRemaining > 0) {
      //           propagationCompute();
      //           totalRemaining -= increments[z];
      //           if (totalRemaining > 0) {
      //             indices[z] += increments[z];
      //             propagationExchanges[z]();
      //           }
      //         }
      //         if (totalRemaining > 0) {
      //           indices[y] += increments[y];
      //           propagationExchanges[y]();
      //           indices[z] = 0;
      //         }
      //       }
      //     } else {
      //       totalRemaining -= increments[x] *
      //         (endIndices[y] - indices[y]) *
      //         (endIndices[z] - indices[z]);
      //     }
      //     if (totalRemaining > 0) {
      //       indices[x] += increments[x];
      //       propagationExchanges[x]();
      //       indices[y] = 0;
      //     }
      //   }
      //
      //   /* If propagationMustReachStartingOffset then the following program
      //      is generated also */
      //
      //   mask = indices > 0;
      //   remainingIterations = (partition * mask) - indices;
      //
      //   while (remainingIterations.z --> 0) {
      //     propagationExchanges[z]();
      //   }
      //   while (remainingIterations.y --> 0) {
      //     propagationExchanges[y]();
      //   }
      //   while (remainingIterations.x --> 0) {
      //     propagationExchanges[x]();
      //   }
      //
      //   /* End of propagationMustReachStartingOffset section */
      //
      //   postPropagation();
      // }
      //

      const auto totalRemaining = [&] {
        std::vector<Tensor> placeholders;
        expr::Any endIndicesProductExpr = expr::PlaceHolder(1);
        for (std::size_t i = 0; i < numDirections; ++i) {
          placeholders.emplace_back(endIndices[i]);
          if (i != 0) {
            endIndicesProductExpr =
                expr::Mul(expr::PlaceHolder(i + 1), endIndicesProductExpr);
          }
        }
        const auto startingIndexFlatExpr =
            expr::Const(getPropagationStartingOffsetFlat(plan));
        return popops::map(
            graph, expr::Sub(endIndicesProductExpr, startingIndexFlatExpr),
            placeholders, progs.back(), {dnai, "totalRemaining"});
      }();

      progs.emplace_back();
      progs.back().add(prePropagation);

      const auto indices = graph.clone(endIndices, {dnai, "indices"});
      const auto initialIndices =
          graph.addConstant(UNSIGNED_INT, endIndices.shape(),
                            ArrayRef<unsigned>(startingIndices), {dnai});
      logging::popsparse::debug("startingIndices={}", startingIndices);
      graph.setTileMapping(initialIndices, controlFlowTile);
      progs.back().add(Copy(initialIndices, indices, false, {dnai}));

      const std::array<std::string, numDirections> dimNames = {"X", "Y", "Z"};

      assert(dimsToIterate.size() >= 1);
      for (auto dimIt = dimsToIterate.rbegin();
           dimIt != std::prev(dimsToIterate.rend()); ++dimIt) {
        progs.emplace_back();

        progs.back().add(Copy(zero, indices[*dimIt], false, {dnai}));
        popops::addInPlace(graph, indices[*std::next(dimIt)], one, progs.back(),
                           {dnai, "adjust" + dimNames[*dimIt] +
                                      "StartingIndicesToLoopBounds"});

        auto prog = std::move(progs.back());
        progs.pop_back();
        Sequence condBody;
        const auto doesNotFitInLoopBound = popops::gteq(
            graph, indices[*dimIt], endIndices[*dimIt], progs.back(),
            {dnai, "is" + dimNames[*dimIt] + "StartingIndexOutsideLoopBounds"});
        progs.back().add(If(doesNotFitInLoopBound, std::move(prog),
                            Sequence{{}, {dnai}}, {dnai}));
      }
      for (std::size_t i = 0; i < dimsToIterate.size(); ++i) {
        progs.emplace_back();
      }
      progs.back().add(propagationCompute);
      popops::subInPlace(graph, totalRemaining,
                         increments[dimsToIterate.back()], progs.back(),
                         {dnai, "decrementTotalRemaining"});
      for (auto dimIt = dimsToIterate.rbegin(); dimIt != dimsToIterate.rend();
           ++dimIt) {
        // If this is the outer-most loop, we have extra information that
        // allows us to potentially skip the body for this iteration.
        if (*dimIt == outerLoopDim) {
          auto prog = std::move(progs.back());
          progs.pop_back();
          progs.emplace_back();

          const auto doIteration = graph.addVariable(
              UNSIGNED_INT, {}, {dnai, "do" + dimNames[*dimIt] + "Iteration"});
          graph.setTileMapping(doIteration, controlFlowTile);
          const auto cs = graph.addComputeSet(
              {dnai, "shouldDo" + dimNames[*dimIt] + "Iteration"});
          const auto v =
              graph.addVertex(cs,
                              templateVertex("popsparse::BitIsSet",
                                             UNSIGNED_SHORT, UNSIGNED_INT),
                              {{"bits", outerLoopIterationsToSkip},
                               {"index", indices[*dimIt]},
                               {"out", doIteration}});
          graph.setTileMapping(v, controlFlowTile);
          progs.back().add(Execute(cs, {dnai}));
          Sequence falseBody;
          // We need to reduce the total remaining by the same number as would
          // have been processed if we performed all the inner loops.
          std::vector<Tensor> placeholders = {totalRemaining};
          expr::Any numInnerIterationsExpr = expr::Const(increments[*dimIt]);
          if (dimIt != dimsToIterate.rbegin()) {
            auto innerDimIt = std::prev(dimIt);
            do {
              numInnerIterationsExpr =
                  expr::Mul(numInnerIterationsExpr,
                            (expr::PlaceHolder(placeholders.size() + 1) -
                             expr::PlaceHolder(placeholders.size() + 2)));
              placeholders.emplace_back(endIndices[*innerDimIt]);
              placeholders.emplace_back(indices[*innerDimIt]);
            } while (innerDimIt-- != dimsToIterate.rbegin());
          }
          popops::mapInPlace(graph, expr::Sub(expr::_1, numInnerIterationsExpr),
                             placeholders, falseBody,
                             {dnai, "decrementTotalRemaining"});
          progs.back().add(
              Switch(doIteration, {{0, falseBody}}, std::move(prog), {dnai}));
        }

        // If there is any work left to do, do the exchange for this dimension,
        // increment indices and zero
        {
          progs.emplace_back();
          progs.back().add(propagationExchanges.at(*dimIt));

          const auto increment =
              graph.addConstant(UNSIGNED_INT, {1}, increments[*dimIt], {dnai});
          graph.setTileMapping(increment, controlFlowTile);
          popops::addInPlace(graph, indices[*dimIt], increment, progs.back(),
                             {dnai, "increment" + dimNames[*dimIt] + "Index"});

          // Zero next inner-most dimension's index (if there is one)
          if (dimIt != dimsToIterate.rbegin()) {
            progs.back().add(
                Copy(zero, indices[*std::prev(dimIt)], false, {dnai}));
          }

          auto prog = std::move(progs.back());
          progs.pop_back();
          progs.back().add(If(totalRemaining, std::move(prog),
                              Sequence{{}, {dnai}}, {dnai}));
        }

        auto prog = std::move(progs.back());
        progs.pop_back();

        Sequence condBody;
        const auto dimIsNotFinished = popops::map(
            graph, (expr::_1 != expr::Const(0)) && (expr::_2 < expr::_3),
            {totalRemaining, indices[*dimIt], endIndices[*dimIt]}, condBody,
            {dnai, "isDim" + dimNames[*dimIt] + "Finished"});
        progs.back().add(RepeatWhileTrue(std::move(condBody), dimIsNotFinished,
                                         std::move(prog), {dnai}));
      }

      // If we need to get back to the starting offset we can check
      // for each dimension if we have completed a number of iterations equal
      // to the total number of partitions in each dimension and if not
      // just exchange until we reach a zero offset again.
      if (propagationMustReachStartingOffset) {
        const auto dimPartitions = [&] {
          auto v = plan.partition.asStdVector<unsigned>();
          // No 'groups' dim
          v.erase(v.begin());
          return v;
        }();
        const auto dimPartitionsT =
            graph.addConstant(UNSIGNED_INT, {dimPartitions.size()},
                              ArrayRef(dimPartitions), {dnai, "dimPartitions"});
        graph.setTileMapping(dimPartitionsT, controlFlowTile);
        const auto remainingExchanges = popops::map(
            graph,
            (expr::_2 * expr::Cast(expr::_1 > expr::Const(0), UNSIGNED_INT)) -
                expr::_1,
            {indices, dimPartitionsT}, progs.back(),
            {dnai, "calcRemainingExchanges"});

        // NOTE: We could speed this up by only exchanging the written buffers
        // that we need to copy in the postPropagation program but this would
        // be more code and effort and for now doesn't really make much
        // difference.
        for (const auto dim : dimsToIterate) {
          Sequence condBody;
          const auto increment =
              graph.addConstant(UNSIGNED_INT, {1}, increments[dim], {dnai});
          graph.setTileMapping(increment, controlFlowTile);
          const auto haveRemainingExchanges =
              popops::gt(graph, remainingExchanges[dim], zero, condBody,
                         {dnai, "haveRemainingExchanges" + dimNames[dim]});
          Sequence updateRemainingExchanges;
          popops::subInPlace(
              graph, remainingExchanges[dim], increment,
              updateRemainingExchanges,
              {dnai, "updateRemainingExchanges" + dimNames[dim]});
          progs.back().add(
              RepeatWhileTrue(std::move(condBody), haveRemainingExchanges,
                              Sequence{{propagationExchanges.at(dim),
                                        std::move(updateRemainingExchanges)},
                                       {dnai}},
                              {dnai}));
        }
      }

      progs.back().add(postPropagation);

      // Wrap the whole program in a conditional to check if the dynamic steps
      // are necessary at all based on totalRemaining.
      {
        auto prog = std::move(progs.back());
        progs.pop_back();
        progs.back().add(
            If(totalRemaining, std::move(prog), Sequence{{}, {dnai}}, {dnai}));
      }

      assert(progs.size() == 1);
      seq.add(std::move(progs.back()));
    }
    for (int level = post.size() - 1; level >= 0; --level) {
      if (static_cast<unsigned>(level) < reductionCSs.size()) {
        for (const auto &cs : reductionCSs[level]) {
          seq.add(Execute(cs, {dnai}));
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
  p.initialDistributionPartitions = plan.initialDistributionPartitions;
  p.propagationPartitions = Vector<unsigned>(1);
  p.mapping = plan.exchangePlan.fwdMapping;
  p.exchangeBuckets = true;
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
  std::swap(p.initialDistributionPartitions.x,
            p.initialDistributionPartitions.y);
  std::swap(p.propagationPartitions.x, p.propagationPartitions.y);
  p.mapping = plan.exchangePlan.gradAMapping;
  p.metaInfoElemsPerBucket = plan.gradAMetaInfoElemsPerBucket;
  p.method = plan.method.gradA;
  p.dimShuffleToFwd = {0, 2, 1, 3};
  return p;
}

static SinglePassPlan getGradWPlan(const Plan &plan) {
  SinglePassPlan p = getFwdPlan(plan);
  std::swap(p.grouping.y, p.grouping.z);
  std::swap(p.partition.y, p.partition.z);
  std::swap(p.initialDistributionPartitions.y,
            p.initialDistributionPartitions.z);
  p.exchangeBuckets = plan.exchangePlan.gradWExchangeBuckets;
  // When exchanging buckets in the gradW pass we don't unroll the
  // initial distribution or propagation loops.
  if (p.exchangeBuckets) {
    p.initialDistributionPartitions = Vector<unsigned>(1);
    p.propagationPartitions = Vector<unsigned>(1);
  } else {
    p.propagationPartitions = p.initialDistributionPartitions;
  }
  p.mapping = plan.exchangePlan.gradWMapping;
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
    const auto levelPNId =
        plan.mapping.getPNIdForPartition(plan.partition, indices[level]);
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
        "Padding of input to meet grouping not yet handled. "
        "Please make sure the batch size is a multiple of 2 "
        "if using fp32, and 4 for fp16.");
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
                           const DebugNameAndId &dnai, bool padSrc = false) {
  assert(src.size() == dst.size());
  // Just flatten to a single source/dest tensor!
  std::vector<Tensor> flatSrc, flatDst;
  for (std::size_t i = 0; i < src.size(); ++i) {
    auto s = src[i];
    auto d = dst[i];
    // For GradW the source can have a smaller shape
    // in its outer-most non-singular dimension. In this case
    // we allow this and simply trim the destination
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
  prog.add(Copy(concat(flatSrc), concat(flatDst), false, {dnai}));
}

// Rearrange partitions and return a view of an already mapped tensor.
static std::vector<Tensor> rearrangePartitions(
    Graph &graph, const ComputeSet &cs, Sequence &preCopies,
    const std::vector<Tensor> &src, const std::vector<Tensor> &dst,
    const std::vector<unsigned> &srcMemOrdering,
    const std::vector<unsigned> &dstMemOrdering, unsigned grouping,
    bool enableStructuredRearrangements, const poplar::DebugNameAndId &dnai) {
  std::vector<Tensor> dstView;
  dstView.resize(dst.size());
  // We only really know how to deal with 2D transposes and don't expect
  // groups for the timebeing.
  assert(srcMemOrdering.at(0) == 0 && dstMemOrdering.at(0) == 0);
  assert(src.size() == dst.size());
  assert(grouping == 4 || grouping == 8 || grouping == 16);
  const auto inverseDstMemOrdering = inversePermutation(dstMemOrdering);

  // Tensors which don't require a vertex to rearrange are copied in the
  // preCopies program
  std::vector<Tensor> copySrcTensors, copyDstTensors;
  copySrcTensors.reserve(src.size());
  copyDstTensors.reserve(dst.size());

  for (std::size_t i = 0; i < src.size(); ++i) {
    auto s = unfactorDims(src[i], 3);
    auto d = unfactorDims(dst[i], 3);
    assert(s.dim(0) == 1 && d.dim(0) == 1);
    assert(s.shape() == d.shape());
    s = s.dimShuffle(srcMemOrdering).squeeze({0});
    const auto &dataType = s.elementType();
    const auto blockSize = getRearrangementBlockSize(dataType);

    // Detect innermost grouping on ordered source as that puts the
    // innermost dimension with the block size
    // TODO: the innermost grouping could be done in parallel
    const auto innermostGrouping = poputil::detectInnermostGrouping(graph, s);

    // A grouping of multiples given the data type as they are efficient to
    // transpose
    const auto groupingForVertex = dataType == FLOAT ? 2 : 4;
    const bool useVertex = innermostGrouping % groupingForVertex == 0;
    d = d.dimShuffle(dstMemOrdering).squeeze({0});
    assert(s.dim(0) % blockSize == 0);
    assert(s.dim(1) % grouping == 0);
    auto dRearranged = d.reshape({d.dim(0) / grouping, d.dim(1) / blockSize,
                                  grouping, blockSize})
                           .dimShuffle({0, 2, 1, 3})
                           .reshape(d.shape())
                           .expand({0})
                           .dimShuffle(inverseDstMemOrdering);
    const auto dstShape = dst[i].shape();
    const std::vector<std::size_t> dstGroupings(dstShape.end() - 3,
                                                dstShape.end());
    dstView.at(i) = factorDims(dRearranged, dstGroupings);
    if (useVertex && enableStructuredRearrangements) {
      unsigned maxSizePerTile =
          (s.elementType() == FLOAT ? 2 : 4) *
          ((1 << (graph.getTarget().getNumStrideBits() - 1)) - 1);
      if (s.dim(1) > maxSizePerTile) {
        throw poplibs_error("ASM vertex doesn't support dimension size. "
                            "Consider adding an unlimited version");
      }
      const auto vertexClass =
          templateVertex("popsparse::BlockTransposeGradW", s.elementType());
      const auto v = graph.addVertex(
          cs, vertexClass, {{"in", s.flatten()}, {"out", d.flatten()}});
      graph.setInitialValue(v["blockSizeXOrY"], grouping);
      const auto numBlocks = s.dim(1) / grouping;
      graph.setInitialValue(v["numXOrYBlocks"], numBlocks);
      graph.setInitialValue(v["numZ"], s.dim(0));
      graph.setInitialValue(
          v["maxXOrYBlocksPerWorker"],
          ceildiv(numBlocks, graph.getTarget().getNumWorkerContexts()));
      // assume whole tensor lives on the tile
      const auto sliceTileMap = graph.getTileMapping(d.slice(0, 1, 0));
      auto it = std::find_if(sliceTileMap.begin(), sliceTileMap.end(),
                             [&](const std::vector<Interval> &regions) {
                               return !regions.empty();
                             });
      assert(it != sliceTileMap.end());
      const auto tile = std::distance(sliceTileMap.begin(), it);
      graph.setTileMapping(v, tile);
    } else {
      copySrcTensors.push_back(src[i].flatten());
      copyDstTensors.push_back(dstView.at(i).flatten());
    }
  }

  if (!copySrcTensors.empty()) {
    logging::popsparse::debug("copies added in GradW {}", dnai.getPathName());
    preCopies.add(
        Copy(concat(copySrcTensors), concat(copyDstTensors), false, {dnai}));
  }
  return dstView;
}

static void allocatePerPartitionInputs(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    bool isActs, const Type &inputType, const std::vector<unsigned> &hierarchy,
    const unsigned level, Tensor &input, const poplar::DebugNameAndId &dnai) {
  const auto &grouping = plan.grouping;
  std::vector<unsigned> memOrdering;
  std::vector<std::size_t> shapeInternal, inputGrouping;
  if (isActs) {
    memOrdering = getOnTileActsOrdering(plan.method);
    shapeInternal = {shape.groups * grouping.groups, shape.y * grouping.y,
                     shape.z * grouping.z};
    inputGrouping = {grouping.groups, grouping.y, grouping.z};
  } else {
    memOrdering = getOnTileWeightsOrdering(plan.method);
    shapeInternal = {shape.groups * grouping.groups, shape.x * grouping.x,
                     shape.y * grouping.y};
    inputGrouping = {grouping.groups, grouping.x, grouping.y};
  }
  std::vector<std::size_t> shapeAllocation(memOrdering.size());
  for (std::size_t i = 0; i < memOrdering.size(); ++i) {
    shapeAllocation[i] = shapeInternal[memOrdering[i]];
  }
  input = graph.addVariable(inputType, shapeAllocation, {dnai})
              .dimShuffle(inversePermutation(memOrdering));
  input = factorDims(input, inputGrouping);
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  graph.setTileMapping(input, tile);
}

template <typename NextLevelInput>
static void allocatePerPartitionInputs(
    Graph &graph, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    bool isActs, const Type &inputType, const std::vector<unsigned> &hierarchy,
    const unsigned level, std::vector<NextLevelInput> &perPartitionInputs,
    const poplar::DebugNameAndId &dnai) {
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
            graph, subShape, subIndices, plan, isActs, inputType, hierarchy,
            level + 1, perPartitionInputs[partitionIndexFlat], {dnai});
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
  assert(product(plan.initialDistributionPartitions.asStdVector()) ==
         plan.initialDistributionPartitions.z);
  auto bucketPartition = plan.partition;
  bucketPartition.z /= plan.initialDistributionPartitions.z;
  const auto totalPartitions = product(partition.asStdVector());
  nextLevelBuckets.resize(totalPartitions);
  const auto &bucketsByPartition =
      getBucketsByPartition(buckets, bucketPartition);
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
      getBucketsByPartition(buckets, plan.partition);

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
                                const poplar::DebugNameAndId &dnai) {
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
      graph.addVariable(options.partialsType, partialsShapeAllocation, {dnai})
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
                    const poplar::DebugNameAndId &dnai) {
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
                      createPartialsDense(graph, subShape, subIndices, plan,
                                          options, hierarchy, level + 1,
                                          partials[partitionIndexFlat], {dnai});
                    });
}

// Tile-level
static void createPartialsSparse(Graph &graph, const Vector<unsigned> &shape,
                                 const std::vector<Vector<unsigned>> &indices,
                                 const SinglePassPlan &plan,
                                 const Options &options,
                                 const std::vector<unsigned> &hierarchy,
                                 unsigned level, Tensor &partials,
                                 const poplar::DebugNameAndId &dnai) {
  // We include partitions in the shape of partials. This makes it easier
  // to get back to the output shape.
  partials = graph.addVariable(options.partialsType,
                               {1, 1, 1, 1, 1, plan.nzElemsPerBucket}, {dnai});
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
                     const poplar::DebugNameAndId &dnai) {
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
                          level + 1, partials[partitionIndexFlat], {dnai});
                    });
}

static void getSubGroupIds(Graph &graph,
                           const std::vector<Vector<unsigned>> &indices,
                           const SinglePassPlan &plan, const Options &options,
                           const std::vector<unsigned> &hierarchy, unsigned &id,
                           const poplar::DebugNameAndId &dnai) {
  id = getSubGroupId(plan, indices);
}

static void getSubGroupIds(Graph &graph,
                           const std::vector<Vector<unsigned>> &indices,
                           const SinglePassPlan &plan, const Options &options,
                           const std::vector<unsigned> &hierarchy, Tensor &id,
                           const poplar::DebugNameAndId &dnai) {
  const auto val = getSubGroupId(plan, indices);
  id = graph.addConstant(UNSIGNED_SHORT, {1}, val, {dnai});
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  graph.setTileMapping(id, tile);
}

template <typename NextLevelIDs>
static void getSubGroupIds(Graph &graph,
                           const std::vector<Vector<unsigned>> &indices,
                           const SinglePassPlan &plan, const Options &options,
                           const std::vector<unsigned> &hierarchy,
                           std::vector<NextLevelIDs> &ids,
                           const poplar::DebugNameAndId &dnai) {
  ids.resize(product(plan.partition.asStdVector()));
  iteratePartitions(plan.partition, [&](const auto &i) {
    const auto partitionIndexFlat =
        flattenIndex(plan.partition.asStdVector(), i.asStdVector());
    auto subIndices = indices;
    subIndices.emplace_back(i);
    getSubGroupIds(graph, subIndices, plan, options, hierarchy,
                   ids[partitionIndexFlat], {dnai});
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
                    const poplar::DebugNameAndId &dnai) {
  const std::string levelPrefix = "l" + std::to_string(level);
  const auto tile = getPartitionTile(hierarchy, plan, indices);

  // Though it complicates this function slightly, for clarity what we give
  // to the onTileImpl is the dimensions of the block, as ordered for this
  // pass. i.e. Forward = {x, y}, GradA = {y, x}, GradW = {x, y}.
  const auto grouping = plan.grouping.asStdVector<std::size_t>();
  std::vector<std::size_t> indicesOfBlockDims(plan.dimShuffleToFwd.begin() + 1,
                                              plan.dimShuffleToFwd.begin() + 3);
  std::sort(indicesOfBlockDims.begin(), indicesOfBlockDims.end());
  std::array<std::size_t, 2> blockDimensions;
  for (std::size_t i = 0; i < indicesOfBlockDims.size(); ++i) {
    blockDimensions[i] = grouping[indicesOfBlockDims[i]];
  }
  onTileImpl(graph, cs, tile, plan.method, zeroPartials, subGroupId,
             shape.asStdVector<std::size_t>(), metaInfo, weights, acts,
             partials, blockDimensions, {dnai, levelPrefix});
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
        const poplar::DebugNameAndId &dnai) {
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
                {dnai});
      });
}

static Tensor finalReduction(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const Type &resultType, const std::vector<Vector<unsigned>> &indices,
    const SinglePassPlan &plan, const Options &options,
    const std::vector<unsigned> &hierarchy, unsigned level,
    const Tensor &partials, bool forGradW, const poplar::DebugNameAndId &dnai) {
  const std::string levelPrefix = "l" + std::to_string(level);
  const Type &levelResultType = options.partialsType;

  Tensor result = partials;
  if (result.elementType() != levelResultType) {
    result =
        popops::cast(graph, result, levelResultType, progBuilder.post.at(level),
                     {dnai, levelPrefix + "/castResult"});
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
               const poplar::DebugNameAndId &dnai) {
  const std::string levelPrefix = "l" + std::to_string(level);
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
                          partials[partitionIndexFlat], forGradW, {dnai});
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
          progBuilder.reductionCSs.at(level), convOpts, {dnai, levelPrefix});
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
                          {dnai, levelPrefix + "/castResult"});
  }

  return result;
}

static void writeUndefPartitions(Sequence &prog, const std::vector<Tensor> &ts,
                                 const DebugNameAndId &dnai) {
  std::vector<Tensor> flattenedTs;
  for (const auto &t : ts) {
    flattenedTs.emplace_back(t.flatten());
  }
  prog.add(WriteUndef(concat(flattenedTs), {dnai}));
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
                            const poplar::DebugNameAndId &dnai) {
  const std::size_t numBuckets = product(plan.partition.asStdVector());
  const auto buckets =
      graph.addVariable(type, {numBuckets, elemsPerBucket}, {dnai});
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
    Tensor &buffer, const poplar::DebugNameAndId &dnai) {
  const auto tile = getPartitionTile(hierarchy, plan, indices);
  buffer = graph.clone(allocationReference, {dnai});
  graph.setTileMapping(buffer, tile);
}

template <typename NextLevelBuffers>
static void createPropagationBuffers(
    Graph &graph, const Type &inputType, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const Options &options, const std::vector<unsigned> &hierarchy,
    unsigned level, const std::vector<NextLevelBuffers> &reference,
    const Tensor &, std::vector<NextLevelBuffers> &buffers,
    const poplar::DebugNameAndId &dnai) {
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
                                 buffers[partitionIndexFlat], {dnai});
      });
}

static void getPropagationExchangeSources(
    Graph &graph, const SinglePassPlan &plan, const Options &options,
    const Vector<unsigned> &offset, const Tensor &buckets,
    const Tensor &prevBuckets, Tensor &nextLevelSources,
    const poplar::DebugNameAndId &dnai) {
  nextLevelSources = prevBuckets;
}

template <typename NextLevelBuckets, typename NextLevelExchangeSources>
static void getPropagationExchangeSources(
    Graph &graph, const SinglePassPlan &plan, const Options &options,
    const Vector<unsigned> &offset,
    const std::vector<NextLevelBuckets> &buckets,
    const std::vector<NextLevelBuckets> &prevBuckets,
    std::vector<NextLevelExchangeSources> &nextLevelSources,
    const poplar::DebugNameAndId &dnai) {
  nextLevelSources.resize(product(plan.partition.asStdVector()));
  iteratePartitions(plan.partition, [&](const auto &i) {
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
    getPropagationExchangeSources(graph, plan, options, offset,
                                  buckets[idxFlat], buckets[prevIdxFlat],
                                  nextLevelSources[idxFlat], {dnai});
  });
}

static void getBroadcastPropagationExchangeSources(
    const SinglePassPlan &plan, const Options &options,
    const Vector<unsigned> &broadcastSrcIdx,
    const Vector<unsigned> &broadcastFactor, const Tensor &source,
    Tensor &broadcastSource) {
  broadcastSource = source;
}

template <typename NextLevelSources>
static void getBroadcastPropagationExchangeSources(
    const SinglePassPlan &plan, const Options &options,
    const Vector<unsigned> &broadcastSrcIdx,
    const Vector<unsigned> &broadcastFactor,
    const std::vector<NextLevelSources> &sources,
    std::vector<NextLevelSources> &broadcastSources) {
  broadcastSources.resize(product(plan.partition.asStdVector()));
#ifndef NDEBUG
  auto lessThanPred = [](const auto &a, const auto &b) { return a < b; };
#endif
  assert(product(broadcastSrcIdx.binaryOp(broadcastFactor, lessThanPred)
                     .asStdVector()) != 0);
  assert(plan.partition % broadcastFactor == Vector<unsigned>(0));
  iteratePartitions(plan.partition, [&](const auto &i) {
    const auto srcI = (i / broadcastFactor) * broadcastFactor + broadcastSrcIdx;
    const auto srcIdxFlat =
        flattenIndex(plan.partition.asStdVector(), srcI.asStdVector());
    const auto idxFlat =
        flattenIndex(plan.partition.asStdVector(), i.asStdVector());
    getBroadcastPropagationExchangeSources(
        plan, options, {}, {}, sources[srcIdxFlat], broadcastSources[idxFlat]);
  });
}

static void addBufferIncrementProg(Graph &graph, const Tensor &t, Sequence &seq,
                                   unsigned tile,
                                   const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "bufIncrement"});
  auto v = graph.addVertex(
      cs, templateVertex("popsparse::BufferIndexUpdate", t.elementType()));
  graph.connect(v["index"], t);
  graph.setTileMapping(v, tile);
  seq.add(Execute(cs, {dnai}));
}

static void addPropagationExchanges(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const SinglePassPlan &plan, const Options &options,
    const std::vector<unsigned> &hierarchy,
    const MetaInfoAndValues<std::vector<Tensor>> &buckets,
    const MetaInfoAndValues<std::array<std::vector<Tensor>, 2>> &buffers,
    const Tensor &bufferIdx, const poplar::DebugNameAndId &dnai) {
  MetaInfoAndValues<std::vector<Tensor>> homeSources;
  const auto homeOffset = getPropagationStartingOffset(plan);
  getPropagationExchangeSources(graph, plan, options, homeOffset,
                                buckets.metaInfo, {}, homeSources.metaInfo,
                                {dnai});
  getPropagationExchangeSources(graph, plan, options, homeOffset,
                                buckets.values, {}, homeSources.values, {dnai});
  const auto controlFlowTile =
      getTileForDynamicControlFlowManagement(graph, plan);
  // We also set buffer index before exchanging.
  constexpr std::size_t initialBufferIdx = 0;
  const auto bufferIdxInitialVal =
      graph.addConstant(UNSIGNED_INT, {1}, initialBufferIdx, {dnai});
  graph.setTileMapping(bufferIdxInitialVal, controlFlowTile);
  progBuilder.prePropagation.add(
      Copy(bufferIdxInitialVal, bufferIdx, false, {dnai}));
  copyPartitions(graph, progBuilder.prePropagation, homeSources.metaInfo,
                 buffers.metaInfo.at(initialBufferIdx), {dnai});
  copyPartitions(graph, progBuilder.prePropagation, homeSources.values,
                 buffers.values.at(initialBufferIdx), {dnai});
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
      getPropagationExchangeSources(graph, plan, options, offset,
                                    buffers.metaInfo.at(sourceBuffer), {},
                                    bufferSources.metaInfo, {dnai});
      getPropagationExchangeSources(graph, plan, options, offset,
                                    buffers.values.at(sourceBuffer), {},
                                    bufferSources.values, {dnai});
      copyPartitions(graph, exchangeProgs.at(destBuffer),
                     bufferSources.metaInfo, buffers.metaInfo.at(destBuffer),
                     {dnai});
      copyPartitions(graph, exchangeProgs.at(destBuffer), bufferSources.values,
                     buffers.values.at(destBuffer), {dnai});
    }
    // We also toggle buffer index before exchanging
    Sequence prog;
    // Toggle buffer index
    addBufferIncrementProg(graph, bufferIdx, prog, controlFlowTile,
                           {dnai, "toggleBuffer"});
    prog.add(Switch(bufferIdx, {{0, exchangeProgs.at(false)}},
                    exchangeProgs.at(true), {dnai}));
    progBuilder.propagationExchanges.at(dirIdx) = prog;
  }
}

static void addPropagationExchangesGradW(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const SinglePassPlan &plan, const Options &options,
    const std::vector<unsigned> &hierarchy, const std::vector<Tensor> &inputs,
    const std::vector<Tensor> &weights, const std::vector<Tensor> &subGroupIds,
    const std::vector<Tensor> &metaInfo, const std::vector<Tensor> &partials,
    const std::array<std::vector<Tensor>, numBuffers> &inputBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &weightBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &subGroupIdBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &metaInfoBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &partialBuffers,
    const Tensor &bufferIdx, const poplar::DebugNameAndId &dnai) {
  std::vector<Tensor> inputHomeSources, weightHomeSources,
      subGroupIdHomeSources, metaInfoHomeSources, partialHomeSources;
  auto homeOffset = getPropagationStartingOffset(plan);
  // If we are exchanging input/output gradients this is opposite direction
  // to Fwd/GradA.
  if (!plan.exchangeBuckets) {
    homeOffset = (plan.partition - homeOffset);
  }
  homeOffset %= plan.partition;

  const auto controlFlowTile =
      getTileForDynamicControlFlowManagement(graph, plan);
  constexpr std::size_t initialBufferIdx = 0b000;
  const auto bufferIdxInitialVal =
      graph.addConstant(UNSIGNED_INT, {1}, initialBufferIdx, {dnai});
  graph.setTileMapping(bufferIdxInitialVal, controlFlowTile);
  progBuilder.prePropagation.add(
      Copy(bufferIdxInitialVal, bufferIdx, false, {dnai}));
  // WriteUndef the initial buffers. They may be partially written
  // by the first copy of partitions from home locations if there
  // is padding necessary.
  if (plan.exchangeBuckets) {
    writeUndefPartitions(progBuilder.prePropagation,
                         metaInfoBuffers.at(initialBufferIdx), {dnai});
    writeUndefPartitions(progBuilder.prePropagation,
                         partialBuffers.at(initialBufferIdx), {dnai});
    getPropagationExchangeSources(graph, plan, options, homeOffset, metaInfo,
                                  {}, metaInfoHomeSources, {dnai});
    copyPartitions(graph, progBuilder.prePropagation, metaInfoHomeSources,
                   metaInfoBuffers.at(initialBufferIdx), {dnai});
    getPropagationExchangeSources(graph, plan, options, homeOffset, partials,
                                  {}, partialHomeSources, {dnai});
    copyPartitions(graph, progBuilder.prePropagation, partialHomeSources,
                   partialBuffers.at(initialBufferIdx), {dnai});
  } else {
    writeUndefPartitions(progBuilder.prePropagation,
                         inputBuffers.at(initialBufferIdx), {dnai});
    writeUndefPartitions(progBuilder.prePropagation,
                         weightBuffers.at(initialBufferIdx), {dnai});
    writeUndefPartitions(progBuilder.prePropagation,
                         subGroupIdBuffers.at(initialBufferIdx), {dnai});
    getPropagationExchangeSources(graph, plan, options, homeOffset, inputs, {},
                                  inputHomeSources, {dnai});
    copyPartitions(graph, progBuilder.prePropagation, inputHomeSources,
                   inputBuffers.at(initialBufferIdx), {dnai}, true);
    getPropagationExchangeSources(graph, plan, options, homeOffset, weights, {},
                                  weightHomeSources, {dnai});
    copyPartitions(graph, progBuilder.prePropagation, weightHomeSources,
                   weightBuffers.at(initialBufferIdx), {dnai}, true);
    getPropagationExchangeSources(graph, plan, options, homeOffset, subGroupIds,
                                  {}, subGroupIdHomeSources, {dnai});
    copyPartitions(graph, progBuilder.prePropagation, subGroupIdHomeSources,
                   subGroupIdBuffers.at(initialBufferIdx), {dnai});
  }
  // WriteUndef the other buffers. We do this after the initial copy from
  // home location to ensure these buffers aren't live at the same
  // time as exchanging them above.
  for (std::size_t buffer = (initialBufferIdx + 1) % numBuffers;
       buffer != initialBufferIdx; buffer = (buffer + 1) % numBuffers) {
    if (plan.exchangeBuckets) {
      writeUndefPartitions(progBuilder.prePropagation,
                           metaInfoBuffers.at(buffer), {dnai});
      writeUndefPartitions(progBuilder.prePropagation,
                           partialBuffers.at(buffer), {dnai});
    } else {
      writeUndefPartitions(progBuilder.prePropagation, inputBuffers.at(buffer),
                           {dnai});
      writeUndefPartitions(progBuilder.prePropagation, weightBuffers.at(buffer),
                           {dnai});
      writeUndefPartitions(progBuilder.prePropagation,
                           subGroupIdBuffers.at(buffer), {dnai});
    }
  }

  auto offsets = plan.propagationPartitions.asStdVector();
  const auto getDir = [&](unsigned dim) -> Vector<unsigned> {
    std::vector<unsigned> v(4, 0);
    const auto &partition = plan.partition.asStdVector();
    v[dim] = offsets[dim];
    if (!plan.exchangeBuckets) {
      v[dim] = partition[dim] - v[dim];
    }
    v[dim] %= partition[dim];
    return v;
  };
  // We have 3 directions, X, Y, and Z, in which to propagate.
  for (std::size_t dirIdx = 0; dirIdx < numDirections; ++dirIdx) {
    const auto offset = getDir(1 + dirIdx);
    Switch exchangeSwitch(bufferIdx, {dnai});

    unsigned bitsToFlip = 0;
    if (plan.exchangeBuckets) {
      bitsToFlip = 1u;
      for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
        const auto sourceBuffer = (buffer + (numBuffers - 1)) % numBuffers;
        Sequence exchangeProg;
        {
          std::vector<Tensor> sources;
          getPropagationExchangeSources(graph, plan, options, offset,
                                        metaInfoBuffers.at(sourceBuffer), {},
                                        sources, {dnai});
          copyPartitions(graph, exchangeProg, sources,
                         metaInfoBuffers.at(buffer), {dnai});
        }
        {
          std::vector<Tensor> sources;
          getPropagationExchangeSources(graph, plan, options, offset,
                                        partialBuffers.at(sourceBuffer), {},
                                        sources, {dnai});
          copyPartitions(graph, exchangeProg, sources,
                         partialBuffers.at(buffer), {dnai});
        }
        exchangeSwitch.add(buffer, std::move(exchangeProg));
      }
    } else {
      const bool moveInput = offset.y + offset.z > 0;
      const bool moveWeights = offset.x + offset.y > 0;
      const bool moveSubGroupIds = offset.x + offset.z > 0;
      bitsToFlip = unsigned(moveInput) << 2u | unsigned(moveWeights) << 1u |
                   unsigned(moveSubGroupIds) << 0u;
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
              getPropagationExchangeSources(graph, plan, options, offset,
                                            inputBuffers.at(inputSourceBuffer),
                                            {}, sources, {dnai});
              copyPartitions(graph, exchangeProg, sources,
                             inputBuffers.at(inputDestBuffer), {dnai});
            }
            if (moveWeights) {
              std::vector<Tensor> sources;
              getPropagationExchangeSources(
                  graph, plan, options, offset,
                  weightBuffers.at(weightSourceBuffer), {}, sources, {dnai});
              copyPartitions(graph, exchangeProg, sources,
                             weightBuffers.at(weightDestBuffer), {dnai});
            }
            if (moveSubGroupIds) {
              std::vector<Tensor> sources;
              getPropagationExchangeSources(
                  graph, plan, options, offset,
                  subGroupIdBuffers.at(subGroupIdSourceBuffer), {}, sources,
                  {dnai});
              copyPartitions(graph, exchangeProg, sources,
                             subGroupIdBuffers.at(subGroupIdDestBuffer),
                             {dnai});
            }
            exchangeSwitch.add(inputDestBuffer * numBuffers * numBuffers +
                                   weightDestBuffer * numBuffers +
                                   subGroupIdDestBuffer,
                               std::move(exchangeProg));
          }
        }
      }
    }
    Sequence prog;
    // Calculate buffer index.
    const auto bitsToFlipT =
        graph.addConstant(UNSIGNED_INT, {1}, bitsToFlip, {dnai});
    graph.setTileMapping(bitsToFlipT, controlFlowTile);
    popops::bitwiseXorInPlace(graph, bufferIdx, bitsToFlipT, prog,
                              {dnai, "toggleBuffers"});
    prog.add(std::move(exchangeSwitch));
    progBuilder.propagationExchanges.at(dirIdx) = std::move(prog);
  }

  // If we're exchanging buckets, we also need to move partial results
  // from buffers back to their home locations as they are written to/
  // updated. We ensure these always end up at their original position
  // when doing propagation phases.
  if (plan.exchangeBuckets) {
    Switch exchangeSwitch(bufferIdx, {dnai});
    for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
      Sequence exchangeProg;
      copyPartitions(graph, exchangeProg, partialBuffers.at(buffer), partials,
                     {dnai});
      exchangeSwitch.add(buffer, std::move(exchangeProg));
    }
    progBuilder.postPropagation.add(exchangeSwitch);
  }
}

// Transpose blocks within all buckets
static Tensor transposeBuckets(Graph &graph, Sequence &prog,
                               const SinglePassPlan &plan,
                               const Options &options, const Tensor &buckets,
                               const poplar::DebugNameAndId &dnai) {
  // Flatten away buckets dimension, and introduce forward pass block
  // dimensions.
  const Vector<std::size_t> fwdGrouping = shuffleVector(
      plan.grouping.asStdVector<std::size_t>(), plan.dimShuffleToFwd);
  const std::vector<std::size_t> blockDimensions = {fwdGrouping.x,
                                                    fwdGrouping.y};
  const auto blockSize = product(blockDimensions);
  // If the block is not 2-dimensional there is no transpose to do.
  if (sum(blockDimensions) > blockSize) {
    return buckets;
  }
  const auto bucketsFlat = buckets.flatten();
  const auto bucketsWithBlocks =
      bucketsFlat.reshape({bucketsFlat.numElements() / blockSize,
                           blockDimensions[0], blockDimensions[1]});
  const auto cs = graph.addComputeSet({dnai, "transposeBuckets"});
  const auto transposedBuckets =
      popops::rearrange::partialTranspose(graph, bucketsWithBlocks, cs, {dnai});
  prog.add(Execute(cs, {dnai}));
  return transposedBuckets.reshape(buckets.shape());
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
    const bool transposedBuckets, const poplar::DebugNameAndId &dnai) {
  // Only supporting single-IPU currently.
  assert(hierarchy.size() == 1);

  const std::string levelPrefix = "l" + std::to_string(level);
  const auto &inputType = acts.elementType();

  // At the top level for now, before doing anything else, enforce and
  // introduce grouping into the given tensors.
  if (level == 0) {
    acts = groupActs(acts, plan.grouping);
  }

  const Type resultType = inputType;

  if (transposedBuckets) {
    nzValueBuckets = transposeBuckets(graph, progBuilder.preDistribution, plan,
                                      options, nzValueBuckets, {dnai});
  }

  // Transpose the inputs if they do not have an inner dimension compatible
  // with the layout needed on-tile based on the plan.
  {
    auto actsUngrouped = unfactorDims(acts, 3);
    const auto actsMemOrder = getOnTileActsOrdering(plan.method);
    const std::size_t preferredGrouping =
        actsUngrouped.dim(actsMemOrder.back());
    actsUngrouped = popops::rearrange::regroupIfBeneficial(
                        graph,
                        actsUngrouped.dimRoll(actsMemOrder.back(),
                                              actsUngrouped.rank() - 1),
                        preferredGrouping, progBuilder.preDistribution,
                        {dnai, "regroupInput"})
                        .dimRoll(actsUngrouped.rank() - 1, actsMemOrder.back());
    acts = groupActs(actsUngrouped, plan.grouping);
  }

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
                      {dnai, "partials"});
  getSubGroupIds(graph, {}, plan, options, hierarchy, subGroupIds,
                 {dnai, "subGroupIds"});
  // For now the buckets to propagate are just those on each tile without
  // broadcast hence these are just home locations.
  getBucketsByPartition(plan, metaInfoBuckets,
                        nextLevelPropagationBuckets.metaInfo);
  getBucketsByPartition(plan, nzValueBuckets,
                        nextLevelPropagationBuckets.values);
  const auto controlFlowTile =
      getTileForDynamicControlFlowManagement(graph, plan);
  const Tensor bufferIdx =
      graph.addVariable(UNSIGNED_INT, {}, {dnai, "bufferIdx"});
  graph.setTileMapping(bufferIdx, controlFlowTile);

  // Pre-arrange the inputs for the next level and hold onto them to
  // avoid exchanging multiple times in propagation phases (if they
  // are present).
  {
    std::vector<Tensor> perPartitionInputs;
    allocatePerPartitionInputs(graph, shape, {}, plan, true, inputType,
                               hierarchy, 0, perPartitionInputs,
                               {dnai, "partitionedInputs"});
    copyPartitions(graph, progBuilder.preDistribution, nextLevelInputs,
                   perPartitionInputs, {dnai});
    std::swap(nextLevelInputs, perPartitionInputs);
  }

  writeUndefPartitions(progBuilder.preDistribution, partials, {dnai});
  const auto distributionCS =
      graph.addComputeSet({dnai, "ComputePartialsInitialDistribution"});
  compute(graph, distributionCS, shape, {}, plan, options, hierarchy, 0, true,
          nextLevelInputs, nextLevelDistributionBuckets.values,
          nextLevelDistributionBuckets.metaInfo, partials, subGroupIds, {dnai});
  progBuilder.distributionCompute.add(Execute(distributionCS, {dnai}));

  // We need a set of buffers on each tile for use during dynamic exchange
  // and compute steps (propagation phase).
  MetaInfoAndValues<std::array<std::vector<Tensor>, numBuffers>>
      propagationBuffers;
  for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
    const auto metaInfoBuffer = createBuckets(
        graph, UNSIGNED_SHORT, plan, plan.metaInfoElemsPerBucket, hierarchy,
        {dnai, "metaInfoPropagationBuffer" + std::to_string(buffer)});
    const auto nzValueBuffer = createBuckets(
        graph, inputType, plan, plan.nzElemsPerBucket, hierarchy,
        {dnai, "nzValuesPropagationBuffer" + std::to_string(buffer)});
    getBucketsByPartition(plan, metaInfoBuffer,
                          propagationBuffers.metaInfo[buffer]);
    getBucketsByPartition(plan, nzValueBuffer,
                          propagationBuffers.values[buffer]);
    // WriteUndef buffers as they are written to/read from during dynamic
    // control flow
    progBuilder.prePropagation.add(WriteUndef(metaInfoBuffer, {dnai}));
    progBuilder.prePropagation.add(WriteUndef(nzValueBuffer, {dnai}));
  }
  addPropagationExchanges(graph, progBuilder, shape, plan, options, hierarchy,
                          nextLevelPropagationBuckets, propagationBuffers,
                          bufferIdx, {dnai});
  std::array<ComputeSet, numBuffers> propagationCS;
  for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
    propagationCS.at(buffer) = graph.addComputeSet(
        {dnai, "ComputePartialsPropagateBuffer" + std::to_string(buffer)});
    compute(
        graph, propagationCS.at(buffer), shape, {}, plan, options, hierarchy, 0,
        false, nextLevelInputs, propagationBuffers.values.at(buffer),
        propagationBuffers.metaInfo.at(buffer), partials, subGroupIds, {dnai});
  }
  progBuilder.propagationCompute = {
      Switch(bufferIdx, {{0, Execute(propagationCS.at(false), {dnai})}},
             Execute(propagationCS.at(true), {dnai}), {dnai})};
  const auto output =
      finalReduction(graph, progBuilder, shape, resultType, {}, plan, options,
                     hierarchy, 0, partials, false, {dnai});

  return unfactorDims(output, 3);
}

static void computeInitialDistributionGradW(
    Graph &graph, Sequence &prog, const Vector<unsigned> &shape,
    const SinglePassPlan &plan, const Options &options,
    const std::vector<unsigned> &hierarchy,
    const std::vector<Tensor> &nextLevelInputs,
    const std::vector<Tensor> &nextLevelWeights,
    const std::vector<Tensor> &nextLevelMetaInfoBuckets,
    const std::vector<Tensor> &partials, const std::vector<Tensor> &subGroupIds,
    const std::vector<Tensor> &inputBuffer,
    const std::vector<Tensor> &weightBuffer,
    const std::vector<Tensor> &subGroupIdBuffer,
    const poplar::DebugNameAndId &dnai) {
  // For GradW we don't have a specific vertex that can handle multiple
  // partitions at a time. Instead we use a special exchange and compute
  // pattern that broadcasts each partition we need to handle to all partitions
  // which at least allows us to take advantage of 64-bit exchange where
  // possible when we move inputs/output gradients over exchange.
  // If we move buckets instead then there is no advantage to this.
  if (product(plan.initialDistributionPartitions.asStdVector()) == 1) {
    const auto cs =
        graph.addComputeSet({dnai, "ComputePartialsInitialDistribution"});
    compute(graph, cs, shape, {}, plan, options, hierarchy, 0, true,
            nextLevelInputs, nextLevelWeights, nextLevelMetaInfoBuckets,
            partials, subGroupIds, {dnai});
    prog.add(Execute(cs, {dnai}));
  } else {
    const auto broadcastFactor = plan.initialDistributionPartitions;
    bool moveSubGroupIds = broadcastFactor.x * broadcastFactor.z > 1;
    bool moveInputs = broadcastFactor.y * broadcastFactor.z > 1;
    bool moveWeights = broadcastFactor.x * broadcastFactor.y > 1;
    std::array<ComputeSet, 2> computeCSs;
    for (bool zeroPartials : {false, true}) {
      computeCSs[zeroPartials] = graph.addComputeSet(
          {dnai, std::string("ComputePartialsInitialDistribution") +
                     (zeroPartials ? "ZeroPartials" : "")});
      compute(graph, computeCSs[zeroPartials], shape, {}, plan, options,
              hierarchy, 0, zeroPartials,
              moveInputs ? inputBuffer : nextLevelInputs,
              moveWeights ? weightBuffer : nextLevelWeights,
              nextLevelMetaInfoBuckets, partials,
              moveSubGroupIds ? subGroupIdBuffer : subGroupIds, {dnai});
    }
    Sequence computeProg;
    bool zeroPartials = true;
    // WriteUndef the buffers we use in the compute programs when Z is
    // split. These may be partially written when copying partitions.
    if (moveInputs) {
      writeUndefPartitions(computeProg, inputBuffer, {dnai});
    }
    if (moveWeights) {
      writeUndefPartitions(computeProg, weightBuffer, {dnai});
    }
    if (moveSubGroupIds) {
      writeUndefPartitions(computeProg, subGroupIdBuffer, {dnai});
    }
    iteratePartitions(broadcastFactor, [&](const auto &i) {
      if (moveInputs) {
        std::vector<Tensor> broadcastInputSource;
        getBroadcastPropagationExchangeSources(plan, options, i,
                                               broadcastFactor, nextLevelInputs,
                                               broadcastInputSource);
        copyPartitions(graph, computeProg, broadcastInputSource, inputBuffer,
                       {dnai}, true);
      }
      if (moveWeights) {
        std::vector<Tensor> broadcastWeightSource;
        getBroadcastPropagationExchangeSources(
            plan, options, i, broadcastFactor, nextLevelWeights,
            broadcastWeightSource);
        copyPartitions(graph, computeProg, broadcastWeightSource, weightBuffer,
                       {dnai}, true);
      }
      if (moveSubGroupIds) {
        std::vector<Tensor> broadcastSubGroupIdSource;
        getBroadcastPropagationExchangeSources(plan, options, i,
                                               broadcastFactor, subGroupIds,
                                               broadcastSubGroupIdSource);
        copyPartitions(graph, computeProg, broadcastSubGroupIdSource,
                       subGroupIdBuffer, {dnai});
      }
      computeProg.add(Execute(computeCSs[zeroPartials], {dnai}));
      zeroPartials = false;
    });
    prog.add(std::move(computeProg));
  }
}

static void computePropagationGradW(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const SinglePassPlan &plan, const Options &options,
    const std::vector<unsigned> &hierarchy, const std::vector<Tensor> &inputs,
    const std::vector<Tensor> &weights, const std::vector<Tensor> &subGroupIds,
    const std::vector<Tensor> &metaInfo, const std::vector<Tensor> &partials,
    const std::array<std::vector<Tensor>, numBuffers> &inputBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &weightBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &subGroupIdBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &metaInfoBuffers,
    const std::array<std::vector<Tensor>, numBuffers> &partialBuffers,
    const Tensor &bufferIdx, const poplar::DebugNameAndId &dnai) {

  std::vector<Program> computeProgs;
  if (plan.exchangeBuckets) {
    for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
      const auto cs = graph.addComputeSet(
          {dnai, "ComputePartialsPropagationBuffer" + std::to_string(buffer)});
      compute(graph, cs, shape, {}, plan, options, hierarchy, 0, false, inputs,
              weights, metaInfoBuffers.at(buffer), partialBuffers.at(buffer),
              subGroupIds, {dnai});
      computeProgs.push_back(Execute(cs, {dnai}));
    }
  } else {
    // Create compute programs that operate on each combination of
    // input/weight/subGroupId buffers.
    std::vector<std::size_t> computeProgIndexedShape = {numBuffers, numBuffers,
                                                        numBuffers};
    computeProgs.resize(product(computeProgIndexedShape));
    for (std::size_t inputBuffer = 0; inputBuffer < numBuffers; ++inputBuffer) {
      for (std::size_t weightBuffer = 0; weightBuffer < numBuffers;
           ++weightBuffer) {
        for (std::size_t subGroupIdBuffer = 0; subGroupIdBuffer < numBuffers;
             ++subGroupIdBuffer) {
          const auto cs = graph.addComputeSet(
              {dnai, "ComputePartialsPropagateBufferIn" +
                         std::to_string(inputBuffer) + "Weights" +
                         std::to_string(weightBuffer) + "SubGroupId" +
                         std::to_string(subGroupIdBuffer)});
          compute(graph, cs, shape, {}, plan, options, hierarchy, 0, false,
                  inputBuffers.at(inputBuffer), weightBuffers.at(weightBuffer),
                  metaInfo, partials, subGroupIdBuffers.at(subGroupIdBuffer),
                  {dnai});
          computeProgs.at(
              flattenIndex(computeProgIndexedShape,
                           {inputBuffer, weightBuffer, subGroupIdBuffer})) =
              Execute(cs, {dnai});
        }
      }
    }

    if (product(plan.propagationPartitions.asStdVector()) > 1) {
      const auto broadcastFactor = plan.propagationPartitions;
      bool moveInputs = broadcastFactor.y * broadcastFactor.z > 1;
      bool moveWeights = broadcastFactor.x * broadcastFactor.y > 1;
      bool moveSubGroupIds = broadcastFactor.x * broadcastFactor.z > 1;

      std::vector<Program> computeCSs(computeProgs.size());
      std::swap(computeProgs, computeCSs);
      for (std::size_t inputBuffer = 0; inputBuffer < numBuffers;
           ++inputBuffer) {
        for (std::size_t weightBuffer = 0; weightBuffer < numBuffers;
             ++weightBuffer) {
          for (std::size_t subGroupIdBuffer = 0; subGroupIdBuffer < numBuffers;
               ++subGroupIdBuffer) {
            const auto computeInputBuffer =
                (inputBuffer + moveInputs) % numBuffers;
            const auto computeWeightBuffer =
                (weightBuffer + moveWeights) % numBuffers;
            const auto computeSubGroupIdBuffer =
                (subGroupIdBuffer + moveSubGroupIds) % numBuffers;
            const auto computeCS = computeCSs.at(
                flattenIndex(computeProgIndexedShape,
                             {computeInputBuffer, computeWeightBuffer,
                              computeSubGroupIdBuffer}));

            Sequence computeProg;
            iteratePartitions(broadcastFactor, [&](const auto &i) {
              if (moveInputs) {
                std::vector<Tensor> broadcastSource;
                getBroadcastPropagationExchangeSources(
                    plan, options, i, broadcastFactor,
                    inputBuffers.at(inputBuffer), broadcastSource);
                copyPartitions(graph, computeProg, broadcastSource,
                               inputBuffers.at(computeInputBuffer), {dnai});
              }
              if (moveWeights) {
                std::vector<Tensor> broadcastSource;
                getBroadcastPropagationExchangeSources(
                    plan, options, i, broadcastFactor,
                    weightBuffers.at(weightBuffer), broadcastSource);
                copyPartitions(graph, computeProg, broadcastSource,
                               weightBuffers.at(computeWeightBuffer), {dnai});
              }
              if (moveSubGroupIds) {
                std::vector<Tensor> broadcastSource;
                getBroadcastPropagationExchangeSources(
                    plan, options, i, broadcastFactor,
                    subGroupIdBuffers.at(subGroupIdBuffer), broadcastSource);
                copyPartitions(graph, computeProg, broadcastSource,
                               subGroupIdBuffers.at(computeSubGroupIdBuffer),
                               {dnai});
              }
              computeProg.add(computeCS);
            });
            computeProgs.at(
                flattenIndex(computeProgIndexedShape,
                             {inputBuffer, weightBuffer, subGroupIdBuffer})) =
                std::move(computeProg);
          }
        }
      }
    }
  }

  auto computeSwitch = Switch(bufferIdx, {dnai});

  for (std::size_t i = 0; i < computeProgs.size(); ++i) {
    computeSwitch.add(i, computeProgs[i]);
  }

  progBuilder.propagationCompute.add(computeSwitch);

  if (plan.exchangeBuckets) {
    progBuilder.setPropagationMustReachStartingOffset();
  }
}

static Tensor fullyConnectedSparseGradWImpl(
    Graph &graph, ProgBuilder &progBuilder, const Vector<unsigned> &shape,
    const std::vector<Vector<unsigned>> &indices, const SinglePassPlan &plan,
    const OnTileMethod &fwdMethod, const OnTileMethod &gradAMethod,
    const Options &options, const std::vector<unsigned> &hierarchy,
    unsigned level, Tensor metaInfoBuckets, Tensor weights, Tensor acts,
    const poplar::DebugNameAndId &dnai) {
  // Only supporting single-IPU currently.
  assert(hierarchy.size() == 1);

  const std::string levelPrefix = "l" + std::to_string(level);
  const auto &inputType = acts.elementType();

  // At the top level for now, before doing anything else, enforce and
  // introduce grouping into the given tensors.
  if (level == 0) {
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
                       {dnai, "partials"});
  std::vector<Tensor> subGroupIds;
  getSubGroupIds(graph, {}, plan, options, hierarchy, subGroupIds,
                 {dnai, "subGroupIds"});

  // Pre-arrange inputs and weights. We use the plans for forward and grada
  // passes to see if a transpose is needed of input/weights. This is aggressive
  // in that we assume the layout of these operands is that we would expect
  // from the plan for forward/grada passes even though it may well not be.
  //
  // The intention is to solve the more fiddly question of more efficient
  // rearrangements of non-weight operands at a later date all at once but
  // let us achieve the performance we expect for now in benchmarking when
  // we know our assumptions about layout are met.
  {
    auto inputSrcMemOrdering = getOnTileActsOrdering(fwdMethod);
    // Ordering of dimensions of activations is transposed as compared to
    // forward pass.
    std::swap(inputSrcMemOrdering.at(1), inputSrcMemOrdering.at(2));
    const auto inputDstMemOrdering = getOnTileActsOrdering(plan.method);
    const auto weightsSrcMemOrdering = getOnTileActsOrdering(gradAMethod);
    const auto weightsDstMemOrdering = getOnTileWeightsOrdering(plan.method);

    // Pre-arrange meta-info as this should stay along-side partials and never
    // move between tiles.
    {
      std::vector<Tensor> perPartition;
      const auto prearranged =
          createBuckets(graph, metaInfoBuckets.elementType(), plan,
                        plan.metaInfoElemsPerBucket, hierarchy,
                        {dnai, "partitionedMetaInfoBuckets"});
      getBucketsByPartition(plan, prearranged, perPartition);
      copyPartitions(graph, progBuilder.preDistribution,
                     nextLevelMetaInfoBuckets, perPartition, {dnai});
      std::swap(nextLevelMetaInfoBuckets, perPartition);
    }

    // Transpose partitions depending on the expected layout of inputs based
    // on the forward/grada plan.
    const auto transposeCS = graph.addComputeSet({dnai, "transposeInputs"});
    // If the transpose vertices are not generated then an explicit copy
    // is done to the destination
    Sequence rearrangePreCopies;
    if (inputSrcMemOrdering != inputDstMemOrdering) {
      std::vector<Tensor> perPartition;
      allocatePerPartitionInputs(graph, shape, {}, plan, true, inputType,
                                 hierarchy, 0, perPartition,
                                 {dnai, "partitionedInputs"});
      auto perPartitionView = rearrangePartitions(
          graph, transposeCS, rearrangePreCopies, nextLevelInputs, perPartition,
          inputSrcMemOrdering, inputDstMemOrdering, plan.grouping.z,
          options.enableStructuredRearrangements, {dnai, "transposeInputs"});
      std::swap(nextLevelInputs, perPartitionView);
    }
    if (weightsSrcMemOrdering != weightsDstMemOrdering) {
      std::vector<Tensor> perPartition;
      allocatePerPartitionInputs(graph, shape, {}, plan, false, inputType,
                                 hierarchy, 0, perPartition,
                                 {dnai, "partitionedWeights"});
      auto perPartitionView = rearrangePartitions(
          graph, transposeCS, rearrangePreCopies, nextLevelWeights,
          perPartition, weightsSrcMemOrdering, weightsDstMemOrdering,
          plan.grouping.x, options.enableStructuredRearrangements,
          {dnai, "transposeWeights"});
      std::swap(nextLevelWeights, perPartitionView);
    }
    progBuilder.preDistribution.add(rearrangePreCopies);
    progBuilder.preDistribution.add(Execute(transposeCS, {dnai}));
  }

  std::array<std::vector<Tensor>, numBuffers> inputPropagationBuffers,
      weightPropagationBuffers, subGroupIdPropagationBuffers,
      metaInfoPropagationBuffers, partialPropagationBuffers;
  for (std::size_t buffer = 0; buffer < numBuffers; ++buffer) {
    if (plan.exchangeBuckets) {
      createPropagationBuffers(
          graph, UNSIGNED_SHORT, shape, {}, plan, options, hierarchy, 0,
          nextLevelMetaInfoBuckets, {}, metaInfoPropagationBuffers.at(buffer),
          {dnai, "metaInfoPropagationBuffer" + std::to_string(buffer)});
      createPropagationBuffers(
          graph, options.partialsType, shape, {}, plan, options, hierarchy, 0,
          partials, {}, partialPropagationBuffers.at(buffer),
          {dnai, "partialPropagationBuffer" + std::to_string(buffer)});
    } else {
      createPropagationBuffers(
          graph, inputType, shape, {}, plan, options, hierarchy, 0,
          nextLevelInputs, {}, inputPropagationBuffers.at(buffer),
          {dnai, "inputPropagationBuffer" + std::to_string(buffer)});
      createPropagationBuffers(
          graph, inputType, shape, {}, plan, options, hierarchy, 0,
          nextLevelWeights, {}, weightPropagationBuffers.at(buffer),
          {dnai, "weightPropagationBuffer" + std::to_string(buffer)});
      createPropagationBuffers(
          graph, inputType, shape, {}, plan, options, hierarchy, 0, subGroupIds,
          {}, subGroupIdPropagationBuffers.at(buffer),
          {dnai, "subGroupIdPropagationBuffer" + std::to_string(buffer)});
    }
  }

  writeUndefPartitions(progBuilder.preDistribution, partials, {dnai});
  computeInitialDistributionGradW(
      graph, progBuilder.distributionCompute, shape, plan, options, hierarchy,
      nextLevelInputs, nextLevelWeights, nextLevelMetaInfoBuckets, partials,
      subGroupIds, inputPropagationBuffers.at(0),
      weightPropagationBuffers.at(0), subGroupIdPropagationBuffers.at(0),
      {dnai});

  const auto controlFlowTile =
      getTileForDynamicControlFlowManagement(graph, plan);
  const auto bufferIdx =
      graph.addVariable(UNSIGNED_INT, {}, {dnai, "bufferIdx"});
  graph.setTileMapping(bufferIdx, controlFlowTile);
  addPropagationExchangesGradW(
      graph, progBuilder, shape, plan, options, hierarchy, nextLevelInputs,
      nextLevelWeights, subGroupIds, nextLevelMetaInfoBuckets, partials,
      inputPropagationBuffers, weightPropagationBuffers,
      subGroupIdPropagationBuffers, metaInfoPropagationBuffers,
      partialPropagationBuffers, bufferIdx, {dnai});

  computePropagationGradW(
      graph, progBuilder, shape, plan, options, hierarchy, nextLevelInputs,
      nextLevelWeights, subGroupIds, nextLevelMetaInfoBuckets, partials,
      inputPropagationBuffers, weightPropagationBuffers,
      subGroupIdPropagationBuffers, metaInfoPropagationBuffers,
      partialPropagationBuffers, bufferIdx, {dnai});

  const auto output =
      finalReduction(graph, progBuilder, shape, resultType, {}, plan, options,
                     hierarchy, 0, partials, true, {dnai});
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
  auto planConstraints = poplin::groupedMatMulPlanConstraints(
      graph, inputType, options.partialsType,
      {fcParams.getNumGroups(), fcParams.getInputChannelsPerGroup(),
       fcParams.getBatchSize()},
      {fcParams.getNumGroups(), fcParams.getBatchSize(),
       fcParams.getOutputChannelsPerGroup()},
      matMulOptions);
  auto serialSplits = poplin::getMatMulSerialSplits(planConstraints);
  return std::make_tuple(std::get<0>(serialSplits), std::get<1>(serialSplits),
                         std::get<2>(serialSplits));
}

static std::vector<std::size_t>
getWeightsShape(const FullyConnectedParams &params) {
  return {params.getNumGroups(), params.getInputChannelsPerGroup(),
          params.getOutputChannelsPerGroup()};
}

static std::vector<std::size_t>
getActivationsShape(const FullyConnectedParams &params) {
  return {params.getNumGroups(), params.getBatchSize(),
          params.getInputChannelsPerGroup()};
}

static poplin::matmul::PlanningCache *getDenseCache(PlanningCache *cache) {
  if (cache == nullptr) {
    return nullptr;
  }
  return cache->impl->denseCache;
}

static OptionFlags getDenseOptionFlags(const Options &options,
                                       std::string pass) {

  OptionFlags result;
  result.set("availableMemoryProportion",
             std::to_string(options.availableMemoryProportion));
  result.set("partialsType", options.partialsType.toString());
  result.set("fullyConnectedPass", std::move(pass));
  return result;
}

static SparseTensor createDenseWeights(
    Graph &graph, const Type &inputType, const FullyConnectedParams &params,
    const poplar::DebugContext &debugContext, PlanningCache *cache,
    const poplar::Tensor &metaInfoTensor,
    std::unique_ptr<TensorMetaDataBase> opMetaData, const Options &options) {
  auto denseTensor = poplin::createMatMulGroupedInputRHS(
      graph, inputType, inputType, getActivationsShape(params),
      getWeightsShape(params), debugContext,
      getDenseOptionFlags(options, options.doGradAPass || options.doGradWPass
                                       ? "TRAINING_FWD"
                                       : "INFERENCE_FWD"),
      getDenseCache(cache));

  return SparseTensor(metaInfoTensor, denseTensor.flatten(),
                      std::move(opMetaData));
}

SparseTensor createFullyConnectedWeights(
    Graph &graph, const Type &inputType, const FullyConnectedParams &params,
    const poplar::DebugContext &debugContext, const OptionFlags &optionFlags,
    PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(inputType, params, optionFlags, cache));

  const auto &options = parseOptionFlags(optionFlags);
  logging::popsparse::debug(
      "popsparse::createFullyConnectedWeights: '{}' params={}, options={}",
      debugContext.getPathName(), params, options);

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
      {di, "metaInfoForward"});
  const auto gradAMetaInfoBuckets =
      plan.sharedBuckets() ? Tensor()
                           : createBuckets(graph, UNSIGNED_SHORT, fwdPlan,
                                           plan.gradAMetaInfoElemsPerBucket,
                                           hierarchy, {di, "metaInfoGradA"});
  const auto nzValueBuckets =
      createBuckets(graph, inputType, fwdPlan, plan.nzElemsPerBucket, hierarchy,
                    {di, "nzValues"});
  const auto weightBuckets =
      SparseTensor(plan.sharedBuckets()
                       ? fwdMetaInfoBuckets
                       : concat(fwdMetaInfoBuckets, gradAMetaInfoBuckets, 1),
                   nzValueBuckets);
  const auto overflowInfoElems = getNumOverflowInfoElems(
      target.getTypeSize(UNSIGNED_SHORT), plan.partition.x, plan.partition.y,
      plan.partition.z);
  const auto controlFlowTile =
      getTileForDynamicControlFlowManagement(graph, fwdPlan);
  std::vector<Tensor> overflowInfoTs;
  for (const auto &name : {"X", "Y", "Z"}) {
    overflowInfoTs.emplace_back(graph.addVariable(
        UNSIGNED_SHORT, {1}, {di, std::string("overflowInfo") + name}));
  }
  assert(overflowInfoElems >= overflowInfoTs.size());
  overflowInfoTs.emplace_back(graph.addVariable(
      UNSIGNED_SHORT, {overflowInfoElems - overflowInfoTs.size()},
      {di, "overflowInfoOuterIterationsToSkip"}));
  const auto overflowInfo = concat(overflowInfoTs);
  graph.setTileMapping(overflowInfo, controlFlowTile);

  const auto packed =
      packWeights(weightBuckets, getTotalMetaInfoElemsPerBuckets(plan),
                  plan.nzElemsPerBucket, overflowInfo);

  // Attach meta-data to the sparse tensor.
  std::unique_ptr<TensorMetaDataBase> opMetaData =
      std::make_unique<FullyConnectedTensorMetaData>(params, options);

  if (plan.useDense) {
    return createDenseWeights(graph, inputType, params, debugContext, cache,
                              packed.getMetaInfoTensor(), std::move(opMetaData),
                              options);
  }
  return SparseTensor(packed.getMetaInfoTensor(), packed.getNzValuesTensor(),
                      std::move(opMetaData));
}

Tensor createFullyConnectedInput(Graph &graph, const Type &inputType,
                                 const FullyConnectedParams &params,
                                 const poplar::DebugContext &debugContext,
                                 const OptionFlags &optionFlags,
                                 PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(inputType, params, optionFlags, cache));
  const auto &options = parseOptionFlags(optionFlags);
  logging::popsparse::debug(
      "popsparse::createFullyConnectedInput: '{}' params={}, options={}",
      debugContext.getPathName(), params, options);

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

  const auto input = graph.addVariable(inputType, inputShapeAllocation, {di})
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

static Tensor denseFullyConnectedFwd(Graph &graph, const SparseTensor &weights,
                                     const Tensor &activations,
                                     const FullyConnectedParams &params,
                                     Sequence &prog, PlanningCache *cache,
                                     const poplar::DebugContext &debugContext,
                                     const Options &options) {
  return poplin::matMulGrouped(
      graph, activations.reshape(getActivationsShape(params)),
      weights.getNzValuesTensor().reshape(getWeightsShape(params)), prog,
      activations.elementType(), debugContext,
      getDenseOptionFlags(options, options.doGradAPass || options.doGradWPass
                                       ? "TRAINING_FWD"
                                       : "INFERENCE_FWD"),
      getDenseCache(cache));
}

Tensor fullyConnectedFwd(Graph &graph, const SparseTensor &weights,
                         const Tensor &activations,
                         const FullyConnectedParams &params, Sequence &prog,
                         const poplar::DebugContext &debugContext,
                         const OptionFlags &optionFlags, PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(weights, activations, params, optionFlags, cache));
  // TODO: Parameter validation - shapes/sizes match given params etc.
  const auto &target = graph.getTarget();
  const auto &inputType = activations.elementType();
  const auto &options = parseOptionFlags(optionFlags);
  logging::popsparse::debug(
      "popsparse::fullyConnectedFwd: '{}' params={}, options={}",
      debugContext.getPathName(), params, options);
  validateSparseOperandMetaData(weights, params, options);
  Plan plan;
  Cost cost;
  std::tie(plan, cost) = getPlan(target, inputType, params, optionFlags, cache);

  if (plan.useDense) {
    return denseFullyConnectedFwd(graph, weights, activations, params, prog,
                                  cache, {{di}, "denseOperation"}, options);
  }

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
  ProgBuilder progBuilder(graph, hierarchy, {di});
  std::vector<Vector<unsigned>> indices;
  constexpr bool transposedBuckets = false;
  const auto &outputActivations = fullyConnectedImpl(
      graph, progBuilder, shape, indices, fwdPlan, options, hierarchy,
      0u /* level */, weightBuckets.getMetaInfoTensor(),
      weightBuckets.getNzValuesTensor(), input, transposedBuckets, {di});
  progBuilder.addToSequence(graph, prog, fwdPlan, overflowInfo, {di});

  return inputInternalToExternalShape(outputActivations, shape.groups);
}

static std::vector<std::size_t>
getOutputShape(const FullyConnectedParams &params) {
  return {params.getNumGroups(), params.getBatchSize(),
          params.getOutputChannelsPerGroup()};
}

static Tensor denseGradAImpl(Graph &graph, const SparseTensor &weights,
                             const Tensor &activations,
                             const FullyConnectedParams &params, Sequence &prog,
                             const poplar::DebugContext &debugContext,
                             PlanningCache *cache, const Options &options) {
  auto denseWeights =
      weights.getNzValuesTensor().reshape(getWeightsShape(params));
  auto acts = activations.reshape(getOutputShape(params));
  auto weightsTransposed = poplin::transposeGroupedMatrix(denseWeights);
  return poplin::matMulGrouped(graph, activations, weightsTransposed, prog,
                               activations.elementType(), debugContext,
                               getDenseOptionFlags(options, "TRAINING_BWD"),
                               getDenseCache(cache));
}

Tensor fullyConnectedGradA(Graph &graph, const SparseTensor &weights,
                           const Tensor &activations,
                           const FullyConnectedParams &params, Sequence &prog,
                           const poplar::DebugContext &debugContext,
                           const OptionFlags &optionFlags,
                           PlanningCache *cache) {
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(weights, activations, params, optionFlags, cache));
  const auto &target = graph.getTarget();
  const auto &inputType = activations.elementType();
  const auto &options = parseOptionFlags(optionFlags);
  logging::popsparse::debug(
      "popsparse::fullyConnectedGradA: '{}' params={}, options={}",
      debugContext.getPathName(), params, options);
  validateSparseOperandMetaData(weights, params, options);
  Plan plan;
  Cost cost;
  std::tie(plan, cost) = getPlan(target, inputType, params, optionFlags, cache);

  if (plan.useDense) {
    return denseGradAImpl(graph, weights, activations, params, prog,
                          {{di}, "denseBwds"}, cache, options);
  }

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
  weightBuckets = SparseTensor(
      getBucketsByPartition(weightBuckets.getMetaInfoTensor(), plan.partition),
      getBucketsByPartition(weightBuckets.getNzValuesTensor(), plan.partition));
  const std::vector<unsigned> shuffleSrc = {0, 1, 2, 3};
  const auto shuffleDest = vectorConvert<unsigned>(gradAPlan.dimShuffleToFwd);
  weightBuckets = SparseTensor(weightBuckets.getMetaInfoTensor()
                                   .dimShufflePartial(shuffleSrc, shuffleDest)
                                   .flatten(0, 5),
                               weightBuckets.getNzValuesTensor()
                                   .dimShufflePartial(shuffleSrc, shuffleDest)
                                   .flatten(0, 5));

  ProgBuilder progBuilder(graph, hierarchy, {di});
  std::vector<Vector<unsigned>> indices;
  constexpr bool transposedBuckets = true;
  const auto &inputGradients = fullyConnectedImpl(
      graph, progBuilder, shape, indices, gradAPlan, options, hierarchy,
      0u /* level */, weightBuckets.getMetaInfoTensor(),
      weightBuckets.getNzValuesTensor(), input, transposedBuckets, {di});
  progBuilder.addToSequence(graph, prog, gradAPlan, overflowInfo, {di});

  return inputInternalToExternalShape(inputGradients, shape.groups);
}

Tensor fullyConnectedSparseGradW(Graph &graph, const Tensor sparsityMetaInfo,
                                 const Tensor &gradA, const Tensor &activations,
                                 const FullyConnectedParams &params,
                                 Sequence &prog,
                                 const poplar::DebugContext &debugContext,
                                 const OptionFlags &optionFlags,
                                 PlanningCache *cache) {
  // TODO: Should this take meta-data for sparse tensor for validation?
  // Validation is the only purpose this serves right now.
  POPSPARSE_TRACEPOINT();
  poputil::PoplibsOpDebugInfo di(debugContext,
                                 DI_ARGS(sparsityMetaInfo, gradA, activations,
                                         params, optionFlags, cache));
  const auto &target = graph.getTarget();
  const auto &inputType = activations.elementType();
  const auto &options = parseOptionFlags(optionFlags);
  logging::popsparse::debug(
      "popsparse::fullyConnectedSparseGradW: '{}' params={}, options={}",
      debugContext.getPathName(), params, options);
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
  metaInfoBuckets = getBucketsByPartition(metaInfoBuckets, plan.partition);
  metaInfoBuckets =
      metaInfoBuckets
          .dimShufflePartial({0, 1, 2, 3},
                             vectorConvert<unsigned>(gradWPlan.dimShuffleToFwd))
          .flatten(0, 5);

  ProgBuilder progBuilder(graph, hierarchy, {di});
  std::vector<Vector<unsigned>> indices;
  auto weightGradientBuckets = fullyConnectedSparseGradWImpl(
      graph, progBuilder, shape, indices, gradWPlan, plan.method.fwd,
      plan.method.gradA, options, hierarchy, 0u /* level */, metaInfoBuckets,
      outputGrad, input, {di});
  progBuilder.addToSequence(graph, prog, gradWPlan, overflowInfo, {di});

  // Rearrange resulting weight gradient buckets into order expected for
  // the forward pass.
  weightGradientBuckets =
      getBucketsByPartition(weightGradientBuckets, gradWPlan.partition);
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
