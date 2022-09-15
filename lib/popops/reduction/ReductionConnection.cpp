// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include "ReductionConnection.hpp"

#include "CycleEstimationFunctions.hpp"
#include "ReductionVertex.hpp"

#include "poplibs_support/logging.hpp"
#include <poplar/Type.hpp>
#include <poplibs_support/Compiler.hpp>
#include <popops/Reduce.hpp>
#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>

#include <boost/optional.hpp>
#include <boost/range/algorithm/transform.hpp>

#include <algorithm>
#include <cassert>

using namespace poplibs_support;

namespace popops {

namespace {

// Divide a by b, rounding up.
template <typename T> T udiv(T a, T b) { return ((a + b) - 1) / b; }

// Return the approximate number of operations per cycle for the given
// type and operation. This doesn't account for type conversion, scale or
// update.
double opsPerCycle(unsigned outputSize, const poplar::Target &target,
                   poplar::Type type, popops::Operation operation) {
  if (outputSize == 0)
    return 1;

  // Very roughly, the number of elements we can process per cycle is as
  // follows.
  //
  // In theory we can do better in some cases with complex optimisations.
  //
  // f16 add, square-add:
  //  outputSize     == 1: 8     rpt {ld128step; f16v8acc}, f16v4add, f16v2add,
  //                             f16v2add
  //  outputSize     == 2: 8     rpt {ld128step; f16v8acc}, f16v4add, f16v2add
  //  outputSize     == 4: 8     rpt {ld128step; f16v8acc}, f16v4add
  //  outputSize     == 8: 8     rpt {ld128step; f16v8acc}
  //  outputSize % 8 == 0: 8     rpt rpt {ld128step; f16v8acc}
  //  outputSize % 4 == 0: 4     rpt rpt {ld64step; f16v4acc}
  //  outputSize % 2 == 0: 2     rpt rpt {ld32step; f16v2add}
  //            otherwise: 1     rpt rpt {lds16step; f16v2add}
  //
  // f16 other ops:
  //  outputSize     == 1: 4
  //  outputSize     == 2: 4
  //  outputSize     == 4: 4     rpt {ld64step; f16v4mul}
  //  outputSize % 4 == 0: 4
  //  outputSize % 2 == 0: 2
  //            otherwise: 1
  //
  // f32 add, square-add:
  //  outputSize     == 1: 4
  //  outputSize     == 2: 4
  //  outputSize     == 4: 4
  //  outputSize % 4 == 0: 4
  //  outputSize % 2 == 0: 2
  //            otherwise: 4/3
  //
  // f32 other ops:
  //  outputSize     == 1: 2     rpt {ld64step; f32v2mul}, f32mul
  //  outputSize     == 2: 2     rpt {ld64step; f32v2mul}
  //  outputSize % 2 == 0: 2
  //            otherwise: 4/3 (we have two do two ld32's instead of one ld64
  //            every other cycle).

  // However this only needs to be rough, so I am ignoring the clever
  // optimisations which means this is simple.
  unsigned vectorWidth = target.getVectorWidth(type);

  // Add and square-add can be done twice as fast for floats.
  if ((operation == popops::Operation::ADD ||
       operation == popops::Operation::SQUARE_ADD) &&
      (type == poplar::HALF || type == poplar::FLOAT))
    vectorWidth *= 2;

  // If the output size is 1/2, 1/4 or 1/8th of the vector width we
  // can just treat it as if it were the vector width with a final tiny
  // reduction on the output value.
  for (unsigned w = vectorWidth; w > 0; w /= 2) {
    if (outputSize == w)
      return vectorWidth;
  }

  // Otherwise we can generally process as many numbers as we can load
  // in one cycle, when taking alignment requirements into account.
  for (unsigned w = vectorWidth; w > 0; w /= 2) {
    if (outputSize % w == 0)
      return w;
  }

  // We should never reach here but the answer is probably 1.
  return 1;
}

struct ReductionAssignment {
  // The worker(s) that this reduction has been assigned to. If it is only
  // assigned to one worker it can be done in one stage. If it is assigned to
  // more than one it is done in two stages.
  std::vector<unsigned> workers;
};

std::uint64_t approximateOverheadCyclesForReduction(const unsigned splits) {
  // Simple overhead based on specialisation 3.
  const unsigned vertexOverhead = 30;
  return splits == 1 ? vertexOverhead : 2 * vertexOverhead;
}

std::uint64_t approximateCyclesForReduction(const poplar::Target &target,
                                            popops::Operation operation,
                                            const RegionReduction &reduction) {
  unsigned totalPartialElements = reduction.getNumPartialsElements();
  double cyclesPerOp =
      1.0 / opsPerCycle(reduction.output.numElements(), target,
                        reduction.output.elementType(), operation);

  // The ceil() is because otherwise some cycle counts come out as 0.5 which is
  // obviously impossible in practice, and it gets rounded to 0 which is even
  // more impossible. This is only rough and there's going to be some overhead
  // which I haven't estimated here anyway.
  return std::ceil(totalPartialElements * cyclesPerOp);
}

// Distribute a set of region reductions between workers on a tile. This never
// splits the individual reductions.
//
// The return value is the worker each reduction is assigned to.
std::vector<unsigned> distributeReductionsBetweenWorkers(
    const poplar::Target &target, popops::Operation operation,
    const std::vector<RegionReduction> &reductions,
    const unsigned remainingWorkers) {

  // If there are more output regions than numWorkers we need to
  // distrubute them between workers.  We can choose to either:
  // a) Distribute them as evenly as possible based on the number of partials
  //    so each vertex has a similar number of partials.  This has the benefit
  //    of balancing the load per worker.
  // b) Try to keep reductions that are input together assigned to the same
  //    worker, which sometimes allows that worker to use a more efficent vertex
  //
  // If there is very little work sharing benefit to be had from approach a) we
  // choose method b).
  // Note that method a) will tend to assign reductions with equal work cost to
  // workers in a round-robin manner which works against combining those
  // reductions to be implemented in a single efficient vertex.

  std::vector<unsigned> assignments;

  if (reductions.empty())
    return assignments;

  assignments.resize(reductions.size());

  // Calculate (very roughly) the number of cycles needed to calculate
  // each reduction. This doesn't have to be exact.
  //
  // The pair is {reductionNumber, cycleCount}.
  std::vector<std::pair<unsigned, std::uint64_t>> reductionCycles;
  reductionCycles.reserve(reductions.size());

  double meanCycles = 0;
  for (const auto &r : reductions) {
    reductionCycles.emplace_back(
        reductionCycles.size(),
        approximateCyclesForReduction(target, operation, r));
    meanCycles += reductionCycles.back().second;
  }

  meanCycles = meanCycles / reductionCycles.size();
  // Check if the cycle counts are all approximately equal.  Define a range
  // based on the heuristic of mean +/- 8 cycles OR +/- 10% whichever is the
  // greater
  const auto toleratedRange = std::max(8.0, 0.1 * meanCycles);
  std::uint64_t minCycles = static_cast<std::uint64_t>(
      meanCycles > toleratedRange ? meanCycles - toleratedRange : 0);
  std::uint64_t maxCycles =
      static_cast<std::uint64_t>(meanCycles + toleratedRange);
  bool equalCycles = true;
  for (const auto &r : reductionCycles) {
    if (r.second < minCycles || r.second > maxCycles) {
      equalCycles = false;
      break;
    }
  }
  if (equalCycles && remainingWorkers != 0) {
    // Assigning work to each worker in a round-robin manner (as below) will
    // often split outputs and / or partials which will constrain our
    // choice of vertex so don't do that if there is no work sharing benefit.
    // Instead assign consecutive reductions to each, eg, with 14 reductions
    // assign to worker 0,0,0,1,1,1,2,2,3,3,4,4,5,5
    // Whereas the round robin method would result in:
    // assign to worker: 0,1,2,3,4,5,0,1,2,3,4,5,0,1
    // (In the second case no adjacent/consecutive reductions are assigned to
    // the same worker so they are unlikely to be combined later)
    unsigned reductionsPerWorker = reductions.size() / remainingWorkers;
    unsigned remainder = reductions.size() % remainingWorkers;
    unsigned index = 0;
    for (unsigned i = 0; i < remainingWorkers && index < reductions.size();
         i++) {
      for (unsigned j = 0; j < reductionsPerWorker + (i < remainder); j++) {
        assignments[index++] = i;
      }
    }
    return assignments;
  }

  // Optimisation: There is definitely scope for further optimisation here.
  // Distributing work evenly is almost NP-complete, but there are apparently
  // efficient algorithms. See https://en.wikipedia.org/wiki/Partition_problem
  // and especially Multi-Way Number Partitioning by Richard Korf:
  // http://www.ijcai.org/Proceedings/09/Papers/096.pdf

  // For now, we just sort them from most cycles to least, and greedily
  // assign them to the vertex with the fewest cycles.

  // Use stable_sort because:
  // Order of items with equal cycles matters for repeatability. Especially the
  // case where it affects reductions that get assigned to a specific worker by
  // the process below and can then get combined into a single vertex if
  // suitable.
  std::stable_sort(
      reductionCycles.begin(), reductionCycles.end(),
      [](std::pair<unsigned, std::uint64_t> a,
         std::pair<unsigned, std::uint64_t> b) { return a.second < b.second; });

  std::vector<std::uint64_t> cyclesPerWorker;
  cyclesPerWorker.reserve(target.getNumWorkerContexts());

  for (auto c : reductionCycles) {
    auto reductionIndex = c.first;
    auto cycles = c.second;

    if (cyclesPerWorker.size() < remainingWorkers) {

      // If we still have spare worker contexts, add another worker.
      assignments[reductionIndex] = cyclesPerWorker.size();
      cyclesPerWorker.push_back(cycles);

    } else {

      // Otherwise add it to the one with the fewest worker cycles so far.
      auto minIt =
          std::min_element(cyclesPerWorker.begin(), cyclesPerWorker.end());
      assignments[reductionIndex] = minIt - cyclesPerWorker.begin();
      *minIt += cycles;
    }
  }
  return assignments;
}

// Get the number of rows, or the reduction factor of a region reduction.
std::size_t reductionFactor(const RegionReduction &reduction) {
  auto outputSize = reduction.output.numElements();

  // Work out the number of rows (reduction ratio) of this reduction.
  std::size_t totalPartialElements = reduction.getNumPartialsElements();
  assert(totalPartialElements % outputSize == 0);

  // Total number of rows for this reduction.
  return totalPartialElements / outputSize;
}

// Given a reduction with a reduction factor of `row` that takes roughly
// `cycles` cycles, what is the highest number of pieces we should split it
// into. We want each piece to have at least 2 rows, and at least
// `minCycles` cycles.
unsigned getMaxSplit(std::size_t rows, std::uint64_t cycles) {

  // Don't split it up so that any piece takes fewer cycles than this.
  // This number was found empirically by trying different numbers on
  // reductions from resnet until it gave the fewest total cycles.
  const std::uint64_t minCycles = 4;

  // Or there are fewer than 2 rows in each piece.
  return std::min(static_cast<std::uint64_t>(rows / 2), cycles / minCycles);
}

// Split the RegionReductions into smaller chunks. This can be for two reasons:
//  - to make more work to utilise all of the workers available
//  - to make the vertex state values fit into their respective types.
// The return value is the number of pieces each reduction is split up into
// (minimum 1) for each reduction.
std::vector<unsigned> splitTwoStageReductionsBetweenWorkers(
    const poplar::Target &target, popops::Operation operation,
    const std::vector<RegionReduction> &reductions,
    const std::size_t vectorListMaxSize, const unsigned remainingWorkers) {
  // initialise the number of splits needed for each reduction by how many times
  // it would be needed for it to fit into a DeltaN VectorList.
  std::vector<unsigned> minimumSplits;
  minimumSplits.reserve(reductions.size());

  const auto out = std::back_inserter(minimumSplits);
  boost::transform(reductions, out, [&](const RegionReduction &reduction) {
    return udiv(reduction.getNumPartials(), vectorListMaxSize);
  });

  std::vector<std::uint64_t> approxCycleCounts(reductions.size());
  for (std::size_t i = 0; i < reductions.size(); ++i) {
    approxCycleCounts[i] =
        approximateOverheadCyclesForReduction(1) +
        approximateCyclesForReduction(target, operation, reductions[i]);
  }

  std::vector<unsigned> splits(minimumSplits.begin(), minimumSplits.end());

  // First work out the maximum number of splits for each worker. It never
  // makes sense to split to less than 2 rows for each piece, and in some
  // cases the limit is more if there aren't many output values.

  std::vector<unsigned> maxSplit(reductions.size(), 1);
  for (unsigned i = 0; i < reductions.size(); ++i) {
    maxSplit[i] =
        getMaxSplit(reductionFactor(reductions[i]), approxCycleCounts[i]);
  }

  unsigned freeWorkers = remainingWorkers > reductions.size()
                             ? remainingWorkers - reductions.size()
                             : 0;
  // As a baseline to check if an improvement in max cycles per reduction
  // is made by splitting work, record the max cycles prior to splitting.
  const auto maxOneStageCycles =
      *std::max_element(approxCycleCounts.begin(), approxCycleCounts.end());

  while (freeWorkers > 0) {
    boost::optional<unsigned> toSplit;
    // Consider splitting the slowest reduction
    const auto highestCyclesIter =
        std::max_element(approxCycleCounts.begin(), approxCycleCounts.end());
    const auto i = std::distance(approxCycleCounts.begin(), highestCyclesIter);

    // Stop if it doesn't want to be split any more, as we can't improve the
    // maximum
    if (splits[i] + 1 >= maxSplit[i])
      break;

    // Work out the rough number of cycles if it would be split more, accounting
    // for the overhead of needing another stage.
    // TODO: T12961 Use the cycle estimation functions to account for cycles
    // accurately throughout this module, given the decision on which reduction
    // vertex is actually going to be called.
    auto cyclesAfterSplit =
        approximateOverheadCyclesForReduction(splits[i] + 1) +
        approximateCyclesForReduction(target, operation, reductions[i]) /
            (splits[i] + 1);
    if (cyclesAfterSplit < approxCycleCounts[i]) {
      approxCycleCounts[i] = cyclesAfterSplit;
      toSplit = i;
    }

    // Check if we don't want to split any more.
    if (!toSplit)
      break;

    ++splits[toSplit.get()];
    --freeWorkers;
  }
  auto highestCyclesAfterSplits =
      *std::max_element(approxCycleCounts.begin(), approxCycleCounts.end());

  // Is the complexity of splitting to use all workers actually improving
  // the overall maximum?
  // If not then return the minimum possible split, avoiding extra complexity
  // and vertex state for no good reason
  return highestCyclesAfterSplits == maxOneStageCycles ? minimumSplits : splits;
}
// Return a vector containing the partials for all of the passed reductions.
// In the case of a reduction having RegularPartials this will be just the
// slices that are to be reduced
std::vector<poplar::Tensor>
extractPartials(const RegionReductionRange reductions) {
  std::vector<poplar::Tensor> result;
  for (auto &red : reductions) {
    if (red.regularPartials()) {
      bool partialsGroupContiguous =
          red.getStride() == red.output.numElements();
      for (unsigned i = 0; i < red.getNumOuterStrides(); i++) {
        if (partialsGroupContiguous) {
          const auto base = red.getOffset() + (i * red.getOuterStride());
          result.emplace_back(red.getPartials()[0].slice(
              base, base + red.output.numElements() * red.innerFactor *
                               red.outerFactor));
        } else {
          for (unsigned j = 0; j < red.outerFactor; j++) {
            const auto base = red.getOffset() + (j * red.getStride()) +
                              (i * red.getOuterStride());
            result.emplace_back(red.getPartials()[0].slice(
                base, base + red.output.numElements() * red.innerFactor));
          }
        }
      }
    } else {
      for (const auto &partial : red.getPartials()) {
        result.emplace_back(partial);
      }
    }
  }
  return result;
}

// Return a single tensor with all partials in the given range flattened
// and concatenated.
poplar::Tensor extractFlatPartials(const RegionReductionRange reductions) {
  std::vector<poplar::Tensor> toConcat;
  for (auto &red : reductions) {
    if (red.regularPartials()) {
      const auto partials = red.getPartials()[0];
      for (unsigned i = 0; i < red.getNumOuterStrides(); ++i) {
        const auto baseOffset = red.getOffset() + i * red.getOuterStride();
        // innerElems is the number of contiguous elements we reference
        // between each stride.
        const auto innerElems = red.output.numElements() * red.innerFactor;
        // We form a 2-dimensional shape from the stride and outerFactor
        // and slice that to get all the partials. Because the partials
        // tensor may be of length < outerFactor * stride we actually
        // take 1 slice for (outerFactor - 1) of the contiguous regions
        // and 1 slice for the last contiguous region.
        const auto sizeOfFirstRegionToSlice =
            (red.outerFactor - 1) * red.getStride();
        if (sizeOfFirstRegionToSlice > 0) {
          const auto regionToSlice =
              partials.slice(baseOffset, baseOffset + sizeOfFirstRegionToSlice)
                  .reshape({red.outerFactor - 1, red.getStride()});
          toConcat.emplace_back(
              regionToSlice.slice(0, innerElems, 1).flatten());
        }
        toConcat.emplace_back(
            partials.slice(baseOffset + sizeOfFirstRegionToSlice,
                           baseOffset + sizeOfFirstRegionToSlice + innerElems));
      }
    } else {
      for (const auto &partial : red.getPartials()) {
        toConcat.emplace_back(partial);
      }
    }
  }
  return concat(toConcat);
}

poplar::Tensor flattenAndCheckPartials(const poplar::Graph &graph,
                                       const RegionReductionRange reductions) {
  // Check that the partials can be received via exchange without
  // requiring a rearrangement due to misalignment. If this is not the
  // case we should be using the generic vertex
  if (!reductions[0].regularPartials()) {
    const auto &target = graph.getTarget();
    const auto atomicWriteSize = target.getAtomicStoreGranularity();
    unsigned numBytes = 0;
    for (const auto &p : reductions[0].getPartials()) {
      if (numBytes % atomicWriteSize) {
        throw poputil::poplibs_error(
            "Generating a singleIO reduction vertex with misaligned partials");
      }
      numBytes += p.numElements() * target.getTypeSize(p.elementType());
    }
  }
  return extractFlatPartials({reductions.begin(), reductions.begin() + 1});
}

template <typename CountsAndStridesType>
static void createStridedReduceVertex(
    poplar::Graph &graph, const RegionReductionRange reductions,
    const bool targetIsCpu, const poplar::VertexRef &vertex,
    const ReductionSpecialisation specialisation, unsigned tile,
    const poplar::DebugNameAndId &dnai) {

  const auto &r0 = reductions[0];
  const auto &target = graph.getTarget();
  const auto partialsType = r0.getPartials()[0].elementType();
  const auto partialsElemWidth = (targetIsCpu && partialsType == poplar::HALF)
                                     ? 2
                                     : target.getTypeSize(partialsType);
  const auto partialsGrainSize = partialsType == poplar::HALF ? 4 : 2;
  unsigned numPartials, partialsWidth, vertexOuterStride, numOuterStridesM1;
  unsigned numOutputs;

  if (r0.regularPartials()) {
    // Partials information is regular and we may be able to use the stride
    std::vector<poplar::Tensor> outputs;
    outputs.reserve(reductions.size());
    for (const auto &region : reductions) {
      outputs.push_back(region.output);
    }
    const auto output = concat(outputs);
    numOutputs = output.numElements();
    graph.connect(vertex["out"], output);

    const auto strideIsCompatible =
        (r0.getStride() * partialsElemWidth) % 8 == 0;
    const auto outerStrideIsCompatible =
        (r0.getOuterStride() * partialsElemWidth) % 8 == 0;
    if (strideIsCompatible && outerStrideIsCompatible) {
      // We can use the stride to connect to a larger region without copies
      const auto oneSliceSize =
          output.numElements() + r0.getStride() * (r0.outerFactor - 1);
      numOuterStridesM1 = r0.getNumOuterStrides() - 1;

      const auto sliceEnd = r0.getOffset() + oneSliceSize +
                            r0.getOuterStride() * numOuterStridesM1;
      graph.connect(vertex["partials"],
                    r0.getPartials()[0].slice(r0.getOffset(), sliceEnd));
      partialsWidth = r0.getStride();
      // Vertex uses an outer stride that is the stride to the next partial
      // having iterated over the previous group of partials
      vertexOuterStride = r0.getOuterStride() == 0
                              ? 0
                              : (r0.getOuterStride() -
                                 (r0.getStride() * (r0.outerFactor - 1))) /
                                    partialsGrainSize;
      numPartials = r0.outerFactor;

      logging::popops::trace("  Offset: {} PartialsWidth: {} "
                             "NumOuterStridesM1: {} OuterStride: {}",
                             r0.getOffset(), partialsWidth, numOuterStridesM1,
                             r0.getOuterStride());
    } else {
      // The size of the stride is not useful so slice and introduce copies.
      graph.connect(
          vertex["partials"],
          extractFlatPartials({reductions.begin(), reductions.begin() + 1}));
      partialsWidth = r0.output.numElements();
      numOuterStridesM1 = 0;
      vertexOuterStride = 0;
      numPartials = r0.outerFactor * r0.getNumOuterStrides();
    }
  } else {
    const auto allPartials =
        extractFlatPartials({reductions.begin(), reductions.begin() + 1});
    numPartials = allPartials.numElements() / r0.output.numElements();
    partialsWidth = r0.output.numElements();
    graph.connect(vertex["partials"], allPartials);

    numOutputs = r0.output.numElements();
    graph.connect(vertex["out"], r0.output);
    // The outer stride is not applicable in this case
    numOuterStridesM1 = 0;
    vertexOuterStride = 0;
  }
  // Check within range
  if (numPartials > std::numeric_limits<unsigned short>::max() &&
      !targetIsCpu) {
    throw poputil::poplibs_error("Number of partials larger than short");
  }
  if (vertexOuterStride > std::numeric_limits<unsigned short>::max() &&
      !targetIsCpu) {
    throw poputil::poplibs_error("Outer stride larger than short");
  }

  // Process these parameters to reflect what the vertex needs, so it doesn't
  // need to be done at runtime
  if (partialsWidth % partialsGrainSize) {
    throw poputil::poplibs_error("Partials width (" +
                                 std::to_string(partialsWidth) +
                                 ") must be a multiple of grainsize (" +
                                 std::to_string(partialsGrainSize) + ")");
  }

  if (numOutputs % partialsGrainSize) {
    throw poputil::poplibs_error("Output width (" + std::to_string(numOutputs) +
                                 ") must be a multiple of grainsize (" +
                                 std::to_string(partialsGrainSize) + ")");
  }
  partialsWidth = partialsWidth / partialsGrainSize;
  numOutputs = numOutputs / partialsGrainSize - 1;

  if (partialsWidth > std::numeric_limits<unsigned short>::max() &&
      !targetIsCpu) {
    throw poputil::poplibs_error("Partials width larger than short");
  }
  if (specialisation == ReductionSpecialisation::STRIDED_REDUCE &&
      numOuterStridesM1) {
    throw poputil::poplibs_error(
        "Logic error in selecting STRIDED_REDUCE vertex");
  }
  CountsAndStrides<CountsAndStridesType> cAndS;
  cAndS.numOutputsM1 = numOutputs;
  cAndS.numPartialsM1 = numPartials - 1;
  cAndS.partialsWidth = partialsWidth;
  cAndS.numOuterStridesM1 = numOuterStridesM1;
  cAndS.outerStride = vertexOuterStride;

  const auto cAndSVector = countsAndStridesAsVector(cAndS);
  auto countsAndStrides = graph.addConstant<CountsAndStridesType>(
      poplar::equivalent_device_type<CountsAndStridesType>().value,
      {cAndSVector.size()}, cAndSVector, {dnai, "countsAndStrides"});

  graph.setTileMapping(countsAndStrides, tile);
  graph.connect(vertex["countsAndStrides"], countsAndStrides);
}

static void createSingleOutputVertex(
    poplar::Graph &graph, const RegionReductionRange reductions,
    const bool targetIsCpu, const poplar::VertexRef &vertex,
    const ReductionSpecialisation specialisation) {

  const auto &r0 = reductions[0];
  const auto allPartials = flattenAndCheckPartials(graph, reductions);

  graph.connect(vertex["partials"], allPartials);

  const auto numPartials = allPartials.numElements();

  if (numPartials > std::numeric_limits<unsigned short>::max() &&
      !targetIsCpu) {
    throw poputil::poplibs_error("Number of partials larger than short");
  }
  graph.setInitialValue(vertex["numPartials"], numPartials);
  graph.connect(vertex["out"], r0.output);
}

static void createReductionVertex(poplar::Graph &graph,
                                  const unsigned numOutputRegions,
                                  const RegionReductionRange reductions,
                                  const poplar::DebugNameAndId &dnai,
                                  const poplar::VertexRef vertex,
                                  const bool targetIsCpu) {
  // Work out the total number of partial regions.
  unsigned numPartialRegions = 0;

  // Number of input partial regions for each output region, start with 0.
  std::vector<unsigned short> numPartials;
  numPartials.reserve(numOutputRegions);

  for (const auto &r : reductions) {
    auto sz = r.getNumPartials();
    if (sz < 1) {
      throw poputil::poplibs_error("output region with no partials");
    }
    if (sz > std::numeric_limits<unsigned short>::max() && !targetIsCpu) {
      // As total memory on Colossus B0 is 2**18, 2**16 * num_workers
      // assuming that the work is split across workers
      // would occupy more memory than we have. If work is not split across
      // workers, then if partials[i].size() < 4 for all reductions
      // could hit this limit.
      // In future we may have to deal with num partials greater than this
      // and create more vertices
      throw poputil::poplibs_error("Number of partials larger than short");
    }
    numPartialRegions += sz;
    numPartials.push_back(static_cast<unsigned short>(sz));
  }

  auto t = graph.addConstant(poplar::UNSIGNED_SHORT, {numPartials.size()},
                             numPartials.data(), {dnai, "numPartials"});
  graph.setTileMapping(t, 0);
  graph.connect(vertex["numPartials"], t);

  std::vector<poplar::Tensor> partials = extractPartials(reductions);
  partials.reserve(numPartialRegions);
  std::vector<poplar::Tensor> outputs;
  outputs.reserve(numOutputRegions);

  for (const auto &r : reductions) {
    outputs.emplace_back(r.output);
  }
  graph.connect(vertex["out"], outputs);
  graph.connect(vertex["partials"], partials);
}

static void
createContinuousReductionVertex(poplar::Graph &graph,
                                const RegionReductionRange reductions,
                                const poplar::VertexRef vertex) {
  const unsigned numOutputs = reductions.size();
  assert(numOutputs != 0);
  graph.setInitialValue(vertex["numOutputsM1"], numOutputs - 1);
  graph.setInitialValue(vertex["numPartials"],
                        reductions.front().getNumPartialsElements());

  std::vector<poplar::Tensor> outputs;
  outputs.reserve(reductions.size());
  for (const auto &red : reductions) {
    outputs.push_back(red.output);
  }
  auto singlePartials = extractFlatPartials(reductions);
  auto singleOutput = concat(outputs);
  graph.connect(vertex["partials"], singlePartials.flatten());
  graph.connect(vertex["out"], singleOutput.flatten());
}
// TODO: T18040 - Either enter correct cycle costs here, or properly
// account for cycles consumed given the number of partials and the choice of
// vertex.

// Simple costs based on vertex inner loop cycles per vector
// just using the defaultBase cost for now to weight decisions against the
// inefficient default case.
constexpr unsigned specialDefaultBaseCycleCost = 2;
constexpr unsigned specialSingleInputBaseCycleCost = 0;
constexpr unsigned specialStridedBaseCycleCost = 0;
constexpr unsigned specialContinuousBaseCycleCost = 0;

// Simple costs based on vertex state sizes in bytes.
constexpr unsigned specialDefaultBaseMemoryCost = 20;
constexpr unsigned specialDefaultPerReductionMemoryCost = 8;

constexpr unsigned specialSingleInputBaseMemoryCost = 12;
// For strided reduce, use the directly included vertex state.  Typically
// the "countsAndStrides" field will be shared at no/very little extra cost.
constexpr unsigned specialStridedBaseMemoryCost = 12;
constexpr unsigned specialContinuousBaseMemoryCost = 12;

unsigned getReductionCost(ReductionSpecialisation specialisation,
                          const RegionReductionRange regions) {
  switch (specialisation) {
  case ReductionSpecialisation::DEFAULT:
    return specialDefaultPerReductionMemoryCost * regions.size() +
           specialDefaultBaseMemoryCost + specialDefaultBaseCycleCost;

  case ReductionSpecialisation::SCALAR_OUTPUT_REGIONS:
    return specialDefaultPerReductionMemoryCost * regions.size() +
           specialDefaultBaseMemoryCost + specialDefaultBaseCycleCost;

  case ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT:
    return specialSingleInputBaseMemoryCost + specialSingleInputBaseCycleCost;

  case ReductionSpecialisation::STRIDED_REDUCE:
    return specialStridedBaseMemoryCost + specialStridedBaseCycleCost;

  case ReductionSpecialisation::ALL_REGIONS_CONTINUOUS:
    return specialContinuousBaseMemoryCost + specialContinuousBaseCycleCost;

  default:
    throw poputil::poplibs_error("Cannot find cost of undefined reduction "
                                 "specialisation");
  };
}
// Find the number of reductions that a reduce vertex of a given type should
// consume for the cost to always be better than an implementation using a
// single expensive vertex.

unsigned findMinReductionsForSpecialisation(ReductionSpecialisation special) {
  if (special == ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
    return specialContinuousBaseMemoryCost /
           specialDefaultPerReductionMemoryCost;
  }
  if (special == ReductionSpecialisation::STRIDED_REDUCE) {
    return specialStridedBaseMemoryCost / specialDefaultPerReductionMemoryCost;
  }
  // Prevent combining reductions using other vertex types
  return UINT_MAX;
}
// Consider the reductions in order and try to split them into a sequence that
// can be deal with by more efficient vertices.  At present continuous reduce
// can deal with multiple reductions, and the other efficient vertex types can
// deal with just one. So we can end up with a sequence where a vertex deals
// with several reductions or just one.
// Eg:
// reduction [0,1) : specialisation 2
// reduction [1,3) : continuousReduce (2 reductions consumed here: lower cost)
// reduction [3,4) : specialisation 3
//
// There are 2 reasons to do this - firstly to extract a group of reductions
// which should always be implemented together by continuous reduce.
// Secondly to consider use of all specialisations to implement the reductions
// with the least cost

enum class ThresholdType { LENGTH_THRESHOLD, COST_THRESHOLD };
// Structure contining the reductions that we are working on.
// RegionReductionsAndRanges.plan can reference some RegionReductions from
// RegionReductionsAndRanges.allReductions and some from
// RegionReductionsAndRanges.remainingReductions, so they are kept together.
struct RegionReductionsAndRanges {
  std::vector<RegionReduction> allReductions;
  std::vector<RegionReduction> remainingReductions;
  std::vector<RegionReductionRange> plan;
  RegionReductionRange range;
};

static bool
specialisationHasZeroPerReductionCost(ReductionSpecialisation specialisation) {
  // Reduction specialisations that can consume multiple reductions with no
  // increase in vertex state size.
  return specialisation == ReductionSpecialisation::ALL_REGIONS_CONTINUOUS ||
         specialisation == ReductionSpecialisation::STRIDED_REDUCE;
}

// Consume reductions from reductionsAndRanges.range, adding them to
// the plan in reductionsAndRanges.plan
void findMultiVertexPlan(const poplar::Graph &graph, const ReduceParams &params,
                         RegionReductionsAndRanges &reductionsAndRanges,
                         poplar::Type partialType, bool reductionUsesInput,
                         ThresholdType thresholdType,
                         boost::optional<unsigned> threshold) {
  if (thresholdType == ThresholdType::LENGTH_THRESHOLD &&
      threshold != boost::none) {
    throw poputil::poplibs_error("Reduction multiVertexPlan will find its own "
                                 " length threshold, don't provide one");
  }
  unsigned sumOfMultiVertexCosts = 0;
  const auto startSize = reductionsAndRanges.plan.size();
  auto reductions = reductionsAndRanges.range;

  for (unsigned i = 0; i < reductions.size(); i++) {
    // Create a range start point and specialisation
    const auto rangeStart = reductions.begin() + i;
    auto specialisation = getReductionVertexSpecialisation(
        graph, params, {rangeStart, rangeStart + 1}, partialType,
        reductionUsesInput);

    auto rangeEnd = reductions.end();
    // Try to consume more reductions into that range, if an efficient vertex
    // will still be used.
    for (unsigned j = i + 1; j < reductions.size(); j++) {
      const auto candidateSpecialisation = getReductionVertexSpecialisation(
          graph, params, {rangeStart, reductions.begin() + j + 1}, partialType,
          reductionUsesInput);
      if (specialisationHasZeroPerReductionCost(candidateSpecialisation)) {
        // The range can be extended, so advance i to reflect that and note the
        // specialisation
        i = j;
        specialisation = candidateSpecialisation;
      } else {
        // End of range
        rangeEnd = reductions.begin() + j;
        break;
      }
    }
    if (thresholdType == ThresholdType::COST_THRESHOLD) {
      // Add the range to the plan, evaluate cost and exit if too great
      reductionsAndRanges.plan.push_back({rangeStart, rangeEnd});

      sumOfMultiVertexCosts +=
          getReductionCost(specialisation, reductionsAndRanges.plan.back());

      if (sumOfMultiVertexCosts > threshold) {
        // Remove what we just made and refer to the whole vector of
        // reductions in the plan as that has a lower cost
        reductionsAndRanges.plan.resize(startSize);
        reductionsAndRanges.plan.push_back(reductions);
        return;
      }
    }
    if (thresholdType == ThresholdType::LENGTH_THRESHOLD) {
      // Reference those with a given length, otherwise gather the remaining
      // ones for the second analysis step
      const auto threshold = findMinReductionsForSpecialisation(specialisation);
      if (static_cast<unsigned>(rangeEnd - rangeStart) >= threshold) {
        reductionsAndRanges.plan.push_back({rangeStart, rangeEnd});
      } else {
        for (unsigned j = rangeStart - reductions.begin();
             j < rangeEnd - reductions.begin(); j++) {
          reductionsAndRanges.remainingReductions.push_back(reductions[j]);
        }
      }
    }
  }
}

// Find a plan for implementing the reductions - choosing vertices so that the
// cost of implementation is minimised
void findVertexPlan(const poplar::Graph &graph, const ReduceParams &params,
                    RegionReductionsAndRanges &reductionsAndRanges,
                    poplar::Type partialType, bool reductionUsesInput) {

  // Potentially we could just do the work in one vertex, this could be the
  // best solution and provides a quick exit.
  auto specialisation = getReductionVertexSpecialisation(
      graph, params, reductionsAndRanges.allReductions, partialType,
      reductionUsesInput);

  if (specialisation != ReductionSpecialisation::DEFAULT &&
      specialisation != ReductionSpecialisation::SCALAR_OUTPUT_REGIONS) {
    reductionsAndRanges.plan.push_back(reductionsAndRanges.allReductions);
    return;
  }
  // Try to improve on the single vertex plan:
  // Note the reductions that will always produce a cost improvement
  // if consumed by continuous reduce regardless of the other vertices used.
  // Eg reductions:  a b c d e f g h i j k l
  // These:            ^ ^ ^ ^   ^ ^ ^ ^    can be consumed by continuous reduce
  // So we update reductionsAndRanges.plan with references to {bcde} and {ghij}
  // and reductionsAndRanges.remainingReductions with {afkl}
  reductionsAndRanges.range = reductionsAndRanges.allReductions;
  findMultiVertexPlan(graph, params, reductionsAndRanges, partialType,
                      reductionUsesInput, ThresholdType::LENGTH_THRESHOLD,
                      boost::none);

  // If there were any left, they will be in remainingReductions,
  // process them too
  if (reductionsAndRanges.remainingReductions.size() != 0) {

    // We aim to either deal with all remaining reductions in one vertex, or use
    // several if that would have a lower cost (based on the size of
    // the vertex state), so find the basline cost of doing it all in one vertex
    // Continuing the example above:
    // We deal with the remaining reductions {afkl}.  They can be dealt with in
    // one vertex, with a cost.  But if they can be deal with using one
    // vertex each, or {a},{f},{kl}, Ie k,l in one vertex and this has a lower
    // cost we will do that.
    auto specialisation = getReductionVertexSpecialisation(
        graph, params, reductionsAndRanges.remainingReductions, partialType,
        reductionUsesInput);
    const unsigned wholeCost = getReductionCost(
        specialisation, reductionsAndRanges.remainingReductions);

    // Add multiple separate region reduction ranges into the plan, or just one
    // range if that would have the lower cost.
    reductionsAndRanges.range = reductionsAndRanges.remainingReductions;
    findMultiVertexPlan(graph, params, reductionsAndRanges, partialType,
                        reductionUsesInput, ThresholdType::COST_THRESHOLD,
                        wholeCost);
  }
  return;
}

// Create the reduction vertex most suited to the data.
void createVertex(poplar::Graph &graph,
                  const std::vector<RegionReduction> &reductions,
                  const ReduceParams &params, const poplar::Type &partialType,
                  const poplar::Type &outputType, const poplar::ComputeSet &cs,
                  const unsigned tile, bool reductionUsesInput,
                  const poplar::DebugNameAndId &dnai) {

  const bool targetIsCpu =
      graph.getTarget().getTargetType() == poplar::TargetType::CPU;
  const bool targetIsIpu =
      graph.getTarget().getTargetType() == poplar::TargetType::IPU;
  // Number of output regions for this vertex.
  auto numOutputRegions = reductions.size();

  if (numOutputRegions < 1)
    throw poputil::poplibs_error("no output regions in reduction");

  // Check the number of partials in each output region
  for (const auto &r : reductions) {
    auto sz = r.getPartials().size();
    if (sz < 1) {
      throw poputil::poplibs_error("output region with no partials");
    }
  }
  // logging begin
  if (logging::popops::shouldLog(logging::Level::Trace)) {
    for (unsigned i = 0; i < reductions.size(); i++) {
      const auto partials =
          extractPartials({reductions.begin() + i, reductions.begin() + i + 1});
      logging::popops::trace(
          "Reduction output size = {}, input vector size = {}",
          reductions[i].output.numElements(), partials.size());
      unsigned size = 0;
      unsigned count = 0;
      for (const auto &p : partials) {
        if (count == 0 || size == p.numElements()) {
          count++;
          size = p.numElements();
        } else {
          logging::popops::trace("    Partials: {} with size: {}", count, size);
          count = 1;
          size = p.numElements();
        }
      }
      if (count != 0) {
        logging::popops::trace("    Partials: {} with size: {}", count, size);
      }
    }
  }
  // Logging end
  auto qualifyVertexSpecialisation =
      [&](const RegionReductionRange &range,
          ReductionSpecialisation specialisation) {
        // Throughout the decision making process the 2 strided reduce
        // specialisations can be considered identical.  At the point of vertex
        // creation however we need to qualify which to use
        if (specialisation == ReductionSpecialisation::STRIDED_REDUCE) {
          if (range[0].regularPartials()) {
            const auto partialsType = range[0].getPartials()[0].elementType();
            const auto partialsElemWidth =
                (targetIsCpu && partialsType == poplar::HALF)
                    ? 2
                    : graph.getTarget().getTypeSize(partialsType);
            const auto strideIsCompatible =
                (range[0].getStride() * partialsElemWidth) % 8 == 0;
            const auto outerStrideIsCompatible =
                (range[0].getOuterStride() * partialsElemWidth) % 8 == 0;
            if (strideIsCompatible && outerStrideIsCompatible &&
                range[0].getNumOuterStrides() > 1) {
              specialisation = ReductionSpecialisation::STRIDED_REDUCE_OUTER;
            }
          }
        }
        return specialisation;
      };

  // The vector of RegionReductions can be implemented by 1 or many vertices.
  // Sometimes, using many will have a lower cost than using one, if those
  // targeted are more efficient than the single one.  The result is a
  // vector of references to each group of reductions to deal with separately.
  // This is held in reductionsAndRanges.plan. reductionsAndRanges also holds
  // the vectors of region reductions which it references.
  RegionReductionsAndRanges reductionsAndRanges;
  for (auto &red : reductions) {
    reductionsAndRanges.allReductions.emplace_back(std::move(red));
  }
  findVertexPlan(graph, params, reductionsAndRanges, partialType,
                 reductionUsesInput);

  for (auto &range : reductionsAndRanges.plan) {
    const auto specialisation = qualifyVertexSpecialisation(
        range, getReductionVertexSpecialisation(
                   graph, params, range, partialType, reductionUsesInput));

    const auto name = getReductionVertexName(params, partialType, outputType,
                                             specialisation, params.useScale);
    logging::popops::trace("{}", name);
    const auto vertex = graph.addVertex(cs, name);
    graph.setTileMapping(vertex, tile);

    if (reductionSupportsScaling(specialisation) && params.useScale) {
      graph.connect(vertex["k"], params.scale.reshape({1}));
    }
    if (specialisation == ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT) {
      createSingleOutputVertex(graph, range, targetIsCpu, vertex,
                               specialisation);
    } else if (specialisation == ReductionSpecialisation::STRIDED_REDUCE ||
               specialisation ==
                   ReductionSpecialisation::STRIDED_REDUCE_OUTER) {
      if (targetIsIpu) {
        createStridedReduceVertex<unsigned short>(
            graph, range, targetIsCpu, vertex, specialisation, tile, {dnai});
      } else {
        createStridedReduceVertex<unsigned>(graph, range, targetIsCpu, vertex,
                                            specialisation, tile, {dnai});
      }

    } else if (specialisation ==
               ReductionSpecialisation::ALL_REGIONS_CONTINUOUS) {
      createContinuousReductionVertex(graph, range, vertex);
    } else {

      createReductionVertex(graph, numOutputRegions, range, {dnai}, vertex,
                            targetIsCpu);
    }
  }
}

// Split `rows` into up to N groups with a minimum of 2 rows per group.
// If possible the number of groups is miminised without increasing the
// maximum number of rows in each group. For example with rows=9, N=4
// the output is {3,3,3} rather than {3,2,2,2}.
std::vector<unsigned> splitRowsToWorkers(unsigned rows, unsigned N,
                                         unsigned grainSize) {
  if (rows <= 3 || N <= 1)
    return {rows};
  auto maxRowsPerWorker = udiv(rows, N);

  auto maxWorkers = rows / 2;
  auto numWorkers = std::min(maxWorkers, udiv(rows, maxRowsPerWorker));

  std::vector<unsigned> split(numWorkers,
                              grainSize * (rows / (numWorkers * grainSize)));
  auto plusOnes = rows % (numWorkers * grainSize);
  for (unsigned i = 0; i < plusOnes / grainSize; ++i) {
    split[i] += grainSize;
  }
  // Add the remaining non-grainSize pieces to the last reduction.
  split.back() += plusOnes % grainSize;
  return split;
}

// Every tensor in `partials` is a 1D tensor with a length that is a multiple
// of `outputSize`. Imagine the partials are all concatenated and then wrapped
// to make an N * `outputSize` 2D tensor. This function splits that tensor
// up by the row (according to the row counts in `rows) and returns
// a set of 1D row tensors for each chunk of rows.
//
// The output is not set in the returned values. You have to do that yourself.
std::vector<RegionReduction>
splitPartialsByRows(const RegionReduction &reduction_,
                    const std::vector<unsigned> &rows) {

  std::vector<RegionReduction> out(rows.size());
  auto reduction = reduction_;
  if (reduction.regularPartials() && reduction.getNumOuterStrides() != 1) {
    reduction.convertToIrregularPartials();
    logging::popops::trace("Converting to irregular partials so that we can"
                           " split partials by row.");
  }
  if (reduction.regularPartials()) {
    // For regular partials this is a case of changing the offset and
    // outerFactor to reference the correct piece of the partials
    unsigned cumulativeRows = 0;
    for (unsigned i = 0; i < out.size(); ++i) {
      out[i] = reduction;
      out[i].getOffset() =
          reduction.getOffset() + reduction.getStride() * cumulativeRows;
      out[i].outerFactor = rows[i];
      cumulativeRows += rows[i];
      out[i].getStride() = reduction.getStride();
    }
    return out;
  }
  // For Irregular partials we have to slice out rows from the vector of
  // partials until we have enough data for each row split
  const auto outputSize = reduction.output.numElements();
  // Get the number of rows in each partial.
  std::vector<std::size_t> rowsInPartials(reduction.getNumPartials());
  for (std::size_t i = 0; i < rowsInPartials.size(); ++i)
    rowsInPartials[i] = reduction.getPartials()[i].numElements() / outputSize;

  assert(std::accumulate(rowsInPartials.begin(), rowsInPartials.end(), 0) ==
         std::accumulate(rows.begin(), rows.end(), 0));

  std::size_t currentPartial = 0;
  std::size_t rowBegin = 0;

  for (unsigned i = 0; i < out.size(); ++i) {
    // Get `rows[i]` rows from the reduction and put them in out[i].
    // Gather them into this and assign at the end.
    IrregularPartials iPartials;

    auto rowsNeeded = rows[i];

    while (rowsNeeded > 0) {
      assert(currentPartial < rowsInPartials.size());

      auto rowsInCurrentPartial = rowsInPartials[currentPartial];

      std::size_t rowEnd = rowBegin + rowsNeeded;
      if (rowEnd > rowsInCurrentPartial)
        rowEnd = rowsInCurrentPartial;

      rowsNeeded -= (rowEnd - rowBegin);

      if (rowBegin == 0 && rowEnd == rowsInCurrentPartial) {
        // No need to slice.
        iPartials.data.push_back(reduction.getPartials()[currentPartial]);
        out[i].outerFactor = rowEnd;
      } else {
        // Take a slice.
        iPartials.data.push_back(
            reduction.getPartials()[currentPartial].flatten().slice(
                rowBegin * outputSize, rowEnd * outputSize));
        out[i].outerFactor = rowEnd - rowBegin;
      }

      rowBegin = rowEnd;
      if (rowBegin >= rowsInCurrentPartial) {
        rowBegin = 0;
        ++currentPartial;
      }
    }
    out[i].partials = iPartials;
  }
  return out;
}

// Call this function to connect reductions where none of them are two-stage
// reductions and there may be more reductions than worker contexts.
// The `assignments` should be the same size as `reductions` and
// contains the worker context/vertex each reduction should be done by.
void connectSingleStageReductions(
    poplar::Graph &graph, poplar::ComputeSet &cs, ReduceParams params,
    poplar::Type partialType, poplar::Type outputType, unsigned tile,
    const std::vector<RegionReduction> &reductions,
    const std::vector<unsigned> &assignments, bool reductionUsesInput,
    const poplar::DebugNameAndId &dnai) {

  assert(reductions.size() == assignments.size());

  logging::popops::trace("Connecting single stage reduction, size: {}",
                         reductions.size());

  // Map from worker context to list of reductions.
  // Optimisation: Copying the RegionReductions around could be avoided.
  std::map<unsigned, std::vector<RegionReduction>> reductionsPerWorker;

  for (std::size_t i = 0; i < assignments.size(); ++i)
    reductionsPerWorker[assignments[i]].emplace_back(reductions[i]);

  assert(reductionsPerWorker.size() <=
         graph.getTarget().getNumWorkerContexts());

  // Connect the single stage reductions.
  for (const auto &it : reductionsPerWorker) {
    const auto &vertexReductions = it.second;

    createVertex(graph, vertexReductions, params, partialType, outputType, cs,
                 tile, reductionUsesInput, {dnai});
  }
}
// Check if we should split some reductions to try to use all the workers.
// We are going to create 2 stages anyhow so that decision is simpler than it
// would be otherwise. We try to split the reductions that we want to implement
// here (reductionsToSplit) up such that the TOTAL number of reductions = the
// number of workers.
// Return the number of times we can split the "reductionsToSplit" reductions
unsigned findMaxSplits(const unsigned numWorkers,
                       const unsigned totalReductions,
                       const unsigned reductionsToSplit) {
  unsigned splits = 1;
  while (splits * reductionsToSplit + (totalReductions - reductionsToSplit) <=
         numWorkers) {
    splits++;
  }
  // If the while loop incremented it would have incremented one too many
  return (splits == 1 ? splits : splits - 1);
}
// Find the largest number of splits by column that should be applied to the
// specific reduction.  This will be less than or equal to splitsToUseWorkers
// and should not result in column splits with a grainSize that is
// inefficient to implement.
unsigned getAllowedColumnSplits(const poplar::Graph &graph,
                                const unsigned splitsToUseWorkers,
                                const RegionReduction &reduction) {
  const unsigned minColumnsToSplit = graph.getTarget().getVectorWidth(
      reduction.getPartials()[0].elementType());
  const unsigned thisStageOutputSize =
      reduction.output.numElements() * reduction.innerFactor;
  unsigned splits = 1;
  for (unsigned i = 2; i <= splitsToUseWorkers; i++) {
    if (reduction.output.numElements() % i == 0 &&
        (thisStageOutputSize / i) % minColumnsToSplit == 0) {
      splits = i;
    }
  }
  return splits;
}
// Split the partials into groups for implementation by each worker.
// We try to split into rowSplits groups if the number of partials and
// partials elements means that this is sensible.
std::pair<unsigned, std::vector<RegionReduction>> splitPartialsIntoGroups(
    const RegionReduction &reduction, const unsigned firstStageOutputSize,
    const unsigned firstStageFactor, const unsigned rowSplits) {
  // Default for there being no split
  unsigned groups = 1;
  const unsigned minRowsToSplit = 4;
  unsigned partialsElements = reduction.getNumPartialsElements();
  const unsigned rows = partialsElements / firstStageOutputSize;

  std::vector<unsigned> rowGroups(rowSplits,
                                  (rows / rowSplits) * firstStageFactor);
  if (rows > rowSplits * minRowsToSplit &&
      partialsElements % firstStageOutputSize == 0) {
    // Go with splitting and account for the remainder
    rowGroups[rowSplits - 1] += (rows % rowSplits) * firstStageFactor;
    groups = rowGroups.size();
  }

  std::vector<RegionReduction> partialsPerWorker;
  if (groups > 1) {
    partialsPerWorker = splitPartialsByRows(reduction, rowGroups);
  } else {
    partialsPerWorker.push_back(reduction);
  }
  return {groups, partialsPerWorker};
}
// Separate reductions that could be implemented according to the function
// passed.  Return true if no reductions are candidates
bool findSuitableReductions(
    const std::vector<RegionReduction> &reductions,
    const std::function<bool(RegionReduction)> &reductionIsSuitable,
    std::vector<RegionReduction> &candidateReductions,
    std::vector<RegionReduction> &remainingReductions) {
  for (auto &reduction : reductions) {
    if (reductionIsSuitable(reduction)) {
      candidateReductions.emplace_back(std::move(reduction));
    } else {
      remainingReductions.emplace_back(std::move(reduction));
    }
  }
  return (candidateReductions.size() == 0);
}
// Some column counts pose a problem and will have a poor implementation due to
// the number of columns not working well with the data type vector size.
// 2 halves is especially bad, and has a simple solution.  Reductions with
// columns A,B. Data A0 B0 A1 B1 A2 B2 ... in memory.
// Given a half and vector size 4 we reduce:
//          A0 B0 A1 B1
//          A2 B2 A3 B3
//          A4 B4 A5 B5
//          ...
// Result = Ae Be Ao Bo = the reduction of all evenA's, evenB's, oddA's, oddB's
// A second stage implements Ae + Ao and Be + Bo
//
// This principle can be applied for any case where the vector width and
// column width (output size) don't work well together but we will limit its
// implementation to these cases, as otherwise the 2 stage partials width
// is large (Usually = vectorWidth * outputWidth):
// Half (vector width 4), output width 2, so twoStagePartials width = 4
// Half                   output width 3, so twoStagePartials width = 12
// Float (vector width 2),output width 3, so twoStagePartials width = 6
std::vector<RegionReduction> connectProblemColumnCountReductions(
    poplar::Graph &graph, ComputeSetList &css, unsigned &reductionComputeSets,
    const ReduceParams &params, poplar::Type inputType,
    poplar::Type partialType, poplar::Type outputType, unsigned tile,
    const std::vector<RegionReduction> &reductions, bool reductionUsesInput,
    unsigned &remainingWorkers, const poplar::DebugNameAndId &dnai) {

  const auto vectorWidth = graph.getTarget().getVectorWidth(inputType);

  const auto reductionIsSuitable = [&](const RegionReduction &r) {
    return r.output.numElements() < 4 && r.output.numElements() != 1 &&
           (r.output.numElements() % vectorWidth);
  };

  const auto outputSize = [&](const RegionReduction &r) {
    return (vectorWidth == 4 && r.output.numElements() == 2)
               ? 4
               : r.output.numElements() * vectorWidth;
  };

  // As we may often not need to do any of this, check the reductions and exit
  // quickly. Note those that we can't do here, those that we could do here and
  // eventually those that we do here.
  std::vector<RegionReduction> remainingReductions;
  std::vector<RegionReduction> candidateReductions;
  std::vector<RegionReduction> consumedReductions;

  if (findSuitableReductions(reductions, reductionIsSuitable,
                             candidateReductions, remainingReductions)) {
    return remainingReductions;
  }
  logging::popops::trace(
      "Considering connecting {} reductions as 2 stages. Reason:"
      " Optimising column count",
      candidateReductions.size());

  // A vector of partials per reduction - each will be empty if not implemented
  // here, contains "Splits" partials otherwise.  Ie one partial if implemented
  // but not split, more if implemented and split
  std::vector<std::vector<poplar::Tensor>> twoStagePartials(
      candidateReductions.size());

  // If possible, we could split each reduction this many times, resulting in
  // all workers being occupied.  It may not be possible due to the size of
  // the individual reductions, but this forms an upper bound on the splits.
  const auto splitsToUseWorkers = findMaxSplits(
      remainingWorkers, reductions.size(), candidateReductions.size());
  unsigned partialsIndex = 0;
  // Create the first stage and note which reductions we are consuming
  for (auto &reduction : candidateReductions) {
    const unsigned firstStageOutputSize = outputSize(reduction);

    // If the partials aren't suitable we need to back out of this for this
    // reduction
    const bool partialsAreSuitable = [&]() {
      if (reduction.regularPartials()) {
        // We can't do this unless the original outputsize = stride. Also there
        // must be reducing to do if we increase the effective output size.
        // Plus the whole area must be a multiple of the new outputsize
        return reduction.getNumPartialsElementsPerOuterStride() !=
                   firstStageOutputSize &&
               (reduction.output.numElements() == reduction.getStride()) &&
               (reduction.getNumPartialsElementsPerOuterStride() %
                firstStageOutputSize) == 0;
      } else {
        return std::all_of(
            reduction.getPartials().begin(), reduction.getPartials().end(),
            [&](const poplar::Tensor &t) {
              return (t.numElements() % firstStageOutputSize) == 0;
            });
      }
    }();
    if (partialsAreSuitable) {
      consumedReductions.emplace_back(std::move(reduction));
      const unsigned firstStageFactor =
          firstStageOutputSize / consumedReductions.back().output.numElements();
      const auto partialsGrouped = splitPartialsIntoGroups(
          consumedReductions.back(), firstStageOutputSize, firstStageFactor,
          splitsToUseWorkers);
      for (unsigned j = 0; j < partialsGrouped.first; j++) {
        // Create and record the output of the 1st stage / partials for the 2nd
        twoStagePartials[partialsIndex].push_back(
            graph.addVariable(partialType, {firstStageOutputSize},
                              {dnai, "secondStagePartials"}));
        graph.setTileMapping(twoStagePartials[partialsIndex].back(), tile);

        RegionReduction firstStage;
        firstStage.output = twoStagePartials[partialsIndex].back();
        firstStage.partials = partialsGrouped.second[j].partials;
        if (firstStage.regularPartials()) {
          firstStage.getStride() = firstStage.getStride() * firstStageFactor;
        }
        firstStage.outerFactor =
            partialsGrouped.second[j].outerFactor / firstStageFactor;

        createVertex(graph, {firstStage}, params.op, inputType, partialType,
                     css.getCs1(reductionComputeSets), tile, reductionUsesInput,
                     {dnai});
        remainingWorkers = (remainingWorkers == 0) ? 0 : remainingWorkers - 1;
      }
      partialsIndex++;
    } else {
      remainingReductions.emplace_back(std::move(reduction));
    }
  }
  if (consumedReductions.size() == 0) {
    logging::popops::trace(
        "No reductions suitable for column count optimisations");
    return remainingReductions;
  }
  logging::popops::trace(
      "On tile second stage reductions for column count optimisations");

  // Create the second stage
  if (reductionComputeSets != 2) {
    css.add(graph, {dnai, "Reduce_Second_Stage"});
    reductionComputeSets++;
  }
  // Don't square again in the second stage.
  ReduceParams secondStageParams = params;
  if (params.op == Operation::SQUARE_ADD) {
    secondStageParams.op = Operation::ADD;
  }
  partialsIndex = 0;
  for (auto &red : consumedReductions) {
    // This covers the cases that we are targeting, a more general statement
    // would be needed if we target more.
    const unsigned ssReductionFactor =
        (red.output.numElements() == 3 && vectorWidth == 4) ? 4 : 2;
    const unsigned ssReductionStride = red.output.numElements();

    for (unsigned j = 0; j < red.output.numElements(); j++) {
      // Create a vector of partials for each reduction
      std::vector<poplar::Tensor> partials;
      for (auto &par : twoStagePartials[partialsIndex]) {
        for (unsigned k = 0; k < ssReductionFactor; k++) {
          partials.push_back(par.slice(j + ssReductionStride * k,
                                       j + ssReductionStride * k + 1));
        }
      }
      // Create the reduction and its output, create the vertex
      RegionReduction secondStageReduction;
      secondStageReduction.output = red.output.slice(j, j + 1);
      IrregularPartials iPartials = {{concat(partials)}};
      secondStageReduction.partials = iPartials;

      createVertex(graph, {secondStageReduction}, secondStageParams,
                   partialType, outputType, css.getCs2(reductionComputeSets),
                   tile, reductionUsesInput, {dnai});
    }
    partialsIndex++;
  }
  logging::popops::trace("Connected {} reductions as 2 stages. Reason:"
                         " Optimising column count",
                         consumedReductions.size());
  return remainingReductions;
}

// If the inner factor parameter requires us to create some two stage reductions
// then do so.  This may not consume all (or any of) the reductions so return
// those not dealt with. The compute sets created can have more reductions added
// to them later.
// We have to use a 2 stage reduction to deal with innerFactor != 1 and
// outputElements > 1.  Patterns with data belonging to columns A,B,C,D such as:
// A0 A1 B0 B1 C0 C1 D0 D1, .... Are reduced to : Aeven Aodd, Beven, Bodd... by
// the first stage, and to A B C D by the second stage
std::vector<RegionReduction> connectSmallInnerFactorReductions(
    poplar::Graph &graph, ComputeSetList &css, unsigned &reductionComputeSets,
    const ReduceParams &params, poplar::Type inputType,
    poplar::Type partialType, poplar::Type outputType, unsigned tile,
    const std::vector<RegionReduction> &reductions, bool reductionUsesInput,
    unsigned &remainingWorkers, const poplar::DebugNameAndId &dnai) {

  const auto reductionIsSuitable = [](const RegionReduction &r) {
    return r.innerFactor != 1 && r.output.numElements() != 1;
  };
  // As we may often not need to do any of this, check the reductions and exit
  // quickly.  The number of suitable reductions is also useful later.
  std::vector<RegionReduction> remainingReductions;
  std::vector<RegionReduction> consumedReductions;
  if (findSuitableReductions(reductions, reductionIsSuitable,
                             consumedReductions, remainingReductions)) {
    return remainingReductions;
  }

  logging::popops::trace("Connecting {} reductions as 2 stages. "
                         "Reason: InnerFactor and OuterFactor both != 1",
                         consumedReductions.size());
  // If possible, we could split each reduction this many times, resulting in
  // all workers being occupied.  It may not be possible due to the size of
  // the individual reductions, but this forms an upper bound on the splits.
  const auto splitsToUseWorkers = findMaxSplits(
      remainingWorkers, reductions.size(), consumedReductions.size());

  // Here we attempt to split by column to make better use of all the workers.
  // Splitting by row works but produces an inefficient second
  // reduction stage as continuous reduce cannot be targeted.
  //
  // Each reduction was previously divided up considering the grainSize
  // for the data type and the number of columns.  InnerFactor was not
  // considered - so if there are enough workers we can divide it further.
  // Here we check each reduction and split it if required.
  std::vector<RegionReduction> implementedReductions;
  for (auto &red : consumedReductions) {
    const auto splitsThisReduction =
        getAllowedColumnSplits(graph, splitsToUseWorkers, red);

    if (splitsThisReduction > 1) {
      const unsigned outWidth = red.output.numElements() / splitsThisReduction;
      const unsigned reshapeDim1 =
          outWidth * red.innerFactor * splitsThisReduction;
      const auto splitWidth = outWidth * red.innerFactor;
      for (unsigned j = 0; j < splitsThisReduction; j++) {
        // Add a reduction per split, containing a subset of the columns of
        // the original reduction
        implementedReductions.push_back({});
        implementedReductions.back().output =
            red.output.slice(j * outWidth, (j + 1) * outWidth);
        implementedReductions.back().innerFactor = red.innerFactor;
        implementedReductions.back().outerFactor = red.outerFactor;

        if (red.regularPartials()) {
          implementedReductions.back().getPartials() = red.getPartials();
          implementedReductions.back().getStride() = red.getStride();
          implementedReductions.back().getOffset() =
              red.getOffset() + (j * splitWidth);
          implementedReductions.back().getOuterStride() = red.getOuterStride();
          implementedReductions.back().getNumOuterStrides() =
              red.getNumOuterStrides();
        } else {
          IrregularPartials iPartials;

          for (auto &par : red.getPartials()) {
            const auto partial =
                par.reshape({par.numElements() / reshapeDim1, reshapeDim1});
            iPartials.data.push_back(
                partial.slice(j * splitWidth, (j + 1) * splitWidth, 1)
                    .flatten());
          }
          implementedReductions.back().partials = iPartials;
        }
      }
    } else {
      // Add reduction to the list to be implemented here, without splitting
      implementedReductions.emplace_back(std::move(red));
    }
  }
  remainingWorkers = implementedReductions.size() > remainingWorkers
                         ? 0
                         : remainingWorkers - implementedReductions.size();
  // Create the first stage and note the partials
  std::vector<boost::optional<poplar::Tensor>> twoStagePartials(
      implementedReductions.size());
  for (unsigned i = 0; i < implementedReductions.size(); i++) {
    const auto &red = implementedReductions[i];
    auto firstStageOutputSize = red.output.numElements() * red.innerFactor;
    // Create and record the partials
    twoStagePartials[i] = graph.addVariable(partialType, {firstStageOutputSize},
                                            {dnai, "secondStagePartials"});
    graph.setTileMapping(twoStagePartials[i].get(), tile);

    // Connect the first stage
    RegionReduction firstStage;
    firstStage.output = twoStagePartials[i].get();
    firstStage.partials = red.partials;

    firstStage.outerFactor = red.outerFactor;

    createVertex(graph, {firstStage}, params.op, inputType, partialType,
                 css.getCs1(reductionComputeSets), tile, reductionUsesInput,
                 {dnai});
  }
  // Create the second stage
  if (reductionComputeSets != 2) {
    css.add(graph, {dnai, "Reduce_Second_Stage"});
    reductionComputeSets++;
  }
  // Don't square again in the second stage.
  ReduceParams secondStageParams = params;
  if (params.op == Operation::SQUARE_ADD) {
    secondStageParams.op = Operation::ADD;
  }
  for (unsigned i = 0; i < implementedReductions.size(); i++) {
    const auto &red = implementedReductions[i];
    // For each original reduction that has a contiguous group of outputs
    // make a vector of region reductions each with a single partial and a
    // single output.  Each vector should target a continuous reduce vertex.
    const auto outRegions = graph.getSortedContiguousRegions(
        red.output, {{0, red.output.numElements()}});
    for (auto &outRegion : outRegions) {
      // Build a vector of reductions to result in the outputs of the original
      // reduction
      std::vector<RegionReduction> ssReductions;
      for (auto &interval : outRegion) {
        for (unsigned j = 0; j < interval.size(); j++) {
          const auto index = interval.begin() + j;
          ssReductions.push_back({});
          ssReductions.back().output = red.output.slice(index, index + 1);

          IrregularPartials iPartials = {{twoStagePartials[i].get().slice(
              red.innerFactor * index, red.innerFactor * (index + 1))}};
          ssReductions.back().partials = iPartials;
        }
      }
      createVertex(graph, ssReductions, secondStageParams, partialType,
                   outputType, css.getCs2(reductionComputeSets), tile,
                   reductionUsesInput, {dnai});
    }
  }
  return remainingReductions;
}

// This is called when the number of reductions is less than or equal to
// the number of workers, and some of them *may* be split. Despite the name
// there may actually be no two-stage reductions, or there may be a mix,
// or there may be no single-stage reductions.
//
// `splits` is the number of pieces to split each reduction into (if 1 it
// means it is a single-stage reduction).
void connectTwoStageReductions(poplar::Graph &graph, ComputeSetList &css,
                               unsigned &reductionComputeSets,
                               const ReduceParams &params,
                               poplar::Type inputType, poplar::Type partialType,
                               poplar::Type outputType, unsigned tile,
                               const std::vector<RegionReduction> &reductions,
                               const std::vector<unsigned> &splits,
                               bool reductionUsesInput,
                               const poplar::DebugNameAndId &dnai) {
  // Triple check...
  assert(splits.size() == reductions.size());
  std::vector<RegionReduction> singleStageReductions;
  std::vector<unsigned> singleStageAssignments;

  for (std::size_t i = 0; i < splits.size(); ++i) {
    if (splits[i] == 1) {
      singleStageReductions.push_back(reductions[i]);
      singleStageAssignments.push_back(singleStageAssignments.size());
      logging::popops::trace("Single stage reduction[{}] {}", i, reductions[i]);
    }
  }

  if (!singleStageReductions.empty()) {
    connectSingleStageReductions(graph, css.getCs1(reductionComputeSets),
                                 params, inputType, outputType, tile,
                                 singleStageReductions, singleStageAssignments,
                                 reductionUsesInput, {dnai});
  }

  // If there are no two-stage reductions, that's it!
  if (singleStageReductions.size() == reductions.size())
    return;

  // Map from reduction number to the partial for second stage reductions.
  std::map<unsigned, poplar::Tensor> secondStagePartials;

  // Now connect the first stage of the two-stage reductions.
  for (std::size_t i = 0; i < splits.size(); ++i) {
    // Ignore the single-stage ones we've already done.
    if (splits[i] <= 1)
      continue;

    // Optimisation: In some cases we'd want to do a two stage reduction anyway
    // e.g. if there are 6 reductions with reduction factors
    //   x1000, x5, x5, x5, x5, x5.

    logging::popops::trace("Two stage reduction[{}] = {}", i, reductions[i]);

    auto outputSize = reductions[i].output.numElements();

    auto totalRows = reductionFactor(reductions[i]);
    const bool rowSizeOne = reductions[i].output.numElements() == 1;

    auto rowsPerWorker = splitRowsToWorkers(
        totalRows, splits[i],
        rowSizeOne ? graph.getTarget().getVectorWidth(inputType) : 1);

    assert(!rowsPerWorker.empty());
    assert(rowsPerWorker.size() <= splits[i]);
    assert(std::accumulate(rowsPerWorker.begin(), rowsPerWorker.end(), 0u) ==
           totalRows);

    auto partialsPerWorker = splitPartialsByRows(reductions[i], rowsPerWorker);

    assert(partialsPerWorker.size() == rowsPerWorker.size());

    // Create a tensor for all the partial results.
    secondStagePartials[i] =
        graph.addVariable(partialType, {outputSize * partialsPerWorker.size()},
                          {dnai, "secondStagePartials"});
    graph.setTileMapping(secondStagePartials[i], tile);

    // Now create the new RegionReductions.
    for (unsigned s = 0; s < partialsPerWorker.size(); ++s) {
      RegionReduction firstStage;
      firstStage.output =
          secondStagePartials[i].slice(s * outputSize, (s + 1) * outputSize);
      firstStage.partials = partialsPerWorker[s].partials;
      firstStage.outerFactor = partialsPerWorker[s].outerFactor;

      // Don't do scale or update in the first stage.
      ReduceParams firstStageParams(params.op);
      createVertex(graph, {firstStage}, firstStageParams, inputType,
                   partialType, css.getCs1(reductionComputeSets), tile,
                   reductionUsesInput, {dnai});
    }
  }

  // And the second stage of the two-stage reductions.

  // Add a second compute set if needed.
  if (reductionComputeSets != 2) {
    css.add(graph, {dnai, "Reduce_Second_Stage"});
    reductionComputeSets++;
  }
  // Work out which vertex should do each second stage. We just assign
  // in a round-robin manner since there shouldn't really ever be more than
  // the number of worker contexts anyway.

  std::vector<std::vector<RegionReduction>> secondStageReductions;
  secondStageReductions.reserve(graph.getTarget().getNumWorkerContexts());

  // Don't square again in the second stage.
  ReduceParams secondStageParams = params;
  if (params.op == Operation::SQUARE_ADD) {
    secondStageParams.op = Operation::ADD;
  }

  for (const auto &r : secondStagePartials) {
    assert(r.first >= 0 && r.first < reductions.size());

    RegionReduction secondStageReduction;
    secondStageReduction.output = reductions[r.first].output;
    // There's only ever one partial for each second stage reduction.
    IrregularPartials iPartials = {{r.second}};
    secondStageReduction.partials = iPartials;

    createVertex(graph, {secondStageReduction}, secondStageParams, partialType,
                 outputType, css.getCs2(reductionComputeSets), tile,
                 reductionUsesInput, {dnai});
  }
}
// Distribute reductions between workers based on the number of outputs each
// reduction has. We attempt to assign the work evenly, but also minimise the
// number of workers if it would make no differentce to the maximum work any one
// worker has to do.
std::vector<unsigned> findSplitsBetweenWorkers(const unsigned maxSplits,
                                               const unsigned outputs) {
  const unsigned splits = std::min(maxSplits, outputs);

  // Each of these represents the number of reduction outputs each worker will
  // compute.
  std::vector<unsigned> splitThresholds(splits, outputs / splits);
  // Allocate the split work with remainder, starting at the first worker.
  for (unsigned j = 0; j < outputs % splits; j++) {
    splitThresholds[j]++;
  }
  // Starting at the end, attempt to remove workers and re-assign their
  // allocation if the largest is unaffected.  Exit as soon as we fail.
  const unsigned largest = splitThresholds[0];
  for (unsigned j = splits; j > 0; j--) {
    bool workerRemoved = false;
    for (unsigned k = 0; k < j - 1; k++) {
      if (splitThresholds[j - 1] + splitThresholds[k] == largest) {
        splitThresholds[k] += splitThresholds[j - 1];
        splitThresholds.resize(splitThresholds.size() - 1);
        workerRemoved = true;
        break;
      }
    }
    if (!workerRemoved) {
      break;
    }
  }

  // Decrementing the first element and adding others makes this
  // zero based and cumulative to compare to the loop index when in use.
  splitThresholds[0]--;
  for (unsigned j = 1; j < splits; j++) {
    splitThresholds[j] += splitThresholds[j - 1];
  }
  return splitThresholds;
}

// These reductions have an innerFactor !=1.  They cannot be implemented
// directly, insted they need to be split into "outputSize" regionReductions,
// each with a new outputSize = 1.  As they should each have a single
// region as the input they should result in a continuous reduce vertex.
std::vector<RegionReduction> connectLargeInnerFactorReductions(
    poplar::Graph &graph, ComputeSetList &css, unsigned reductionComputeSets,
    const ReduceParams &params, poplar::Type partialType,
    poplar::Type outputType, unsigned tile,
    const std::vector<RegionReduction> &reductions, bool reductionUsesInput,
    unsigned &remainingWorkers, const poplar::DebugNameAndId &dnai) {

  const auto reductionIsSuitable = [&](const RegionReduction &r) {
    return r.innerFactor != 1 && r.outerFactor == 1 &&
           r.output.numElements() != 1;
  };
  // Exit quickly if we have nothing to do, and find the number of reductions
  // which we will consume here (There is no other way to implement them)
  std::vector<RegionReduction> consumedReductions;
  // Store those that we haven't consumed
  std::vector<RegionReduction> remainingReductions;
  if (findSuitableReductions(reductions, reductionIsSuitable,
                             consumedReductions, remainingReductions)) {
    return remainingReductions;
  }

  // We intend to split each reduction this many times to share those we
  // consume here between workers, leaving 1 worker for each reduction we don't
  // consume here
  const auto maxSplits = findMaxSplits(remainingWorkers, reductions.size(),
                                       consumedReductions.size());
  logging::popops::trace(
      "Splitting and connecting {} reductions due to innerFactor != 1",
      consumedReductions.size());
  for (auto &reduction : consumedReductions) {
    const auto splitThresholds =
        findSplitsBetweenWorkers(maxSplits, reduction.output.numElements());
    unsigned splitIndex = 0;
    // A vector of region reductions with innerFactor=1 which we construct to
    // describe a single regionReduction with innerFactor!=1
    std::vector<RegionReduction> red;

    for (unsigned j = 0; j < reduction.output.numElements(); j++) {
      red.push_back({reduction.output.slice(j, j + 1), {}});

      // Assume the  partials could be a multiple of the whole
      // sequence 000011112222, so reshape:
      // 0000111112222
      // 0000111122222
      // ......
      // and slice out all the 0000's 1111's etc.
      // Presently, we only allow this to be called when outerFactor == 1
      // means that we will only have 1 row, but are capable of dealing with
      // many.
      if (reduction.regularPartials()) {
        red.back().getPartials() = reduction.getPartials();
        red.back().getOffset() =
            reduction.getOffset() + j * reduction.innerFactor;
        red.back().outerFactor = reduction.outerFactor;
        red.back().innerFactor = reduction.innerFactor;
        red.back().getStride() = reduction.getStride();
      } else {
        IrregularPartials iPartials;
        for (auto &partials : reduction.getPartials()) {
          const unsigned reshapeDim1 =
              reduction.output.numElements() * reduction.innerFactor;
          auto par = partials.reshape(
              {partials.numElements() / reshapeDim1, reshapeDim1});

          iPartials.data.push_back(par.slice(j * reduction.innerFactor,
                                             (j + 1) * reduction.innerFactor, 1)
                                       .flatten());
        }
        red.back().partials = iPartials;
      }
      if (j == splitThresholds[splitIndex]) {
        // We have assigned enough reductions to a worker - make the
        // vertex and move onto the next worker if there is one
        createVertex(graph, red, params, partialType, outputType,
                     css.getCs1(reductionComputeSets), tile, reductionUsesInput,
                     {dnai});
        splitIndex++;
        red.clear();
        if (remainingWorkers > 0) {
          remainingWorkers--;
        }
      }
    }
  }
  return remainingReductions;
}

} // anonymous namespace

void RegionReduction::convertToIrregularPartials() {
  if (regularPartials()) {
    const std::vector<RegionReduction> regions = {*this};
    const auto partials = extractPartials(regions);
    this->partials = IrregularPartials{partials};
  }
}

void connectReductions(poplar::Graph &graph, ComputeSetList &css,
                       ReduceParams params, poplar::Type inputType,
                       poplar::Type partialType, poplar::Type outputType,
                       unsigned tile,
                       const std::vector<RegionReduction> &reductions,
                       bool reductionUsesInput,
                       const ReduceManyInfo &reductionInfo,
                       const poplar::DebugNameAndId &dnai) {

  const auto &target = graph.getTarget();
  // Optimisation: If there is just one partial for an output we don't need to
  // calculate it, but hooking that up is a bit fiddly.

  // Check that all the inputs and outputs are of the appropriate sizes
  // and types.
  logging::popops::trace("Connecting {} reductions", reductions.size());

  for (const auto &r : reductions) {

    auto outputSize = r.output.numElements();
    if (outputSize == 0) {
      throw poputil::poplibs_error("Zero-sized reduction output");
    }

    if (r.output.elementType() != outputType) {
      throw poputil::poplibs_error("Reduction output is incorrect type");
    }

    if (r.regularPartials()) {
      const auto &p = r.getPartials();
      if (p[0].numElements() == 0)
        throw poputil::poplibs_error("Zero-sized reduction partial");

      if (p[0].elementType() != inputType)
        throw poputil::poplibs_error("Reduction partial is incorrect type");
    } else {
      for (const auto &p : r.getPartials()) {
        if (p.numElements() == 0)
          throw poputil::poplibs_error("Zero-sized reduction partial");

        if (p.numElements() % outputSize != 0)
          throw poputil::poplibs_error("Reduction partial size is not a "
                                       "multiple of the output size");

        if (p.elementType() != inputType)
          throw poputil::poplibs_error("Reduction partial is incorrect type");
      }
    }
  }

  const std::size_t vectorListMaxSize = [&] {
    // output region size does not change the field dimension of the vertex and
    // it doesn't matter if the last parameter is true or false. We just set it
    // to true without checking if the size is one or not
    const auto vertex = getReductionVertexName(
        params, partialType, outputType, ReductionSpecialisation::DEFAULT);
    return graph.getMaxFieldDim(vertex, "partials", 0);
  }();

  // There can be lots of reasons for creating 2 on tile reduction stages here.
  // We will make at least one compute set, maybe 2.  Start with one and keep
  // track of how many we have made
  css.add(graph, {dnai, +"Reduce"});
  unsigned reductionComputeSets = 1;

  unsigned remainingWorkers =
      reductionInfo.assignWorkers(target.getNumWorkerContexts());

  auto remainingReductions = connectLargeInnerFactorReductions(
      graph, css, reductionComputeSets, params, inputType, outputType, tile,
      reductions, reductionUsesInput, remainingWorkers, {dnai});
  if (remainingReductions.size() == 0) {
    return;
  }

  remainingReductions = connectSmallInnerFactorReductions(
      graph, css, reductionComputeSets, params, inputType, partialType,
      outputType, tile, remainingReductions, reductionUsesInput,
      remainingWorkers, {dnai});
  if (remainingReductions.size() == 0) {
    return;
  }

  // reductions contains a list of ReductionRegions, if any of the regions
  // contain more partials than we can fit into a DeltaN list we must split
  // those regions up.
  const auto partialsTooLarge = [&](const RegionReduction &reduction) {
    return reduction.getPartials().size() >= vectorListMaxSize;
  };

  const bool somePartialsTooLarge =
      std::any_of(std::begin(remainingReductions),
                  std::end(remainingReductions), partialsTooLarge);

  if (!somePartialsTooLarge) {
    remainingReductions = connectProblemColumnCountReductions(
        graph, css, reductionComputeSets, params, inputType, partialType,
        outputType, tile, remainingReductions, reductionUsesInput,
        remainingWorkers, {dnai});
    if (remainingReductions.size() == 0) {
      return;
    }
  }
  // Reductions may need splitting into two stages for other reasons, either
  // to share work between workers or because the partials are too large.
  // Some reductions may already have been consumed by the steps above and will
  // already have committed us to making 2 stages and occupied some of the
  // workers in the 1st and 2nd stages. So at this point we are not
  // neccessarily starting with a clean slate.
  // Trying to co-ordinate allocating workers when some were already allocated
  // in other 2-stage reductions looks messy. Mostly we get a reduction that
  // is all dealt with in "InnerFactor", "problemColumn" or not at all.  So
  // leave each function to allocate workers reasonably without knowledge of
  // what the others may do.

  // Consider all reductions when deciding if to split, not those remaining
  const bool useTwoStageReduction =
      remainingReductions.size() < remainingWorkers || somePartialsTooLarge;

  // See if there is the possibility of easily splitting reductions
  // into two-level ones.
  if (useTwoStageReduction) {
    // Try to split some into multi-stage reductions.
    auto splits = splitTwoStageReductionsBetweenWorkers(
        target, params.op, remainingReductions, vectorListMaxSize,
        remainingWorkers);
    const auto workersUsed = std::accumulate(splits.begin(), splits.end(), 0u);
    logging::popops::trace(
        "Splitting {} reductions between {} vertices on tile {} {}",
        remainingReductions.size(), workersUsed, tile,
        remainingReductions.size() == workersUsed
            ? " "
            : "(Plus reductions to combine those split by row)");

    for (size_t i = 0; i < remainingReductions.size(); i++) {

      // Address splits which can't be represented as regular partials
      if (splits[i] != 1 && remainingReductions[i].regularPartials() &&
          remainingReductions[i].innerFactor != 1 &&
          remainingReductions[i].outerFactor != 1) {
        remainingReductions[i].convertToIrregularPartials();
      }
    }
    connectTwoStageReductions(graph, css, reductionComputeSets, params,
                              inputType, partialType, outputType, tile,
                              remainingReductions, splits, reductionUsesInput,
                              {dnai});

  } else {
    logging::popops::trace("Using single stage reduction on tile {}", tile);
    // Distribute the reductionTensors among 6 (or whatever) vertices.
    auto reductionAssignments = distributeReductionsBetweenWorkers(
        target, params.op, remainingReductions, remainingWorkers);

    connectSingleStageReductions(graph, css.getCs1(reductionComputeSets),
                                 params, inputType, outputType, tile,
                                 remainingReductions, reductionAssignments,
                                 reductionUsesInput, {dnai});
  }
}

/// reduction with a scalar output and word-aligned inputs can be executed
/// using specialised vertices
/// \param graph  The compute graph
/// \param params The parameters of this reduction
/// \param r      The reductions to be performed
static bool isScalarOutputReduction(const poplar::Graph &graph,
                                    const ReduceParams &params,
                                    const RegionReductionRange r) {

  // This must be a reduction of a single region
  if (r.size() != 1) {
    return false;
  }
  const auto &target = graph.getTarget();
  const auto &r0 = r.front();
  // The output must be scalar
  auto outputElements = r0.output.numElements();
  if (outputElements > 1) {
    return false;
  }
  if (params.update == true || params.useScale == true) {
    return false;
  }
  const auto atomicWriteSize = target.getAtomicStoreGranularity();
  const auto inType = r0.getPartials()[0].elementType();
  const auto inTypeSize = target.getTypeSize(inType);
  if (r0.regularPartials()) {
    return (outputElements * inTypeSize) % atomicWriteSize == 0;
  } else {
    bool nextIsMisaligned = false;
    for (const auto &p : r0.getPartials()) {
      // Each incoming partial region must be for a full set of outputs
      if (p.numElements() % outputElements != 0) {
        return false;
      }
      // It must be possible to receive each partial over exchange without
      // requiring a gather.
      if (nextIsMisaligned) {
        return false;
      }
      nextIsMisaligned = (p.numElements() * inTypeSize) % atomicWriteSize;
    }
    return true;
  }
}

static bool isStridedReduction(const poplar::Graph &graph,
                               const ReduceParams &params,
                               const RegionReductionRange r,
                               bool reductionUsesInput) {

  const auto &target = graph.getTarget();
  const auto &r0 = r.front();
  const auto inType = r0.getPartials()[0].elementType();
  const auto inTypeSize = target.getTypeSize(inType);
  const auto outTypeSize = target.getTypeSize(r0.output.elementType());
  const auto atomicWriteSize = target.getAtomicStoreGranularity();
  // The output must be 4byte writable
  if ((r0.output.numElements() * outTypeSize) % atomicWriteSize != 0) {
    return false;
  }
  std::vector<poplar::Tensor> outputs;
  for (const auto &region : r) {
    outputs.push_back(region.output);
  }
  const auto output = concat(outputs);
  if ((output.numElements() * inTypeSize) % 8 != 0) {
    return false;
  }
  auto outputElements = r0.output.numElements();

  // This must be a reduction of a single region or of multiple adjacent regions
  if (r.size() != 1) {
    if (!r0.regularPartials()) {
      return false;
    }
    if ((r0.getStride() * inTypeSize) % 8 != 0) {
      return false;
    }
    if (r0.innerFactor != 1) {
      return false;
    }
    unsigned referenceOffset = r0.getOffset() + outputElements;
    for (unsigned i = 1; i < r.size(); i++) {
      if (!r[i].regularPartials() || r[i].getStride() != r0.getStride() ||
          r[i].getOffset() != referenceOffset || r[i].innerFactor != 1 ||
          r[i].getPartials() != r0.getPartials() ||
          r[i].outerFactor != r0.outerFactor ||
          r[i].getOuterStride() != r0.getOuterStride()) {
        return false;
      }
      referenceOffset += r[i].output.numElements();
    }
  }
  if (r0.regularPartials()) {
    return (r0.output.numElements() * inTypeSize) % atomicWriteSize == 0;
  } else {
    bool partialsCopyIsCheap = true;
    bool nextIsMisaligned = false;
    for (const auto &p : r0.getPartials()) {
      // Each incoming partial region must be for a full set of outputs
      if (p.numElements() % outputElements != 0) {
        partialsCopyIsCheap = false;
        break;
      }
      // If not contiguous it should be possible to copy all the partials
      // together cheaply. This could be worthwhile even if the copy isn't this
      // cheap.
      if (nextIsMisaligned) {
        partialsCopyIsCheap = false;
        break;
      }
      nextIsMisaligned = (p.numElements() * outTypeSize) % atomicWriteSize;
      if (!partialsCopyIsCheap) {
        break;
      }
    }
    return partialsCopyIsCheap || (!reductionUsesInput);
  }
}

static bool allRegionsContinuous(const poplar::Graph &graph,
                                 const RegionReductionRange regions,
                                 const ReduceParams &params,
                                 const poplar::Type &partialsType,
                                 const bool reductionUsesInput) {
  boost::optional<unsigned> requiredPartialsElements;
  std::vector<poplar::Tensor> outputs;
  std::vector<poplar::Tensor> partials;
  if (regions.empty()) {
    return false;
  }
  for (const auto &red : regions) {
    if (red.output.numElements() != 1) {
      return false;
    }
    // Total number of partial elements must be equal for each reduction and
    // not equal to 1
    const unsigned thisReductionPartialsElements = red.getNumPartialsElements();

    if (requiredPartialsElements) {
      if (thisReductionPartialsElements != requiredPartialsElements.get()) {
        return false;
      }
    } else {
      requiredPartialsElements = thisReductionPartialsElements;
    }

    outputs.push_back(red.output);
  }
  const auto singleOut = concat(outputs).isContiguous();

  bool nextIsMisaligned = false;
  bool partialsCopyIsCheap = true;
  if (!extractFlatPartials(regions).isContiguous()) {
    const auto &target = graph.getTarget();
    const auto atomicWriteSize = target.getAtomicStoreGranularity();
    const auto bytesPerPartial = target.getTypeSize(partialsType);
    for (const auto &red : regions) {
      // If not contiguous it should be possible to copy all the partials
      // together cheaply. This could be worthwhile even if the copy isn't this
      // cheap.
      if (red.regularPartials()) {
        if (nextIsMisaligned) {
          partialsCopyIsCheap = false;
          break;
        }

        const auto elemsPerRegion = red.innerFactor * red.output.numElements();
        nextIsMisaligned = (elemsPerRegion * bytesPerPartial) % atomicWriteSize;
      } else {
        for (const auto &p : red.getPartials()) {
          if (nextIsMisaligned) {
            partialsCopyIsCheap = false;
            break;
          }
          nextIsMisaligned =
              (p.numElements() * bytesPerPartial) % atomicWriteSize;
        }
        if (!partialsCopyIsCheap) {
          break;
        }
      }
    }
  }
  return singleOut && (partialsCopyIsCheap || (!reductionUsesInput));
}

ReductionSpecialisation getReductionVertexSpecialisation(
    const poplar::Graph &graph, const ReduceParams &params,
    const RegionReductionRange regions, poplar::Type partialType,
    bool reductionUsesInput) {

  auto specialisation = ReductionSpecialisation::DEFAULT;
  // We don't have assembler implementations for some specialisations
  // other than for ADD and SQUARE_ADD operations. The code generated by the
  // compiler is very inefficient for these operations.
  bool opIsAddOrSquareAdd =
      params.op == Operation::ADD || params.op == Operation::SQUARE_ADD;
  bool opIsMaxOrMin =
      params.op == Operation::MAX || params.op == Operation::MIN;
  // For LOG_ADD there is no assembler - don't let this restrict its use
  bool opIsLogAdd = params.op == Operation::LOG_ADD;

  if (allRegionsContinuous(graph, regions, params, partialType,
                           reductionUsesInput) &&
      (opIsAddOrSquareAdd || opIsMaxOrMin || opIsLogAdd)) {
    return ReductionSpecialisation::ALL_REGIONS_CONTINUOUS;
  } else if (isScalarOutputReduction(graph, params, regions) &&
             (opIsAddOrSquareAdd || opIsLogAdd)) {
    return ReductionSpecialisation::SCALAR_OUTPUT_SINGLE_INPUT;
  } else if (isStridedReduction(graph, params, regions, reductionUsesInput)) {
    // both input and output must be full width accumulators
    const auto outElemType = regions[0].output.elementType();
    const auto partialsElemType = regions[0].getPartials()[0].elementType();
    const auto outElems = regions[0].output.numElements();
    bool addOpHasAssembly =
        (outElemType == poplar::FLOAT || outElemType == poplar::HALF) &&
        opIsAddOrSquareAdd;
    bool maxMinOpHasAssembly =
        (outElemType == poplar::FLOAT || outElemType == poplar::HALF) &&
        opIsMaxOrMin;
    const auto &target = graph.getTarget();
    const auto targetIsCpu = target.getTargetType() == poplar::TargetType::CPU;
    const auto partialsElemSize =
        (targetIsCpu && partialsElemType == poplar::HALF)
            ? 2
            : target.getTypeSize(partialsElemType);

    if ((addOpHasAssembly || maxMinOpHasAssembly || opIsLogAdd) &&
        outElems * target.getTypeSize(outElemType) % 4 == 0 &&
        outElems * partialsElemSize % 8 == 0) {
      // output must be whole words
      return ReductionSpecialisation::STRIDED_REDUCE;
    }

  } else {
    // find if all output regions are of size 1
    auto allOutputRegionsOfSizeOne = std::all_of(
        regions.begin(), regions.end(), [](const popops::RegionReduction &r) {
          return r.output.numElements() == 1;
        });
    if (allOutputRegionsOfSizeOne) {
      specialisation = ReductionSpecialisation::SCALAR_OUTPUT_REGIONS;
    }
  }
  return specialisation;
}

} // namespace popops
