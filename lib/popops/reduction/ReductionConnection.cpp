#include "ReductionConnection.hpp"

#include <boost/optional.hpp>
#include <boost/range/algorithm/transform.hpp>

#include <poputil/Util.hpp>
#include <poputil/VertexTemplates.hpp>
#include <poputil/exceptions.hpp>
#include <popops/Reduce.hpp>
#include <poplibs_support/Compiler.hpp>

#include "CycleEstimationFunctions.hpp"
#include "ReductionVertex.hpp"

namespace popops {

namespace {

// Divide a by b, rounding up.
template <typename T>
T udiv(T a, T b) {
  return ((a + b) - 1) / b;
};

// Return the approximate number of operations per cycle for the given
// type and operation. This doesn't account for type conversion, scale or
// update.
double opsPerCycle(unsigned outputSize,
                   const poplar::Target &target,
                   poplar::Type type,
                   popops::Operation operation) {
  if (outputSize == 0)
    return 1;

  // Very roughly, the number of elements we can process per cycle is as
  // follows.
  //
  // In theory we can do better in some cases with complex optimisations.
  //
  // f16 add, square-add, abs-add:
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
  // f32 add, square-add, abs-add:
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

  // Add, square-add and abs-add can be done twice as fast for floats.
  if ((operation == popops::Operation::ADD ||
       operation == popops::Operation::SQUARE_ADD) // Or ABS_ADD
      && (type == poplar::HALF || type == poplar::FLOAT))
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

std::uint64_t approximateCyclesForReduction(const poplar::Target &target,
                                            popops::Operation operation,
                                            const RegionReduction &reduction) {
  unsigned totalPartialElements = 0;
  for (const auto &p : reduction.partials)
    totalPartialElements += p.numElements();

  double cyclesPerOp = 1.0 / opsPerCycle(reduction.output.numElements(),
                                         target,
                                         reduction.output.elementType(),
                                         operation);

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
    const poplar::Target &target,
    popops::Operation operation,
    const std::vector<RegionReduction> &reductions) {

  // If there are more or equal output regions than numWorkers we
  // distribute them as evenly as possible based on the number of partials
  // so each vertex has a similar number of partials.

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

  for (const auto &r : reductions) {
    reductionCycles.emplace_back(
          reductionCycles.size(),
          approximateCyclesForReduction(target, operation, r)
        );
  }

  // Optimisation: There is definitely scope for further optimisation here.
  // Distributing work evenly is almost NP-complete, but there are apparently
  // efficient algorithms. See https://en.wikipedia.org/wiki/Partition_problem
  // and especially Multi-Way Number Partitioning by Richard Korf:
  // http://www.ijcai.org/Proceedings/09/Papers/096.pdf

  // For now, we just sort them from most cycles to least, and greedily
  // assign them to the vertex with the fewest cycles.

  std::sort(reductionCycles.begin(),
            reductionCycles.end(),
            [](std::pair<unsigned, std::uint64_t> a,
               std::pair<unsigned, std::uint64_t> b) {
    return a.second < b.second;
  });

  std::vector<std::uint64_t> cyclesPerWorker;
  cyclesPerWorker.reserve(target.getNumWorkerContexts());

  for (auto c : reductionCycles) {
    auto reductionIndex = c.first;
    auto cycles = c.second;

    if (cyclesPerWorker.size() < target.getNumWorkerContexts()) {

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
  std::size_t totalPartialElements = 0;
  for (const auto &partial : reduction.partials)
    totalPartialElements += partial.numElements();
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
  return std::min(static_cast<std::uint64_t>(rows/2), cycles/minCycles);
}

// Split the RegionReductions into smaller chunks. This can be for two reasons:
//  - to make more work to utilise all of the workers available
//  - to make the vertex state values fit into their respective types.
// The return value is the number of pieces each reduction is split up into
// (minimum 1) for each reduction.
std::vector<unsigned> splitTwoStageReductionsBetweenWorkers(
    const poplar::Target &target,
    popops::Operation operation,
    const std::vector<RegionReduction> &reductions,
    const std::size_t vectorListMaxSize) {
  // initialise the number of splits needed for each reduction by how many times
  // it would be needed for it to fit into a DeltaN VectorList.
  std::vector<unsigned> splits;
  splits.reserve(reductions.size());

  const auto out = std::back_inserter(splits);
  boost::transform(reductions, out, [&](const RegionReduction &reduction) {
    return udiv(reduction.partials.size(), vectorListMaxSize);
  });

  std::vector<std::uint64_t> approxCycleCounts(reductions.size());
  for (std::size_t i = 0; i < reductions.size(); ++i) {
    approxCycleCounts[i] =
          approximateCyclesForReduction(target, operation, reductions[i]);
  }

  // First work out the maximum number of splits for each worker. It never
  // makes sense to split to less than 2 rows for each piece, and in some
  // cases the limit is more if there aren't many output values.

  std::vector<unsigned> maxSplit(reductions.size(), 1);
  for (unsigned i = 0; i < reductions.size(); ++i) {
    maxSplit[i] = getMaxSplit(reductionFactor(reductions[i]),
                              approxCycleCounts[i]);
  }

  unsigned freeWorkers = target.getNumWorkerContexts() - reductions.size();

  while (freeWorkers > 0) {
    boost::optional<unsigned> toSplit;
    std::size_t highestCyclesAfterSplit = 0;

    for (unsigned i = 0; i < splits.size(); ++i) {
      // Ignore this if it doesn't want to be split any more.
      if (splits[i] + 1 >= maxSplit[i])
        continue;

      // Work out the rough number of cycles if it would be split more.
      auto cyclesAfterSplit = approxCycleCounts[i] / (splits[i] + 1);
      if (cyclesAfterSplit > highestCyclesAfterSplit) {
        highestCyclesAfterSplit = cyclesAfterSplit;
        toSplit = i;
      }
    }

    // Check if we don't want to split any more.
    if (!toSplit)
      break;

    ++splits[toSplit.get()];
    --freeWorkers;
  }

  return splits;
}


// Connect the input and output regions for the vertex.
void connectVertexEdges(poplar::Graph &graph,
                        const std::vector<RegionReduction> &reductions,
                        poplar::VertexRef &vertex) {

  // Number of output regions for this vertex.
  auto numOutputRegions = reductions.size();

  if (numOutputRegions < 1)
    throw poputil::poplibs_error("no output regions in reduction");

  // Work out the total number of partial regions.
  unsigned numPartialRegions = 0;

  // Number of input partial regions for each output region, start with 0.
  std::vector<unsigned short> numPartials;
  numPartials.reserve(numOutputRegions);

  for (const auto &r : reductions) {
    auto sz = r.partials.size();

    if (sz < 1) {
      throw poputil::poplibs_error("output region with no partials");
    }
    if (sz > std::numeric_limits<unsigned short>::max()) {
      // As total memory on Colossus B0 is 2**18, 2**16 * num_workers
      // assuming that the work is split across workers
      // would occupy more memory than we have. If work is not split across
      // workers, then if partials[i].size() < 4 for all reductions
      // could hit this limit.
      // Come MK2 may have to deal with num partials greater than this
      // and create more vertices
      throw poputil::poplibs_error("Number of partials larger than short");
    }
    numPartialRegions += sz;
    numPartials.push_back(static_cast<unsigned short>(sz));
  }

  auto t = graph.addConstant(poplar::UNSIGNED_SHORT, {numPartials.size()},
                             numPartials.data());
  graph.setTileMapping(t, 0);
  graph.connect(vertex["numPartials"], t);

  std::vector<poplar::Tensor> partials;
  partials.reserve(numPartialRegions);
  std::vector<poplar::Tensor> outputs;
  outputs.reserve(numOutputRegions);

  for (const auto &r : reductions) {
    outputs.emplace_back(r.output);

    for (const auto &partial : r.partials) {
      partials.emplace_back(partial);
    }
  }

  graph.connect(vertex["out"], outputs);
  graph.connect(vertex["partials"], partials);
}

// Split `rows` into up to N groups with a minimum of 2 rows per group.
// If possible the number of groups is miminised without increasing the
// maximum number of rows in each group. For example with rows=9, N=4
// the output is {3,3,3} rather than {3,2,2,2}.
std::vector<unsigned> splitRowsToWorkers(unsigned rows, unsigned N) {
  if (rows <= 3 || N <= 1)
    return {rows};

  auto maxRowsPerWorker = udiv(rows, N);

  auto maxWorkers = rows / 2;
  auto numWorkers = std::min(maxWorkers, udiv(rows, maxRowsPerWorker));

  std::vector<unsigned> split(numWorkers, rows/numWorkers);
  auto plusOnes = rows % numWorkers;
  for (unsigned i = 0; i < plusOnes; ++i)
    ++split[i];
  return split;
}

// Every tensor in `partials` is a 1D tensor with a length that is a multiple
// of `outputSize`. Imagine the partials are all concatenated and then wrapped
// to make an N * `outputSize` 2D tensor. This function splits that tensor
// up by the row (according to the row counts in `rows) and returns
// a set of 1D row tensors for each chunk of rows.
//
// The output is not set in the returned values. You have to do that yourself.
std::vector<RegionReduction> splitPartialsByRows(
    const RegionReduction &reduction,
    const std::vector<unsigned> &rows) {

  assert(reduction.partialsDebugInfo.size() == reduction.partials.size());

  std::vector<RegionReduction> out(rows.size());

  const auto outputSize = reduction.output.numElements();

  // Get the number of rows in each partial.
  std::vector<std::size_t> rowsInPartials(reduction.partials.size());
  for (std::size_t i = 0; i < rowsInPartials.size(); ++i)
    rowsInPartials[i] = reduction.partials[i].numElements() / outputSize;

  assert(std::accumulate(rowsInPartials.begin(), rowsInPartials.end(), 0) ==
         std::accumulate(rows.begin(), rows.end(), 0));

  std::size_t currentPartial = 0;
  std::size_t rowBegin = 0;

  for (unsigned i = 0; i < out.size(); ++i) {
    // Get `rows[i]` rows from the reduction and put them in out[i].

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
        out[i].partials.push_back(reduction.partials[currentPartial]);
        out[i].partialsDebugInfo.push_back(
              reduction.partialsDebugInfo[currentPartial]);
      } else {
        // Take a slice.
        out[i].partials.push_back(
              reduction.partials[currentPartial].flatten().slice(
                rowBegin * outputSize,
                rowEnd * outputSize
              )
            );

        // Debug info
        auto di = reduction.partialsDebugInfo[currentPartial];
        if (di.sourceCols.size() == outputSize) {
          di.sourceRows = {di.sourceRows.begin() + rowBegin,
                           di.sourceRows.begin() + rowEnd};
        } else {
          di.sourceRows = {0, 1};
          di.sourceCols = {rowBegin * outputSize, rowEnd * outputSize};
        }
        out[i].partialsDebugInfo.push_back(di);
      }

      rowBegin = rowEnd;
      if (rowBegin >= rowsInCurrentPartial) {
        rowBegin = 0;
        ++currentPartial;
      }
    }
  }
  return out;
}

// Call this function to connect reductions where none of them are two-stage
// reductions and there may be more reductions than worker contexts.
// The `assignments` should be the same size as `reductions` and
// contains the worker context/vertex each reduction should be done by.
void connectSingleStageReductions(
    poplar::Graph &graph,
    poplar::ComputeSet &cs,
    ReduceParams params,
    poplar::Type partialType,
    poplar::Type outputType,
    unsigned tile,
    const std::vector<RegionReduction> &reductions,
    const std::vector<unsigned> &assignments,
    ReductionDebug::TileReduction *tileDebug) {

  assert(reductions.size() == assignments.size());

  // Map from worker context to list of reductions.
  // Optimisation: Copying the RegionReductions around could be avoided.
  std::map<unsigned, std::vector<RegionReduction>> reductionsPerWorker;

  for (std::size_t i = 0; i < assignments.size(); ++i)
    reductionsPerWorker[assignments[i]].emplace_back(reductions[i]);

  assert(reductionsPerWorker.size()
         <= graph.getTarget().getNumWorkerContexts());

  // The name of the vertex to use.
  std::string vertexName =
      getReductionVertexName(params, partialType, outputType);

  // Connect the single stage reductions.
  for (const auto &it : reductionsPerWorker) {
    const auto &vertexReductions = it.second;

    // Add a vertex.
    auto vertex = graph.addVertex(cs, vertexName);

    // Map it to this tile.
    graph.setTileMapping(vertex, tile);

    // This field is present even in codelets that don't use it at the moment.
    graph.setInitialValue(vertex["k"], params.scale);

    // Connect its inputs and outputs.
    connectVertexEdges(graph, vertexReductions, vertex);

    if (tileDebug != nullptr) {
      for (const auto &reduction : vertexReductions) {
        ReductionDebug::RegionReduction rr;
        rr.vertex = it.first;
        rr.output = reduction.outputDebugInfo;
        rr.partials = reduction.partialsDebugInfo;
        tileDebug->firstStage.singleStageRegions.push_back(rr);
      }
    }
  }
}

// This is called when the number of reductions is less than or equal to
// the number of workers, and some of them *may* be split. Despite the name
// there may actually be no two-stage reductions, or there may be a mix,
// or there may be no single-stage reductions.
//
// `splits` is the number of pieces to split each reduction into (if 1 it
// means it is a single-stage reduction).
void connectTwoStageReductions(poplar::Graph &graph,
                               ComputeSetList &css,
                               ReduceParams params,
                               poplar::Type partialType,
                               poplar::Type outputType,
                               unsigned tile,
                               const std::vector<RegionReduction> &reductions,
                               const std::vector<unsigned> &splits,
                               const std::string &debugPrefix,
                               ReductionDebug::TileReduction *tileDebug) {
  // Triple check...
  assert(splits.size() == reductions.size());

  std::vector<RegionReduction> singleStageReductions;
  std::vector<unsigned> singleStageAssignments;

  for (std::size_t i = 0; i < splits.size(); ++i) {
    if (splits[i] == 1) {
      singleStageReductions.push_back(reductions[i]);
      singleStageAssignments.push_back(singleStageAssignments.size());
    }
  }

  auto firstCs = css.add(graph, debugPrefix + "/Reduce");

  if (!singleStageReductions.empty()) {

    connectSingleStageReductions(graph,
                                 firstCs,
                                 params,
                                 partialType,
                                 outputType,
                                 tile,
                                 singleStageReductions,
                                 singleStageAssignments,
                                 tileDebug);

  }

  // If there are no two-stage reductions, that's it!
  if (singleStageReductions.size() == reductions.size())
    return;

  // The name of the vertex to use. Don't do scale or update in the first
  // stage.
  std::string firstStageVertexName =
      getReductionVertexName({params.op}, partialType, outputType);

  // Map from reduction number to the partial for second stage reductions.
  std::map<unsigned, poplar::Tensor> secondStagePartials;

  unsigned currentVertex = singleStageReductions.size();

  // Now connect the first stage of the two-stage reductions.
  for (std::size_t i = 0; i < splits.size(); ++i) {
    // Ignore the single-stage ones we've already done.
    if (splits[i] <= 1)
      continue;

    // Optimisation: In some cases we'd want to do a two stage reduction anyway
    // e.g. if there are 6 reductions with reduction factors
    //   x1000, x5, x5, x5, x5, x5.

    auto outputSize = reductions[i].output.numElements();

    auto totalRows = reductionFactor(reductions[i]);

    auto rowsPerWorker = splitRowsToWorkers(
                           totalRows,
                           splits[i]
                         );

    assert(!rowsPerWorker.empty());
    assert(rowsPerWorker.size() <= splits[i]);
    assert(std::accumulate(rowsPerWorker.begin(),
                           rowsPerWorker.end(), 0u) == totalRows);

    auto partialsPerWorker = splitPartialsByRows(reductions[i],
                                                 rowsPerWorker);

    assert(partialsPerWorker.size() == rowsPerWorker.size());

    // Create a tensor for all the partial results.
    secondStagePartials[i] = graph.addVariable(
                               outputType,
                               {outputSize * partialsPerWorker.size()},
                               debugPrefix + "/secondStagePartials"
                             );
    graph.setTileMapping(secondStagePartials[i], tile);

    // Now create the new RegionReductions.
    for (unsigned s = 0; s < rowsPerWorker.size(); ++s) {
      RegionReduction firstStage;
      firstStage.output = secondStagePartials[i].slice(s * outputSize,
                                                       (s+1) * outputSize);
      firstStage.partials = partialsPerWorker[s].partials;

      // Add a vertex for that reduction.

      // Add a vertex.
      auto vertex = graph.addVertex(firstCs, firstStageVertexName);

      // Map it to this tile.
      graph.setTileMapping(vertex, tile);

      // Don't scale! Although this field should be unused anyway.
      graph.setInitialValue(vertex["k"], 1.0f);

      // Connect its inputs and outputs.
      connectVertexEdges(graph, {firstStage}, vertex);

      // Debug info.
      if (tileDebug != nullptr) {
        ReductionDebug::RegionReduction rr;
        rr.vertex = currentVertex++;
        rr.output.outputRegion = reductions[i].outputDebugInfo.outputRegion;
        rr.output.dataRegion = {s * outputSize, (s+1) * outputSize};
        rr.partials = partialsPerWorker[s].partialsDebugInfo;

        tileDebug->firstStage.firstStageRegions.push_back(rr);
      }
    }
  }

  // And the second stage of the two-stage reductions.

  // Add a second compute set if needed.
  auto secondCs = css.add(graph, debugPrefix + "/Reduce_Second_Stage");

  // Work out which vertex should do each second stage. We just assign
  // in a round-robin manner since there shouldn't really ever be more than
  // the number of worker contexts anyway.

  std::vector<std::vector<RegionReduction>> secondStageReductions;
  secondStageReductions.reserve(graph.getTarget().getNumWorkerContexts());

  // Don't square again in the second stage.
  if (params.op == Operation::SQUARE_ADD)
    params.op = Operation::ADD;

  std::string secondStageVertexName =
      getReductionVertexName(params, outputType, outputType);

  currentVertex = 0;

  std::size_t partialColsHighlightOffset = 0;

  for (const auto &r : secondStagePartials) {
    assert(r.first >= 0 && r.first < reductions.size());


    RegionReduction secondStageReduction;
    secondStageReduction.output = reductions[r.first].output;
    // There's only ever one partial for each second stage reduction.
    secondStageReduction.partials.emplace_back(r.second);

    // Add a vertex to the second compute set.
    auto vertex = graph.addVertex(secondCs, secondStageVertexName);

    // Map it to this tile.
    graph.setTileMapping(vertex, tile);

    // Set the scale if needed.
    graph.setInitialValue(vertex["k"], params.scale);

    // Connect its inputs and outputs.
    connectVertexEdges(graph, {secondStageReduction}, vertex);

    // Debug info
    if (tileDebug != nullptr) {
      ReductionDebug::RegionReduction rr;
      rr.vertex = currentVertex++;
      rr.output = reductions[r.first].outputDebugInfo;

      ReductionDebug::Partial p;
      p.sourceTile = tile;
      p.sourceRows = {0, 1};
      // This is not quite true but it makes the visualisation work.
      // Basically each region reduction gets its own intermediate
      // partial, and they are all drawn next to each other with
      // the same highlight div in the visualisation, so the sourceCols
      // here are relative to the first one.
      p.sourceCols = {partialColsHighlightOffset,
                      partialColsHighlightOffset + r.second.numElements()};
      partialColsHighlightOffset += r.second.numElements();
      rr.partials.push_back(p);

      tileDebug->secondStage.secondStageRegions.push_back(rr);
    }
  }
}


} // anonymous namespace

void connectReductions(poplar::Graph &graph,
                       ComputeSetList &css,
                       ReduceParams params,
                       poplar::Type partialType,
                       poplar::Type outputType,
                       unsigned tile,
                       const std::vector<RegionReduction> &reductions,
                       const std::string &debugPrefix,
                       ReductionDebug::TileReduction *tileDebug) {

  const auto &target = graph.getTarget();

  // Optimisation: If there is just one partial for an output we don't need to
  // calculate it, but hooking that up is a bit fiddly.

  // Check that all the inputs and outputs are of the appropriate sizes
  // and types.
  for (const auto &r : reductions) {

    auto outputSize = r.output.numElements();
    if (outputSize == 0)
      throw poputil::poplibs_error("Zero-sized reduction output");

    if (r.output.elementType() != outputType)
      throw poputil::poplibs_error("Reduction output is incorrect type");

    for (const auto &p : r.partials) {
      if (p.numElements() == 0)
        throw poputil::poplibs_error("Zero-sized reduction partial");
      if (p.numElements() % outputSize != 0)
        throw poputil::poplibs_error("Reduction partial size is not a multiple "
                                    "of the output size");

      if (p.elementType() != partialType)
        throw poputil::poplibs_error("Reduction partial is incorrect type");
    }
  }

  const std::size_t vectorListMaxSize = [&] {
    const auto vertex = getReductionVertexName(params, partialType, outputType);
    return graph.getMaxFieldDim(vertex, "partials", 0);
  }();

  // reductions contains a list of ReductionRegions, if any of the regions
  // contain more partials than we can fit into a DeltaN list we must split
  // those regions up.
  const auto partialsTooLarge = [&](const RegionReduction &reduction) {
    return reduction.partials.size() >= vectorListMaxSize;
  };

  const bool useTwoStageReduction =
    reductions.size() < target.getNumWorkerContexts() ||
    std::any_of(std::begin(reductions), std::end(reductions), partialsTooLarge);

  // See if there is the possibility of easily splitting reductions
  // into two-level ones.
  if (useTwoStageReduction) {
    // Try to split some into multi-stage reductions.
    auto splits = splitTwoStageReductionsBetweenWorkers(
                    target, params.op, reductions, vectorListMaxSize);

    connectTwoStageReductions(graph,
                              css,
                              params,
                              partialType,
                              outputType,
                              tile,
                              reductions,
                              splits,
                              debugPrefix,
                              tileDebug);


  } else {
    // Distribute the reductionTensors among 6 (or whatever) vertices.
    auto reductionAssignments = distributeReductionsBetweenWorkers(
        target, params.op, reductions);

    // We need at least one compute set for the single stage reduction.
    auto cs = css.add(graph, debugPrefix + "/Reduce");

    connectSingleStageReductions(graph,
                                 cs,
                                 params,
                                 partialType,
                                 outputType,
                                 tile,
                                 reductions,
                                 reductionAssignments,
                                 tileDebug);
  }

  if (tileDebug != nullptr)
    tileDebug->tileIndex = tile;

}

}
