// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/Embedding.hpp"
#include "FullyConnectedPNMapping.hpp"
#include "FullyConnectedPlan.hpp"
#include "FullyConnectedUtils.hpp"
#include "FullyConnectedVector.hpp"
#include "MatMulUtils.hpp"
#include "SparseCodeletMetaInfoScale.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/Zero.hpp"
#include "popsparse/MatMul.hpp"
#include "popsparse/SparsePartitioner.hpp"
#include "poputil/TileMapping.hpp"
#include <popops/Expr.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/VertexTemplates.hpp>

#include <boost/optional.hpp>

#include <variant>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;
using namespace popsparse;
using namespace popsparse::dynamic;
using namespace popops::expr;
using namespace popops;

namespace popsparse {
namespace dynamic {
namespace {

constexpr std::size_t minIndicesPerTile = 32;

struct PlannedSplits {
  // Information about the plan for memory layout for the sparse input tensor.
  // this dictates the plan for the slice operation.
  std::size_t rows;
  std::size_t splitRows;
  unsigned rowSplits;
  // Both the sparse input and the sliced result tensor are split in the same
  // way in this dimension
  std::size_t columns;
  std::size_t splitColumns;
  unsigned columnSplits;
  // The division of data the matmul planned in the z-dimension
  // (result operand columns)
  std::size_t z;
  std::size_t splitZ;
  unsigned zSplits;

  // The mapping order of the matmul plan
  fullyconnected::PartitionToPNMapping mappingOrder;
  unsigned groups;

  // Plan for the split of the sliced result tensor (columns split as above)
  std::size_t slicesSplitRows;
  unsigned slicesRowSplits;
  std::size_t slicesSplitIndices;
  unsigned slicesIndicesSplits;

  PlannedSplits(std::size_t numSlices, const FullyConnectedParams &params,
                const fullyconnected::Plan &plan,
                boost::optional<std::size_t> plannedColumns = boost::none) {
    rows = params.getOutputChannelsPerGroup();
    if (plannedColumns) {
      columns = plannedColumns.get();
    } else {
      columns = params.getInputChannelsPerGroup();
    }
    z = params.getBatchSize();
    // Just note the splits from the plan, and find the number of elements
    rowSplits = plan.partition.x;
    splitRows = (rows + rowSplits - 1) / rowSplits;

    columnSplits = plan.partition.y;
    splitColumns = (columns + columnSplits - 1) / columnSplits;

    zSplits = plan.partition.z;
    splitZ = (z + zSplits - 1) / zSplits;

    groups = plan.partition.groups;
    // Extract the mapping order
    mappingOrder = plan.exchangePlan.fwdMapping;

    // Decide how to split the slice result?
    // 1. Split by column in the same way as the input.
    // 2. The "Z-dim" of the input will have all NZ&metaData broadcast to
    //    all other tiles in that dimension, so next split the indices over
    //    that dimension.  This means that when slicing there are fewer indices
    //    per tile. (This is splitting by row but with the advantage described)
    // 3. Finally split by row.  Anything split in this last way means that
    //    the indices and slices will require a second stage as indices are
    //    checked for on multiple candidate tiles.
    // Named:
    // 1. columnSplits,
    // 2. slicesIndicesSplits
    // 3. slicesRowSplits

    // Split rows over the z-dimension
    slicesSplitIndices = (numSlices + zSplits - 1) / zSplits;
    slicesIndicesSplits =
        (numSlices + slicesSplitIndices - 1) / slicesSplitIndices;
    // Split the remaining rows
    slicesSplitRows = (slicesSplitIndices + rowSplits - 1) / rowSplits;
    slicesRowSplits =
        (slicesSplitIndices + slicesSplitRows - 1) / slicesSplitRows;

    if (logging::popsparse::shouldLog(logging::Level::Debug)) {
      logging::popsparse::debug("Sparse Dense Embedding plan with:\n"
                                "baseT rows:{} rowSplits:{} rowsPerSplit:{}",
                                rows, rowSplits, splitRows);
      logging::popsparse::debug(
          "baseT columns:{} columnSplits:{} columnsPerSplit:{}", columns,
          columnSplits, splitColumns);
      logging::popsparse::debug("baseT z:{} zSplits:{} zPerSplit:{}", z,
                                zSplits, splitZ);
      logging::popsparse::debug("Slices rows:{} rowSplits:{} rowsPerSplit:{}",
                                numSlices, slicesRowSplits, slicesSplitRows);
      logging::popsparse::debug(
          "Slices columns:{} columnSplits:{} columnsPerSplit:{}", columns,
          columnSplits, splitColumns);
      logging::popsparse::debug("Slices zSplits:{} zPerSplit:{}",
                                slicesIndicesSplits, slicesSplitIndices);
    }
  }

  fullyconnected::Vector<unsigned> getPartitions(void) const {
    return {groups, rowSplits, columnSplits, zSplits};
  }
};
// Find the range of columns for a given column split
Interval columnRange(unsigned cs, const PlannedSplits &plannedSplits,
                     std::size_t columns) {
  const auto columnStart = cs * plannedSplits.splitColumns;
  const auto columnEnd =
      std::min((cs + 1) * plannedSplits.splitColumns, columns);
  return {columnStart, columnEnd};
}

// Find the range of rows for a given row split
Interval rowRange(unsigned rs, unsigned is, unsigned rowSplits,
                  unsigned splitRows, unsigned numIndices) {
  const auto rowGroup = rs + is * rowSplits;

  const auto rowStart = std::min(rowGroup * splitRows, numIndices);
  const auto rowEnd = std::min((rowGroup + 1) * splitRows, numIndices);
  return {rowStart, rowEnd};
}

// The existing multiSlice vertices are used in the second stage of the
// embedding slice.  This is a copy operation of dense to dense data
// (normal tensors) using modified indices to access data from each of a group
// of tiles where the input to the first stage was split by row, BUT NOT over
// the Z-split dimension, as the sparse data would be broadcast over that
// dimension before slicing began
void generateDenseDenseMultiSliceVertices(Graph &graph, Sequence &prog,
                                          const Tensor &offsets,
                                          const Tensor &base, Tensor &slices,
                                          const PlannedSplits &plannedSplits,
                                          const std::string &debugStr) {

  auto computeSet = graph.addComputeSet(debugStr);
  const auto inputType = base.elementType();
  const auto &target = graph.getTarget();
  const auto numWorkers = target.getNumWorkerContexts();
  const auto numSlices = slices.dim(0) / plannedSplits.slicesIndicesSplits;
  const auto rowsPerGroup = plannedSplits.rowSplits * numSlices;
  const auto offsetsPerThread =
      (plannedSplits.slicesSplitRows + numWorkers - 1) / numWorkers;

  const auto vertexClass = templateVertex("popops::MultiSlice", inputType);

  for (unsigned cs = 0; cs < plannedSplits.columnSplits; cs++) {
    for (unsigned is = 0; is < plannedSplits.slicesIndicesSplits; is++) {
      for (unsigned rs = 0; rs < plannedSplits.slicesRowSplits; rs++) {
        const auto tile = plannedSplits.mappingOrder.getPNIdForPartition(
            plannedSplits.getPartitions(), {0, rs, cs, is});
        // Find the range of columns in the result found on this tile, the last
        // split tile can have a truncated result if columns didn't divide
        // exactly
        const auto columns = columnRange(cs, plannedSplits, slices.dim(1));

        // Slice the partials result tensors from source tiles that are relevant
        // to this tile.
        std::vector<Tensor> baseT(plannedSplits.rowSplits);
        const auto rowBase =
            rs * plannedSplits.slicesSplitRows + is * rowsPerGroup;

        for (unsigned st = 0; st < plannedSplits.rowSplits; st++) {
          const auto rowStart = rowBase + st * numSlices;
          const auto rowEnd = rowStart + plannedSplits.slicesSplitRows;
          baseT[st] =
              base.slice({rowStart, columns.begin()}, {rowEnd, columns.end()})
                  .flatten();
        }

        const auto index = rs + is * plannedSplits.slicesRowSplits;
        const auto tileOffsets =
            offsets[index].slice(0, plannedSplits.slicesSplitRows);

        const auto subTRows =
            rowRange(rs, is, plannedSplits.slicesRowSplits,
                     plannedSplits.slicesSplitRows, slices.dim(0));

        auto tileSubT = slices.slice({subTRows.begin(), columns.begin()},
                                     {subTRows.end(), columns.end()});
        tileSubT = tileSubT.reshape(
            {tileSubT.numElements() / columns.size(), columns.size()});

        // Split the indices between workers and generate vertices
        for (unsigned worker = 0; worker < numWorkers; worker++) {
          auto workerRowStart = offsetsPerThread * worker;
          if (workerRowStart >= tileSubT.dim(0)) {
            break;
          }
          auto workerRowEnd =
              std::min(workerRowStart + offsetsPerThread, tileSubT.dim(0));

          const auto vertex = graph.addVertex(computeSet, vertexClass);
          graph.setTileMapping(vertex, tile);

          graph.connect(vertex["baseT"], concat(baseT).flatten());

          auto workerOffsets = tileOffsets.slice(workerRowStart, workerRowEnd);
          graph.connect(vertex["offsets"], workerOffsets);

          auto workerSubT = tileSubT.slice(workerRowStart, workerRowEnd);
          graph.connect(vertex["subT"], workerSubT.flatten());

          graph.setInitialValue(vertex["baseOffset"], 0);
          graph.setInitialValue(vertex["numBaseElements"],
                                plannedSplits.rowSplits *
                                    plannedSplits.slicesSplitRows);
          graph.setInitialValue(vertex["regionSize"], columns.size());
        }
      }
    }
  }
  prog.add(Execute(computeSet));
}

void createSparseDenseTileVertices(
    Graph &graph, ComputeSet &computeSet, const std::string &vertexClass,
    unsigned tile, const Tensor &offsets, const std::vector<Tensor> &baseTNZ,
    const std::vector<Tensor> &baseTMeta, const Tensor &subT, unsigned columns,
    unsigned rowStart, unsigned rowsPerPartition, unsigned id, unsigned splitZ,
    const boost::optional<Tensor> &scale,
    const boost::optional<Tensor> &yPartitionToProcess) {
  // TODO - split the indices between workers (At least for slice)
  // Or do we use the worker information in the metainformation?
  // An issue with this is sub-word writes for the "half" case.
  // If indices are split between workers (As a worker vertex) then each
  // worker's region is "safe" and it can sub-word write freely.
  // If a supervisor vertex we could align subT and share work by index: with
  // an even number of indices per worker. Or even connect to "workers"
  // subT edges.
  //
  // For update we'll be writing into a contiguous area of NZValues with
  // updates based on the indices and metadata.  There seems no way to
  // avoid sub word writes clashing between workers. The solution will
  // be to cast the NZ values to float, update and cast back.
  const auto vertex = graph.addVertex(computeSet, vertexClass);
  graph.setTileMapping(vertex, tile);

  graph.connect(vertex["offsets"], offsets.flatten());
  graph.connect(vertex["subT"], subT);

  graph.connect(vertex["baseTMetaInfo"], baseTMeta);
  graph.connect(vertex["baseTNZ"], baseTNZ);

  graph.setInitialValue(vertex["subColumns"], columns);
  graph.setInitialValue(vertex["nzScaleFactor"], reciprocalMulFactor(splitZ));
  if (scale) {
    graph.setInitialValue(vertex["rowsPerPartition"], rowsPerPartition);
    graph.connect(vertex["yPartitionToProcess"], yPartitionToProcess.get());
    graph.connect(vertex["scale"], scale.get().reshape({}));
  } else {
    graph.setInitialValue(vertex["rowOffset"], rowStart);
    graph.setInitialValue(vertex["subGroupIdToProcess"], id);
  }
}

void generateSparseDenseMultiSliceVertices(
    Graph &graph, Sequence &prog, const Tensor &offsets,
    const SparseTensor &base, const std::vector<std::size_t> &tShape,
    const Tensor &slices, const PlannedSplits &plannedSplits,
    const std::string &debugStr) {

  auto computeSet = graph.addComputeSet(debugStr);
  const auto inputType = base.getNzValuesTensor().elementType();
  const auto vertexClass =
      templateVertex("popsparse::SparseDenseMultiSliceElementWise", inputType);
  const auto paddedSlicesRows = slices.dim(0) /
                                plannedSplits.slicesIndicesSplits /
                                plannedSplits.rowSplits;

  logging::popsparse::debug("creating {} vertices", vertexClass);

  auto metaInfoBuckets = fullyconnected::getBucketsByPartition(
      base.getMetaInfoTensor(), plannedSplits.getPartitions());
  auto nzBuckets = fullyconnected::getBucketsByPartition(
      base.getNzValuesTensor(), plannedSplits.getPartitions());

  const auto splitRows = slices.dim(0) / (plannedSplits.rowSplits *
                                          plannedSplits.slicesIndicesSplits);

  for (unsigned cs = 0; cs < plannedSplits.columnSplits; cs++) {
    for (unsigned rs = 0; rs < plannedSplits.rowSplits; rs++) {
      const auto rowStart = rs * plannedSplits.splitRows;

      const auto columns = columnRange(cs, plannedSplits, tShape[1]);

      // Gather the sparse NZ & Metadata over the set of tiles in the z-split
      // group that relates to the slices on this tile
      std::vector<Tensor> baseTNZ(plannedSplits.zSplits);
      std::vector<Tensor> baseTMeta(plannedSplits.zSplits);
      for (unsigned zSplit = 0; zSplit < plannedSplits.zSplits; zSplit++) {
        baseTNZ[zSplit] = nzBuckets[0][rs][cs][zSplit].flatten();
        baseTMeta[zSplit] = metaInfoBuckets[0][rs][cs][zSplit].flatten();
      }
      // Create a vertex on each tile in the indices split group.  Each
      // operates on a subset of the indices, but has the whole z-group data
      // available to it (We broadcast rows of baseT to a group of tiles
      // and spread the indices over those tiles)
      const auto tileID = fullyconnected::calculateSubGroupId(
          plannedSplits.rowSplits, plannedSplits.columnSplits, rs, cs);
      for (unsigned is = 0; is < plannedSplits.slicesIndicesSplits; is++) {
        const auto iTile = plannedSplits.mappingOrder.getPNIdForPartition(
            plannedSplits.getPartitions(), {0, rs, cs, is});

        auto rows =
            rowRange(rs, is, plannedSplits.rowSplits, splitRows, slices.dim(0));
        auto subT = slices.slice({rows.begin(), columns.begin()},
                                 {rows.end(), columns.end()});

        const auto offsetsStart = is * paddedSlicesRows;
        if (offsetsStart >= offsets.numElements()) {
          break;
        }
        const auto offsetsEnd =
            std::min((is + 1) * paddedSlicesRows, offsets.numElements());
        auto offsetsSlice = offsets.slice(offsetsStart, offsetsEnd);

        createSparseDenseTileVertices(
            graph, computeSet, vertexClass, iTile, offsetsSlice, baseTNZ,
            baseTMeta, subT.flatten(), columns.size(), rowStart,
            plannedSplits.splitRows, tileID, plannedSplits.splitZ, boost::none,
            boost::none);
      }
    }
  }
  prog.add(Execute(computeSet));
}

void generateSparseDenseMultiUpdateVertices(
    Graph &graph, Sequence &prog, const Tensor &offsets,
    const Tensor &columnPartition, const SparseTensor &base,
    const std::vector<std::size_t> &tShape, const Tensor &slices,
    const Tensor &scale, const PlannedSplits &plannedSplits,
    const std::string &debugStr) {

  auto computeSet = graph.addComputeSet(debugStr);
  const auto inputType = base.getNzValuesTensor().elementType();
  const auto vertexClass = templateVertex(
      "popsparse::SparseDenseMultiUpdateAddElementWise", inputType);

  logging::popsparse::debug("creating {} vertices", vertexClass);

  auto metaInfoBuckets = fullyconnected::getBucketsByPartition(
      base.getMetaInfoTensor(), plannedSplits.getPartitions());
  auto nzBuckets = fullyconnected::getBucketsByPartition(
      base.getNzValuesTensor(), plannedSplits.getPartitions());

  // Make a plan variable that represents the layout, but with only 1 column
  // per partition
  auto offsetsSplits = plannedSplits;
  offsetsSplits.splitColumns = 1;
  offsetsSplits.columns = offsetsSplits.columnSplits;

  for (unsigned cs = 0; cs < plannedSplits.columnSplits; cs++) {
    for (unsigned rs = 0; rs < plannedSplits.rowSplits; rs++) {
      const auto rowStart = rs * plannedSplits.splitRows;
      const auto columns = columnRange(cs, plannedSplits, tShape[1]);
      const auto offsetsColumns =
          columnRange(cs, offsetsSplits, offsetsSplits.columnSplits);

      // Create vertices on each tile in the Z-split to udpate the portion
      // of baseT that it holds. All slices are used, split in the
      // columns dimension
      for (unsigned zSplit = 0; zSplit < plannedSplits.zSplits; zSplit++) {
        auto zTile = plannedSplits.mappingOrder.getPNIdForPartition(
            plannedSplits.getPartitions(), {0, rs, cs, zSplit});

        auto baseTNZ = nzBuckets[0][rs][cs][zSplit].flatten();
        auto baseTMeta = metaInfoBuckets[0][rs][cs][zSplit].flatten();

        // Select the slice that belongs to the tile the vertex is
        // being executed on
        const auto gatherRows =
            rowRange(rs, zSplit, plannedSplits.slicesRowSplits,
                     plannedSplits.slicesSplitRows, slices.dim(0));
        const auto offsetsSlice =
            offsets.slice({gatherRows.begin(), offsetsColumns.begin()},
                          {gatherRows.end(), offsetsColumns.end()});

        const auto subT = slices
                              .slice({gatherRows.begin(), columns.begin()},
                                     {gatherRows.end(), columns.end()})
                              .flatten();

        createSparseDenseTileVertices(
            graph, computeSet, vertexClass, zTile, offsetsSlice, {baseTNZ},
            {baseTMeta}, subT, columns.size(), rowStart,
            plannedSplits.splitRows, cs, plannedSplits.splitZ, scale,
            columnPartition[zSplit][rs][cs]);
      }
    }
  }
  prog.add(Execute(computeSet));
}

// Create the slice result tensor, or the partial intermediate result for a
// 2 stage slice operation.  Specifically map the tensor according to the
// plan, mirroring the location of the input tensor
Tensor createSliceTensor(Graph &graph, Type inputType,
                         const std::vector<std::size_t> &tShape,
                         std::size_t numIndices,
                         const PlannedSplits &plannedSplits,
                         bool firstOfTwoStages,
                         const std::string &debugPrefix) {

  auto result = graph.addVariable(inputType, {numIndices, tShape[1]},
                                  debugPrefix + "/slices");
  logging::popsparse::debug("Creating slice {} result tensor, shape {}",
                            firstOfTwoStages ? "intermediate" : "final",
                            result.shape());

  // Splits for the first of a two stage slice will be based on the input tensor
  // splits, otherwise based on the result splits.  Row splits differ in the
  // two allocations, but column splits do not
  const auto rowSplits = firstOfTwoStages ? plannedSplits.rowSplits
                                          : plannedSplits.slicesRowSplits;

  const auto splitRows =
      firstOfTwoStages
          ? numIndices / (rowSplits * plannedSplits.slicesIndicesSplits)
          : plannedSplits.slicesSplitRows;

  for (unsigned cs = 0; cs < plannedSplits.columnSplits; cs++) {
    for (unsigned is = 0; is < plannedSplits.slicesIndicesSplits; is++) {
      for (unsigned rs = 0; rs < rowSplits; rs++) {
        const auto rows = rowRange(rs, is, rowSplits, splitRows, numIndices);
        if (rows.begin() >= numIndices) {
          continue;
        }
        const auto columns = columnRange(cs, plannedSplits, tShape[1]);
        // Set the mapping of a slice of the tensor to the intended tile
        const auto tile = plannedSplits.mappingOrder.getPNIdForPartition(
            plannedSplits.getPartitions(), {0, rs, cs, is});
        graph.setTileMapping(result.slice({rows.begin(), columns.begin()},
                                          {rows.end(), columns.end()}),
                             tile);
      }
    }
  }
  return result;
}

fullyconnected::Plan getFullyConnectedPlan(Graph &graph, Type inputType,
                                           const FullyConnectedParams &params,
                                           const OptionFlags &optionFlags,
                                           PlanningCache *cache) {
  const auto target = graph.getTarget();
  const auto options = parseMatMulOptionFlags(optionFlags);
  fullyconnected::Plan plan;
  fullyconnected::Cost cost;
  std::tie(plan, cost) = fullyconnected::getPlan(
      target, inputType, params, getFullyConnectedOptions(options), cache);
  return plan;
}

std::tuple<Tensor, SparseTensor>
getInternalSparseTensor(Graph &graph, const SparseTensor &baseT,
                        const fullyconnected::Plan &plan) {
  const auto target = graph.getTarget();

  const auto overflowInfoElems = fullyconnected::getNumOverflowInfoElems(
      target.getTypeSize(UNSIGNED_SHORT), plan.partition.x, plan.partition.y,
      plan.partition.z);

  auto [baseTBuckets, overflowInfo] = fullyconnected::unpackWeights(
      baseT, overflowInfoElems,
      fullyconnected::getTotalMetaInfoElemsPerBuckets(plan),
      plan.nzElemsPerBucket);

  // Get meta-info required for forward pass.
  return {overflowInfo, fullyconnected::weightsInternalSliceBuckets(
                            baseTBuckets, 0u, plan.fwdMetaInfoElemsPerBucket)};
}

// Given a base PlannedSplits structure, apply the same partitioning of rows,
// columns but alter the number of rows and/or columns per partition to a
// single row or column.
std::tuple<PlannedSplits, std::vector<std::size_t>>
createSplitsAndShape(const PlannedSplits &baseSplits, std::size_t outerDim,
                     bool singleRowPerSplit, bool singleColumnPerSplit) {

  auto splits = baseSplits;
  if (singleColumnPerSplit) {
    splits.splitColumns = 1;
    splits.columns = splits.columnSplits;
  }
  if (singleRowPerSplit) {
    splits.splitRows = 1;
    splits.slicesSplitRows = 1;
    splits.rows = splits.rowSplits;
  }
  std::vector<size_t> shape = {outerDim, splits.columnSplits};
  return {splits, shape};
}
// Make exchange buffers.  There are 2 "source" buffers each with destination
// views.  Copying from src[0] to rowDst[1] results in exchange from tile to
// tile over the z,row partition of tiles.  Copying from src[0] to columnDst[1]
// results in exchange from tile to tile over the column partition of tiles.
// Then from src[1] to dstRow[0] or dstColumn[0] continues the sequence.
// Using these 2 alternately data can be propogated as many times as required
struct ExchangeTensors {
  std::vector<Tensor> src;
  std::vector<Tensor> rowDst;
  std::vector<Tensor> columnDst;
};
ExchangeTensors createExchangeTensors(Graph &graph, Type dataType,
                                      const PlannedSplits &plannedSplits,
                                      unsigned numBuffers, unsigned rows,
                                      const std::string &debugPrefix) {

  std::vector<Tensor> src(numBuffers), rowDst(numBuffers), colDst(numBuffers);
  for (unsigned i = 0; i < numBuffers; i++) {
    // Make a source buffer, mapped to tiles in the same way as slices tensors
    // in our plannedSplits
    src[i] = createSliceTensor(
        graph, dataType, {plannedSplits.rows, plannedSplits.columns}, rows,
        plannedSplits, false, debugPrefix + "/ExBuf" + std::to_string(i));

    // Create views into the src tensors which become the destination
    // These are just a view into the source tensor but with the
    // rows circularly modified so that a tile's data gets copied to the
    // next tile (The last slicesSplitRows = the first slicesSplitRows,
    // and all others shuffle up)
    rowDst[i] =
        concat(src[i].slice({plannedSplits.slicesSplitRows, src[i].dim(0)}, 0),
               src[i].slice({0, plannedSplits.slicesSplitRows}, 0), 0);

    // Similarly for columns, where the last splitColumns become the first
    // splitColumns and all the others shuffle left.  Here the way this is
    // split matters as it determines the exchage order.  As we don't always go
    // full circle this direction matters.
    auto splitPoint =
        plannedSplits.splitColumns * (plannedSplits.columnSplits - 1);
    colDst[i] = concat(src[i].slice({splitPoint, src[i].dim(1)}, 1),
                       src[i].slice({0, splitPoint}, 1), 1);
  }
  return {src, rowDst, colDst};
}

// Create a program to implement a loop countdown from either an initial value
// held in a tensor or an unsigned int.  Returns the program and a bool
// decision tensor
std::tuple<Sequence, Tensor>
createDecisionProg(Graph &graph, Sequence &prog,
                   std::variant<unsigned, Tensor> loopCount,
                   const std::string &debugPrefix) {

  auto decisionCount = graph.addVariable(UNSIGNED_INT, {}, debugPrefix);
  graph.setTileMapping(decisionCount, 0);
  Tensor decisionInitialValue;
  if (std::get_if<Tensor>(&loopCount)) {
    decisionInitialValue = cast(graph, std::get<Tensor>(loopCount),
                                UNSIGNED_INT, prog, debugPrefix);
  } else {
    decisionInitialValue = graph.addConstant<unsigned>(
        UNSIGNED_INT, {}, std::get<unsigned>(loopCount), debugPrefix);
    graph.setTileMapping(decisionInitialValue, 0);
  }

  prog.add(Copy(decisionInitialValue, decisionCount));
  Sequence decisionProg;
  // Sample the result then subtract is an equivalent to a decrement after the
  // decision and avoids adding 1 to the loop count
  const auto decision =
      cast(graph, decisionCount, BOOL, decisionProg, debugPrefix);
  subInPlace(graph, decisionCount, 1u, decisionProg, debugPrefix);
  return {decisionProg, decision};
}

// Create a vector of programs to process (run update vertices) for each of the
// buffers, exchange over the row,z partitions combined and toggle the
// buffer select variable
std::vector<Sequence> createComputeProg(
    Graph &graph, Sequence &prog, const PlannedSplits &plannedSplits,
    const SparseTensor &baseTBuckets, const Tensor &bufferSelect,
    const Tensor &scale, unsigned numBuffers,
    const ExchangeTensors &slicesExBuf, const ExchangeTensors &offsetsExBuf,
    const ExchangeTensors &columnPartitionExBuf,
    const std::string &debugPrefix) {

  // Create programs to run in a loop, alternately on each pass
  std::vector<Sequence> loopBodyProg(numBuffers);
  for (unsigned srcBuf = 0; srcBuf < numBuffers; srcBuf++) {
    const auto dstBuf = srcBuf ? 0u : 1u;
    // Create the update vertices with the "source view" of the buffer just
    // copied
    generateSparseDenseMultiUpdateVertices(
        graph, loopBodyProg[srcBuf], offsetsExBuf.src[srcBuf],
        columnPartitionExBuf.src[srcBuf], baseTBuckets,
        {plannedSplits.rows, plannedSplits.columns}, slicesExBuf.src[srcBuf],
        scale, plannedSplits, debugPrefix);

    // Exhchange for next time
    loopBodyProg[srcBuf].add(
        Copy(slicesExBuf.src[srcBuf], slicesExBuf.rowDst[dstBuf]));
    loopBodyProg[srcBuf].add(
        Copy(offsetsExBuf.src[srcBuf], offsetsExBuf.rowDst[dstBuf]));

    // We don't really need to exchange this here but we do need it to
    // end up in the correct srcBuf[0] or [1] to then run an exchange prog.
    // It's small so shouldn't be too much overhead.
    loopBodyProg[srcBuf].add(Copy(columnPartitionExBuf.src[srcBuf],
                                  columnPartitionExBuf.rowDst[dstBuf]));
    // Toggle the buffer used for next time: src<->dst
    bitwiseXorInPlace(graph, bufferSelect, 1u, loopBodyProg[srcBuf]);
  }
  return loopBodyProg;
}

// Create a buffer select variable, plus  a vector of programs to exchange over
// the columns partition and toggle the buffer select variable
std::tuple<std::vector<Sequence>, Tensor> createExchangeProg(
    Graph &graph, Sequence &prog, const PlannedSplits &plannedSplits,
    const SparseTensor &baseTBuckets, const Tensor &scale, unsigned numBuffers,
    const ExchangeTensors &slicesExBuf, const ExchangeTensors &offsetsExBuf,
    const ExchangeTensors &columnPartitionExBuf,
    const std::string &debugPrefix) {

  // Create and initialise the buffer select variable
  auto bufferSelect = graph.addVariable(UNSIGNED_INT, {});
  graph.setTileMapping(bufferSelect, 0);
  auto bufferSelectInitialValue =
      graph.addConstant<unsigned>(UNSIGNED_INT, {}, 0u);
  prog.add(Copy(bufferSelectInitialValue, bufferSelect));
  graph.setTileMapping(bufferSelectInitialValue, 0);

  // Create programs to run in a loop, alternately on each pass
  std::vector<Sequence> loopBodyProg(numBuffers);
  for (unsigned srcBuf = 0; srcBuf < numBuffers; srcBuf++) {
    const auto dstBuf = srcBuf ? 0u : 1u;
    loopBodyProg[srcBuf].add(
        Copy(slicesExBuf.src[srcBuf], slicesExBuf.columnDst[dstBuf]));
    loopBodyProg[srcBuf].add(
        Copy(offsetsExBuf.src[srcBuf], offsetsExBuf.columnDst[dstBuf]));
    loopBodyProg[srcBuf].add(Copy(columnPartitionExBuf.src[srcBuf],
                                  columnPartitionExBuf.columnDst[dstBuf]));
    // Toggle the buffer used for next time: src<->dst
    bitwiseXorInPlace(graph, bufferSelect, 1u, loopBodyProg[srcBuf]);
  }
  return {loopBodyProg, bufferSelect};
}
} // end anonymous namespace

Tensor createIndicesTensor(Graph &graph, const std::vector<std::size_t> &dims,
                           const std::size_t numIndices,
                           const std::string &debugPrefix) {
  logging::popsparse::info("createIndicesTensor for {} / {}", numIndices, dims);
  const auto indices = graph.addVariable(
      UNSIGNED_INT, {numIndices, dims.size()}, debugPrefix + "/indices");
  mapTensorLinearly(graph, indices, minIndicesPerTile, 1);
  return indices;
}

Tensor embeddingSlice(Graph &graph, const SparseTensor &baseT,
                      const Tensor &offsets, Sequence &prog,
                      const FullyConnectedParams &params,
                      const std::string &debugPrefix_,
                      const OptionFlags &options, PlanningCache *cache) {
  const auto inputType = baseT.getNzValuesTensor().elementType();

  const auto plan =
      getFullyConnectedPlan(graph, inputType, params, options, cache);
  const PlannedSplits plannedSplits(offsets.dim(0), params, plan);
  const std::vector<std::size_t> tShape = {plannedSplits.rows,
                                           plannedSplits.columns};
  Tensor overflowInfo;
  SparseTensor baseTBuckets;
  std::tie(overflowInfo, baseTBuckets) =
      getInternalSparseTensor(graph, baseT, plan);

  const auto debugPrefix = debugPrefix_ + "/embeddingSlice";
  // Create the result tensor, with padding in cases of the sliced results
  // not being exactly divided.  Partly this makes things easier but also
  // when there are row splits the act of gathering partials results and
  // generating indices to do the second slice stage requires no odd ends.

  // TODO - where there is padding we need to writeUndef if it isn't used
  const auto paddedSize = plannedSplits.slicesIndicesSplits *
                          plannedSplits.slicesRowSplits *
                          plannedSplits.slicesSplitRows;
  const auto paddedPartialsSize = plannedSplits.rowSplits * paddedSize;

  auto slices = createSliceTensor(graph, inputType, tShape, paddedSize,
                                  plannedSplits, false, debugPrefix);

  if (plannedSplits.rowSplits == 1) {
    // Slice straight into the result as all tiles contain part of all rows
    // As we slice only the locations with Nz values it needs zeroing
    zero(graph, slices, prog, debugPrefix);
    generateSparseDenseMultiSliceVertices(graph, prog, offsets, baseTBuckets,
                                          tShape, slices, plannedSplits,
                                          debugPrefix + "/SingleStage");
  } else {
    // Rows are spread over 'rowSplits' tiles. We need a partials tensor
    // with dims {offsets, splitColumns} per tile
    auto partials =
        createSliceTensor(graph, inputType, tShape, paddedPartialsSize,
                          plannedSplits, true, debugPrefix);

    // Slice into the partials
    // As we slice only the locations with Nz values they need zeroing.  The
    // end result does not, as dense partials are copied into it
    zero(graph, partials, prog, debugPrefix);
    generateSparseDenseMultiSliceVertices(graph, prog, offsets, baseTBuckets,
                                          tShape, partials, plannedSplits,
                                          debugPrefix + "/Stage1");

    // For the second stage we need an offset into the partials information
    // given by the other tiles.  This can be calculated from the initial index
    // as we know the allocation of the input.
    // For example if allocated:
    // Tile 0: Rows 0-16 Tile 1: Rows 16-32 Tile2: Rows 32-48 Tile3: Rows 48-64
    // With indices 17,12,50,33,33,2 ....
    // We know the results will be found on tile 1,0,3,2,2,0 ... (At runtime)
    // So when gathered a tile sees a result (Say we have 3 results per tile),
    // The tile that has results 0,1,2 based on indices 0,1,2 = 17,12,50 gets:
    // idx0 from tile0 = Don't care
    // idx1 from tile0 = Genuine result - idx1 result from offset 1
    // idx2 from tile0 = Don't care
    // idx0 from tile1 = Genuine result - idx0 result from offset 3
    // idx1 from tile1 = Don't care
    // idx2 from tile1 = Don't care
    // idx0 from tile2 = Don't care
    // idx1 from tile2 = Don't care
    // idx2 from tile2 = Don't care
    // idx0 from tile3 = Don't care
    // idx1 from tile3 = Don't care
    // idx2 from tile3 = Genuine result - idx2 result from offset 11
    // The second stage slice needs to convert 17,12,50 to 3,1,11
    // Given by resultsPerTile * (index/rowsPerSourceTile) + offset
    // Where offset = 0,1,2,3... per index on a result tile. So:
    // 3 * (17/16) + 0 = 3
    // 3 * (12/16) + 1 = 1
    // 3 * (50/16) + 2 = 11
    const auto offsetsType = offsets.elementType();

    std::vector<unsigned> offsetsPerTile(plannedSplits.slicesSplitRows);
    // TODO - does this get big ?  If so compute it, but is that slower - so
    //        optionally?
    std::iota(offsetsPerTile.begin(), offsetsPerTile.end(), 0);
    auto offsetsT = graph.addConstant<unsigned>(
        offsetsType, {1, plannedSplits.slicesSplitRows}, offsetsPerTile,
        "/offSetsPerTile");
    graph.setTileMapping(offsetsT, 0);

    // Pad so that the padded size is divisible by splits exactly.  This makes
    // indexing into the "odd bit at the end" when the result isn't split
    // exactly over tiles the same as all the other tiles.  It also makes
    // slicing the same regardless of the "row odd bit at the end"

    auto padT = graph.addConstant<unsigned>(
        offsetsType, {paddedSize - offsets.numElements()}, 0u,
        debugPrefix + "/PadOffsets");
    graph.setTileMapping(padT, 0);
    auto offsetsReshaped = concat(offsets.flatten(), padT);
    offsetsReshaped = offsetsReshaped.reshape(
        {offsetsReshaped.numElements() / plannedSplits.slicesSplitRows,
         plannedSplits.slicesSplitRows});

    auto stage2Offsets = map(
        graph,
        (plannedSplits.slicesSplitRows * (_1 / plannedSplits.splitRows)) + _2,
        {offsetsReshaped, offsetsT}, prog, debugPrefix + "/stage2Indices");
    graph.setTileMapping(stage2Offsets, 0);

    // Slice from partials into the result (All data is now dense)
    generateDenseDenseMultiSliceVertices(
        graph, prog, stage2Offsets, partials, slices, plannedSplits,
        debugPrefix + "/embeddingSlice/Stage2");
  }
  // Return the result, slicing off the padding if any was added in a 2 stage
  // multislice
  return slices.slice(0, offsets.dim(0));
}

// There is only 1 stage here, as every tile needs access to every row
// of the `slices` input and the slice in the columns dimension
// that matches up with the slice in the planned sparse baseT tile allocation.
// There is the additional issue of bucket spills, where tiles in a row
// partition spill their content to other tiles in  the same row partition,
// and likewise in column partitions.
// All of this is dealt with by serialisation (which also reduces the
// temporary memory requirement). Data is
// a) used to update and then exchanged from tile to tile in each
//    `row and z partition combined` until it has passed full circle.
// b) Exchanged in the column partition
//
// After b) a fully cycle of a) is cariied out once more - repeatedly until
// all data has visited all the tiles it needs to and updates are complete.
//
// Sparse data (being updated) remains ontile, dense data and indices move.

void embeddingUpdateAdd(Graph &graph, const SparseTensor &baseT,
                        const Tensor &slices_, const Tensor &offsets_,
                        const Tensor &scale, Sequence &prog,
                        const FullyConnectedParams &params,
                        const std::string &debugPrefix_,
                        const OptionFlags &options, PlanningCache *cache) {

  const auto debugPrefix = debugPrefix_ + "/embeddingUpdateAdd";
  const auto plan = getFullyConnectedPlan(
      graph, baseT.getNzValuesTensor().elementType(), params, options, cache);

  // Pad offsets to a consistent shape, otherwise when exchanging
  // and processing every step needs to be customised to the number of
  // offsets on each tile.  The pad is initialised with max unsigned to
  // prevent actual processing (Such an offset won't be found in the sparse
  // data).  Therefore the slice data doesn't need to be padded as it isn't
  // accessed.
  const auto offsets = [&]() {
    const PlannedSplits splits(offsets_.dim(0), params, plan);
    const auto paddedSize =
        splits.zSplits * splits.rowSplits * splits.slicesSplitRows;
    const auto paddedValue = std::numeric_limits<unsigned>::max();
    auto pad = graph.addConstant<unsigned>(
        offsets_.elementType(), {paddedSize - offsets_.dim(0), 1}, paddedValue,
        debugPrefix + "/PadOffsets");
    graph.setTileMapping(pad, 0);
    return concat(offsets_, pad);
  }();

  // Where the number of columns wasn't exactly divisible by the column splits
  // we need to pad to make the last, smaller slice the same as all the others.
  // This is to ensure that when it is passed to other column partitions it is
  // treated correctly. The meta-data content will ensure that padded elements
  // are never accessed
  const auto slices = [&]() {
    const PlannedSplits splits(offsets_.dim(0), params, plan);
    const auto paddedSize = splits.columnSplits * splits.splitColumns;
    auto pad = graph.addVariable(slices_.elementType(),
                                 {slices_.dim(0), paddedSize - slices_.dim(1)},
                                 debugPrefix + "/PadSlices");
    graph.setTileMapping(pad, 0);
    return concat(slices_, pad, 1);
  }();

  // We want the same plan so present the same parameters, but the total
  // columns we operate on is different as we padded above
  const PlannedSplits plannedSplits(offsets.dim(0), params, plan,
                                    slices.dim(1));
  const auto [overflowInfo, baseTBuckets] =
      getInternalSparseTensor(graph, baseT, plan);
  const auto inputType = baseT.getNzValuesTensor().elementType();

  // Create splits plan matching that of the slices, but containing the
  // offsets (In other words the same, but with 1 column per split)
  auto [offsetsSplits, broadcastOffsetsShape] =
      createSplitsAndShape(plannedSplits, offsets.dim(0), false, true);
  // Create a broadcast offsets tensor to spread over tiles with each
  // column split - a broadcast of the offsets over the planned splits
  auto offsetsBroadcast = offsets;
  broadcastToMatch(offsetsBroadcast, broadcastOffsetsShape);

  // Create splits plan matching that of the slices, but containing the
  // columnPartition variable (In other words the same, but with 1 column per
  // split and 1 row per split)
  auto [columnPartitionSplits, columnPartitionShape] = createSplitsAndShape(
      plannedSplits, plannedSplits.zSplits * plannedSplits.rowSplits, true,
      true);
  // Create a columnPartition tensor to spread over tiles with each
  // column split: {0,1,2,...columnSplits} broadcast  rowSplits*zSplits times

  std::vector<unsigned short> columnPartitions(
      columnPartitionSplits.columnSplits);
  std::iota(columnPartitions.begin(), columnPartitions.end(), 0);
  auto columnPartitionBroadcast = graph.addConstant<unsigned short>(
      baseT.getMetaInfoTensor().elementType(),
      {1, columnPartitionSplits.columnSplits}, columnPartitions);

  broadcastToMatch(columnPartitionBroadcast, columnPartitionShape);

  columnPartitionBroadcast = columnPartitionBroadcast.reshape(
      {columnPartitionSplits.zSplits, columnPartitionSplits.rowSplits,
       columnPartitionSplits.columnSplits});
  graph.setTileMapping(columnPartitionBroadcast, 0);

  // Create pairs of exchange buffers with the same mapping as the slices
  // and same shape, plus a view into the buffer representing the exchange
  // patterns required to propogate over rows (z and row splits) and columns.
  // We need this for slices, offsets and an columnPartition.
  // Slices - the actual slice data
  // Offsets - the offsets/indices matching the slices
  // ColumnPartition - the column partition number matching the data. 0,1,2...
  //                   per columnpartition. Needed when columns are spilled.
  const auto numBuffers = 2;
  const auto slicesExBuf =
      createExchangeTensors(graph, inputType, plannedSplits, numBuffers,
                            offsets.dim(0), debugPrefix + "/Slices");

  const auto offsetsExBuf = createExchangeTensors(
      graph, offsets.elementType(), offsetsSplits, numBuffers, offsets.dim(0),
      debugPrefix + "/Offsets");

  auto columnPartitionExBuf = createExchangeTensors(
      graph, columnPartitionBroadcast.elementType(), columnPartitionSplits,
      numBuffers,
      columnPartitionBroadcast.dim(0) * columnPartitionBroadcast.dim(1),
      debugPrefix + "/ColumnPartition");

  for (unsigned i = 0; i < numBuffers; i++) {
    const auto shape = columnPartitionBroadcast.shape();
    columnPartitionExBuf.src[i] = columnPartitionExBuf.src[i].reshape(shape);
    columnPartitionExBuf.rowDst[i] =
        columnPartitionExBuf.rowDst[i].reshape(shape);
    columnPartitionExBuf.columnDst[i] =
        columnPartitionExBuf.columnDst[i].reshape(shape);
  }

  // Create program fragments used below

  // Use the overflow info to determine the number of column loops to make
  const auto yPartitionOverflowIndex = 1;
  auto columnOverFlowCount = overflowInfo.slice(
      {yPartitionOverflowIndex, yPartitionOverflowIndex + 1});
  auto [outerDecisionProg, outerDecisionFlag] = createDecisionProg(
      graph, prog, columnOverFlowCount, debugPrefix + "/outerLoop");

  const auto [exchangeProg, bufferSelect] = createExchangeProg(
      graph, prog, plannedSplits, baseTBuckets, scale, numBuffers, slicesExBuf,
      offsetsExBuf, columnPartitionExBuf, debugPrefix);

  auto computeProg = createComputeProg(
      graph, prog, plannedSplits, baseTBuckets, bufferSelect, scale, numBuffers,
      slicesExBuf, offsetsExBuf, columnPartitionExBuf, debugPrefix);

  // Prime the 1st buffer
  prog.add(Copy(slices, slicesExBuf.src[0].slice(0, slices.dim(0))));
  prog.add(Copy(offsetsBroadcast, offsetsExBuf.src[0]));
  prog.add(Copy(columnPartitionBroadcast, columnPartitionExBuf.src[0]));

  // Now use the programs and variables created to make the program below. The
  // switch program is used to select the computeProg or exchangeProg
  // based on srcIndex:
  //
  // srcIndex = 0;
  // dstIndex = 1;
  // columnLoops = column spill distance
  // while(columnLoops!=0) {
  //   rowLoops = rs * is;  //(This must perform a full rotation)
  //   while(rowLoops!=0) {
  //     // This loop body is `computeProg[0] or [1]`
  //     createVertices[srcIndex];
  //     exchangeSlices(src[srcIndex], dstRow[dstIndex]);
  //     exchangeIndices(src[srcIndex], dstRow[dstIndex]);
  //     exchangeColumnPartitions(src[srcIndex], dstRow[dstIndex]);
  //     srcIndex=xor(srcIndex,1);
  //     dstIndex=xor(dstIndex,1);
  //     rowLoops--;
  //   }
  //   // This is `exchangeProg[0] or [1]`
  //   exchangeSlices(src[srcIndex], dstColumn[dstIndex]);
  //   exchangeIndices(src[srcIndex], dstColumn[dstIndex]);
  //   exchangeColumnPartitions(src[srcIndex], dstColumn[dstIndex]);
  //   srcIndex=xor(srcIndex,1);
  //   dstIndex=xor(dstIndex,1);
  //   columnLoops--;
  // }

  Sequence innerProg, outerProg;
  innerProg.add(
      Switch(bufferSelect, {{0, computeProg[0]}, {1, computeProg[1]}}));

  outerProg.add(
      Repeat(plannedSplits.slicesRowSplits * plannedSplits.slicesIndicesSplits,
             innerProg));
  outerProg.add(
      Switch(bufferSelect, {{0, exchangeProg[0]}, {1, exchangeProg[1]}}));

  prog.add(RepeatWhileTrue(outerDecisionProg, outerDecisionFlag, outerProg));
}

// External function to be used to create the slice input for an update
// operation
Tensor createSliceTensor(Graph &graph, const SparseTensor &t,
                         std::size_t numIndices,
                         const FullyConnectedParams &params,
                         const std::string &debugPrefix,
                         const OptionFlags &options, PlanningCache *cache) {
  const auto plan = getFullyConnectedPlan(
      graph, t.getNzValuesTensor().elementType(), params, options, cache);
  const PlannedSplits plannedSplits(numIndices, params, plan);
  const auto inputType = t.getNzValuesTensor().elementType();
  return createSliceTensor(graph, inputType,
                           {plannedSplits.rows, plannedSplits.columns},
                           numIndices, plannedSplits, false, debugPrefix);
}

} // end namespace dynamic
} // end namespace popsparse
