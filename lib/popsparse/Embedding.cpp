// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include "popsparse/Embedding.hpp"
#include "FullyConnectedOptions.hpp"
#include "FullyConnectedPNMapping.hpp"
#include "FullyConnectedPlan.hpp"
#include "FullyConnectedUtils.hpp"
#include "FullyConnectedVector.hpp"
#include "SparseCodeletMetaInfoScale.hpp"
#include "poplibs_support/logging.hpp"
#include "popops/Cast.hpp"
#include "popops/ElementWise.hpp"
#include "popops/ScaledAdd.hpp"
#include "popops/Zero.hpp"
#include "popsparse/MatMul.hpp"
#include "popsparse/SparsePartitioner.hpp"
#include "poputil/TileMapping.hpp"
#include <popops/Expr.hpp>
#include <poputil/Broadcast.hpp>
#include <poputil/DebugInfo.hpp>
#include <poputil/VertexTemplates.hpp>

#include <boost/optional.hpp>

#include <variant>

using namespace poplar;
using namespace poplar::program;
using namespace poputil;
using namespace poplibs_support;
using namespace popsparse;
using namespace popsparse::dynamic;
using namespace popops;
using namespace popops::expr;

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

  // BlockSizes (both =1 for elementWise sparsity)
  unsigned blockRows;
  unsigned blockColumns;

  PlannedSplits(std::size_t numSlices, const FullyConnectedParams &params,
                const fullyconnected::Plan &plan,
                const poplar::DebugNameAndId &dnai,
                boost::optional<std::size_t> plannedColumns = boost::none,
                bool respectBlockSize = true) {

    blockRows = plan.method.grouping.x;
    blockColumns = plan.method.grouping.y;

    rows = params.getOutputChannelsPerGroup();
    if (plannedColumns) {
      columns = plannedColumns.get();
    } else {
      columns = params.getInputChannelsPerGroup();
    }
    z = params.getBatchSize();

    // Where we are planning the layout of slices we need to round up any split
    // to the next multiple of the block size.  Where we are using
    // `PlannedSplits` to plan exchange buffers we need to use the sizes given
    // with no rounding up.
    const auto planningBlockRows = respectBlockSize ? blockRows : 1;
    const auto planningBlockColumns = respectBlockSize ? blockColumns : 1;

    // Just note the splits from the plan, and find the number of elements
    rowSplits = plan.partition.x;

    const auto splitRowBlocks = (rows + rowSplits * planningBlockRows - 1) /
                                (rowSplits * planningBlockRows);
    splitRows = planningBlockRows * splitRowBlocks;

    columnSplits = plan.partition.y;
    const auto splitColumnBlocks =
        (columns + columnSplits * planningBlockColumns - 1) /
        (columnSplits * planningBlockColumns);
    splitColumns = planningBlockColumns * splitColumnBlocks;

    // Unlike X,Y the splitting of Z is not subject to block size boundaries
    zSplits = plan.partition.z;
    splitZ = (z + zSplits - 1) / zSplits;

    groups = plan.partition.groups;
    // Extract the mapping order
    mappingOrder = plan.exchangePlan.fwdMapping;

    // Decide how to split the slice result?
    // 1. Split by column in the same way as the input.
    // 2. Split by row - following the Z partition splits of the sparse
    //    input.  If we were to gather data over the Z partition (not at
    //    present) each partition of slices split in this way could be
    //    sliced into with no further exchange.
    //    (This is splitting by row but with the advantage described)
    // 3. Finally split by row - following the row (X) partition splits of the
    //    sparse input
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
      logging::popsparse::debug("Sparse Dense Embedding plan for {}:",
                                dnai.getPathName());
      logging::popsparse::debug("Sparse tensor has block size {},{}", blockRows,
                                blockColumns);
      logging::popsparse::debug(
          "    baseT rows:{} rowSplits:{} rowsPerSplit:{}", rows, rowSplits,
          splitRows);
      logging::popsparse::debug(
          "    baseT columns:{} columnSplits:{} columnsPerSplit:{}", columns,
          columnSplits, splitColumns);
      logging::popsparse::debug("    baseT z:{} zSplits:{} zPerSplit:{}", z,
                                zSplits, splitZ);
      logging::popsparse::debug(
          "    Slices rows:{} rowSplits:{} rowsPerSplit:{}", numSlices,
          slicesRowSplits, slicesSplitRows);
      logging::popsparse::debug(
          "    Slices columns:{} columnSplits:{} columnsPerSplit:{}", columns,
          columnSplits, splitColumns);
      logging::popsparse::debug("    Slices zSplits:{} zPerSplit:{}",
                                slicesIndicesSplits, slicesSplitIndices);
    }
  }

  fullyconnected::Vector<unsigned> getPartitions(void) const {
    return {groups, rowSplits, columnSplits, zSplits};
  }

  bool isElementWise(void) const { return blockRows == 1 && blockColumns == 1; }
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

void createSparseDenseTileVertices(
    Graph &graph, ComputeSet &computeSet, const std::string &vertexClass,
    unsigned tile, const Tensor &offsets, const std::vector<Tensor> &baseTNZ,
    const std::vector<Tensor> &baseTMeta, const Tensor &subT, unsigned columns,
    const PlannedSplits &plannedSplits,
    const std::variant<unsigned, Tensor> &yPartitionToProcess,
    const boost::optional<Tensor> &scale = boost::none) {

  // Using supervisor vertices, so work division is done in the vertex.
  const auto vertex = graph.addVertex(computeSet, vertexClass);
  graph.setTileMapping(vertex, tile);

  graph.connect(vertex["offsets"], offsets.flatten());
  graph.connect(vertex["subT"], subT);

  graph.connect(vertex["baseTMetaInfo"], baseTMeta);
  graph.connect(vertex["baseTNZ"], baseTNZ);
  graph.setInitialValue(vertex["subColumns"], columns);
  graph.setInitialValue(vertex["rowsPerPartition"], plannedSplits.splitRows);
  graph.setInitialValue(vertex["numOffsets"], offsets.numElements());

  if (plannedSplits.isElementWise()) {
    graph.setInitialValue(vertex["nzScaleFactor"],
                          reciprocalMulFactor(plannedSplits.splitZ));
  } else {
    graph.setInitialValue(vertex["blockRows"], plannedSplits.blockRows);
    graph.setInitialValue(vertex["blockColumns"], plannedSplits.blockColumns);
  }

  if (scale) {
    graph.connect(vertex["scale"], scale.get().reshape({}));
  }

  if (std::get_if<Tensor>(&yPartitionToProcess)) {
    graph.connect(vertex["yPartitionToProcess"],
                  std::get<Tensor>(yPartitionToProcess));
  } else {
    graph.setInitialValue(vertex["yPartitionToProcess"],
                          std::get<unsigned>(yPartitionToProcess));
  }
}

void generateSparseDenseMultiSliceVertices(
    Graph &graph, Sequence &prog, const Tensor &offsets,
    const Tensor &nzBuckets, const PlannedSplits &nzSplits,
    const Tensor &metaInfoBuckets, const PlannedSplits &metaInfoSplits,
    const Tensor &slices, const PlannedSplits &plannedSplits,
    const poplar::DebugNameAndId &dnai) {

  auto computeSet = graph.addComputeSet({dnai});
  const auto inputType = nzBuckets.elementType();

  auto bytesPerBlockRow =
      graph.getTarget().getTypeSize(inputType) * plannedSplits.blockColumns;
  const unsigned vectorWidthInBytes =
      (bytesPerBlockRow % 8 == 0) ? 8 : ((bytesPerBlockRow % 4 == 0) ? 4 : 2);
  const auto vertexClass =
      plannedSplits.isElementWise()
          ? templateVertex("popsparse::SparseDenseMultiSliceElementWise",
                           inputType)
          : templateVertex("popsparse::SparseDenseMultiSliceBlock", inputType,
                           vectorWidthInBytes);

  logging::popsparse::debug("creating {} vertices", vertexClass);

  // Loop and create vertices based on the existence of offsets to process
  // and so use the plannedSplits.slicesRowSplits, SplitRows etc... variables
  for (unsigned cs = 0; cs < plannedSplits.columnSplits; cs++) {
    for (unsigned rs = 0; rs < plannedSplits.slicesRowSplits; rs++) {
      for (unsigned is = 0; is < plannedSplits.slicesIndicesSplits; is++) {
        const auto offsetsPartition = is * plannedSplits.slicesRowSplits + rs;
        const auto offsetsStart =
            offsetsPartition * plannedSplits.slicesSplitRows;
        if (offsetsStart >= offsets.dim(0)) {
          continue;
        }
        const auto offsetsEnd =
            std::min((offsetsPartition + 1) * plannedSplits.slicesSplitRows,
                     offsets.dim(0));
        const auto rows =
            rowRange(rs, is, plannedSplits.slicesRowSplits,
                     plannedSplits.slicesSplitRows, slices.dim(0));
        const auto columns =
            columnRange(cs, plannedSplits, plannedSplits.columns);

        // Slice the offsets, slices, Nz and metadata
        const auto offsetsSlice =
            offsets.slice({offsetsStart, cs}, {offsetsEnd, cs + 1});
        const auto subT = slices
                              .slice({rows.begin(), columns.begin()},
                                     {rows.end(), columns.end()})
                              .flatten();

        const auto nzColumns = columnRange(cs, nzSplits, nzSplits.columns);
        const auto metaInfoColumns =
            columnRange(cs, metaInfoSplits, metaInfoSplits.columns);

        const auto tilePartition = is * nzSplits.rowSplits + rs;
        const auto baseTNZ = nzBuckets
                                 .slice({tilePartition, nzColumns.begin()},
                                        {tilePartition + 1, nzColumns.end()})
                                 .flatten();
        const auto baseTMeta =
            metaInfoBuckets
                .slice({tilePartition, metaInfoColumns.begin()},
                       {tilePartition + 1, metaInfoColumns.end()})
                .flatten();

        const auto iTile = plannedSplits.mappingOrder.getPNIdForPartition(
            plannedSplits.getPartitions(), {0, rs, cs, is});

        createSparseDenseTileVertices(graph, computeSet, vertexClass, iTile,
                                      offsetsSlice, {baseTNZ}, {baseTMeta},
                                      subT, columns.size(), plannedSplits, cs);
      }
    }
  }
  prog.add(Execute(computeSet, {dnai}));
}

void generateSparseDenseMultiUpdateVertices(
    Graph &graph, Sequence &prog, const Tensor &offsets,
    const Tensor &columnPartition, const Tensor &metaInfoBuckets,
    const Tensor &nzBuckets, const Tensor &slices, const Tensor &scale,
    const PlannedSplits &plannedSplits, const poplar::DebugNameAndId &dnai) {

  auto computeSet = graph.addComputeSet({dnai});
  const auto target = graph.getTarget();
  const auto inputType = slices.elementType();
  const auto outputVectorWidth = target.getVectorWidth(nzBuckets.elementType());
  bool vectorise = (plannedSplits.blockColumns % outputVectorWidth) == 0;
  const auto vertexClass =
      plannedSplits.isElementWise()
          ? templateVertex("popsparse::SparseDenseMultiUpdateAddElementWise",
                           inputType)
          : templateVertex("popsparse::SparseDenseMultiUpdateAddBlock",
                           inputType, vectorise);

  logging::popsparse::debug("creating {} vertices", vertexClass);

  // Make a plan variable that represents the layout, but with only 1 column
  // per partition
  auto offsetsSplits = plannedSplits;
  offsetsSplits.splitColumns = 1;
  offsetsSplits.columns = offsetsSplits.columnSplits;

  for (unsigned cs = 0; cs < plannedSplits.columnSplits; cs++) {
    for (unsigned rs = 0; rs < plannedSplits.rowSplits; rs++) {
      const auto columns =
          columnRange(cs, plannedSplits, plannedSplits.columns);
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

        createSparseDenseTileVertices(graph, computeSet, vertexClass, zTile,
                                      offsetsSlice, {baseTNZ}, {baseTMeta},
                                      subT, columns.size(), plannedSplits,
                                      columnPartition[zSplit][rs][cs], scale);
      }
    }
  }
  prog.add(Execute(computeSet, {dnai}));
}

// Create the slice result tensor, or exchange buffers for data that will
// propogate during serial exchange stages.  Specifically map the tensor
// according to the plan, mirroring the location of the input tensor
Tensor createSliceTensor(Graph &graph, Type inputType, std::size_t numIndices,
                         const PlannedSplits &plannedSplits,
                         const poplar::DebugNameAndId &dnai) {

  auto result =
      graph.addVariable(inputType, {numIndices, plannedSplits.columns}, {dnai});
  logging::popsparse::debug("Creating slice tensor with shape {} for {}",
                            result.shape(), dnai.getPathName());

  for (unsigned cs = 0; cs < plannedSplits.columnSplits; cs++) {
    for (unsigned is = 0; is < plannedSplits.slicesIndicesSplits; is++) {
      for (unsigned rs = 0; rs < plannedSplits.slicesRowSplits; rs++) {
        const auto rows = rowRange(rs, is, plannedSplits.slicesRowSplits,
                                   plannedSplits.slicesSplitRows, numIndices);
        if (rows.begin() >= numIndices) {
          continue;
        }
        const auto columns =
            columnRange(cs, plannedSplits, plannedSplits.columns);
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
  fullyconnected::Plan plan;
  fullyconnected::Cost cost;
  std::tie(plan, cost) =
      fullyconnected::getPlan(target, inputType, params, optionFlags, cache);
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
ExchangeTensors createExchangeTensors(Graph &graph, Sequence &prog,
                                      Type dataType,
                                      const PlannedSplits &plannedSplits,
                                      unsigned numBuffers, unsigned rows,
                                      bool isSlice,
                                      const poplar::DebugNameAndId &dnai) {

  std::vector<Tensor> src(numBuffers), rowDst(numBuffers), colDst(numBuffers);
  for (unsigned i = 0; i < numBuffers; i++) {
    // Make a source buffer, mapped to tiles in the same way as slices tensors
    // in our plannedSplits
    src[i] = createSliceTensor(graph, dataType, rows, plannedSplits,
                               {dnai, "ExBuf" + std::to_string(i)});
    // Ensure that he exchange buffers don't remain always live. This can
    // happen as they are not necessarily completely written, or it is not
    // clear that they are completely written due to the dynamic program flow
    prog.add(WriteUndef(src[i], {dnai}));
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

    // If Slice we exchange in the opposite direction to Update, as for Slice
    // the sparse data moves, in Update the dense data moves
    const auto splitPoint =
        isSlice
            ? plannedSplits.splitColumns
            : (plannedSplits.splitColumns * (plannedSplits.columnSplits - 1));

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
                   const poplar::DebugNameAndId &dnai) {

  auto decisionCount = graph.addVariable(UNSIGNED_INT, {}, {dnai});
  graph.setTileMapping(decisionCount, 0);
  Tensor decisionInitialValue;
  if (std::get_if<Tensor>(&loopCount)) {
    decisionInitialValue =
        cast(graph, std::get<Tensor>(loopCount), UNSIGNED_INT, prog, {dnai});
  } else {
    decisionInitialValue = graph.addConstant<unsigned>(
        UNSIGNED_INT, {}, std::get<unsigned>(loopCount), {dnai});
    graph.setTileMapping(decisionInitialValue, 0);
  }

  prog.add(Copy(decisionInitialValue, decisionCount, false, {dnai}));
  Sequence decisionProg({}, {dnai});
  // Sample the result then subtract is an equivalent to a decrement after the
  // decision and avoids adding 1 to the loop count
  const auto decision = cast(graph, decisionCount, BOOL, decisionProg, {dnai});
  subInPlace(graph, decisionCount, 1u, decisionProg, {dnai});
  return {decisionProg, decision};
}

// Update (toggle 0<->1) the buffer index
void bufferIndexUpdate(Graph &graph, const Tensor &index, Sequence &prog,
                       const poplar::DebugNameAndId &dnai) {
  auto cs = graph.addComputeSet({dnai, "bufIncrement"});
  auto v = graph.addVertex(
      cs, templateVertex("popsparse::BufferIndexUpdate", index.elementType()));
  graph.connect(v["index"], index);
  graph.setTileMapping(v, 0);
  prog.add(Execute(cs, {dnai}));
}

// Create a vector of programs to process (run Update vertices) for each of the
// buffers, exchange over the row,z partitions combined and toggle the
// buffer select variable
std::vector<Sequence> createUpdateComputeProg(
    Graph &graph, Sequence &prog, const PlannedSplits &plannedSplits,
    const Tensor &metaInfoBuckets, const Tensor &nzBuckets,
    const Tensor &bufferSelect, const Tensor &scale, unsigned numBuffers,
    const ExchangeTensors &slicesExBuf, const ExchangeTensors &offsetsExBuf,
    const ExchangeTensors &columnPartitionExBuf,
    const poplar::DebugNameAndId &dnai) {

  // Create programs to run in a loop, alternately on each pass
  std::vector<Sequence> loopBodyProg(numBuffers);
  for (unsigned srcBuf = 0; srcBuf < numBuffers; srcBuf++) {
    const auto dstBuf = srcBuf ? 0u : 1u;
    // Create the update vertices with the "source view" of the buffer just
    // copied
    generateSparseDenseMultiUpdateVertices(
        graph, loopBodyProg[srcBuf], offsetsExBuf.src[srcBuf],
        columnPartitionExBuf.src[srcBuf], metaInfoBuckets, nzBuckets,
        slicesExBuf.src[srcBuf], scale, plannedSplits, {dnai});

    // Exchange for next time
    loopBodyProg[srcBuf].add(Copy(slicesExBuf.src[srcBuf],
                                  slicesExBuf.rowDst[dstBuf], false, {dnai}));
    loopBodyProg[srcBuf].add(Copy(offsetsExBuf.src[srcBuf],
                                  offsetsExBuf.rowDst[dstBuf], false, {dnai}));

    // We don't really need to exchange this here but we do need it to
    // end up in the correct srcBuf[0] or [1] to then run an exchange prog.
    // It's small so shouldn't be too much overhead.
    loopBodyProg[srcBuf].add(Copy(columnPartitionExBuf.src[srcBuf],
                                  columnPartitionExBuf.rowDst[dstBuf], false,
                                  {dnai}));
    // Toggle the buffer used for next time: src<->dst
    bufferIndexUpdate(graph, bufferSelect, loopBodyProg[srcBuf], {dnai});
  }
  return loopBodyProg;
}

std::vector<Sequence> createSliceComputeProg(
    Graph &graph, Sequence &prog, const PlannedSplits &plannedSplits,
    const Tensor &slices, const Tensor &offsets, const Tensor &bufferSelect,
    unsigned numBuffers, const ExchangeTensors &nzExBuf,
    const PlannedSplits &nzSplits, const ExchangeTensors &metaInfoExBuf,
    const PlannedSplits &metaInfoSplits, const poplar::DebugNameAndId &dnai) {

  // Create programs to run in a loop, alternately on each pass
  std::vector<Sequence> loopBodyProg(numBuffers);
  for (unsigned srcBuf = 0; srcBuf < numBuffers; srcBuf++) {
    const auto dstBuf = srcBuf ? 0u : 1u;
    // Create the update vertices with the "source view" of the buffer just
    // copied
    generateSparseDenseMultiSliceVertices(
        graph, loopBodyProg[srcBuf], offsets, nzExBuf.src[srcBuf], nzSplits,
        metaInfoExBuf.src[srcBuf], metaInfoSplits, slices, plannedSplits,
        {dnai});

    // Exchange for next time
    loopBodyProg[srcBuf].add(
        Copy(nzExBuf.src[srcBuf], nzExBuf.rowDst[dstBuf], false, {dnai}));
    loopBodyProg[srcBuf].add(Copy(metaInfoExBuf.src[srcBuf],
                                  metaInfoExBuf.rowDst[dstBuf], false, {dnai}));

    // Toggle the buffer used for next time: src<->dst
    bufferIndexUpdate(graph, bufferSelect, loopBodyProg[srcBuf], {dnai});
  }
  return loopBodyProg;
}

// Create a buffer select variable, plus  a vector of programs to exchange over
// the columns partition and toggle the buffer select variable
std::tuple<std::vector<Sequence>, Tensor>
createExchangeProg(Graph &graph, Sequence &prog, unsigned numBuffers,
                   const std::vector<ExchangeTensors> &exBufs,
                   const poplar::DebugNameAndId &dnai) {

  // Create and initialise the buffer select variable
  auto bufferSelect = graph.addVariable(UNSIGNED_INT, {}, {dnai});
  graph.setTileMapping(bufferSelect, 0);
  auto bufferSelectInitialValue =
      graph.addConstant<unsigned>(UNSIGNED_INT, {}, 0u, {dnai});
  prog.add(Copy(bufferSelectInitialValue, bufferSelect, false, {dnai}));
  graph.setTileMapping(bufferSelectInitialValue, 0);

  // Create programs to run in a loop, alternately on each pass
  std::vector<Sequence> loopBodyProg(numBuffers);
  for (unsigned srcBuf = 0; srcBuf < numBuffers; srcBuf++) {
    const auto dstBuf = srcBuf ? 0u : 1u;
    for (unsigned i = 0; i < exBufs.size(); i++) {
      loopBodyProg[srcBuf].add(Copy(
          exBufs[i].src[srcBuf], exBufs[i].columnDst[dstBuf], false, {dnai}));
    }
    // Toggle the buffer used for next time: src<->dst
    bufferIndexUpdate(graph, bufferSelect, loopBodyProg[srcBuf], {dnai});
  }
  return {loopBodyProg, bufferSelect};
}

// Create a 2D shape based on the shape of the input tensor. This is used for
// building a series of partitions representing the mapping of the sparse data
// per tile.
std::vector<std::size_t> getShape2D(const Tensor &input) {
  auto shape = input.shape();
  const unsigned xDim = 1;
  const unsigned yDim = 2;
  const unsigned zDim = 3;
  const unsigned elemsPerBucketDim = 5;
  return {shape[xDim] * shape[zDim], shape[yDim] * shape[elemsPerBucketDim]};
}

void to2DShape(Tensor &input) {
  // Buckets have shape [group][XSplit][YSplit][ZSplit]....
  // and so need reshaping so we can treat them as a row, column per tile for
  // exchange
  const auto shape2D = getShape2D(input);
  input = input.dimRoll(3, 1).reshape(shape2D);
}

} // end anonymous namespace

Tensor createIndicesTensor(Graph &graph, const FullyConnectedParams &params,
                           const std::size_t numIndices,
                           const OptionFlags &optionFlags,
                           const poplar::DebugContext &debugContext) {
  poputil::PoplibsOpDebugInfo di(debugContext, DI_ARGS(params, numIndices));

  logging::popsparse::info("createIndicesTensor with {} indices", numIndices);
  const auto indices =
      graph.addVariable(UNSIGNED_INT, {numIndices}, {di, "indices"});
  mapTensorLinearly(graph, indices, minIndicesPerTile, 1);
  return indices;
}

// The method used here is that the dense slice data and the offsets remain
// on the same tile throughout.  The sparse NZ,metadata are exchanged so that
// they "visit" the slice data.  On a visit the dense data is partially
// poplulated with the NZ data that is relevant to its slice.
// ** As the offsets are dynamic there is a further benefit to this approach
//    for this specific operation.
//
// This method requires a fairly small amount of temporary memory (source and
// destination exchange buffers for NZ,metadata), it copes with spilled buckets
// and doesn't need a second stage to cope with the dynamic offsets.
//
// The sparse NZ,metadata data is
// a) used to slice and then exchanged from tile to tile in each
//    `row and z partition combined` until it has passed full circle.
// b) Exchanged in the column partition
//
// After b) a full cycle of a) is carried out once more - repeatedly until
// all data has visited all the tiles it needs to and slices are complete.
//
// Dense data (being sliced into) remains ontile, NZ, metadata move.
// The column partiton that the vertex is populating remains the same, so there
// is no additional columnPartiton data to exchange (as there is in Update)

Tensor embeddingSlice(Graph &graph, const SparseTensor &baseT,
                      const Tensor &offsets, Sequence &prog,
                      const FullyConnectedParams &params,
                      const poplar::DebugContext &debugContext,
                      const OptionFlags &options, PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(baseT, offsets, params, options, cache));
  const auto inputType = baseT.getNzValuesTensor().elementType();
  const std::string layer = "embeddingSlice";
  const auto target = graph.getTarget();
  const auto plan =
      getFullyConnectedPlan(graph, inputType, params, options, cache);

  // Create the result tensor. If the data type is half and the number of
  // columns in any partition is odd we need to pad so that each partition of
  // the slices has an even number of columns.  This greatly simplifies the
  // vertex implementation.  It has no real cost here.

  // Create a plan based on the actual input and use that to find the number of
  // columns that we would have with padding such that each partition is the
  // same size and has an even number of columns

  const PlannedSplits splits(offsets.dim(0), params, plan, {di, layer});
  const auto elementsPerWrite =
      target.getAtomicStoreGranularity() / target.getTypeSize(inputType);
  const auto requiredPadding = splits.splitColumns % elementsPerWrite;
  const auto paddedColumns =
      splits.columnSplits * (splits.splitColumns + requiredPadding);

  // Create a result tensor with the padding and a plan to describe it
  const PlannedSplits paddedSplits(offsets.dim(0), params, plan, {di, layer},
                                   paddedColumns);
  const auto paddedSlices = createSliceTensor(graph, inputType, offsets.dim(0),
                                              paddedSplits, {di, layer});
  // Create a result, based on slicing the padded tensor
  const auto slices = [&]() {
    // Remove the padding that was added to each partition to make its width
    // even
    auto slices =
        paddedSlices.reshape({paddedSlices.dim(0), paddedSplits.columnSplits,
                              paddedSplits.splitColumns});
    slices = slices.slice(0, splits.splitColumns, 2);
    // Remove padding that was added to make all the partitions the same width
    slices =
        slices.reshape({slices.dim(0), slices.numElements() / slices.dim(0)});
    return slices.slice(0, splits.columns, 1);
  }();

  const auto [overflowInfo, baseTBuckets] =
      getInternalSparseTensor(graph, baseT, plan);

  auto metaInfoBuckets = fullyconnected::getBucketsByPartition(
      baseTBuckets.getMetaInfoTensor(), paddedSplits.getPartitions());
  const auto metaInfoShape2D = getShape2D(metaInfoBuckets);

  auto nzBuckets = fullyconnected::getBucketsByPartition(
      baseTBuckets.getNzValuesTensor(), paddedSplits.getPartitions());
  const auto nzShape2D = getShape2D(nzBuckets);
  logging::popsparse::debug("Creating embedding slice exchange tensors "
                            "based on NZ-data shape {} and metaInfo shape {}",
                            nzBuckets.shape(), metaInfoBuckets.shape());

  // Buckets need reshaping so we can use rank 2 tensors with
  // rows, columns representing tiles for exchange.
  to2DShape(nzBuckets);
  to2DShape(metaInfoBuckets);

  // Create splits plan matching that of the slices, but containing the
  // offsets (In other words the same, but with 1 column per split)
  auto [offsetsSplits, broadcastOffsetsShape] =
      createSplitsAndShape(paddedSplits, offsets.dim(0), false, true);
  // Create a broadcast offsets tensor to spread over tiles with each
  // column split - a broadcast of the offsets over the planned splits.
  auto offsetsBroadcast = offsets.expand({1});
  broadcastToMatch(offsetsBroadcast, broadcastOffsetsShape);
  // Create a helpfully mapped tensor for the broadcast offsets to avoid
  // exchanging every loop pass.  This gets primed with the other buffers but
  // offsets are not exchanged any more during this process
  const auto offsetsBroadcastPerTile =
      createSliceTensor(graph, offsets.elementType(), offsets.dim(0),
                        offsetsSplits, {di, layer + "/Offsets"});

  // Create pairs of exchange buffers with the same mapping as the nz,metadata
  // and the 2D shape, plus a view into the buffer representing the exchange
  // patterns required to propogate over rows (z and row splits) and columns.
  const auto numBuffers = 2;
  auto metaInfoSplits = PlannedSplits(metaInfoShape2D[0], params, plan,
                                      {di, layer + "/metainfoExchange"},
                                      metaInfoShape2D[1], false);
  auto nzSplits =
      PlannedSplits(nzShape2D[0], params, plan, {di, layer + "/nzExchange"},
                    nzShape2D[1], false);

  // Create exchange buffers for the NZ and metaInfo
  const auto metaInfoExBuf = createExchangeTensors(
      graph, prog, metaInfoBuckets.elementType(), metaInfoSplits, numBuffers,
      metaInfoShape2D[0], true, {di, layer + "/metaInfo"});
  const auto nzExBuf =
      createExchangeTensors(graph, prog, inputType, nzSplits, numBuffers,
                            nzShape2D[0], true, {di, layer + "/Nzero"});

  // Create program fragments used below

  // Use the overflow info to determine the number of column loops to make
  const auto yPartitionOverflowIndex = 1;
  auto columnOverFlowCount = overflowInfo.slice(
      {yPartitionOverflowIndex, yPartitionOverflowIndex + 1});
  auto [outerDecisionProg, outerDecisionFlag] = createDecisionProg(
      graph, prog, columnOverFlowCount, {di, layer + "/outerLoop"});

  const auto [exchangeProg, bufferSelect] = createExchangeProg(
      graph, prog, numBuffers, {nzExBuf, metaInfoExBuf}, {di, layer});

  const auto computeProg = createSliceComputeProg(
      graph, prog, paddedSplits, paddedSlices, offsetsBroadcastPerTile,
      bufferSelect, numBuffers, nzExBuf, nzSplits, metaInfoExBuf,
      metaInfoSplits, {di, layer});

  // Prime the buffers.
  prog.add(Copy(metaInfoBuckets, metaInfoExBuf.src[0], false, {di}));
  prog.add(Copy(nzBuckets, nzExBuf.src[0], false, {di}));
  prog.add(Copy(offsetsBroadcast, offsetsBroadcastPerTile, false, {di}));

  // Zero the slices once as we'll gradually populate them with the sparse
  // NZ values on each call to the slice vertices.
  zero(graph, paddedSlices, prog, {di, layer});

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
  //     exchangeNz(src[srcIndex], dstRow[dstIndex]);
  //     exchangeMetaData(src[srcIndex], dstRow[dstIndex]);
  //     srcIndex=xor(srcIndex,1);
  //     dstIndex=xor(dstIndex,1);
  //     rowLoops--;
  //   }
  //   // This is `exchangeProg[0] or [1]`
  //   exchangeNz(src[srcIndex], dstColumn[dstIndex]);
  //   exchangeMetaData(src[srcIndex], dstColumn[dstIndex]);
  //   srcIndex=xor(srcIndex,1);
  //   dstIndex=xor(dstIndex,1);
  //   columnLoops--;
  // }

  Sequence innerProg, outerProg({}, {di});
  innerProg.add(
      Switch(bufferSelect, {{0, computeProg[0]}, {1, computeProg[1]}}, {di}));
  outerProg.add(
      Repeat(paddedSplits.rowSplits * paddedSplits.zSplits, innerProg, {di}));
  outerProg.add(
      Switch(bufferSelect, {{0, exchangeProg[0]}, {1, exchangeProg[1]}}, {di}));
  prog.add(
      RepeatWhileTrue(outerDecisionProg, outerDecisionFlag, outerProg, {di}));

  return slices;
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
// After b) a full cycle of a) is carried out once more - repeatedly until
// all data has visited all the tiles it needs to and updates are complete.
//
// Sparse data (being updated) remains ontile, dense data and indices move.
// On each exchange step b) the sparse data is being updated using dense data
// from a different column partiton.  The vertex needs to know which partition
// this is from, therefore a columnPartition Tensor needs to be exchanged with
// the dense data.

void embeddingUpdateAdd(Graph &graph, const SparseTensor &baseT,
                        const Tensor &slices_, const Tensor &offsets_,
                        const Tensor &scale, Sequence &prog,
                        const FullyConnectedParams &params,
                        const poplar::DebugContext &debugContext,
                        const OptionFlags &options, PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext,
      DI_ARGS(baseT, slices_, offsets_, scale, params, options, cache));

  const auto inputType = baseT.getNzValuesTensor().elementType();
  const std::string layer = "embeddingUpdateAdd";
  const auto plan =
      getFullyConnectedPlan(graph, inputType, params, options, cache);
  const auto &opts = fullyconnected::parseOptionFlags(options);
  if (!opts.doGradWPass) {
    throw poputil::poplibs_error("popsparse embeddingUpdateAdd requires "
                                 "the doGradWPass option to be set.");
  }

  // Pad offsets to a consistent shape, otherwise when exchanging
  // and processing every step needs to be customised to the number of
  // offsets on each tile.  The pad is initialised with max unsigned to
  // prevent actual processing (Such an offset won't be found in the sparse
  // data).  Therefore the slice data doesn't need to be padded as it isn't
  // accessed.
  const auto offsets = [&]() {
    const PlannedSplits splits(offsets_.dim(0), params, plan,
                               {di, layer + "/prePad"});
    const auto paddedSize =
        splits.zSplits * splits.rowSplits * splits.slicesSplitRows;
    const auto paddedValue = std::numeric_limits<unsigned>::max();
    auto pad = graph.addConstant<unsigned>(
        offsets_.elementType(), {paddedSize - offsets_.dim(0)}, paddedValue,
        {di, layer + "/PadOffsets"});
    graph.setTileMapping(pad, 0);
    return concat(offsets_, pad);
  }();

  // Where the number of columns wasn't exactly divisible by the column splits
  // we need to pad to make the last, smaller slice the same as all the others.
  // This is to ensure that when it is passed to other column partitions it is
  // treated correctly. The meta-data content will ensure that padded elements
  // are never accessed
  const auto slices = [&]() {
    const auto slicesType = slices_.elementType();
    const PlannedSplits splits(offsets_.dim(0), params, plan,
                               {di, layer + "/prePad"});
    const auto paddedSize = splits.columnSplits * splits.splitColumns;
    auto pad = graph.addVariable(slicesType,
                                 {slices_.dim(0), paddedSize - slices_.dim(1)},
                                 {di, layer + "/PadSlices"});
    graph.setTileMapping(pad, 0);
    auto slices = concat(slices_, pad, 1);
    // Now that's nice and regular we can pad again to make the columns in each
    // partition even which removes the need for a copy after exchange in each
    // loop pass.  As the metadata won't reference the padded columns
    // they are never accessed and so have no effect on the result.
    const auto padToEven = (slicesType == HALF && splits.splitColumns % 2);
    if (padToEven) {
      slices = slices.reshape(
          {slices.dim(0), splits.columnSplits, splits.splitColumns});
      auto pad =
          graph.addVariable(slicesType, {slices.dim(0), splits.columnSplits, 1},
                            {di, layer + "/PadSlices"});
      graph.setTileMapping(pad, 0);
      slices = concat(slices, pad, 2);
    }
    return slices.reshape(
        {slices.dim(0), slices.numElements() / slices.dim(0)});
  }();

  // We want the same plan so present the same parameters, but the total
  // columns we operate on is different as we padded above
  const PlannedSplits plannedSplits(offsets.dim(0), params, plan,
                                    {di, layer + "/padded"}, slices.dim(1));
  const auto [overflowInfo, baseTBuckets] =
      getInternalSparseTensor(graph, baseT, plan);

  // Create splits plan matching that of the slices, but containing the
  // offsets (In other words the same, but with 1 column per split)
  auto [offsetsSplits, broadcastOffsetsShape] =
      createSplitsAndShape(plannedSplits, offsets.dim(0), false, true);
  // Create a broadcast offsets tensor to spread over tiles with each
  // column split - a broadcast of the offsets over the planned splits
  auto offsetsBroadcast = offsets.expand({1});
  broadcastToMatch(offsetsBroadcast, broadcastOffsetsShape);

  // Create splits plan matching that of the slices, but containing the
  // columnPartition variable (In other words the same, but with 1 column per
  // split and 1 row per split)
  auto [columnPartitionSplits, columnPartitionShape] = createSplitsAndShape(
      plannedSplits, plannedSplits.zSplits * plannedSplits.rowSplits, true,
      true);
  // Create a columnPartition tensor to spread over tiles with each
  // column split: {0,1,2,...columnSplits} broadcast  rowSplits*zSplits times

  std::vector<unsigned> columnPartitions(columnPartitionSplits.columnSplits);
  std::iota(columnPartitions.begin(), columnPartitions.end(), 0);
  auto columnPartitionBroadcast = graph.addConstant<unsigned>(
      UNSIGNED_INT, {1, columnPartitionSplits.columnSplits}, columnPartitions);

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
      createExchangeTensors(graph, prog, inputType, plannedSplits, numBuffers,
                            offsets.dim(0), false, {di, layer + "/Slices"});

  const auto offsetsExBuf = createExchangeTensors(
      graph, prog, offsets.elementType(), offsetsSplits, numBuffers,
      offsets.dim(0), false, {di, layer + "/Offsets"});

  auto columnPartitionExBuf = createExchangeTensors(
      graph, prog, columnPartitionBroadcast.elementType(),
      columnPartitionSplits, numBuffers,
      columnPartitionBroadcast.dim(0) * columnPartitionBroadcast.dim(1), false,
      {di, layer + "/ColumnPartition"});

  for (unsigned i = 0; i < numBuffers; i++) {
    const auto shape = columnPartitionBroadcast.shape();
    columnPartitionExBuf.src[i] = columnPartitionExBuf.src[i].reshape(shape);
    columnPartitionExBuf.rowDst[i] =
        columnPartitionExBuf.rowDst[i].reshape(shape);
    columnPartitionExBuf.columnDst[i] =
        columnPartitionExBuf.columnDst[i].reshape(shape);
  }
  // Extract the tensors from the SparseTensor. Make a partials tensor of
  // type float which is mapped the same as the NZ data, zeroed and updated with
  // scaled inputs.  It is added to the original NZ data to complete the update.
  // This maintains resolution while potentially accumulating many updates.
  const auto metaInfoBuckets = fullyconnected::getBucketsByPartition(
      baseTBuckets.getMetaInfoTensor(), plannedSplits.getPartitions());
  const auto nzBuckets = fullyconnected::getBucketsByPartition(
      baseTBuckets.getNzValuesTensor(), plannedSplits.getPartitions());

  const auto nzVertexOutput =
      graph.clone(FLOAT, nzBuckets, {di, layer + "/nzPartials"});
  zero(graph, nzVertexOutput, prog, {di, layer});

  // Create program fragments used below

  // Use the overflow info to determine the number of column loops to make
  const auto yPartitionOverflowIndex = 1;
  const auto columnOverFlowCount = overflowInfo.slice(
      {yPartitionOverflowIndex, yPartitionOverflowIndex + 1});
  const auto [outerDecisionProg, outerDecisionFlag] = createDecisionProg(
      graph, prog, columnOverFlowCount, {di, layer + "/outerLoop"});

  const auto [exchangeProg, bufferSelect] = createExchangeProg(
      graph, prog, numBuffers,
      {slicesExBuf, offsetsExBuf, columnPartitionExBuf}, {di, layer});

  const auto computeProg = createUpdateComputeProg(
      graph, prog, plannedSplits, metaInfoBuckets, nzVertexOutput, bufferSelect,
      scale, numBuffers, slicesExBuf, offsetsExBuf, columnPartitionExBuf,
      {di, layer});

  // Prime the 1st buffer (Only the existing slices rows go in, although the
  // buffer can be larger if padding slices rows)
  prog.add(
      Copy(slices, slicesExBuf.src[0].slice(0, slices.dim(0)), false, {di}));
  prog.add(Copy(offsetsBroadcast, offsetsExBuf.src[0], false, {di}));
  prog.add(
      Copy(columnPartitionBroadcast, columnPartitionExBuf.src[0], false, {di}));

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

  Sequence innerProg, outerProg({}, {di});
  innerProg.add(
      Switch(bufferSelect, {{0, computeProg[0]}, {1, computeProg[1]}}, {di}));

  outerProg.add(
      Repeat(plannedSplits.slicesRowSplits * plannedSplits.slicesIndicesSplits,
             innerProg, {di}));
  outerProg.add(
      Switch(bufferSelect, {{0, exchangeProg[0]}, {1, exchangeProg[1]}}, {di}));

  prog.add(
      RepeatWhileTrue(outerDecisionProg, outerDecisionFlag, outerProg, {di}));

  // Add the partials to the Nz-data. Scale was already applied
  // while adding to the partial result.
  scaledAddTo(graph, nzBuckets, nzVertexOutput, 1.0, prog, {di, layer});
}

// External function to be used to create the slice input for an update
// operation
Tensor createSliceTensor(Graph &graph, const Type &dataType,
                         const FullyConnectedParams &params,
                         std::size_t numIndices,
                         const poplar::DebugContext &debugContext,
                         const OptionFlags &options, PlanningCache *cache) {
  poputil::PoplibsOpDebugInfo di(
      debugContext, DI_ARGS(dataType, params, numIndices, options, cache));
  const auto plan =
      getFullyConnectedPlan(graph, dataType, params, options, cache);
  const PlannedSplits plannedSplits(numIndices, params, plan, {di});
  return createSliceTensor(graph, dataType, numIndices, plannedSplits, {di});
}

} // end namespace dynamic
} // end namespace popsparse
